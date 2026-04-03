"""
Validation video:
  Per-fish:   Panel 1 = raw crop + mask overlay  |  Panel 2 = head detection + turn indicator
  Grid video: Brady Bunch compilation of all fish, same 2-panel layout per cell
  Turn stats: printed summary per fish
"""

import json
import math
import os
import sys
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import skeletonize
from idtrackerai import ListOfBlobs

# ── Session folder browser ────────────────────────────────────────────────────

root = tk.Tk()
root.withdraw()
SESSION_DIR = filedialog.askdirectory(title="Select idtrackerai session folder")
root.destroy()

if not SESSION_DIR:
    print("No folder selected, exiting.")
    sys.exit(0)

# Read video path and other metadata from session.json
_meta       = json.load(open(os.path.join(SESSION_DIR, "session.json")))
SOURCE_VIDEO = _meta["video_paths"][0]
OUTPUT_DIR   = os.path.join(SESSION_DIR, "head_detection_output")
CROP_SIZE    = 158
HALF         = CROP_SIZE // 2
MEDIAN_KERNEL     = 5      # temporal median filter (frames)
MIN_SKELETON_PX   = 5
CONTINUITY_WEIGHT = 2.0
MASK_ALPHA        = 0.25
MASK_SHADE_BGR    = (0, 200, 255)
TURN_THRESHOLD_DEG = 7     # smoothed °/frame to count as a turn
TURN_SMOOTH_WINDOW = 3     # frames over which to smooth angular velocity
MIN_TURN_FRAMES    = 2     # minimum consecutive frames above threshold
MIN_TURN_ANGLE_DEG = 20    # minimum total angle change over the bout
TURN_BAR_H         = 20    # pixel height of the turn indicator bar
HEAD_SEARCH_R      = 15    # pixels inward from tip to search for max DT
GRAPH_W            = 120   # width of the real-time tally panel

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading blobs...")
list_of_blobs = ListOfBlobs.load(f"{SESSION_DIR}/preprocessing/list_of_blobs.pickle")

cap = cv2.VideoCapture(SOURCE_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap.release()

session_meta = _meta
n_fish       = session_meta["number_of_animals"]
tracked_end  = max(iv[1] for iv in session_meta["tracking_intervals"])
n_frames     = min(int(round(fps * 60)), total_source_frames,
                   len(list_of_blobs.blobs_in_video), tracked_end)
print(f"  {n_frames} frames ({n_frames/fps:.1f} s) | {n_fish} fish | {fps:.2f} fps")

print("Loading trajectories...")
traj_data    = np.load(f"{SESSION_DIR}/trajectories/trajectories.npy", allow_pickle=True).item()
trajectories = traj_data["trajectories"].copy()   # (total, n_fish, 2) — (x, y)

for fi in range(trajectories.shape[1]):
    for coord in range(2):
        y = trajectories[:, fi, coord]
        nans = np.isnan(y)
        if nans.any() and (~nans).sum() > 1:
            x = np.arange(len(y))
            y[nans] = np.interp(x[nans], x[~nans], y[~nans])

traj_centroids = {fid: trajectories[:n_frames, fid - 1] for fid in range(1, n_fish + 1)}


# ── Mask building ─────────────────────────────────────────────────────────────

def build_raw_masks(fish_identity):
    masks = np.zeros((n_frames, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    for fn in range(n_frames):
        blobs = list_of_blobs.blobs_in_video[fn]
        blob_by_id = {b.identity: b for b in blobs
                      if not b.is_a_crossing and b.identity is not None}
        if fish_identity not in blob_by_id:
            continue
        blob = blob_by_id[fish_identity]
        cx, cy = blob.centroid
        icx, icy = int(round(cx)), int(round(cy))
        bc = blob.bbox_corners
        x_min, y_min = bc.bottom, bc.left
        bm = blob.get_bbox_mask()
        bh, bw = bm.shape
        cc = x_min - (icx - HALF)
        cr = y_min - (icy - HALF)
        sr0, sc0 = max(0, -cr), max(0, -cc)
        dr0, dc0 = max(0, cr), max(0, cc)
        ch = min(bh - sr0, CROP_SIZE - dr0)
        cw = min(bw - sc0, CROP_SIZE - dc0)
        if ch > 0 and cw > 0:
            masks[fn, dr0:dr0+ch, dc0:dc0+cw] = (bm[sr0:sr0+ch, sc0:sc0+cw] > 0) * 255
    return masks


# ── Head detection ────────────────────────────────────────────────────────────

def find_endpoints(skel):
    k = np.array([[1,1,1],[1,10,1],[1,1,1]], np.uint8)
    return np.argwhere(cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=0) == 11)


def score_endpoints(endpoints, dt, vel_xy):
    dt_s = np.array([dt[r, c] for r, c in endpoints], dtype=float)
    rng  = dt_s.max() - dt_s.min()
    dt_n = (dt_s - dt_s.min()) / rng if rng > 1e-3 else np.zeros_like(dt_s)

    centre = np.array([HALF, HALF], dtype=float)
    vx, vy = vel_xy
    v = np.array([vy, vx], dtype=float)
    vn = np.linalg.norm(v)
    if vn > 1e-3:
        v /= vn
        vel_s = np.array([float(np.dot((ep - centre) / max(np.linalg.norm(ep - centre), 1e-3), v))
                          for ep in endpoints])
    else:
        vel_s = np.zeros(len(endpoints))
    return dt_n + (vel_s + 1) / 2


def detect_head_sequence(masks, velocities):
    heads, prev = [None] * len(masks), None
    for fn, (mask, vel) in enumerate(zip(masks, velocities)):
        binary = (mask > 0).astype(np.uint8)
        if binary.sum() < MIN_SKELETON_PX:
            prev = None; continue
        skel = skeletonize(binary).astype(np.uint8)
        eps  = find_endpoints(skel)
        if len(eps) < 2:
            prev = None; continue
        dt   = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        base = score_endpoints(eps, dt, vel)
        if prev is not None:
            dists   = np.array([np.linalg.norm(ep - np.array(prev)) for ep in eps])
            md      = dists.max()
            cont    = 1.0 - dists / md if md > 1e-3 else np.full(len(eps), 0.5)
            combined = base + CONTINUITY_WEIGHT * cont
        else:
            combined = base
        idx      = combined.argmax()
        heads[fn] = tuple(eps[idx])
        prev      = eps[idx]
    return heads


# ── Turn detection ────────────────────────────────────────────────────────────

def smooth_head_positions(heads, window=5):
    """
    Apply a short moving-average to (row, col) head positions.
    Frames with no head (None) are skipped and left as None.
    Prevents pixel-level jitter while keeping the position responsive.
    """
    n    = len(heads)
    rows = np.array([h[0] if h is not None else np.nan for h in heads], dtype=float)
    cols = np.array([h[1] if h is not None else np.nan for h in heads], dtype=float)
    hw   = window // 2
    sr, sc = rows.copy(), cols.copy()
    for fn in range(n):
        w = slice(max(0, fn - hw), min(n, fn + hw + 1))
        vr = rows[w][~np.isnan(rows[w])]
        vc = cols[w][~np.isnan(cols[w])]
        if len(vr):
            sr[fn] = vr.mean()
            sc[fn] = vc.mean()
    return [
        (int(round(sr[fn])), int(round(sc[fn]))) if not np.isnan(sr[fn]) else None
        for fn in range(n)
    ]


def compute_turns(heads):
    """
    Returns:
        angles      – heading angle in degrees per frame (NaN if no head)
        turn_dir    – +1 left (CCW), -1 right (CW), 0 straight, per frame
                      (only set for bouts that pass all three gates)
        turn_counts – dict with 'left', 'right', 'total'

    A bout must pass all three gates to count as a turn:
      1. Smoothed angular velocity exceeds TURN_THRESHOLD_DEG
      2. Bout lasts at least MIN_TURN_FRAMES consecutive frames
      3. Total accumulated angle change >= MIN_TURN_ANGLE_DEG
    """
    n      = len(heads)
    angles = np.full(n, np.nan)
    for fn, h in enumerate(heads):
        if h is not None:
            hr, hc     = h
            angles[fn] = np.degrees(np.arctan2(HALF - hr, hc - HALF))

    # Frame-to-frame angular velocity, wrapped to [-180, 180]
    ang_vel = np.full(n, np.nan)
    for fn in range(1, n):
        if not (np.isnan(angles[fn]) or np.isnan(angles[fn - 1])):
            ang_vel[fn] = (angles[fn] - angles[fn - 1] + 180) % 360 - 180

    # Light smoothing
    hw   = TURN_SMOOTH_WINDOW // 2
    av_s = np.full(n, np.nan)
    for fn in range(n):
        w = ang_vel[max(0, fn - hw):min(n, fn + hw + 1)]
        v = w[~np.isnan(w)]
        if len(v):
            av_s[fn] = v.mean()

    # Raw candidate direction per frame (gate 1 only)
    raw_dir = np.zeros(n, dtype=int)
    raw_dir[np.nan_to_num(av_s) >  TURN_THRESHOLD_DEG] =  1
    raw_dir[np.nan_to_num(av_s) < -TURN_THRESHOLD_DEG] = -1

    # Extract bouts and apply gates 2 + 3
    turn_dir = np.zeros(n, dtype=int)
    left = right = 0
    fn = 0
    while fn < n:
        if raw_dir[fn] == 0:
            fn += 1
            continue
        # Start of a bout
        direction = raw_dir[fn]
        start     = fn
        while fn < n and raw_dir[fn] == direction:
            fn += 1
        end = fn  # exclusive

        duration    = end - start
        total_angle = np.nansum(np.abs(ang_vel[start:end]))

        if duration >= MIN_TURN_FRAMES and total_angle >= MIN_TURN_ANGLE_DEG:
            turn_dir[start:end] = direction
            if direction == 1:
                left  += 1
            else:
                right += 1

    return angles, turn_dir, {"left": left, "right": right, "total": left + right}


# ── Frame rendering ───────────────────────────────────────────────────────────

def crop_raw(source_frame, cx, cy):
    icx, icy = int(round(cx)), int(round(cy))
    r0, r1 = max(icy - HALF, 0), min(icy + HALF, frame_h_src)
    c0, c1 = max(icx - HALF, 0), min(icx + HALF, frame_w_src)
    crop = source_frame[r0:r1, c0:c1]
    pt, pb = max(0, HALF - icy), max(0, (icy + HALF) - frame_h_src)
    pl, pr = max(0, HALF - icx), max(0, (icx + HALF) - frame_w_src)
    if any([pt, pb, pl, pr]):
        crop = np.pad(crop, ((pt, pb), (pl, pr), (0, 0)))
    return crop


def make_panel1(raw_bgr, mask):
    """Raw crop with cyan contour + subtle shading."""
    out   = raw_bgr.copy()
    shade = np.zeros_like(raw_bgr)
    shade[mask > 0] = MASK_SHADE_BGR
    out   = cv2.addWeighted(out, 1.0, shade, MASK_ALPHA, 0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 255), 1)
    return out


def head_inscribed_circle(skel, dt, head_rc):
    """
    Find the skeleton point with maximum DT within HEAD_SEARCH_R pixels of
    the head endpoint — the widest part of the head region, not the bare tip.
    Returns (row, col), radius.
    """
    hr, hc = head_rc
    pts    = np.argwhere(skel)
    if len(pts) == 0:
        return head_rc, 2
    dists  = np.hypot(pts[:, 0] - hr, pts[:, 1] - hc)
    nearby = pts[dists <= HEAD_SEARCH_R]
    if len(nearby) == 0:
        return head_rc, 2
    dt_vals = np.array([dt[r, c] for r, c in nearby])
    best    = nearby[dt_vals.argmax()]
    return tuple(best), max(2, int(round(dt_vals.max())))


def make_panel2(mask, head, turn_dir):
    """Binary mask + skeleton + inscribed circle at head region + turn bar."""
    frame  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    binary = (mask > 0).astype(np.uint8)
    skel   = None
    if binary.sum() >= MIN_SKELETON_PX:
        skel = skeletonize(binary)
        frame[skel] = (255, 0, 0)
    if head is not None and skel is not None:
        dt              = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        centre_rc, radius = head_inscribed_circle(skel, dt, head)
        cr, cc          = centre_rc
        cv2.circle(frame, (cc, cr), radius, (0, 0, 255), 1)

    # Colour bar: green = left turn, magenta = right turn
    if turn_dir == 1:
        frame[:TURN_BAR_H, :] = (0, 220, 0)
    elif turn_dir == -1:
        frame[:TURN_BAR_H, :] = (220, 0, 220)

    return frame


def make_tally_panel(cum_left, cum_right, max_count, fn, fish_id):
    """
    Vertical bar chart showing cumulative left (green) and right (magenta)
    turn counts up to frame fn, growing in real time.
    """
    panel  = np.zeros((CROP_SIZE, GRAPH_W, 3), dtype=np.uint8)
    margin = 25   # bottom margin for labels
    bar_w  = 30
    bar_area = CROP_SIZE - margin - 10   # usable bar height

    l = int(cum_left[fn])
    r = int(cum_right[fn])
    scale = bar_area / max(max_count, 1)

    # Left bar
    lx = 12
    lh = max(1, int(l * scale)) if l > 0 else 0
    if lh:
        cv2.rectangle(panel,
                      (lx, CROP_SIZE - margin - lh),
                      (lx + bar_w, CROP_SIZE - margin),
                      (0, 210, 0), -1)
    cv2.putText(panel, str(l),
                (lx + 4, CROP_SIZE - margin - lh - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "L",
                (lx + 9, CROP_SIZE - margin + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 0), 1, cv2.LINE_AA)

    # Right bar
    rx = lx + bar_w + 16
    rh = max(1, int(r * scale)) if r > 0 else 0
    if rh:
        cv2.rectangle(panel,
                      (rx, CROP_SIZE - margin - rh),
                      (rx + bar_w, CROP_SIZE - margin),
                      (210, 0, 210), -1)
    cv2.putText(panel, str(r),
                (rx + 4, CROP_SIZE - margin - rh - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 0, 210), 1, cv2.LINE_AA)
    cv2.putText(panel, "R",
                (rx + 9, CROP_SIZE - margin + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 0, 210), 1, cv2.LINE_AA)

    # Baseline
    cv2.line(panel,
             (lx - 2, CROP_SIZE - margin),
             (rx + bar_w + 2, CROP_SIZE - margin),
             (160, 160, 160), 1)

    # Time + fish label
    cv2.putText(panel, f"#{fish_id}  {fn/fps:.0f}s",
                (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    return panel


# ── Pre-compute all fish data ─────────────────────────────────────────────────

print("Building mask buffers + head detection + turns...")
fish_heads    = {}
fish_turn_dir = {}
fish_masks    = {}
fish_cum_left  = {}
fish_cum_right = {}
fish_max_count = {}

for fid in range(1, n_fish + 1):
    print(f"  Fish {fid}...")
    raw  = build_raw_masks(fid)
    filt = (median_filter(raw.astype(np.float32),
                          size=(MEDIAN_KERNEL, 1, 1)) > 127).astype(np.uint8) * 255
    fish_masks[fid] = filt

    tc   = traj_centroids[fid]
    vels = [tc[min(fn+2, n_frames-1)] - tc[max(fn-2, 0)] for fn in range(n_frames)]
    raw_heads = detect_head_sequence(filt, vels)
    fish_heads[fid] = smooth_head_positions(raw_heads)  # smoothed for display only

    angles, turn_dir, counts = compute_turns(raw_heads)  # raw for turn signal
    fish_turn_dir[fid] = turn_dir

    # Cumulative counts: increment at the start of each new bout
    cum_l = np.zeros(n_frames, dtype=int)
    cum_r = np.zeros(n_frames, dtype=int)
    for fn in range(1, n_frames):
        cum_l[fn] = cum_l[fn - 1] + (1 if turn_dir[fn] == 1  and turn_dir[fn - 1] != 1  else 0)
        cum_r[fn] = cum_r[fn - 1] + (1 if turn_dir[fn] == -1 and turn_dir[fn - 1] != -1 else 0)
    fish_cum_left[fid]  = cum_l
    fish_cum_right[fid] = cum_r
    fish_max_count[fid] = max(int(cum_l[-1]) + int(cum_r[-1]), 1)

    dur = n_frames / fps
    print(f"    Turns — left: {counts['left']}  right: {counts['right']}  "
          f"total: {counts['total']}  ({counts['total']/dur*60:.1f}/min)")


# ── Grid layout ───────────────────────────────────────────────────────────────

PANEL_W    = CROP_SIZE * 2 + GRAPH_W   # two video panels + tally graph
grid_cols  = math.ceil(math.sqrt(n_fish))
grid_rows  = math.ceil(n_fish / grid_cols)
grid_w     = grid_cols * PANEL_W
grid_h     = grid_rows * CROP_SIZE

fourcc = cv2.VideoWriter_fourcc(*"XVID")

# Open per-fish writers
fish_writers = {
    fid: cv2.VideoWriter(
        f"{OUTPUT_DIR}/fish_{fid}_validation.avi",
        fourcc, fps, (PANEL_W, CROP_SIZE), isColor=True,
    )
    for fid in range(1, n_fish + 1)
}
grid_writer = cv2.VideoWriter(
    f"{OUTPUT_DIR}/all_fish_grid.avi",
    fourcc, fps, (grid_w, grid_h), isColor=True,
)

# ── Single pass through source video ─────────────────────────────────────────

print("Writing videos (single pass through source)...")
cap = cv2.VideoCapture(SOURCE_VIDEO)

for fn in range(n_frames):
    if fn % 300 == 0:
        print(f"  Frame {fn}/{n_frames}...")
    ret, source_frame = cap.read()
    if not ret:
        break

    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for fid in range(1, n_fish + 1):
        cx, cy   = traj_centroids[fid][fn]
        raw      = crop_raw(source_frame, cx, cy)
        mask     = fish_masks[fid][fn]
        head     = fish_heads[fid][fn]
        turn     = fish_turn_dir[fid][fn]

        p1       = make_panel1(raw, mask)
        p2       = make_panel2(mask, head, turn)
        p3       = make_tally_panel(fish_cum_left[fid], fish_cum_right[fid],
                                    fish_max_count[fid], fn, fid)
        composite = np.concatenate([p1, p2, p3], axis=1)

        # Write individual fish video
        fish_writers[fid].write(composite)

        # Place in grid
        gi   = fid - 1
        gr, gc = divmod(gi, grid_cols)
        rr   = gr * CROP_SIZE
        cc   = gc * PANEL_W
        grid[rr:rr + CROP_SIZE, cc:cc + PANEL_W] = composite

        # Fish label in grid cell
        cv2.putText(grid, f"#{fid}", (cc + 3, rr + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    grid_writer.write(grid)

cap.release()
for w in fish_writers.values():
    w.release()
grid_writer.release()

# ── Export turn data as CSV ───────────────────────────────────────────────────

import csv

# Per-frame data: one row per frame per fish
with open(f"{OUTPUT_DIR}/turns_per_frame.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fish_id", "frame", "time_s", "turn_direction", "cum_left", "cum_right"])
    for fid in range(1, n_fish + 1):
        td = fish_turn_dir[fid]
        cl = fish_cum_left[fid]
        cr = fish_cum_right[fid]
        for fn in range(n_frames):
            if td[fn] != 0 or cl[fn] != (cl[fn-1] if fn else 0) or cr[fn] != (cr[fn-1] if fn else 0):
                w.writerow([fid, fn, f"{fn/fps:.3f}", td[fn], cl[fn], cr[fn]])

# Per-fish summary
with open(f"{OUTPUT_DIR}/turns_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fish_id", "duration_s", "left_turns", "right_turns",
                "total_turns", "turns_per_min", "left_bias"])
    dur = n_frames / fps
    for fid in range(1, n_fish + 1):
        l = int(fish_cum_left[fid][-1])
        r = int(fish_cum_right[fid][-1])
        total = l + r
        bias  = round((l - r) / total, 3) if total else 0  # +1 = all left, -1 = all right
        w.writerow([fid, round(dur, 1), l, r, total,
                    round(total / dur * 60, 1), bias])

print(f"\nDone!  Videos + CSVs saved to {OUTPUT_DIR}/")
