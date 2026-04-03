"""
Head detection test using:
  - Temporal median filter to suppress dropped-frame flicker
  - Distance transform at skeleton endpoints (tail tip is sharper → lower DT)
  - Velocity direction from trajectory to confirm/flip the DT choice
"""

import os
import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from idtrackerai import ListOfBlobs

SESSION_DIR = "session_batch1control_freeswim_2026-03-27-141723-good"
SOURCE_VIDEO = (
    r"C:\Users\grich\Macquarie University\Morsch Group - Documents\Grant R"
    r"\Fish videos\Pradeep Fish Free Swim Videos"
    r"\27032026 Pradeep's Fish Free Swim\batch1control_freeswim_2026-03-27-141723-good.avi"
)
OUTPUT_DIR = "head_detection_output"
N_FRAMES = 500
CROP_SIZE = 158
HALF = CROP_SIZE // 2
MEDIAN_KERNEL = 5   # frames, must be odd
MIN_SKELETON_PX = 5  # skip tiny/degenerate skeletons

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────

print("Loading blobs...")
list_of_blobs = ListOfBlobs.load(f"{SESSION_DIR}/preprocessing/list_of_blobs.pickle")
n_frames = min(N_FRAMES, len(list_of_blobs.blobs_in_video))

cap = cv2.VideoCapture(SOURCE_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

print("Loading trajectories...")
traj_data = np.load(
    f"{SESSION_DIR}/trajectories/trajectories.npy", allow_pickle=True
).item()
# shape (n_total_frames, n_fish, 2)  — (x, y) in full-frame pixel coords
trajectories = traj_data["trajectories"].copy()

# Interpolate NaN gaps so velocity is always defined
for fish_idx in range(trajectories.shape[1]):
    for coord in range(2):
        y = trajectories[:, fish_idx, coord]
        nans = np.isnan(y)
        if nans.any() and (~nans).sum() > 1:
            x = np.arange(len(y))
            y[nans] = np.interp(x[nans], x[~nans], y[~nans])


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_skeleton_endpoints(skel: np.ndarray) -> np.ndarray:
    """Return (N, 2) array of (row, col) skeleton endpoints."""
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)
    filtered = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=0)
    pts = np.argwhere(filtered == 11)  # 10 (self) + 1 (one neighbour)
    return pts  # (row, col)


def dt_head_score(skel: np.ndarray, dt: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
    """
    Score each endpoint by the DT value there.
    Higher DT → wider body at that tip → more likely to be the head.
    Returns a score per endpoint (higher = more likely head).
    """
    return np.array([dt[r, c] for r, c in endpoints])


def velocity_head_score(endpoints_rc: np.ndarray, velocity_xy: np.ndarray) -> np.ndarray:
    """
    Score each endpoint by how well it aligns with the velocity direction.
    Endpoints are in canvas (row, col) coords; velocity is (vx, vy) full-frame.
    We compare the endpoint direction from canvas centre against velocity.
    """
    centre = np.array([HALF, HALF])
    # convert velocity (vx, vy) → (v_col, v_row) = (vx, vy)
    v = np.array([velocity_xy[1], velocity_xy[0]])  # (v_row, v_col) — flip x/y
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-3:
        return np.zeros(len(endpoints_rc))
    v = v / v_norm
    scores = []
    for ep in endpoints_rc:
        d = ep - centre
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-3:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(d / d_norm, v)))
    return np.array(scores)


def detect_head(mask: np.ndarray, velocity_xy: np.ndarray):
    """
    Given a binary mask (uint8, 0/255) and velocity vector (vx, vy),
    return (head_row, head_col) or None if detection fails.
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() < MIN_SKELETON_PX:
        return None

    skel = skeletonize(binary).astype(np.uint8)
    endpoints = find_skeleton_endpoints(skel)

    if len(endpoints) < 2:
        return None

    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    dt_scores = dt_head_score(skel, dt, endpoints)
    vel_scores = velocity_head_score(endpoints, velocity_xy)

    # Normalise each to [-1, 1] then combine with equal weight.
    # DT score: normalise by range so both signals have comparable scale.
    dt_range = dt_scores.max() - dt_scores.min()
    dt_norm = (dt_scores - dt_scores.min()) / dt_range if dt_range > 1e-3 else np.zeros_like(dt_scores)

    # Velocity score is already a cosine similarity in [-1, 1]; shift to [0, 1]
    vel_norm = (vel_scores + 1) / 2

    combined = dt_norm + vel_norm
    head_idx = combined.argmax()
    return tuple(endpoints[head_idx])  # (row, col)


# ── Build per-fish mask buffers ──────────────────────────────────────────────

print("Building mask buffers and running head detection...")
fourcc = cv2.VideoWriter_fourcc(*"XVID")

for fish_identity in range(1, 7):
    fish_idx = fish_identity - 1
    print(f"  Fish {fish_identity}...")

    # 1. Collect raw masks
    raw_masks = np.zeros((n_frames, CROP_SIZE, CROP_SIZE), dtype=np.uint8)

    for fn in range(n_frames):
        blobs = list_of_blobs.blobs_in_video[fn]
        blob_by_id = {
            b.identity: b for b in blobs
            if not b.is_a_crossing and b.identity is not None
        }
        if fish_identity not in blob_by_id:
            continue

        blob = blob_by_id[fish_identity]
        cx, cy = blob.centroid
        cx, cy = int(round(cx)), int(round(cy))

        bc = blob.bbox_corners
        x_min, y_min = bc.bottom, bc.left
        blob_mask = blob.get_bbox_mask()
        bh, bw = blob_mask.shape

        canvas_col = x_min - (cx - HALF)
        canvas_row = y_min - (cy - HALF)

        src_r0 = max(0, -canvas_row)
        src_c0 = max(0, -canvas_col)
        dst_r0 = max(0, canvas_row)
        dst_c0 = max(0, canvas_col)
        copy_h = min(bh - src_r0, CROP_SIZE - dst_r0)
        copy_w = min(bw - src_c0, CROP_SIZE - dst_c0)

        if copy_h > 0 and copy_w > 0:
            region = blob_mask[src_r0:src_r0 + copy_h, src_c0:src_c0 + copy_w]
            raw_masks[fn, dst_r0:dst_r0 + copy_h, dst_c0:dst_c0 + copy_w] = (
                (region > 0).astype(np.uint8) * 255
            )

    # 2. Temporal median filter along frame axis
    filtered_masks = median_filter(raw_masks.astype(np.float32), size=(MEDIAN_KERNEL, 1, 1))
    filtered_masks = (filtered_masks > 127).astype(np.uint8) * 255

    # 3. Head detection per frame → annotated BGR video
    out = cv2.VideoWriter(
        f"{OUTPUT_DIR}/fish_{fish_identity}_head.avi",
        fourcc, fps, (CROP_SIZE, CROP_SIZE), isColor=True,
    )

    for fn in range(n_frames):
        mask = filtered_masks[fn]

        # Velocity: central difference on interpolated trajectory
        fn_prev = max(fn - 2, 0)
        fn_next = min(fn + 2, n_frames - 1)
        pos_prev = trajectories[fn_prev, fish_idx]
        pos_next = trajectories[fn_next, fish_idx]
        velocity_xy = pos_next - pos_prev  # (vx, vy)

        head = detect_head(mask, velocity_xy)

        # Annotate: mask → BGR, draw skeleton in blue, head dot in red
        frame_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        binary = (mask > 0).astype(np.uint8)
        if binary.sum() >= MIN_SKELETON_PX:
            skel = skeletonize(binary)
            frame_bgr[skel, 0] = 255   # blue channel → skeleton
            frame_bgr[skel, 1] = 0
            frame_bgr[skel, 2] = 0

        if head is not None:
            hr, hc = head
            cv2.circle(frame_bgr, (hc, hr), 4, (0, 0, 255), -1)  # red dot

        out.write(frame_bgr)

    out.release()

print(f"Done! Annotated videos saved to {OUTPUT_DIR}/")
