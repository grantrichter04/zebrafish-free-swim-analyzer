"""
fish_posture_analyzer.py
========================
Extract per-frame body midlines from idtracker.ai individual cropped fish videos.

For each fish video the script:
  1. Segments the fish from the bright background using Otsu thresholding
  2. Skeletonizes the binary mask to find the body midline
  3. Detects endpoints and the head via curvature analysis
  4. Orders midline points head→tail and fits a B-spline
  5. Samples the spline at N_MIDLINE_POINTS evenly spaced points

Outputs (written to <session_folder>/posture_output/):
  - individual_<id>_midlines.csv   one row per frame; columns:
        frame, valid, head_x, head_y, tail_x, tail_y,
        x_00..x_19, y_00..y_19   (20 evenly-sampled midline points)
  - individual_<id>_posture.avi    side-by-side visualisation:
        [original crop | binary mask | skeleton + midline overlay]

Usage
-----
Run directly — a file dialog opens so you can select either:
  • An individual_<id>.avi file  →  processes that one video
  • A session folder              →  processes all individual_*.avi files found

Requirements: opencv-python, numpy, scipy, scikit-image
Run in the idtrackerai conda environment (or any env with the above packages):
    python fish_posture_analyzer.py
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

# ── tuneable parameters ────────────────────────────────────────────────────────
N_MIDLINE_POINTS = 20       # number of points sampled along the midline spline
SMOOTH_SIGMA     = 5        # gaussian smoothing sigma for curvature estimation
MIN_BLOB_AREA    = 50       # minimum pixel area to consider a mask valid
MORPH_KERNEL     = 3        # morphological open/close kernel size (pixels)
# ──────────────────────────────────────────────────────────────────────────────


# =============================================================================
# Segmentation
# =============================================================================

def segment_fish(frame_gray: np.ndarray) -> np.ndarray:
    """Return a uint8 binary mask (255 = fish) using inverted Otsu thresholding.

    The individual fish videos have a bright background (~230+) with the fish
    body appearing darker (~30-150).  Otsu's method reliably finds the boundary.
    """
    _, mask = cv2.threshold(
        frame_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # Clean up small noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    # Keep only the largest connected component
    return _keep_largest_component(mask)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask
    # stats row 0 is background; find largest foreground component
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest).astype(np.uint8) * 255)


# =============================================================================
# Midline extraction  (adapted from the idtrackerai sample code)
# =============================================================================

def extract_midline(binary_mask: np.ndarray):
    """Extract an ordered midline from a binary fish mask.

    Parameters
    ----------
    binary_mask : uint8 array, 255 = fish body

    Returns
    -------
    midline : (N, 2) float array of (x, y) coordinates ordered head→tail,
              or None if extraction fails
    spline  : scipy splprep output tuple, or None
    skel    : boolean skeleton array (same shape as binary_mask)
    """
    binary_bool = binary_mask > 0
    if binary_bool.sum() < MIN_BLOB_AREA:
        return None, None, np.zeros_like(binary_bool)

    skel = skeletonize(binary_bool)
    if not skel.any():
        return None, None, skel

    # midline as (x, y) — note np.where returns (row, col) = (y, x)
    midline = np.asarray(np.where(skel))[::-1].T   # shape (N, 2)
    if len(midline) < 4:
        return None, None, skel

    end_points = _find_end_points(skel)
    if len(end_points) < 2:
        return None, None, skel

    head = _find_head(binary_mask)
    if head is None:
        return None, None, skel

    order   = _midline_order(midline, end_points=end_points, head=head)
    midline = midline[order]

    try:
        spline, _ = interpolate.splprep(midline.T, s=len(midline) * 0.5)
        return midline, spline, skel
    except Exception:
        return midline, None, skel


def sample_spline(spline, n: int = N_MIDLINE_POINTS) -> np.ndarray:
    """Sample a splprep spline at n evenly-spaced parameter values.

    Returns (n, 2) array of (x, y) points.
    """
    u = np.linspace(0, 1, n)
    pts = interpolate.splev(u, spline)
    return np.column_stack(pts)   # (n, 2)


# ── internal helpers (directly from idtrackerai sample code) ──────────────────

def _find_head(binary_img: np.ndarray):
    """Locate the fish head using contour curvature analysis."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = contours[0].squeeze().astype(np.float32)
    if contour.ndim < 2 or len(contour) < 5:
        return None

    smoother = gaussian_filter1d(contour, SMOOTH_SIGMA, mode="wrap", axis=0)
    curv     = np.abs(_curvature(smoother))
    max_i    = argrelmax(curv, mode="wrap")[0]

    if len(max_i) == 0:
        return contour[np.argmax(curv)]

    b_s = max_i[np.argsort(curv[max_i])]
    if len(b_s) < 2:
        return contour[b_s[-1]]
    # second-highest curvature peak = head (highest = tail)
    return contour[b_s[-2]]


def _curvature(contour: np.ndarray) -> np.ndarray:
    dx1, dy1 = scipy.ndimage.convolve1d(
        contour, [-0.5, 0.0, 0.5], mode="wrap", axis=0
    ).T
    dx2, dy2 = scipy.ndimage.convolve1d(
        contour, [1.0, -2.0, 1.0], mode="wrap", axis=0
    ).T
    denom = np.power(dx1 * dx1 + dy1 * dy1, 1.5)
    denom[denom == 0] = np.finfo(float).eps
    return (dx1 * dy2 - dy1 * dx2) / denom


def _midline_order(midline: np.ndarray, end_points: np.ndarray,
                   head: np.ndarray) -> list:
    """Order midline indices from the endpoint closest to head outward."""
    sorted_indices = []
    free_indices   = list(range(len(midline)))

    dist_to_head  = cdist(head[None, :], end_points)
    first_ep      = end_points[dist_to_head.argmin()]
    last_ep       = end_points[dist_to_head.argmax()]

    d = cdist(first_ep[None, :], midline[free_indices])
    sorted_indices.append(free_indices.pop(int(d.argmin())))

    while free_indices:
        d = (
            cdist(midline[sorted_indices[-1]][None, :], midline[free_indices])
            - cdist(last_ep[None, :], midline[free_indices])
        )
        sorted_indices.append(free_indices.pop(int(d.argmin())))
        if np.allclose(midline[sorted_indices[-1]], last_ep):
            break

    return sorted_indices


def _find_end_points(skel: np.ndarray) -> np.ndarray:
    """Find skeleton branch endpoints (pixels with exactly one neighbour)."""
    kernel   = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)
    filtered = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=0)
    return np.asarray(np.where(filtered == 11))[::-1].T   # (x, y)


# =============================================================================
# Visualisation helpers
# =============================================================================

def _draw_overlay(frame_gray: np.ndarray, mask: np.ndarray,
                  skel: np.ndarray, midline_pts) -> np.ndarray:
    """Create a 3-panel BGR image: original | mask | skeleton+midline."""
    h, w = frame_gray.shape

    # Panel 1: original (convert to BGR)
    orig_bgr  = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # Panel 2: mask
    mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Panel 3: skeleton + midline on grayscale background
    overlay   = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    # Draw skeleton in blue
    skel_pts  = np.argwhere(skel)   # (row, col)
    for r, c in skel_pts:
        overlay[r, c] = (200, 100, 0)
    # Draw sampled midline points in green, connected
    if midline_pts is not None and len(midline_pts) > 1:
        pts_int = midline_pts.astype(np.int32)
        cv2.polylines(overlay, [pts_int[:, ::-1].reshape(-1, 1, 2)],
                      isClosed=False, color=(0, 220, 0), thickness=1)
        # Head in red, tail in cyan
        cv2.circle(overlay, tuple(pts_int[0,  ::-1]), 3, (0, 0, 255), -1)
        cv2.circle(overlay, tuple(pts_int[-1, ::-1]), 3, (255, 200, 0), -1)

    return np.hstack([orig_bgr, mask_bgr, overlay])


# =============================================================================
# Per-video processing
# =============================================================================

def process_video(video_path: Path, output_dir: Path):
    """Process one individual fish video and write CSV + visualisation AVI."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path.name}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stem     = video_path.stem                        # e.g. "individual_1"
    csv_path = output_dir / f"{stem}_midlines.csv"
    avi_path = output_dir / f"{stem}_posture.avi"

    # Visualisation writer (3 panels side by side)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(avi_path), fourcc, fps, (frame_w * 3, frame_h))

    # CSV header
    col_x = [f"x_{i:02d}" for i in range(N_MIDLINE_POINTS)]
    col_y = [f"y_{i:02d}" for i in range(N_MIDLINE_POINTS)]
    header = ["frame", "valid", "head_x", "head_y", "tail_x", "tail_y"] + col_x + col_y

    rows = []
    n_valid = 0

    for frame_idx in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Black frames = crossing / untracked — skip but record as invalid
        if frame_bgr.max() == 0:
            row = [frame_idx, 0] + [float("nan")] * (4 + N_MIDLINE_POINTS * 2)
            rows.append(row)
            panel = np.zeros((frame_h, frame_w * 3, 3), np.uint8)
            writer.write(panel)
            continue

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = segment_fish(gray)

        midline, spline, skel = extract_midline(mask)

        if spline is not None:
            pts = sample_spline(spline, N_MIDLINE_POINTS)
            head_xy = pts[0]
            tail_xy = pts[-1]
            xs = pts[:, 0].tolist()
            ys = pts[:, 1].tolist()
            row = [frame_idx, 1, head_xy[0], head_xy[1],
                   tail_xy[0], tail_xy[1]] + xs + ys
            n_valid += 1
        else:
            row = [frame_idx, 0] + [float("nan")] * (4 + N_MIDLINE_POINTS * 2)
            pts = None

        rows.append(row)

        panel = _draw_overlay(gray, mask, skel, pts if spline is not None else None)
        writer.write(panel)

        if frame_idx % 300 == 0:
            pct = 100 * frame_idx / max(total_frames, 1)
            print(f"    frame {frame_idx}/{total_frames}  ({pct:.0f}%)  valid so far: {n_valid}")

    cap.release()
    writer.release()

    # Write CSV
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"  → {csv_path.name}  ({len(rows)} frames, {n_valid} valid midlines)")
    print(f"  → {avi_path.name}")


# =============================================================================
# Entry point
# =============================================================================

def browse_target() -> Path | None:
    """Open a file/folder dialog and return the selected path."""
    root = tk.Tk()
    root.withdraw()

    choice = messagebox.askquestion(
        "Select input",
        "Select a SESSION FOLDER to process all fish?\n\n"
        "(Choose 'No' to pick a single individual_*.avi file instead)",
        icon="question"
    )
    if choice == "yes":
        path = filedialog.askdirectory(title="Select idtracker.ai session folder")
    else:
        path = filedialog.askopenfilename(
            title="Select individual fish video",
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
        )
    root.destroy()
    return Path(path) if path else None


def main():
    target = browse_target()
    if target is None:
        print("No file selected — exiting.")
        sys.exit(0)

    # Collect video files to process
    if target.is_dir():
        # Look for individual_videos/ subfolder first, then the folder itself
        search_dir = target / "individual_videos"
        if not search_dir.exists():
            search_dir = target
        videos = sorted(search_dir.glob("individual_*.avi"))
        if not videos:
            print(f"No individual_*.avi files found in {search_dir}")
            sys.exit(1)
        output_dir = target / "posture_output"
    else:
        videos    = [target]
        output_dir = target.parent / "posture_output"

    output_dir.mkdir(exist_ok=True)
    print(f"Output folder: {output_dir}")
    print(f"Videos to process: {len(videos)}\n")

    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {vp.name}")
        process_video(vp, output_dir)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
