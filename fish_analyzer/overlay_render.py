"""
fish_analyzer/overlay_render.py
===============================
Pure overlay compositing for the Video Inspector.

Deliberately free of tkinter: the live view and the clip exporter both call
compose_frame, so what gets exported cannot drift from what is displayed, and
the drawing can be tested without a display.

All frames are RGB uint8. VideoFrameReader converts BGR2RGB on read and PIL
expects RGB, so there is no BGR anywhere in this module.
"""
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from .shoaling import nearest_neighbour_distances

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass(frozen=True)
class OverlaySettings:
    """Everything the compositor needs to know, snapshotted from the GUI.

    Frozen so an export cannot change appearance halfway through because a
    checkbox was toggled while it ran.
    """
    show_positions: bool = False
    show_nnd: bool = False
    show_hull: bool = False
    show_iid: bool = False
    iid_focus: int = 0
    show_bout_ring: bool = False
    bout_fish: int = 0
    trail_length: int = 0
    trail_opacity: float = 0.6
    trail_width: float = 1.0
    dot_radius: int = 6


def fish_colors(n_fish: int) -> np.ndarray:
    """Per-fish RGBA colours, matching the convention used across the app.

    analysis_tab.py, bout_tab.py, shoaling_tab.py and spatial_tab.py all sample
    tab10 this way, so a fish keeps one colour between the video and the plots.
    """
    return plt.cm.tab10(np.linspace(0, 1, max(1, n_fish)))


def rgb_uint8(rgba) -> tuple:
    """matplotlib RGBA floats (0-1) to an RGB 0-255 tuple."""
    return (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))


def compose_frame(base, trajectories, frame_idx, settings, scale, colors=None):
    """Draw the overlays for one frame onto a copy of `base`.

    Parameters
    ----------
    base : np.ndarray
        RGB uint8 frame, shape (h, w, 3). Not modified.
    trajectories : np.ndarray
        Shape (n_frames, n_fish, 2) in pixel coordinates. NaN where untracked.
    frame_idx : int
    settings : OverlaySettings
    scale : float
        calibration.scale_factor - pixels to the calibrated unit, used for the
        distance labels.
    colors : np.ndarray, optional
        Result of fish_colors(n_fish). Passed in by callers that render many
        frames so it is not recomputed per frame.

    Returns
    -------
    np.ndarray
        A new RGB uint8 array.
    """
    display = base.copy()
    n_fish = trajectories.shape[1]
    if colors is None:
        colors = fish_colors(n_fish)

    positions = trajectories[frame_idx]

    if CV2_AVAILABLE:
        _draw_cv2(display, trajectories, positions, frame_idx, n_fish,
                  colors, settings, scale)
    else:
        _draw_numpy(display, positions, n_fish, colors, settings)
    return display


def _draw_numpy(display, positions, n_fish, colors, settings):
    """Minimal fallback so the inspector still shows something without cv2."""
    if not settings.show_positions:
        return
    r = settings.dot_radius
    for i in range(n_fish):
        if np.isnan(positions[i, 0]):
            continue
        px, py = int(positions[i, 0]), int(positions[i, 1])
        y_grid, x_grid = np.ogrid[-r:r + 1, -r:r + 1]
        mask = x_grid ** 2 + y_grid ** 2 <= r ** 2
        y_start, y_end = max(0, py - r), min(display.shape[0], py + r + 1)
        x_start, x_end = max(0, px - r), min(display.shape[1], px + r + 1)
        mask_y = slice(max(0, r - py),
                       r + 1 + min(0, display.shape[0] - py - r - 1))
        mask_x = slice(max(0, r - px),
                       r + 1 + min(0, display.shape[1] - px - r - 1))
        display[y_start:y_end, x_start:x_end][mask[mask_y, mask_x]] = \
            rgb_uint8(colors[i])


def _draw_cv2(display, trajectories, positions, frame_idx, n_fish, colors,
              settings, scale):
    """Draw every enabled overlay directly onto the array."""
    _draw_positions(display, positions, n_fish, colors, settings)


def _draw_positions(display, positions, n_fish, colors, settings):
    if not settings.show_positions:
        return
    r = settings.dot_radius
    for i in range(n_fish):
        if np.isnan(positions[i, 0]):
            continue
        px, py = int(positions[i, 0]), int(positions[i, 1])
        color = rgb_uint8(colors[i])
        cv2.circle(display, (px, py), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(display, (px, py), r, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        font_scale = max(0.3, r / 20.0)
        text = str(i)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, 1)
        cv2.putText(display, text, (px - tw // 2, py + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
