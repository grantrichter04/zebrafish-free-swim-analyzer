"""Tests for the pure overlay compositor.

These need no display and no video file, so unlike the GUI tests they run
everywhere. cv2 is required for the drawing path and skipped if absent.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

cv2 = pytest.importorskip("cv2")

from fish_analyzer.overlay_render import (  # noqa: E402
    OverlaySettings,
    compose_frame,
    fish_colors,
)


def blank(h=100, w=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


def one_fish_at(x, y, n_frames=1):
    """Trajectories array of shape (n_frames, 1, 2)."""
    traj = np.full((n_frames, 1, 2), np.nan)
    traj[:, 0, 0] = x
    traj[:, 0, 1] = y
    return traj


def test_all_overlays_off_returns_the_frame_unchanged():
    base = blank()
    base[10, 10] = (7, 8, 9)
    out = compose_frame(base, one_fish_at(50, 50), 0, OverlaySettings(), 1.0)
    assert np.array_equal(out, base)


def test_compose_frame_does_not_mutate_the_input():
    base = blank()
    before = base.copy()
    settings = OverlaySettings(show_positions=True, dot_radius=20)
    compose_frame(base, one_fish_at(50, 50), 0, settings, 1.0)
    assert np.array_equal(base, before)


def test_fish_dot_uses_the_tab10_colour_in_rgb_order():
    """Regression test for the BGR/RGB channel swap.

    The frame is RGB (video_utils converts BGR2RGB on read and PIL expects
    RGB), so fish 0 must be drawn tab10 blue (31, 119, 180). Before the fix it
    was drawn (180, 119, 31) - orange - which did not match the colour the
    Individual Analysis plots use for the same fish.
    """
    settings = OverlaySettings(show_positions=True, dot_radius=20)
    out = compose_frame(blank(), one_fish_at(50, 50), 0, settings, 1.0)

    # 15px above centre: inside the r=20 disc, clear of the 2px border and of
    # the fish-number glyph drawn at the centre.
    assert tuple(int(v) for v in out[35, 50]) == (31, 119, 180)


def test_untracked_fish_draws_nothing_and_does_not_raise():
    traj = np.full((1, 1, 2), np.nan)
    settings = OverlaySettings(show_positions=True, dot_radius=20)
    out = compose_frame(blank(), traj, 0, settings, 1.0)
    assert out.sum() == 0


def test_fish_colors_matches_the_app_wide_convention():
    """Same sampling as analysis_tab.py:924, so per-fish colours agree."""
    import matplotlib.pyplot as plt

    expected = plt.cm.tab10(np.linspace(0, 1, 6))
    assert np.allclose(fish_colors(6), expected)
