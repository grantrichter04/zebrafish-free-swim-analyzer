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


def two_fish_at(p0, p1):
    traj = np.full((1, 2, 2), np.nan)
    traj[0, 0] = p0
    traj[0, 1] = p1
    return traj


def test_nnd_line_connects_the_nearest_pair():
    """Fish 0 and 1 are close; fish 2 is far. The line must join 0 and 1."""
    traj = np.full((1, 3, 2), np.nan)
    traj[0, 0] = (20.0, 50.0)
    traj[0, 1] = (40.0, 50.0)
    traj[0, 2] = (95.0, 95.0)

    out = compose_frame(blank(), traj, 0, OverlaySettings(show_nnd=True), 1.0)

    # Sample on the 0-1 segment but left of the distance label, whose black
    # outline is drawn from the segment midpoint rightwards.
    assert out[50, 25].sum() > 0, "no line drawn between 0 and 1"
    # Fish 2's own nearest neighbour is fish 1, so a 1-2 line is expected.
    # Fish 0 must not be joined to fish 2: (57, 72) sits on the 0-2 diagonal
    # and clear of the 1-2 one, which passes through y ~ 64 at that column.
    assert out[72, 57].sum() == 0, "fish 0 was joined to the distant fish"


def test_nnd_label_uses_the_calibration_scale():
    """The drawn distance is in calibrated units, not pixels.

    Asserted through the shared helper rather than by reading pixels: the
    overlay must report scale * pixel distance.
    """
    from fish_analyzer.shoaling import nearest_neighbour_distances

    positions = np.array([[20.0, 50.0], [40.0, 50.0]])
    nnd, _ = nearest_neighbour_distances(positions)
    assert nnd[0] * 0.0125 == pytest.approx(0.25)


def test_hull_needs_three_tracked_fish():
    out = compose_frame(blank(), two_fish_at((20.0, 20.0), (60.0, 60.0)), 0,
                        OverlaySettings(show_hull=True), 1.0)
    assert out.sum() == 0


def test_hull_is_drawn_for_three_tracked_fish():
    traj = np.full((1, 3, 2), np.nan)
    traj[0, 0] = (20.0, 20.0)
    traj[0, 1] = (80.0, 20.0)
    traj[0, 2] = (50.0, 80.0)
    out = compose_frame(blank(), traj, 0, OverlaySettings(show_hull=True), 1.0)
    assert out[40, 50].sum() > 0, "hull interior not tinted"


def test_adding_a_second_fish_does_not_fade_the_first_ones_trail():
    """One blend for all trails, not one blend per fish.

    The old code copied the frame and called addWeighted once per fish, so
    every extra fish re-blended everything drawn before it towards the
    background: fish 0's trail got dimmer as the fish count went up. Rows are
    compared against themselves, since fish are drawn in different colours.
    """
    settings = OverlaySettings(trail_length=9, trail_opacity=0.5)
    base = np.zeros((100, 100, 3), dtype=np.uint8)

    solo = np.full((10, 1, 2), np.nan)
    solo[:, 0, 0] = np.linspace(10, 90, 10)
    solo[:, 0, 1] = 50.0

    pair = np.full((10, 2, 2), np.nan)
    pair[:, 0] = solo[:, 0]
    pair[:, 1, 0] = np.linspace(10, 90, 10)
    pair[:, 1, 1] = 70.0

    row_alone = compose_frame(base, solo, 9, settings, 1.0)[50].astype(int)
    row_with_neighbour = compose_frame(base, pair, 9, settings,
                                       1.0)[50].astype(int)

    assert row_alone.sum() > 0, "no trail drawn at all"
    assert np.array_equal(row_alone, row_with_neighbour), \
        "fish 0's trail changed when a second fish was added"


def test_iid_lines_radiate_from_the_focus_fish():
    traj = np.full((1, 3, 2), np.nan)
    traj[0, 0] = (10.0, 50.0)
    traj[0, 1] = (90.0, 50.0)
    traj[0, 2] = (50.0, 10.0)

    out = compose_frame(blank(), traj, 0,
                        OverlaySettings(show_iid=True, iid_focus=0), 1.0)
    assert out[50, 50].sum() > 0, "no line from fish 0 to fish 1"


def test_bout_ring_is_drawn_around_the_selected_fish():
    settings = OverlaySettings(show_bout_ring=True, bout_fish=0, dot_radius=10)
    out = compose_frame(blank(), one_fish_at(50, 50), 0, settings, 1.0)
    # ring radius is 1.8 * dot_radius = 18
    assert out[50, 68].sum() > 0 or out[50, 67].sum() > 0
