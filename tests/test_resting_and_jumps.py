"""The Video Inspector's rings: which frames the analysis counts a fish as
resting, and which it dropped as tracking jumps."""
import numpy as np
import pytest

from fish_analyzer.overlay_render import (JUMP_COLOR, RESTING_COLOR,
                                          OverlaySettings, compose_frame)
from fish_analyzer.processing import (ProcessingParameters,
                                      process_and_analyze_file, resting_frames)

from synthetic_tracks import FPS, make_file


def _still_then_swimming(n_still=120, n_swim=120):
    """A fish sitting still for 4 s, then swimming 3 px a frame."""
    still = np.tile([[300.0, 300.0]], (n_still, 1))
    swim = np.stack([300.0 + 3.0 * np.arange(1, n_swim + 1),
                     np.full(n_swim, 300.0)], axis=1)
    return np.concatenate([still, swim])[:, None, :]


def test_resting_frames_are_the_frames_the_freeze_metric_counts():
    loaded = make_file(_still_then_swimming())
    fish = process_and_analyze_file(loaded)[0]

    resting = resting_frames(fish, ProcessingParameters())

    assert resting[5:115].all(), "sitting still is resting"
    assert not resting[130:].any(), "swimming is not"
    assert resting.sum() / FPS == pytest.approx(
        fish.metrics["freeze_total_duration_s"])


def test_a_higher_threshold_counts_slow_swimming_as_resting():
    loaded = make_file(_still_then_swimming())
    fish = process_and_analyze_file(loaded)[0]
    # 3 px a frame at 50 px per BL and 30 fps is 1.8 BL/s.
    assert not resting_frames(fish, ProcessingParameters())[130:].any()
    assert resting_frames(fish, ProcessingParameters(
        rest_speed_threshold=2.0))[130:].all()


def test_the_frames_dropped_as_jumps_are_kept_by_number():
    traj = _still_then_swimming()
    traj[200:, 0, 0] += 400.0                    # one leap of 8 body lengths
    fish = process_and_analyze_file(make_file(traj))[0]

    assert list(fish.jump_frames) == [200]
    assert fish.n_jumps == 1


@pytest.mark.parametrize("name, color", [("resting", RESTING_COLOR),
                                         ("jumps", JUMP_COLOR)])
def test_a_ring_is_drawn_only_where_the_state_is_true(name, color):
    pytest.importorskip("cv2")
    base = np.full((200, 200, 3), 200, dtype=np.uint8)
    traj = np.array([[[50.0, 100.0], [150.0, 100.0]]])
    state = np.array([[True, False]])
    settings = OverlaySettings(show_positions=True, dot_radius=6,
                               show_resting=True, show_jumps=True)

    out = compose_frame(base, traj, 0, settings, 1.0, states={name: state})

    def has_ring_colour(x0, x1):
        patch = out[70:130, x0:x1].reshape(-1, 3)
        return bool((np.abs(patch.astype(int) - color).sum(axis=1) < 30).any())

    assert has_ring_colour(20, 80), "the fish in that state has a ring"
    assert not has_ring_colour(120, 180), "the other fish has none"

    off = compose_frame(base, traj, 0, OverlaySettings(show_positions=True),
                        1.0, states={name: state})
    assert not (np.abs(off[70:130, 20:80].reshape(-1, 3).astype(int) - color)
                .sum(axis=1) < 30).any(), "no ring unless asked for"


def test_a_trail_stops_at_a_jump_instead_of_crossing_the_tank():
    pytest.importorskip("cv2")
    base = np.full((200, 400, 3), 200, dtype=np.uint8)
    # A track at the left that hands over to a fish at the right at frame 5.
    traj = np.array([[[20.0 + i, 100.0]] for i in range(5)]
                    + [[[360.0 + i, 100.0]] for i in range(5)])
    jumps = np.zeros((10, 1), dtype=bool)
    jumps[5, 0] = True
    settings = OverlaySettings(trail_length=10, trail_opacity=1.0)

    joined = compose_frame(base, traj, 9, settings, 1.0)
    cut = compose_frame(base, traj, 9, settings, 1.0, states={"jumps": jumps})

    middle = (slice(95, 106), slice(150, 250))
    assert (joined[middle] != 200).any(), "uncut, the trail crosses the tank"
    assert (cut[middle] == 200).all(), "cut at the jump, it does not"


def test_without_identities_the_dots_have_no_number_and_one_colour():
    pytest.importorskip("cv2")
    from fish_analyzer.overlay_render import ANONYMOUS_COLOR
    base = np.full((200, 200, 3), 120, dtype=np.uint8)
    traj = np.array([[[50.0, 100.0], [150.0, 100.0]]])
    settings = OverlaySettings(show_positions=True, dot_radius=12)

    out = compose_frame(base, traj, 0, settings, 1.0, labels=["1", "2"],
                        anonymous=True)

    for x in (50, 150):
        assert tuple(out[100, x]) == ANONYMOUS_COLOR, "same colour, no number on it"


def test_a_tank_rests_for_the_share_of_its_tracked_time_its_tracks_rest(tmp_path):
    from test_no_identities import _session
    from fish_analyzer import TrajectoryFileLoader, results
    # Two tracks: one sits still the whole time, one swims the whole time.
    still = _still_then_swimming(n_still=240, n_swim=0)[:, 0]
    swim = _still_then_swimming(n_still=1, n_swim=239)[:, 0] + [0.0, 300.0]
    loaded = TrajectoryFileLoader.load_from_session_folder(
        _session(tmp_path, np.stack([still, swim], axis=1)), "tank")
    loaded.processed_data = process_and_analyze_file(loaded)

    row = results.results_table({"tank": loaded}).iloc[0]

    assert row["RestingTime"] == pytest.approx(50.0, abs=1.0)
