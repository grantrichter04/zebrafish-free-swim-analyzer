"""Sessions idtracker.ai tracked without identities: recognised on loading,
cut at handovers, and reported as one value per tank."""
import json

import numpy as np
import pytest

from fish_analyzer import TrajectoryFileLoader, process_and_analyze_file, results

FPS = 30.0
BODY = 50.0


def _session(tmp_path, positions, without_identities=True, fish_in_log=2):
    """A session folder as idtracker.ai writes it, holding `positions`."""
    folder = tmp_path / "session_tank"
    (folder / "trajectories").mkdir(parents=True)
    n_tracks = positions.shape[1]
    np.save(folder / "trajectories" / "trajectories.npy", np.array({
        "trajectories": positions, "width": 1000, "height": 1000,
        "frames_per_second": FPS, "body_length": BODY,
        "estimated_accuracy": 1.0, "fraction_identified": 1.0,
        "identities_labels": [str(i + 1) for i in range(n_tracks)],
        "version": "6.0.8"}, dtype=object), allow_pickle=True)
    # Without identities idtracker.ai stores the most blobs it saw here, not
    # the number of fish it was given.
    (folder / "session.json").write_text(json.dumps({
        "track_wo_identities": without_identities,
        "number_of_animals": n_tracks}), encoding="utf-8")
    if fish_in_log is not None:
        (folder / "idtrackerai.log").write_text(
            "  number_of_animals=%d\n  use_bkg=True\n" % fish_in_log,
            encoding="utf-8")
    return folder


def _swimmers(n=900, step=2.0):
    """Two fish swimming steadily along x, one above the other, in pixels."""
    t = np.arange(n, dtype=float)
    a = np.stack([100 + step * t, np.full(n, 300.0)], axis=1)
    b = np.stack([100 + step * t, np.full(n, 700.0)], axis=1)
    return np.stack([a, b], axis=1)


def test_a_session_tracked_without_identities_is_recognised(tmp_path):
    loaded = TrajectoryFileLoader.load_from_session_folder(
        _session(tmp_path, _swimmers()), "tank")

    assert not loaded.with_identities
    assert loaded.fish_in_tank == 2


def test_the_fish_count_comes_from_the_log_not_the_columns(tmp_path):
    # Three columns: in one stretch idtracker.ai saw a third blob.
    positions = np.concatenate([_swimmers(), np.full((900, 1, 2), np.nan)], axis=1)
    positions[10:12, 2] = [500.0, 500.0]

    loaded = TrajectoryFileLoader.load_from_session_folder(
        _session(tmp_path, positions, fish_in_log=2), "tank")

    assert loaded.n_fish == 3
    assert loaded.fish_in_tank == 2


def test_without_a_log_the_fish_count_is_the_columns(tmp_path):
    loaded = TrajectoryFileLoader.load_from_session_folder(
        _session(tmp_path, _swimmers(), fish_in_log=None), "tank")

    assert loaded.fish_in_tank == 2


def test_a_session_tracked_with_identities_is_read_as_fish(tmp_path):
    loaded = TrajectoryFileLoader.load_from_session_folder(
        _session(tmp_path, _swimmers(), without_identities=False), "tank")
    loaded.processed_data = process_and_analyze_file(loaded)

    assert loaded.with_identities
    assert list(results.results_table({"tank": loaded})["Fish"]) == ["1", "2"]


def test_a_handover_adds_no_distance_and_no_speed(tmp_path):
    # At frame 450 the two tracks swap fish: each jumps 400 px, 8 body lengths.
    positions = _swimmers()
    positions[450:] = positions[450:, ::-1]
    loaded = TrajectoryFileLoader.load_from_session_folder(
        _session(tmp_path, positions), "tank")
    loaded.processed_data = process_and_analyze_file(loaded)

    row = results.results_table({"tank": loaded}).iloc[0]

    assert row["Jumps"] == 2, "one jump on each track"
    # 2 px a frame at 50 px per BL and 30 fps is 1.2 BL/s, every frame.
    assert row["Speed99"] == pytest.approx(1.2, abs=0.01)
    assert row["DistancePerTrackedMin"] == pytest.approx(72.0, abs=0.5)


def test_the_tank_is_one_row_pooling_every_track(tmp_path):
    positions = _swimmers()
    positions[:300, 1] = np.nan          # one track is short; it still counts
    loaded = TrajectoryFileLoader.load_from_session_folder(
        _session(tmp_path, positions), "tank")
    loaded.processed_data = process_and_analyze_file(loaded)

    table = results.results_table({"tank": loaded})

    assert list(table["Fish"]) == [results.POOLED]
    assert loaded.excluded_fish == {}
    tracked = (900 + 600) / (2 * 900) * 100
    assert table["Tracked_pct"].iloc[0] == pytest.approx(tracked, abs=0.1)
    # Distance is everything swum, shared over the two fish.
    swum = (899 + 599 - 2) * 2.0 / BODY      # minus the frames lost to smoothing
    assert table["Distance"].iloc[0] == pytest.approx(swum / 2, rel=0.01)


def test_the_tank_gets_one_speed_curve_and_one_line_per_minute(tmp_path):
    loaded = TrajectoryFileLoader.load_from_session_folder(
        _session(tmp_path, _swimmers()), "tank")
    loaded.processed_data = process_and_analyze_file(loaded)

    samples = results.speed_samples({"tank": loaded})
    minutes = results.minute_table({"tank": loaded})

    assert [s.fish for s in samples] == [results.POOLED]
    assert set(minutes["Fish"]) == {results.POOLED}
    assert len(minutes) == results.bin_count(900 / FPS)


def test_distance_is_shown_per_tracked_minute_once_a_tank_is_pooled(
        tmp_path, synthetic_npy):
    with_ids = TrajectoryFileLoader.load_file(synthetic_npy, "fish")
    with_ids.processed_data = process_and_analyze_file(with_ids)
    assert results.metrics_for({"fish": with_ids})[0].column == "Distance"

    tank = TrajectoryFileLoader.load_from_session_folder(
        _session(tmp_path, _swimmers()), "tank")
    tank.processed_data = process_and_analyze_file(tank)

    shown = results.metrics_for({"fish": with_ids, "tank": tank})
    assert shown[0].column == "DistancePerTrackedMin"
    assert [m.column for m in shown[1:]] == [m.column for m in results.METRICS[1:]]
