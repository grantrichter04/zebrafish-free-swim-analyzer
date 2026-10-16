"""How each session was tracked travels with its results (Audit I, I3):
idtracker.ai's settings, what this app changed about the run, and whether
the session was finished early or reviewed."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from fish_analyzer import (TrajectoryFileLoader, process_and_analyze_file,
                           provenance, results_export, tracking)

from test_tracking import WRITES_TRAJECTORIES, _pretend_idtrackerai, _touch

SESSION_JSON = {
    "version": "6.0.14",
    "number_of_animals": 3,
    "intensity_ths": [36, 255],
    "area_ths": [250.0, float("inf")],
    "use_bkg": True,
    "background_subtraction_stat": "median",
    "resolution_reduction": 0.8,
    "tracking_intervals": [[0, 600]],
    "roi_list": ["+ Polygon [[0.0, 0.0], [1024.0, 0.0], [1024.0, 1024.0], [0.0, 1024.0]]"],
    "track_wo_identities": False,
    "last_validated": "2026-10-12T09:30:00",
}


def _session(root, name, log="", own=None, **changes):
    """A session folder as idtracker.ai and the Tracking tab leave it."""
    folder = root / f"session_{name}"
    (folder / "trajectories").mkdir(parents=True)
    return folder, {**SESSION_JSON, **changes}, log, own


def _write(folder, session_json, log, own, npy):
    shutil.copy(npy, folder / "trajectories" / "trajectories.npy")
    # json.dump writes inf as Infinity, as idtracker.ai's session.json does.
    (folder / "session.json").write_text(json.dumps(session_json), encoding="utf-8")
    (folder / "idtrackerai.log").write_text(log, encoding="utf-8")
    if own is not None:
        (folder / tracking.TRACKED_WITH_FILE).write_text(json.dumps(own),
                                                          encoding="utf-8")


def _loaded(folder, name):
    loaded = TrajectoryFileLoader.load_from_session_folder(folder)
    loaded.nickname = name
    loaded.processed_data = process_and_analyze_file(loaded)
    return loaded


@pytest.fixture
def two_sessions(synthetic_npy, tmp_path):
    early, *rest = _session(tmp_path, "early",
                            log="Training interrupted by user\n",
                            own={"analyzer_version": "2.3.0 (commit abc1234)",
                                 "rolling_background": True,
                                 "setup": "rig.toml",
                                 "setup_text": "intensity_ths = [36, 255]\n"})
    _write(early, *rest, synthetic_npy)
    plain, *rest = _session(tmp_path, "plain", last_validated=None)
    _write(plain, *rest, synthetic_npy)
    return {"early": _loaded(early, "early"), "plain": _loaded(plain, "plain")}


def test_the_export_has_a_row_of_tracking_settings_per_session(two_sessions, tmp_path):
    folder = results_export.export_results_folder(two_sessions, None, tmp_path)
    table = pd.read_csv(folder / "tracking_settings.csv").set_index("Session")

    early = table.loc["early"]
    assert early["IdtrackerVersion"] == "6.0.14"
    assert early["FishInTank"] == 3
    assert early["IntensityThreshold"] == "36-255"
    assert early["AreaThreshold"] == "250-inf"
    assert early["Background"] == "rolling median, 1 min"
    assert early["ResolutionReduction"] == 0.8
    assert early["TrackingIntervals"] == "0-600"
    assert early["RegionOfInterest"].startswith("+ Polygon [[0.0, 0.0], [1024.0, 0.0]")
    assert early["IdentityTraining"] == "finished early"
    assert early["ReviewedOn"] == "2026-10-12"
    assert early["TrackedBy"] == "Free Swim Analyzer 2.3.0 (commit abc1234)"
    assert early["Setup"] == "rig.toml"

    plain = table.loc["plain"]
    assert plain["IdentityTraining"] == "ran to the end"
    assert pd.isna(plain["ReviewedOn"])
    # Tracked before this was recorded, or outside the app: the background
    # is what idtracker.ai stored, and whether it was rolling is unknown.
    assert plain["Background"] == "median (rolling not recorded)"
    assert plain["TrackedBy"] == "not recorded"


def test_the_setup_used_is_saved_beside_the_export(two_sessions, tmp_path):
    folder = results_export.export_results_folder(two_sessions, None, tmp_path)
    assert (folder / "setups" / "early.toml").read_text(encoding="utf-8") == \
        "intensity_ths = [36, 255]\n"
    assert not (folder / "setups" / "plain.toml").exists()


def test_the_tank_file_carries_the_analysis_settings(two_sessions, tmp_path):
    folder = results_export.export_results_folder(two_sessions, None, tmp_path)
    tanks = pd.read_csv(folder / "results_per_tank.csv")
    per_fish = pd.read_csv(folder / "results_per_fish.csv")
    settings = [c for c in per_fish.columns if c.startswith("Setting_")]
    assert settings, "the fish file has the settings"
    for column in settings:
        assert column in tanks.columns
        assert set(tanks[column]) == set(per_fish[column].dropna())


def test_the_tank_outline_and_where_it_came_from_are_exported(two_sessions, tmp_path):
    """Time near the wall can be recomputed only if the outline is kept
    (Audit I, I4): "early" uses idtracker.ai's region of interest, "plain"
    one drawn in this app."""
    from fish_analyzer.spatial import (ThigmotaxisCalculator, arena_in_units,
                                       idtrackerai_arena)
    roi = idtrackerai_arena(two_sessions["early"])
    drawn = arena_in_units(np.array([[0, 0], [800, 0], [800, 800], [0, 800]]),
                           two_sessions["plain"])
    for name, arena in (("early", roi), ("plain", drawn)):
        two_sessions[name].thigmotaxis_results = ThigmotaxisCalculator(
            two_sessions[name], arena).calculate()

    folder = results_export.export_results_folder(two_sessions, None, tmp_path)
    outlines = pd.read_csv(folder / "tank_outlines.csv")

    early = outlines[outlines["Session"] == "early"]
    assert set(early["Source"]) == {"idtracker.ai region of interest"}
    assert early[["X_px", "Y_px"]].to_numpy().tolist() == [
        [0.0, 0.0], [1024.0, 0.0], [1024.0, 1024.0], [0.0, 1024.0]]
    plain = outlines[outlines["Session"] == "plain"]
    assert set(plain["Source"]) == {"drawn in Free Swim Analyzer"}
    assert len(plain) == 4
    assert set(outlines["WallZoneWidth_pct"]) == {15.0}


def test_tracking_records_how_the_app_ran_it(tmp_path, monkeypatch):
    video = _touch(tmp_path / "exp.avi")
    setup = tmp_path / "rig.toml"
    setup.write_text("use_bkg = true\nbackground_subtraction_stat = 'rolling median'\n",
                     encoding="utf-8")
    _pretend_idtrackerai(monkeypatch, WRITES_TRAJECTORIES)

    assert tracking.run_tracking(video, setup, lambda _: None).ok

    record = json.loads((tmp_path / "session_exp" / tracking.TRACKED_WITH_FILE)
                        .read_text(encoding="utf-8"))
    assert record == {"analyzer_version": provenance.analyzer_version(),
                      "rolling_background": True,
                      "setup": "rig.toml",
                      "setup_text": setup.read_text(encoding="utf-8"),
                      "tracking_seconds": record["tracking_seconds"]}
    assert record["tracking_seconds"] >= 0
    assert tracking.tracking_seconds(video) == record["tracking_seconds"]


def test_a_failed_run_records_nothing(tmp_path, monkeypatch):
    video = _touch(tmp_path / "exp.avi")
    _pretend_idtrackerai(monkeypatch, "print('no trajectories')")
    assert not tracking.run_tracking(video, tmp_path / "rig.toml", lambda _: None).ok
    assert not (tmp_path / "session_exp").exists()



def test_tracking_time_reads_none_when_not_recorded(tmp_path):
    video = _touch(tmp_path / "exp.avi")
    assert tracking.tracking_seconds(video) is None
    (tmp_path / "session_exp").mkdir()
    (tmp_path / "session_exp" / tracking.TRACKED_WITH_FILE).write_text(
        json.dumps({"setup": "rig.toml"}), encoding="utf-8")
    assert tracking.tracking_seconds(video) is None


def test_durations_read_as_clock_times():
    assert tracking.format_duration(65) == "1:05"
    assert tracking.format_duration(3 * 3600 + 4 * 60 + 9.6) == "3:04:10"
