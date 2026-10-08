"""The headline results table and its SuperPlot."""
import shutil

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from fish_analyzer import TrajectoryFileLoader, process_and_analyze_file, results


def _analysed(npy, name):
    loaded = TrajectoryFileLoader.load_file(npy, name)
    loaded.processed_data = process_and_analyze_file(loaded)
    return loaded


@pytest.fixture
def two_groups(synthetic_npy, synthetic_npy_larger_fish):
    return {
        "control_1": _analysed(synthetic_npy, "control_1"),
        "control_2": _analysed(synthetic_npy_larger_fish, "control_2"),
        "treated_1": _analysed(synthetic_npy, "treated_1"),
    }


def test_table_has_one_row_per_fish_with_the_headline_metrics(two_groups):
    table = results.results_table(two_groups)

    assert len(table) == 9
    assert list(table.columns) == [
        "Group", "Session", "Fish", "Tracked_pct", "Distance", "TypicalSpeed",
        "PeakSpeed", "Straightness", "NearWall", "WallZoneArea_pct", "Unit",
        "PixelsPerUnit"]
    assert table["NearWall"].isna().all(), "no arena, so no wall time yet"
    assert set(table["Group"]) == {"control", "treated"}, \
        "ungrouped sessions fall into the group their name implies"
    first = two_groups["control_1"].processed_data[0]
    row = table.iloc[0]
    assert row["Distance"] == first.metrics["total_distance"]
    assert row["TypicalSpeed"] == first.metrics["median_speed"]
    assert row["PeakSpeed"] == first.metrics["speed_p99"]
    assert row["PeakSpeed"] >= row["TypicalSpeed"]
    assert 0 <= row["Straightness"] <= 1


def test_chosen_groups_override_the_names(two_groups):
    table = results.results_table(two_groups, {"control_2": "treated"})
    assert dict(zip(table["Session"], table["Group"]))["control_2"] == "treated"


def test_sessions_that_were_not_analysed_are_left_out(two_groups):
    two_groups["treated_1"].processed_data = None
    assert set(results.results_table(two_groups)["Session"]) == {"control_1", "control_2"}
    assert results.results_table({}).empty


def test_the_group_line_is_the_mean_of_session_means_not_of_fish():
    """The tank is the unit. Pooling fish would let a tank with more fish, or
    one odd fish, pull the group estimate around."""
    table = pd.DataFrame({
        "Group": ["g"] * 5, "Session": ["a", "a", "a", "a", "b"],
        "Fish": list("12345"), "Tracked_pct": [100.0] * 5,
        "Distance": [10.0, 10.0, 10.0, 10.0, 50.0],
        "TypicalSpeed": [1.0] * 5, "PeakSpeed": [2.0] * 5,
        "Straightness": [0.5] * 5, "NearWall": [60.0] * 5,
        "WallZoneArea_pct": [51.0] * 5, "Unit": ["cm"] * 5,
        "PixelsPerUnit": [30.0] * 5,
    })
    means = results.session_means(table)
    assert list(means["Distance"]) == [10.0, 50.0]

    ax = Figure().subplots()
    results.superplot(ax, table, results.METRICS[0],
                      {"a": (0, 0, 1, 1), "b": (1, 0, 0, 1)})

    line = [c for c in ax.collections if type(c).__name__ == "LineCollection"][0]
    assert line.get_segments()[0][0][1] == 30.0, \
        "mean of the two session means (10, 50), not of the five fish (18)"
    dots = [c for c in ax.collections if type(c).__name__ == "PathCollection"]
    assert sorted(len(c.get_offsets()) for c in dots) == [1, 1, 1, 4], \
        "fish dots for each session, plus one large marker per session"
    assert ax.get_ylabel() == "cm"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["g"]


def test_speeds_are_labelled_per_second_and_straightness_has_no_unit():
    labels = {m.column: m.axis_label("BL") for m in results.METRICS}
    assert labels == {"Distance": "BL", "TypicalSpeed": "BL/s",
                      "PeakSpeed": "BL/s", "Straightness": "0 to 1",
                      "NearWall": "% of time"}
    assert all(len(m.meaning) > 40 for m in results.METRICS), \
        "every measure says what it is"


# --- the tab --------------------------------------------------------------------

def test_results_tab_fills_after_run_all_analysis(
        app, tmp_path, synthetic_npy, synthetic_npy_larger_fish, monkeypatch):
    from fish_analyzer.gui import data_tab
    for name in ("showinfo", "showerror"):
        monkeypatch.setattr(data_tab.messagebox, name, lambda *a, **k: None)
    monkeypatch.setattr(app, "_report_batch_outcome", lambda *a, **k: None)
    for name, source in (("control_1", synthetic_npy),
                         ("treated_1", synthetic_npy_larger_fish)):
        target = tmp_path / f"session_{name}" / "trajectories"
        target.mkdir(parents=True)
        shutil.copy(source, target / "trajectories.npy")
    assert app.results_export_button["state"] == "disabled"
    assert "No results yet" in app.results_note_var.get()

    app._add_path(tmp_path)
    app._run_analysis_and_switch_tab()

    assert app.notebook.select() == str(app.results_tab_frame)
    rows = [app.results_tree.item(i, "values") for i in app.results_tree.get_children()]
    assert len(rows) == 6
    assert {r[0] for r in rows} == {"control", "treated"}
    assert "6 fish in 2 session(s), 2 group(s)" in app.results_note_var.get()
    assert "one session cannot be tested" in app.results_note_var.get()
    assert app.results_tree.heading("Distance")["text"] == "Distance (BL)"
    assert app.results_export_button["state"] == "normal"

    out = tmp_path / "out.csv"
    monkeypatch.setattr("fish_analyzer.gui.results_tab.ask_csv_save_path",
                        lambda *a: out)
    monkeypatch.setattr("fish_analyzer.gui.results_tab.messagebox.showinfo",
                        lambda *a, **k: None)
    app._export_results()
    exported = pd.read_csv(out)
    assert len(exported) == 6 and "PixelsPerUnit" in exported.columns
    assert np.allclose(exported["Distance"], app._results_table["Distance"])

    # Changing units throws the results away, and the tab says so.
    app.units_choice.set("cm")
    app.cm_pixels_var.set("300")
    app.cm_length_var.set("10")
    app._apply_units()
    assert not app.results_tree.get_children()
    assert "No results yet" in app.results_note_var.get()


# --- speed distributions ---------------------------------------------------------

def test_speed_samples_are_per_fish_with_gaps_removed(two_groups):
    samples = results.speed_samples(two_groups)

    assert len(samples) == 9
    assert {s.session for s in samples} == set(two_groups)
    assert all(not np.isnan(s.speeds).any() and len(s.speeds) > 100 for s in samples)
    assert samples[0].group == "control" and samples[0].unit == "BL"


def test_a_fish_that_sits_still_shows_as_a_peak_at_zero():
    """The reason to look at distributions at all: the mean of this fish looks
    unremarkable, its distribution does not."""
    rng = np.random.default_rng(1)
    swimmer = results.SpeedSample("g", "tank", "1", rng.normal(3, 0.5, 6000).clip(0), "BL")
    sitter = results.SpeedSample(
        "g", "tank", "2",
        np.concatenate([rng.normal(0.1, 0.05, 3000).clip(0),
                        rng.normal(3, 0.5, 3000).clip(0)]), "BL")
    bins = results._speed_bins([swimmer, sitter])

    still = results._density(sitter.speeds, bins)[:5].sum()
    moving = results._density(swimmer.speeds, bins)[:5].sum()
    assert still > 20 * max(moving, 1e-9)

    figure = Figure()
    left, right = figure.subplots(1, 2)
    colors = {"tank": (0, 0, 1, 1)}
    results.plot_session_speed_ecdf(left, [swimmer, sitter], colors)
    results.plot_fish_speed_ridges(right, [swimmer, sitter], colors)
    curves = [line for line in left.lines if len(line.get_xdata()) > 2]
    assert len(curves) == 1, "one cumulative curve per session"
    x, y = curves[0].get_xdata(), curves[0].get_ydata()
    assert y[0] == 0 and y[-1] == 100 and np.all(np.diff(x) >= 0)
    pooled = np.concatenate([swimmer.speeds, sitter.speeds])
    assert x[np.searchsorted(y, 50)] == pytest.approx(np.median(pooled), rel=0.02), \
        "where the curve crosses 50% is the typical speed"
    assert [t.get_text() for t in right.get_yticklabels()] == [
        "tank \u00b7 1", "tank \u00b7 2"]
    assert left.get_xlabel() == "Speed (BL/s)"


def test_results_tab_switches_to_speed_distributions(
        app, tmp_path, synthetic_npy, monkeypatch):
    from fish_analyzer.gui import data_tab
    for name in ("showinfo", "showerror"):
        monkeypatch.setattr(data_tab.messagebox, name, lambda *a, **k: None)
    monkeypatch.setattr(app, "_report_batch_outcome", lambda *a, **k: None)
    target = tmp_path / "session_control_1" / "trajectories"
    target.mkdir(parents=True)
    shutil.copy(synthetic_npy, target / "trajectories.npy")
    app._add_path(tmp_path / "session_control_1")
    app._run_analysis()
    drawn = []
    monkeypatch.setattr(results, "plot_fish_speed_ridges",
                        lambda ax, samples, colors: drawn.append(len(samples)))

    app.results_view.set("distributions")
    app._draw_results_plot()

    assert drawn == [3]
    assert app.results_plot_frame.winfo_children(), "a figure was embedded"
    app.results_view.set("comparison")
    app._draw_results_plot()


# --- time near the wall -----------------------------------------------------------

def _session_with_arena(tmp_path, source_npy, name, roi_list):
    import json
    session = tmp_path / f"session_{name}"
    (session / "trajectories").mkdir(parents=True)
    shutil.copy(source_npy, session / "trajectories" / "trajectories.npy")
    (session / "session.json").write_text(json.dumps({"roi_list": roi_list}))
    return session


def test_the_arena_drawn_in_idtrackerai_is_read_from_the_session(tmp_path, synthetic_npy):
    from fish_analyzer.spatial import idtrackerai_arena

    session = _session_with_arena(
        tmp_path, synthetic_npy, "a",
        ["+ Polygon [[100, 200], [900, 200], [900, 800], [100, 800]]"])
    loaded = TrajectoryFileLoader.load_from_session_folder(session)

    arena = idtrackerai_arena(loaded)

    assert arena.vertices_pixels.tolist() == [[100, 200], [900, 200], [900, 800], [100, 800]]
    # In the session's unit (40 px per BL) with y pointing up (video is 1024 high).
    assert arena.vertices_bl[0].tolist() == [100 / 40, (1024 - 200) / 40]


def test_no_arena_is_assumed_when_the_session_does_not_have_exactly_one(
        tmp_path, synthetic_npy):
    from fish_analyzer.spatial import idtrackerai_arena

    for index, roi_list in enumerate((
            [], "",
            ["+ Polygon [[0,0],[5,0],[5,5]]", "+ Polygon [[9,9],[12,9],[12,12]]"],
            ["- Polygon [[0,0],[5,0],[5,5]]"], ["+ Blob [1, 2]"])):
        session = _session_with_arena(tmp_path, synthetic_npy, f"s{index}", roi_list)
        loaded = TrajectoryFileLoader.load_from_session_folder(session)
        assert idtrackerai_arena(loaded) is None, roi_list

    assert idtrackerai_arena(TrajectoryFileLoader.load_file(synthetic_npy)) is None, \
        "a bare trajectories file has no session.json at all"


def test_an_ellipse_outline_becomes_a_polygon(tmp_path, synthetic_npy):
    from fish_analyzer.spatial import idtrackerai_arena

    session = _session_with_arena(
        tmp_path, synthetic_npy, "round",
        ["+ Ellipse {'center': [500, 500], 'axes': [300, 200], 'angle': 0}"])
    arena = idtrackerai_arena(TrajectoryFileLoader.load_from_session_folder(session))

    xs, ys = arena.vertices_pixels[:, 0], arena.vertices_pixels[:, 1]
    assert (xs.min(), xs.max()) == pytest.approx((200, 800), abs=1)
    assert (ys.min(), ys.max()) == pytest.approx((300, 700), abs=1)


def test_run_all_analysis_fills_time_near_the_wall_from_that_arena(
        app, tmp_path, synthetic_npy, monkeypatch):
    """The synthetic fish wander around (400, 400) px. An arena of 0-800 px
    has a 120 px border, so they are in the centre almost all the time."""
    from fish_analyzer.gui import data_tab
    for name in ("showinfo", "showerror", "showwarning"):
        monkeypatch.setattr(data_tab.messagebox, name, lambda *a, **k: None)
    monkeypatch.setattr(app, "_report_batch_outcome", lambda *a, **k: None)
    session = _session_with_arena(
        tmp_path, synthetic_npy, "control_1",
        ["+ Polygon [[0, 0], [800, 0], [800, 800], [0, 800]]"])
    app._add_path(session)

    app._run_analysis()

    table = app._results_table
    assert table["NearWall"].between(0, 100).all()
    assert table["NearWall"].mean() < 20
    assert table["WallZoneArea_pct"].iloc[0] == pytest.approx(51.0, abs=0.1), \
        "a 15% border round a square is 1 - 0.7 x 0.7 of its area"
    assert "control_1" in app.file_arena_definitions, \
        "the Spatial tab sees the same arena"
    values = app.results_tree.item(app.results_tree.get_children()[0], "values")
    assert values[-1] != ""

    # The arena is held in the session's unit; changing the unit rescales it.
    before = app.file_arena_definitions["control_1"].vertices_bl.copy()
    app.units_choice.set("cm")
    app.cm_pixels_var.set("200")
    app.cm_length_var.set("10")
    app._apply_units()
    after = app.file_arena_definitions["control_1"].vertices_bl
    assert np.allclose(after, before * 40 / 20), "40 px/BL became 20 px/cm"


def test_wall_panel_says_so_when_there_is_no_arena(two_groups):
    table = results.results_table(two_groups)
    ax = Figure().subplots()

    results.superplot(ax, table, results.METRICS[-1], {})

    assert "No arena outline" in ax.texts[0].get_text()


def test_measures_are_explained_in_the_app(app):
    window = app._explain_measures()
    try:
        import tkinter as tk
        shown = next(w for w in window.winfo_children()
                     if isinstance(w, tk.Text)).get("1.0", "end")
        assert "divided by the" in shown and "Path straightness" in shown
        assert "99th percentile" in shown and "even" in shown
    finally:
        window.destroy()

