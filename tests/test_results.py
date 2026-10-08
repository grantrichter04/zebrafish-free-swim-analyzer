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
        "PeakSpeed", "Straightness", "Unit", "PixelsPerUnit"]
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
        "Straightness": [0.5] * 5, "Unit": ["cm"] * 5, "PixelsPerUnit": [30.0] * 5,
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
                      "PeakSpeed": "BL/s", "Straightness": "1 = straight"}


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
