"""The Shoaling tab and the summary behind it."""
import json
import shutil

import numpy as np
import pytest
from matplotlib.figure import Figure

from fish_analyzer import TrajectoryFileLoader, shoal_summary
from fish_analyzer.shoaling import ShoalingCalculator, ShoalingParameters

SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


def test_random_reference_matches_the_known_answer_for_a_square():
    """Two points placed at random in a unit square are 0.5214 apart on
    average (a textbook result). With two fish, NND and IID are that distance."""
    nnd, iid = shoal_summary.random_expectation(SQUARE, n_fish=2, n_draws=20000)

    assert nnd == pytest.approx(0.5214, abs=0.01)
    assert iid == pytest.approx(0.5214, abs=0.01)


def test_more_fish_have_nearer_neighbours_but_the_same_spread():
    nnd_2, iid_2 = shoal_summary.random_expectation(SQUARE, 2)
    nnd_6, iid_6 = shoal_summary.random_expectation(SQUARE, 6)

    assert nnd_6 < nnd_2 * 0.6, "with more fish, the closest one is closer"
    assert iid_6 == pytest.approx(iid_2, abs=0.02), \
        "the average distance to everyone does not depend on how many there are"
    assert shoal_summary.random_expectation(SQUARE, 6) == (nnd_6, iid_6), \
        "seeded, so the reference line does not move between redraws"


def test_random_reference_needs_two_fish_and_a_real_outline():
    assert shoal_summary.random_expectation(SQUARE, 1) is None
    assert shoal_summary.random_expectation(np.array([[0, 0], [1, 1]]), 3) is None


def _with_shoaling(npy, name):
    loaded = TrajectoryFileLoader.load_file(npy, name)
    loaded.shoaling_results = ShoalingCalculator(
        loaded, ShoalingParameters(30)).calculate()
    return loaded


def test_table_has_a_row_per_fish_and_summary_a_row_per_session(
        synthetic_npy, synthetic_npy_larger_fish):
    files = {"control_1": _with_shoaling(synthetic_npy, "control_1"),
             "treated_1": _with_shoaling(synthetic_npy_larger_fish, "treated_1"),
             "not_run": TrajectoryFileLoader.load_file(synthetic_npy, "not_run")}

    table = shoal_summary.shoaling_table(files)
    summary = shoal_summary.session_summary(table)

    assert len(table) == 6 and set(table["Session"]) == {"control_1", "treated_1"}
    assert (table["IID"] >= table["NND"] - 1e-9).all(), \
        "the average distance to everyone is never less than to the nearest"
    assert table["RandomNND"].isna().all(), "no arena, so no reference"
    shoal = files["control_1"].shoaling_results
    assert summary.loc[0, "NND"] == pytest.approx(shoal.mean_nnd, rel=1e-6)
    assert summary.loc[0, "Fish"] == 3
    assert list(summary["Group"]) == ["control", "treated"]


def test_over_time_draws_one_line_per_session(synthetic_npy, synthetic_npy_larger_fish):
    files = {"a": _with_shoaling(synthetic_npy, "a"),
             "b": _with_shoaling(synthetic_npy_larger_fish, "b")}
    ax = Figure().subplots()

    shoal_summary.plot_over_time(ax, files, "NND", "Nearest neighbour",
                                 {"a": (0, 0, 1, 1), "b": (1, 0, 0, 1)},
                                 smooth_seconds=5)

    assert [line.get_label() for line in ax.lines] == ["a", "b"]
    assert ax.get_ylabel() == "BL" and ax.get_xlabel() == "Time (minutes)"


# --- the tab ----------------------------------------------------------------------

def test_run_all_analysis_fills_the_shoaling_tab(
        app, tmp_path, synthetic_npy, monkeypatch):
    from fish_analyzer.gui import data_tab
    for name in ("showinfo", "showerror", "showwarning"):
        monkeypatch.setattr(data_tab.messagebox, name, lambda *a, **k: None)
    monkeypatch.setattr(app, "_report_batch_outcome", lambda *a, **k: None)
    session = tmp_path / "session_control_1"
    (session / "trajectories").mkdir(parents=True)
    shutil.copy(synthetic_npy, session / "trajectories" / "trajectories.npy")
    (session / "session.json").write_text(json.dumps(
        {"roi_list": ["+ Polygon [[0, 0], [800, 0], [800, 800], [0, 800]]"]}))
    assert "No results yet" in app.shoaling_note_var.get()
    assert app.shoaling_export_button["state"] == "disabled"

    app._add_path(session)
    app._run_analysis()

    loaded = app.loaded_files["control_1"]
    assert loaded.shoaling_results is not None, "shoaling ran with everything else"
    assert loaded.shoaling_results.sample_interval_frames == 30, "one sample a second"
    rows = [app.shoaling_tree.item(i, "values") for i in app.shoaling_tree.get_children()]
    assert len(rows) == 1 and rows[0][1] == "control_1" and rows[0][2] == "3"
    nnd, random_nnd = float(rows[0][3]), float(rows[0][4])
    assert nnd > 0 and random_nnd > 0
    assert app.shoaling_tree.heading("NND")["text"] == "Nearest neighbour (BL)"
    assert app.shoaling_export_button["state"] == "normal"
    assert app.shoaling_plot_frame.winfo_children(), "the comparison drew"

    app.shoaling_view.set("time")
    app._draw_shoaling_plot()
    assert app.shoaling_plot_frame.winfo_children(), "the over-time view drew"
    app.shoaling_view.set("comparison")

    # Changing units throws shoaling results away along with the rest.
    app.units_choice.set("cm")
    app.cm_pixels_var.set("300")
    app.cm_length_var.set("10")
    app._apply_units()
    assert not app.shoaling_tree.get_children()
    assert "No results yet" in app.shoaling_note_var.get()


def test_shoaling_measures_are_explained_in_the_app(app):
    import tkinter as tk
    window = app._explain_shoaling()
    try:
        shown = next(w for w in window.winfo_children()
                     if isinstance(w, tk.Text)).get("1.0", "end")
        assert "Nearest-neighbour distance" in shown
        assert "placed at" in shown and "every fish was tracked" in shown
    finally:
        window.destroy()
