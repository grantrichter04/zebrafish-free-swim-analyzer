"""The Sessions & Units tab: loading sessions, and keeping them on one scale."""
import shutil

import numpy as np
import pytest

from fish_analyzer.gui import data_tab, measure_dialog


def _make_session(folder, source_npy, name):
    target = folder / f"session_{name}" / "trajectories"
    target.mkdir(parents=True)
    shutil.copy(source_npy, target / "trajectories.npy")
    return target.parent


@pytest.fixture
def quiet(monkeypatch):
    """Record the tab's dialogs instead of showing them."""
    shown = {"info": [], "error": []}
    monkeypatch.setattr(data_tab.messagebox, "showinfo",
                        lambda title, text="", **k: shown["info"].append((title, text)))
    monkeypatch.setattr(data_tab.messagebox, "showerror",
                        lambda title, text="", **k: shown["error"].append((title, text)))
    monkeypatch.setattr(data_tab.messagebox, "askyesno", lambda *a, **k: True)
    return shown


def _rows(app):
    tree = app.sessions_tree
    return {i: dict(zip(tree["columns"], tree.item(i, "values")))
            for i in tree.get_children()}


# --- loading -------------------------------------------------------------------

def test_one_session_folder_loads_and_fills_the_table(
        app, quiet, tmp_path, synthetic_npy):
    session = _make_session(tmp_path, synthetic_npy, "control")

    assert app._add_path(session) == 1

    row = _rows(app)["control"]
    assert row["fish"] == "3"
    assert row["body"] == "40.0"
    assert row["fps"] == "30.00 fps"
    assert row["accuracy"] == "98.0%"
    assert row["reviewed"] == "", "never opened in the validator"
    assert quiet["info"] == [], "one session needs no confirmation dialog"


def test_a_folder_of_sessions_loads_them_all(
        app, quiet, tmp_path, synthetic_npy, synthetic_npy_larger_fish):
    _make_session(tmp_path, synthetic_npy, "control")
    _make_session(tmp_path, synthetic_npy_larger_fish, "treated")
    (tmp_path / "notes").mkdir()
    (tmp_path / "control.avi").write_bytes(b"")

    assert app._add_path(tmp_path) == 2

    assert list(app.loaded_files) == ["control", "treated"]
    assert "Loaded 2 session(s)" in quiet["info"][0][1]

    # Loading the same folder again adds nothing and says why.
    assert app._add_path(tmp_path) == 0
    assert "Already loaded" in quiet["info"][1][1]


def test_a_bare_trajectories_file_still_loads(app, quiet, synthetic_npy):
    assert app._add_path(synthetic_npy) == 1
    assert app.loaded_files["trajectories"].n_fish == 3


def test_a_folder_without_sessions_is_reported(app, quiet, tmp_path):
    assert app._add_path(tmp_path) == 0
    assert quiet["error"][0][0] == "No sessions found"
    assert not app.loaded_files


# --- one scale for every session ------------------------------------------------

def test_sessions_share_one_body_length_not_their_own(
        app, quiet, tmp_path, synthetic_npy, synthetic_npy_larger_fish):
    """Per-video body length moves with lighting and threshold, so two videos
    from one rig would be put on different physical scales."""
    _make_session(tmp_path, synthetic_npy, "control")              # 40 px
    _make_session(tmp_path, synthetic_npy_larger_fish, "treated")  # 52 px

    app._add_path(tmp_path)

    assert app.bl_pixels_var.get() == "46.0", "the median of 40 and 52"
    assert app.rest_threshold_var.get() == "0.5", \
        "the suggested body length moving must not rewrite the threshold"
    scales = {n: (f.calibration.unit_name, f.calibration.pixels_per_unit)
              for n, f in app.loaded_files.items()}
    assert scales == {"control": ("BL", 46.0), "treated": ("BL", 46.0)}
    assert "40.0 to 52.0 px" in app.bl_hint_label["text"]
    assert "for all 2 session(s)" in app.units_status_var.get()
    assert app.apply_units_button["state"] == "disabled"


def test_a_typed_body_length_survives_loading_another_session(
        app, quiet, tmp_path, synthetic_npy, synthetic_npy_larger_fish):
    first = _make_session(tmp_path, synthetic_npy, "control")
    app._add_path(first)
    app._bl_edited_by_user()
    app.bl_pixels_var.set("44")
    app._apply_units()

    app._add_path(_make_session(tmp_path, synthetic_npy_larger_fish, "treated"))

    assert app.bl_pixels_var.get() == "44"
    assert {f.calibration.pixels_per_unit for f in app.loaded_files.values()} == {44.0}


def test_centimetres_apply_to_every_session_and_discard_old_results(
        app, quiet, tmp_path, synthetic_npy, synthetic_npy_larger_fish):
    _make_session(tmp_path, synthetic_npy, "control")
    _make_session(tmp_path, synthetic_npy_larger_fish, "treated")
    app._add_path(tmp_path)
    app.loaded_files["control"].processed_data = ["computed in BL"]
    quiet["info"].clear()

    app.units_choice.set("cm")
    app.cm_pixels_var.set("680")
    app.cm_length_var.set("20")
    assert "not applied yet" in app.units_status_var.get()
    assert app.apply_units_button["state"] == "normal"

    assert app._apply_units()

    for loaded in app.loaded_files.values():
        assert loaded.calibration.unit_name == "cm"
        assert loaded.calibration.pixels_per_unit == 34.0
        assert loaded.calibration.frame_rate == 30.0
    assert app.loaded_files["control"].processed_data is None
    assert "discarded" in quiet["info"][0][1]
    assert app.rest_unit_label["text"] == "cm/s"


def test_freeze_threshold_keeps_its_physical_speed_when_units_change(
        app, quiet, tmp_path, synthetic_npy):
    """0.5 BL/s at 40 px/BL is 20 px/s. At 34 px/cm that is 0.588 cm/s, not
    0.5 cm/s; leaving the number alone would silently redefine 'frozen'."""
    app._add_path(_make_session(tmp_path, synthetic_npy, "control"))
    assert app.rest_threshold_var.get() == "0.5"

    app.units_choice.set("cm")
    app.cm_pixels_var.set("680")
    app.cm_length_var.set("20")
    app._apply_units()

    assert float(app.rest_threshold_var.get()) == pytest.approx(0.588, abs=0.001)


def test_centimetres_without_a_scale_is_refused_in_plain_words(
        app, quiet, tmp_path, synthetic_npy):
    app._add_path(_make_session(tmp_path, synthetic_npy, "control"))
    app.units_choice.set("cm")
    app._update_units_status()

    assert "Centimetres needs a scale" in app.units_status_var.get()
    assert not app._run_analysis()
    assert quiet["error"][0][0] == "Units"
    assert app.loaded_files["control"].processed_data is None


def test_run_analysis_applies_an_edit_the_user_forgot_to_apply(
        app, quiet, tmp_path, synthetic_npy, monkeypatch):
    monkeypatch.setattr(app, "_report_batch_outcome", lambda *a, **k: None)
    app._add_path(_make_session(tmp_path, synthetic_npy, "control"))
    app.units_choice.set("cm")
    app.cm_pixels_var.set("300")
    app.cm_length_var.set("10")

    assert app._run_analysis()

    loaded = app.loaded_files["control"]
    assert loaded.calibration.unit_name == "cm"
    assert loaded.processed_data, "analysed, in the units on screen"


def test_removing_a_session_updates_the_table(
        app, quiet, tmp_path, synthetic_npy, synthetic_npy_larger_fish):
    _make_session(tmp_path, synthetic_npy, "control")
    _make_session(tmp_path, synthetic_npy_larger_fish, "treated")
    app._add_path(tmp_path)
    app.sessions_tree.selection_set("treated")

    app._remove_file()

    assert list(_rows(app)) == ["control"]
    assert app.bl_pixels_var.get() == "40.0"


# --- measuring the scale on a frame -----------------------------------------------

class _Click:
    def __init__(self, ax, x, y):
        self.inaxes, self.xdata, self.ydata = ax, x, y


def test_measuring_two_points_fills_the_centimetre_scale(
        app, quiet, tmp_path, synthetic_npy, monkeypatch):
    app._add_path(_make_session(tmp_path, synthetic_npy, "control"))
    monkeypatch.setattr(data_tab, "load_frame_image",
                        lambda loaded_file: np.zeros((100, 200), dtype=np.uint8))
    opened = []
    real_dialog = measure_dialog.MeasureScaleDialog
    monkeypatch.setattr(data_tab, "MeasureScaleDialog",
                        lambda *a, **k: opened.append(real_dialog(*a, **k)))
    app.units_choice.set("cm")

    app._measure_scale()
    dialog = opened[0]
    dialog._on_click(_Click(dialog.ax, 10, 20))
    assert dialog.measured_pixels() is None
    dialog._on_click(_Click(dialog.ax, 130, 70))
    assert dialog.measured_pixels() == pytest.approx(130.0)
    dialog.length_var.set("5")
    dialog._confirm()

    assert app.cm_pixels_var.get() == "130.0"
    assert app.cm_length_var.get() == "5"
    assert "26.00 pixels per cm" in app.units_status_var.get()


def test_measure_dialog_wants_two_points_and_a_length(app, monkeypatch):
    asked = []
    monkeypatch.setattr(measure_dialog.messagebox, "showinfo",
                        lambda title, text, **k: asked.append(text))
    used = []
    dialog = measure_dialog.MeasureScaleDialog(
        app.root, np.zeros((50, 50), dtype=np.uint8), "s",
        lambda pixels, cm: used.append((pixels, cm)))
    try:
        dialog._confirm()
        dialog._on_click(_Click(dialog.ax, 0, 0))
        dialog._on_click(_Click(dialog.ax, 30, 40))
        dialog._confirm()
        assert used == []
        assert "two different points" in asked[0] and "in cm" in asked[1]

        # A third click starts a new measurement.
        dialog._on_click(_Click(dialog.ax, 5, 5))
        assert dialog.measured_pixels() is None
    finally:
        dialog.window.destroy()


# --- names and groups -------------------------------------------------------------

def test_renaming_a_session_carries_everything_with_it(
        app, quiet, tmp_path, synthetic_npy, synthetic_npy_larger_fish, monkeypatch):
    _make_session(tmp_path, synthetic_npy, "control1_long_camera_name")
    _make_session(tmp_path, synthetic_npy_larger_fish, "treated1")
    app._add_path(tmp_path)
    old = "control1_long_camera_name"
    loaded = app.loaded_files[old]
    from fish_analyzer import process_and_analyze_file
    analysed = process_and_analyze_file(loaded)
    loaded.processed_data = analysed
    app.file_arena_definitions[old] = "arena"
    monkeypatch.setattr(data_tab.simpledialog, "askstring", lambda *a, **k: " ctrl 1 ")
    app.sessions_tree.selection_set(old)

    app._rename_session()

    assert list(app.loaded_files) == ["ctrl 1", "treated1"], "order is kept"
    assert app.loaded_files["ctrl 1"] is loaded and loaded.nickname == "ctrl 1"
    assert loaded.processed_data is analysed, "nothing is recomputed or lost"
    assert {app.results_tree.item(i, "values")[1]
            for i in app.results_tree.get_children()} == {"ctrl 1"}, \
        "the Results tab follows the new name"
    assert app.file_arena_definitions == {"ctrl 1": "arena"}
    assert _rows(app)["ctrl 1"]["group"] == "ctrl", \
        "a group nobody chose is derived from the new name"
    assert list(app.inspector_file_dropdown["values"]) == ["ctrl 1", "treated1"]


def test_a_session_cannot_take_another_sessions_name(
        app, quiet, tmp_path, synthetic_npy, synthetic_npy_larger_fish, monkeypatch):
    _make_session(tmp_path, synthetic_npy, "control")
    _make_session(tmp_path, synthetic_npy_larger_fish, "treated")
    app._add_path(tmp_path)
    monkeypatch.setattr(data_tab.simpledialog, "askstring", lambda *a, **k: "treated")
    app.sessions_tree.selection_set("control")

    app._rename_session()

    assert list(app.loaded_files) == ["control", "treated"]
    assert quiet["error"][0][0] == "Rename"


def test_several_sessions_can_be_put_in_one_group(
        app, quiet, tmp_path, synthetic_npy, synthetic_npy_larger_fish, monkeypatch):
    _make_session(tmp_path, synthetic_npy, "tank3")
    _make_session(tmp_path, synthetic_npy_larger_fish, "tank7")
    app._add_path(tmp_path)
    assert _rows(app)["tank3"]["group"] == "tank", "the default strips a trailing number"
    monkeypatch.setattr(data_tab.simpledialog, "askstring", lambda *a, **k: "treated")
    app.sessions_tree.selection_set(["tank3", "tank7"])

    app._set_group()

    assert app.file_groups == {"tank3": "treated", "tank7": "treated"}
    assert {r["group"] for r in _rows(app).values()} == {"treated"}


def test_a_chosen_group_survives_a_rename(
        app, quiet, tmp_path, synthetic_npy, monkeypatch):
    app._add_path(_make_session(tmp_path, synthetic_npy, "tank3"))
    app.file_groups["tank3"] = "wild type"
    monkeypatch.setattr(data_tab.simpledialog, "askstring", lambda *a, **k: "wt_a")
    app.sessions_tree.selection_set("tank3")

    app._rename_session()

    assert _rows(app)["wt_a"]["group"] == "wild type"


# --- the tank outline -------------------------------------------------------------

def test_one_outline_lands_on_the_same_pixels_in_every_session(
        app, quiet, tmp_path, synthetic_npy, synthetic_npy_larger_fish, monkeypatch):
    """Sessions with different body lengths are on different scales, so the
    outline is placed per session from its pixels. Copying its scaled corners
    instead resized the tank relative to the fish."""
    from fish_analyzer.gui import outline_dialog
    _make_session(tmp_path, synthetic_npy, "control1")
    _make_session(tmp_path, synthetic_npy_larger_fish, "treated1")
    app._add_path(tmp_path)
    # One shared scale is the app's default; give each its own to test this.
    for loaded in app.loaded_files.values():
        loaded.calibration.pixels_per_unit = loaded.metadata.body_length
    monkeypatch.setattr(data_tab, "load_frame_image",
                        lambda loaded_file: np.zeros((1024, 1024), dtype=np.uint8))
    opened = []
    real_dialog = outline_dialog.TankOutlineDialog
    monkeypatch.setattr(data_tab, "TankOutlineDialog",
                        lambda *a, **k: opened.append(real_dialog(*a, **k)))

    app._draw_tank_outline()
    dialog = opened[0]
    corners = [(100, 200), (900, 200), (900, 800), (100, 800)]
    for x, y in corners:
        dialog._on_click(_Click(dialog.ax, x, y))
    dialog._confirm()

    arenas = app.file_arena_definitions
    assert set(arenas) == {"control1", "treated1"}, "no selection means every session"
    for arena in arenas.values():
        assert arena.vertices_pixels.tolist() == [list(map(float, c)) for c in corners]
    assert not np.allclose(arenas["control1"].vertices_bl, arenas["treated1"].vertices_bl)


def test_a_new_outline_updates_time_near_the_wall(
        app, quiet, tmp_path, synthetic_npy, monkeypatch):
    monkeypatch.setattr(app, "_report_batch_outcome", lambda *a, **k: None)
    app._add_path(_make_session(tmp_path, synthetic_npy, "control1"))
    app._run_analysis()
    loaded = app.loaded_files["control1"]
    assert loaded.thigmotaxis_results is None, "no outline, so no wall time yet"

    app._set_tank_outline(["control1"], [[0, 0], [1024, 0], [1024, 1024], [0, 1024]])

    assert loaded.thigmotaxis_results is not None
    near_wall = [app.results_tree.item(i, "values")[-1]
                 for i in app.results_tree.get_children()]
    assert all(value != "" for value in near_wall), "the Results table follows"


def test_outline_dialog_wants_three_corners(app, monkeypatch):
    from fish_analyzer.gui import outline_dialog
    asked = []
    monkeypatch.setattr(outline_dialog.messagebox, "showinfo",
                        lambda title, text, **k: asked.append(text))
    used = []
    dialog = outline_dialog.TankOutlineDialog(
        app.root, np.zeros((50, 50), dtype=np.uint8), "s", used.append,
        existing=[[1, 1], [40, 1], [40, 40]])
    try:
        dialog._on_click(_Click(dialog.ax, 0, 0))
        dialog._on_click(_Click(dialog.ax, 30, 0))
        dialog._confirm()
        assert used == [] and "three corners" in asked[0]

        dialog._start_again()
        assert dialog.corners() is None
        for x, y in ((0, 0), (30, 0), (30, 40)):
            dialog._on_click(_Click(dialog.ax, x, y))
        dialog._confirm()
        assert used[0].tolist() == [[0, 0], [30, 0], [30, 40]]
    finally:
        if dialog.window.winfo_exists():
            dialog.window.destroy()


def test_the_analysis_settings_reach_the_parameters_and_follow_the_unit(
        app, quiet, tmp_path, synthetic_npy):
    """Every choice that changes what a number means is a box on this tab,
    and the straightness speed converts with the unit like the freeze one."""
    app._add_path(_make_session(tmp_path, synthetic_npy, "control"))
    app.min_tracked_var.set("60")
    app.freeze_min_var.set("2")
    app.smoothing_var.set("0")
    params = app._get_processing_parameters_from_gui()
    assert params.min_valid_percentage == pytest.approx(0.6)
    assert params.min_freeze_seconds == 2.0
    assert params.smoothing_seconds == 0.0
    assert params.straightness_min_speed == 1.0

    app.units_choice.set("cm")
    app.cm_pixels_var.set("680")
    app.cm_length_var.set("20")
    app._apply_units()
    assert app.straight_unit_label["text"] == "cm/s."
    assert float(app.straight_speed_var.get()) == pytest.approx(1.176, abs=0.005)


def test_a_setting_that_is_not_a_number_is_refused_by_name(
        app, quiet, tmp_path, synthetic_npy):
    app._add_path(_make_session(tmp_path, synthetic_npy, "control"))
    app.freeze_min_var.set("one")
    with pytest.raises(ValueError, match="shortest freeze"):
        app._get_processing_parameters_from_gui()
