"""Regression tests for the fixes applied after AUDIT_D_GUI.md.

Each test pins one defect that produced a wrong or invisible result for the
user. They are cheap and need no real session data; the fixtures live in
conftest.py.

See AUDIT_D_GUI.md for the traced failure paths these correspond to.
"""
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# C5 — the Methods paragraph reported constructor defaults, not the run
# ---------------------------------------------------------------------------

def test_processing_params_reflect_the_gui(app):
    """self.processing_params must track the controls the user actually set.

    It was assigned once in GUIBase.__init__ and read only by the Methods text
    generator, so a manuscript paragraph claimed 'Smoothing: OFF' no matter
    what was ticked.
    """
    app.apply_smoothing_var.set(True)
    app.smoothing_window_var.set("7")
    app.rest_threshold_var.set("1.25")

    params = app._get_processing_parameters_from_gui()
    app.processing_params = params          # what _run_analysis now does

    assert app.processing_params.apply_smoothing is True
    assert app.processing_params.smoothing_window == 7
    assert app.processing_params.rest_speed_threshold == 1.25


# ---------------------------------------------------------------------------
# C1 — calibration changed after analysis, exporting old numbers under new units
# ---------------------------------------------------------------------------

def test_calibration_change_invalidates_cached_results(app, synthetic_npy):
    """Changing calibration must discard results computed under the old one.

    Metrics are scaled at processing time but the unit label is read at export
    time, so keeping them produced a CSV with body-length numbers under a 'cm'
    header — silently.
    """
    from fish_analyzer import TrajectoryFileLoader

    nick = "s1"
    app.loaded_files[nick] = TrajectoryFileLoader.load_file(synthetic_npy, nick)
    app.active_file = nick

    loaded = app.loaded_files[nick]
    loaded.processed_data = ["stale"]
    loaded.shoaling_results = "stale"
    loaded.thigmotaxis_results = "stale"
    app.bout_results[nick] = ["stale"]

    cleared = app._invalidate_results_for([nick])

    assert loaded.processed_data is None
    assert loaded.shoaling_results is None
    assert loaded.thigmotaxis_results is None
    assert nick not in app.bout_results
    assert cleared == [nick]
    # The user has to be told, or they just see their results vanish.
    assert "discarded" in app._invalidation_notice(cleared).lower()


def test_invalidation_notice_is_empty_when_nothing_was_cached(app):
    """No results, no scary message."""
    assert app._invalidation_notice([]) == ""


# ---------------------------------------------------------------------------
# C2/C3 — removing or replacing a file left its state behind
# ---------------------------------------------------------------------------

def test_purge_file_state_clears_every_side_dictionary(app):
    """Per-file state keyed by nickname must not outlive the file.

    Stale bout results were still written to the exported CSV — with a
    hardcoded 30 fps fallback, so every timestamp for that file was wrong on a
    recording at any other frame rate.
    """
    nick = "gone"

    class FakeReader:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    reader = FakeReader()
    app.file_arena_definitions[nick] = "arena"
    app.file_roi_definitions[nick] = "roi"
    app.file_groups[nick] = "group"
    app.bout_results[nick] = ["bout"]
    app.video_readers[nick] = reader
    app.current_arena_file = nick

    app._purge_file_state(nick)

    for mapping in (app.file_arena_definitions, app.file_roi_definitions,
                    app.file_groups, app.bout_results, app.video_readers):
        assert nick not in mapping

    assert reader.closed, "the cv2 capture must be released, not just dropped"
    assert app.current_arena_file is None


# ---------------------------------------------------------------------------
# C4 — an arena copied between files landed in the wrong place
# ---------------------------------------------------------------------------

def _arena_for(loaded, vertices_bl):
    """Build an ArenaDefinition the way _complete_arena does."""
    from fish_analyzer.spatial import ArenaDefinition
    verts_bl = np.asarray(vertices_bl, dtype=float)
    verts_px = verts_bl * loaded.metadata.body_length
    verts_px[:, 1] = loaded.metadata.video_height - verts_px[:, 1]
    return ArenaDefinition(vertices_pixels=verts_px, vertices_bl=verts_bl)


def test_arena_rescale_preserves_pixel_geometry(app, synthetic_npy,
                                                synthetic_npy_larger_fish):
    """Rescaling must keep the polygon on the same pixels of the frame.

    vertices_bl was divided by the *source* file's body length, while fish
    positions are divided by the *target's*, so a plain copy resized the arena
    relative to the fish.
    """
    from fish_analyzer import TrajectoryFileLoader

    src = TrajectoryFileLoader.load_file(synthetic_npy, "src")
    tgt = TrajectoryFileLoader.load_file(synthetic_npy_larger_fish, "tgt")
    assert src.metadata.body_length != tgt.metadata.body_length

    arena = _arena_for(src, [[1, 1], [9, 1], [9, 7], [1, 7]])
    rescaled = app._rescale_arena_for(arena, tgt)

    # Invert the target's transform: we must land back on the same pixels.
    back = rescaled.vertices_bl * tgt.metadata.body_length
    back[:, 1] = tgt.metadata.video_height - back[:, 1]
    assert np.allclose(back, arena.vertices_pixels, atol=1e-9)


def test_arena_rescale_actually_changes_bl_vertices(app, synthetic_npy,
                                                    synthetic_npy_larger_fish):
    """Guards the point of the fix: a plain copy is not equivalent.

    If this ever passes trivially, _rescale_arena_for has regressed to a copy.
    """
    from fish_analyzer import TrajectoryFileLoader

    src = TrajectoryFileLoader.load_file(synthetic_npy, "src")
    tgt = TrajectoryFileLoader.load_file(synthetic_npy_larger_fish, "tgt")

    arena = _arena_for(src, [[1, 1], [9, 1], [9, 7], [1, 7]])
    rescaled = app._rescale_arena_for(arena, tgt)

    assert not np.allclose(rescaled.vertices_bl, arena.vertices_bl), (
        "different body lengths must produce different body-length vertices"
    )


# ---------------------------------------------------------------------------
# D2 — root.update() let the user re-enter a running analysis
# ---------------------------------------------------------------------------

def test_progress_helper_disables_and_restores_the_button(app):
    """The triggering button must be dead while its own handler is running."""
    button = app.run_analysis_button
    seen, states = [], []

    for item in app._with_progress(["a", "b", "c"], "Testing", button=button,
                                   progressbar=app.analysis_progress):
        seen.append(item)
        states.append(str(button.cget("state")))

    assert seen == ["a", "b", "c"]
    assert states == ["disabled"] * 3
    assert str(button.cget("state")) == "normal"


def test_progress_helper_restores_the_button_on_exception(app):
    """A crash mid-batch must not leave the button permanently disabled."""
    import gc

    button = app.run_bout_button
    with pytest.raises(RuntimeError):
        for _ in app._with_progress(["a"], "Testing", button=button):
            raise RuntimeError("boom")
    gc.collect()   # generator close runs the finally block

    assert str(button.cget("state")) == "normal"


def test_gui_does_not_pump_the_full_event_queue():
    """root.update() must not come back.

    update() dispatches user input, so a second click re-entered the handler
    mid-run. update_idletasks() repaints without that hazard.
    """
    offenders = []
    for path in (REPO / "fish_analyzer" / "gui").rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            # A call statement ends the line; prose mentioning it (docstrings,
            # comments explaining why it was removed) does not.
            if line.strip().endswith(".root.update()"):
                offenders.append(f"{path.relative_to(REPO)}:{i}")
    assert not offenders, f"root.update() reintroduced at: {offenders}"


# ---------------------------------------------------------------------------
# A2/A3/A5 — exceptions and log history went nowhere the user could see
# ---------------------------------------------------------------------------

def test_error_handlers_are_installed(app):
    """Tk and matplotlib must both route uncaught callbacks to the reporter.

    Their defaults print to stderr and continue, so a failed button click or a
    failed arena click looked exactly like a control that did nothing.
    """
    from fish_analyzer.gui import utils as gui_utils

    assert app.root.report_callback_exception == app._report_uncaught
    assert gui_utils._FIGURE_ERROR_HANDLER == app._report_uncaught


def test_embedded_canvases_get_the_error_handler(app):
    """Every canvas built through the shared helper must be wired up."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from fish_analyzer.gui.utils import install_canvas_error_handler

    canvas = FigureCanvasTkAgg(Figure(), master=app.root)
    install_canvas_error_handler(canvas)
    assert canvas.callbacks.exception_handler == app._report_uncaught


def test_report_uncaught_accepts_both_call_signatures(app, monkeypatch):
    """Tk passes (type, value, tb); matplotlib passes one exception instance."""
    shown = []
    monkeypatch.setattr("fish_analyzer.gui.base.messagebox.showerror",
                        lambda title, msg: shown.append((title, msg)))

    app._report_uncaught(ValueError, ValueError("tk style"), None)
    app._report_uncaught(ValueError("matplotlib style"))

    assert len(shown) == 2
    assert "tk style" in shown[0][1]
    assert "matplotlib style" in shown[1][1]


@contextmanager
def _routed_through_gui(app):
    """Point sys.stdout/stderr at the app's redirectors for the duration.

    The app installs them at construction, but pytest reinstalls its own
    capture at the start of each test phase, which clobbers them. Re-pointing
    here keeps the test end-to-end through print()/sys.stderr rather than
    poking the redirector objects directly.
    """
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = app._log_redirector, app._err_redirector
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved


def test_animation_reports_render_errors_instead_of_swallowing(app, monkeypatch):
    """A5 — the playback loop used to `except Exception: pass`.

    The frame counter kept advancing while the image stayed frozen, which is
    indistinguishable from a still video. Playback must stop and report once.
    """
    reported = []
    monkeypatch.setattr(app, "_report_uncaught", lambda *a, **k: reported.append(a))
    monkeypatch.setattr(app, "_inspector_update_fast",
                        lambda: (_ for _ in ()).throw(RuntimeError("render boom")))

    app.animation_running = True
    app._inspector_animate_step()

    assert app.animation_running is False, "playback must stop, not spin on the error"
    assert len(reported) == 1, "the error must be reported exactly once"


def test_stderr_is_captured_and_the_log_is_reachable(app):
    """get_log() was defined and never called; stderr was not captured at all."""
    with _routed_through_gui(app):
        print("marker-stdout")
        sys.stderr.write("marker-stderr\n")

    log = app.get_log()
    assert "marker-stdout" in log
    assert "[error] marker-stderr" in log, "stderr must reach the log history"


def test_log_history_is_bounded(app):
    """The buffer must not grow without limit over a long session."""
    from fish_analyzer.gui.base import GUILogRedirector

    with _routed_through_gui(app):
        for i in range(GUILogRedirector.MAX_LINES + 250):
            print(f"line {i}")

    assert len(app._log_lines) <= GUILogRedirector.MAX_LINES
    assert "line 0" not in app.get_log(), "oldest lines should be dropped"


# ---------------------------------------------------------------------------
# D1 — the inspector rebuilt its whole widget tree every frame by default
# ---------------------------------------------------------------------------

def test_inspector_rebuild_flag_is_independent_of_the_time_panel(app):
    """_insp_fig is only assigned when a time panel is shown.

    Using it as the rebuild sentinel meant that with Time Panel = 'None' — the
    default — every slider tick destroyed and rebuilt the display.
    """
    assert hasattr(app, "_insp_needs_rebuild")

    app._insp_needs_rebuild = False
    app._insp_fig = None                      # the old sentinel, time panel off
    assert app._insp_needs_rebuild is False

    app._inspector_rebuild_needed()           # explicit request still works
    # _inspector_update_fast clears it only after actually rebuilding, which
    # needs a selected file; with none selected it returns early.
    assert app._insp_needs_rebuild is True


# ---------------------------------------------------------------------------
# Audit B Phase 0 — eight noise-dominated metrics were withdrawn
# ---------------------------------------------------------------------------

def test_analysis_tab_renders_after_the_metric_withdrawal(app, synthetic_npy):
    """Drive the summary tables, the comparison figure and the Methods panel.

    Removing columns from a treeview is the classic way to get a header list
    and a row tuple out of step, which raises only once real rows are inserted
    -- i.e. never during a construction-only smoke test.
    """
    from fish_analyzer.file_loading import TrajectoryFileLoader
    from fish_analyzer.processing import (ProcessingParameters,
                                          process_and_analyze_file)

    loaded = TrajectoryFileLoader.load_file(synthetic_npy)
    loaded.processed_data = process_and_analyze_file(
        loaded, ProcessingParameters.default_for_fish())
    app.loaded_files[loaded.nickname] = loaded
    names = [loaded.nickname]

    app._update_analysis_summary(names)
    app._plot_behavioral_comparison(names)
    app._update_analysis_methods_text(names)
    app.root.update_idletasks()

    methods = app.analysis_methods_text.get("1.0", "end")
    for withdrawn in ("Burst accel threshold", "Erratic turn threshold",
                      "Angular velocity was computed", "Burst events were detected"):
        assert withdrawn not in methods
    assert "deliberately not reported" in methods


def test_shoaling_tab_labels_axes_with_the_files_own_unit(app, synthetic_npy):
    """Audit B7: axis labels and table headers were hardcoded "BL" while the
    calculator ignored the file's calibration, so a cm-calibrated session was
    plotted against a BL axis."""
    from fish_analyzer.data_structures import CalibrationSettings
    from fish_analyzer.file_loading import TrajectoryFileLoader
    from fish_analyzer.shoaling import ShoalingCalculator, ShoalingParameters

    loaded = TrajectoryFileLoader.load_file(
        synthetic_npy,
        calibration=CalibrationSettings.from_physical_measurement(
            10.0, "cm", 30.0))
    results = {loaded.nickname:
               ShoalingCalculator(loaded, ShoalingParameters(30)).calculate()}
    app.loaded_files[loaded.nickname] = loaded

    app._display_shoaling_comparison(results)
    app._plot_nnd_comparison(results)
    app._plot_hull_comparison(results)
    app.root.update_idletasks()

    from fish_analyzer.gui.shoaling_tab import _unit
    assert _unit(results) == "cm"


def test_mixed_calibrations_are_labelled_as_such_rather_than_guessed():
    """Two files calibrated differently cannot share an axis; saying so beats
    picking one of them."""
    from fish_analyzer.gui.shoaling_tab import _unit

    class R:
        def __init__(self, u):
            self.unit_name = u

    assert _unit({"a": R("cm"), "b": R("cm")}) == "cm"
    assert _unit({"a": R("cm"), "b": R("BL")}) == "mixed units"


# ---------------------------------------------------------------------------
# The overlay settings snapshot handed to the shared compositor
# ---------------------------------------------------------------------------

def test_render_settings_reflect_the_inspector_controls(app):
    """The snapshot handed to the compositor must be what the user ticked.

    Same failure mode as C5: a settings object that ignores the GUI renders
    something other than what the controls say.
    """
    app.inspector_show_nnd_var.set(True)
    app.inspector_show_hull_var.set(False)
    app.inspector_dot_size_var.set(11)
    app.inspector_trail_var.set(45)

    settings = app.render_settings_from_vars()

    assert settings.show_nnd is True
    assert settings.show_hull is False
    assert settings.dot_radius == 11
    assert settings.trail_length == 45


# ---------------------------------------------------------------------------
# Export range markers
# ---------------------------------------------------------------------------

def test_in_out_markers_normalise_when_set_backwards(app):
    """Marking Out before In must not produce an empty or negative range."""
    app.inspector_mark_in = 900
    app.inspector_mark_out = 100

    start, end = app._inspector_export_range(n_frames=1000)

    assert (start, end) == (100, 900)


def test_unset_markers_default_to_the_whole_recording(app):
    app.inspector_mark_in = None
    app.inspector_mark_out = None

    start, end = app._inspector_export_range(n_frames=1000)

    assert (start, end) == (0, 999)


def test_one_marker_set_bounds_only_that_end(app):
    app.inspector_mark_in = 250
    app.inspector_mark_out = None

    assert app._inspector_export_range(n_frames=1000) == (250, 999)


def test_markers_are_clamped_to_the_recording(app):
    """A marker left over from a longer file must not run off the end."""
    app.inspector_mark_in = 0
    app.inspector_mark_out = 5000

    assert app._inspector_export_range(n_frames=1000) == (0, 999)


def test_markers_reset_when_a_different_file_is_selected(app, synthetic_npy):
    """A range marked on one recording must not survive onto another."""
    from fish_analyzer.file_loading import TrajectoryFileLoader

    app.loaded_files["s1"] = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.inspector_file_var.set("s1")
    app._on_inspector_file_selected()

    app.inspector_mark_in = 100
    app.inspector_mark_out = 200

    app._on_inspector_file_selected()

    assert app.inspector_mark_in is None
    assert app.inspector_mark_out is None


def test_saved_frame_is_full_video_resolution_not_canvas_size(app,
                                                              synthetic_npy):
    """The canvas downscales to fit the widget; a figure needs the real thing."""
    from fish_analyzer.file_loading import TrajectoryFileLoader

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.loaded_files["s1"] = loaded
    app.inspector_file_var.set("s1")
    app.inspector_show_positions_var.set(True)

    composed, returned, frame_idx = app._inspector_current_composite()

    assert composed.shape[0] == loaded.metadata.video_height
    assert composed.shape[1] == loaded.metadata.video_width
    assert returned is loaded


def test_current_composite_reports_no_file_rather_than_raising(app):
    composed, loaded, frame_idx = app._inspector_current_composite()

    assert composed is None and loaded is None and frame_idx is None


# ---------------------------------------------------------------------------
# Export guards
# ---------------------------------------------------------------------------

def test_export_refuses_when_the_time_panel_needs_missing_shoaling(
        app, synthetic_npy):
    """The on-screen panel says 'Run Shoaling Analysis first'; an export must
    refuse rather than bake that placeholder into a video."""
    from fish_analyzer.file_loading import TrajectoryFileLoader

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.loaded_files["s1"] = loaded
    app.inspector_file_var.set("s1")
    app.inspector_time_mode_var.set("nnd")

    assert not loaded.shoaling_results

    ok, reason = app._inspector_can_export()

    assert ok is False
    assert "Shoaling" in reason


def test_export_refuses_when_the_bout_panel_has_no_results(app, synthetic_npy):
    from fish_analyzer.file_loading import TrajectoryFileLoader

    app.loaded_files["s1"] = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.inspector_file_var.set("s1")
    app.inspector_time_mode_var.set("bout")

    ok, reason = app._inspector_can_export()

    assert ok is False
    assert "Bout" in reason


def test_export_is_allowed_when_the_time_panel_is_off(app, synthetic_npy):
    from fish_analyzer.file_loading import TrajectoryFileLoader

    app.loaded_files["s1"] = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.inspector_file_var.set("s1")
    app.inspector_time_mode_var.set("none")

    ok, _ = app._inspector_can_export()

    assert ok is True


def test_export_refuses_with_no_file_selected(app):
    ok, reason = app._inspector_can_export()

    assert ok is False
    assert "Select a file" in reason


# ---------------------------------------------------------------------------
# Time-panel blitting across a resize
# ---------------------------------------------------------------------------

def test_resizing_the_time_panel_drops_the_stale_blit_background(app):
    """A background cached at one canvas size must not be restored at another.

    The figure is created 10 inches wide but its widget is packed fill="x", so
    Tk stretches it and matplotlib redraws at the new width. Restoring the
    narrow cached raster over the wide canvas left the earlier rendering
    visible beside it — the plot appeared duplicated at the bottom of the tab.
    """
    app._insp_bg_cache = object()          # stand-in for a cached raster

    app._on_inspector_time_canvas_resize()

    assert app._insp_bg_cache is None


def test_window_resize_also_drops_the_time_panel_background(app):
    app._insp_bg_cache = object()

    app._on_inspector_resize()

    assert app._insp_bg_cache is None


def test_recapture_is_a_noop_when_there_is_no_time_panel(app):
    """Time Panel 'none' means no figure at all; recapture must not raise."""
    app._insp_canvas = None
    app._insp_fig = None

    app._inspector_recapture_time_background()

    assert app._insp_bg_cache is None


def test_export_converts_seconds_to_the_panel_time_units(app):
    """The NND/IID/Hull panels are drawn against minutes, the export counts
    seconds. Without the conversion the exported cursor pins to the right edge
    and never advances, which is silent - nothing raises."""
    assert app._inspector_time_scale_for("nnd") == pytest.approx(1.0 / 60.0)
    assert app._inspector_time_scale_for("iid") == pytest.approx(1.0 / 60.0)
    assert app._inspector_time_scale_for("hull") == pytest.approx(1.0 / 60.0)


def test_bout_panel_is_already_in_seconds(app):
    assert app._inspector_time_scale_for("bout") == pytest.approx(1.0)


def test_export_strip_covers_only_the_exported_segment(app, synthetic_npy):
    """The panel is drawn over the whole recording, so a short clip moved the
    cursor ~1% of the panel width. The axis is narrowed to the clip instead."""
    import numpy as np
    from fish_analyzer.file_loading import TrajectoryFileLoader

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.loaded_files["s1"] = loaded
    app.inspector_file_var.set("s1")
    fps = loaded.calibration.frame_rate

    # Stand in for a rebuilt time panel spanning the whole recording.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(10, 2), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.08, 0.22, 0.88, 0.68])
    ax.plot(np.linspace(0, loaded.n_frames / fps / 60.0, 50), np.zeros(50))
    ax.set_xlim(0, loaded.n_frames / fps / 60.0)
    app._insp_fig, app._insp_ax_time = fig, ax

    strip = app._inspector_build_time_strip(loaded, "nnd", 640,
                                            start=120, end=239)

    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(120 / fps / 60.0)
    assert hi == pytest.approx(239 / fps / 60.0)
    assert strip is not None


def test_export_strip_cursor_sweeps_the_whole_clip(app, synthetic_npy):
    """With the axis narrowed, the cursor should traverse most of the width."""
    import numpy as np
    from fish_analyzer.file_loading import TrajectoryFileLoader

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.loaded_files["s1"] = loaded
    app.inspector_file_var.set("s1")
    fps = loaded.calibration.frame_rate

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(10, 2), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.08, 0.22, 0.88, 0.68])
    ax.plot(np.linspace(0, loaded.n_frames / fps / 60.0, 50), np.zeros(50))
    app._insp_fig, app._insp_ax_time = fig, ax

    start, end = 120, 239
    strip = app._inspector_build_time_strip(loaded, "nnd", 640, start, end)

    pristine = strip.at(start / fps).copy()

    def col(img):
        d = np.abs(img.astype(int) - pristine.astype(int)).sum(axis=(0, 2))
        return int(np.argmax(d))

    first = col(strip.at((start + 1) / fps).copy())
    last = col(strip.at(end / fps).copy())

    # Across the clip the cursor should cross a large fraction of the panel,
    # not the ~1% it managed when the axis spanned the whole recording.
    assert (last - first) > strip.width_px * 0.5


# ---------------------------------------------------------------------------
# The IID panel must describe the fish the overlay draws
# ---------------------------------------------------------------------------

def test_iid_focus_change_forces_a_panel_rebuild(app, synthetic_npy):
    """The IID panel plots one fish, so switching focus has to redraw it.

    Only bout_fish was in the rebuild condition, so the trace kept describing
    whichever fish was selected when the panel was first built.
    """
    from fish_analyzer.file_loading import TrajectoryFileLoader

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.loaded_files["s1"] = loaded
    app.inspector_file_var.set("s1")
    app.inspector_time_mode_var.set("iid")

    app._insp_needs_rebuild = False
    app._insp_cached_file = "s1"
    app._insp_cached_time_mode = "iid"
    app._insp_cached_overlays = app.inspector_video_var.get()
    app._insp_cached_iid_focus = 0

    app.inspector_iid_focus_var.set("2")

    assert app._inspector_iid_focus_index(loaded.n_fish) == 2
    assert app._insp_cached_iid_focus != \
        app._inspector_iid_focus_index(loaded.n_fish), \
        "cached focus should differ, which is what triggers the rebuild"


def test_iid_focus_index_clamps_to_an_existing_fish(app, synthetic_npy):
    """A focus left over from a six-fish session must not index a three-fish one."""
    from fish_analyzer.file_loading import TrajectoryFileLoader

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.inspector_iid_focus_var.set("5")

    assert app._inspector_iid_focus_index(loaded.n_fish) == 0

    app.inspector_iid_focus_var.set("not a number")
    assert app._inspector_iid_focus_index(loaded.n_fish) == 0


# ---------------------------------------------------------------------------
# Transport bar layout
# ---------------------------------------------------------------------------

def test_transport_survives_a_figure_rebuild(app, synthetic_npy):
    """The scrubber sits between the video and the time panel, both of which
    _inspector_rebuild_figure destroys and recreates. It needs its own
    container or it would vanish whenever the time mode changed."""
    from fish_analyzer.file_loading import TrajectoryFileLoader

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.loaded_files["s1"] = loaded
    app.inspector_file_var.set("s1")

    slider_before = app.inspector_frame_slider
    play_before = app.inspector_play_button

    app._inspector_rebuild_figure("s1", loaded, 0, "none")

    assert app.inspector_frame_slider is slider_before
    assert app.inspector_frame_slider.winfo_exists()
    assert app.inspector_play_button is play_before
    assert app.inspector_play_button.winfo_exists()
    assert app.inspector_mark_label.winfo_exists()


def test_scrubber_is_not_confined_to_the_control_panel(app):
    """It was a 180px slider in the left column; it now stretches under the
    video, so it must not be packed with a fixed short length."""
    # length= is only a default request; what makes it stretch is the packing.
    info = app.inspector_frame_slider.pack_info()
    assert info["fill"] in ("x", "both")
    assert str(info["expand"]) in ("1", "True", "true")

    # Its ancestry must not run through the scrollable controls canvas.
    names = []
    w = app.inspector_frame_slider
    while w is not None:
        names.append(str(w))
        w = getattr(w, "master", None)
    assert not any("labelframe" in n.lower() for n in names), \
        f"scrubber still inside a control LabelFrame: {names}"


# ---------------------------------------------------------------------------
# The typed path box
# ---------------------------------------------------------------------------

def test_typing_a_session_path_loads_it(app, synthetic_npy, monkeypatch):
    """The Session Folder box is editable, so a typed path must do something.

    It previously did nothing: no Load button, no <Return> binding, and
    _load_selected_file - the only route to loading a bare trajectories.npy -
    had no callers at all.
    """
    from fish_analyzer.gui import data_tab

    monkeypatch.setattr(data_tab.simpledialog, "askstring",
                        lambda *a, **k: "typed")

    app.file_path_var.set(str(synthetic_npy))
    app._load_selected_file()

    assert "typed" in app.loaded_files
    assert app.loaded_files["typed"].n_fish == 3


def test_typing_a_nonsense_path_reports_rather_than_loading(app, monkeypatch):
    from fish_analyzer.gui import data_tab

    shown = []
    monkeypatch.setattr(data_tab.messagebox, "showerror",
                        lambda title, msg: shown.append(title))

    app.file_path_var.set(r"C:\definitely\not\a\session")
    app._load_selected_file()

    assert shown == ["Invalid Path"]
    assert not app.loaded_files


def test_empty_path_box_is_reported_not_ignored(app, monkeypatch):
    from fish_analyzer.gui import data_tab

    shown = []
    monkeypatch.setattr(data_tab.messagebox, "showerror",
                        lambda title, msg: shown.append(title))

    app.file_path_var.set("")
    app._load_selected_file()

    assert shown == ["Error"]


def test_csv_save_dialog_returns_a_path_or_none(monkeypatch):
    """Every CSV export shares one Save-as dialog.

    It returns a Path rather than the raw string tkinter gives back, because
    every caller immediately wanted one, and None on cancel so callers can bail
    with a falsy check.
    """
    from fish_analyzer.gui import utils

    monkeypatch.setattr(utils.filedialog, "asksaveasfilename",
                        lambda **kw: r"C:\tmp\out.csv")
    got = utils.ask_csv_save_path("Title", "out.csv")
    assert isinstance(got, Path)
    assert got.name == "out.csv"

    monkeypatch.setattr(utils.filedialog, "asksaveasfilename",
                        lambda **kw: "")
    assert utils.ask_csv_save_path("Title", "out.csv") is None


def test_csv_save_dialog_offers_csv_first(monkeypatch):
    """A .csv default extension is what stops silently-extensionless exports."""
    from fish_analyzer.gui import utils

    seen = {}
    monkeypatch.setattr(utils.filedialog, "asksaveasfilename",
                        lambda **kw: seen.update(kw) or "x.csv")
    utils.ask_csv_save_path("Export Thing", "thing.csv")

    assert seen["defaultextension"] == ".csv"
    assert seen["filetypes"][0] == ("CSV files", "*.csv")
    assert seen["title"] == "Export Thing"
    assert seen["initialfile"] == "thing.csv"
