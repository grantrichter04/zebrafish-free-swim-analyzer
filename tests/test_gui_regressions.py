"""Regression tests for the fixes applied after docs/audit/AUDIT_D_GUI.md.

Each test pins one defect that produced a wrong or invisible result for the
user. They are cheap and need no real session data; the fixtures live in
conftest.py.

See docs/audit/AUDIT_D_GUI.md for the traced failure paths these correspond to.
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
    generator, so a manuscript paragraph described the defaults no matter what
    had been typed.
    """
    app.rest_threshold_var.set("1.25")

    params = app._get_processing_parameters_from_gui()
    app.processing_params = params          # what _run_analysis now does

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

    loaded = app.loaded_files[nick]
    loaded.processed_data = ["stale"]
    loaded.shoaling_results = "stale"
    loaded.thigmotaxis_results = "stale"

    cleared = app._invalidate_results_for([nick])

    assert loaded.processed_data is None
    assert loaded.shoaling_results is None
    assert loaded.thigmotaxis_results is None
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

    A stale video reader was still displayed, and its capture never released.
    """
    nick = "gone"

    class FakeReader:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    reader = FakeReader()
    app.file_arena_definitions[nick] = "arena"
    app.file_groups[nick] = "group"
    app.video_readers[nick] = reader

    app._purge_file_state(nick)

    for mapping in (app.file_arena_definitions,
                    app.file_groups, app.video_readers):
        assert nick not in mapping

    assert reader.closed, "the cv2 capture must be released, not just dropped"


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

    button = app.run_analysis_button
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
    loaded.shoaling_results = ShoalingCalculator(
        loaded, ShoalingParameters(30)).calculate()
    app.loaded_files[loaded.nickname] = loaded

    app._update_shoaling()
    app.root.update_idletasks()

    assert app.shoaling_tree.heading("NND")["text"] == "Nearest neighbour (cm)"
    from fish_analyzer.gui.shoaling_tab import _unit
    assert _unit({loaded.nickname: loaded.shoaling_results}) == "cm"


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
    app.inspector_show_positions_var.set(False)
    app.inspector_show_hull_var.set(True)
    app.inspector_dot_size_var.set(7)
    app.inspector_trail_seconds_var.set("1.5")
    try:
        settings = app.render_settings_from_vars()

        assert settings.show_nnd is True
        assert settings.show_positions is False
        assert settings.show_hull is True and settings.show_iid is False
        assert settings.dot_radius == 7
        assert settings.trail_length == 45, "1.5 seconds at 30 frames a second"

        zoomed = app.render_settings_from_vars(zoom=4.0)
        assert zoomed.line_scale == 0.25 and zoomed.dot_radius == 2, \
            "zoomed in, overlays are drawn finer so they stay the same size on screen"
        assert settings.line_scale == 1.0, "an export is never thinned"

        app.inspector_trail_seconds_var.set("a while")
        assert app.render_settings_from_vars().trail_length == 0, \
            "a trail length that is not a number is no trail, not a crash"
    finally:
        app.inspector_show_nnd_var.set(False)
        app.inspector_show_positions_var.set(True)
        app.inspector_show_hull_var.set(False)
        app.inspector_dot_size_var.set(12)
        app.inspector_trail_seconds_var.set("0")


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
    """The on-screen panel says 'Press Run All Analysis'; an export must
    refuse rather than bake that placeholder into a video."""
    from fish_analyzer.file_loading import TrajectoryFileLoader

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.loaded_files["s1"] = loaded
    app.inspector_file_var.set("s1")
    app.inspector_time_mode_var.set("nnd")

    assert not loaded.shoaling_results

    ok, reason = app._inspector_can_export()

    assert ok is False
    assert "Run All Analysis" in reason


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
    assert "Choose a session" in reason


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
    """The time panel is drawn against minutes, the export counts seconds.
    Without the conversion the exported cursor pins to the right edge and
    never advances, which is silent - nothing raises."""
    assert app._inspector_time_scale_for("nnd") == pytest.approx(1.0 / 60.0)


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
# The time cursor must not be baked into the chart it moves over
# ---------------------------------------------------------------------------

def test_the_time_cursor_is_left_out_of_the_cached_chart(
        app, tmp_path, synthetic_npy, monkeypatch):
    """The chart is drawn once and the cursor blitted over it. A chart
    re-cached while the cursor sat mid-recording kept a second, stationary
    cursor - on screen and, through the export strip, in clips."""
    from fish_analyzer.gui import data_tab
    monkeypatch.setattr(data_tab.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(app, "_report_batch_outcome", lambda *a, **k: None)
    target = tmp_path / "session_s1" / "trajectories"
    target.mkdir(parents=True)
    import shutil
    shutil.copy(synthetic_npy, target / "trajectories.npy")
    app._add_path(tmp_path / "session_s1")
    app._run_analysis()
    app.inspector_file_var.set("s1")
    app._on_inspector_file_selected()
    app.inspector_time_mode_var.set("nnd")
    app._inspector_rebuild_needed()

    def chart():
        app._insp_canvas.draw()
        return np.asarray(app._insp_canvas.buffer_rgba()).copy()

    try:
        assert app._insp_time_marker is not None
        at_start = chart()
        app.inspector_frame_var.set(app.loaded_files["s1"].n_frames // 2)
        app._inspector_update_fast()
        assert np.array_equal(chart(), at_start), \
            "a full draw of the chart must not include the cursor"
    finally:
        app.inspector_time_mode_var.set("none")
        app._inspector_rebuild_needed()


# ---------------------------------------------------------------------------
# Zoom, the plot under the video, and the fish the lines start from
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zoom, centre, expected", [
    (1.0, None, (0, 0, 1000, 800)),
    (2.0, None, (250, 200, 500, 400)),
    (4.0, (0, 0), (0, 0, 250, 200)),            # held inside the frame
    (4.0, (990, 790), (750, 600, 250, 200)),
    (0.5, None, (0, 0, 1000, 800)),             # never smaller than the frame
])
def test_zoom_view_stays_inside_the_frame(zoom, centre, expected):
    from fish_analyzer.gui.inspector_tab import zoom_view
    assert zoom_view(1000, 800, zoom, centre) == expected


class _Wheel:
    def __init__(self, x, y, delta):
        self.x, self.y, self.delta = x, y, delta


def test_scrolling_zooms_about_the_pointer_and_double_click_resets(app, synthetic_npy):
    from fish_analyzer.file_loading import TrajectoryFileLoader
    from fish_analyzer.gui.inspector_tab import zoom_view

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.loaded_files["s1"] = loaded
    app.inspector_file_var.set("s1")
    app._on_inspector_file_selected()
    width, height = loaded.metadata.video_width, loaded.metadata.video_height
    try:
        assert app._insp_zoom == 1.0
        x0, y0, scale_fit, left, top = app._insp_view
        # A point a quarter of the way across and down the frame.
        pointer = (left + width * 0.25 * scale_fit, top + height * 0.25 * scale_fit)
        before = app._inspector_frame_point(*pointer)

        app._on_inspector_wheel(_Wheel(*pointer, 120))

        assert app._insp_zoom == pytest.approx(1.25)
        after = app._inspector_frame_point(*pointer)
        assert after == pytest.approx(before, abs=2.0), \
            "the spot under the pointer stays under the pointer"
        x0, y0, view_w, view_h = zoom_view(width, height, app._insp_zoom,
                                           app._insp_zoom_centre)
        assert 0 <= x0 and x0 + view_w <= width and 0 <= y0 and y0 + view_h <= height

        for _ in range(40):
            app._on_inspector_wheel(_Wheel(*pointer, 120))
        assert app._insp_zoom == app.MAX_ZOOM
        for _ in range(40):
            app._on_inspector_wheel(_Wheel(*pointer, -120))
        assert app._insp_zoom == 1.0

        app._on_inspector_wheel(_Wheel(*pointer, 120))
        app._on_inspector_zoom_reset()
        assert app._insp_zoom == 1.0 and app._insp_zoom_centre is None
    finally:
        app._on_inspector_zoom_reset()


def test_the_list_under_the_video_chooses_what_is_plotted(app):
    try:
        for text, mode in app.TIME_PANELS:
            app.inspector_time_panel_box.set(text)
            app._inspector_on_time_panel_chosen()
            assert app.inspector_time_mode_var.get() == mode
    finally:
        app.inspector_time_panel_box.current(0)
        app._inspector_on_time_panel_chosen()


def test_lines_to_all_others_start_from_the_fish_named_in_the_list(app, synthetic_npy):
    """The list shows idtracker.ai's labels, which start at 1; the renderer
    wants a row of the trajectories, which starts at 0."""
    from fish_analyzer.file_loading import TrajectoryFileLoader

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.loaded_files["s1"] = loaded
    app.inspector_file_var.set("s1")
    app._on_inspector_file_selected()

    labels = [str(label) for label in loaded.metadata.identity_labels]
    assert list(app.inspector_iid_focus_combo["values"]) == labels
    app.inspector_iid_focus_var.set(labels[2])
    assert app.render_settings_from_vars().iid_focus == 2
    app.inspector_iid_focus_var.set("no such fish")
    assert app.render_settings_from_vars().iid_focus == 0


def test_outlines_are_refused_politely_when_the_session_did_not_save_them(
        app, synthetic_npy, monkeypatch):
    from fish_analyzer.file_loading import TrajectoryFileLoader
    from fish_analyzer.gui import inspector_tab

    told = []
    monkeypatch.setattr(inspector_tab.messagebox, "showinfo",
                        lambda title, text="", **k: told.append(title))
    app.loaded_files["s1"] = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.inspector_file_var.set("s1")
    app._insp_outline_finders.clear()
    app.inspector_show_outlines_var.set(True)

    app._inspector_on_outlines_toggle()

    assert told == ["No outlines for this session"]
    assert app.inspector_show_outlines_var.get() is False
    assert app._inspector_outlines_for("s1", np.zeros((10, 10, 3), np.uint8)) is None


# ---------------------------------------------------------------------------
# One slider position per frame
# ---------------------------------------------------------------------------

def test_the_scrubber_and_the_step_buttons_move_in_single_frames(app, synthetic_npy):
    from fish_analyzer.file_loading import TrajectoryFileLoader

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.loaded_files["s1"] = loaded
    app.inspector_file_var.set("s1")
    app._on_inspector_file_selected()

    assert int(app.inspector_frame_slider.cget("to")) == loaded.n_frames - 1
    app._inspector_step_forward()
    app._inspector_step_forward()
    assert app._get_inspector_frame_idx() == 2
    app._inspector_step_back()
    assert app._get_inspector_frame_idx() == 1
    for _ in range(3):
        app._inspector_step_back()
    assert app._get_inspector_frame_idx() == 0, "it stops at the first frame"
    app.inspector_frame_var.set(loaded.n_frames - 1)
    app._inspector_step_forward()
    assert app._get_inspector_frame_idx() == loaded.n_frames - 1, "and at the last"


@pytest.mark.parametrize("speed, stride, tick_ms", [
    ("0.25x", 1, 133), ("1x", 1, 33), ("4x", 4, 33), ("8x", 8, 33)])
def test_fast_playback_skips_frames_instead_of_outrunning_the_drawing(
        app, speed, stride, tick_ms):
    """Drawing a frame takes about as long as a frame lasts, so 8x cannot
    mean eight times as many frames a second."""
    app.inspector_speed_var.set(speed)
    try:
        assert app._inspector_playback_pace(30.0) == (stride, tick_ms)
    finally:
        app.inspector_speed_var.set("1x")


def test_a_session_without_its_video_offers_to_find_it(app, synthetic_npy):
    from fish_analyzer.file_loading import TrajectoryFileLoader

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    loaded.video_file_path = None
    app.loaded_files["s1"] = loaded
    app.inspector_file_var.set("s1")
    app._on_inspector_file_selected()

    assert app.inspector_video_var.get() is False
    assert "not found" in app.inspector_video_status.cget("text")
    assert app.inspector_find_video_button.winfo_manager() == "pack"

    app._inspector_show_video_status("tank.avi", True)
    assert app.inspector_find_video_button.winfo_manager() == "", \
        "nothing to find once a video is showing"


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


def test_check_setup_window_shows_the_self_check_report(app, monkeypatch):
    """The RA's way to answer 'is this laptop set up?' without a terminal."""
    import tkinter as tk
    from fish_analyzer import selfcheck

    monkeypatch.setattr(selfcheck, "run_checks", lambda: [
        selfcheck.CheckItem("Analyzer", True, "9.9.9", required=True),
        selfcheck.CheckItem("idtracker.ai", False, "not installed",
                            required=False),
    ])

    win = app._show_setup_check()
    try:
        text = next(w for w in win.winfo_children() if isinstance(w, tk.Text))
        shown = text.get("1.0", "end")
        assert "[ OK ] Analyzer" in shown and "9.9.9" in shown
        assert "[WARN] idtracker.ai" in shown
    finally:
        win.destroy()
