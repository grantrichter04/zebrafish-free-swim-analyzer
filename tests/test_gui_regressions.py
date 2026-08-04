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
