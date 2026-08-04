"""Shared fixtures for the fish_analyzer test suite.

Everything here is synthetic — the tests need no real session data and write
nothing outside pytest's tmp dirs.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture(scope="session")
def synthetic_npy(tmp_path_factory):
    """An idtracker.ai-shaped trajectories.npy with known properties.

    Same layout fish_analyzer expects: a 0-d object array holding a dict with
    a (n_frames, n_fish, 2) 'trajectories' array plus metadata fields.
    """
    return _write_session(tmp_path_factory.mktemp("session"), body_length=40.0)


@pytest.fixture(scope="session")
def synthetic_npy_larger_fish(tmp_path_factory):
    """A second session, same frame size but a different body length.

    Real recordings differ here — across four wild-type sessions the measured
    body length ranged 71.3 to 82.2 px — which is exactly the condition that
    made copying an arena between files misplace it.
    """
    return _write_session(tmp_path_factory.mktemp("session_big"), body_length=52.0)


def _write_session(directory: Path, body_length: float, n_frames: int = 600,
                   n_fish: int = 3, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(0, 2.0, size=(n_frames, n_fish, 2)), axis=0) + 400.0
    pos[rng.random((n_frames, n_fish)) < 0.02] = np.nan  # tracking dropouts
    d = {
        'trajectories': pos,
        'width': 1024,
        'height': 1024,
        'frames_per_second': 30.0,
        'body_length': body_length,
        'estimated_accuracy': 0.98,
        'fraction_identified': 0.97,
        'identities_labels': [str(i + 1) for i in range(n_fish)],
        'version': '6.0.14',
    }
    p = directory / "trajectories.npy"
    np.save(p, np.array(d, dtype=object), allow_pickle=True)
    return p


@pytest.fixture(scope="session")
def display_available():
    """Probe for a usable display exactly once per session.

    Probing per test meant creating and destroying a Tk root for every GUI
    test on top of the app's own root. That turned out to be flaky on Windows
    — different tests reported 'no display available' on different runs — and
    a test that silently skips is protecting nothing.
    """
    tk = pytest.importorskip("tkinter")
    try:
        probe = tk.Tk()
    except Exception:
        return False
    probe.destroy()
    return True


@pytest.fixture(scope="session")
def _gui_app(display_available):
    """One hidden EnhancedFishAnalyzer for the whole session.

    Constructing one per test creates a Tk root per test, and Tcl starts
    refusing new interpreters after a handful of them — which surfaced as
    tk.Tk() raising intermittently, in a different test on every run. One
    instance, reset between tests, avoids that entirely.
    """
    if not display_available:
        pytest.skip("no display available")

    from fish_analyzer import EnhancedFishAnalyzer

    # The app replaces sys.stdout/stderr at construction and only restores
    # them in run()'s finally block, which tests never reach.
    saved_out, saved_err = sys.stdout, sys.stderr
    instance = EnhancedFishAnalyzer()
    instance.root.withdraw()
    try:
        yield instance
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err
        try:
            instance.root.destroy()
        except Exception:
            pass


@pytest.fixture
def app(_gui_app):
    """The shared app, with per-test state cleared so tests stay independent."""
    from fish_analyzer.processing import ProcessingParameters

    for mapping in (_gui_app.loaded_files, _gui_app.bout_results,
                    _gui_app.file_arena_definitions,
                    _gui_app.file_roi_definitions, _gui_app.file_groups,
                    _gui_app.video_readers):
        mapping.clear()

    _gui_app.active_file = None
    _gui_app.current_arena_file = None
    _gui_app.arena_definition = None
    _gui_app.arena_vertices = []
    _gui_app.processing_params = ProcessingParameters.default_for_fish()
    _gui_app._log_lines.clear()
    _gui_app._insp_needs_rebuild = True
    _gui_app._insp_fig = None

    # pytest reinstalls its own capture each phase, so re-point the app's
    # redirectors at whatever it just installed.
    _gui_app._log_redirector.original_stream = sys.stdout
    _gui_app._err_redirector.original_stream = sys.stderr

    yield _gui_app
