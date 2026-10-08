"""Smoke tests for fish_analyzer.

Run from the repository root:

    pip install pytest
    pytest -q

These are deliberately cheap (~4 s) and use a synthetic idtracker.ai-shaped
trajectory file, so they need no real data and commit nothing.
"""

from pathlib import Path

import numpy as np
import pytest

# These tests read repository files - the startup banner, the README
# example - rather than importing the package, so they need the repo root as a
# path. Not a sys.path insert: the package is imported normally.
REPO = Path(__file__).resolve().parent.parent


#: `synthetic_npy` and the `app` fixture now live in conftest.py, so the GUI
#: regression tests can share them.


def test_package_imports():
    import fish_analyzer
    assert fish_analyzer.__version__


def test_version_matches_banner():
    """The startup banner must not drift from __version__."""
    src = (REPO / "fish_analyzer" / "__main__.py").read_text(encoding="utf-8")
    assert "__version__" in src, "banner should interpolate __version__, not hardcode it"


def test_readme_api_example_signature():
    """ShoalingCalculator.calculate is an instance method taking only self.

    The README documented it as a static method for a while; this pins the
    real shape so the docs and the code cannot drift apart again.
    """
    import inspect
    from fish_analyzer import ShoalingCalculator
    assert list(inspect.signature(ShoalingCalculator.calculate).parameters) == ["self"]


def test_console_output_is_ascii_safe():
    """print() in the package must survive a cp1252 stdout.

    Non-ASCII in print() raises UnicodeEncodeError whenever stdout is a pipe
    or a redirect on Windows, which killed the whole processing run.
    """
    offenders = []
    for path in (REPO / "fish_analyzer").rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "print(" in line and not line.lstrip().startswith("#"):
                if any(ord(c) > 127 for c in line):
                    offenders.append(f"{path.relative_to(REPO)}:{i}")
    assert not offenders, f"non-ASCII inside print(): {offenders}"


def test_load_and_process(synthetic_npy):
    from fish_analyzer import TrajectoryFileLoader, process_and_analyze_file
    loaded = TrajectoryFileLoader.load_file(synthetic_npy)
    assert loaded.n_fish == 3
    assert loaded.n_frames == 600

    fish_list = process_and_analyze_file(loaded)
    assert len(fish_list) == 3
    for fish in fish_list:
        assert fish.metrics["total_distance"] > 0
        assert np.isfinite(fish.metrics["mean_speed"])


def test_shoaling_runs(synthetic_npy):
    from fish_analyzer import (TrajectoryFileLoader, ShoalingCalculator,
                               ShoalingParameters)
    loaded = TrajectoryFileLoader.load_file(synthetic_npy)
    results = ShoalingCalculator(loaded, ShoalingParameters()).calculate()
    assert results.mean_nnd > 0
    # Nearest-neighbour distance is by definition <= mean pairwise distance.
    assert results.mean_nnd <= results.mean_iid


def test_gui_constructs(app):
    """Every tab's widget construction ran, across the whole mixin stack.

    Uses the shared session app rather than building a second one. It used to
    construct its own EnhancedFishAnalyzer on top of the fixture's, and
    conftest.py documents why that is a trap: "Tcl starts refusing new
    interpreters after a handful of them - which surfaced as tk.Tk() raising
    intermittently, in a different test on every run." This test was the one
    left doing it, and it skipped or failed roughly one run in five.

    Reaching the fixture at all means construction succeeded, so the assertions
    are about the result being complete rather than about it not raising.
    """
    tabs = [app.notebook.tab(i, "text")
            for i in range(app.notebook.index("end"))]

    assert tabs == [
        "Tracking",
        "Sessions & Units",
        "Results",
        "Individual Analysis",
        "Bout Analysis",
        "Shoaling Analysis",
        "Spatial Analysis",
        "Video Inspector",
    ]

    # One control from each tab's mixin, so a tab that silently built nothing
    # would be caught rather than merely counted.
    for attr in ("tracking_videos_tree",     # TrackingTabMixin
                 "sessions_tree",            # DataTabMixin
                 "speed_dist_collapse_var",  # AnalysisTabMixin
                 "inspector_frame_slider",   # InspectorTabMixin
                 "inspector_mark_label"):    # InspectorExportMixin
        assert hasattr(app, attr), f"{attr} missing - a tab did not build"


def test_declared_dependencies_fit_alongside_idtrackerai():
    """idtracker.ai installs opencv-python-headless. Declaring opencv-python
    here would put two packages in the same cv2 directory in a shared env."""
    from importlib.metadata import metadata, requires

    reqs = [r.replace(" ", "") for r in requires("fish-analyzer") or []]
    runtime = [r for r in reqs if "extra==" not in r]

    assert any(r.startswith("opencv-python-headless") for r in runtime)
    assert not any(r.startswith("opencv-python>") or r == "opencv-python"
                   for r in runtime)
    assert any(r.startswith("idtrackerai") and 'extra=="tracking"' in r
               for r in reqs), "the tracking extra should pull in idtrackerai"
    assert metadata("fish-analyzer")["Requires-Python"] == ">=3.10"


def test_requirements_txt_is_gone():
    """pyproject.toml is the only dependency list."""
    assert not (REPO / "requirements.txt").exists()


def test_launcher_shows_its_splash_before_loading_the_package():
    """launch.pyw exists so something is on screen during the slow imports.
    A module-level fish_analyzer import would run them first and defeat it."""
    import ast

    tree = ast.parse((REPO / "launch.pyw").read_text(encoding="utf-8"))
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top_level if isinstance(n, ast.Import) for a in n.names]
    names += [n.module for n in top_level if isinstance(n, ast.ImportFrom)]
    assert names == ["tkinter"]
