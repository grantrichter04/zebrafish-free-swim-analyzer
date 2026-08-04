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

# These tests read repository files - the banner in run_analyzer.py, the README
# example - rather than importing the package, so they need the repo root as a
# path. Not a sys.path insert: the package is imported normally.
REPO = Path(__file__).resolve().parent.parent


#: `synthetic_npy` and the `app` fixture now live in conftest.py, so the GUI
#: regression tests can share them.


def test_package_imports():
    import fish_analyzer
    assert fish_analyzer.__version__


def test_version_matches_banner():
    """run_analyzer.py's banner must not drift from __version__."""
    import fish_analyzer
    src = (REPO / "run_analyzer.py").read_text(encoding="utf-8")
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


def test_gui_constructs():
    """Exercises the whole mixin stack and every tab's widget construction."""
    tk = pytest.importorskip("tkinter")
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    probe.destroy()

    from fish_analyzer import EnhancedFishAnalyzer
    app = EnhancedFishAnalyzer()
    app.root.withdraw()
    app.root.update_idletasks()
    app.root.destroy()


def test_requirements_txt_matches_pyproject():
    """Two files list the same dependencies, so they can drift apart.

    The conda path in the README was already a hand-typed package list that had
    no mechanism keeping it in step; this at least pins the two pip ones.
    Compares against the installed metadata, which is what pyproject produced.
    """
    from importlib.metadata import requires

    declared = set()
    for req in requires("fish-analyzer") or []:
        if "extra ==" in req:          # dev / standalone extras, not runtime
            continue
        declared.add(req.replace(" ", ""))

    listed = set()
    for line in (REPO / "requirements.txt").read_text(
            encoding="utf-8-sig").splitlines():
        line = line.split("#")[0].strip()
        if line:
            listed.add(line.replace(" ", ""))

    assert listed == declared, (
        "requirements.txt and pyproject.toml disagree.\n"
        f"  only in requirements.txt: {sorted(listed - declared)}\n"
        f"  only in pyproject.toml:   {sorted(declared - listed)}"
    )
