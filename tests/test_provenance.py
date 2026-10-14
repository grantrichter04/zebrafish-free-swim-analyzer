"""Which code made an export: the version, and the commit when run from a
checkout, so numbers in a paper can be traced back (Audit I, I2)."""
import shutil
import subprocess
from pathlib import Path

import pytest

from fish_analyzer import __version__, provenance

REPO = Path(__file__).resolve().parent.parent


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


needs_git_checkout = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO / ".git").exists(),
    reason="needs git and a git checkout")


def test_the_changelog_top_entry_is_this_version():
    """Changes that move the numbers go under a version number, never under
    "Unreleased" while the app reports the previous one."""
    headings = [line for line in (REPO / "CHANGELOG.md").read_text(
        encoding="utf-8").splitlines() if line.startswith("## ")]
    assert headings[0].startswith(f"## {__version__} ")


def test_outside_a_checkout_it_is_the_version_alone(tmp_path):
    assert provenance.analyzer_version(tmp_path) == __version__


@needs_git_checkout
def test_in_a_checkout_it_names_the_commit():
    version = provenance.analyzer_version(REPO)
    commit = _git("rev-parse", "HEAD")[:7]
    assert version.startswith(f"{__version__} (commit {commit}")


@needs_git_checkout
def test_it_says_when_the_code_differs_from_the_commit():
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    assert ("modified" in provenance.analyzer_version(REPO)) == dirty


def test_the_shoaling_summary_says_which_analyzer_made_it(synthetic_npy, tmp_path):
    import csv
    from fish_analyzer import (ShoalingCalculator, ShoalingParameters,
                               TrajectoryFileLoader, export_shoaling_summary_csv)
    loaded = TrajectoryFileLoader.load_file(synthetic_npy)
    loaded.shoaling_results = ShoalingCalculator(loaded, ShoalingParameters()).calculate()
    out = tmp_path / "shoaling_summary.csv"
    export_shoaling_summary_csv({"tank": loaded}, out)
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["AnalyzerVersion"] == provenance.analyzer_version()


def _fake_checkout(root: Path, commit: str, packed: bool) -> None:
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    if packed:
        (git / "packed-refs").write_text(
            "# pack-refs with: peeled fully-peeled sorted\n"
            f"{commit} refs/heads/main\n")
    else:
        (git / "refs" / "heads" / "main").write_text(commit + "\n")


@pytest.mark.parametrize("packed", [False, True])
def test_without_git_installed_the_commit_is_read_from_the_checkout(
        tmp_path, monkeypatch, packed):
    commit = "0123456789abcdef0123456789abcdef01234567"
    _fake_checkout(tmp_path, commit, packed)
    monkeypatch.setattr(provenance.shutil, "which", lambda name: None)
    # Without git it cannot tell whether files changed, so it does not say.
    assert provenance.analyzer_version(tmp_path) == f"{__version__} (commit 0123456)"
