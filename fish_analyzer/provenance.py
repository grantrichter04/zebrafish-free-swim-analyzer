"""
fish_analyzer/provenance.py
===========================
Which code made an export, so a number in a paper can be traced back to it.

The version alone is not enough: the lab laptop runs the app from a git
checkout, and numbers can change between releases. So when the package sits
in a checkout, the version names the commit too, and says "modified" when
the code differs from that commit:

    2.3.0 (commit 6e84edf, modified)

git need not be installed for the commit: it is read from the checkout's
files. Without git there is no telling whether files were changed, so then
nothing is said about it.
"""
import shutil
import subprocess
from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _commit_from_files(git_dir: Path) -> Optional[str]:
    """The commit HEAD points at, read from .git without running git."""
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head  # a detached HEAD holds the commit itself
        ref = head[len("ref: "):]
        loose = git_dir / ref
        if loose.exists():
            return loose.read_text().strip()
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        pass
    return None


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def analyzer_version(root: Path = PACKAGE_ROOT) -> str:
    """The version, with the commit when `root` is a git checkout."""
    # Imported here: the package imports export.py, which imports this
    # module, before it defines __version__.
    from . import __version__
    if not (root / ".git").exists():
        return __version__
    modified = None
    commit = None
    if shutil.which("git"):
        commit = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain", "--untracked-files=no")
        modified = bool(status) if status is not None else None
    if not commit and (root / ".git").is_dir():
        commit = _commit_from_files(root / ".git")
    if not commit:
        return __version__
    return f"{__version__} (commit {commit[:7]}{', modified' if modified else ''})"
