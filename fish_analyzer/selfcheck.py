"""
fish_analyzer/selfcheck.py
==========================
Is this machine set up to run the analyzer, and to track videos?

One function returns the answer as data, so `fish-analyzer --check`, the
Check Setup button and install.bat all report the same thing.

A check is either required or not. Only a failed required check makes the
exit code non-zero: without idtracker.ai or a GPU the analysis tabs still
work, so those are reported as warnings.
"""
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import List, Optional, Tuple

MIN_PYTHON = (3, 10)


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str
    required: bool


def _python_version() -> Tuple[int, ...]:
    return tuple(sys.version_info[:3])


def _package_version(distribution: str) -> Optional[str]:
    """Installed version of a distribution, or None. Does not import it."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _gpu_status() -> Tuple[bool, str]:
    """(usable, description). Imports torch, which takes a few seconds."""
    try:
        import torch
    except Exception as exc:
        return False, f"PyTorch could not be loaded ({exc.__class__.__name__})"
    if not torch.cuda.is_available():
        return False, "no NVIDIA GPU visible to PyTorch"
    return True, torch.cuda.get_device_name(0)


def run_checks() -> List[CheckItem]:
    from . import __version__

    version = _python_version()
    python = ".".join(str(part) for part in version)
    python_ok = version[:2] >= MIN_PYTHON
    items = [
        CheckItem("Python", python_ok,
                  python if python_ok else
                  f"{python} - needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer",
                  required=True),
        CheckItem("Analyzer", True, __version__, required=True),
    ]

    opencv = (_package_version("opencv-python-headless")
              or _package_version("opencv-python"))
    items.append(CheckItem(
        "OpenCV", opencv is not None,
        opencv or "not installed - the Video Inspector is disabled",
        required=False))

    idtrackerai = _package_version("idtrackerai")
    items.append(CheckItem(
        "idtracker.ai", idtrackerai is not None,
        idtrackerai or "not installed - the Tracking tab is disabled",
        required=False))

    torch_version = _package_version("torch")
    items.append(CheckItem(
        "PyTorch", torch_version is not None,
        torch_version or "not installed", required=False))

    gpu_ok, gpu_detail = _gpu_status()
    items.append(CheckItem("GPU", gpu_ok, gpu_detail, required=False))
    return items


def exit_code(items: List[CheckItem]) -> int:
    return 1 if any(i.required and not i.ok for i in items) else 0


def format_report(items: List[CheckItem]) -> str:
    def mark(item: CheckItem) -> str:
        if item.ok:
            return "[ OK ]"
        return "[FAIL]" if item.required else "[WARN]"

    lines = [f"{mark(i)} {i.name:<13} {i.detail}" for i in items]
    lines.append("")
    if exit_code(items):
        lines.append("The analyzer cannot run on this setup.")
    elif all(i.ok for i in items):
        lines.append("Everything is ready.")
    else:
        lines.append("Analysis will work. Tracking will not until the "
                     "warnings above are fixed - re-run install.bat.")
    return "\n".join(lines)
