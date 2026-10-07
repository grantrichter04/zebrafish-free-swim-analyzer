# Phase 1: Installation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One conda env holds idtracker.ai and the analyzer, installed by double-clicking `install.bat`, with a self-check that says in plain words whether the laptop is ready.

**Architecture:** `pyproject.toml` becomes the only dependency list and stops conflicting with idtracker.ai's OpenCV. A new `fish_analyzer/selfcheck.py` returns structured check results that the `--check` flag and a "Check Setup" button both display. `install.bat` builds the env from pinned versions and runs that check.

**Tech Stack:** Python 3.12, setuptools, conda, pip constraints file, Windows batch, tkinter, pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-ra-ready-pipeline-design.md`, Phase 1.

**One deviation from the spec:** the spec says the self-check is reachable "from the app's Help menu". The app has no menu bar; it has a status bar with a "Show Log" button. This plan adds a "Check Setup" button beside it instead of introducing a menu bar for one item.

**Conventions for every task:**

- Work on branch `polish/ra-ready`.
- The interpreter is the test env built on 2026-10-08. In Git Bash:
  `PY=~/.conda/envs/freeswim/python.exe`
- Run tests with `$PY -m pytest -q`. Baseline before this plan: 188 passed.
- End every commit message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File map

| File | Change | Responsibility |
|---|---|---|
| `pyproject.toml` | modify | Sole dependency list; headless OpenCV; `tracking` extra; Python ≥ 3.10 |
| `requirements.txt` | delete | Duplicate of the above |
| `environment.yml` | rewrite | Python 3.12 and pip only |
| `.github/workflows/tests.yml` | modify | CI on 3.12 only |
| `fish_analyzer/video_utils.py`, `fish_analyzer/gui/inspector_tab.py` | modify | Install hints name the headless package |
| `fish_analyzer/selfcheck.py` | create | Run the checks, format the report, decide the exit code |
| `fish_analyzer/__main__.py` | modify | `--check` flag |
| `fish_analyzer/gui/base.py` | modify | "Check Setup" button and window |
| `constraints-win-cu128.txt` | create | Pinned versions of the tested env |
| `install.bat` | create | One-time laptop setup |
| `README.md` | modify | Installation section only (full rewrite is Phase 3) |
| `tests/test_smoke.py` | modify | Dependency declarations |
| `tests/test_selfcheck.py` | create | Self-check behaviour |
| `tests/test_gui_regressions.py` | modify | "Check Setup" window |

---

### Task 1: One dependency list, no OpenCV conflict

**Files:**
- Modify: `pyproject.toml`
- Delete: `requirements.txt`
- Modify: `environment.yml`, `.github/workflows/tests.yml`
- Modify: `fish_analyzer/video_utils.py:27,114`, `fish_analyzer/gui/inspector_tab.py:967,1014`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Replace the requirements test with a dependency-declaration test**

In `tests/test_smoke.py`, delete the whole `test_requirements_txt_matches_pyproject` function and put this in its place:

```python
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
```

- [ ] **Step 2: Run it and watch it fail**

Run: `$PY -m pytest tests/test_smoke.py -q -k "dependencies or requirements_txt"`
Expected: 2 failed (headless OpenCV not declared; `requirements.txt` exists).

- [ ] **Step 3: Edit `pyproject.toml`**

Change `requires-python = ">=3.9"` to:

```toml
requires-python = ">=3.10"
```

Replace the comment and `dependencies` list with:

```toml
# The only dependency list. environment.yml installs from here.
dependencies = [
    "numpy>=1.24",
    "pandas>=1.5",
    "matplotlib>=3.7",
    "scipy>=1.10",
    "traja==25.0.1",
    # traja imports sklearn unconditionally but does not declare it, so
    # `import traja` fails without this. Not optional.
    "scikit-learn>=1.0",
    # Functionally optional - without them thigmotaxis and the video inspector
    # are disabled rather than crashing - but installed by default because
    # nearly every workflow uses them.
    "shapely>=2.0",
    # Headless, not opencv-python: idtracker.ai installs the headless build
    # and the two cannot share an environment. Nothing here opens a cv2 window.
    "opencv-python-headless>=4.10",
]
```

In `[project.optional-dependencies]`, add after the `dev` entry:

```toml
# pip install -e ".[tracking]" - adds idtracker.ai for the Tracking tab.
# Install GPU PyTorch first; see install.bat.
tracking = [
    "idtrackerai>=6.0.14,<6.1",
]
```

- [ ] **Step 4: Delete `requirements.txt` and rewrite `environment.yml`**

```bash
git rm -q requirements.txt
```

`environment.yml`, whole file:

```yaml
# Conda environment for the Zebrafish Free Swim Analyzer.
#
# On the lab laptop, run install.bat instead: it also installs GPU PyTorch,
# idtracker.ai and the desktop shortcut.
#
# For analysis only (no tracking), on any machine:
#   conda env create -f environment.yml
#   conda activate freeswim
#   pip install -e ".[dev]"
#
# Dependencies are declared once, in pyproject.toml.
name: freeswim

channels:
  - conda-forge

dependencies:
  - python=3.12
  - pip
```

- [ ] **Step 5: CI on 3.12 only**

In `.github/workflows/tests.yml`, replace the three matrix lines

```yaml
        # The floor and the recommended default. README claims both work; this
        # is what makes that claim mean something.
        python-version: ["3.9", "3.12"]
```

with

```yaml
        # The version install.bat builds. idtracker.ai needs 3.10 or newer.
        python-version: ["3.12"]
```

- [ ] **Step 6: Fix the four install hints**

In `fish_analyzer/video_utils.py` (lines 27 and 114) and `fish_analyzer/gui/inspector_tab.py` (lines 967 and 1014), replace every `opencv-python` with `opencv-python-headless`. Check nothing was missed:

Run: `grep -rn "opencv-python" fish_analyzer/ | grep -v headless`
Expected: no output.

- [ ] **Step 7: Refresh the installed metadata and run the suite**

```bash
$PY -m pip install -q --no-warn-script-location -e ".[dev,tracking]"
$PY -m pip check
$PY -m pytest -q
```

Expected: `pip check` prints `No broken requirements found.`; pytest reports 189 passed (188 − 1 removed + 2 added).

- [ ] **Step 8: Commit**

```bash
git add -A pyproject.toml requirements.txt environment.yml .github/workflows/tests.yml fish_analyzer/video_utils.py fish_analyzer/gui/inspector_tab.py tests/test_smoke.py
git commit -m "Make pyproject the only dependency list and share OpenCV with idtracker.ai"
```

---

### Task 2: The self-check module

**Files:**
- Create: `fish_analyzer/selfcheck.py`
- Test: `tests/test_selfcheck.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_selfcheck.py`, whole file:

```python
"""The setup self-check: what install.bat and the Check Setup button report.

Nothing here needs idtracker.ai, torch or a GPU - the lookups are replaced -
so it runs the same in CI as on the lab laptop.
"""
import fish_analyzer
from fish_analyzer import selfcheck


def _with(monkeypatch, versions, gpu):
    monkeypatch.setattr(selfcheck, "_package_version", versions.get)
    monkeypatch.setattr(selfcheck, "_gpu_status", lambda: gpu)
    return {item.name: item for item in selfcheck.run_checks()}


FULL = {"opencv-python-headless": "5.0.0.93", "idtrackerai": "6.0.14",
        "torch": "2.11.0+cu128"}


def test_fully_set_up_laptop_passes_everything(monkeypatch):
    items = _with(monkeypatch, FULL, (True, "NVIDIA GeForce RTX 4070"))

    assert [i.name for i in selfcheck.run_checks()] == [
        "Python", "Analyzer", "OpenCV", "idtracker.ai", "PyTorch", "GPU"]
    assert all(i.ok for i in items.values())
    assert items["Analyzer"].detail == fish_analyzer.__version__
    assert items["GPU"].detail == "NVIDIA GeForce RTX 4070"
    assert selfcheck.exit_code(list(items.values())) == 0


def test_missing_idtrackerai_is_a_warning_not_a_failure(monkeypatch):
    """Analysis still works without tracking, so this must not fail the check."""
    items = _with(monkeypatch, {"opencv-python-headless": "5.0.0.93"},
                  (False, "PyTorch is not installed"))

    assert not items["idtracker.ai"].ok
    assert not items["idtracker.ai"].required
    assert "not installed" in items["idtracker.ai"].detail
    assert not items["GPU"].ok
    assert selfcheck.exit_code(list(items.values())) == 0


def test_old_python_fails_the_check(monkeypatch):
    monkeypatch.setattr(selfcheck, "_python_version", lambda: (3, 9, 18))
    items = _with(monkeypatch, FULL, (True, "GPU"))

    assert not items["Python"].ok
    assert items["Python"].required
    assert selfcheck.exit_code(list(items.values())) == 1


def test_report_says_what_is_wrong_in_plain_words(monkeypatch):
    items = _with(monkeypatch, {"opencv-python-headless": "5.0.0.93"},
                  (False, "PyTorch is not installed"))
    report = selfcheck.format_report(list(items.values()))

    assert "[ OK ] Analyzer" in report
    assert "[WARN] idtracker.ai" in report
    assert "Analysis will work. Tracking will not" in report


def test_report_for_a_ready_laptop(monkeypatch):
    items = _with(monkeypatch, FULL, (True, "GPU"))
    assert "Everything is ready." in selfcheck.format_report(list(items.values()))


def test_regular_opencv_counts_when_headless_is_absent(monkeypatch):
    items = _with(monkeypatch, {"opencv-python": "4.12.0"}, (False, "x"))
    assert items["OpenCV"].ok and items["OpenCV"].detail == "4.12.0"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `$PY -m pytest tests/test_selfcheck.py -q`
Expected: collection error, `cannot import name 'selfcheck'`.

- [ ] **Step 3: Write the module**

`fish_analyzer/selfcheck.py`, whole file:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_selfcheck.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add fish_analyzer/selfcheck.py tests/test_selfcheck.py
git commit -m "Add a setup self-check that reports analyzer, idtracker.ai and GPU status"
```

---

### Task 3: `fish-analyzer --check`

**Files:**
- Modify: `fish_analyzer/__main__.py`
- Test: `tests/test_selfcheck.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_selfcheck.py`:

```python
def test_check_flag_prints_the_report_and_does_not_open_the_gui(
        monkeypatch, capsys):
    from fish_analyzer import __main__ as entry

    monkeypatch.setattr(selfcheck, "_package_version", FULL.get)
    monkeypatch.setattr(selfcheck, "_gpu_status", lambda: (True, "GPU"))
    monkeypatch.setattr(fish_analyzer, "EnhancedFishAnalyzer",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("--check must not start the GUI")))

    assert entry.main(["--check"]) == 0
    assert "Everything is ready." in capsys.readouterr().out


def test_check_flag_returns_nonzero_when_a_required_check_fails(monkeypatch):
    from fish_analyzer import __main__ as entry

    monkeypatch.setattr(selfcheck, "_python_version", lambda: (3, 9, 18))
    monkeypatch.setattr(selfcheck, "_package_version", FULL.get)
    monkeypatch.setattr(selfcheck, "_gpu_status", lambda: (True, "GPU"))

    assert entry.main(["--check"]) == 1
```

- [ ] **Step 2: Run them and watch them fail**

Run: `$PY -m pytest tests/test_selfcheck.py -q -k check_flag`
Expected: 2 failed, `main() takes 0 positional arguments but 1 was given`.

- [ ] **Step 3: Rewrite `fish_analyzer/__main__.py`**

Whole file:

```python
"""
fish_analyzer/__main__.py
=========================
Entry point for `python -m fish_analyzer` and the `fish-analyzer` command.

    fish-analyzer            open the application
    fish-analyzer --check    report whether this machine is set up, and exit

run_analyzer.py at the repo root opens the application too and still works
from a plain checkout, without installing anything. This module is what an
installed copy and the desktop shortcut use.
"""
import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="fish-analyzer",
        description="Zebrafish Free Swim Analyzer")
    parser.add_argument(
        "--check", action="store_true",
        help="report whether this machine is set up, then exit")
    args = parser.parse_args(argv)

    if args.check:
        from . import selfcheck
        items = selfcheck.run_checks()
        print(selfcheck.format_report(items))
        return selfcheck.exit_code(items)

    from . import EnhancedFishAnalyzer, __version__

    print("=" * 60)
    print(f"Fish Trajectory Analyzer v{__version__}")
    print("=" * 60)
    print()
    print("Starting GUI application...")
    print("(Close the window to exit)")
    print()

    EnhancedFishAnalyzer().run()

    print("Application closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests, then the real command**

Run: `$PY -m pytest tests/test_selfcheck.py -q`
Expected: 8 passed.

Run: `$PY -m fish_analyzer --check; echo "exit=$?"`
Expected: six lines all `[ OK ]`, the GPU line naming the RTX card, `Everything is ready.`, `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add fish_analyzer/__main__.py tests/test_selfcheck.py
git commit -m "Add fish-analyzer --check"
```

---

### Task 4: "Check Setup" button

**Files:**
- Modify: `fish_analyzer/gui/base.py` (`_setup_gui`, and a new method after `_show_log_window`)
- Test: `tests/test_gui_regressions.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui_regressions.py`:

```python
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
```

- [ ] **Step 2: Run it and watch it fail**

Run: `$PY -m pytest tests/test_gui_regressions.py -q -k check_setup`
Expected: 1 failed, `AttributeError: ... has no attribute '_show_setup_check'`.

- [ ] **Step 3: Add the button**

In `fish_analyzer/gui/base.py`, in `_setup_gui`, directly after the `tk.Button(... text="Show Log" ...).pack(...)` statement, add:

```python
        tk.Button(
            status_frame, text="Check Setup", command=self._show_setup_check,
            font=("Arial", 8), relief="flat", padx=6
        ).pack(side="right", padx=(0, 4), pady=1)
```

- [ ] **Step 4: Add the window**

In the same file, directly after the `_show_log_window` method, add:

```python
    def _show_setup_check(self):
        """Open a window reporting whether this machine is set up.

        The same report `fish-analyzer --check` prints. Loading PyTorch to ask
        about the GPU takes a few seconds, hence the status message first.
        """
        from .. import selfcheck

        self.set_status("Checking setup...")
        self.root.update_idletasks()
        report = selfcheck.format_report(selfcheck.run_checks())
        self.set_status("Ready")

        win = tk.Toplevel(self.root)
        win.title("Setup Check")
        win.geometry("700x260")
        text = tk.Text(win, wrap="word", font=("Courier", 10), padx=10, pady=10)
        text.pack(fill="both", expand=True)
        text.insert("1.0", report)
        text.config(state="disabled")
        return win
```

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest -q`
Expected: 198 passed (189 + 8 self-check + 1 GUI).

- [ ] **Step 6: Commit**

```bash
git add fish_analyzer/gui/base.py tests/test_gui_regressions.py
git commit -m "Add a Check Setup button to the status bar"
```

---

### Task 5: Pinned versions and `install.bat`

**Files:**
- Create: `constraints-win-cu128.txt`
- Create: `install.bat`

- [ ] **Step 1: Generate the constraints file from the tested env**

```bash
$PY -c "
import subprocess, sys
lines = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--exclude-editable'],
                       capture_output=True, text=True, check=True).stdout.splitlines()
keep = [l.split('+')[0] for l in lines if '==' in l and not l.startswith('fish-analyzer')]
header = '''# Versions of the environment tested on 2026-10-08 (Windows 11, Python 3.12,
# CUDA 12.8 builds of PyTorch). install.bat installs against this file so a
# reinstall reproduces the tested environment instead of whatever is newest.
#
# To move to newer versions: build a fresh env without -c, run the test suite
# and idtrackerai_test, then regenerate this file from pip freeze.
'''
open('constraints-win-cu128.txt', 'w', newline='\n').write(header + '\n'.join(sorted(keep, key=str.lower)) + '\n')
"
grep -E "^(torch|torchvision|idtrackerai|opencv-python-headless|numpy|traja)==" constraints-win-cu128.txt
```

Expected output:

```
idtrackerai==6.0.14
numpy==2.5.2
opencv-python-headless==5.0.0.93
torch==2.11.0
torchvision==0.26.0
traja==25.0.1
```

(`+cu128` is stripped on purpose: a constraint of `torch==2.11.0` accepts the `2.11.0+cu128` build, and local version labels are not allowed in constraints for packages fetched from PyPI.)

- [ ] **Step 2: Write `install.bat`**

Whole file. Save with CRLF line endings.

```bat
@echo off
rem ===========================================================================
rem  Zebrafish Free Swim Analyzer - one-time setup for the lab laptop.
rem
rem  Double-click this file. It builds one conda environment holding both
rem  idtracker.ai and the analyzer, checks it, and puts a shortcut on the
rem  desktop. Safe to run again: it reuses the environment if it exists.
rem
rem  Needs: Miniconda or Anaconda, an NVIDIA GPU with a current driver, and
rem  an internet connection (about 5 GB is downloaded).
rem
rem  For testing: set FREESWIM_ENV to use another environment name,
rem  FREESWIM_NO_SHORTCUT=1 to skip the shortcut, FREESWIM_NO_PAUSE=1 to
rem  skip the final pause.
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
if not defined FREESWIM_ENV set "FREESWIM_ENV=freeswim"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"

echo.
echo  Zebrafish Free Swim Analyzer - setup
echo  ------------------------------------
echo.

rem --- 1. Find conda --------------------------------------------------------
set "CONDA="
for /f "delims=" %%I in ('where conda.bat 2^>nul') do if not defined CONDA set "CONDA=%%I"
for %%P in ("%ProgramData%\miniconda3" "%UserProfile%\miniconda3" "%LocalAppData%\miniconda3" "%ProgramData%\anaconda3" "%UserProfile%\anaconda3") do (
    if not defined CONDA if exist "%%~P\condabin\conda.bat" set "CONDA=%%~P\condabin\conda.bat"
)
if not defined CONDA (
    echo  [STOP] Conda was not found.
    echo         Install Miniconda from https://www.anaconda.com/download/success
    echo         then run this file again.
    goto :fail
)
echo  [1/6] Found conda: %CONDA%

rem --- 2. Check for an NVIDIA GPU -------------------------------------------
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo  [STOP] No NVIDIA driver was found ^(nvidia-smi is missing^).
    echo         Tracking needs an NVIDIA GPU. Install the current driver from
    echo         https://www.nvidia.com/drivers then run this file again.
    goto :fail
)
echo  [2/6] Found an NVIDIA driver.

rem --- 3. Create or reuse the environment -----------------------------------
call "%CONDA%" env list | findstr /R /C:"^%FREESWIM_ENV% " >nul
if errorlevel 1 (
    echo  [3/6] Creating the "%FREESWIM_ENV%" environment...
    call "%CONDA%" create -n %FREESWIM_ENV% python=3.12 pip -y -q
    if errorlevel 1 goto :fail
) else (
    echo  [3/6] Reusing the existing "%FREESWIM_ENV%" environment.
)
set "PY="
for /f "delims=" %%I in ('call "%CONDA%" run -n %FREESWIM_ENV% python -c "import sys; print(sys.executable)"') do set "PY=%%I"
if not defined PY (
    echo  [STOP] Could not find Python inside the "%FREESWIM_ENV%" environment.
    goto :fail
)

rem --- 4. GPU PyTorch -------------------------------------------------------
echo  [4/6] Installing PyTorch for the GPU ^(the large download^)...
"%PY%" -m pip install -q --no-warn-script-location torch torchvision --index-url %TORCH_INDEX% -c constraints-win-cu128.txt
if errorlevel 1 goto :fail

rem --- 5. idtracker.ai and the analyzer -------------------------------------
echo  [5/6] Installing idtracker.ai and the analyzer...
"%PY%" -m pip install -q --no-warn-script-location -e ".[tracking]" -c constraints-win-cu128.txt
if errorlevel 1 goto :fail

rem --- 6. Check, then shortcut ----------------------------------------------
echo  [6/6] Checking the installation...
echo.
"%PY%" -m fish_analyzer --check
if errorlevel 1 goto :fail
echo.

if "%FREESWIM_NO_SHORTCUT%"=="1" goto :done
for %%I in ("%PY%") do set "PYW=%%~dpIpythonw.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'Free Swim Analyzer.lnk'));" ^
  "$s.TargetPath = $env:PYW; $s.Arguments = '-m fish_analyzer';" ^
  "$s.WorkingDirectory = [Environment]::GetFolderPath('MyDocuments');" ^
  "$s.Description = 'Zebrafish Free Swim Analyzer'; $s.Save()"
if errorlevel 1 (
    echo  [WARN] The desktop shortcut could not be created. The install itself is fine.
) else (
    echo  A "Free Swim Analyzer" shortcut is on the desktop.
)

:done
echo.
echo  Setup finished.
if not "%FREESWIM_NO_PAUSE%"=="1" pause
exit /b 0

:fail
echo.
echo  Setup did NOT finish. Read the message above, fix it, and run this again.
if not "%FREESWIM_NO_PAUSE%"=="1" pause
exit /b 1
```

- [ ] **Step 3: Run the installer against a throwaway environment**

This is the real test of the installer: a from-nothing build. It downloads about 5 GB and takes several minutes. Run from the repo root in Git Bash:

```bash
FREESWIM_ENV=freeswim-installtest FREESWIM_NO_SHORTCUT=1 FREESWIM_NO_PAUSE=1 cmd //c install.bat; echo "exit=$?"
```

Expected: steps `[1/6]` to `[6/6]`, then six `[ OK ]` lines, `Everything is ready.`, `Setup finished.`, `exit=0`.

- [ ] **Step 4: Run the test suite inside the throwaway environment**

```bash
T=~/.conda/envs/freeswim-installtest/python.exe
$T -m pip check
$T -m pip install -q pytest
$T -m pytest -q
```

Expected: `No broken requirements found.` and 198 passed.

- [ ] **Step 5: Run it a second time to confirm it is re-runnable**

```bash
FREESWIM_ENV=freeswim-installtest FREESWIM_NO_SHORTCUT=1 FREESWIM_NO_PAUSE=1 cmd //c install.bat; echo "exit=$?"
```

Expected: step 3 prints `Reusing the existing "freeswim-installtest" environment.`, `exit=0`, and it finishes in under a minute.

- [ ] **Step 6: Create the real shortcut and confirm it opens the app**

```bash
FREESWIM_NO_PAUSE=1 cmd //c install.bat; echo "exit=$?"
```

Expected: `A "Free Swim Analyzer" shortcut is on the desktop.` and `exit=0`. Then double-click the shortcut: the analyzer window opens with no console window behind it, and "Check Setup" in the status bar shows the same six `[ OK ]` lines.

- [ ] **Step 7: Remove the throwaway environment**

```bash
conda env remove -n freeswim-installtest -y
```

- [ ] **Step 8: Commit**

```bash
git add constraints-win-cu128.txt install.bat
git commit -m "Add install.bat and the pinned versions it installs"
```

---

### Task 6: README installation section

**Files:**
- Modify: `README.md` (the `## Installation` section only, from that heading to the `---` before `## Usage`)

- [ ] **Step 1: Replace the section**

Replace everything from `## Installation` up to, but not including, the `---` line that precedes `## Usage` with:

````markdown
## Installation

### On the lab laptop (tracking and analysis)

1. Install [Miniconda](https://www.anaconda.com/download/success) and a current
   NVIDIA driver, if they are not already there.
2. Download or clone this repository.
3. Double-click **`install.bat`**.

It builds one conda environment, `freeswim`, holding both idtracker.ai and the
analyzer, checks it, and puts a **Free Swim Analyzer** shortcut on the desktop.
It downloads about 5 GB and is safe to run again.

To check a machine at any time, press **Check Setup** at the bottom right of
the app, or run:

```bash
python -m fish_analyzer --check
```

### On any other machine (analysis only)

```bash
conda env create -f environment.yml
conda activate freeswim
pip install -e ".[dev]"
pytest -q
```

This skips idtracker.ai and PyTorch. Every analysis tab works; tracking does
not.

### Requirements

- **Python 3.10 or newer**; 3.12 is what `install.bat` builds and what CI tests.
- Dependencies are declared once, in [`pyproject.toml`](pyproject.toml).
  [`constraints-win-cu128.txt`](constraints-win-cu128.txt) pins the exact
  versions the lab laptop install was tested with.
- OpenCV is installed as `opencv-python-headless`, the same build idtracker.ai
  uses. Do not also install `opencv-python` into the same environment.

The standalone scripts (`fish_posture_analyzer.py`, `head_detection/`) need
more:

```bash
pip install -e ".[standalone]"
```

````

- [ ] **Step 2: Check nothing in the README still points at removed things**

Run: `grep -n "requirements.txt\|fishanalyzer\|3\.9" README.md`
Expected: no output.

- [ ] **Step 3: Run the suite**

Run: `$PY -m pytest -q`
Expected: 198 passed (one smoke test reads the README's API example; it must still pass).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document install.bat and the single environment"
```

---

## Done when

- `$PY -m pytest -q` reports 198 passed.
- `$PY -m pip check` reports no broken requirements in an env holding both idtracker.ai and the analyzer.
- `install.bat` built a working env from nothing (Task 5, Step 3) and the desktop shortcut opens the app.
- CI is green on the pushed branch.
