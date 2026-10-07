"""
fish_analyzer/tracking.py
=========================
Running idtracker.ai on a folder of videos. No GUI code lives here.

An experiment folder holds videos, the setup files (.toml) saved from
idtracker.ai's own window, and one session_<video name> folder per tracked
video - which is where file_loading.py already looks for a session and for the
video that belongs to it:

    experiment/
        fish_A.avi
        fish_B.avi
        my_setup.toml
        session_fish_A/trajectories/trajectories.npy

idtracker.ai always runs as a separate process. Its window is Qt and ours is
tkinter, and a crash while tracking must not take the analyzer down with it.
"""
import os
import subprocess
import sys
import threading
import time
from importlib import util
from pathlib import Path
from typing import Callable, List, Optional

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg"}

NOT_TRACKED = "not tracked"
TRACKED = "tracked"
INCOMPLETE = "incomplete"


def find_videos(folder: Path) -> List[Path]:
    """Video files directly inside `folder`, sorted by name."""
    return sorted(
        (p for p in Path(folder).iterdir()
         if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda p: p.name.lower())


def find_setups(folder: Path) -> List[Path]:
    """Setup files (.toml) directly inside `folder`, sorted by name."""
    return sorted(
        (p for p in Path(folder).iterdir()
         if p.is_file() and p.suffix.lower() == ".toml"),
        key=lambda p: p.name.lower())


def session_folder_for(video: Path) -> Path:
    """Where idtracker.ai writes this video's session when given --name stem."""
    video = Path(video)
    return video.parent / f"session_{video.stem}"


def tracking_status(video: Path) -> str:
    """TRACKED, INCOMPLETE or NOT_TRACKED, read from the folder alone.

    Nothing is remembered between runs, so the answer is right after a
    restart and a batch picks up where it stopped.
    """
    session = session_folder_for(video)
    if (session / "trajectories" / "trajectories.npy").is_file():
        return TRACKED
    if session.is_dir():
        return INCOMPLETE
    return NOT_TRACKED


def idtrackerai_available() -> bool:
    """Whether idtracker.ai is installed. Does not import it (or torch)."""
    return util.find_spec("idtrackerai") is not None


def python_executable() -> str:
    """The interpreter to run idtracker.ai with.

    The desktop shortcut starts the app with pythonw.exe, which has no
    stdout. idtracker.ai logs to stdout, so it gets python.exe from the same
    environment and its window is suppressed in run_process instead.
    """
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        console = exe.with_name("python.exe")
        if console.is_file():
            return str(console)
    return str(exe)


def build_configure_command(video: Path, setup: Optional[Path] = None) -> List[str]:
    """Open idtracker.ai's own window on `video`, optionally loading a setup."""
    command = [python_executable(), "-m", "idtrackerai.start"]
    if setup is not None:
        command += ["--load", str(setup)]
    return command + ["--video_paths", str(video)]


def build_track_command(video: Path, setup: Path) -> List[str]:
    """Track `video` with `setup`, with no window.

    idtracker.ai applies command-line arguments after the setup file, so the
    video and name stored in a setup saved from another video are overridden.
    """
    video = Path(video)
    return [python_executable(), "-m", "idtrackerai.start",
            "--load", str(setup),
            "--video_paths", str(video),
            "--name", video.stem,
            "--track"]


def _stop_process_tree(process: subprocess.Popen) -> None:
    """idtracker.ai starts worker processes; ending only the parent leaves
    them holding the GPU."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                       capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        process.terminate()


def run_process(command: List[str], cwd: Path,
                on_line: Callable[[str], None],
                should_stop: Callable[[], bool] = lambda: False) -> int:
    """Run `command`, passing each line it prints to `on_line`.

    Blocks until the process ends, so call it from a worker thread. If
    `should_stop` returns true the process and its children are ended.
    Returns the exit code.
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               COLUMNS="160")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(
        command, cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", creationflags=flags)

    def pump():
        for line in process.stdout:
            on_line(line.rstrip("\r\n"))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    while process.poll() is None:
        if should_stop():
            _stop_process_tree(process)
            break
        time.sleep(0.2)
    process.wait()
    reader.join(timeout=5)
    return process.returncode
