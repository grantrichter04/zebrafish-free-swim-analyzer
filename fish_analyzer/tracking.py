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
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
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


def reviewed_on(video: Path) -> Optional[str]:
    """The date (YYYY-MM-DD) the session was last saved from idtracker.ai's
    validator, or None if it never was."""
    try:
        with open(session_folder_for(video) / "session.json", encoding="utf-8") as file:
            stamp = json.load(file).get("last_validated")
    except (OSError, ValueError):
        return None
    return str(stamp)[:10] if stamp else None


def build_review_command(video: Path) -> List[str]:
    """Open idtracker.ai's validator on a tracked video's session.

    The validator shows the video with each fish's identity drawn on it, lists
    the frames it is unsure about, and lets identities be corrected. Saving
    there rewrites the session's trajectories file in place, which is the file
    the analysis loads.
    """
    return [python_executable(), "-c",
            "from idtrackerai.extra_tools.validator import "
            "idtrackerai_validate_entrypoint as run; run()",
            str(session_folder_for(video))]


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


SETUP_WINDOW_SCRIPT = Path(__file__).with_name("idtrackerai_setup_window.py")


def build_configure_command(video: Path, save_to: Path,
                            load: Optional[Path] = None) -> List[str]:
    """Open idtracker.ai's window on `video` to make or check a setup.

    The window saves to `save_to` and cannot start tracking; see
    idtrackerai_setup_window.py. `load` pre-fills it from an existing setup.
    """
    command = [python_executable(), str(SETUP_WINDOW_SCRIPT),
               "--video", str(video), "--save-to", str(save_to)]
    if load is not None:
        command += ["--load", str(load)]
    return command


def setup_path_for(folder: Path, name: str) -> Optional[Path]:
    """The .toml path for a setup called `name`, or None if the name cannot
    be a file name."""
    name = name.strip()
    if name.lower().endswith(".toml"):
        name = name[:-5].strip()
    if not name or any(c in name for c in '\\/:*?"<>|') or name in (".", ".."):
        return None
    return Path(folder) / f"{name}.toml"


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


@dataclass
class TrackOutcome:
    """How tracking one video went."""
    video: Path
    ok: bool
    stopped: bool
    seconds: float
    last_lines: List[str] = field(default_factory=list)

    def reason(self) -> str:
        """One line saying why it failed, taken from idtracker.ai's output.

        Its last lines are usually about where the log file was copied, so an
        error line is preferred when there is one.
        """
        def clean(line: str) -> str:
            # Drop the "file.py:123" column idtracker.ai's logger appends,
            # and the timestamp it starts some lines with.
            line = re.sub(r"\s+\S+\.py:\d+\s*$", "", line)
            return re.sub(r"^\d\d:\d\d:\d\d\s+", "", line.strip()).strip()

        lines = [text for text in map(clean, self.last_lines) if text]
        errors = [text for text in lines
                  if re.search(r"CRITICAL|ERROR|Error|Exception", text)]
        if errors:
            return errors[-1][:200]
        return lines[-1][:200] if lines else "idtracker.ai gave no output"


def run_tracking(video: Path, setup: Path,
                 on_line: Callable[[str], None],
                 should_stop: Callable[[], bool] = lambda: False) -> TrackOutcome:
    """Track one video. Blocks; call from a worker thread.

    Success is judged by the trajectories file existing afterwards, not by
    the exit code: idtracker.ai exits with 0 whether or not tracking worked.
    """
    video = Path(video)
    tail: deque = deque(maxlen=60)
    stopped = [False]

    def line(text: str) -> None:
        tail.append(text)
        on_line(text)

    def stop() -> bool:
        stopped[0] = stopped[0] or should_stop()
        return stopped[0]

    started = time.monotonic()
    run_process(build_track_command(video, setup), video.parent, line, stop)
    return TrackOutcome(
        video=video,
        ok=not stopped[0] and tracking_status(video) == TRACKED,
        stopped=stopped[0],
        seconds=time.monotonic() - started,
        last_lines=list(tail))
