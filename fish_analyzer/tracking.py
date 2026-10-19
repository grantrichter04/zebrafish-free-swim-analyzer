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
        fish_B.toml
        session_fish_A/trajectories/trajectories.npy

A setup named after a video (fish_B.toml) is that video's own and is used for
it in place of the shared one.

idtracker.ai always runs as a separate process. Its window is Qt and ours is
tkinter, and a crash while tracking must not take the analyzer down with it.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from importlib import util
from pathlib import Path
from typing import Callable, List, Optional

from . import idtrackerai_rolling, provenance

#: Written into a session folder after tracking: what idtracker.ai's own
#: session.json cannot say about the run. The rolling background is passed
#: to idtracker.ai as a plain statistic, and the setup beside the videos may
#: be changed later, so both are kept here, with this app's version.
TRACKED_WITH_FILE = "freeswim_tracking.json"

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
    """Shared setup files (.toml) directly inside `folder`, sorted by name.

    A setup named after a video belongs to that video alone (see
    own_setup_for) and is left out.
    """
    own = {own_setup_for(video).name.lower() for video in find_videos(folder)}
    return sorted(
        (p for p in Path(folder).iterdir()
         if p.is_file() and p.suffix.lower() == ".toml"
         and p.name.lower() not in own),
        key=lambda p: p.name.lower())


def own_setup_for(video: Path) -> Path:
    """Where a video's own setup is kept: beside it, under its name."""
    return Path(video).with_suffix(".toml")


def setup_for(video: Path, shared: Optional[Path]) -> Optional[Path]:
    """The setup `video` is tracked with: its own if it has one, else the
    shared one."""
    own = own_setup_for(video)
    return own if own.is_file() else shared


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
    """When a video's session was last saved from the validator, or None."""
    return session_reviewed_on(session_folder_for(video))


def session_reviewed_on(session_folder: Path) -> Optional[str]:
    """The date (YYYY-MM-DD) a session was last saved from idtracker.ai's
    validator, or None if it never was."""
    try:
        with open(Path(session_folder) / "session.json", encoding="utf-8") as file:
            stamp = json.load(file).get("last_validated")
    except (OSError, ValueError):
        return None
    return str(stamp)[:10] if stamp else None


#: Kept in a session folder: what was chosen for the session in this app and
#: should come back with it, so far its experimental group.
SESSION_CHOICES_FILE = "freeswim_session.json"


def saved_group(session_folder: Path) -> Optional[str]:
    """The group last set for the session in its folder, or None."""
    try:
        with open(Path(session_folder) / SESSION_CHOICES_FILE,
                  encoding="utf-8") as file:
            group = json.load(file).get("group")
    except (OSError, ValueError, AttributeError):
        return None
    return str(group) if group else None


def save_group(session_folder: Path, group: Optional[str]) -> bool:
    """Keep `group` in the session folder (None forgets it). Returns False if
    the folder could not be written, for example on a read-only drive."""
    path = Path(session_folder) / SESSION_CHOICES_FILE
    try:
        with open(path, encoding="utf-8") as file:
            choices = json.load(file)
        if not isinstance(choices, dict):
            choices = {}
    except (OSError, ValueError):
        choices = {}
    if group:
        choices["group"] = group
    else:
        choices.pop("group", None)
    try:
        path.write_text(json.dumps(choices, indent=2), encoding="utf-8")
    except OSError:
        return False
    return True


def build_review_command(session_folder: Path) -> List[str]:
    """Open idtracker.ai's validator on a session folder.

    The validator shows the video with each fish's identity drawn on it, lists
    the frames it is unsure about, and lets identities be corrected. Saving
    there rewrites the session's trajectories file in place, which is the file
    the analysis loads.
    """
    return [python_executable(), "-c",
            "from idtrackerai.extra_tools.validator import "
            "idtrackerai_validate_entrypoint as run; run()",
            str(session_folder)]


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
TRACK_SCRIPT = Path(__file__).with_name("idtrackerai_track.py")


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


def build_track_command(video: Path, setup: Path,
                        finish_flag: Path) -> List[str]:
    """Track `video` with `setup`, with no window.

    It runs through idtrackerai_track.py, which is idtracker.ai plus a way to
    finish its identity training early: creating the file `finish_flag`. A
    setup that asks for a rolling background gets one there too, and
    idtracker.ai is given the plain statistic, the only kind it knows.
    idtracker.ai applies command-line arguments after the setup file, so the
    video and name stored in a setup saved from another video are overridden.
    """
    video = Path(video)
    rolling, stat = idtrackerai_rolling.setup_statistic(setup)
    return [python_executable(), str(TRACK_SCRIPT),
            "--finish-flag", str(finish_flag),
            *(["--rolling-background", stat] if rolling else []),
            "--load", str(setup),
            *(["--background_subtraction_stat", stat] if rolling else []),
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
                 should_stop: Callable[[], bool] = lambda: False,
                 should_finish_training: Callable[[], bool] = lambda: False
                 ) -> TrackOutcome:
    """Track one video. Blocks; call from a worker thread.

    When `should_finish_training` returns true, idtracker.ai is asked, once,
    to stop learning identities and carry on with its best result so far.

    Success is judged by the trajectories file existing afterwards, not by
    the exit code: idtracker.ai exits with 0 whether or not tracking worked.
    """
    video = Path(video)
    tail: deque = deque(maxlen=60)
    stopped = [False]

    def line(text: str) -> None:
        tail.append(text)
        on_line(text)

    asked = [False]

    def stop() -> bool:
        # Polled several times a second while idtracker.ai runs, which makes
        # it the place to pass on a request to finish training too.
        if not asked[0] and should_finish_training():
            asked[0] = True
            finish_flag.touch()
        stopped[0] = stopped[0] or should_stop()
        return stopped[0]

    started = time.monotonic()
    with tempfile.TemporaryDirectory() as folder:
        finish_flag = Path(folder) / "finish_training"
        run_process(build_track_command(video, setup, finish_flag),
                    video.parent, line, stop)
    seconds = time.monotonic() - started
    ok = not stopped[0] and tracking_status(video) == TRACKED
    if ok:
        _record_how_it_was_tracked(video, setup, seconds)
    return TrackOutcome(
        video=video,
        ok=ok,
        stopped=stopped[0],
        seconds=seconds,
        last_lines=list(tail))


def tracking_seconds(video: Path) -> Optional[float]:
    """How long this app took to track `video`, or None if not recorded
    (tracked before this was kept, or outside the app)."""
    try:
        with open(session_folder_for(video) / TRACKED_WITH_FILE,
                  encoding="utf-8") as file:
            seconds = json.load(file).get("tracking_seconds")
    except (OSError, ValueError, AttributeError):
        return None
    return float(seconds) if isinstance(seconds, (int, float)) else None


def format_duration(seconds: float) -> str:
    """1:05:09 for over an hour, 12:34 for less."""
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return (f"{hours}:{minutes:02d}:{secs:02d}" if hours
            else f"{minutes}:{secs:02d}")


def _record_how_it_was_tracked(video: Path, setup: Path,
                               seconds: Optional[float] = None) -> None:
    """Write TRACKED_WITH_FILE into the video's new session folder."""
    try:
        setup_text = Path(setup).read_text(encoding="utf-8")
    except OSError:
        setup_text = None
    record = {"analyzer_version": provenance.analyzer_version(),
              "rolling_background": idtrackerai_rolling.setup_statistic(setup)[0],
              "setup": Path(setup).name,
              "setup_text": setup_text,
              "tracking_seconds": None if seconds is None else round(seconds, 1)}
    try:
        (session_folder_for(video) / TRACKED_WITH_FILE).write_text(
            json.dumps(record, indent=2), encoding="utf-8")
    except OSError:
        pass  # the tracking itself succeeded; the export will say "not recorded"
