"""fish_analyzer.tracking: finding videos and setups, and running idtracker.ai.

None of this needs idtracker.ai or a GPU. Folder layouts are built in a temp
dir, and run_process is driven by a stand-in Python one-liner.
"""
import sys
import time
from pathlib import Path

from fish_analyzer import tracking


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_find_videos_lists_only_top_level_videos_sorted(tmp_path):
    _touch(tmp_path / "b_fish.AVI")
    _touch(tmp_path / "a_fish.mp4")
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "setup.toml")
    _touch(tmp_path / "session_a_fish" / "inner.avi")

    assert [p.name for p in tracking.find_videos(tmp_path)] == [
        "a_fish.mp4", "b_fish.AVI"]


def test_find_setups_lists_toml_files(tmp_path):
    _touch(tmp_path / "ir_rig.toml")
    _touch(tmp_path / "video.avi")
    _touch(tmp_path / "session_video" / "other.toml")

    assert [p.name for p in tracking.find_setups(tmp_path)] == ["ir_rig.toml"]


def test_status_follows_the_session_folder(tmp_path):
    video = _touch(tmp_path / "control 1.avi")
    assert tracking.session_folder_for(video) == tmp_path / "session_control 1"
    assert tracking.tracking_status(video) == tracking.NOT_TRACKED

    (tmp_path / "session_control 1").mkdir()
    assert tracking.tracking_status(video) == tracking.INCOMPLETE

    _touch(tmp_path / "session_control 1" / "trajectories" / "trajectories.npy")
    assert tracking.tracking_status(video) == tracking.TRACKED


def test_track_command_overrides_the_video_and_name_in_the_setup(tmp_path):
    video, setup = tmp_path / "exp 1.avi", tmp_path / "rig.toml"
    command = tracking.build_track_command(video, setup)

    assert command[1:3] == ["-m", "idtrackerai.start"]
    assert command[command.index("--load") + 1] == str(setup)
    assert command[command.index("--video_paths") + 1] == str(video)
    assert command[command.index("--name") + 1] == "exp 1"
    assert command[-1] == "--track"


def test_configure_command_opens_the_window_instead_of_tracking(tmp_path):
    video, setup = tmp_path / "exp.avi", tmp_path / "rig.toml"

    fresh = tracking.build_configure_command(video)
    assert "--track" not in fresh and "--load" not in fresh
    assert fresh[fresh.index("--video_paths") + 1] == str(video)

    editing = tracking.build_configure_command(video, setup)
    assert editing[editing.index("--load") + 1] == str(setup)
    assert "--track" not in editing


def test_pythonw_is_swapped_for_python(tmp_path, monkeypatch):
    """Under the desktop shortcut sys.executable is pythonw.exe, which has no
    stdout for idtracker.ai to log to."""
    _touch(tmp_path / "python.exe")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "pythonw.exe"))
    assert tracking.python_executable() == str(tmp_path / "python.exe")


def test_run_process_reports_lines_and_exit_code(tmp_path):
    lines = []
    code = tracking.run_process(
        [sys.executable, "-c",
         "import os, sys; print('one'); print(os.getcwd()); sys.exit(3)"],
        cwd=tmp_path, on_line=lines.append)

    assert code == 3
    assert lines[0] == "one"
    assert Path(lines[1]).resolve() == tmp_path.resolve()


def test_run_process_can_be_stopped(tmp_path):
    lines = []
    started = time.monotonic()
    code = tracking.run_process(
        [sys.executable, "-c",
         "import time; print('started', flush=True); time.sleep(60)"],
        cwd=tmp_path, on_line=lines.append,
        should_stop=lambda: "started" in lines)

    assert time.monotonic() - started < 20
    assert code != 0
