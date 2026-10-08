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


def test_configure_command_opens_the_setup_window_not_tracking(tmp_path):
    video, setup = tmp_path / "exp.avi", tmp_path / "rig.toml"

    fresh = tracking.build_configure_command(video, save_to=setup)
    assert fresh[1] == str(tracking.SETUP_WINDOW_SCRIPT)
    assert tracking.SETUP_WINDOW_SCRIPT.is_file()
    assert fresh[fresh.index("--video") + 1] == str(video)
    assert fresh[fresh.index("--save-to") + 1] == str(setup)
    assert "--track" not in fresh and "--load" not in fresh

    checking = tracking.build_configure_command(video, save_to=setup, load=setup)
    assert checking[checking.index("--load") + 1] == str(setup)


def test_setup_names_become_toml_paths(tmp_path):
    assert tracking.setup_path_for(tmp_path, " IR rig ") == tmp_path / "IR rig.toml"
    assert tracking.setup_path_for(tmp_path, "rig.toml") == tmp_path / "rig.toml"
    assert tracking.setup_path_for(tmp_path, "") is None
    assert tracking.setup_path_for(tmp_path, "a/b") is None
    assert tracking.setup_path_for(tmp_path, "what?") is None


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


def _pretend_idtrackerai(monkeypatch, script: str):
    """Make build_track_command run a Python one-liner instead. The script
    gets the video path as sys.argv[1]."""
    monkeypatch.setattr(
        tracking, "build_track_command",
        lambda video, setup: [sys.executable, "-c", script, str(video)])


WRITES_TRAJECTORIES = (
    "import sys, pathlib; v = pathlib.Path(sys.argv[1]);"
    "d = v.parent / ('session_' + v.stem) / 'trajectories';"
    "d.mkdir(parents=True); (d / 'trajectories.npy').write_bytes(b'');"
    "print('Success')")


def test_run_tracking_succeeds_when_trajectories_appear(tmp_path, monkeypatch):
    video = _touch(tmp_path / "exp.avi")
    _pretend_idtrackerai(monkeypatch, WRITES_TRAJECTORIES)

    outcome = tracking.run_tracking(video, tmp_path / "rig.toml", lambda _: None)

    assert outcome.ok and not outcome.stopped
    assert outcome.video == video


def test_exit_code_zero_without_trajectories_is_a_failure(tmp_path, monkeypatch):
    """idtracker.ai exits 0 even when tracking fails, so the exit code is not
    evidence of anything."""
    video = _touch(tmp_path / "exp.avi")
    _pretend_idtrackerai(
        monkeypatch,
        "print('Loading video'); "
        "print('09:15:02 IdtrackeraiError: too many blobs      run.py:80'); "
        "print('         Log file copied to                    run.py:106'); print('')")

    outcome = tracking.run_tracking(video, tmp_path / "rig.toml", lambda _: None)

    assert not outcome.ok and not outcome.stopped
    assert outcome.reason() == "IdtrackeraiError: too many blobs"


def test_a_stopped_run_is_not_a_success_even_if_files_exist(tmp_path, monkeypatch):
    video = _touch(tmp_path / "exp.avi")
    _pretend_idtrackerai(
        monkeypatch,
        WRITES_TRAJECTORIES + "; import time; print('x', flush=True); time.sleep(60)")
    lines = []

    outcome = tracking.run_tracking(video, tmp_path / "rig.toml", lines.append,
                                    should_stop=lambda: "x" in lines)

    assert outcome.stopped and not outcome.ok


def test_review_command_opens_the_validator_on_the_session(tmp_path):
    command = tracking.build_review_command(tmp_path / "session_exp 1")

    assert command[1] == "-c" and "validator" in command[2]
    assert command[-1] == str(tmp_path / "session_exp 1")


def test_reviewed_on_reads_the_date_the_validator_saved(tmp_path):
    video = _touch(tmp_path / "exp.avi")
    assert tracking.reviewed_on(video) is None, "no session at all"

    session = tmp_path / "session_exp"
    session.mkdir()
    (session / "session.json").write_text('{"last_validated": null}')
    assert tracking.reviewed_on(video) is None

    (session / "session.json").write_text(
        '{"last_validated": "2026-10-08T11:02:33.123456"}')
    assert tracking.reviewed_on(video) == "2026-10-08"
