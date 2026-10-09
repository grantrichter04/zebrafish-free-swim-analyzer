"""The rolling background offered beside idtracker.ai's background statistic.

No idtracker.ai needed: the videos are a few synthetic frames written to a
temp dir.
"""
import importlib.util
import sys

import cv2
import numpy as np
import pytest

from fish_analyzer import tracking

# The module is run-by-path in production (see its docstring), so it is loaded
# by path here too.
_spec = importlib.util.spec_from_file_location(
    "idtrackerai_rolling", tracking.TRACK_SCRIPT.with_name("idtrackerai_rolling.py"))
rolling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rolling)

FPS = 10


@pytest.fixture
def drifting_video(tmp_path):
    """Four minutes at 10 frames a second. The scene is at brightness 100 for
    the first half and 180 for the second: lighting no single still image
    fits."""
    path = tmp_path / "drift.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS,
                             (64, 48), isColor=False)
    for frame in range(240 * FPS):
        writer.write(np.full((48, 64), 100 if frame < 120 * FPS else 180, np.uint8))
    writer.release()
    return path


def test_a_setup_asks_for_rolling_with_a_prefix(tmp_path):
    assert rolling.split_statistic("rolling median") == (True, "median")
    assert rolling.split_statistic("Rolling Max") == (True, "max")
    assert rolling.split_statistic("median") == (False, "median")
    assert rolling.split_statistic("rolling stones.png") == (False, "rolling stones.png")
    assert rolling.split_statistic(None) == (False, "")

    setup = tmp_path / "rig.toml"
    setup.write_text("use_bkg = true\nbackground_subtraction_stat = 'rolling median'\n")
    assert rolling.setup_statistic(setup) == (True, "median")
    setup.write_text("use_bkg = false\nbackground_subtraction_stat = 'rolling median'\n")
    assert rolling.setup_statistic(setup) == (False, "median"), "subtraction is off"
    assert rolling.setup_statistic(tmp_path / "missing.toml") == (False, "")


def test_the_minute_is_centred_and_kept_inside_the_video():
    middle = rolling.sample_frames(3000, 3500, fps=30.0, frames_in_video=18000)
    assert len(middle) == rolling.SAMPLES
    assert (middle[0], middle[-1]) == (2350, 4149), "a minute around frame 3250"

    start = rolling.sample_frames(0, 500, fps=30.0, frames_in_video=18000)
    assert (start[0], start[-1]) == (0, 1799), "slid forward, still a full minute"
    end = rolling.sample_frames(17500, 18000, fps=30.0, frames_in_video=18000)
    assert (end[0], end[-1]) == (16200, 17999)

    short = rolling.sample_frames(0, 100, fps=30.0, frames_in_video=100)
    assert (short[0], short[-1]) == (0, 99), "a video shorter than a minute"


def test_each_stretch_gets_the_background_of_its_own_minute(drifting_video):
    early = rolling.background_for(drifting_video, 0, 100)
    late = rolling.background_for(drifting_video, 2000, 2100)

    assert early.shape == (48, 64) and early.dtype == np.uint8
    assert abs(int(early.mean()) - 100) <= 2
    assert abs(int(late.mean()) - 180) <= 2


def test_a_rolling_setup_is_tracked_with_a_rolling_background(tmp_path):
    """idtracker.ai is handed the plain statistic, which it understands, and
    the launcher is told to roll it."""
    video, flag = tmp_path / "exp.avi", tmp_path / "flag"
    setup = tmp_path / "rig.toml"
    setup.write_text("use_bkg = true\nbackground_subtraction_stat = 'rolling median'\n")

    command = tracking.build_track_command(video, setup, flag)

    assert command[command.index("--rolling-background") + 1] == "median"
    assert command.index("--background_subtraction_stat") > command.index("--load"), \
        "idtracker.ai applies arguments in order; the plain statistic must win"
    assert command[command.index("--background_subtraction_stat") + 1] == "median"

    setup.write_text("use_bkg = true\nbackground_subtraction_stat = 'median'\n")
    plain = tracking.build_track_command(video, setup, flag)
    assert "--rolling-background" not in plain
    assert "--background_subtraction_stat" not in plain
