"""
fish_analyzer/idtrackerai_rolling.py
====================================
A rolling background for idtracker.ai's background subtraction.

idtracker.ai subtracts one still background, made from frames spread over the
whole video. When the lighting drifts during a recording - sunlight creeping
across the tank - no single image fits, and the edge of the lit patch is
detected as extra animals. Here each stretch of video gets its own background,
made from the minute around it:

    the background for a stretch of frames is the median (or the setup's
    chosen statistic) of 31 frames spread evenly over the 60 seconds centred
    on that stretch.

An animal that stays in one spot for more than half of that minute becomes
part of the background and is not detected while it sits there.

A setup asks for it by prefixing its statistic: background_subtraction_stat =
"rolling median". idtracker.ai itself does not know the prefix and would take
it for a file name, so a rolling setup fails loudly if run without these
scripts rather than quietly using a still background.

Imported by idtrackerai_setup_window.py and idtrackerai_track.py, which are
run by path; like them it must not import fish_analyzer.
"""
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

PREFIX = "rolling "
WINDOW_SECONDS = 60.0
SAMPLES = 31
STATISTICS = {"median": np.median, "mean": np.mean, "max": np.max, "min": np.min}


def split_statistic(stat) -> Tuple[bool, str]:
    """(is it rolling, the plain statistic) for a setup's
    background_subtraction_stat."""
    text = str(stat or "")
    if text.lower().startswith(PREFIX) and text[len(PREFIX):].lower() in STATISTICS:
        return True, text[len(PREFIX):].lower()
    return False, text


def setup_statistic(setup: Path) -> Tuple[bool, str]:
    """split_statistic for a setup file; not rolling if it cannot be read or
    has background subtraction switched off."""
    import tomllib
    try:
        with open(setup, "rb") as file:
            parameters = tomllib.load(file)
    except (OSError, ValueError):
        return False, ""
    rolling, stat = split_statistic(parameters.get("background_subtraction_stat"))
    return rolling and bool(parameters.get("use_bkg")), stat


def sample_frames(first_frame: int, last_frame: int, fps: float,
                  frames_in_video: int) -> np.ndarray:
    """Which frames make the background for frames [first_frame, last_frame).

    The minute is centred on the stretch, and slid inwards at either end of
    the video so that it is always a full minute where the video allows.
    """
    span = min(int(round(WINDOW_SECONDS * fps)), frames_in_video) - 1
    start = int(round((first_frame + last_frame - 1) / 2 - span / 2))
    start = max(0, min(start, frames_in_video - 1 - span))
    return np.linspace(start, start + span, SAMPLES).round().astype(int)


def background_for(video_path, first_frame: int, last_frame: int,
                   stat: str = "median") -> np.ndarray:
    """The background for frames [first_frame, last_frame) of one video file."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stack = []
        for index in sample_frames(first_frame, last_frame, fps, frames_in_video):
            if index != int(cap.get(cv2.CAP_PROP_POS_FRAMES)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                stack.append(frame if frame.ndim == 2
                             else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    if not stack:
        raise OSError(f"No frames could be read from {video_path} around "
                      f"frame {first_frame} for its rolling background")
    return STATISTICS[stat](np.stack(stack), axis=0).astype(np.uint8)
