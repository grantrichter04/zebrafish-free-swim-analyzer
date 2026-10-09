"""
fish_analyzer/idtrackerai_track.py
==================================
Track one video with idtracker.ai, with two additions: its identity training
can be told to finish early, and its background can be a rolling one.

Run as a script, by path, in a separate process (see tracking.py):

    python idtrackerai_track.py --finish-flag FILE [--rolling-background STAT]
                                <idtracker.ai's arguments>

idtracker.ai learns to tell the fish apart until a score reaches a target, and
stops early on Ctrl+C, keeping the best result so far. The Tracking tab runs it
without a console, so there is no Ctrl+C to press. Instead the tab creates
FILE, and the next time idtracker.ai checks its score it is given the same
KeyboardInterrupt Ctrl+C would raise. That check is inside the training loop
and nowhere else, so the request cannot interrupt any other step.

The request is a file, not a line on standard input: on Windows a thread
waiting to read standard input makes idtracker.ai's worker processes hang as
they start.

With --rolling-background each stretch of video is segmented against a
background made from the minute around it (see idtrackerai_rolling.py) in
place of idtracker.ai's one still background. idtracker.ai segments in worker
processes, which start by importing this file again, so the replacement is
made at import time and the choice travels in an environment variable.

This reaches into idtracker.ai's training class and its segmentation, which
are not public interfaces. If a future version renames what the early finish
relies on, tracking runs unchanged and says so. If the rolling background
cannot be put in place, tracking stops: carrying on would quietly track with a
different background from the one the setup was checked with.

Do not import this from the package: it is run by path so that it does not
load fish_analyzer (and with it several seconds of imports).
"""
import argparse
import functools
import os
import sys
from pathlib import Path

ROLLING_VARIABLE = "FREESWIM_ROLLING_BACKGROUND"


def _allow_finishing_training(flag: Path) -> None:
    from idtrackerai.base.tracker.contrastive import ContrastiveLearning

    original_validate = ContrastiveLearning.validate

    def validate(self):
        if flag.exists():
            flag.unlink(missing_ok=True)
            raise KeyboardInterrupt
        return original_validate(self)

    ContrastiveLearning.validate = validate


def _use_rolling_background(stat: str) -> None:
    import idtrackerai.base.animals_detection.segmentation as segmentation
    import idtrackerai_rolling as rolling

    original_segment_episode = segmentation.segment_episode

    # wraps() keeps the name idtracker.ai's worker processes look it up by.
    @functools.wraps(original_segment_episode)
    def segment_episode(inputs):
        episode, parameters = inputs
        if parameters.get("bkg_model") is not None:
            parameters = dict(parameters, bkg_model=rolling.background_for(
                episode.video_path, episode.local_start, episode.local_end, stat))
        return original_segment_episode((episode, parameters))

    segmentation.segment_episode = segment_episode


if __name__ == "__mp_main__" and os.environ.get(ROLLING_VARIABLE):
    # One of idtracker.ai's worker processes, started for a rolling setup.
    _use_rolling_background(os.environ[ROLLING_VARIABLE])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finish-flag", required=True, type=Path)
    parser.add_argument("--rolling-background", metavar="STAT")
    args, idtrackerai_arguments = parser.parse_known_args()

    if args.rolling_background:
        os.environ[ROLLING_VARIABLE] = args.rolling_background
        _use_rolling_background(args.rolling_background)
        print("Rolling background: each stretch of video is segmented against "
              f"the {args.rolling_background} of the minute around it.",
              flush=True)

    try:
        _allow_finishing_training(args.finish_flag)
    except Exception as exc:
        print("Could not adapt idtracker.ai's training to this version "
              f"({exc!r}). Tracking as usual; identity training cannot be "
              "finished early.", flush=True)

    sys.argv = ["idtrackerai", *idtrackerai_arguments]
    from idtrackerai.start.__main__ import main as idtrackerai_main
    idtrackerai_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
