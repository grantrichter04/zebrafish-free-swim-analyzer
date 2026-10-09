"""
fish_analyzer/idtrackerai_track.py
==================================
Track one video with idtracker.ai, with one addition: its identity training
can be told to finish early.

Run as a script, by path, in a separate process (see tracking.py):

    python idtrackerai_track.py --finish-flag FILE <idtracker.ai's arguments>

idtracker.ai learns to tell the fish apart until a score reaches a target, and
stops early on Ctrl+C, keeping the best result so far. The Tracking tab runs it
without a console, so there is no Ctrl+C to press. Instead the tab creates
FILE, and the next time idtracker.ai checks its score it is given the same
KeyboardInterrupt Ctrl+C would raise. That check is inside the training loop
and nowhere else, so the request cannot interrupt any other step.

The request is a file, not a line on standard input: on Windows a thread
waiting to read standard input makes idtracker.ai's worker processes hang as
they start.

This reaches into idtracker.ai's training class, which is not a public
interface. If a future version renames what it relies on, tracking runs
unchanged and says so.

Do not import this from the package: it is run by path so that it does not
load fish_analyzer (and with it several seconds of imports).
"""
import argparse
import sys
from pathlib import Path


def _allow_finishing_training(flag: Path) -> None:
    from idtrackerai.base.tracker.contrastive import ContrastiveLearning

    original_validate = ContrastiveLearning.validate

    def validate(self):
        if flag.exists():
            flag.unlink(missing_ok=True)
            raise KeyboardInterrupt
        return original_validate(self)

    ContrastiveLearning.validate = validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finish-flag", required=True, type=Path)
    args, idtrackerai_arguments = parser.parse_known_args()

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
