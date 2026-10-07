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
