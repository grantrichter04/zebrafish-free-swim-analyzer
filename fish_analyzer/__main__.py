"""
fish_analyzer/__main__.py
=========================
Entry point for `python -m fish_analyzer` and the `fish-analyzer` command.

run_analyzer.py at the repo root does the same thing and still works from a
plain checkout, without installing anything. This module is what an installed
copy uses, and unlike the script it does not need the working directory to be
the repository.
"""


def main():
    """Launch the Fish Trajectory Analyzer GUI."""
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


if __name__ == "__main__":
    main()
