#!/usr/bin/env python3
"""
run_analyzer.py - Main entry point for the Fish Trajectory Analyzer

HOW TO RUN:
    python run_analyzer.py

This script imports and launches the GUI application from the fish_analyzer package.
The actual code is organized into modules within the fish_analyzer/ folder.

PACKAGE STRUCTURE:
    fish_analyzer/
        __init__.py          - Package initialization and exports
        data_structures.py   - Data classes (metadata, calibration, etc.)
        file_loading.py      - Load .npy files from idtracker.ai
        processing.py        - Trajectory processing and metrics
        shoaling.py          - Group behavior analysis (NND, IID, hull)
        spatial.py           - Thigmotaxis and heatmaps
        gui.py               - The graphical user interface

DEPENDENCIES:
    pip install numpy pandas matplotlib traja scipy
    pip install shapely  # Optional, for thigmotaxis

ALTERNATIVE: Using components programmatically (without the GUI):

    from pathlib import Path
    from fish_analyzer import (
        TrajectoryFileLoader,
        process_and_analyze_file,
        ShoalingCalculator,
        ShoalingParameters
    )
    
    # Load a file
    loaded_file = TrajectoryFileLoader.load_file(Path("my_trajectory.npy"))
    
    # Run individual analysis
    fish_list = process_and_analyze_file(loaded_file)
    for fish in fish_list:
        print(f"Fish {fish.fish_id}: {fish.metrics['total_distance']:.1f} BL traveled")
    
    # Run shoaling analysis
    params = ShoalingParameters(sample_interval_frames=30)
    calculator = ShoalingCalculator(loaded_file, params)
    results = calculator.calculate()
    print(f"Mean NND: {results.mean_nnd:.2f} BL")
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import fish_analyzer
# This is only needed if running from a different directory
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Now import and run the application
from fish_analyzer import EnhancedFishAnalyzer


def main():
    """Launch the Fish Trajectory Analyzer GUI."""
    print("=" * 60)
    print("Fish Trajectory Analyzer v2.0")
    print("=" * 60)
    print()
    print("Starting GUI application...")
    print("(Close the window to exit)")
    print()
    
    # Create and run the application
    app = EnhancedFishAnalyzer()
    app.run()
    
    print("Application closed.")


if __name__ == "__main__":
    main()
