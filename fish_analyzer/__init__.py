"""
fish_analyzer - A modular tool for analyzing fish trajectory data from idtracker.ai

This package provides tools for:
- Loading and calibrating trajectory data
- Individual behavior metrics (speed, distance, freezing, bursting, path straightness)
- Bout analysis (swim bout detection, laterality, per-bout metrics)
- Shoaling analysis (NND, IID, convex hull)
- Spatial analysis (thigmotaxis, heatmaps)

Usage:
    # Run the GUI application
    from fish_analyzer import EnhancedFishAnalyzer
    app = EnhancedFishAnalyzer()
    app.run()

    # Or use individual components programmatically
    from fish_analyzer import TrajectoryFileLoader, process_and_analyze_file
    
    loaded_file = TrajectoryFileLoader.load_file(Path("my_data.npy"))
    results = process_and_analyze_file(loaded_file)
"""

# Make key classes available at package level for convenient imports
# Users can do: from fish_analyzer import EnhancedFishAnalyzer
# Instead of: from fish_analyzer.gui import EnhancedFishAnalyzer

from .data_structures import (
    IdTrackerMetadata,
    CalibrationSettings,
    LoadedTrajectoryFile,
)

from .file_loading import TrajectoryFileLoader

from .processing import (
    ProcessingParameters,
    FishTrajectory,
    TrajectoryProcessor,
    MetricsCalculator,
    process_and_analyze_file,
)

from .shoaling import (
    ShoalingParameters,
    ShoalingResults,
    ShoalingCalculator,
)

from .bout_analysis import (
    BoutParameters,
    BoutResults,
    BoutDetector,
    analyze_bouts_for_file,
)

from .spatial import (
    ArenaDefinition,
    ThigmotaxisResults,
    ThigmotaxisCalculator,
    HeatmapGenerator,
    SHAPELY_AVAILABLE,
)

# Import GUI from the modular subpackage
# GUI requires tkinter which may not be available in all environments
try:
    from .gui import EnhancedFishAnalyzer
    GUI_AVAILABLE = True
except ImportError as e:
    EnhancedFishAnalyzer = None
    GUI_AVAILABLE = False
    import warnings
    warnings.warn(f"GUI not available: {e}. Install tkinter to use the GUI.")

from .export import (
    export_individual_metrics_csv,
    export_shoaling_metrics_csv,
    export_shoaling_summary_csv,
    export_thigmotaxis_csv,
)

from .video_utils import (
    VideoFrameReader,
    VideoFrameCache,
    CV2_AVAILABLE,
)

# Package metadata
__version__ = "2.1.0"
__author__ = "Fish Trajectory Analyzer"
