"""
fish_analyzer/gui/__init__.py
=============================
GUI Package - Modular graphical user interface for fish trajectory analysis.

This package splits the GUI into logical components:
- base.py: Core initialization and shared state
- data_tab.py: File loading and calibration
- analysis_tab.py: Individual trajectory analysis
- bout_tab.py: Bout detection and laterality analysis
- shoaling_tab.py: Group behavior analysis
- spatial_tab.py: Thigmotaxis and heatmap analysis
- inspector_tab.py: Unified video inspector with overlay controls
- utils.py: Shared helper functions

The EnhancedFishAnalyzer class combines all mixins to provide the complete GUI.
"""

from .base import GUIBase
from .data_tab import DataTabMixin
from .analysis_tab import AnalysisTabMixin
from .bout_tab import BoutTabMixin
from .shoaling_tab import ShoalingTabMixin
from .spatial_tab import SpatialTabMixin
from .inspector_tab import InspectorTabMixin
from .utils import smooth_time_series


class EnhancedFishAnalyzer(GUIBase, DataTabMixin, AnalysisTabMixin,
                           BoutTabMixin, ShoalingTabMixin, SpatialTabMixin,
                           InspectorTabMixin):
    """
    Main application class - orchestrates the entire GUI.

    This class combines:
    - GUIBase: Window creation, shared state management
    - DataTabMixin: File loading, calibration, processing parameters
    - AnalysisTabMixin: Individual trajectory visualization
    - BoutTabMixin: Swim bout detection and laterality analysis
    - ShoalingTabMixin: Group behavior analysis
    - SpatialTabMixin: Thigmotaxis, heatmaps
    - InspectorTabMixin: Unified video inspector with overlays

    Usage:
        app = EnhancedFishAnalyzer()
        app.run()
    """
    pass


# Export the main class and utility function
__all__ = ['EnhancedFishAnalyzer', 'smooth_time_series']
