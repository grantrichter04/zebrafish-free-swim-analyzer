"""
fish_analyzer/gui/__init__.py
=============================
GUI Package - Modular graphical user interface for fish trajectory analysis.

This package splits the GUI into logical components:
- base.py: Core initialization and shared state
- tracking_tab.py: Running idtracker.ai on a folder of videos
- data_tab.py: File loading and calibration
- results_tab.py: The headline metrics, speed distributions and swim paths
- shoaling_tab.py: Group behavior analysis
- inspector_tab.py: Unified video inspector with overlay controls
- utils.py: Shared helper functions

The EnhancedFishAnalyzer class combines all mixins to provide the complete GUI.
"""

from .base import GUIBase
from .tracking_tab import TrackingTabMixin
from .data_tab import DataTabMixin
from .results_tab import ResultsTabMixin
from .shoaling_tab import ShoalingTabMixin
from .inspector_tab import InspectorTabMixin
from .inspector_export import InspectorExportMixin


class EnhancedFishAnalyzer(GUIBase, TrackingTabMixin, DataTabMixin,
                           ResultsTabMixin,
                           ShoalingTabMixin,
                           InspectorTabMixin, InspectorExportMixin):
    """
    Main application class - orchestrates the entire GUI.

    This class combines:
    - GUIBase: Window creation, shared state management
    - TrackingTabMixin: Videos, idtracker.ai setups, tracking
    - DataTabMixin: Sessions, units, running the analysis
    - ResultsTabMixin: The headline metrics as SuperPlots
    - ShoalingTabMixin: Group behavior analysis
    - InspectorTabMixin: Unified video inspector with overlays
    - InspectorExportMixin: The inspector's frame and clip export controls

    Usage:
        app = EnhancedFishAnalyzer()
        app.run()
    """
    pass


__all__ = ['EnhancedFishAnalyzer']
