"""
fish_analyzer/gui/base.py
=========================
Base class containing initialization, shared state, and core window setup.

This provides the foundation that all tab mixins build upon.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')  # MUST be before importing pyplot

# Import from our package
from ..data_structures import LoadedTrajectoryFile, CalibrationSettings
from ..processing import ProcessingParameters
from ..shoaling import ShoalingParameters


class GUIBase:
    """
    Base class containing shared state and initialization for the GUI.
    
    This class manages:
    - The main tkinter window and notebook
    - Loaded files dictionary
    - Default parameters
    - Animation and video reader state
    - Arena definition state
    
    Tab-specific methods are provided by mixin classes.
    """

    def __init__(self):
        """Initialize the application and create the GUI."""
        # Data storage
        self.loaded_files: Dict[str, LoadedTrajectoryFile] = {}
        self.active_file: Optional[str] = None
        
        # Default parameters
        self.processing_params = ProcessingParameters.default_for_fish()
        self.shoaling_params = ShoalingParameters()

        # Animation state
        self.animation_running = False
        self.animation_after_id = None

        # Video reader state
        self.video_readers: Dict[str, Any] = {}  # Per-file video readers

        # Arena drawing state
        self.arena_vertices = []
        self.arena_definition = None
        self.arena_fig = None
        self.arena_ax = None
        self.arena_canvas = None
        self._arena_width_bl = None
        self._arena_height_bl = None
        
        # Per-file arena storage
        self.file_arena_definitions: Dict[str, Any] = {}
        self.current_arena_file: Optional[str] = None

        # Frame viewer state
        self.frame_view_fig = None
        self.frame_view_canvas = None
        self.frame_view_ax_main = None
        self.frame_view_ax_time = None
        self.frame_view_time_marker = None

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Fish Trajectory Analyzer")
        self.root.geometry("1200x850")

        # Build the GUI
        self._setup_gui()

    def _setup_gui(self):
        """Create the main window structure with tabbed interface."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Create each tab (methods provided by mixins)
        self._create_data_tab()
        self._create_analysis_tab()
        self._create_shoaling_tab()
        self._create_spatial_tab()

    def run(self):
        """Start the application main loop."""
        self.root.mainloop()
