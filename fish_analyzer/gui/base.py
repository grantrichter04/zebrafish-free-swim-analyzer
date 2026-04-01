"""
fish_analyzer/gui/base.py
=========================
Base class containing initialization, shared state, and core window setup.

This provides the foundation that all tab mixins build upon.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
import io
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')  # MUST be before importing pyplot

# Import from our package
from ..data_structures import LoadedTrajectoryFile, CalibrationSettings
from ..processing import ProcessingParameters
from ..shoaling import ShoalingParameters


class GUILogRedirector(io.TextIOBase):
    """
    Redirects print() output to a GUI status bar so users see messages
    that would otherwise only appear in a terminal console.

    Also keeps a copy going to the real stderr for debugging.
    """

    def __init__(self, status_label: tk.Label, original_stdout):
        self.status_label = status_label
        self.original_stdout = original_stdout
        self._log_lines: List[str] = []

    def write(self, text: str):
        # Always write to original stdout too (for debugging in terminals)
        if self.original_stdout:
            self.original_stdout.write(text)

        # Skip empty/whitespace-only writes
        stripped = text.strip()
        if stripped:
            self._log_lines.append(stripped)
            # Keep only last 100 lines
            if len(self._log_lines) > 100:
                self._log_lines = self._log_lines[-100:]
            # Update status bar with most recent message
            try:
                self.status_label.config(text=f"  {stripped}")
            except tk.TclError:
                pass  # Widget may have been destroyed
        return len(text)

    def flush(self):
        if self.original_stdout:
            self.original_stdout.flush()

    def get_log(self) -> str:
        """Return the full log history as a string."""
        return "\n".join(self._log_lines)


class GUIBase:
    """
    Base class containing shared state and initialization for the GUI.

    This class manages:
    - The main tkinter window and notebook
    - Loaded files dictionary
    - Default parameters
    - Animation and video reader state
    - Arena definition state
    - Status bar for user feedback

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

        # Group assignments: file nickname → group label (for collapsed distributions)
        self.file_groups: Dict[str, str] = {}

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Fish Trajectory Analyzer")

        # Size window to fit screen (cap at 1200x850, shrink for small displays)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1200, int(screen_w * 0.9))
        win_h = min(850, int(screen_h * 0.85))
        self.root.geometry(f"{win_w}x{win_h}")

        # Build the GUI
        self._setup_gui()

        # Set up log redirection (print → status bar)
        self._setup_log_redirect()

    def _setup_gui(self):
        """Create the main window structure with tabbed interface."""
        # Main content area
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # Status bar at bottom — shows print() messages and progress info
        status_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill="x", side="bottom", padx=10, pady=(0, 5))

        self.status_label = tk.Label(
            status_frame, text="  Ready", anchor="w",
            font=("Arial", 9), fg="gray40"
        )
        self.status_label.pack(fill="x", padx=5, pady=2)

        # Create each tab (methods provided by mixins)
        self._create_data_tab()
        self._create_analysis_tab()
        self._create_bout_tab()
        self._create_shoaling_tab()
        self._create_spatial_tab()
        self._create_inspector_tab()

    def _setup_log_redirect(self):
        """Redirect print() output to the GUI status bar."""
        self._original_stdout = sys.stdout
        self._log_redirector = GUILogRedirector(self.status_label, self._original_stdout)
        sys.stdout = self._log_redirector

    def set_status(self, message: str):
        """Update the status bar message directly (for GUI code)."""
        self.status_label.config(text=f"  {message}")
        self.root.update_idletasks()

    def run(self):
        """Start the application main loop."""
        try:
            self.root.mainloop()
        finally:
            # Restore stdout when GUI closes
            sys.stdout = self._original_stdout
