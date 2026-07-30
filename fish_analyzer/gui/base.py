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
import traceback
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')  # MUST be before importing pyplot

# Import from our package
from ..data_structures import LoadedTrajectoryFile, CalibrationSettings
from ..processing import ProcessingParameters
from ..shoaling import ShoalingParameters
from .utils import set_figure_error_handler


class GUILogRedirector(io.TextIOBase):
    """
    Redirects a stream to a GUI status bar so users see messages that would
    otherwise only appear in a terminal console.

    Both stdout and stderr are wrapped, sharing one `log_lines` buffer, so that
    tracebacks — which Python writes to stderr, and which Tkinter and
    matplotlib produce for swallowed callback exceptions — end up somewhere the
    user can actually read. Before this, stderr was untouched and every such
    traceback went to a console that a double-click launch does not have.

    Also keeps a copy going to the real stream for debugging.
    """

    MAX_LINES = 500

    def __init__(self, status_label: tk.Label, original_stream,
                 log_lines: List[str], prefix: str = ""):
        self.status_label = status_label
        self.original_stream = original_stream
        self.prefix = prefix
        self._log_lines = log_lines   # shared across stdout/stderr redirectors

    def write(self, text: str):
        # Always write to the original stream too (for debugging in terminals)
        if self.original_stream:
            try:
                self.original_stream.write(text)
            except Exception:
                pass  # e.g. pythonw with no console attached

        # Skip empty/whitespace-only writes
        stripped = text.strip()
        if stripped:
            self._log_lines.append(f"{self.prefix}{stripped}")
            if len(self._log_lines) > self.MAX_LINES:
                del self._log_lines[:-self.MAX_LINES]
            # Update status bar with most recent message
            try:
                self.status_label.config(text=f"  {self.prefix}{stripped}")
            except tk.TclError:
                pass  # Widget may have been destroyed
        return len(text)

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass

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

        # Set up log redirection (print/stderr → status bar + log history)
        # before building the GUI, so failures during construction are caught.
        self._setup_gui()
        self._setup_log_redirect()

        # Route otherwise-swallowed Tk and matplotlib callback exceptions
        self._setup_error_reporting()

    def _setup_gui(self):
        """Create the main window structure with tabbed interface."""
        # Main content area
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # Status bar at bottom — shows print() messages and progress info
        status_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill="x", side="bottom", padx=10, pady=(0, 5))

        # "Show Log" gives the 500-line history a way out. Until this existed,
        # get_log() was defined and never called from anywhere, so every
        # message but the most recent one was unreachable.
        tk.Button(
            status_frame, text="Show Log", command=self._show_log_window,
            font=("Arial", 8), relief="flat", padx=6
        ).pack(side="right", padx=(0, 4), pady=1)

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
        """Redirect print() and stderr output to the GUI status bar and log."""
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._log_lines: List[str] = []

        self._log_redirector = GUILogRedirector(
            self.status_label, self._original_stdout, self._log_lines
        )
        self._err_redirector = GUILogRedirector(
            self.status_label, self._original_stderr, self._log_lines,
            prefix="[error] "
        )
        sys.stdout = self._log_redirector
        sys.stderr = self._err_redirector

    def get_log(self) -> str:
        """Return the combined stdout/stderr history as a string."""
        return "\n".join(self._log_lines)

    def _show_log_window(self):
        """Open a scrollable window showing the message history."""
        win = tk.Toplevel(self.root)
        win.title("Message Log")
        win.geometry("900x500")

        scroll = tk.Scrollbar(win)
        scroll.pack(side="right", fill="y")
        text = tk.Text(win, wrap="none", font=("Courier", 9),
                       yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.config(command=text.yview)

        log = self.get_log() or "(no messages yet)"
        text.insert("1.0", log)
        text.see("end")
        text.config(state="disabled")

    # =========================================================================
    # ERROR REPORTING
    # =========================================================================

    def _setup_error_reporting(self):
        """Route exceptions Tk and matplotlib would otherwise discard.

        Tkinter sends uncaught callback exceptions to
        Tk.report_callback_exception, and matplotlib sends uncaught canvas
        event-handler exceptions to CallbackRegistry.exception_handler. Both
        defaults print to stderr and continue, so a failed button click or a
        failed arena click looked exactly like a button that did nothing.
        """
        self._reporting_error = False
        self.root.report_callback_exception = self._report_uncaught
        set_figure_error_handler(self._report_uncaught)

    def _report_uncaught(self, exc, val=None, tb=None):
        """Show, and log, an exception that would otherwise be swallowed.

        Accepts both call signatures in play: Tk passes (type, value, tb);
        matplotlib passes a single exception instance.
        """
        if val is None and isinstance(exc, BaseException):
            exc, val, tb = type(exc), exc, exc.__traceback__

        # A failure inside the reporter itself must not recurse.
        if self._reporting_error:
            return
        self._reporting_error = True
        try:
            detail = "".join(traceback.format_exception(exc, val, tb))
            # Goes through the stderr redirector → status bar + log history.
            sys.stderr.write(detail)
            messagebox.showerror(
                "Unexpected Error",
                f"{getattr(exc, '__name__', exc)}: {val}\n\n"
                f"The action did not complete. Nothing was saved.\n"
                f"Use 'Show Log' at the bottom of the window for the "
                f"full traceback."
            )
        finally:
            self._reporting_error = False

    # =========================================================================
    # BATCH PROGRESS
    # =========================================================================

    def _with_progress(self, items, label="Processing", button=None,
                       progressbar=None):
        """Iterate `items`, reporting progress and locking out re-entry.

        Replaces the ad-hoc `self.root.update()` calls that used to sit inside
        the processing loops. `update()` dispatches the *full* event queue,
        including user input, so a second click on the button that started the
        run re-entered the handler while the first was still iterating —
        two runs then wrote results for the same files, and a click on
        'Remove' could mutate loaded_files mid-iteration.

        `update_idletasks()` repaints without dispatching input, and the
        triggering button is disabled for the duration, which removes that
        whole class of hazard without introducing threads.

        Parameters
        ----------
        items : iterable
            Work items; str items are named in the status message.
        label : str
            Verb shown in the status bar, e.g. "Processing".
        button : tk.Button, optional
            Disabled while the loop runs, restored afterwards.
        progressbar : ttk.Progressbar, optional
            Shown and advanced per item, hidden afterwards.
        """
        items = list(items)
        total = len(items)

        prev_state = None
        if button is not None:
            prev_state = str(button.cget('state'))
            button.config(state=tk.DISABLED)

        if progressbar is not None:
            progressbar.pack(pady=(5, 0))
            progressbar['maximum'] = max(total, 1)
            progressbar['value'] = 0

        try:
            for idx, item in enumerate(items, 1):
                named = f": {item}" if isinstance(item, str) else ""
                self.set_status(f"{label} {idx}/{total}{named}...")
                yield item
                if progressbar is not None:
                    progressbar['value'] = idx
                self.root.update_idletasks()
        finally:
            if progressbar is not None:
                progressbar.pack_forget()
            if button is not None:
                button.config(state=prev_state or tk.NORMAL)

    @staticmethod
    def _format_outcome_list(header: str, entries: List[str], limit: int = 8) -> str:
        """Render a capped bullet list for the batch-outcome dialog."""
        shown = "\n".join(f"  - {e}" for e in entries[:limit])
        if len(entries) > limit:
            shown += f"\n  ... and {len(entries) - limit} more"
        return f"\n{header}\n{shown}"

    def _report_batch_outcome(self, what: str, total: int,
                              succeeded: List[str], failed: List[str],
                              degraded: List[str]):
        """Report a batch run's outcome, naming every file that did not work.

        Silent partial failure was the main way a degraded result reached the
        user: per-file errors were printed to a status bar that the next
        message immediately overwrote, and the completion dialog reported only
        a count. A file that produced no results now says so, by name.
        """
        text = f"{len(succeeded)} of {total} file(s) completed."
        if failed:
            text += self._format_outcome_list("FAILED - no results produced:", failed)
        if degraded:
            text += self._format_outcome_list("Completed, with problems:", degraded)

        if failed:
            messagebox.showerror(
                f"{what}: {len(failed)} file(s) failed",
                text + "\n\nUse 'Show Log' at the bottom of the window for "
                       "the full tracebacks."
            )
        elif degraded:
            messagebox.showwarning(f"{what} complete, with warnings", text)
        else:
            messagebox.showinfo(f"{what} complete", text)

    def set_status(self, message: str):
        """Update the status bar message directly (for GUI code)."""
        self.status_label.config(text=f"  {message}")
        self.root.update_idletasks()

    def run(self):
        """Start the application main loop."""
        try:
            self.root.mainloop()
        finally:
            # Restore the real streams when the GUI closes
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
