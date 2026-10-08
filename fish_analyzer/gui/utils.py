"""
fish_analyzer/gui/utils.py
==========================
Shared utility functions for the GUI components.
"""

from pathlib import Path

from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


#: Handler installed on every embedded canvas, set once by GUIBase.
#: matplotlib binds its default exception handler as a function default
#: argument, so it cannot be replaced by patching the module attribute — it
#: has to be set per CallbackRegistry instance, i.e. per canvas.
_FIGURE_ERROR_HANDLER = None


def set_figure_error_handler(handler):
    """Register the handler for exceptions raised inside canvas event callbacks."""
    global _FIGURE_ERROR_HANDLER
    _FIGURE_ERROR_HANDLER = handler


def install_canvas_error_handler(canvas):
    """Route a canvas's callback exceptions to the registered handler.

    Without this, matplotlib prints the traceback to stderr and continues, so
    an exception inside a click handler is invisible to the user.
    """
    if _FIGURE_ERROR_HANDLER is not None:
        canvas.callbacks.exception_handler = _FIGURE_ERROR_HANDLER
    return canvas


def embed_figure_with_toolbar(fig, parent):
    """
    Embed a matplotlib figure with interactive navigation toolbar.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to embed
    parent : tk.Widget
        Parent widget

    Returns
    -------
    FigureCanvasTkAgg
        The canvas widget
    """
    canvas = FigureCanvasTkAgg(fig, master=parent)
    install_canvas_error_handler(canvas)
    toolbar = NavigationToolbar2Tk(canvas, parent)
    toolbar.update()
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas


def ask_csv_save_path(title: str, initial_file: str):
    """The Save-as dialog every CSV export uses.

    Five export handlers repeated these six lines verbatim, differing only in
    the title and the suggested filename.

    Returns
    -------
    Path or None
        None when the user cancels, which callers treat as "do nothing".
    """
    path = filedialog.asksaveasfilename(
        title=title,
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=initial_file,
    )
    return Path(path) if path else None
