"""
fish_analyzer/gui/utils.py
==========================
Shared utility functions for the GUI components.
"""

from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def create_sortable_treeview(parent, columns, data, title=None):
    """
    Create a sortable ttk.Treeview table widget.

    Parameters
    ----------
    parent : tk.Widget
        Parent widget to pack into
    columns : list of (col_id, header_text, width)
        Column definitions
    data : list of tuples
        Row data matching column order
    title : str, optional
        Title label above the table

    Returns
    -------
    ttk.Treeview
        The created treeview widget
    """
    frame = tk.Frame(parent)
    frame.pack(fill="both", expand=True, padx=5, pady=5)

    if title:
        tk.Label(frame, text=title, font=("Arial", 11, "bold")).pack(anchor="w", padx=5, pady=(5, 2))

    # Create treeview with scrollbars
    tree_frame = tk.Frame(frame)
    tree_frame.pack(fill="both", expand=True)

    y_scroll = ttk.Scrollbar(tree_frame, orient="vertical")
    x_scroll = ttk.Scrollbar(tree_frame, orient="horizontal")

    col_ids = [c[0] for c in columns]
    tree = ttk.Treeview(tree_frame, columns=col_ids, show="headings",
                        yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    y_scroll.config(command=tree.yview)
    x_scroll.config(command=tree.xview)

    y_scroll.pack(side="right", fill="y")
    x_scroll.pack(side="bottom", fill="x")
    tree.pack(side="left", fill="both", expand=True)

    # Configure columns
    for col_id, header, width in columns:
        tree.heading(col_id, text=header,
                     command=lambda c=col_id: _treeview_sort_column(tree, c, False))
        tree.column(col_id, width=width, minwidth=50, anchor="center")

    # Insert data
    for row in data:
        tree.insert("", "end", values=row)

    return tree


def _treeview_sort_column(tree, col, reverse):
    """Sort treeview column when header is clicked."""
    data = [(tree.set(child, col), child) for child in tree.get_children('')]

    # Try numeric sort first
    try:
        data.sort(key=lambda t: float(t[0].replace('%', '').replace(',', '')), reverse=reverse)
    except (ValueError, TypeError):
        data.sort(key=lambda t: t[0], reverse=reverse)

    for index, (val, child) in enumerate(data):
        tree.move(child, '', index)

    tree.heading(col, command=lambda: _treeview_sort_column(tree, col, not reverse))


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
