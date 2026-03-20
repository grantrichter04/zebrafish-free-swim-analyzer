"""
fish_analyzer/gui/utils.py
==========================
Shared utility functions for the GUI components.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.ndimage import uniform_filter1d
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
    toolbar = NavigationToolbar2Tk(canvas, parent)
    toolbar.update()
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas


def smooth_time_series(data: np.ndarray, window_seconds: float, frame_rate: float) -> np.ndarray:
    """
    Apply smoothing to a time series using a rolling average.
    
    Parameters
    ----------
    data : np.ndarray
        The time series data to smooth
    window_seconds : float
        The smoothing window size in seconds
    frame_rate : float
        The frame rate of the data (fps)
        
    Returns
    -------
    np.ndarray
        Smoothed data (same length as input)
    """
    if window_seconds <= 0:
        return data
    
    # Calculate window size in samples
    window_samples = int(window_seconds * frame_rate)
    if window_samples < 1:
        window_samples = 1
    
    # Handle NaN values by interpolating, smoothing, then restoring NaN positions
    nan_mask = np.isnan(data)
    if np.all(nan_mask):
        return data
    
    # Create a copy for smoothing
    smoothed = data.copy()
    
    # For data with NaNs, we'll work only on valid sections
    if np.any(nan_mask):
        # Simple approach: interpolate over NaNs, smooth, then restore NaNs
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) < 2:
            return data
        
        # Linear interpolation over NaN gaps
        smoothed = np.interp(
            np.arange(len(data)),
            valid_indices,
            data[valid_indices]
        )
        
        # Apply uniform filter (moving average)
        smoothed = uniform_filter1d(smoothed, size=window_samples, mode='nearest')
    else:
        smoothed = uniform_filter1d(data, size=window_samples, mode='nearest')
    
    return smoothed
