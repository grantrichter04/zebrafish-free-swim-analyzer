"""
fish_analyzer/gui/outline_dialog.py
===================================
Draw the tank outline on a video frame, by clicking its corners in order.

Only needed when a session has no outline from idtracker.ai's setup window, or
that one is wrong. Time near the wall and the random-placement reference for
shoaling are both measured against this outline.
"""
import tkinter as tk
from tkinter import messagebox
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .utils import install_canvas_error_handler


class TankOutlineDialog:
    """A window showing `image`; calls on_done(corners in pixels) on OK.

    `existing` is the outline in use now, drawn faintly for reference.
    """

    def __init__(self, parent, image: np.ndarray, applies_to: str,
                 on_done: Callable[[np.ndarray], None],
                 existing: Optional[Sequence[Sequence[float]]] = None):
        self.on_done = on_done
        self.points: List[Tuple[float, float]] = []

        self.window = tk.Toplevel(parent)
        self.window.title(f"Tank outline - {applies_to}")
        self.window.geometry("1000x820")

        tk.Label(
            self.window, justify=tk.LEFT, font=("Arial", 10),
            text="Click each inside corner of the tank in turn, going round "
                 "it; a round tank needs a dozen or so points along its edge.\n"
                 "Use the zoom tool below the picture for a more exact click, "
                 "then switch it off to click."
                 + ("\nThe dashed line is the outline in use now."
                    if existing is not None else "")
        ).pack(anchor="w", padx=10, pady=(8, 4))

        controls = tk.Frame(self.window)
        controls.pack(side="bottom", fill="x", padx=10, pady=8)
        tk.Button(controls, text="Start again",
                  command=self._start_again).pack(side="left", padx=5)
        self.status_var = tk.StringVar()
        tk.Label(controls, textvariable=self.status_var,
                 font=("Arial", 10, "bold"), fg="#1f4e79").pack(side="left", padx=20)
        tk.Button(controls, text="Cancel",
                  command=self.window.destroy).pack(side="right", padx=5)
        tk.Button(controls, text="Use this outline", command=self._confirm,
                  bg="lightgreen", font=("Arial", 10, "bold")).pack(side="right", padx=5)

        figure = Figure(figsize=(9, 7), dpi=100)
        self.ax = figure.add_axes([0, 0, 1, 1])
        self.ax.imshow(image, cmap="gray")
        self.ax.set_axis_off()
        if existing is not None:
            current = np.asarray(existing, dtype=float)
            current = np.vstack([current, current[:1]])
            self.ax.plot(current[:, 0], current[:, 1], "--", color="#ffd60a",
                         linewidth=1.2)
        self.canvas = FigureCanvasTkAgg(figure, master=self.window)
        install_canvas_error_handler(self.canvas)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.window)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._line = None
        self._redraw()

    def _on_click(self, event):
        # While the toolbar's zoom or pan is active, a click belongs to it.
        if event.inaxes is not self.ax or self.toolbar.mode:
            return
        self.points.append((event.xdata, event.ydata))
        self._redraw()

    def _start_again(self):
        self.points = []
        self._redraw()

    def _redraw(self):
        if self._line is not None:
            self._line.remove()
            self._line = None
        if self.points:
            # Closed back to the first corner once there is an area to close.
            shown = self.points + (self.points[:1] if len(self.points) > 2 else [])
            xs, ys = zip(*shown)
            (self._line,) = self.ax.plot(
                xs, ys, "o-", color="#ff3b30", markersize=6, linewidth=2)
        count = len(self.points)
        self.status_var.set(
            "Click the first corner." if count == 0 else
            f"{count} corner{'s' if count > 1 else ''}. At least 3 are needed."
            if count < 3 else f"{count} corners.")
        self.canvas.draw_idle()

    def corners(self) -> Optional[np.ndarray]:
        """The clicked corners in pixels, or None until there are three."""
        if len(self.points) < 3:
            return None
        return np.asarray(self.points, dtype=float)

    def _confirm(self):
        corners = self.corners()
        if corners is None:
            messagebox.showinfo(
                "Tank outline", "Click at least three corners of the tank first.",
                parent=self.window)
            return
        self.window.destroy()
        self.on_done(corners)
