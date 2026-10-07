"""
fish_analyzer/gui/measure_dialog.py
===================================
Measure the pixel scale on a video frame: click the two ends of something
whose real length is known - a ruler, or the inside width of the tank - and
type that length.
"""
import math
import tkinter as tk
from tkinter import messagebox
from typing import Callable, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .utils import install_canvas_error_handler


def load_frame_image(loaded_file) -> Optional[np.ndarray]:
    """A frame to measure on: the session's first video frame, or failing
    that idtracker.ai's background image. None if neither can be read."""
    video = getattr(loaded_file, "video_file_path", None)
    if video is not None:
        try:
            import cv2
            capture = cv2.VideoCapture(str(video))
            ok, frame = capture.read()
            capture.release()
            if ok and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Note: could not read a frame from {video}: {e}")

    background = getattr(loaded_file, "background_image_path", None)
    if background is not None:
        try:
            import matplotlib.image as mpimg
            return mpimg.imread(str(background))
        except Exception as e:
            print(f"Note: could not read {background}: {e}")
    return None


def pixel_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


class MeasureScaleDialog:
    """A window showing `image`; calls on_done(pixels, centimetres) on OK."""

    def __init__(self, parent, image: np.ndarray, session_name: str,
                 on_done: Callable[[float, float], None]):
        self.on_done = on_done
        self.points: List[Tuple[float, float]] = []

        self.window = tk.Toplevel(parent)
        self.window.title(f"Measure the scale - {session_name}")
        self.window.geometry("1000x820")

        tk.Label(
            self.window, justify=tk.LEFT, font=("Arial", 10),
            text="Click the two ends of something whose real length you know: "
                 "a ruler, or the inside width of the tank.\n"
                 "A third click starts again. Use the zoom tool below the "
                 "picture for a more exact click, then switch it off to click."
        ).pack(anchor="w", padx=10, pady=(8, 4))

        controls = tk.Frame(self.window)
        controls.pack(side="bottom", fill="x", padx=10, pady=8)
        tk.Label(controls, text="That length is").pack(side="left")
        self.length_var = tk.StringVar()
        tk.Entry(controls, textvariable=self.length_var, width=8).pack(side="left", padx=5)
        tk.Label(controls, text="cm").pack(side="left")
        self.distance_var = tk.StringVar(value="Click the first point.")
        tk.Label(controls, textvariable=self.distance_var,
                 font=("Arial", 10, "bold"), fg="#1f4e79").pack(side="left", padx=20)
        tk.Button(controls, text="Cancel",
                  command=self.window.destroy).pack(side="right", padx=5)
        tk.Button(controls, text="Use this scale", command=self._confirm,
                  bg="lightgreen", font=("Arial", 10, "bold")).pack(side="right", padx=5)

        figure = Figure(figsize=(9, 7), dpi=100)
        self.ax = figure.add_axes([0, 0, 1, 1])
        self.ax.imshow(image, cmap="gray")
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(figure, master=self.window)
        install_canvas_error_handler(self.canvas)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.window)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._line = None
        self.canvas.draw()

    def _on_click(self, event):
        # While the toolbar's zoom or pan is active, a click belongs to it.
        if event.inaxes is not self.ax or self.toolbar.mode:
            return
        if len(self.points) == 2:
            self.points = []
        self.points.append((event.xdata, event.ydata))
        self._redraw()

    def _redraw(self):
        if self._line is not None:
            self._line.remove()
            self._line = None
        if self.points:
            xs, ys = zip(*self.points)
            (self._line,) = self.ax.plot(
                xs, ys, "o-", color="#ff3b30", markersize=7, linewidth=2)
        pixels = self.measured_pixels()
        if pixels is not None:
            self.distance_var.set(f"{pixels:.1f} pixels between the two points")
        elif self.points:
            self.distance_var.set("Click the second point.")
        else:
            self.distance_var.set("Click the first point.")
        self.canvas.draw_idle()

    def measured_pixels(self) -> Optional[float]:
        if len(self.points) != 2:
            return None
        return pixel_distance(*self.points)

    def _confirm(self):
        pixels = self.measured_pixels()
        if pixels is None or pixels < 2:
            messagebox.showinfo(
                "Measure", "Click two different points on the picture first.",
                parent=self.window)
            return
        try:
            centimetres = float(self.length_var.get())
            if centimetres <= 0:
                raise ValueError
        except ValueError:
            messagebox.showinfo(
                "Measure", "Type the real length between the two points, in cm.",
                parent=self.window)
            return
        self.window.destroy()
        self.on_done(pixels, centimetres)
