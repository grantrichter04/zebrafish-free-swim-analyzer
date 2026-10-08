"""
fish_analyzer/gui/inspector_tab.py
===================================
Video Inspector Tab - Unified frame-by-frame viewer with configurable overlays.

Plays a session's video with the tracking drawn on it: fish positions, the
outlines idtracker.ai segmented, trails and lines to each fish's nearest
neighbour, with a shoaling measure over time underneath. The less used
overlays sit under "More options". Scrolling on the video zooms it. Frames
and clips are exported from here.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from ..shoaling import ShoalingParameters, ShoalingCalculator
from ..fish_outlines import OutlineFinder
from ..overlay_render import OverlaySettings, compose_frame, fish_colors

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from PIL import Image as _PIL_Image, ImageTk as _PIL_ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def zoom_view(frame_w, frame_h, zoom, centre=None):
    """The part of a frame shown at `zoom`: (x0, y0, width, height) in frame
    pixels, the frame's own shape, kept inside the frame. `centre` is where
    the view should be centred; the middle of the frame if None."""
    zoom = max(1.0, float(zoom))
    width = max(1, int(round(frame_w / zoom)))
    height = max(1, int(round(frame_h / zoom)))
    cx, cy = centre if centre is not None else (frame_w / 2.0, frame_h / 2.0)
    x0 = int(round(min(max(cx - width / 2.0, 0), frame_w - width)))
    y0 = int(round(min(max(cy - height / 2.0, 0), frame_h - height)))
    return x0, y0, width, height


class InspectorTabMixin:
    """
    Mixin providing the Video Inspector tab.

    Expects from GUIBase:
        - self.root, self.notebook
        - self.loaded_files
        - self.video_readers
        - self.animation_running, self.animation_after_id
    """

    def _create_inspector_tab(self):
        """Create the Video Inspector tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Video Inspector")

        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=0)

        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=1)

        self._create_inspector_controls(left_panel)
        self._create_inspector_display(right_panel)

    # =========================================================================
    # CONTROLS
    # =========================================================================

    TRAIL_OPACITY = 0.5
    MAX_ZOOM = 12.0
    #: What can be plotted under the video: (shown in the list, mode).
    TIME_PANELS = (("Nothing", "none"),
                   ("Nearest-neighbour distance", "nnd"),
                   ("Inter-individual distance", "iid"),
                   ("Area the shoal covers", "hull"))

    def _create_inspector_controls(self, parent):
        """The left column: which session, what to draw on it, and export."""
        column = tk.Frame(parent, width=250)
        column.pack(fill="y", expand=True, padx=8)
        column.pack_propagate(False)

        def heading(text):
            tk.Label(column, text=text, font=("Arial", 10, "bold")
                     ).pack(anchor="w", pady=(14, 3))

        # --- Session ---
        heading("Session")
        self.inspector_file_var = tk.StringVar()
        self.inspector_file_dropdown = ttk.Combobox(
            column, textvariable=self.inspector_file_var, state="readonly")
        self.inspector_file_dropdown.pack(fill="x")
        self.inspector_file_dropdown.bind(
            "<<ComboboxSelected>>", self._on_inspector_file_selected)

        # The video is found from the session; this is only for when it is not.
        #: True while frames come from the video rather than the background.
        self.inspector_video_var = tk.BooleanVar(value=False)
        self.inspector_video_status = tk.Label(
            column, text="", font=("Arial", 8), fg="gray", wraplength=235,
            justify="left")
        self.inspector_video_status.pack(anchor="w", pady=(3, 0))
        self._inspector_find_video_row = tk.Frame(column)
        self._inspector_find_video_row.pack(fill="x")
        self.inspector_find_video_button = tk.Button(
            self._inspector_find_video_row, text="Find video...",
            command=self._inspector_browse_video)

        # --- What to draw ---
        heading("Show")
        self.inspector_show_positions_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            column, text="Fish positions",
            variable=self.inspector_show_positions_var,
            command=self._inspector_on_overlay_change
        ).pack(anchor="w")

        self.inspector_show_outlines_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            column, text="Fish outlines, as idtracker.ai saw them",
            variable=self.inspector_show_outlines_var,
            command=self._inspector_on_outlines_toggle
        ).pack(anchor="w")

        self.inspector_show_nnd_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            column, text="Lines to nearest neighbour",
            variable=self.inspector_show_nnd_var,
            command=self._inspector_on_overlay_change
        ).pack(anchor="w")

        trail_row = tk.Frame(column)
        trail_row.pack(fill="x", pady=(4, 0))
        tk.Label(trail_row, text="Trail: the last").pack(side="left")
        self.inspector_trail_seconds_var = tk.StringVar(value="0")
        trail_box = tk.Spinbox(
            trail_row, from_=0, to=3600, increment=0.5, width=6,
            textvariable=self.inspector_trail_seconds_var,
            command=self._inspector_update_fast)
        trail_box.pack(side="left", padx=4)
        for event in ("<Return>", "<FocusOut>"):
            trail_box.bind(event, lambda e: self._inspector_update_fast())
        tk.Label(trail_row, text="seconds").pack(side="left")

        # A string because the renderer and the exporter both switch on which
        # panel, if any, sits under the video.
        self.inspector_time_mode_var = tk.StringVar(value="none")
        tk.Label(column, text="Under the video, plot:").pack(anchor="w", pady=(8, 0))
        self.inspector_time_panel_box = ttk.Combobox(
            column, state="readonly", values=[text for text, _ in self.TIME_PANELS])
        self.inspector_time_panel_box.current(0)
        self.inspector_time_panel_box.pack(fill="x")
        self.inspector_time_panel_box.bind(
            "<<ComboboxSelected>>", self._inspector_on_time_panel_chosen)

        # --- Used less often, so out of the way until asked for ---
        more_button = tk.Button(column, text="\u25B6 More options", relief="flat",
                                anchor="w", font=("Arial", 9, "bold"))
        more_button.pack(fill="x", pady=(10, 0))
        more = tk.Frame(column)

        def toggle_more():
            if more.winfo_manager():
                more.pack_forget()
                more_button.config(text="\u25B6 More options")
            else:
                more.pack(fill="x", after=more_button)
                more_button.config(text="\u25BC More options")

        more_button.config(command=toggle_more)
        self._inspector_more_frame = more

        self.inspector_show_hull_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            more, text="Outline around the shoal",
            variable=self.inspector_show_hull_var,
            command=self._inspector_on_overlay_change
        ).pack(anchor="w")

        iid_row = tk.Frame(more)
        iid_row.pack(fill="x")
        self.inspector_show_iid_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            iid_row, text="Lines to all others, from fish",
            variable=self.inspector_show_iid_var,
            command=self._inspector_on_overlay_change
        ).pack(side="left")
        self.inspector_iid_focus_var = tk.StringVar(value="")
        self.inspector_iid_focus_combo = ttk.Combobox(
            iid_row, textvariable=self.inspector_iid_focus_var,
            state="readonly", width=4)
        self.inspector_iid_focus_combo.pack(side="left")
        self.inspector_iid_focus_combo.bind(
            "<<ComboboxSelected>>", lambda e: self._inspector_on_overlay_change())

        dot_row = tk.Frame(more)
        dot_row.pack(fill="x", pady=(2, 0))
        tk.Label(dot_row, text="Dot size:").pack(side="left", anchor="s")
        self.inspector_dot_size_var = tk.IntVar(value=12)
        tk.Scale(
            dot_row, from_=3, to=24, orient=tk.HORIZONTAL,
            variable=self.inspector_dot_size_var,
            command=lambda v: self._inspector_update_fast(),
            showvalue=False
        ).pack(side="left", padx=3, fill="x", expand=True)

        tk.Label(column,
                 text="Scroll on the video to zoom, drag to move, "
                      "double-click to see it all again.",
                 font=("Arial", 8), fg="gray", wraplength=235,
                 justify="left").pack(anchor="w", pady=(10, 0))

        # --- Export ---
        heading("Export")
        tk.Button(column, text="Save frame (PNG)...",
                  command=self._inspector_save_frame,
                  bg="lightgreen").pack(fill="x", pady=2)
        tk.Button(column, text="Export clip...",
                  command=self._inspector_export_clip_dialog,
                  bg="lightgreen").pack(fill="x", pady=2)
        tk.Label(column,
                 text="Exports what is shown, at full video resolution. "
                      "A clip runs from Set In to Set Out, under the video.",
                 font=("Arial", 8), fg="gray", wraplength=235,
                 justify="left").pack(anchor="w", pady=(2, 0))

    # =========================================================================
    # DISPLAY AREA
    # =========================================================================

    def _create_inspector_transport(self, parent):
        """Player-style transport bar: playback, scrubber, range markers.

        Lives under the video rather than in the left column, where the slider
        was only 180px wide and scrubbing an 18,000 frame recording meant about
        100 frames per pixel.
        """
        row = tk.Frame(parent, bg="#ececec")
        row.pack(fill="x", padx=8, pady=(4, 2))

        self.inspector_play_button = tk.Button(
            row, text="> Play", command=self._inspector_toggle_playback,
            bg="lightblue", width=8
        )
        self.inspector_play_button.pack(side="left", padx=2)

        tk.Button(row, text="◀", width=3,
                  command=self._inspector_step_back).pack(side="left", padx=1)
        tk.Button(row, text="▶", width=3,
                  command=self._inspector_step_forward).pack(side="left",
                                                             padx=1)

        tk.Label(row, text="Speed:", bg="#ececec").pack(side="left",
                                                        padx=(8, 2))
        self.inspector_speed_var = tk.StringVar(value="1x")
        ttk.Combobox(
            row, textvariable=self.inspector_speed_var,
            values=["0.25x", "0.5x", "1x", "2x", "4x", "8x"],
            width=5, state="readonly"
        ).pack(side="left")

        self.inspector_frame_var = tk.IntVar(value=0)
        self.inspector_frame_slider = tk.Scale(
            row, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.inspector_frame_var,
            command=self._on_inspector_slider_change,
            showvalue=False, bg="#ececec", highlightthickness=0,
            sliderlength=18, width=14
        )
        self.inspector_frame_slider.pack(side="left", fill="x", expand=True,
                                         padx=8)

        mark_row = tk.Frame(parent, bg="#ececec")
        mark_row.pack(fill="x", padx=8, pady=(0, 4))
        tk.Button(mark_row, text="Set In", command=self._inspector_set_mark_in,
                  bg="lightblue").pack(side="left", padx=2)
        tk.Button(mark_row, text="Set Out",
                  command=self._inspector_set_mark_out,
                  bg="lightblue").pack(side="left", padx=2)
        tk.Button(mark_row, text="Clear",
                  command=self._inspector_clear_marks).pack(side="left",
                                                            padx=2)
        self.inspector_mark_label = tk.Label(
            mark_row, text="In -- | Out --", font=("Arial", 9), bg="#ececec"
        )
        self.inspector_mark_label.pack(side="left", padx=10)

        # On the second row, so row one is buttons plus a scrubber that gets
        # everything else.
        self.inspector_info_label = tk.Label(
            mark_row, text="", font=("Arial", 9),
            bg="#ececec", anchor="e"
        )
        self.inspector_info_label.pack(side="right", padx=(8, 2))

    def _create_inspector_display(self, parent):
        """Create the inspector display area.

        Three stacked regions, top to bottom: the video, the transport bar and
        the time panel. The transport and the time panel each get their own
        persistent container because _inspector_rebuild_figure destroys and
        recreates everything in the video region.
        """
        # Packed bottom-first, so the visual order ends up video, transport,
        # time panel.
        self.inspector_time_frame = tk.Frame(parent, bg="white")
        self.inspector_time_frame.pack(side="bottom", fill="x")

        transport = tk.Frame(parent, bg="#ececec")
        transport.pack(side="bottom", fill="x")
        self._create_inspector_transport(transport)

        self.inspector_plot_frame = tk.Frame(parent, bg="white")
        self.inspector_plot_frame.pack(fill="both", expand=True)

        tk.Label(
            self.inspector_plot_frame,
            text="Load sessions on the Sessions & Units tab, then choose "
                 "one here.",
            font=("Arial", 11), fg="gray"
        ).pack(expand=True)

        # Inspector figure state
        # NOTE: rebuild is tracked by its own flag, not by "_insp_fig is None".
        # _insp_fig is only ever assigned when a time panel is shown, so using
        # it as the sentinel made every update rebuild the whole widget tree
        # whenever Time Panel was set to "None" (the default).
        self._insp_needs_rebuild = True
        self._insp_fig = None
        self._insp_canvas = None
        self._insp_ax_main = None
        self._insp_ax_time = None
        self._insp_ax_time2 = None
        self._insp_time_marker = None
        self._insp_dynamic_artists = []
        self._insp_cached_file = None
        self._insp_cached_overlays = None
        self._insp_cached_time_mode = None
        self._insp_video_bg_artist = None
        self._insp_cached_background = None
        self._insp_cached_width_bl = None
        self._insp_cached_height_bl = None
        self._insp_bg_cache = None
        self._insp_composite_artist = None
        # PIL/ImageTk video display (replaces matplotlib imshow for speed)
        self._insp_video_canvas = None
        self._insp_video_photo = None
        self._insp_video_canvas_item = None
        self._insp_title_item = None
        # Per-fish colours, recomputed only when the fish count changes.
        self._insp_fish_colors = None
        # Zoom: how far in, and the frame pixel the view is centred on.
        self._insp_zoom = 1.0
        self._insp_zoom_centre = None
        self._insp_view = None
        # One OutlineFinder per session; None once it is known to be missing.
        self._insp_outline_finders = {}
        # Export range. None means "not marked", not "frame 0".
        self.inspector_mark_in = None
        self.inspector_mark_out = None
        self._insp_export_after_id = None
        self._insp_recapture_after_id = None


    def render_settings_from_vars(self, zoom: float = 1.0) -> OverlaySettings:
        """Snapshot the overlay controls.

        The single place tk state becomes an OverlaySettings - the live view
        and the exporter both go through here, so they cannot disagree about
        what is being drawn. Only the live view passes its `zoom`, which
        thins the overlays so they stay the same size on screen; an export
        is always the whole frame.
        """
        return OverlaySettings(
            show_positions=self.inspector_show_positions_var.get(),
            show_nnd=self.inspector_show_nnd_var.get(),
            show_hull=self.inspector_show_hull_var.get(),
            show_iid=self.inspector_show_iid_var.get(),
            iid_focus=self._inspector_iid_focus_index(),
            trail_length=self._inspector_trail_frames(),
            trail_opacity=self.TRAIL_OPACITY,
            dot_radius=max(2, int(round(self.inspector_dot_size_var.get() / zoom))),
            line_scale=1.0 / zoom,
        )

    def _inspector_trail_frames(self) -> int:
        """The trail box, in frames of the session on show. Anything that is
        not a number counts as no trail."""
        try:
            seconds = max(0.0, float(self.inspector_trail_seconds_var.get()))
        except (ValueError, TypeError, tk.TclError):
            return 0
        loaded = self.loaded_files.get(self.inspector_file_var.get())
        fps = loaded.calibration.frame_rate if loaded is not None else 30.0
        return int(round(seconds * fps))

    def _inspector_iid_focus_index(self) -> int:
        """Which fish the lines-to-all-others start from, as a row of the
        trajectories. The list shows idtracker.ai's labels."""
        values = list(self.inspector_iid_focus_combo["values"])
        chosen = self.inspector_iid_focus_var.get()
        return values.index(chosen) if chosen in values else 0

    def _inspector_on_time_panel_chosen(self, event=None):
        chosen = self.inspector_time_panel_box.get()
        self.inspector_time_mode_var.set(dict(self.TIME_PANELS).get(chosen, "none"))
        self._inspector_rebuild_needed()

    def _inspector_on_outlines_toggle(self):
        """Say so, once, if this session cannot show outlines."""
        selected = self.inspector_file_var.get()
        if (self.inspector_show_outlines_var.get() and selected in self.loaded_files
                and self._inspector_outline_finder(selected) is None):
            self.inspector_show_outlines_var.set(False)
            messagebox.showinfo(
                "No outlines for this session",
                "The outlines are found again from the thresholds and the "
                "background idtracker.ai saved in the session folder, and "
                f"those could not be read for '{selected}'.")
            return
        self._inspector_update_fast()

    def _inspector_outline_finder(self, selected: str):
        if selected not in self._insp_outline_finders:
            session_folder = Path(self.loaded_files[selected].file_path).parent.parent
            self._insp_outline_finders[selected] = OutlineFinder.for_session(session_folder)
        return self._insp_outline_finders[selected]

    def _inspector_outlines_for(self, selected: str, frame):
        """The blob outlines in `frame`, or None when they are not shown."""
        if not self.inspector_show_outlines_var.get():
            return None
        finder = self._inspector_outline_finder(selected)
        return finder.find(frame) if finder is not None else None

    # =========================================================================
    # FILE SELECTION
    # =========================================================================

    def _update_inspector_file_dropdown(self):
        """Update the inspector file dropdown with loaded files."""
        file_list = list(self.loaded_files.keys())
        self.inspector_file_dropdown['values'] = file_list
        if file_list and not self.inspector_file_var.get():
            self.inspector_file_dropdown.current(0)
            self._on_inspector_file_selected()

    def _on_inspector_file_selected(self, event=None):
        """Handle file selection change."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            return

        loaded = self.loaded_files[selected]

        # One slider position per frame.
        self.inspector_frame_slider.configure(to=max(0, loaded.n_frames - 1))
        self.inspector_frame_var.set(0)

        # A range marked on one recording means nothing on another, and
        # silently carrying it over would export the wrong stretch.
        self._inspector_clear_marks()

        labels = [str(label) for label in loaded.metadata.identity_labels]
        self.inspector_iid_focus_combo["values"] = labels
        if labels:
            self.inspector_iid_focus_combo.current(0)
        self._insp_zoom, self._insp_zoom_centre = 1.0, None

        self.inspector_video_var.set(
            self._inspector_try_auto_load_video(selected))

        # Force rebuild
        self._insp_needs_rebuild = True
        self._update_inspector_info()
        self._inspector_update_fast()

    # =========================================================================
    # FRAME INFO
    # =========================================================================

    def _get_inspector_frame_idx(self):
        """The frame the slider is on."""
        return self.inspector_frame_var.get()

    def _update_inspector_info(self):
        """Update the frame info label."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            return

        loaded = self.loaded_files[selected]
        frame_idx = self._get_inspector_frame_idx()
        frame_idx = min(frame_idx, loaded.n_frames - 1)
        fps = loaded.calibration.frame_rate
        time_s = frame_idx / fps

        info = f"Frame {frame_idx} | {time_s:.1f} s"

        # The nearest shoaling sample, when the session has been analysed.
        if loaded.shoaling_results:
            results = loaded.shoaling_results
            nearest = int(np.argmin(np.abs(results.frame_indices - frame_idx)))
            info += (f" | Nearest neighbour "
                     f"{results.mean_nnd_per_sample[nearest]:.2f} "
                     f"{loaded.calibration.unit_name}")

        self.inspector_info_label.config(text=info)

    # =========================================================================
    # NAVIGATION
    # =========================================================================

    def _on_inspector_slider_change(self, value):
        """Called when frame slider changes."""
        self._update_inspector_info()
        self._inspector_update_fast()

    def _inspector_step(self, frames: int):
        """Move by `frames`, staying inside the recording."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            return
        last = self.loaded_files[selected].n_frames - 1
        self.inspector_frame_var.set(
            max(0, min(last, self._get_inspector_frame_idx() + frames)))
        self._update_inspector_info()
        self._inspector_update_fast()

    def _inspector_step_back(self):
        self._inspector_step(-1)

    def _inspector_step_forward(self):
        self._inspector_step(1)

    # =========================================================================
    # PLAYBACK
    # =========================================================================

    def _inspector_toggle_playback(self):
        """Toggle animation playback."""
        if self.animation_running:
            self._inspector_stop_playback()
        else:
            self._inspector_start_playback()

    def _inspector_start_playback(self):
        """Start animation."""
        self.animation_running = True
        self.inspector_play_button.config(text="|| Pause", bg="salmon")

        self._inspector_animate_step()

    def _inspector_playback_pace(self, fps: float):
        """(frames to advance per tick, milliseconds per tick).

        Drawing a frame takes about as long as a frame lasts, so playback
        faster than real time shows every second, fourth or eighth frame at
        an unchanged tick rather than trying to draw them all.
        """
        try:
            speed = float(self.inspector_speed_var.get().replace('x', ''))
        except ValueError:
            speed = 1.0
        stride = max(1, int(round(speed)))
        return stride, max(16, int(stride / (fps * speed) * 1000))

    def _inspector_stop_playback(self):
        """Stop animation."""
        self.animation_running = False
        self.inspector_play_button.config(text="> Play", bg="lightblue")
        if self.animation_after_id:
            self.root.after_cancel(self.animation_after_id)
            self.animation_after_id = None

    def _on_inspector_time_canvas_resize(self, event=None):
        """Drop the blit background when the time panel changes size.

        A cached background is only valid for the canvas size it was captured
        at. Restoring a stale one leaves the previous, differently-sized
        rendering visible alongside it - the plot appears duplicated.

        Until it is recaptured the cursor path falls back to draw_idle(), which
        is correct but slower, so the recapture is debounced rather than run on
        every <Configure> during a drag.
        """
        self._insp_bg_cache = None

        if getattr(self, '_insp_recapture_after_id', None):
            self.root.after_cancel(self._insp_recapture_after_id)
        self._insp_recapture_after_id = self.root.after(
            200, self._inspector_recapture_time_background
        )

    def _inspector_recapture_time_background(self):
        """Redraw the time panel and cache it for blitting at the new size."""
        self._insp_recapture_after_id = None
        if self._insp_canvas is None or self._insp_fig is None:
            return
        try:
            self._insp_canvas.draw()
            self._insp_bg_cache = self._insp_canvas.copy_from_bbox(
                self._insp_fig.bbox
            )
        except Exception:
            self._insp_bg_cache = None

    def _on_inspector_resize(self, event=None):
        """Pause animation during window resize to prevent UI freeze.

        The video canvas fires <Configure> continuously while the user drags
        the window edge. Rendering a resized PIL image on every event would
        race with Tkinter's layout engine and lock up the UI. Instead we
        pause the animation loop and restart it 200 ms after the last event.
        """
        # Cancel any pending restart
        if hasattr(self, '_resize_after_id') and self._resize_after_id:
            self.root.after_cancel(self._resize_after_id)
            self._resize_after_id = None

        # Pause the animation loop (without changing the play/pause button)
        was_running = self.animation_running
        if was_running:
            self.animation_running = False
            if self.animation_after_id:
                self.root.after_cancel(self.animation_after_id)
                self.animation_after_id = None

        # Invalidate the canvas items so they get repositioned on next draw
        self._insp_video_canvas_item = None
        self._insp_title_item = None
        # The time panel resizes with the window too, and its cached blit
        # background is only valid at the size it was captured.
        self._insp_bg_cache = None

        def _resume():
            self._resize_after_id = None
            if was_running:
                self.animation_running = True
                self.inspector_play_button.config(text="|| Pause", bg="salmon")
                self._inspector_animate_step()
            else:
                self._inspector_update_fast()

        self._resize_after_id = self.root.after(200, _resume)

    def _inspector_animate_step(self):
        """Advance one step in animation."""
        if not self.animation_running:
            return

        import time

        selected = self.inspector_file_var.get()
        stride, target_ms = 1, 100
        if selected and selected in self.loaded_files:
            stride, target_ms = self._inspector_playback_pace(
                self.loaded_files[selected].calibration.frame_rate)

        t_start = time.perf_counter()

        try:
            current = self.inspector_frame_var.get()
            max_val = int(self.inspector_frame_slider.cget('to'))

            if current + stride <= max_val:
                self.inspector_frame_var.set(current + stride)
            else:
                self.inspector_frame_var.set(0)

            self._update_inspector_info()
            self._inspector_update_fast()
        except Exception as e:
            # This used to be a bare `pass`, on the grounds that a render error
            # must not break the animation chain. But it swallowed the error on
            # every channel: the frame counter kept advancing while the image
            # stayed frozen, which is indistinguishable from a still video.
            # Stopping playback first means the error is reported exactly once
            # instead of on every frame.
            self._inspector_stop_playback()
            print(f"[FAILED] Inspector render at frame "
                  f"{self._get_inspector_frame_idx()}: {e}")
            self._report_uncaught(e)
            return

        # Subtract render time; keep a floor of 8 ms so Tkinter can breathe
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        delay_ms = max(8, int(target_ms - elapsed_ms))

        self.animation_after_id = self.root.after(
            delay_ms, self._inspector_animate_step
        )

    # =========================================================================
    # ZOOM
    # =========================================================================

    def _inspector_frame_point(self, canvas_x, canvas_y):
        """The frame pixel under a canvas position, or None before a draw."""
        if self._insp_view is None:
            return None
        x0, y0, scale_fit, left, top = self._insp_view
        return x0 + (canvas_x - left) / scale_fit, y0 + (canvas_y - top) / scale_fit

    def _inspector_frame_size(self):
        loaded = self.loaded_files.get(self.inspector_file_var.get())
        if loaded is None:
            return None
        return loaded.metadata.video_width, loaded.metadata.video_height

    def _on_inspector_wheel(self, event):
        """Zoom in or out, keeping the spot under the pointer where it is."""
        size = self._inspector_frame_size()
        under = self._inspector_frame_point(event.x, event.y)
        if size is None or under is None:
            return
        old = self._insp_zoom
        new = min(self.MAX_ZOOM, max(1.0, old * (1.25 if event.delta > 0 else 0.8)))
        if new == old:
            return
        x0, y0, _, _ = zoom_view(*size, old, self._insp_zoom_centre)
        # The pointer keeps its place in the view: same fraction across it.
        fraction_x = (under[0] - x0) / (size[0] / old)
        fraction_y = (under[1] - y0) / (size[1] / old)
        self._insp_zoom = new
        self._insp_zoom_centre = (
            under[0] + (0.5 - fraction_x) * size[0] / new,
            under[1] + (0.5 - fraction_y) * size[1] / new)
        self._inspector_update_fast()

    def _on_inspector_drag_start(self, event):
        self._insp_drag_from = (event.x, event.y)

    def _on_inspector_drag(self, event):
        """Move the zoomed view with the mouse."""
        size = self._inspector_frame_size()
        if size is None or self._insp_view is None or self._insp_zoom <= 1.0:
            return
        start = getattr(self, "_insp_drag_from", None) or (event.x, event.y)
        x0, y0, view_w, view_h = zoom_view(*size, self._insp_zoom, self._insp_zoom_centre)
        scale_fit = self._insp_view[2]
        self._insp_zoom_centre = (
            x0 + view_w / 2.0 - (event.x - start[0]) / scale_fit,
            y0 + view_h / 2.0 - (event.y - start[1]) / scale_fit)
        self._insp_drag_from = (event.x, event.y)
        self._inspector_update_fast()

    def _on_inspector_zoom_reset(self, event=None):
        self._insp_zoom, self._insp_zoom_centre = 1.0, None
        self._inspector_update_fast()

    # =========================================================================
    # OVERLAY / REBUILD TRIGGERS
    # =========================================================================

    def _inspector_on_overlay_change(self):
        """Called when any overlay checkbox changes.
        With CV2 compositing, overlays are drawn each frame — no rebuild needed.
        """
        self._inspector_update_fast()

    def _inspector_rebuild_needed(self, event=None):
        """Force a figure rebuild on next update."""
        self._insp_needs_rebuild = True
        self._inspector_update_fast()

    # =========================================================================
    # VIDEO
    # =========================================================================

    def _inspector_show_video_status(self, text: str, found: bool):
        """Say which video is showing, and offer to find one only if none is."""
        self.inspector_video_status.config(
            text=text, fg="gray" if found else "#b35c00")
        if found:
            self.inspector_find_video_button.pack_forget()
        else:
            self.inspector_find_video_button.pack(anchor="w", pady=(3, 0))

    def _inspector_try_auto_load_video(self, selected: str) -> bool:
        """
        Try to load the video that was auto-detected at file-load time.

        Returns True if a video is now ready in self.video_readers[selected].
        Without one the inspector draws on idtracker.ai's background image.
        """
        if selected in self.video_readers:
            self._inspector_show_video_status(
                self.video_readers[selected].video_path.name, True)
            return True

        loaded = self.loaded_files.get(selected)
        if loaded is None:
            return False

        video_path = loaded.video_file_path
        if video_path is None or not video_path.exists():
            self._inspector_show_video_status(
                "The video for this session was not found, so the tracking "
                "is drawn on a still background.", False)
            return False

        try:
            from ..video_utils import VideoFrameReader, CV2_AVAILABLE
            if not CV2_AVAILABLE:
                self._inspector_show_video_status(
                    "OpenCV is missing, so the video cannot be shown. "
                    "Run install.bat again.", True)
                return False

            self.video_readers[selected] = VideoFrameReader(video_path)
            self._inspector_show_video_status(video_path.name, True)
            return True

        except Exception as e:
            self._inspector_show_video_status(
                f"{video_path.name} could not be opened: {e}", False)
            return False

    def _inspector_browse_video(self):
        """Choose the session's video by hand."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            messagebox.showwarning("No session", "Choose a session first.")
            return

        path = filedialog.askopenfilename(
            title=f"Video for {selected}",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        try:
            from ..video_utils import VideoFrameReader

            if selected in self.video_readers:
                self.video_readers[selected].close()

            self.video_readers[selected] = VideoFrameReader(Path(path))
            self.loaded_files[selected].video_file_path = Path(path)
        except Exception as e:
            messagebox.showerror("Video Error",
                                 f"Could not open video:\n{e}")
            return

        self._inspector_show_video_status(Path(path).name, True)
        self.inspector_video_var.set(True)
        self._insp_needs_rebuild = True
        self._inspector_update_fast()

    # =========================================================================
    # MAIN RENDERING DISPATCHER
    # =========================================================================

    def _inspector_update_fast(self):
        """Main update entry point - rebuild if needed, then update dynamic."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            return

        loaded = self.loaded_files[selected]
        frame_idx = self._get_inspector_frame_idx()
        frame_idx = min(frame_idx, loaded.n_frames - 1)

        time_mode = self.inspector_time_mode_var.get()

        # A found video changes what the overlays are composited onto.
        video_on = self.inspector_video_var.get()

        rebuild = (
            self._insp_needs_rebuild
            or self._insp_cached_file != selected
            or self._insp_cached_time_mode != time_mode
            or self._insp_cached_overlays != video_on
        )

        if rebuild:
            self.set_status("Rebuilding inspector...")
            self.root.update_idletasks()
            self._inspector_rebuild_figure(
                selected, loaded, frame_idx, time_mode
            )
            self.set_status("Ready")
            self._insp_needs_rebuild = False
            self._insp_cached_file = selected
            self._insp_cached_time_mode = time_mode
            self._insp_cached_overlays = video_on

        self._inspector_update_dynamic(
            selected, loaded, frame_idx, time_mode
        )

    # =========================================================================
    # FIGURE REBUILD (expensive, only when layout changes)
    # =========================================================================

    def _inspector_rebuild_figure(self, selected, loaded, frame_idx,
                                   time_mode):
        """Rebuild the inspector layout from scratch.

        Video is displayed via PIL/ImageTk on a tk.Canvas (fast, direct pixel
        transfer). The time panel beneath uses a small matplotlib Figure only
        for the static chart + moving cursor (blitting works well there).
        """
        for widget in self.inspector_plot_frame.winfo_children():
            widget.destroy()
        # The time panel has its own container so the transport bar between
        # them survives a rebuild.
        for widget in self.inspector_time_frame.winfo_children():
            widget.destroy()

        # Reset all canvas/artist state
        self._insp_video_canvas = None
        self._insp_video_photo = None
        self._insp_video_canvas_item = None
        self._insp_title_item = None
        self._insp_fig = None
        self._insp_canvas = None
        self._insp_ax_main = None
        self._insp_ax_time = None
        self._insp_ax_time2 = None
        self._insp_composite_artist = None
        self._insp_bg_cache = None
        self._insp_time_marker = None
        self._insp_dynamic_artists = []

        show_time = time_mode != "none"

        vid_h = loaded.metadata.video_height
        vid_w = loaded.metadata.video_width
        pixels_to_bl = loaded.calibration.scale_factor
        self._insp_cached_width_bl = vid_w * pixels_to_bl
        self._insp_cached_height_bl = vid_h * pixels_to_bl

        # Always its own container, below the transport bar.
        time_parent = self.inspector_time_frame

        # --- Video canvas (PIL/ImageTk — replaces matplotlib imshow) ---
        self._insp_video_canvas = tk.Canvas(self.inspector_plot_frame,
                                            bg="black",
                                            highlightthickness=0)
        self._insp_video_canvas.pack(fill="both", expand=True)
        self._insp_video_canvas.bind("<Configure>", self._on_inspector_resize)
        self._insp_video_canvas.bind("<MouseWheel>", self._on_inspector_wheel)
        self._insp_video_canvas.bind("<ButtonPress-1>", self._on_inspector_drag_start)
        self._insp_video_canvas.bind("<B1-Motion>", self._on_inspector_drag)
        self._insp_video_canvas.bind("<Double-Button-1>", self._on_inspector_zoom_reset)

        # --- Time panel: small matplotlib figure below the video ---
        if show_time:
            self._insp_fig = Figure(figsize=(10, 2.0), dpi=100)
            self._insp_fig.patch.set_facecolor('#f5f5f5')
            self._insp_ax_time = self._insp_fig.add_axes(
                [0.08, 0.22, 0.88, 0.68]
            )
            self._insp_ax_time2 = None

            self._insp_canvas = FigureCanvasTkAgg(
                self._insp_fig, master=time_parent
            )
            time_widget = self._insp_canvas.get_tk_widget()
            time_widget.pack(fill="x", side="bottom")
            # The figure is created 10in wide but the widget is packed fill="x",
            # so Tk stretches it. FigureCanvasTkAgg has its own <Configure>
            # binding that resizes the figure to match; add="+" is essential
            # here, because a plain bind() would replace it and leave the
            # figure painting only part of its widget - which is what produced
            # a second, stale copy of the plot beside the real one.
            time_widget.bind("<Configure>",
                             self._on_inspector_time_canvas_resize, add="+")

            # Populate the time axes
            if loaded.shoaling_results:
                results = loaded.shoaling_results
                time_min = results.timestamps / 60.0
                ax_t = self._insp_ax_time
                unit = loaded.calibration.unit_name
                values, label = {
                    "nnd": (results.mean_nnd_per_sample, f"Nearest neighbour ({unit})"),
                    "iid": (results.mean_iid_per_sample, f"Inter-individual ({unit})"),
                    "hull": (results.convex_hull_area_per_sample,
                             f"Area covered ({unit}\u00b2)"),
                }[time_mode]
                ax_t.plot(time_min, values, 'b-', lw=1, alpha=0.7)
                ax_t.set_ylabel(label, fontsize=8)
                ax_t.set_xlabel('Time (min)', fontsize=8)
                ax_t.set_xlim(time_min[0], time_min[-1])
                ax_t.grid(True, alpha=0.3)
                # Animated, so a full draw of the chart leaves it out. Without
                # that, a chart cached while the cursor sat mid-recording kept
                # a second, stationary cursor, on screen and in exported clips.
                self._insp_time_marker = ax_t.axvline(
                    x=0, color='red', lw=2, linestyle='--', animated=True
                )
            else:
                self._insp_ax_time.text(
                    0.5, 0.5,
                    "No shoaling results for this session yet.\n"
                    "Press Run All Analysis on Sessions & Units.",
                    transform=self._insp_ax_time.transAxes,
                    ha='center', va='center', fontsize=10, color='gray'
                )

            # Draw static chart content and cache background for blitting
            self._insp_canvas.draw()
            try:
                self._insp_bg_cache = self._insp_canvas.copy_from_bbox(
                    self._insp_fig.bbox
                )
            except Exception:
                self._insp_bg_cache = None

        # --- Background image for compositing (used when video is off) ---
        self._insp_cached_background = None
        self._insp_video_bg_artist = None
        use_video = self.inspector_video_var.get()
        if (not use_video and loaded.background_image_path
                and loaded.background_image_path.exists()):
            try:
                bg = plt.imread(str(loaded.background_image_path))
                if bg.dtype != np.uint8:
                    bg = (np.clip(bg, 0, 1) * 255).astype(np.uint8)
                if len(bg.shape) == 2:
                    bg = np.stack([bg, bg, bg], axis=-1)
                elif bg.shape[2] == 4:
                    bg = bg[:, :, :3]
                self._insp_cached_background = bg
            except Exception:
                pass

    # =========================================================================
    # DYNAMIC UPDATE (fast, every frame/step)
    # =========================================================================

    def _inspector_update_dynamic(self, selected, loaded, frame_idx,
                                   time_mode):
        """Update the display — composites overlays onto the frame.

        Uses CV2 drawing when available (much faster), falls back to
        matplotlib artists otherwise.
        """
        if self._insp_video_canvas is None:
            return

        n_fish = loaded.n_fish
        vid_h = loaded.metadata.video_height
        vid_w = loaded.metadata.video_width
        scale = loaded.calibration.scale_factor
        fps = loaded.calibration.frame_rate
        time_s = frame_idx / fps

        # --- Get base frame ---
        use_video = self.inspector_video_var.get()
        display = None

        if use_video and selected in self.video_readers:
            reader = self.video_readers[selected]
            frame = reader.read_frame(frame_idx)
            if frame is not None:
                display = frame.copy()

        if display is None and self._insp_cached_background is not None:
            display = self._insp_cached_background.copy()

        if display is None:
            display = np.ones((vid_h, vid_w, 3), dtype=np.uint8) * 200

        # Ensure correct format
        if display.dtype != np.uint8:
            display = (np.clip(display, 0, 1) * 255).astype(np.uint8)
        if len(display.shape) == 2:
            display = np.stack([display, display, display], axis=-1)
        if display.shape[2] == 4:
            display = display[:, :, :3]

        # Resize to match arena dimensions if needed
        if display.shape[0] != vid_h or display.shape[1] != vid_w:
            if _CV2_AVAILABLE:
                display = _cv2.resize(display, (vid_w, vid_h))
            else:
                display = np.ones((vid_h, vid_w, 3), dtype=np.uint8) * 200

        if (self._insp_fish_colors is None
                or len(self._insp_fish_colors) != n_fish):
            self._insp_fish_colors = fish_colors(n_fish)

        display = compose_frame(display, loaded.trajectories, frame_idx,
                                self.render_settings_from_vars(self._insp_zoom),
                                scale, self._insp_fish_colors,
                                loaded.metadata.identity_labels,
                                self._inspector_outlines_for(selected, display))

        # --- Zoom: show only part of the frame, at the same canvas size ---
        x0, y0, view_w, view_h = zoom_view(vid_w, vid_h, self._insp_zoom,
                                           self._insp_zoom_centre)
        if self._insp_zoom > 1.0:
            display = display[y0:y0 + view_h, x0:x0 + view_w]

        # --- Render video frame via PIL/ImageTk (fast direct pixel display) ---
        canvas_w = self._insp_video_canvas.winfo_width()
        canvas_h = self._insp_video_canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            # Canvas not yet laid out — use a sensible default
            canvas_w, canvas_h = 800, 600

        img_h, img_w = display.shape[:2]
        scale_fit = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale_fit))
        new_h = max(1, int(img_h * scale_fit))

        if _CV2_AVAILABLE and (new_w != img_w or new_h != img_h):
            display_small = _cv2.resize(display, (new_w, new_h),
                                        interpolation=_cv2.INTER_LINEAR)
        else:
            display_small = display

        # Where the view sits on the canvas, for turning a mouse position
        # back into a frame pixel.
        self._insp_view = (x0, y0, scale_fit, (canvas_w - new_w) / 2.0,
                           (canvas_h - new_h) / 2.0)

        if _PIL_AVAILABLE:
            pil_img = _PIL_Image.fromarray(display_small)
            photo = _PIL_ImageTk.PhotoImage(image=pil_img)
            # Replace reference BEFORE using canvas — this lets Python's refcount
            # immediately GC the old PhotoImage and release its Tk image resource.
            # Keeping self._insp_video_photo is also required so Tk doesn't lose
            # the image while the canvas item still references it by name.
            self._insp_video_photo = photo
            cx, cy = canvas_w // 2, canvas_h // 2
            if self._insp_video_canvas_item is None:
                self._insp_video_canvas_item = self._insp_video_canvas.create_image(
                    cx, cy, anchor="center", image=photo
                )
            else:
                self._insp_video_canvas.itemconfig(
                    self._insp_video_canvas_item, image=photo
                )
                self._insp_video_canvas.coords(
                    self._insp_video_canvas_item, cx, cy
                )

            # Frame/time label overlaid on the canvas
            title_text = f'Frame {frame_idx}  |  {time_s:.1f}s'
            if self._insp_zoom > 1.0:
                title_text += f'  |  zoom {self._insp_zoom:.1f}x'
            if self._insp_title_item is None:
                self._insp_title_item = self._insp_video_canvas.create_text(
                    cx, 14, text=title_text,
                    fill="white", font=("Arial", 11, "bold"),
                    anchor="center"
                )
            else:
                self._insp_video_canvas.itemconfig(
                    self._insp_title_item, text=title_text
                )
                self._insp_video_canvas.coords(self._insp_title_item, cx, 14)

        # --- Update time panel cursor (matplotlib blitting — only the cursor moves) ---
        if self._insp_canvas is None:
            return

        if self._insp_time_marker:
            current_min = time_s / 60.0
            self._insp_time_marker.set_xdata([current_min, current_min])
            if self._insp_bg_cache is None:
                # Dropped by a resize: draw the chart again at the new size.
                self._inspector_recapture_time_background()
            if self._insp_bg_cache is not None:
                self._insp_canvas.restore_region(self._insp_bg_cache)
            self._insp_ax_time.draw_artist(self._insp_time_marker)
            self._insp_canvas.blit(self._insp_fig.bbox)
