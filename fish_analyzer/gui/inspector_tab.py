"""
fish_analyzer/gui/inspector_tab.py
===================================
Video Inspector Tab - Unified frame-by-frame viewer with configurable overlays.

Consolidates the shoaling Frame View and bout Bout Viewer into a single
interactive inspector with overlay checkboxes for:
- Fish positions and trails
- Shoaling overlays (NND lines, convex hull, IID lines)
- Bout speed trace + heading change panel
- Video frame overlay
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

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


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

    @staticmethod
    def _make_collapsible(parent, title, initially_open=True):
        """Create a collapsible section with a toggle button.

        Returns (outer_frame, content_frame) — pack widgets into content_frame.
        """
        outer = tk.Frame(parent)
        outer.pack(fill="x", padx=5, pady=2)

        prefix = "\u25BC " if initially_open else "\u25B6 "
        btn = tk.Button(outer, text=prefix + title, font=("Arial", 9, "bold"),
                        relief="flat", anchor="w", bg="#e0e0e0",
                        activebackground="#d0d0d0")
        btn.pack(fill="x")

        content = tk.Frame(outer)
        if initially_open:
            content.pack(fill="x")

        def toggle():
            if content.winfo_viewable():
                content.pack_forget()
                btn.config(text="\u25B6 " + title)
            else:
                content.pack(fill="x")
                btn.config(text="\u25BC " + title)
            # Update scrollregion
            parent.event_generate("<Configure>")

        btn.config(command=toggle)
        return outer, content

    def _create_inspector_controls(self, parent):
        """Create all inspector controls in a scrollable panel."""
        canvas = tk.Canvas(parent, width=300, highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # --- File Selection ---
        file_frame = tk.LabelFrame(scroll_frame, text="File",
                                    font=("Arial", 10, "bold"))
        file_frame.pack(fill="x", padx=5, pady=5)

        self.inspector_file_var = tk.StringVar()
        self.inspector_file_dropdown = ttk.Combobox(
            file_frame, textvariable=self.inspector_file_var,
            state="readonly", width=25
        )
        self.inspector_file_dropdown.pack(fill="x", padx=5, pady=3)
        self.inspector_file_dropdown.bind(
            "<<ComboboxSelected>>", self._on_inspector_file_selected
        )

        # --- Frame Navigation ---
        nav_frame = tk.LabelFrame(scroll_frame, text="Navigation",
                                   font=("Arial", 10, "bold"))
        nav_frame.pack(fill="x", padx=5, pady=5)

        # Frame slider
        slider_row = tk.Frame(nav_frame)
        slider_row.pack(fill="x", padx=5, pady=3)
        tk.Label(slider_row, text="Frame:").pack(side="left")
        self.inspector_frame_var = tk.IntVar(value=0)
        self.inspector_frame_slider = tk.Scale(
            slider_row, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.inspector_frame_var,
            command=self._on_inspector_slider_change,
            length=180, showvalue=False
        )
        self.inspector_frame_slider.pack(side="left", padx=3, fill="x",
                                          expand=True)

        # Step size
        step_row = tk.Frame(nav_frame)
        step_row.pack(fill="x", padx=5, pady=2)
        tk.Label(step_row, text="Step:").pack(side="left")
        self.inspector_step_var = tk.StringVar(value="30")
        step_combo = ttk.Combobox(
            step_row, textvariable=self.inspector_step_var,
            values=["1", "5", "10", "30", "60", "150", "300"],
            width=5, state="readonly"
        )
        step_combo.pack(side="left", padx=3)
        step_combo.bind("<<ComboboxSelected>>", self._on_inspector_step_change)
        tk.Label(step_row, text="frames/step").pack(side="left")

        # Jump controls
        jump_row = tk.Frame(nav_frame)
        jump_row.pack(fill="x", padx=5, pady=3)
        tk.Label(jump_row, text="Jump to:").pack(side="left")

        tk.Label(jump_row, text="Frame:").pack(side="left", padx=(5, 2))
        self.inspector_jump_frame_var = tk.StringVar()
        entry_f = tk.Entry(jump_row, textvariable=self.inspector_jump_frame_var,
                           width=7)
        entry_f.pack(side="left")
        entry_f.bind("<Return>", lambda e: self._inspector_jump_to_input())

        tk.Label(jump_row, text="Time:").pack(side="left", padx=(5, 2))
        self.inspector_jump_time_var = tk.StringVar()
        entry_t = tk.Entry(jump_row, textvariable=self.inspector_jump_time_var,
                           width=7)
        entry_t.pack(side="left")
        entry_t.bind("<Return>", lambda e: self._inspector_jump_to_input())

        tk.Button(jump_row, text="Go", command=self._inspector_jump_to_input,
                  width=3).pack(side="left", padx=3)

        # Frame info
        self.inspector_info_label = tk.Label(
            nav_frame, text="Frame: -- | Time: --", font=("Arial", 9)
        )
        self.inspector_info_label.pack(anchor="w", padx=5, pady=2)

        # Playback
        play_row = tk.Frame(nav_frame)
        play_row.pack(fill="x", padx=5, pady=3)
        self.inspector_play_button = tk.Button(
            play_row, text="> Play",
            command=self._inspector_toggle_playback,
            bg="lightblue", width=8
        )
        self.inspector_play_button.pack(side="left", padx=2)

        tk.Label(play_row, text="Speed:").pack(side="left", padx=(10, 2))
        self.inspector_speed_var = tk.StringVar(value="1x")
        ttk.Combobox(
            play_row, textvariable=self.inspector_speed_var,
            values=["0.5x", "1x", "2x", "4x", "8x"],
            width=5, state="readonly"
        ).pack(side="left")

        # Single-frame step buttons
        step_btn_row = tk.Frame(nav_frame)
        step_btn_row.pack(fill="x", padx=5, pady=2)
        tk.Button(
            step_btn_row, text="\u25C0 -1 frame",
            command=self._inspector_step_back, width=10
        ).pack(side="left", padx=2)
        tk.Button(
            step_btn_row, text="+1 frame \u25B6",
            command=self._inspector_step_forward, width=10
        ).pack(side="left", padx=2)

        # --- Trails (collapsible) ---
        _, trail_frame = self._make_collapsible(scroll_frame, "Trails")

        tl_row = tk.Frame(trail_frame)
        tl_row.pack(fill="x", padx=5, pady=3)
        tk.Label(tl_row, text="Length:").pack(side="left")
        self.inspector_trail_var = tk.IntVar(value=0)
        tk.Scale(
            tl_row, from_=0, to=200, orient=tk.HORIZONTAL,
            variable=self.inspector_trail_var,
            command=lambda v: self._inspector_update_fast(),
            length=120, showvalue=True
        ).pack(side="left", padx=3, fill="x", expand=True)
        tk.Label(tl_row, text="frames").pack(side="left")

        # --- Drawing Style (collapsible, initially closed) ---
        _, style_frame = self._make_collapsible(scroll_frame, "Drawing Style",
                                                 initially_open=False)

        # Dot size
        dot_row = tk.Frame(style_frame)
        dot_row.pack(fill="x", padx=5, pady=2)
        tk.Label(dot_row, text="Dot size:").pack(side="left")
        self.inspector_dot_size_var = tk.IntVar(value=12)
        tk.Scale(
            dot_row, from_=5, to=30, orient=tk.HORIZONTAL,
            variable=self.inspector_dot_size_var,
            command=lambda v: self._inspector_update_fast(),
            length=100, showvalue=True
        ).pack(side="left", padx=3, fill="x", expand=True)
        tk.Label(dot_row, text="px").pack(side="left")

        # Trail opacity
        opacity_row = tk.Frame(style_frame)
        opacity_row.pack(fill="x", padx=5, pady=2)
        tk.Label(opacity_row, text="Trail opacity:").pack(side="left")
        self.inspector_trail_opacity_var = tk.DoubleVar(value=0.5)
        tk.Scale(
            opacity_row, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
            variable=self.inspector_trail_opacity_var, resolution=0.1,
            command=lambda v: self._inspector_update_fast(),
            length=100, showvalue=True
        ).pack(side="left", padx=3, fill="x", expand=True)

        # Trail width
        width_row = tk.Frame(style_frame)
        width_row.pack(fill="x", padx=5, pady=2)
        tk.Label(width_row, text="Trail width:").pack(side="left")
        self.inspector_trail_width_var = tk.DoubleVar(value=1.0)
        tk.Scale(
            width_row, from_=0.5, to=3.0, orient=tk.HORIZONTAL,
            variable=self.inspector_trail_width_var, resolution=0.5,
            command=lambda v: self._inspector_update_fast(),
            length=100, showvalue=True
        ).pack(side="left", padx=3, fill="x", expand=True)

        # --- Video Overlay (collapsible) ---
        _, video_frame = self._make_collapsible(scroll_frame, "Video Overlay",
                                                 initially_open=False)

        self.inspector_video_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            video_frame, text="Use video frames",
            variable=self.inspector_video_var,
            command=self._inspector_on_video_toggle
        ).pack(anchor="w", padx=5)

        btn_row = tk.Frame(video_frame)
        btn_row.pack(fill="x", padx=5, pady=2)
        tk.Button(btn_row, text="Browse Video...",
                  command=self._inspector_browse_video,
                  width=15).pack(side="left")

        self.inspector_video_status = tk.Label(
            video_frame, text="No video loaded",
            font=("Arial", 8), fg="gray"
        )
        self.inspector_video_status.pack(anchor="w", padx=5, pady=2)

        qual_row = tk.Frame(video_frame)
        qual_row.pack(fill="x", padx=5, pady=2)
        tk.Label(qual_row, text="Quality:").pack(side="left")
        self.inspector_quality_var = tk.StringVar(value="high")
        ttk.Combobox(
            qual_row, textvariable=self.inspector_quality_var,
            values=["high", "medium", "low"],
            width=8, state="readonly"
        ).pack(side="left", padx=3)

        # --- Spatial Overlays (collapsible) ---
        _, overlay_frame = self._make_collapsible(scroll_frame,
                                                    "Spatial Overlays")

        self.inspector_show_positions_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            overlay_frame, text="Fish positions",
            variable=self.inspector_show_positions_var,
            command=self._inspector_on_overlay_change
        ).pack(anchor="w", padx=5)

        self.inspector_show_nnd_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            overlay_frame, text="NND lines",
            variable=self.inspector_show_nnd_var,
            command=self._inspector_on_overlay_change
        ).pack(anchor="w", padx=5)

        self.inspector_show_hull_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            overlay_frame, text="Convex hull",
            variable=self.inspector_show_hull_var,
            command=self._inspector_on_overlay_change
        ).pack(anchor="w", padx=5)

        iid_row = tk.Frame(overlay_frame)
        iid_row.pack(fill="x", padx=5)
        self.inspector_show_iid_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            iid_row, text="IID lines",
            variable=self.inspector_show_iid_var,
            command=self._inspector_on_overlay_change
        ).pack(side="left")
        tk.Label(iid_row, text="Focus:").pack(side="left", padx=(10, 2))
        self.inspector_iid_focus_var = tk.StringVar(value="0")
        self.inspector_iid_focus_combo = ttk.Combobox(
            iid_row, textvariable=self.inspector_iid_focus_var,
            state="readonly", width=4
        )
        self.inspector_iid_focus_combo.pack(side="left")

        # --- Bout Overlay (collapsible) ---
        _, bout_frame = self._make_collapsible(scroll_frame, "Bout Overlay")

        self.inspector_show_bouts_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            bout_frame, text="Highlight bout fish + show traces",
            variable=self.inspector_show_bouts_var,
            command=self._inspector_on_bout_toggle
        ).pack(anchor="w", padx=5)

        self.bout_overlay_status = tk.Label(bout_frame, text="Run Bout Analysis first",
                                             font=("Arial", 8), fg="gray")
        self.bout_overlay_status.pack(anchor="w", padx=5)

        fish_row = tk.Frame(bout_frame)
        fish_row.pack(fill="x", padx=5, pady=2)
        tk.Label(fish_row, text="Fish:").pack(side="left")
        self.inspector_bout_fish_var = tk.StringVar()
        self.inspector_bout_fish_combo = ttk.Combobox(
            fish_row, textvariable=self.inspector_bout_fish_var,
            state="readonly", width=5
        )
        self.inspector_bout_fish_combo.pack(side="left", padx=3)
        self.inspector_bout_fish_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self._inspector_rebuild_needed()
        )

        window_row = tk.Frame(bout_frame)
        window_row.pack(fill="x", padx=5, pady=2)
        tk.Label(window_row, text="Window:").pack(side="left")
        self.inspector_bout_window_var = tk.IntVar(value=10)
        tk.Scale(
            window_row, from_=2, to=60, orient=tk.HORIZONTAL,
            variable=self.inspector_bout_window_var,
            length=100, showvalue=True
        ).pack(side="left", padx=3)
        tk.Label(window_row, text="sec").pack(side="left")

        # --- Time Panel Mode (collapsible) ---
        _, time_frame = self._make_collapsible(scroll_frame, "Time Panel")

        self.inspector_time_mode_var = tk.StringVar(value="none")
        modes = [
            ("None", "none"),
            ("NND", "nnd"),
            ("IID", "iid"),
            ("Hull Area", "hull"),
            ("Bout Speed + Heading", "bout"),
        ]
        for text, val in modes:
            tk.Radiobutton(
                time_frame, text=text,
                variable=self.inspector_time_mode_var, value=val,
                command=self._inspector_rebuild_needed
            ).pack(anchor="w", padx=5)

    # =========================================================================
    # DISPLAY AREA
    # =========================================================================

    def _create_inspector_display(self, parent):
        """Create the inspector display area."""
        self.inspector_plot_frame = tk.Frame(parent, bg="white")
        self.inspector_plot_frame.pack(fill="both", expand=True)

        tk.Label(
            self.inspector_plot_frame,
            text="Load files in Data tab, then select a file to inspect",
            font=("Arial", 11), fg="gray"
        ).pack(expand=True)

        # Inspector figure state
        self._insp_fig = None
        self._insp_canvas = None
        self._insp_ax_main = None
        self._insp_ax_time = None
        self._insp_ax_time2 = None
        self._insp_time_marker = None
        self._insp_bout_cursor_speed = None
        self._insp_bout_cursor_heading = None
        self._insp_dynamic_artists = []
        self._insp_cached_file = None
        self._insp_cached_overlays = None
        self._insp_cached_time_mode = None
        self._insp_cached_bout_fish = None
        self._insp_video_bg_artist = None
        self._insp_cached_background = None
        self._insp_cached_width_bl = None
        self._insp_cached_height_bl = None
        self._insp_bg_cache = None
        self._insp_composite_artist = None
        self._insp_ax_zoom = None
        self._insp_zoom_artist = None

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
        n_frames = loaded.n_frames

        # Update frame slider range
        step = int(self.inspector_step_var.get())
        max_steps = max(0, (n_frames - 1) // step)
        self.inspector_frame_slider.configure(to=max_steps)
        self.inspector_frame_var.set(0)

        # Update fish dropdowns
        fish_opts = [str(i) for i in range(loaded.n_fish)]
        self.inspector_iid_focus_combo['values'] = fish_opts
        self.inspector_bout_fish_combo['values'] = fish_opts
        if fish_opts:
            self.inspector_iid_focus_combo.current(0)
            self.inspector_bout_fish_combo.current(0)

        # Force rebuild
        self._insp_fig = None
        self._update_inspector_info()
        self._inspector_update_fast()

    def _on_inspector_step_change(self, event=None):
        """Recalculate slider range when step size changes."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            return
        loaded = self.loaded_files[selected]
        step = int(self.inspector_step_var.get())
        max_steps = max(0, (loaded.n_frames - 1) // step)
        self.inspector_frame_slider.configure(to=max_steps)
        self.inspector_frame_var.set(0)
        self._update_inspector_info()
        self._insp_fig = None
        self._inspector_update_fast()

    # =========================================================================
    # FRAME INFO
    # =========================================================================

    def _get_inspector_frame_idx(self):
        """Get the actual frame index from the slider position."""
        step = int(self.inspector_step_var.get())
        return self.inspector_frame_var.get() * step

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

        info = f"Frame: {frame_idx} | Time: {time_s:.1f}s"

        # Add shoaling metrics if available (nearest sample)
        if loaded.shoaling_results:
            results = loaded.shoaling_results
            diffs = np.abs(results.frame_indices - frame_idx)
            nearest = int(np.argmin(diffs))
            nnd = results.mean_nnd_per_sample[nearest]
            iid = results.mean_iid_per_sample[nearest]
            hull = results.convex_hull_area_per_sample[nearest]
            info += f" | NND: {nnd:.2f} | IID: {iid:.2f} | Hull: {hull:.1f}"

        self.inspector_info_label.config(text=info)
        self.inspector_jump_frame_var.set(str(frame_idx))
        self.inspector_jump_time_var.set(f"{time_s:.1f}")

    # =========================================================================
    # NAVIGATION
    # =========================================================================

    def _on_inspector_slider_change(self, value):
        """Called when frame slider changes."""
        self._update_inspector_info()
        self._inspector_update_fast()

    def _inspector_jump_to_input(self):
        """Jump to specified frame or time."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            return

        loaded = self.loaded_files[selected]
        fps = loaded.calibration.frame_rate
        step = int(self.inspector_step_var.get())

        frame_str = self.inspector_jump_frame_var.get().strip()
        time_str = self.inspector_jump_time_var.get().strip()

        target_frame = None
        if frame_str:
            try:
                target_frame = int(frame_str)
            except ValueError:
                pass

        if target_frame is None and time_str:
            try:
                target_frame = int(float(time_str) * fps)
            except ValueError:
                pass

        if target_frame is not None:
            max_step = int(self.inspector_frame_slider.cget('to'))
            target_step = max(0, min(target_frame // step, max_step))
            self.inspector_frame_var.set(target_step)
            self._update_inspector_info()
            self._inspector_update_fast()

    def _inspector_step_back(self):
        """Step back by exactly 1 frame."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            return
        loaded = self.loaded_files[selected]
        step = int(self.inspector_step_var.get())
        frame_idx = self._get_inspector_frame_idx()
        target = max(0, frame_idx - 1)
        target_step = target // step
        self.inspector_frame_var.set(target_step)
        self._update_inspector_info()
        self._inspector_update_fast()

    def _inspector_step_forward(self):
        """Step forward by exactly 1 frame."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            return
        loaded = self.loaded_files[selected]
        step = int(self.inspector_step_var.get())
        frame_idx = self._get_inspector_frame_idx()
        target = min(loaded.n_frames - 1, frame_idx + 1)
        target_step = target // step
        self.inspector_frame_var.set(target_step)
        self._update_inspector_info()
        self._inspector_update_fast()

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

        # Pre-cache video frames
        selected = self.inspector_file_var.get()
        if selected and selected in self.video_readers:
            current_frame = self._get_inspector_frame_idx()
            step = int(self.inspector_step_var.get())
            reader = self.video_readers[selected]
            n_frames = self.loaded_files[selected].n_frames
            for i in range(50):
                f = current_frame + i * step
                if f < n_frames:
                    reader.read_frame(f)

        self._inspector_animate_step()

    def _inspector_stop_playback(self):
        """Stop animation."""
        self.animation_running = False
        self.inspector_play_button.config(text="> Play", bg="lightblue")
        if self.animation_after_id:
            self.root.after_cancel(self.animation_after_id)
            self.animation_after_id = None

    def _inspector_animate_step(self):
        """Advance one step in animation."""
        if not self.animation_running:
            return

        current = self.inspector_frame_var.get()
        max_val = int(self.inspector_frame_slider.cget('to'))

        if current < max_val:
            self.inspector_frame_var.set(current + 1)
        else:
            self.inspector_frame_var.set(0)

        self._update_inspector_info()
        self._inspector_update_fast()

        # Calculate delay based on real-time playback
        step = int(self.inspector_step_var.get())
        selected = self.inspector_file_var.get()
        delay_ms = 100
        if selected and selected in self.loaded_files:
            fps = self.loaded_files[selected].calibration.frame_rate
            real_time = step / fps

            speed_str = self.inspector_speed_var.get()
            try:
                speed_mult = float(speed_str.replace('x', ''))
            except ValueError:
                speed_mult = 1.0

            delay_ms = max(16, int((real_time / speed_mult) * 1000))

        self.animation_after_id = self.root.after(
            delay_ms, self._inspector_animate_step
        )

    # =========================================================================
    # OVERLAY / REBUILD TRIGGERS
    # =========================================================================

    def _inspector_on_overlay_change(self):
        """Called when any overlay checkbox changes.
        With CV2 compositing, overlays are drawn each frame — no rebuild needed.
        """
        self._inspector_update_fast()

    def _inspector_on_bout_toggle(self):
        """Called when bout overlay checkbox is toggled."""
        if self.inspector_show_bouts_var.get():
            # Auto-switch time panel to bout mode
            self.inspector_time_mode_var.set("bout")
            # Check if bout data exists
            selected = self.inspector_file_var.get()
            bout_results = getattr(self, 'bout_results', {})
            if selected and selected in bout_results:
                self.bout_overlay_status.config(
                    text=f"Bout data available", fg="green")
            else:
                self.bout_overlay_status.config(
                    text="No bout data — run Bout Analysis first", fg="red")
        self._insp_fig = None
        self._inspector_update_fast()

    def _inspector_rebuild_needed(self, event=None):
        """Force a figure rebuild on next update."""
        self._insp_fig = None
        self._inspector_update_fast()

    # =========================================================================
    # VIDEO
    # =========================================================================

    def _inspector_browse_video(self):
        """Browse for a video file."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            messagebox.showwarning("No File", "Select a file first.")
            return

        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        try:
            from ..video_utils import VideoFrameReader, CV2_AVAILABLE

            if not CV2_AVAILABLE:
                messagebox.showerror(
                    "OpenCV Required",
                    "OpenCV is required for video frame reading.\n\n"
                    "Install with: pip install opencv-python"
                )
                return

            if selected in self.video_readers:
                self.video_readers[selected].close()

            reader = VideoFrameReader(Path(path))
            self.video_readers[selected] = reader
            self.loaded_files[selected].video_file_path = Path(path)

            self.inspector_video_status.config(
                text=f"Loaded: {reader.total_frames} frames, "
                     f"{reader.width}x{reader.height}",
                fg="green"
            )

            self.inspector_video_var.set(True)
            self._insp_fig = None
            self._inspector_update_fast()

            messagebox.showinfo(
                "Video Loaded",
                f"Video loaded successfully!\n\n"
                f"Resolution: {reader.width}x{reader.height}\n"
                f"Frames: {reader.total_frames}\n"
                f"FPS: {reader.fps:.1f}"
            )

        except Exception as e:
            messagebox.showerror("Video Error",
                                 f"Could not open video:\n{e}")

    def _inspector_on_video_toggle(self):
        """Handle video toggle checkbox."""
        if self.inspector_video_var.get():
            selected = self.inspector_file_var.get()
            if selected not in self.video_readers:
                messagebox.showinfo(
                    "No Video",
                    "Please browse for and load a video file first.\n\n"
                    "Click 'Browse Video...' to select your video file."
                )
                self.inspector_video_var.set(False)
                return

        self._insp_fig = None
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
        bout_fish = self.inspector_bout_fish_var.get()

        # Video toggle needs rebuild (changes background compositing source)
        video_on = self.inspector_video_var.get()

        rebuild = (
            self._insp_fig is None
            or self._insp_cached_file != selected
            or self._insp_cached_time_mode != time_mode
            or self._insp_cached_overlays != video_on
            or (time_mode == 'bout'
                and self._insp_cached_bout_fish != bout_fish)
        )

        if rebuild:
            self._inspector_rebuild_figure(
                selected, loaded, frame_idx, time_mode
            )
            self._insp_cached_file = selected
            self._insp_cached_time_mode = time_mode
            self._insp_cached_overlays = video_on
            self._insp_cached_bout_fish = bout_fish

        self._inspector_update_dynamic(
            selected, loaded, frame_idx, time_mode
        )

    # =========================================================================
    # FIGURE REBUILD (expensive, only when layout changes)
    # =========================================================================

    def _inspector_rebuild_figure(self, selected, loaded, frame_idx,
                                   time_mode):
        """Rebuild the entire figure from scratch.

        Uses a persistent imshow artist for the arena — overlays are drawn
        directly onto the frame (via CV2 when available) so only a single
        set_data() call is needed per frame instead of dozens of scatter/plot
        artists.
        """
        for widget in self.inspector_plot_frame.winfo_children():
            widget.destroy()

        show_time = time_mode != "none"
        show_bout_heading = time_mode == "bout"

        # Determine layout — bout mode adds a crop-zoom panel to the right
        if show_bout_heading:
            self._insp_fig = Figure(figsize=(12, 10), dpi=100)
            self._insp_ax_main = self._insp_fig.add_axes(
                [0.02, 0.38, 0.62, 0.58]
            )
            self._insp_ax_zoom = self._insp_fig.add_axes(
                [0.66, 0.42, 0.32, 0.54]
            )
            self._insp_ax_time = self._insp_fig.add_axes(
                [0.06, 0.05, 0.90, 0.28]
            )
            self._insp_ax_time2 = None  # Combined into single panel now
        elif show_time:
            self._insp_fig = Figure(figsize=(10, 8), dpi=100)
            self._insp_ax_main = self._insp_fig.add_axes(
                [0.02, 0.22, 0.96, 0.74]
            )
            self._insp_ax_time = self._insp_fig.add_axes(
                [0.08, 0.05, 0.88, 0.15]
            )
            self._insp_ax_time2 = None
            self._insp_ax_zoom = None
        else:
            self._insp_fig = Figure(figsize=(10, 8), dpi=100)
            self._insp_ax_main = self._insp_fig.add_axes(
                [0.02, 0.02, 0.96, 0.94]
            )
            self._insp_ax_time = None
            self._insp_ax_time2 = None
            self._insp_ax_zoom = None

        ax = self._insp_ax_main

        # Coordinate system
        pixels_to_bl = 1.0 / loaded.metadata.body_length
        width_bl = loaded.metadata.video_width * pixels_to_bl
        height_bl = loaded.metadata.video_height * pixels_to_bl
        self._insp_cached_width_bl = width_bl
        self._insp_cached_height_bl = height_bl

        # Load background image for compositing (don't display it separately)
        self._insp_cached_background = None
        self._insp_video_bg_artist = None
        use_video = self.inspector_video_var.get()

        if (not use_video and loaded.background_image_path
                and loaded.background_image_path.exists()):
            try:
                bg = plt.imread(str(loaded.background_image_path))
                # Ensure uint8 RGB for compositing
                if bg.dtype != np.uint8:
                    bg = (np.clip(bg, 0, 1) * 255).astype(np.uint8)
                if len(bg.shape) == 2:
                    bg = np.stack([bg, bg, bg], axis=-1)
                elif bg.shape[2] == 4:
                    bg = bg[:, :, :3]  # Drop alpha channel
                self._insp_cached_background = bg
            except Exception:
                pass

        # Create persistent imshow for composited frames
        vid_h = loaded.metadata.video_height
        vid_w = loaded.metadata.video_width
        blank = np.ones((vid_h, vid_w, 3), dtype=np.uint8) * 180
        self._insp_composite_artist = ax.imshow(
            blank, extent=[0, width_bl, 0, height_bl],
            aspect='auto', origin='upper', zorder=0,
            interpolation='bilinear'
        )

        ax.set_xlim(0, width_bl)
        ax.set_ylim(0, height_bl)
        ax.set_xlabel('X (BL)', fontsize=9)
        ax.set_ylabel('Y (BL)', fontsize=9)

        # Zoom panel (bout mode)
        if self._insp_ax_zoom is not None:
            zoom_blank = np.ones((200, 200, 3), dtype=np.uint8) * 180
            self._insp_zoom_artist = self._insp_ax_zoom.imshow(
                zoom_blank, aspect='equal', interpolation='bilinear'
            )
            self._insp_ax_zoom.set_title('Fish Zoom', fontsize=10,
                                          fontweight='bold')
            self._insp_ax_zoom.axis('off')
        else:
            self._insp_zoom_artist = None

        # --- Time panel: shoaling metrics ---
        self._insp_time_marker = None
        self._insp_bout_cursor_speed = None
        self._insp_bout_cursor_heading = None

        if time_mode in ('nnd', 'iid', 'hull') and loaded.shoaling_results:
            results = loaded.shoaling_results
            time_min = results.timestamps / 60.0
            ax_t = self._insp_ax_time

            if time_mode == 'nnd':
                ax_t.plot(time_min, results.mean_nnd_per_sample,
                          'b-', lw=1, alpha=0.7)
                ax_t.set_ylabel('NND (BL)', fontsize=8)
            elif time_mode == 'iid':
                ax_t.plot(time_min, results.mean_iid_per_sample,
                          'g-', lw=1, alpha=0.7)
                ax_t.set_ylabel('IID (BL)', fontsize=8)
            else:
                ax_t.plot(time_min, results.convex_hull_area_per_sample,
                          'r-', lw=1, alpha=0.7)
                ax_t.set_ylabel('Hull (BL\u00b2)', fontsize=8)

            ax_t.set_xlabel('Time (min)', fontsize=8)
            ax_t.set_xlim(time_min[0], time_min[-1])
            ax_t.grid(True, alpha=0.3)
            self._insp_time_marker = ax_t.axvline(
                x=0, color='red', lw=2, linestyle='--'
            )

        elif time_mode in ('nnd', 'iid', 'hull') and not loaded.shoaling_results:
            ax_t = self._insp_ax_time
            if ax_t:
                ax_t.text(
                    0.5, 0.5,
                    "Run Shoaling Analysis first",
                    transform=ax_t.transAxes, ha='center', va='center',
                    fontsize=10, color='gray'
                )

        elif time_mode == 'bout':
            self._inspector_build_bout_panels(selected, loaded)

        # Create canvas
        self._insp_canvas = FigureCanvasTkAgg(
            self._insp_fig, master=self.inspector_plot_frame
        )
        self._insp_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._insp_dynamic_artists = []

        # Draw static content and cache background for blitting
        self._insp_canvas.draw()
        try:
            self._insp_bg_cache = self._insp_canvas.copy_from_bbox(
                self._insp_fig.bbox
            )
        except Exception:
            self._insp_bg_cache = None

    def _inspector_build_bout_panels(self, selected, loaded):
        """Build the bout speed trace + heading panels (full recording)."""
        ax_speed = self._insp_ax_time
        ax_heading = self._insp_ax_time2

        if ax_speed is None:
            return

        # Check for bout results
        bout_results = getattr(self, 'bout_results', {})
        if selected not in bout_results:
            ax_speed.text(
                0.5, 0.5, "Run Bout Analysis first",
                transform=ax_speed.transAxes, ha='center', va='center',
                fontsize=10, color='gray'
            )
            return

        try:
            fish_id = int(self.inspector_bout_fish_var.get())
        except (ValueError, TypeError):
            return

        # Find bout result for this fish
        bout_result = None
        for r in bout_results[selected]:
            if r.fish_id == fish_id:
                bout_result = r
                break
        if bout_result is None:
            ax_speed.text(
                0.5, 0.5, f"No bout data for Fish {fish_id}",
                transform=ax_speed.transAxes, ha='center', va='center',
                fontsize=10, color='gray'
            )
            return

        fps = loaded.calibration.frame_rate
        scale = loaded.calibration.scale_factor
        n_frames = loaded.n_frames

        # Compute speed trace for this fish
        raw = loaded.trajectories[:, fish_id, :]
        x_pos = raw[:, 0] * scale
        y_pos = (loaded.metadata.video_height - raw[:, 1]) * scale
        dx = np.diff(x_pos)
        dy = np.diff(y_pos)
        speed = np.sqrt(dx ** 2 + dy ** 2) * fps
        speed[np.isnan(speed)] = 0.0
        time_s = np.arange(len(speed)) / fps

        # Get threshold
        threshold = 0.5
        if hasattr(self, 'bout_threshold_var'):
            try:
                threshold = float(self.bout_threshold_var.get())
            except ValueError:
                pass

        unit = loaded.calibration.unit_name

        # --- Combined speed + heading panel ---
        # Top: speed trace with colored bout bands
        # Bottom row: |heading| bars going UP, colored by direction
        # This saves space and avoids the confusing up/down heading display

        ax_speed.plot(time_s, speed, color='black', lw=0.5, alpha=0.6)
        ax_speed.axhline(
            threshold, color='orange', linestyle='--', lw=1, alpha=0.7,
            label=f'Threshold ({threshold} {unit}/s)'
        )

        for bout in bout_result.bouts:
            t0 = bout.start_frame / fps
            t1 = bout.end_frame / fps
            hc = bout.heading_change_deg
            if abs(hc) > 5:
                color = '#1976d2' if hc > 0 else '#d32f2f'
            else:
                color = '#bdbdbd'
            ax_speed.axvspan(t0, t1, color=color, alpha=0.25, zorder=0)

        # Add |heading| as colored bars below the speed trace (negative y space)
        # Scale heading bars to sit below the x-axis line
        if len(speed) > 0:
            y_cap = max(np.percentile(speed, 99) * 1.2, threshold * 3)
        else:
            y_cap = threshold * 3

        # Heading bars: plotted as negative values (below zero) for visual separation
        max_abs_hc = 30.0
        if bout_result.bouts:
            max_abs_hc = max(
                max(abs(b.heading_change_deg) for b in bout_result.bouts),
                30.0
            )
        heading_scale = y_cap * 0.35 / max_abs_hc  # Scale heading to use ~35% of plot

        for bout in bout_result.bouts:
            t_mid = (bout.start_frame + bout.end_frame) / 2 / fps
            hc = bout.heading_change_deg
            abs_hc = abs(hc)
            if abs_hc > 5:
                c = '#1976d2' if hc > 0 else '#d32f2f'
            else:
                c = '#757575'
            bar_h = abs_hc * heading_scale
            ax_speed.bar(
                t_mid, -bar_h, bottom=0,
                width=max(0.03, bout.duration_s),
                color=c, alpha=0.7, edgecolor='none', zorder=1
            )

        ax_speed.axhline(0, color='black', lw=0.5, zorder=2)
        ax_speed.set_ylabel(f"Speed ({unit}/s)", fontsize=8)
        ax_speed.set_xlabel("Time (s)", fontsize=8)
        ax_speed.set_title(
            f"Fish {fish_id} \u2014 {len(bout_result.bouts)} bouts "
            f"(\u25A0 blue=left  \u25A0 red=right  \u25A0 gray=straight)",
            fontsize=9
        )
        ax_speed.legend(fontsize=7, loc='upper right')
        ax_speed.set_ylim(-y_cap * 0.4, y_cap)

        # Add secondary y-axis on right side showing degree scale for heading bars
        ax_deg = ax_speed.twinx()
        ax_deg.set_ylim(-max_abs_hc, y_cap / heading_scale if heading_scale > 0 else 180)
        ax_deg.set_ylabel("|Turn| (\u00b0)", fontsize=8, color='gray')
        ax_deg.tick_params(axis='y', colors='gray', labelsize=7)
        # Only show ticks in the negative region (heading bars)
        import matplotlib.ticker as mticker
        ax_deg.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax_deg.spines['right'].set_color('gray')
        ax_deg.spines['right'].set_alpha(0.5)

        # Cursor line
        self._insp_bout_cursor_speed = ax_speed.axvline(
            x=0, color='green', lw=1.5, alpha=0.8
        )

    # =========================================================================
    # DYNAMIC UPDATE (fast, every frame/step)
    # =========================================================================

    def _inspector_update_dynamic(self, selected, loaded, frame_idx,
                                   time_mode):
        """Update the display — composites overlays onto the frame.

        Uses CV2 drawing when available (much faster), falls back to
        matplotlib artists otherwise.
        """
        ax = self._insp_ax_main
        if ax is None or self._insp_composite_artist is None:
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

        # Save raw frame for zoom (before overlays)
        self._insp_zoom_raw_frame = display.copy()

        # --- Get fish positions in pixel coordinates ---
        raw_pos = loaded.trajectories[frame_idx].copy()  # (n_fish, 2) pixels
        tab10 = plt.cm.tab10(np.linspace(0, 1, n_fish))

        # Reading settings
        dot_radius = self.inspector_dot_size_var.get()
        trail_len = self.inspector_trail_var.get()
        trail_opacity = self.inspector_trail_opacity_var.get()
        trail_width_setting = self.inspector_trail_width_var.get()

        if _CV2_AVAILABLE:
            self._inspector_draw_cv2(
                display, loaded, raw_pos, frame_idx, n_fish, tab10,
                dot_radius, trail_len, trail_opacity, trail_width_setting,
                vid_h, vid_w, scale
            )
        else:
            self._inspector_draw_numpy(
                display, raw_pos, n_fish, tab10, dot_radius
            )

        # --- Update persistent imshow ---
        self._insp_composite_artist.set_data(display)

        # --- Title ---
        ax.set_title(
            f'Frame {frame_idx} | {time_s:.1f}s',
            fontsize=12, fontweight='bold'
        )

        # --- Crop zoom for bout mode (uses raw frame, not composited) ---
        if (self._insp_zoom_artist is not None
                and self.inspector_show_bouts_var.get()):
            try:
                bout_fish = int(self.inspector_bout_fish_var.get())
                if bout_fish < n_fish and not np.isnan(raw_pos[bout_fish, 0]):
                    self._inspector_update_zoom(
                        self._insp_zoom_raw_frame, raw_pos, bout_fish,
                        vid_h, vid_w
                    )
            except (ValueError, TypeError):
                pass

        # --- Update time panel cursor ---
        if time_mode in ('nnd', 'iid', 'hull') and self._insp_time_marker:
            current_min = time_s / 60.0
            self._insp_time_marker.set_xdata([current_min, current_min])

        elif time_mode == 'bout':
            center_s = time_s
            window_s = self.inspector_bout_window_var.get()
            total_s = loaded.n_frames / fps
            t_start = max(0, center_s - window_s / 2)
            t_end = t_start + window_s
            if t_end > total_s:
                t_end = total_s
                t_start = max(0, t_end - window_s)

            if self._insp_ax_time:
                self._insp_ax_time.set_xlim(t_start, t_end)

            if self._insp_bout_cursor_speed:
                self._insp_bout_cursor_speed.set_xdata(
                    [center_s, center_s]
                )

        # --- Render ---
        # Use blitting when possible (not bout mode which scrolls xlim)
        use_blitting = (
            self._insp_bg_cache is not None
            and time_mode != 'bout'
        )

        if use_blitting:
            try:
                self._insp_canvas.restore_region(self._insp_bg_cache)
                ax.draw_artist(self._insp_composite_artist)
                ax.draw_artist(ax.title)
                if self._insp_zoom_artist is not None:
                    self._insp_ax_zoom.draw_artist(self._insp_zoom_artist)
                if self._insp_time_marker:
                    self._insp_ax_time.draw_artist(self._insp_time_marker)
                self._insp_canvas.blit(self._insp_fig.bbox)
            except Exception:
                self._insp_canvas.draw_idle()
        else:
            self._insp_canvas.draw_idle()

    # =========================================================================
    # CV2 DRAWING (fast path — draws overlays directly on the frame)
    # =========================================================================

    @staticmethod
    def _rgba_to_bgr(rgba):
        """Convert matplotlib RGBA (0-1) to OpenCV BGR (0-255)."""
        return (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))

    @staticmethod
    def _rgba_to_rgb_uint8(rgba):
        """Convert matplotlib RGBA (0-1) to RGB (0-255) tuple."""
        return (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))

    def _inspector_draw_cv2(self, display, loaded, raw_pos, frame_idx,
                             n_fish, tab10, dot_radius, trail_len,
                             trail_opacity, trail_width_setting,
                             vid_h, vid_w, scale):
        """Draw all overlays directly onto the frame using OpenCV.

        This is 10-50x faster than creating matplotlib scatter/line/text
        artists because cv2 drawing operates directly on the numpy array.
        """
        # --- Trails ---
        if trail_len > 0:
            start = max(0, frame_idx - trail_len)
            end = frame_idx + 1
            for i in range(n_fish):
                traj = loaded.trajectories[start:end, i, :]
                valid = ~np.isnan(traj[:, 0])
                if np.sum(valid) < 2:
                    continue
                color_bgr = self._rgba_to_bgr(tab10[i])
                pts = traj[valid].astype(np.int32)
                thickness = max(1, int(trail_width_setting * 2))
                # Create overlay for alpha blending
                overlay = display.copy()
                _cv2.polylines(overlay, [pts], False, color_bgr, thickness,
                              lineType=_cv2.LINE_AA)
                alpha = min(1.0, trail_opacity)
                _cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0,
                                display)

        # --- NND lines (white lines with black-outlined text) ---
        if self.inspector_show_nnd_var.get():
            for i in range(n_fish):
                if np.isnan(raw_pos[i, 0]):
                    continue
                min_dist_px = np.inf
                nn_idx = -1
                for j in range(n_fish):
                    if i == j or np.isnan(raw_pos[j, 0]):
                        continue
                    d = np.sqrt((raw_pos[i, 0] - raw_pos[j, 0]) ** 2
                                + (raw_pos[i, 1] - raw_pos[j, 1]) ** 2)
                    if d < min_dist_px:
                        min_dist_px = d
                        nn_idx = j
                if nn_idx >= 0:
                    p1 = (int(raw_pos[i, 0]), int(raw_pos[i, 1]))
                    p2 = (int(raw_pos[nn_idx, 0]), int(raw_pos[nn_idx, 1]))
                    _cv2.line(display, p1, p2, (255, 255, 255), 2,
                             lineType=_cv2.LINE_AA)
                    dist_bl = min_dist_px * scale
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    # Black outline then white text for readability
                    _cv2.putText(display, f'{dist_bl:.1f}', mid,
                                _cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 3, _cv2.LINE_AA)
                    _cv2.putText(display, f'{dist_bl:.1f}', mid,
                                _cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, _cv2.LINE_AA)

        # --- Convex hull ---
        if self.inspector_show_hull_var.get():
            valid_pts = []
            for i in range(n_fish):
                if not np.isnan(raw_pos[i, 0]):
                    valid_pts.append(raw_pos[i])
            if len(valid_pts) >= 3:
                pts_arr = np.array(valid_pts, dtype=np.float32)
                hull = _cv2.convexHull(pts_arr.astype(np.int32))
                overlay = display.copy()
                _cv2.fillPoly(overlay, [hull], (100, 200, 100))
                _cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
                _cv2.polylines(display, [hull], True, (0, 180, 0), 2,
                              lineType=_cv2.LINE_AA)

        # --- IID lines (magenta lines with outlined text) ---
        if self.inspector_show_iid_var.get():
            try:
                focus = int(self.inspector_iid_focus_var.get())
            except (ValueError, TypeError):
                focus = 0
            if focus < n_fish and not np.isnan(raw_pos[focus, 0]):
                pf = (int(raw_pos[focus, 0]), int(raw_pos[focus, 1]))
                for j in range(n_fish):
                    if j == focus or np.isnan(raw_pos[j, 0]):
                        continue
                    pj = (int(raw_pos[j, 0]), int(raw_pos[j, 1]))
                    _cv2.line(display, pf, pj, (255, 100, 255), 2,
                             lineType=_cv2.LINE_AA)
                    d_bl = np.sqrt((raw_pos[focus, 0] - raw_pos[j, 0]) ** 2
                                   + (raw_pos[focus, 1] - raw_pos[j, 1]) ** 2
                                   ) * scale
                    mid = ((pf[0] + pj[0]) // 2, (pf[1] + pj[1]) // 2)
                    _cv2.putText(display, f'{d_bl:.1f}', mid,
                                _cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 3, _cv2.LINE_AA)
                    _cv2.putText(display, f'{d_bl:.1f}', mid,
                                _cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 100, 255), 1, _cv2.LINE_AA)

        # --- Fish positions ---
        if self.inspector_show_positions_var.get():
            for i in range(n_fish):
                if np.isnan(raw_pos[i, 0]):
                    continue
                px, py = int(raw_pos[i, 0]), int(raw_pos[i, 1])
                color_bgr = self._rgba_to_bgr(tab10[i])
                # Filled circle
                _cv2.circle(display, (px, py), dot_radius, color_bgr, -1,
                           lineType=_cv2.LINE_AA)
                # Black border
                _cv2.circle(display, (px, py), dot_radius, (0, 0, 0), 2,
                           lineType=_cv2.LINE_AA)
                # Fish number (white text)
                font_scale = max(0.3, dot_radius / 20.0)
                text = str(i)
                (tw, th), _ = _cv2.getTextSize(
                    text, _cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                _cv2.putText(display, text,
                            (px - tw // 2, py + th // 2),
                            _cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), 1, _cv2.LINE_AA)

        # --- Bout fish highlight ring ---
        if self.inspector_show_bouts_var.get():
            try:
                bout_fish = int(self.inspector_bout_fish_var.get())
                if (bout_fish < n_fish
                        and not np.isnan(raw_pos[bout_fish, 0])):
                    px = int(raw_pos[bout_fish, 0])
                    py = int(raw_pos[bout_fish, 1])
                    ring_r = int(dot_radius * 1.8)
                    _cv2.circle(display, (px, py), ring_r, (0, 255, 255),
                               3, lineType=_cv2.LINE_AA)
            except (ValueError, TypeError):
                pass

    def _inspector_draw_numpy(self, display, raw_pos, n_fish, tab10,
                               dot_radius):
        """Minimal fallback drawing using numpy (no cv2 needed)."""
        if not self.inspector_show_positions_var.get():
            return
        for i in range(n_fish):
            if np.isnan(raw_pos[i, 0]):
                continue
            px, py = int(raw_pos[i, 0]), int(raw_pos[i, 1])
            r = dot_radius
            color = self._rgba_to_rgb_uint8(tab10[i])
            # Draw filled circle via numpy
            y_grid, x_grid = np.ogrid[-r:r + 1, -r:r + 1]
            mask = x_grid ** 2 + y_grid ** 2 <= r ** 2
            y_start = max(0, py - r)
            y_end = min(display.shape[0], py + r + 1)
            x_start = max(0, px - r)
            x_end = min(display.shape[1], px + r + 1)
            mask_y = slice(max(0, r - py), r + 1 + min(0, display.shape[0] - py - r - 1))
            mask_x = slice(max(0, r - px), r + 1 + min(0, display.shape[1] - px - r - 1))
            sub_mask = mask[mask_y, mask_x]
            display[y_start:y_end, x_start:x_end][sub_mask] = color

    def _inspector_update_zoom(self, display, raw_pos, fish_id, vid_h, vid_w):
        """Update the crop-zoom panel centered on the bout fish."""
        px = int(raw_pos[fish_id, 0])
        py = int(raw_pos[fish_id, 1])
        crop_r = 80  # Radius in pixels

        y1 = max(0, py - crop_r)
        y2 = min(vid_h, py + crop_r)
        x1 = max(0, px - crop_r)
        x2 = min(vid_w, px + crop_r)

        crop = display[y1:y2, x1:x2].copy()

        if crop.size == 0:
            return

        # Scale up for visibility
        if _CV2_AVAILABLE and crop.shape[0] > 0 and crop.shape[1] > 0:
            crop = _cv2.resize(crop, (250, 250),
                              interpolation=_cv2.INTER_LINEAR)
            # Draw crosshair at center
            cx, cy = 125, 125
            _cv2.line(crop, (cx - 15, cy), (cx + 15, cy),
                     (0, 255, 255), 1, _cv2.LINE_AA)
            _cv2.line(crop, (cx, cy - 15), (cx, cy + 15),
                     (0, 255, 255), 1, _cv2.LINE_AA)

        self._insp_zoom_artist.set_data(crop)
