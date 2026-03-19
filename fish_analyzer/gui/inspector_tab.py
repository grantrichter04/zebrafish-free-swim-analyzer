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

        # --- Trails ---
        trail_frame = tk.LabelFrame(scroll_frame, text="Trails",
                                     font=("Arial", 10, "bold"))
        trail_frame.pack(fill="x", padx=5, pady=5)

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

        # --- Video Overlay ---
        video_frame = tk.LabelFrame(scroll_frame, text="Video Overlay",
                                     font=("Arial", 10, "bold"))
        video_frame.pack(fill="x", padx=5, pady=5)

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

        # --- Spatial Overlays ---
        overlay_frame = tk.LabelFrame(scroll_frame, text="Spatial Overlays",
                                       font=("Arial", 10, "bold"))
        overlay_frame.pack(fill="x", padx=5, pady=5)

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

        # --- Bout Overlay ---
        bout_frame = tk.LabelFrame(scroll_frame, text="Bout Overlay",
                                    font=("Arial", 10, "bold"))
        bout_frame.pack(fill="x", padx=5, pady=5)

        self.inspector_show_bouts_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            bout_frame, text="Show bout speed + heading",
            variable=self.inspector_show_bouts_var,
            command=self._inspector_on_overlay_change
        ).pack(anchor="w", padx=5)

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

        # --- Time Panel Mode ---
        time_frame = tk.LabelFrame(scroll_frame, text="Time Panel",
                                    font=("Arial", 10, "bold"))
        time_frame.pack(fill="x", padx=5, pady=5)

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
        """Called when any overlay checkbox changes."""
        self._insp_fig = None  # Force full rebuild
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

        overlay_key = (
            self.inspector_show_positions_var.get(),
            self.inspector_show_nnd_var.get(),
            self.inspector_show_hull_var.get(),
            self.inspector_show_iid_var.get(),
            self.inspector_show_bouts_var.get(),
            self.inspector_video_var.get(),
        )

        rebuild = (
            self._insp_fig is None
            or self._insp_cached_file != selected
            or self._insp_cached_time_mode != time_mode
            or self._insp_cached_overlays != overlay_key
            or (time_mode == 'bout'
                and self._insp_cached_bout_fish != bout_fish)
        )

        if rebuild:
            self._inspector_rebuild_figure(
                selected, loaded, frame_idx, time_mode
            )
            self._insp_cached_file = selected
            self._insp_cached_time_mode = time_mode
            self._insp_cached_overlays = overlay_key
            self._insp_cached_bout_fish = bout_fish

        self._inspector_update_dynamic(
            selected, loaded, frame_idx, time_mode
        )

    # =========================================================================
    # FIGURE REBUILD (expensive, only when layout changes)
    # =========================================================================

    def _inspector_rebuild_figure(self, selected, loaded, frame_idx,
                                   time_mode):
        """Rebuild the entire figure from scratch."""
        for widget in self.inspector_plot_frame.winfo_children():
            widget.destroy()

        show_time = time_mode != "none"
        show_bout_heading = time_mode == "bout"

        # Determine layout
        if show_bout_heading:
            # 3 panels: arena, speed trace, heading
            self._insp_fig = Figure(figsize=(10, 10), dpi=100)
            self._insp_ax_main = self._insp_fig.add_axes(
                [0.08, 0.42, 0.88, 0.55]
            )
            self._insp_ax_time = self._insp_fig.add_axes(
                [0.08, 0.22, 0.88, 0.16]
            )
            self._insp_ax_time2 = self._insp_fig.add_axes(
                [0.08, 0.04, 0.88, 0.14]
            )
        elif show_time:
            # 2 panels: arena + time series
            self._insp_fig = Figure(figsize=(10, 8), dpi=100)
            self._insp_ax_main = self._insp_fig.add_axes(
                [0.08, 0.25, 0.88, 0.70]
            )
            self._insp_ax_time = self._insp_fig.add_axes(
                [0.08, 0.05, 0.88, 0.15]
            )
            self._insp_ax_time2 = None
        else:
            # Just arena
            self._insp_fig = Figure(figsize=(10, 8), dpi=100)
            self._insp_ax_main = self._insp_fig.add_axes(
                [0.05, 0.05, 0.90, 0.90]
            )
            self._insp_ax_time = None
            self._insp_ax_time2 = None

        ax = self._insp_ax_main

        # Coordinate system
        pixels_to_bl = 1.0 / loaded.metadata.body_length
        width_bl = loaded.metadata.video_width * pixels_to_bl
        height_bl = loaded.metadata.video_height * pixels_to_bl
        self._insp_cached_width_bl = width_bl
        self._insp_cached_height_bl = height_bl

        # Background image
        self._insp_cached_background = None
        self._insp_video_bg_artist = None
        use_video = self.inspector_video_var.get()

        if (not use_video and loaded.background_image_path
                and loaded.background_image_path.exists()):
            try:
                self._insp_cached_background = plt.imread(
                    str(loaded.background_image_path)
                )
            except Exception:
                pass

        if not use_video and self._insp_cached_background is not None:
            bg = self._insp_cached_background
            cmap = 'gray' if len(bg.shape) == 2 else None
            ax.imshow(
                bg, extent=[0, width_bl, 0, height_bl],
                aspect='equal', origin='upper', alpha=0.7, zorder=0,
                cmap=cmap
            )

        ax.set_xlim(0, width_bl)
        ax.set_ylim(0, height_bl)
        ax.set_xlabel('X (BL)')
        ax.set_ylabel('Y (BL)')
        ax.set_aspect('equal')
        if self._insp_cached_background is None and not use_video:
            ax.grid(True, alpha=0.3)

        # --- Time panel: shoaling metrics (static data, cursor is dynamic) ---
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

        # --- Time panel: bout (draw full trace, scroll via xlim) ---
        elif time_mode == 'bout':
            self._inspector_build_bout_panels(selected, loaded)

        # Create canvas
        self._insp_canvas = FigureCanvasTkAgg(
            self._insp_fig, master=self.inspector_plot_frame
        )
        self._insp_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._insp_dynamic_artists = []

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

        # --- Speed panel: full trace + all bout bands ---
        ax_speed.plot(time_s, speed, color='black', lw=0.5, alpha=0.6)
        ax_speed.axhline(
            threshold, color='orange', linestyle='--', lw=1, alpha=0.7,
            label=f'Threshold ({threshold} {unit}/s)'
        )

        for bout in bout_result.bouts:
            t0 = bout.start_frame / fps
            t1 = bout.end_frame / fps
            hc = bout.heading_change_deg
            if hc > 5:
                color = '#1976d2'
            elif hc < -5:
                color = '#d32f2f'
            else:
                color = '#bdbdbd'
            ax_speed.axvspan(t0, t1, color=color, alpha=0.25, zorder=0)

        ax_speed.set_ylabel(f"Speed ({unit}/s)", fontsize=8)
        ax_speed.set_title(
            f"Fish {fish_id} — {len(bout_result.bouts)} bouts total "
            f"(blue=left, red=right, gray=straight)",
            fontsize=9
        )
        ax_speed.legend(fontsize=7, loc='upper right')

        # Cap y at 99th percentile
        if len(speed) > 0:
            y_cap = max(np.percentile(speed, 99) * 1.2, threshold * 3)
            ax_speed.set_ylim(0, y_cap)

        # Cursor line
        self._insp_bout_cursor_speed = ax_speed.axvline(
            x=0, color='green', lw=1.5, alpha=0.8
        )

        # --- Heading panel: all bout heading bars ---
        if ax_heading:
            for bout in bout_result.bouts:
                t_mid = (bout.start_frame + bout.end_frame) / 2 / fps
                hc = bout.heading_change_deg
                if hc > 5:
                    c = '#1976d2'
                elif hc < -5:
                    c = '#d32f2f'
                else:
                    c = '#757575'
                ax_heading.bar(
                    t_mid, hc, width=max(0.03, bout.duration_s),
                    color=c, alpha=0.7, edgecolor='black', lw=0.3
                )

            ax_heading.axhline(0, color='black', lw=0.5)
            ax_heading.axhline(5, color='gray', lw=0.5, linestyle=':')
            ax_heading.axhline(-5, color='gray', lw=0.5, linestyle=':')
            ax_heading.set_ylabel("Heading change (\u00b0)", fontsize=8)
            ax_heading.set_xlabel("Time (s)", fontsize=8)

            if bout_result.bouts:
                max_hc = max(
                    abs(b.heading_change_deg) for b in bout_result.bouts
                )
                ylim = max(max_hc * 1.1, 30)
                ax_heading.set_ylim(-ylim, ylim)

            self._insp_bout_cursor_heading = ax_heading.axvline(
                x=0, color='green', lw=1.5, alpha=0.8
            )

    # =========================================================================
    # DYNAMIC UPDATE (fast, every frame/step)
    # =========================================================================

    def _inspector_update_dynamic(self, selected, loaded, frame_idx,
                                   time_mode):
        """Update only the dynamic elements (positions, overlays, cursor)."""
        ax = self._insp_ax_main
        if ax is None:
            return

        # Remove previous dynamic artists
        for artist in self._insp_dynamic_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._insp_dynamic_artists = []

        # Video frame
        use_video = self.inspector_video_var.get()
        if use_video and selected in self.video_readers:
            self._inspector_update_video_frame(selected, frame_idx, ax)

        # Get fish positions via ShoalingCalculator
        params = ShoalingParameters(sample_interval_frames=1)
        calculator = ShoalingCalculator(loaded, params)
        positions_bl, is_complete = calculator.get_positions_at_frame(frame_idx)
        n_fish = loaded.n_fish
        colors = plt.cm.tab10(np.linspace(0, 1, n_fish))

        # Trails
        trail_len = self.inspector_trail_var.get()
        if trail_len > 0:
            self._inspector_draw_trails(
                ax, loaded, calculator, frame_idx, trail_len, n_fish, colors
            )

        # Shoaling overlays
        if self.inspector_show_nnd_var.get():
            self._inspector_draw_nnd(ax, positions_bl, n_fish)

        if self.inspector_show_hull_var.get():
            hull_vertices = calculator.get_convex_hull_vertices(frame_idx)
            if hull_vertices is not None and len(hull_vertices) >= 3:
                hull_closed = np.vstack([hull_vertices, hull_vertices[0]])
                fill = ax.fill(
                    hull_closed[:, 0], hull_closed[:, 1],
                    alpha=0.2, color='green', zorder=1
                )
                self._insp_dynamic_artists.extend(fill)
                line, = ax.plot(
                    hull_closed[:, 0], hull_closed[:, 1],
                    'g-', lw=2, alpha=0.7, zorder=2
                )
                self._insp_dynamic_artists.append(line)

        if self.inspector_show_iid_var.get():
            self._inspector_draw_iid(ax, positions_bl, n_fish)

        # Fish positions
        if self.inspector_show_positions_var.get():
            for i in range(n_fish):
                if not np.isnan(positions_bl[i, 0]):
                    sc = ax.scatter(
                        positions_bl[i, 0], positions_bl[i, 1],
                        c=[colors[i]], s=200, edgecolors='black',
                        linewidths=2, zorder=5
                    )
                    self._insp_dynamic_artists.append(sc)
                    txt = ax.annotate(
                        str(i),
                        (positions_bl[i, 0], positions_bl[i, 1]),
                        ha='center', va='center', fontsize=10,
                        fontweight='bold', color='white', zorder=6
                    )
                    self._insp_dynamic_artists.append(txt)

        # Highlight selected bout fish
        if self.inspector_show_bouts_var.get():
            try:
                bout_fish = int(self.inspector_bout_fish_var.get())
                if (bout_fish < n_fish
                        and not np.isnan(positions_bl[bout_fish, 0])):
                    ring = ax.scatter(
                        positions_bl[bout_fish, 0],
                        positions_bl[bout_fish, 1],
                        s=400, facecolors='none', edgecolors='yellow',
                        linewidths=3, zorder=7
                    )
                    self._insp_dynamic_artists.append(ring)
            except (ValueError, TypeError):
                pass

        # Title
        fps = loaded.calibration.frame_rate
        time_s = frame_idx / fps
        ax.set_title(
            f'Frame {frame_idx} | {time_s:.1f}s',
            fontsize=12, fontweight='bold'
        )

        # Update time panel cursor
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

            # Scroll the bout panels by setting xlim
            if self._insp_ax_time:
                self._insp_ax_time.set_xlim(t_start, t_end)
            if self._insp_ax_time2:
                self._insp_ax_time2.set_xlim(t_start, t_end)

            # Move cursor lines
            if self._insp_bout_cursor_speed:
                self._insp_bout_cursor_speed.set_xdata(
                    [center_s, center_s]
                )
            if self._insp_bout_cursor_heading:
                self._insp_bout_cursor_heading.set_xdata(
                    [center_s, center_s]
                )

        self._insp_canvas.draw_idle()

    # =========================================================================
    # DRAWING HELPERS
    # =========================================================================

    def _inspector_update_video_frame(self, selected, frame_idx, ax):
        """Update the video frame overlay on the arena axes."""
        quality = self.inspector_quality_var.get()

        # Frame skipping during animation for performance
        if self.animation_running:
            skip = getattr(self, '_insp_frame_skip', 0) + 1
            self._insp_frame_skip = skip
            if quality == "low" and skip % 3 != 0:
                return
            elif quality == "medium" and skip % 2 != 0:
                return

        reader = self.video_readers[selected]
        frame = reader.read_frame(frame_idx)
        if frame is None:
            return

        if quality == "low":
            frame = frame[::4, ::4]
        elif quality == "medium":
            frame = frame[::2, ::2]

        if self._insp_video_bg_artist is not None:
            try:
                self._insp_video_bg_artist.remove()
            except (ValueError, AttributeError):
                pass

        self._insp_video_bg_artist = ax.imshow(
            frame,
            extent=[0, self._insp_cached_width_bl, 0,
                    self._insp_cached_height_bl],
            aspect='equal', origin='upper', alpha=0.6, zorder=0,
            interpolation='bilinear' if quality == 'high' else 'nearest'
        )

    def _inspector_draw_nnd(self, ax, positions_bl, n_fish):
        """Draw NND lines between each fish and its nearest neighbor."""
        for i in range(n_fish):
            if np.isnan(positions_bl[i, 0]):
                continue
            min_dist = np.inf
            nn_idx = -1
            for j in range(n_fish):
                if i == j or np.isnan(positions_bl[j, 0]):
                    continue
                d = np.sqrt(
                    (positions_bl[i, 0] - positions_bl[j, 0]) ** 2
                    + (positions_bl[i, 1] - positions_bl[j, 1]) ** 2
                )
                if d < min_dist:
                    min_dist = d
                    nn_idx = j
            if nn_idx >= 0:
                line, = ax.plot(
                    [positions_bl[i, 0], positions_bl[nn_idx, 0]],
                    [positions_bl[i, 1], positions_bl[nn_idx, 1]],
                    'b-', lw=1.5, alpha=0.5, zorder=2
                )
                self._insp_dynamic_artists.append(line)
                mid_x = (positions_bl[i, 0] + positions_bl[nn_idx, 0]) / 2
                mid_y = (positions_bl[i, 1] + positions_bl[nn_idx, 1]) / 2
                txt = ax.text(
                    mid_x, mid_y, f'{min_dist:.1f}', fontsize=7,
                    color='blue', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.7),
                    zorder=3
                )
                self._insp_dynamic_artists.append(txt)

    def _inspector_draw_iid(self, ax, positions_bl, n_fish):
        """Draw IID lines from focus fish to all others."""
        try:
            focus = int(self.inspector_iid_focus_var.get())
        except (ValueError, TypeError):
            focus = 0

        if focus >= n_fish or np.isnan(positions_bl[focus, 0]):
            return

        for j in range(n_fish):
            if j == focus or np.isnan(positions_bl[j, 0]):
                continue
            d = np.sqrt(
                (positions_bl[focus, 0] - positions_bl[j, 0]) ** 2
                + (positions_bl[focus, 1] - positions_bl[j, 1]) ** 2
            )
            line, = ax.plot(
                [positions_bl[focus, 0], positions_bl[j, 0]],
                [positions_bl[focus, 1], positions_bl[j, 1]],
                'g-', lw=1.5, alpha=0.4, zorder=2
            )
            self._insp_dynamic_artists.append(line)
            mid_x = (positions_bl[focus, 0] + positions_bl[j, 0]) / 2
            mid_y = (positions_bl[focus, 1] + positions_bl[j, 1]) / 2
            txt = ax.text(
                mid_x, mid_y, f'{d:.1f}', fontsize=7,
                color='darkgreen', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.7),
                zorder=3
            )
            self._insp_dynamic_artists.append(txt)

    def _inspector_draw_trails(self, ax, loaded, calculator, frame_idx,
                                trail_len, n_fish, colors):
        """Draw fading trajectory trails for each fish."""
        start_frame = max(0, frame_idx - trail_len)
        if start_frame >= frame_idx:
            return

        trail_frames = range(start_frame, frame_idx + 1)
        trail_x = np.full((n_fish, len(trail_frames)), np.nan)
        trail_y = np.full((n_fish, len(trail_frames)), np.nan)

        for t_idx, f in enumerate(trail_frames):
            try:
                pos, _ = calculator.get_positions_at_frame(f)
                for i in range(n_fish):
                    if not np.isnan(pos[i, 0]):
                        trail_x[i, t_idx] = pos[i, 0]
                        trail_y[i, t_idx] = pos[i, 1]
            except (IndexError, ValueError):
                continue

        for i in range(n_fish):
            x = trail_x[i]
            y = trail_y[i]
            if np.sum(~np.isnan(x)) < 2:
                continue
            for j in range(1, len(x)):
                if np.isnan(x[j]) or np.isnan(x[j - 1]):
                    continue
                alpha = 0.1 + 0.6 * (j / len(x))
                lw = 0.5 + 1.5 * (j / len(x))
                line, = ax.plot(
                    [x[j - 1], x[j]], [y[j - 1], y[j]],
                    color=colors[i], alpha=alpha, linewidth=lw, zorder=2
                )
                self._insp_dynamic_artists.append(line)
