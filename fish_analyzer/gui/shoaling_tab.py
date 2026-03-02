"""
fish_analyzer/gui/shoaling_tab.py
=================================
Shoaling Tab Mixin - Group behavior analysis and interactive frame viewer.

This mixin provides all methods related to:
- NND (Nearest Neighbor Distance) analysis
- IID (Inter-Individual Distance) analysis
- Convex Hull area analysis
- Interactive frame scrubber with animation
- Video frame overlay support
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from ..shoaling import ShoalingParameters, ShoalingResults, ShoalingCalculator
from .utils import smooth_time_series


class ShoalingTabMixin:
    """
    Mixin providing Shoaling Analysis Tab functionality.
    
    Expects the following attributes from base class:
    - self.root: tk.Tk
    - self.notebook: ttk.Notebook
    - self.loaded_files: Dict[str, LoadedTrajectoryFile]
    - self.shoaling_params: ShoalingParameters
    - self.animation_running: bool
    - self.animation_after_id: Optional[int]
    - self.video_readers: Dict[str, Any]
    - self.frame_view_fig, self.frame_view_canvas, etc.
    """

    def _create_shoaling_tab(self):
        """Create the shoaling analysis tab with interactive frame scrubber."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Shoaling Analysis")

        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)

        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=2)

        # ----- LEFT PANEL: Controls -----
        self._create_shoaling_controls(left_panel)

        # ----- RIGHT PANEL: Results -----
        self._create_shoaling_results(right_panel)

    def _create_shoaling_controls(self, parent):
        """Create the control panel for shoaling tab."""
        # Parameters
        params_frame = tk.LabelFrame(parent, text="Shoaling Parameters", font=("Arial", 11, "bold"))
        params_frame.pack(fill="x", padx=10, pady=10)

        interval_frame = tk.Frame(params_frame)
        interval_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(interval_frame, text="Sample every:").pack(side="left")
        self.shoaling_interval_var = tk.StringVar(value="30")
        tk.Entry(interval_frame, textvariable=self.shoaling_interval_var, width=6).pack(side="left", padx=5)
        tk.Label(interval_frame, text="frames").pack(side="left")

        tk.Label(params_frame, text="At 30fps, sampling every 30 frames = 1 sample/second",
                font=("Arial", 8), fg="gray").pack(anchor="w", padx=10)

        smooth_frame = tk.Frame(params_frame)
        smooth_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(smooth_frame, text="Plot smoothing:").pack(side="left")
        self.shoaling_smooth_var = tk.StringVar(value="5.0")
        tk.Entry(smooth_frame, textvariable=self.shoaling_smooth_var, width=6).pack(side="left", padx=5)
        tk.Label(smooth_frame, text="seconds").pack(side="left")

        # File selection
        file_frame = tk.LabelFrame(parent, text="Select Files to Compare", font=("Arial", 11, "bold"))
        file_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(file_frame, text="Hold Ctrl/Cmd to select multiple files",
                font=("Arial", 8), fg="gray").pack(anchor="w", padx=10, pady=2)

        list_container = tk.Frame(file_frame)
        list_container.pack(fill="x", padx=10, pady=5)

        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")

        self.shoaling_files_listbox = tk.Listbox(
            list_container, selectmode=tk.EXTENDED, height=5,
            yscrollcommand=scrollbar.set
        )
        self.shoaling_files_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.shoaling_files_listbox.yview)

        tk.Button(
            parent, text="Run Shoaling Analysis",
            command=self._run_shoaling_analysis,
            bg="lightgreen", font=("Arial", 12, "bold"), height=2
        ).pack(fill="x", padx=10, pady=10)

        # Frame scrubber section
        self._create_frame_scrubber_controls(parent)

    def _create_frame_scrubber_controls(self, parent):
        """Create the frame scrubber control section."""
        scrubber_frame = tk.LabelFrame(parent, text="Interactive Frame Viewer", font=("Arial", 11, "bold"))
        scrubber_frame.pack(fill="x", padx=10, pady=10)

        # File selection for visualization
        file_viz_frame = tk.Frame(scrubber_frame)
        file_viz_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(file_viz_frame, text="File:").pack(side="left")
        self.viz_shoaling_file_var = tk.StringVar()
        self.viz_shoaling_file_dropdown = ttk.Combobox(
            file_viz_frame, textvariable=self.viz_shoaling_file_var,
            state="readonly", width=18
        )
        self.viz_shoaling_file_dropdown.pack(side="left", padx=5)
        self.viz_shoaling_file_dropdown.bind("<<ComboboxSelected>>", self._on_viz_shoaling_file_selected)

        # Visualization mode
        mode_frame = tk.Frame(scrubber_frame)
        mode_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(mode_frame, text="Show:").pack(side="left")
        self.viz_mode_var = tk.StringVar(value="nnd")

        rb_frame = tk.Frame(mode_frame)
        rb_frame.pack(side="left", padx=5)
        tk.Radiobutton(rb_frame, text="NND", variable=self.viz_mode_var, value="nnd").grid(row=0, column=0, sticky="w")
        tk.Radiobutton(rb_frame, text="Hull", variable=self.viz_mode_var, value="hull").grid(row=0, column=1, sticky="w")
        tk.Radiobutton(rb_frame, text="IID", variable=self.viz_mode_var, value="iid").grid(row=0, column=2, sticky="w")

        # Focus fish for IID
        focus_frame = tk.Frame(scrubber_frame)
        focus_frame.pack(fill="x", padx=10, pady=2)
        tk.Label(focus_frame, text="IID Focus Fish:").pack(side="left")
        self.iid_focus_fish_var = tk.StringVar(value="0")
        self.iid_focus_fish_dropdown = ttk.Combobox(
            focus_frame, textvariable=self.iid_focus_fish_var,
            state="readonly", width=8
        )
        self.iid_focus_fish_dropdown.pack(side="left", padx=5)
        tk.Label(focus_frame, text="(shows distances from this fish to all others)",
                font=("Arial", 8), fg="gray").pack(side="left")

        # Frame slider
        slider_frame = tk.Frame(scrubber_frame)
        slider_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(slider_frame, text="Sample:").pack(side="left")
        self.frame_slider_var = tk.IntVar(value=0)
        self.frame_slider = tk.Scale(
            slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.frame_slider_var, command=self._on_slider_change,
            length=150, showvalue=False
        )
        self.frame_slider.pack(side="left", padx=5, fill="x", expand=True)

        # Jump to frame/time
        jump_frame = tk.Frame(scrubber_frame)
        jump_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(jump_frame, text="Jump to:").pack(side="left")

        tk.Label(jump_frame, text="Frame:").pack(side="left", padx=(10, 2))
        self.jump_frame_var = tk.StringVar()
        jump_frame_entry = tk.Entry(jump_frame, textvariable=self.jump_frame_var, width=8)
        jump_frame_entry.pack(side="left")
        jump_frame_entry.bind("<Return>", self._jump_to_frame)

        tk.Label(jump_frame, text="Time(s):").pack(side="left", padx=(10, 2))
        self.jump_time_var = tk.StringVar()
        jump_time_entry = tk.Entry(jump_frame, textvariable=self.jump_time_var, width=8)
        jump_time_entry.pack(side="left")
        jump_time_entry.bind("<Return>", self._jump_to_time)

        tk.Button(jump_frame, text="Go", command=self._jump_to_input, width=4).pack(side="left", padx=5)

        # Current frame info
        self.frame_info_label = tk.Label(scrubber_frame, text="Frame: -- | Time: --", font=("Arial", 9))
        self.frame_info_label.pack(anchor="w", padx=10)

        # Playback controls
        playback_frame = tk.Frame(scrubber_frame)
        playback_frame.pack(fill="x", padx=10, pady=5)

        self.play_button = tk.Button(playback_frame, text="> Play", command=self._toggle_playback,
                                    bg="lightblue", width=8)
        self.play_button.pack(side="left", padx=2)

        tk.Label(playback_frame, text="Speed:").pack(side="left", padx=(10, 2))
        self.playback_speed_var = tk.StringVar(value="1x")
        speed_combo = ttk.Combobox(playback_frame, textvariable=self.playback_speed_var,
                                   values=["0.5x", "1x", "2x", "4x", "8x"], width=5, state="readonly")
        speed_combo.pack(side="left")
        tk.Label(playback_frame, text="(real-time)").pack(side="left", padx=2)

        # Video frame overlay controls
        self._create_video_overlay_controls(scrubber_frame)

    def _create_video_overlay_controls(self, parent):
        """Create the video frame overlay controls."""
        video_frame = tk.LabelFrame(parent, text="Video Frame Overlay", font=("Arial", 10, "bold"))
        video_frame.pack(fill="x", padx=10, pady=5)

        self.use_video_frames_var = tk.BooleanVar(value=False)
        tk.Checkbutton(video_frame, text="Use video frames (instead of background)",
                      variable=self.use_video_frames_var,
                      command=self._on_video_toggle).pack(anchor="w", padx=10)

        video_btn_frame = tk.Frame(video_frame)
        video_btn_frame.pack(fill="x", padx=10, pady=3)

        tk.Button(video_btn_frame, text="Browse for Video...",
                 command=self._browse_for_video, width=18).pack(side="left", padx=2)

        self.video_status_label = tk.Label(video_frame, text="No video loaded", 
                                           font=("Arial", 8), fg="gray")
        self.video_status_label.pack(anchor="w", padx=10, pady=2)

        # Quality dropdown for playback performance
        quality_frame = tk.Frame(video_frame)
        quality_frame.pack(fill="x", padx=10, pady=3)
        tk.Label(quality_frame, text="Playback quality:").pack(side="left")
        self.video_quality_var = tk.StringVar(value="high")
        quality_combo = ttk.Combobox(quality_frame, textvariable=self.video_quality_var,
                                     values=["high", "medium", "low"], width=8, state="readonly")
        quality_combo.pack(side="left", padx=5)
        tk.Label(quality_frame, text="(lower = faster playback)").pack(side="left")

    def _create_shoaling_results(self, parent):
        """Create the results panel for shoaling tab."""
        results_notebook = ttk.Notebook(parent)
        results_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Text results
        text_tab = ttk.Frame(results_notebook)
        results_notebook.add(text_tab, text="Results Summary")

        scrollbar = tk.Scrollbar(text_tab)
        scrollbar.pack(side="right", fill="y")

        self.shoaling_results_text = tk.Text(
            text_tab, wrap=tk.WORD, font=("Courier", 10),
            yscrollcommand=scrollbar.set
        )
        self.shoaling_results_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.shoaling_results_text.yview)

        self.shoaling_results_text.insert("1.0",
            "Shoaling analysis results will appear here.\n\n"
            "To run analysis:\n"
            "1. Load trajectory files in the 'Data Setup' tab\n"
            "2. Select one or more files (Ctrl+click for multiple)\n"
            "3. Click 'Run Shoaling Analysis'\n\n"
            "Metrics:\n"
            "- NND: Nearest Neighbor Distance (closest fish)\n"
            "- IID: Inter-Individual Distance (all pairs average)\n"
            "- Hull: Convex Hull Area (group footprint)\n"
        )

        # NND Plot
        nnd_plot_tab = ttk.Frame(results_notebook)
        results_notebook.add(nnd_plot_tab, text="NND Time Series")
        self.shoaling_nnd_plot_frame = tk.Frame(nnd_plot_tab, bg="white")
        self.shoaling_nnd_plot_frame.pack(fill="both", expand=True)

        # IID Plot
        iid_plot_tab = ttk.Frame(results_notebook)
        results_notebook.add(iid_plot_tab, text="IID Time Series")
        self.shoaling_iid_plot_frame = tk.Frame(iid_plot_tab, bg="white")
        self.shoaling_iid_plot_frame.pack(fill="both", expand=True)

        # Hull Area Plot
        hull_plot_tab = ttk.Frame(results_notebook)
        results_notebook.add(hull_plot_tab, text="Hull Area")
        self.shoaling_hull_plot_frame = tk.Frame(hull_plot_tab, bg="white")
        self.shoaling_hull_plot_frame.pack(fill="both", expand=True)

        # Frame visualization
        frame_tab = ttk.Frame(results_notebook)
        results_notebook.add(frame_tab, text="Frame View")
        self.shoaling_frame_plot = tk.Frame(frame_tab, bg="white")
        self.shoaling_frame_plot.pack(fill="both", expand=True)

        tk.Label(self.shoaling_frame_plot,
                text="Run analysis, then use the slider to scrub through frames",
                font=("Arial", 11), fg="gray").pack(expand=True)

    # =========================================================================
    # SHOALING FILE DROPDOWN UPDATE
    # =========================================================================

    def _update_shoaling_file_dropdown(self):
        """Update all file selection widgets across tabs."""
        # Update shoaling files listbox
        self.shoaling_files_listbox.delete(0, tk.END)
        for nickname in self.loaded_files.keys():
            self.shoaling_files_listbox.insert(tk.END, nickname)

        # Update visualization dropdown
        file_list = list(self.loaded_files.keys())
        self.viz_shoaling_file_dropdown['values'] = file_list
        if file_list and not self.viz_shoaling_file_var.get():
            self.viz_shoaling_file_dropdown.current(0)
            self._on_viz_shoaling_file_selected()

        # Update analysis files listbox (if method exists)
        if hasattr(self, '_update_analysis_files_listbox'):
            self._update_analysis_files_listbox()

        # Update spatial analysis dropdown (if method exists)
        if hasattr(self, '_update_spatial_file_dropdown'):
            self._update_spatial_file_dropdown()

    # =========================================================================
    # SHOALING ANALYSIS METHODS
    # =========================================================================

    def _run_shoaling_analysis(self):
        """Run shoaling analysis on selected files."""
        selected_indices = self.shoaling_files_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("No Files Selected", "Please select one or more files.")
            return

        file_keys = list(self.loaded_files.keys())
        selected_files = [file_keys[i] for i in selected_indices]

        try:
            sample_interval = int(self.shoaling_interval_var.get())
            smooth_window = float(self.shoaling_smooth_var.get())
            params = ShoalingParameters(
                sample_interval_frames=sample_interval,
                smoothing_window_seconds=smooth_window
            )
            params.validate()
            self.shoaling_params = params
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", f"Check parameters:\n{e}")
            return

        all_results = {}
        for filename in selected_files:
            loaded_file = self.loaded_files[filename]
            try:
                calculator = ShoalingCalculator(loaded_file, params)
                results = calculator.calculate()
                loaded_file.shoaling_results = results
                all_results[filename] = results
            except Exception as e:
                messagebox.showwarning("Analysis Warning", f"Failed '{filename}':\n{e}")
                import traceback
                traceback.print_exc()

        if not all_results:
            messagebox.showerror("Analysis Failed", "Could not analyze any files.")
            return

        self._display_shoaling_comparison(all_results)
        self._plot_nnd_comparison(all_results)
        self._plot_iid_comparison(all_results)
        self._plot_hull_comparison(all_results)

        # Update visualization dropdown
        self.viz_shoaling_file_dropdown['values'] = list(all_results.keys())
        if all_results:
            self.viz_shoaling_file_dropdown.current(0)
            self._on_viz_shoaling_file_selected()

        messagebox.showinfo("Analysis Complete", f"Shoaling analysis complete for {len(all_results)} file(s).")

    def _display_shoaling_comparison(self, all_results: Dict[str, ShoalingResults]):
        """Display comparison of shoaling results."""
        self.shoaling_results_text.delete("1.0", tk.END)

        lines = ["=" * 90, "SHOALING ANALYSIS COMPARISON", "=" * 90, ""]

        lines.append("SUMMARY COMPARISON:")
        lines.append("-" * 90)
        lines.append(f"{'File':<20} {'Fish':<6} {'Mean NND':<12} {'Mean IID':<12} {'Mean Hull':<14} {'Complete %':<12}")
        lines.append("-" * 90)

        for filename, results in all_results.items():
            short_name = filename[:19] if len(filename) > 19 else filename
            lines.append(
                f"{short_name:<20} {results.n_fish:<6} {results.mean_nnd:<12.3f} "
                f"{results.mean_iid:<12.3f} {results.mean_hull_area:<14.2f} "
                f"{results.completeness_percentage:<12.1f}"
            )

        lines.append("-" * 90)
        lines.append("")

        for filename, results in all_results.items():
            lines.append(f"\nFILE: {filename}")
            lines.append(results.summary())

        self.shoaling_results_text.insert("1.0", "\n".join(lines))

    def _plot_nnd_comparison(self, all_results: Dict[str, ShoalingResults]):
        """Plot NND time series with smoothing."""
        for widget in self.shoaling_nnd_plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        try:
            smooth_seconds = float(self.shoaling_smooth_var.get())
        except:
            smooth_seconds = 5.0

        for (filename, results), color in zip(all_results.items(), colors):
            time_minutes = results.timestamps / 60.0
            nnd_data = results.mean_nnd_per_sample.copy()
            
            if smooth_seconds > 0 and len(nnd_data) > 1:
                effective_fps = 1.0 / (results.sample_interval_frames / results.frame_rate)
                nnd_data = smooth_time_series(nnd_data, smooth_seconds, effective_fps)
            
            ax.plot(time_minutes, nnd_data, color=color, linewidth=2,
                   label=f'{filename} ({results.mean_nnd:.2f} BL)', alpha=0.9)

        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Mean Nearest Neighbor Distance (BL)', fontsize=12)
        ax.set_title('NND Comparison Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.shoaling_nnd_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_iid_comparison(self, all_results: Dict[str, ShoalingResults]):
        """Plot IID time series with smoothing."""
        for widget in self.shoaling_iid_plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        try:
            smooth_seconds = float(self.shoaling_smooth_var.get())
        except:
            smooth_seconds = 5.0

        for (filename, results), color in zip(all_results.items(), colors):
            time_minutes = results.timestamps / 60.0
            iid_data = results.mean_iid_per_sample.copy()
            
            if smooth_seconds > 0 and len(iid_data) > 1:
                effective_fps = 1.0 / (results.sample_interval_frames / results.frame_rate)
                iid_data = smooth_time_series(iid_data, smooth_seconds, effective_fps)
            
            ax.plot(time_minutes, iid_data, color=color, linewidth=2,
                   label=f'{filename} ({results.mean_iid:.2f} BL)', alpha=0.9)

        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Mean Inter-Individual Distance (BL)', fontsize=12)
        ax.set_title('IID (Group Cohesion) Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.shoaling_iid_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_hull_comparison(self, all_results: Dict[str, ShoalingResults]):
        """Plot hull area time series with smoothing."""
        for widget in self.shoaling_hull_plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        try:
            smooth_seconds = float(self.shoaling_smooth_var.get())
        except:
            smooth_seconds = 5.0

        for (filename, results), color in zip(all_results.items(), colors):
            time_minutes = results.timestamps / 60.0
            hull_data = results.convex_hull_area_per_sample.copy()
            
            if smooth_seconds > 0 and len(hull_data) > 1:
                effective_fps = 1.0 / (results.sample_interval_frames / results.frame_rate)
                hull_data = smooth_time_series(hull_data, smooth_seconds, effective_fps)
            
            ax.plot(time_minutes, hull_data, color=color, linewidth=2,
                   label=f'{filename} ({results.mean_hull_area:.1f} BL^2)', alpha=0.9)

        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Convex Hull Area (BL^2)', fontsize=12)
        ax.set_title('Group Spread Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.shoaling_hull_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # =========================================================================
    # FRAME SCRUBBER METHODS
    # =========================================================================

    def _on_viz_shoaling_file_selected(self, event=None):
        """Update slider range when file is selected."""
        selected_file = self.viz_shoaling_file_var.get()
        if not selected_file or selected_file not in self.loaded_files:
            return

        loaded_file = self.loaded_files[selected_file]

        # Update focus fish dropdown
        fish_options = [str(i) for i in range(loaded_file.n_fish)]
        self.iid_focus_fish_dropdown['values'] = fish_options
        if fish_options:
            self.iid_focus_fish_dropdown.current(0)

        if loaded_file.shoaling_results is None:
            self.frame_slider.configure(to=0)
            self.frame_info_label.config(text="Analysis not yet run for this file")
            return

        results = loaded_file.shoaling_results
        max_sample = results.n_samples_used - 1
        self.frame_slider.configure(to=max_sample)
        self.frame_slider_var.set(0)
        self._update_frame_info()
        self.frame_view_fig = None

    def _on_slider_change(self, value):
        """Called when slider value changes."""
        self._update_frame_info()
        self._update_frame_view_fast()

    def _update_frame_info(self):
        """Update the frame info label."""
        selected_file = self.viz_shoaling_file_var.get()
        if not selected_file or selected_file not in self.loaded_files:
            return

        loaded_file = self.loaded_files[selected_file]
        if loaded_file.shoaling_results is None:
            return

        results = loaded_file.shoaling_results
        sample_idx = self.frame_slider_var.get()

        if sample_idx < len(results.frame_indices):
            frame_num = int(results.frame_indices[sample_idx])
            time_sec = results.timestamps[sample_idx]
            nnd_val = results.mean_nnd_per_sample[sample_idx]
            iid_val = results.mean_iid_per_sample[sample_idx]
            hull_val = results.convex_hull_area_per_sample[sample_idx]

            self.frame_info_label.config(
                text=f"Frame: {frame_num} | Time: {time_sec:.1f}s | "
                     f"NND: {nnd_val:.2f} | IID: {iid_val:.2f} | Hull: {hull_val:.1f}"
            )

            self.jump_frame_var.set(str(frame_num))
            self.jump_time_var.set(f"{time_sec:.1f}")

    def _jump_to_frame(self, event=None):
        self._jump_to_input()

    def _jump_to_time(self, event=None):
        self._jump_to_input()

    def _jump_to_input(self):
        """Jump to frame or time based on input."""
        selected_file = self.viz_shoaling_file_var.get()
        if not selected_file or selected_file not in self.loaded_files:
            return

        loaded_file = self.loaded_files[selected_file]
        if loaded_file.shoaling_results is None:
            return

        results = loaded_file.shoaling_results
        frame_str = self.jump_frame_var.get().strip()
        time_str = self.jump_time_var.get().strip()

        target_sample = None

        if frame_str:
            try:
                target_frame = int(frame_str)
                differences = np.abs(results.frame_indices - target_frame)
                target_sample = int(np.argmin(differences))
            except ValueError:
                pass

        if target_sample is None and time_str:
            try:
                target_time = float(time_str)
                differences = np.abs(results.timestamps - target_time)
                target_sample = int(np.argmin(differences))
            except ValueError:
                pass

        if target_sample is not None:
            target_sample = max(0, min(target_sample, results.n_samples_used - 1))
            self.frame_slider_var.set(target_sample)
            self._update_frame_info()
            self._update_frame_view_fast()

    def _toggle_playback(self):
        """Toggle animation playback."""
        if self.animation_running:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        """Start animation."""
        self.animation_running = True
        self.play_button.config(text="|| Pause", bg="salmon")

        # Pre-load frames for smooth playback
        selected_file = self.viz_shoaling_file_var.get()
        if selected_file and selected_file in self.video_readers:
            current = self.frame_slider_var.get()
            loaded_file = self.loaded_files[selected_file]
            results = loaded_file.shoaling_results
            if results:
                start_idx = current
                end_idx = min(current + 50, len(results.frame_indices))
                reader = self.video_readers[selected_file]
                for i in range(start_idx, end_idx):
                    if i < len(results.frame_indices):
                        frame_num = int(results.frame_indices[i])
                        reader.read_frame(frame_num)
        self._animate_step()

    def _stop_playback(self):
        """Stop animation."""
        self.animation_running = False
        self.play_button.config(text="> Play", bg="lightblue")
        if self.animation_after_id:
            self.root.after_cancel(self.animation_after_id)
            self.animation_after_id = None

    def _animate_step(self):
        """Advance one step in animation."""
        if not self.animation_running:
            return

        current = self.frame_slider_var.get()
        max_val = self.frame_slider.cget('to')

        if current < max_val:
            self.frame_slider_var.set(current + 1)
        else:
            self.frame_slider_var.set(0)

        self._update_frame_info()
        self._update_frame_view_fast()

        # Calculate delay
        delay_ms = 100
        selected_file = self.viz_shoaling_file_var.get()
        if selected_file and selected_file in self.loaded_files:
            loaded_file = self.loaded_files[selected_file]
            if loaded_file.shoaling_results:
                results = loaded_file.shoaling_results
                real_time_per_sample = results.sample_interval_frames / results.frame_rate

                speed_str = self.playback_speed_var.get()
                try:
                    speed_mult = float(speed_str.replace('x', ''))
                except:
                    speed_mult = 1.0

                delay_ms = max(16, int((real_time_per_sample / speed_mult) * 1000))

        self.animation_after_id = self.root.after(delay_ms, self._animate_step)

    def _update_frame_view_fast(self):
        """Update frame visualization with cached background to reduce flickering."""
        selected_file = self.viz_shoaling_file_var.get()
        if not selected_file or selected_file not in self.loaded_files:
            return

        loaded_file = self.loaded_files[selected_file]
        if loaded_file.shoaling_results is None:
            return

        results = loaded_file.shoaling_results
        sample_idx = self.frame_slider_var.get()

        if sample_idx >= len(results.frame_indices):
            return

        frame_idx = int(results.frame_indices[sample_idx])
        viz_mode = self.viz_mode_var.get()

        # Check if we need to rebuild the figure
        rebuild_needed = (
            self.frame_view_fig is None or
            not hasattr(self, '_cached_file') or 
            self._cached_file != selected_file or
            not hasattr(self, '_cached_mode') or
            self._cached_mode != viz_mode
        )

        if rebuild_needed:
            self._rebuild_frame_view_figure(selected_file, loaded_file, results, viz_mode)

        # Update dynamic elements
        self._update_frame_view_dynamic(selected_file, loaded_file, results, frame_idx, sample_idx, viz_mode)

    def _rebuild_frame_view_figure(self, selected_file, loaded_file, results, viz_mode):
        """Rebuild the frame view figure from scratch."""
        for widget in self.shoaling_frame_plot.winfo_children():
            widget.destroy()

        self.frame_view_fig = Figure(figsize=(10, 8), dpi=100)
        self.frame_view_ax_main = self.frame_view_fig.add_axes([0.1, 0.25, 0.85, 0.7])
        self.frame_view_ax_time = self.frame_view_fig.add_axes([0.1, 0.05, 0.85, 0.15])

        # Calculate coordinate bounds
        pixels_to_bl = 1.0 / loaded_file.metadata.body_length
        width_bl = loaded_file.metadata.video_width * pixels_to_bl
        height_bl = loaded_file.metadata.video_height * pixels_to_bl
        self._cached_width_bl = width_bl
        self._cached_height_bl = height_bl
        self._cached_pixels_to_bl = pixels_to_bl

        # Load and cache background image
        self._cached_background = None
        use_video = self.use_video_frames_var.get() if hasattr(self, 'use_video_frames_var') else False
        
        if not use_video and loaded_file.background_image_path and loaded_file.background_image_path.exists():
            try:
                self._cached_background = plt.imread(str(loaded_file.background_image_path))
            except Exception as e:
                print(f"Failed to load background: {e}")

        # Display background
        if use_video:
            self._video_background_artist = None
        elif self._cached_background is not None:
            if len(self._cached_background.shape) == 2:
                self.frame_view_ax_main.imshow(self._cached_background, 
                    extent=[0, width_bl, 0, height_bl],
                    aspect='equal', origin='upper', alpha=0.7, zorder=0, cmap='gray')
            else:
                self.frame_view_ax_main.imshow(self._cached_background, 
                    extent=[0, width_bl, 0, height_bl],
                    aspect='equal', origin='upper', alpha=0.7, zorder=0)

        # Set up main axis
        self.frame_view_ax_main.set_xlim(0, width_bl)
        self.frame_view_ax_main.set_ylim(0, height_bl)
        self.frame_view_ax_main.set_xlabel('X Position (BL)', fontsize=10)
        self.frame_view_ax_main.set_ylabel('Y Position (BL)', fontsize=10)
        self.frame_view_ax_main.set_aspect('equal')
        if self._cached_background is None:
            self.frame_view_ax_main.grid(True, alpha=0.3)

        # Set up time series axis
        time_minutes = results.timestamps / 60.0
        if viz_mode == 'nnd':
            self.frame_view_ax_time.plot(time_minutes, results.mean_nnd_per_sample, 'b-', linewidth=1, alpha=0.7)
            self.frame_view_ax_time.set_ylabel('NND (BL)', fontsize=8)
        elif viz_mode == 'iid':
            self.frame_view_ax_time.plot(time_minutes, results.mean_iid_per_sample, 'g-', linewidth=1, alpha=0.7)
            self.frame_view_ax_time.set_ylabel('IID (BL)', fontsize=8)
        else:
            self.frame_view_ax_time.plot(time_minutes, results.convex_hull_area_per_sample, 'r-', linewidth=1, alpha=0.7)
            self.frame_view_ax_time.set_ylabel('Hull (BL^2)', fontsize=8)
        self.frame_view_ax_time.set_xlabel('Time (minutes)', fontsize=8)
        self.frame_view_ax_time.set_xlim(time_minutes[0], time_minutes[-1])
        self.frame_view_ax_time.grid(True, alpha=0.3)

        # Create the time marker line
        self.frame_view_time_marker = self.frame_view_ax_time.axvline(x=0, color='red', linewidth=2, linestyle='--')

        # Create canvas
        self.frame_view_canvas = FigureCanvasTkAgg(self.frame_view_fig, master=self.shoaling_frame_plot)
        self.frame_view_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Cache state
        self._cached_file = selected_file
        self._cached_mode = viz_mode
        self._dynamic_artists = []

    def _update_frame_view_dynamic(self, selected_file, loaded_file, results, frame_idx, sample_idx, viz_mode):
        """Update only the dynamic elements of the frame view."""
        ax_main = self.frame_view_ax_main
        
        # Remove previous dynamic artists
        for artist in getattr(self, '_dynamic_artists', []):
            try:
                artist.remove()
            except:
                pass
        self._dynamic_artists = []

        # Update video frame if using video
        use_video = self.use_video_frames_var.get() if hasattr(self, 'use_video_frames_var') else False
        if use_video and selected_file in self.video_readers:
            self._update_video_frame(selected_file, frame_idx, ax_main)

        # Get positions
        params = ShoalingParameters(sample_interval_frames=1)
        calculator = ShoalingCalculator(loaded_file, params)
        positions_bl, is_complete = calculator.get_positions_at_frame(frame_idx)
        n_fish = loaded_file.n_fish
        colors = plt.cm.tab10(np.linspace(0, 1, n_fish))

        # Draw visualization overlays
        self._draw_frame_overlays(ax_main, positions_bl, n_fish, viz_mode, calculator, frame_idx)

        # Draw fish positions
        for i in range(n_fish):
            if not np.isnan(positions_bl[i, 0]):
                scatter = ax_main.scatter(positions_bl[i, 0], positions_bl[i, 1],
                              c=[colors[i]], s=200, edgecolors='black', linewidths=2, zorder=5)
                self._dynamic_artists.append(scatter)
                txt = ax_main.annotate(str(i), (positions_bl[i, 0], positions_bl[i, 1]),
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               color='white', zorder=6)
                self._dynamic_artists.append(txt)

        # Update title
        time_sec = results.timestamps[sample_idx]
        ax_main.set_title(f'Frame {frame_idx} | {time_sec:.1f}s | {viz_mode.upper()}',
                         fontsize=12, fontweight='bold')

        # Update time marker
        current_time_min = time_sec / 60.0
        self.frame_view_time_marker.set_xdata([current_time_min, current_time_min])

        self.frame_view_canvas.draw_idle()

    def _update_video_frame(self, selected_file, frame_idx, ax_main):
        """Update the video frame overlay."""
        quality = self.video_quality_var.get() if hasattr(self, 'video_quality_var') else "high"

        # Frame skipping for lower quality during playback
        if self.animation_running:
            if quality == "low":
                if not hasattr(self, '_frame_skip_counter'):
                    self._frame_skip_counter = 0
                self._frame_skip_counter += 1
                if self._frame_skip_counter % 3 != 0:
                    self.frame_view_canvas.draw_idle()
                    return
            elif quality == "medium":
                if not hasattr(self, '_frame_skip_counter'):
                    self._frame_skip_counter = 0
                self._frame_skip_counter += 1
                if self._frame_skip_counter % 2 != 0:
                    self.frame_view_canvas.draw_idle()
                    return

        reader = self.video_readers[selected_file]
        video_frame = reader.read_frame(frame_idx)
        
        if video_frame is not None:
            if quality == "low":
                video_frame = video_frame[::4, ::4]
            elif quality == "medium":
                video_frame = video_frame[::2, ::2]

            if hasattr(self, '_video_background_artist') and self._video_background_artist is not None:
                try:
                    self._video_background_artist.remove()
                except:
                    pass
            
            self._video_background_artist = ax_main.imshow(
                video_frame,
                extent=[0, self._cached_width_bl, 0, self._cached_height_bl],
                aspect='equal', origin='upper', alpha=0.6, zorder=0,
                interpolation='bilinear' if quality == "high" else 'nearest'
            )

    def _draw_frame_overlays(self, ax_main, positions_bl, n_fish, viz_mode, calculator, frame_idx):
        """Draw visualization overlays based on mode."""
        if viz_mode == 'nnd':
            for i in range(n_fish):
                if np.isnan(positions_bl[i, 0]):
                    continue
                min_dist = np.inf
                nn_idx = -1
                for j in range(n_fish):
                    if i == j or np.isnan(positions_bl[j, 0]):
                        continue
                    dist = np.sqrt((positions_bl[i, 0] - positions_bl[j, 0])**2 + 
                                  (positions_bl[i, 1] - positions_bl[j, 1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        nn_idx = j
                if nn_idx >= 0:
                    line, = ax_main.plot([positions_bl[i, 0], positions_bl[nn_idx, 0]],
                               [positions_bl[i, 1], positions_bl[nn_idx, 1]],
                               'b-', linewidth=1.5, alpha=0.5, zorder=2)
                    self._dynamic_artists.append(line)
                    mid_x = (positions_bl[i, 0] + positions_bl[nn_idx, 0]) / 2
                    mid_y = (positions_bl[i, 1] + positions_bl[nn_idx, 1]) / 2
                    txt = ax_main.text(mid_x, mid_y, f'{min_dist:.1f}', fontsize=7, 
                                      color='blue', ha='center', va='center',
                                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                                      zorder=3)
                    self._dynamic_artists.append(txt)
                               
        elif viz_mode == 'hull':
            hull_vertices = calculator.get_convex_hull_vertices(frame_idx)
            if hull_vertices is not None and len(hull_vertices) >= 3:
                hull_closed = np.vstack([hull_vertices, hull_vertices[0]])
                fill = ax_main.fill(hull_closed[:, 0], hull_closed[:, 1], 
                           alpha=0.2, color='green', zorder=1)
                self._dynamic_artists.extend(fill)
                line, = ax_main.plot(hull_closed[:, 0], hull_closed[:, 1], 
                           'g-', linewidth=2, alpha=0.7, zorder=2)
                self._dynamic_artists.append(line)
                           
        elif viz_mode == 'iid':
            try:
                focus_fish = int(self.iid_focus_fish_var.get())
            except:
                focus_fish = 0
            if focus_fish < n_fish and not np.isnan(positions_bl[focus_fish, 0]):
                for j in range(n_fish):
                    if j == focus_fish or np.isnan(positions_bl[j, 0]):
                        continue
                    dist = np.sqrt((positions_bl[focus_fish, 0] - positions_bl[j, 0])**2 + 
                                  (positions_bl[focus_fish, 1] - positions_bl[j, 1])**2)
                    line, = ax_main.plot([positions_bl[focus_fish, 0], positions_bl[j, 0]],
                               [positions_bl[focus_fish, 1], positions_bl[j, 1]],
                               'g-', linewidth=1.5, alpha=0.4, zorder=2)
                    self._dynamic_artists.append(line)
                    mid_x = (positions_bl[focus_fish, 0] + positions_bl[j, 0]) / 2
                    mid_y = (positions_bl[focus_fish, 1] + positions_bl[j, 1]) / 2
                    txt = ax_main.text(mid_x, mid_y, f'{dist:.1f}', fontsize=7, 
                                      color='darkgreen', ha='center', va='center',
                                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                                      zorder=3)
                    self._dynamic_artists.append(txt)

    # =========================================================================
    # VIDEO METHODS
    # =========================================================================

    def _browse_for_video(self):
        """Browse for and load video file for current visualization file."""
        selected_file = self.viz_shoaling_file_var.get()
        if not selected_file or selected_file not in self.loaded_files:
            messagebox.showwarning("No File", "Please select a file first.")
            return
        
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV"),
                ("All files", "*.*")
            ]
        )
        
        if not video_path:
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
            
            # Close existing reader if any
            if selected_file in self.video_readers:
                self.video_readers[selected_file].close()
            
            # Create new reader
            reader = VideoFrameReader(Path(video_path))
            self.video_readers[selected_file] = reader
            
            # Store path in loaded file
            self.loaded_files[selected_file].video_file_path = Path(video_path)
            
            # Update status
            self.video_status_label.config(
                text=f"✓ Video loaded: {reader.total_frames} frames",
                fg="green"
            )
            
            self.use_video_frames_var.set(True)
            self.frame_view_fig = None
            self._update_frame_view_fast()
            
            messagebox.showinfo(
                "Video Loaded",
                f"Video loaded successfully!\n\n"
                f"Resolution: {reader.width}x{reader.height}\n"
                f"Frames: {reader.total_frames}\n"
                f"FPS: {reader.fps:.1f}"
            )
            
        except Exception as e:
            messagebox.showerror("Error Loading Video", f"Failed to load video:\n\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _on_video_toggle(self):
        """Handle video frame toggle checkbox."""
        if self.use_video_frames_var.get():
            selected_file = self.viz_shoaling_file_var.get()
            if selected_file not in self.video_readers:
                messagebox.showinfo(
                    "No Video",
                    "Please browse for and load a video file first.\n\n"
                    "Click 'Browse for Video...' to select your video file."
                )
                self.use_video_frames_var.set(False)
                return
        
        self.frame_view_fig = None
        self._update_frame_view_fast()
