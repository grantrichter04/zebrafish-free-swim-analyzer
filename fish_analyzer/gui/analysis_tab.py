"""
fish_analyzer/gui/analysis_tab.py
=================================
Analysis Tab Mixin - Individual trajectory metrics and visualizations.

This mixin provides all methods related to:
- Speed time series plots
- Speed distribution histograms
- Distance and behavioral comparison
- Trajectory view (spaghetti and grid modes)
"""

from typing import List
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from .utils import smooth_time_series
from ..export import export_individual_metrics_csv


class AnalysisTabMixin:
    """
    Mixin providing Individual Analysis Tab functionality.

    Expects the following attributes from base class:
    - self.root: tk.Tk
    - self.notebook: ttk.Notebook
    - self.loaded_files: Dict[str, LoadedTrajectoryFile]
    """

    def _create_analysis_tab(self):
        """Create the individual analysis results tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Individual Analysis")

        # Split into left controls and right results
        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)

        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=3)

        # ----- LEFT PANEL: Controls -----
        self._create_analysis_controls(left_panel)

        # ----- RIGHT PANEL: Results -----
        self._create_analysis_results(right_panel)

    def _create_analysis_controls(self, parent):
        """Create the control panel for analysis tab."""
        # File selection
        select_frame = tk.LabelFrame(parent, text="Data Selection", font=("Arial", 11, "bold"))
        select_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(select_frame, text="Select files to compare:", font=("Arial", 9)).pack(anchor="w", padx=10, pady=2)
        tk.Label(select_frame, text="(Ctrl+click for multiple)", font=("Arial", 8), fg="gray").pack(anchor="w", padx=10)

        list_container = tk.Frame(select_frame)
        list_container.pack(fill="x", padx=10, pady=5)

        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")

        self.analysis_files_listbox = tk.Listbox(
            list_container, selectmode=tk.EXTENDED, height=5,
            yscrollcommand=scrollbar.set
        )
        self.analysis_files_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.analysis_files_listbox.yview)

        # Plot settings
        plot_frame = tk.LabelFrame(parent, text="Plot Settings", font=("Arial", 11, "bold"))
        plot_frame.pack(fill="x", padx=10, pady=10)

        smooth_frame = tk.Frame(plot_frame)
        smooth_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(smooth_frame, text="Smoothing:").pack(side="left")
        self.analysis_smooth_var = tk.StringVar(value="2.0")
        tk.Entry(smooth_frame, textvariable=self.analysis_smooth_var, width=6).pack(side="left", padx=5)
        tk.Label(smooth_frame, text="seconds").pack(side="left")

        self.show_individual_fish_var = tk.BooleanVar(value=True)
        tk.Checkbutton(plot_frame, text="Show individual fish",
                      variable=self.show_individual_fish_var).pack(anchor="w", padx=10)

        self.show_file_average_var = tk.BooleanVar(value=True)
        tk.Checkbutton(plot_frame, text="Show file average (bold)",
                      variable=self.show_file_average_var).pack(anchor="w", padx=10)

        tk.Button(
            parent, text="Update Visualizations",
            command=self._update_analysis_visualizations,
            bg="lightblue", font=("Arial", 11, "bold")
        ).pack(fill="x", padx=10, pady=10)

        tk.Button(
            parent, text="Export Results to CSV",
            command=self._export_individual_csv,
            font=("Arial", 10)
        ).pack(fill="x", padx=10, pady=(0, 10))

        # Distribution settings
        dist_frame = tk.LabelFrame(parent, text="Distribution Settings", font=("Arial", 11, "bold"))
        dist_frame.pack(fill="x", padx=10, pady=10)

        bins_frame = tk.Frame(dist_frame)
        bins_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(bins_frame, text="Histogram bins:").pack(side="left")
        self.histogram_bins_var = tk.StringVar(value="30")
        tk.Entry(bins_frame, textvariable=self.histogram_bins_var, width=6).pack(side="left", padx=5)

    def _create_analysis_results(self, parent):
        """Create the results panel for analysis tab."""
        results_notebook = ttk.Notebook(parent)
        results_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Summary tab
        summary_tab = ttk.Frame(results_notebook)
        results_notebook.add(summary_tab, text="Summary")

        summary_scroll = tk.Scrollbar(summary_tab)
        summary_scroll.pack(side="right", fill="y")

        self.analysis_summary_text = tk.Text(
            summary_tab, wrap=tk.WORD, font=("Courier", 10),
            yscrollcommand=summary_scroll.set
        )
        self.analysis_summary_text.pack(side="left", fill="both", expand=True)
        summary_scroll.config(command=self.analysis_summary_text.yview)

        self.analysis_summary_text.insert("1.0",
            "Individual trajectory analysis results will appear here.\n\n"
            "To run analysis:\n"
            "1. Load trajectory files in the 'Data Setup' tab\n"
            "2. Click 'Run Individual Trajectory Analysis' to process\n"
            "3. Select files from the list and click 'Update Visualizations'\n\n"
            "Metrics calculated:\n"
            "- Total distance traveled\n"
            "- Mean/max speed\n"
            "- Freeze count and duration (anxiety indicator)\n"
            "- Burst count and speed (locomotor vigor)\n"
            "- Angular velocity (turning rate)\n"
            "- Erratic movements (sudden direction changes)\n"
            "- Path straightness (how direct the swimming path is)\n"
        )

        # Speed Time Series tab
        speed_ts_tab = ttk.Frame(results_notebook)
        results_notebook.add(speed_ts_tab, text="Speed Over Time")
        self.speed_timeseries_frame = tk.Frame(speed_ts_tab, bg="white")
        self.speed_timeseries_frame.pack(fill="both", expand=True)

        # Speed Distribution tab
        speed_dist_tab = ttk.Frame(results_notebook)
        results_notebook.add(speed_dist_tab, text="Speed Distribution")
        self.speed_distribution_frame = tk.Frame(speed_dist_tab, bg="white")
        self.speed_distribution_frame.pack(fill="both", expand=True)

        # Behavioral Comparison tab (replaces "Distance & Sinuosity")
        behavior_tab = ttk.Frame(results_notebook)
        results_notebook.add(behavior_tab, text="Behavioral Comparison")
        self.distance_comparison_frame = tk.Frame(behavior_tab, bg="white")
        self.distance_comparison_frame.pack(fill="both", expand=True)

        # Trajectory View tab
        traj_tab = ttk.Frame(results_notebook)
        results_notebook.add(traj_tab, text="Trajectory View")

        traj_control_frame = tk.Frame(traj_tab)
        traj_control_frame.pack(fill="x", padx=5, pady=5)

        tk.Label(traj_control_frame, text="View mode:").pack(side="left", padx=5)
        self.trajectory_view_mode_var = tk.StringVar(value="spaghetti")
        tk.Radiobutton(traj_control_frame, text="Combined (spaghetti)",
                      variable=self.trajectory_view_mode_var, value="spaghetti").pack(side="left")
        tk.Radiobutton(traj_control_frame, text="Grid (individual fish)",
                      variable=self.trajectory_view_mode_var, value="grid").pack(side="left")

        tk.Button(traj_control_frame, text="Refresh",
                 command=self._refresh_trajectory_view).pack(side="left", padx=10)

        self.trajectory_view_frame = tk.Frame(traj_tab, bg="white")
        self.trajectory_view_frame.pack(fill="both", expand=True)

        # Methods tab
        methods_tab = ttk.Frame(results_notebook)
        results_notebook.add(methods_tab, text="Methods")
        methods_scroll = tk.Scrollbar(methods_tab)
        methods_scroll.pack(side="right", fill="y")
        self.analysis_methods_text = tk.Text(
            methods_tab, wrap=tk.WORD, font=("Courier", 10),
            yscrollcommand=methods_scroll.set
        )
        self.analysis_methods_text.pack(side="left", fill="both", expand=True)
        methods_scroll.config(command=self.analysis_methods_text.yview)
        self.analysis_methods_text.insert("1.0",
            "Methods text will appear here after running analysis.\n\n"
            "This provides a draft paragraph describing the analysis methods\n"
            "and parameters used, suitable for a manuscript methods section."
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _update_analysis_files_listbox(self):
        """Update the analysis tab file listbox."""
        self.analysis_files_listbox.delete(0, tk.END)
        for nickname in self.loaded_files.keys():
            self.analysis_files_listbox.insert(tk.END, nickname)

    def _update_analysis_visualizations(self):
        """Update all individual analysis visualizations for selected files."""
        selected_indices = self.analysis_files_listbox.curselection()

        if not selected_indices:
            messagebox.showinfo("No Selection", "Please select one or more files to visualize.")
            return

        file_keys = list(self.loaded_files.keys())
        selected_files = [file_keys[i] for i in selected_indices]

        # Check if files have been analyzed
        for filename in selected_files:
            loaded_file = self.loaded_files[filename]
            if not loaded_file.processed_data:
                messagebox.showwarning("Not Analyzed",
                                      f"'{filename}' has not been analyzed yet.\n"
                                      "Please run 'Run Individual Trajectory Analysis' from the Data Setup tab first.")
                return

        # Update all visualizations
        self._update_analysis_summary(selected_files)
        self._plot_speed_timeseries(selected_files)
        self._plot_speed_distribution(selected_files)
        self._plot_behavioral_comparison(selected_files)
        self._plot_trajectory_view(selected_files)
        self._update_analysis_methods_text(selected_files)

    def _update_analysis_summary(self, selected_files: List[str]):
        """Update the text summary of individual analysis."""
        self.analysis_summary_text.delete("1.0", tk.END)

        lines = []
        lines.append("=" * 100)
        lines.append("INDIVIDUAL TRAJECTORY ANALYSIS")
        lines.append("=" * 100)
        lines.append("")

        # Determine unit from first file
        first_file = self.loaded_files[selected_files[0]]
        u = first_file.calibration.unit_name

        # Comparison table
        lines.append("COMPARISON TABLE (File Averages):")
        lines.append("-" * 100)
        lines.append(
            f"{'File':<22} {'Fish':<5} "
            f"{'Distance':>10} {'MeanSpd':>9} {'MaxSpd':>9} "
            f"{'Freeze#':>8} {'FreezePct':>10} "
            f"{'Burst#':>8} {'Erratic#':>9} "
            f"{'Straight':>9} {'Lateral':>8}"
        )
        lines.append(
            f"{'':22} {'':5} "
            f"{'(' + u + ')':>10} {'(' + u + '/s)':>9} {'(' + u + '/s)':>9} "
            f"{'':>8} {'(%)':>10} "
            f"{'':>8} {'(/min)':>9} "
            f"{'(0-1)':>9} {'(-1..1)':>8}"
        )
        lines.append("-" * 110)

        all_file_data = []

        for filename in selected_files:
            loaded_file = self.loaded_files[filename]
            fish_list = loaded_file.processed_data
            unit = loaded_file.calibration.unit_name

            if not fish_list:
                continue

            distances = [f.metrics.get('total_distance', np.nan) for f in fish_list]
            mean_speeds = [f.metrics.get('mean_speed', np.nan) for f in fish_list]
            max_speeds = [f.metrics.get('max_speed', np.nan) for f in fish_list]
            freeze_counts = [f.metrics.get('freeze_count', 0) for f in fish_list]
            freeze_pcts = [f.metrics.get('freeze_fraction_pct', 0) for f in fish_list]
            burst_counts = [f.metrics.get('burst_count', 0) for f in fish_list]
            erratic_rates = [f.metrics.get('erratic_movements_per_min', 0) for f in fish_list]
            straightness = [f.metrics.get('mean_path_straightness', np.nan) for f in fish_list]
            laterality = [f.metrics.get('laterality_index', np.nan) for f in fish_list]

            short_name = filename[:21] if len(filename) > 21 else filename
            lines.append(
                f"{short_name:<22} {len(fish_list):<5} "
                f"{np.nanmean(distances):>10.1f} {np.nanmean(mean_speeds):>9.2f} {np.nanmean(max_speeds):>9.2f} "
                f"{np.mean(freeze_counts):>8.1f} {np.mean(freeze_pcts):>10.1f} "
                f"{np.mean(burst_counts):>8.1f} {np.mean(erratic_rates):>9.1f} "
                f"{np.nanmean(straightness):>9.2f} {np.nanmean(laterality):>8.2f}"
            )

            all_file_data.append({
                'filename': filename, 'fish_list': fish_list, 'unit': unit
            })

        lines.append("-" * 110)
        lines.append("")

        # Detailed per-fish results
        lines.append("=" * 110)
        lines.append("DETAILED RESULTS BY FILE")
        lines.append("=" * 100)

        for data in all_file_data:
            filename = data['filename']
            fish_list = data['fish_list']
            unit = data['unit']
            loaded_file = self.loaded_files[filename]

            lines.append("")
            lines.append(f"FILE: {filename}")
            lines.append(f"Duration: {loaded_file.duration_minutes:.1f} min | Units: {unit} | Fish: {len(fish_list)}")
            lines.append("-" * 70)

            for fish in fish_list:
                m = fish.metrics
                lines.append(f"  Fish {fish.fish_id} ({fish.identity_label}):")
                lines.append(f"    Data quality:       {fish.valid_percentage:.1%} valid frames")
                lines.append(f"    Total distance:     {m.get('total_distance', np.nan):.1f} {unit}")
                lines.append(f"    Mean speed:         {m.get('mean_speed', np.nan):.2f} {unit}/s")
                lines.append(f"    Max speed:          {m.get('max_speed', np.nan):.2f} {unit}/s")
                lines.append(f"    Path straightness:  {m.get('mean_path_straightness', np.nan):.2f} (1.0 = perfectly straight)")
                lines.append(f"    ---")
                lines.append(f"    Freeze episodes:    {m.get('freeze_count', 0)} "
                             f"(total {m.get('freeze_total_duration_s', 0):.1f}s, "
                             f"mean {m.get('freeze_mean_duration_s', 0):.1f}s, "
                             f"{m.get('freeze_fraction_pct', 0):.1f}% of time)")
                lines.append(f"    Burst events:       {m.get('burst_count', 0)} "
                             f"({m.get('burst_frequency_per_min', 0):.1f}/min, "
                             f"mean peak {m.get('burst_mean_speed', 0):.2f} {unit}/s)")
                lines.append(f"    Angular velocity:   {m.get('mean_angular_velocity_deg_s', np.nan):.1f} °/s")
                lines.append(f"    Erratic movements:  {m.get('erratic_movement_count', 0)} "
                             f"({m.get('erratic_movements_per_min', 0):.1f}/min)")
                lines.append(f"    ---")
                lat = m.get('laterality_index', np.nan)
                lat_label = "no bias"
                if not np.isnan(lat):
                    if lat > 0.1:
                        lat_label = "CW (right) bias"
                    elif lat < -0.1:
                        lat_label = "CCW (left) bias"
                lines.append(f"    Laterality index:   {lat:.3f} ({lat_label})")
                lines.append(f"    Turns:              {m.get('n_right_turns', 0)} right (CW), "
                             f"{m.get('n_left_turns', 0)} left (CCW)")
                lines.append(f"    Cumulative heading: {m.get('cumulative_heading_change_deg', np.nan):.1f}°")
                lines.append(f"    Signed ang. vel.:   {m.get('mean_signed_angular_velocity_deg_s', np.nan):.1f} °/s "
                             f"(+CCW / -CW)")
                lines.append("")

        self.analysis_summary_text.insert("1.0", "\n".join(lines))

    def _plot_speed_timeseries(self, selected_files: List[str]):
        """Plot speed over time for selected files."""
        for widget in self.speed_timeseries_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        try:
            smooth_seconds = float(self.analysis_smooth_var.get())
        except ValueError:
            smooth_seconds = 2.0
            self.analysis_smooth_var.set("2.0")
            messagebox.showwarning("Invalid Smoothing",
                                   "Smoothing value was not a valid number.\n"
                                   "Using default: 2.0 seconds.")

        show_individual = self.show_individual_fish_var.get()
        show_average = self.show_file_average_var.get()

        file_colors = plt.cm.tab10(np.linspace(0, 1, len(selected_files)))

        for file_idx, filename in enumerate(selected_files):
            loaded_file = self.loaded_files[filename]
            fish_list = loaded_file.processed_data
            file_color = file_colors[file_idx]
            frame_rate = loaded_file.calibration.frame_rate
            unit = loaded_file.calibration.unit_name

            all_speeds = []
            all_times = []

            for fish in fish_list:
                speed_data = fish.metrics.get('speed_time_series')
                if speed_data is None:
                    continue

                time = speed_data['time'].copy()
                speed = speed_data['speed'].copy()

                valid_mask = ~np.isnan(speed)
                time = time[valid_mask]
                speed = speed[valid_mask]

                if len(speed) == 0:
                    continue

                if smooth_seconds > 0 and len(speed) > 1:
                    speed = smooth_time_series(speed, smooth_seconds, frame_rate)

                time_minutes = time / 60.0
                all_speeds.append(speed)
                all_times.append(time_minutes)

                if show_individual:
                    ax.plot(time_minutes, speed, color=file_color, linewidth=0.8, alpha=0.4)

            if show_average and all_speeds:
                min_time = min(t.min() for t in all_times)
                max_time = max(t.max() for t in all_times)
                common_time = np.linspace(min_time, max_time, 1000)

                interpolated_speeds = []
                for t, s in zip(all_times, all_speeds):
                    interp_speed = np.interp(common_time, t, s)
                    interpolated_speeds.append(interp_speed)

                avg_speed = np.nanmean(interpolated_speeds, axis=0)
                ax.plot(common_time, avg_speed, color=file_color, linewidth=2.5, label=f'{filename}', alpha=0.9)

        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel(f'Speed ({unit}/s)', fontsize=12)
        ax.set_title('Swimming Speed Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.speed_timeseries_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_speed_distribution(self, selected_files: List[str]):
        """Plot speed distribution histograms with shared axes for comparison."""
        for widget in self.speed_distribution_frame.winfo_children():
            widget.destroy()

        try:
            n_bins = int(self.histogram_bins_var.get())
            if n_bins < 1:
                raise ValueError("Must be at least 1")
        except ValueError:
            n_bins = 30
            self.histogram_bins_var.set("30")
            messagebox.showwarning("Invalid Bins",
                                   "Histogram bins value was not valid.\n"
                                   "Using default: 30 bins.")

        fish_info = []
        file_colors = plt.cm.tab10(np.linspace(0, 1, len(selected_files)))

        for file_idx, filename in enumerate(selected_files):
            loaded_file = self.loaded_files[filename]
            fish_list = loaded_file.processed_data
            file_color = file_colors[file_idx]
            unit = loaded_file.calibration.unit_name

            for fish in fish_list:
                speed_data = fish.metrics.get('speed_time_series')
                if speed_data is None:
                    continue
                speed = speed_data['speed']
                speed = speed[~np.isnan(speed)]
                if len(speed) == 0:
                    continue
                fish_info.append((filename, fish, speed, file_color, unit))

        if not fish_info:
            return

        all_speeds_combined = np.concatenate([info[2] for info in fish_info])
        global_x_min = 0
        global_x_max = np.percentile(all_speeds_combined, 99.5)
        global_bins = np.linspace(global_x_min, global_x_max, n_bins + 1)

        all_hist_counts = []
        for filename, fish, speed, file_color, unit in fish_info:
            counts, _ = np.histogram(speed, bins=global_bins)
            all_hist_counts.append(counts)

        global_y_max = max(counts.max() for counts in all_hist_counts) * 1.1

        total_fish = len(fish_info)
        n_cols = min(3, total_fish)
        n_rows = (total_fish + n_cols - 1) // n_cols

        fig = Figure(figsize=(4 * n_cols, 3 * n_rows), dpi=100)

        for plot_idx, (filename, fish, speed, file_color, unit) in enumerate(fish_info):
            ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1)
            ax.hist(speed, bins=global_bins, color=file_color, alpha=0.7, edgecolor='black', linewidth=0.5)

            mean_speed = np.mean(speed)
            ax.axvline(mean_speed, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_speed:.2f}')

            ax.set_xlabel(f'Speed ({unit}/s)', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.set_xlim(global_x_min, global_x_max)
            ax.set_ylim(0, global_y_max)

            short_name = filename[:15] if len(filename) > 15 else filename
            ax.set_title(f'{short_name} - Fish {fish.fish_id}', fontsize=10)
            ax.legend(fontsize=7, loc='upper right')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.speed_distribution_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_behavioral_comparison(self, selected_files: List[str]):
        """Plot behavioral comparison as bar charts: distance, freeze%, burst rate, straightness."""
        for widget in self.distance_comparison_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(14, 8), dpi=100)

        file_names = []
        avg_distances = []
        avg_freeze_pcts = []
        avg_burst_rates = []
        avg_straightness = []
        avg_erratic_rates = []
        units = []

        for filename in selected_files:
            loaded_file = self.loaded_files[filename]
            fish_list = loaded_file.processed_data
            unit = loaded_file.calibration.unit_name

            distances = [f.metrics.get('total_distance', np.nan) for f in fish_list]
            freeze_pcts = [f.metrics.get('freeze_fraction_pct', 0) for f in fish_list]
            burst_rates = [f.metrics.get('burst_frequency_per_min', 0) for f in fish_list]
            straightness = [f.metrics.get('mean_path_straightness', np.nan) for f in fish_list]
            erratic_rates = [f.metrics.get('erratic_movements_per_min', 0) for f in fish_list]

            file_names.append(filename[:20] if len(filename) > 20 else filename)
            avg_distances.append(np.nanmean(distances))
            avg_freeze_pcts.append(np.mean(freeze_pcts))
            avg_burst_rates.append(np.mean(burst_rates))
            avg_straightness.append(np.nanmean(straightness))
            avg_erratic_rates.append(np.mean(erratic_rates))
            units.append(unit)

        x = np.arange(len(file_names))
        colors = plt.cm.tab10(np.linspace(0, 1, len(file_names)))

        # 2x3 grid of subplots
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.bar(x, avg_distances, color=colors, edgecolor='black')
        ax1.set_ylabel(f'Total Distance ({units[0] if units else "units"})', fontsize=10)
        ax1.set_title('Total Distance', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(file_names, rotation=45, ha='right', fontsize=8)

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.bar(x, avg_freeze_pcts, color=colors, edgecolor='black')
        ax2.set_ylabel('Time Frozen (%)', fontsize=10)
        ax2.set_title('Freezing', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(file_names, rotation=45, ha='right', fontsize=8)

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.bar(x, avg_burst_rates, color=colors, edgecolor='black')
        ax3.set_ylabel('Bursts / min', fontsize=10)
        ax3.set_title('Burst Frequency', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(file_names, rotation=45, ha='right', fontsize=8)

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.bar(x, avg_straightness, color=colors, edgecolor='black')
        ax4.set_ylabel('Straightness (0–1)', fontsize=10)
        ax4.set_title('Path Straightness', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(file_names, rotation=45, ha='right', fontsize=8)
        ax4.set_ylim(0, 1)
        ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.bar(x, avg_erratic_rates, color=colors, edgecolor='black')
        ax5.set_ylabel('Erratic Movements / min', fontsize=10)
        ax5.set_title('Erratic Movements', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(file_names, rotation=45, ha='right', fontsize=8)

        # Remove 6th subplot if unused
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.distance_comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # Keep old name as alias for backwards compatibility
    _plot_distance_comparison = _plot_behavioral_comparison

    def _refresh_trajectory_view(self):
        """Refresh trajectory view."""
        selected_indices = self.analysis_files_listbox.curselection()
        if not selected_indices:
            return
        file_keys = list(self.loaded_files.keys())
        selected_files = [file_keys[i] for i in selected_indices]
        self._plot_trajectory_view(selected_files)

    def _plot_trajectory_view(self, selected_files: List[str]):
        """Plot spatial trajectories."""
        for widget in self.trajectory_view_frame.winfo_children():
            widget.destroy()

        n_files = len(selected_files)
        if n_files == 0:
            return

        view_mode = self.trajectory_view_mode_var.get()

        if view_mode == "spaghetti":
            n_cols = min(2, n_files)
            n_rows = (n_files + n_cols - 1) // n_cols
            fig = Figure(figsize=(6 * n_cols, 5 * n_rows), dpi=100)

            for plot_idx, filename in enumerate(selected_files):
                ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1)

                loaded_file = self.loaded_files[filename]
                fish_list = loaded_file.processed_data
                unit = loaded_file.calibration.unit_name

                colors = plt.cm.tab10(np.linspace(0, 1, len(fish_list)))

                for fish, color in zip(fish_list, colors):
                    x = fish.trajectory['x'].values
                    y = fish.trajectory['y'].values
                    ax.plot(x, y, color=color, linewidth=0.8, alpha=0.7, label=f'Fish {fish.fish_id}')

                ax.set_xlabel(f'X ({unit})', fontsize=10)
                ax.set_ylabel(f'Y ({unit})', fontsize=10)
                ax.set_title(f'{filename}', fontsize=12, fontweight='bold')
                ax.set_aspect('equal')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)

            fig.tight_layout()
        else:
            all_fish_data = []
            for filename in selected_files:
                loaded_file = self.loaded_files[filename]
                if loaded_file.processed_data:
                    for fish in loaded_file.processed_data:
                        all_fish_data.append((filename, loaded_file, fish))

            if not all_fish_data:
                return

            n_fish_total = len(all_fish_data)
            n_cols = min(4, n_fish_total)
            n_rows = (n_fish_total + n_cols - 1) // n_cols

            fig = Figure(figsize=(3.5 * n_cols, 3 * n_rows), dpi=100)

            for plot_idx, (filename, loaded_file, fish) in enumerate(all_fish_data):
                ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1)
                unit = loaded_file.calibration.unit_name

                x = fish.trajectory['x'].values
                y = fish.trajectory['y'].values

                file_idx = selected_files.index(filename)
                file_color = plt.cm.tab10(file_idx / max(1, len(selected_files) - 1))

                ax.plot(x, y, color=file_color, linewidth=0.6, alpha=0.6)

                short_name = filename[:15] if len(filename) > 15 else filename
                ax.set_title(f'{short_name} - Fish {fish.fish_id}', fontsize=9)
                ax.set_aspect('equal')
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)

            fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.trajectory_view_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # =========================================================================
    # METHODS TEXT
    # =========================================================================

    def _update_analysis_methods_text(self, selected_files: List[str]):
        """Generate a methods paragraph for the individual trajectory analysis."""
        self.analysis_methods_text.delete("1.0", tk.END)

        # Gather parameters from the first analyzed file
        first_file = None
        for f in selected_files:
            if f in self.loaded_files and self.loaded_files[f].processed_data:
                first_file = self.loaded_files[f]
                break
        if first_file is None:
            return

        cal = first_file.calibration
        p = self.processing_params
        n_files = len(selected_files)
        total_fish = sum(len(self.loaded_files[f].processed_data)
                         for f in selected_files
                         if f in self.loaded_files and self.loaded_files[f].processed_data)

        smooth_text = (
            f"Position data were smoothed using a Savitzky-Golay filter "
            f"(window = {p.smoothing_window} frames, polynomial order = "
            f"{p.smoothing_polynomial_order}) prior to metric computation. "
            f"NaN gaps were linearly interpolated before smoothing and restored after."
        ) if p.apply_smoothing else (
            "No position smoothing was applied to preserve the temporal "
            "resolution of bout-based locomotion."
        )

        text = (
            "METHODS — Individual Trajectory Analysis\n"
            "========================================\n\n"
            "The following is a draft methods paragraph. Edit as needed.\n\n"
            "---\n\n"
            f"Zebrafish trajectories were tracked using idtracker.ai and "
            f"analyzed using the Zebrafish Free Swim Analyzer (v2.1). "
            f"Trajectory coordinates were converted from pixels to "
            f"{cal.unit_name} using a calibration factor of "
            f"{cal.scale_factor:.6f} {cal.unit_name}/pixel "
            f"(frame rate: {cal.frame_rate:.1f} fps). "
            f"{smooth_text} "
            f"Fish with fewer than {p.min_valid_points} valid position "
            f"frames were excluded from analysis.\n\n"
            f"For each fish, the following metrics were computed: total "
            f"distance traveled (sum of frame-to-frame displacements), "
            f"mean and maximum speed, and path straightness (ratio of "
            f"net displacement to path distance over sliding "
            f"{p.straightness_window_seconds:.1f}-second windows; "
            f"1.0 = straight, 0.0 = circling). "
            f"Freezing episodes were defined as consecutive periods where "
            f"speed remained below {p.rest_speed_threshold} {cal.unit_name}/s "
            f"for at least {p.min_freeze_frames} frames "
            f"({p.min_freeze_frames / cal.frame_rate * 1000:.0f} ms). "
            f"Burst events were detected when frame-to-frame acceleration "
            f"exceeded {p.burst_accel_threshold} {cal.unit_name}/s\u00b2. "
            f"Angular velocity was computed as the absolute frame-to-frame "
            f"heading change (degrees/second), restricted to frames where "
            f"the fish was actively moving. Erratic movements were counted "
            f"as heading changes exceeding {p.erratic_turn_threshold}\u00b0 "
            f"during active movement. "
            f"Turning bias was quantified using a laterality index "
            f"(right turns \u2212 left turns) / total turns, where positive "
            f"values indicate clockwise bias and negative values indicate "
            f"counterclockwise bias.\n\n"
            f"Analysis included {n_files} file(s) with {total_fish} fish total.\n\n"
            "---\n\n"
            "KEY PARAMETERS:\n"
            f"  Calibration unit:          {cal.unit_name}\n"
            f"  Scale factor:              {cal.scale_factor:.6f} {cal.unit_name}/px\n"
            f"  Frame rate:                {cal.frame_rate:.1f} fps\n"
            f"  Smoothing:                 {'ON (window=' + str(p.smoothing_window) + ')' if p.apply_smoothing else 'OFF'}\n"
            f"  Rest speed threshold:      {p.rest_speed_threshold} {cal.unit_name}/s\n"
            f"  Min freeze duration:       {p.min_freeze_frames} frames ({p.min_freeze_frames / cal.frame_rate * 1000:.0f} ms)\n"
            f"  Burst accel threshold:     {p.burst_accel_threshold} {cal.unit_name}/s\u00b2\n"
            f"  Erratic turn threshold:    {p.erratic_turn_threshold}\u00b0\n"
            f"  Path straightness window:  {p.straightness_window_seconds} s\n"
            f"  Min valid frames:          {p.min_valid_points}\n"
        )

        self.analysis_methods_text.insert("1.0", text)

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def _export_individual_csv(self):
        """Export individual trajectory metrics to a CSV file."""
        analyzed = {k: v for k, v in self.loaded_files.items() if v.processed_data}
        if not analyzed:
            messagebox.showwarning("No Data",
                                   "No files have been analyzed yet.\n"
                                   "Run 'Run Individual Trajectory Analysis' first.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Export Individual Metrics CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="individual_metrics.csv"
        )
        if not output_path:
            return

        try:
            n_rows = export_individual_metrics_csv(analyzed, Path(output_path))
            self.set_status(f"Exported {n_rows} fish to {Path(output_path).name}")
            messagebox.showinfo("Export Complete",
                                f"Exported {n_rows} fish rows to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{e}")
