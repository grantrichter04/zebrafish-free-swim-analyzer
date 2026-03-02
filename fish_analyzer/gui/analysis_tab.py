"""
fish_analyzer/gui/analysis_tab.py
=================================
Analysis Tab Mixin - Individual trajectory metrics and visualizations.

This mixin provides all methods related to:
- Speed time series plots
- Speed distribution histograms
- Distance and sinuosity comparison
- Trajectory view (spaghetti and grid modes)
"""

from typing import List
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from .utils import smooth_time_series


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
            "2. Click 'Analyze All Files' to process\n"
            "3. Select files from the list and click 'Update Visualizations'\n\n"
            "Metrics calculated:\n"
            "- Total distance traveled\n"
            "- Mean/max speed\n"
            "- Net displacement (start to end)\n"
            "- Sinuosity (path tortuosity)\n"
            "- Turn angle statistics\n"
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

        # Distance Comparison tab
        distance_tab = ttk.Frame(results_notebook)
        results_notebook.add(distance_tab, text="Distance & Sinuosity")
        self.distance_comparison_frame = tk.Frame(distance_tab, bg="white")
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
                                      "Please run 'Analyze All Files' first.")
                return

        # Update all visualizations
        self._update_analysis_summary(selected_files)
        self._plot_speed_timeseries(selected_files)
        self._plot_speed_distribution(selected_files)
        self._plot_distance_comparison(selected_files)
        self._plot_trajectory_view(selected_files)

    def _update_analysis_summary(self, selected_files: List[str]):
        """Update the text summary of individual analysis."""
        self.analysis_summary_text.delete("1.0", tk.END)

        lines = []
        lines.append("=" * 90)
        lines.append("INDIVIDUAL TRAJECTORY ANALYSIS")
        lines.append("=" * 90)
        lines.append("")

        # Comparison table
        lines.append("COMPARISON TABLE (File Averages):")
        lines.append("-" * 90)
        lines.append(f"{'File':<18} {'Fish':<5} {'Distance':<12} {'Mean Speed':<12} {'Max Speed':<12} {'Sinuosity':<10}")
        lines.append("-" * 90)

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
            sinuosities = [f.metrics.get('sinuosity', np.nan) for f in fish_list]

            avg_distance = np.nanmean(distances)
            avg_mean_speed = np.nanmean(mean_speeds)
            avg_max_speed = np.nanmean(max_speeds)
            avg_sinuosity = np.nanmean(sinuosities)

            short_name = filename[:17] if len(filename) > 17 else filename
            lines.append(
                f"{short_name:<18} {len(fish_list):<5} "
                f"{avg_distance:<12.1f} {avg_mean_speed:<12.2f} "
                f"{avg_max_speed:<12.2f} {avg_sinuosity:<10.2f}"
            )

            all_file_data.append({
                'filename': filename, 'fish_list': fish_list, 'unit': unit
            })

        lines.append("-" * 90)
        lines.append("")

        # Detailed per-fish results
        lines.append("=" * 90)
        lines.append("DETAILED RESULTS BY FILE")
        lines.append("=" * 90)

        for data in all_file_data:
            filename = data['filename']
            fish_list = data['fish_list']
            unit = data['unit']
            loaded_file = self.loaded_files[filename]

            lines.append("")
            lines.append(f"FILE: {filename}")
            lines.append(f"Duration: {loaded_file.duration_minutes:.1f} min | Units: {unit} | Fish: {len(fish_list)}")
            lines.append("-" * 60)

            for fish in fish_list:
                m = fish.metrics
                lines.append(f"  Fish {fish.fish_id} ({fish.identity_label}):")
                lines.append(f"    Data quality: {fish.valid_percentage:.1%} valid frames")
                lines.append(f"    Total distance:  {m.get('total_distance', np.nan):.1f} {unit}")
                lines.append(f"    Mean speed:      {m.get('mean_speed', np.nan):.2f} {unit}/s")
                lines.append(f"    Sinuosity:       {m.get('sinuosity', np.nan):.2f}")
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
        except:
            smooth_seconds = 2.0

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

                # Apply smoothing to individual fish data
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
        except:
            n_bins = 30

        # Collect all speed data
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

        # Calculate global x-axis limits (speed range) across ALL fish
        all_speeds_combined = np.concatenate([info[2] for info in fish_info])
        global_x_min = 0
        global_x_max = np.percentile(all_speeds_combined, 99.5)
        
        # Create histograms with same bins for all fish
        global_bins = np.linspace(global_x_min, global_x_max, n_bins + 1)
        
        # Pre-compute all histograms to find global y-axis max
        all_hist_counts = []
        for filename, fish, speed, file_color, unit in fish_info:
            counts, _ = np.histogram(speed, bins=global_bins)
            all_hist_counts.append(counts)
        
        global_y_max = max(counts.max() for counts in all_hist_counts) * 1.1

        # Determine grid size
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

            short_name = filename[:8] if len(filename) > 8 else filename
            ax.set_title(f'{short_name} - Fish {fish.fish_id}', fontsize=10)
            ax.legend(fontsize=7, loc='upper right')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.speed_distribution_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_distance_comparison(self, selected_files: List[str]):
        """Plot distance and sinuosity comparison as bar charts."""
        for widget in self.distance_comparison_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(12, 5), dpi=100)

        file_names = []
        avg_distances = []
        avg_sinuosities = []
        units = []

        for filename in selected_files:
            loaded_file = self.loaded_files[filename]
            fish_list = loaded_file.processed_data
            unit = loaded_file.calibration.unit_name

            distances = [f.metrics.get('total_distance', np.nan) for f in fish_list]
            sinuosities = [f.metrics.get('sinuosity', np.nan) for f in fish_list]

            file_names.append(filename[:15] if len(filename) > 15 else filename)
            avg_distances.append(np.nanmean(distances))
            avg_sinuosities.append(np.nanmean(sinuosities))
            units.append(unit)

        x = np.arange(len(file_names))
        colors = plt.cm.tab10(np.linspace(0, 1, len(file_names)))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.bar(x, avg_distances, color=colors, edgecolor='black')
        ax1.set_xlabel('File', fontsize=12)
        ax1.set_ylabel(f'Total Distance ({units[0] if units else "units"})', fontsize=12)
        ax1.set_title('Total Distance Traveled', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(file_names, rotation=45, ha='right')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.bar(x, avg_sinuosities, color=colors, edgecolor='black')
        ax2.set_xlabel('File', fontsize=12)
        ax2.set_ylabel('Sinuosity', fontsize=12)
        ax2.set_title('Path Sinuosity', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(file_names, rotation=45, ha='right')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.distance_comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

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

                short_name = filename[:8] if len(filename) > 8 else filename
                ax.set_title(f'{short_name} - Fish {fish.fish_id}', fontsize=9)
                ax.set_aspect('equal')
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)

            fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.trajectory_view_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
