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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

from .utils import smooth_time_series, create_sortable_treeview, embed_figure_with_toolbar
from ..export import export_individual_metrics_csv, export_combined_summary_csv


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
        ).pack(fill="x", padx=10, pady=(0, 4))

        tk.Button(
            parent, text="Export Combined Summary (per fish)",
            command=self._export_combined_summary,
            bg="#c8e6c9", font=("Arial", 10, "bold")
        ).pack(fill="x", padx=10, pady=(0, 10))

        # Distribution settings
        dist_frame = tk.LabelFrame(parent, text="Distribution Settings", font=("Arial", 11, "bold"))
        dist_frame.pack(fill="x", padx=10, pady=10)

        # View mode
        tk.Label(dist_frame, text="View mode:", font=("Arial", 9)).pack(anchor="w", padx=10, pady=(5, 0))
        self.speed_dist_mode_var = tk.StringVar(value="ridge")
        tk.Radiobutton(dist_frame, text="Histograms (grid)",
                       variable=self.speed_dist_mode_var, value="histogram").pack(anchor="w", padx=20)
        tk.Radiobutton(dist_frame, text="Ridge plot (joy plot)",
                       variable=self.speed_dist_mode_var, value="ridge").pack(anchor="w", padx=20)

        bins_frame = tk.Frame(dist_frame)
        bins_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(bins_frame, text="Histogram bins:").pack(side="left")
        self.histogram_bins_var = tk.StringVar(value="30")
        tk.Entry(bins_frame, textvariable=self.histogram_bins_var, width=6).pack(side="left", padx=5)

        # Collapse level
        ttk.Separator(dist_frame, orient="horizontal").pack(fill="x", padx=10, pady=(8, 0))
        tk.Label(dist_frame, text="Collapse level:", font=("Arial", 9)).pack(anchor="w", padx=10, pady=(5, 0))
        self.speed_dist_collapse_var = tk.StringVar(value="individual")
        tk.Radiobutton(dist_frame, text="Individual fish",
                       variable=self.speed_dist_collapse_var, value="individual").pack(anchor="w", padx=20)
        tk.Radiobutton(dist_frame, text="Per video (pool fish)",
                       variable=self.speed_dist_collapse_var, value="video").pack(anchor="w", padx=20)
        tk.Radiobutton(dist_frame, text="Per group (pool videos)",
                       variable=self.speed_dist_collapse_var, value="group").pack(anchor="w", padx=20)
        tk.Button(dist_frame, text="Assign Groups...",
                  command=self._assign_groups_dialog,
                  font=("Arial", 8)).pack(anchor="w", padx=20, pady=(4, 6))

    def _create_analysis_results(self, parent):
        """Create the results panel for analysis tab."""
        results_notebook = ttk.Notebook(parent)
        results_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Summary tab (uses Treeview tables)
        summary_tab = ttk.Frame(results_notebook)
        results_notebook.add(summary_tab, text="Summary")

        self.analysis_summary_frame = tk.Frame(summary_tab)
        self.analysis_summary_frame.pack(fill="both", expand=True)

        tk.Label(self.analysis_summary_frame,
            text="Individual trajectory analysis results will appear here.\n\n"
            "To run analysis:\n"
            "1. Load trajectory files in the 'Data Setup' tab\n"
            "2. Click 'Run All Analysis' to process\n"
            "3. Select files from the list and click 'Update Visualizations'",
            font=("Arial", 10), fg="gray", justify=tk.LEFT
        ).pack(padx=20, pady=20)

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
        """Update the summary tab with Treeview tables."""
        for widget in self.analysis_summary_frame.winfo_children():
            widget.destroy()

        first_file = self.loaded_files[selected_files[0]]
        u = first_file.calibration.unit_name

        # --- File Averages Table ---
        file_columns = [
            ("file", "File", 150), ("fish", "Fish", 50),
            ("distance", f"Distance ({u})", 90), ("mean_spd", f"Mean Spd ({u}/s)", 90),
            ("max_spd", f"Max Spd ({u}/s)", 90), ("freeze_n", "Freeze #", 70),
            ("freeze_pct", "Freeze %", 70), ("burst_n", "Burst #", 70),
            ("erratic", "Erratic /min", 80), ("straight", "Straightness", 80),
            ("lateral", "Laterality", 70),
        ]

        file_rows = []
        all_fish_rows = []

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

            file_rows.append((
                filename, len(fish_list),
                f"{np.nanmean(distances):.1f}", f"{np.nanmean(mean_speeds):.2f}",
                f"{np.nanmean(max_speeds):.2f}", f"{np.mean(freeze_counts):.1f}",
                f"{np.mean(freeze_pcts):.1f}", f"{np.mean(burst_counts):.1f}",
                f"{np.mean(erratic_rates):.1f}", f"{np.nanmean(straightness):.2f}",
                f"{np.nanmean(laterality):.2f}",
            ))

            # Per-fish rows
            for fish in fish_list:
                m = fish.metrics
                lat = m.get('laterality_index', np.nan)
                all_fish_rows.append((
                    filename, f"Fish {fish.fish_id}",
                    f"{fish.valid_percentage:.0%}",
                    f"{m.get('total_distance', np.nan):.1f}",
                    f"{m.get('mean_speed', np.nan):.2f}",
                    f"{m.get('max_speed', np.nan):.2f}",
                    f"{m.get('freeze_count', 0)}",
                    f"{m.get('freeze_fraction_pct', 0):.1f}",
                    f"{m.get('burst_count', 0)}",
                    f"{m.get('burst_frequency_per_min', 0):.1f}",
                    f"{m.get('erratic_movements_per_min', 0):.1f}",
                    f"{m.get('mean_path_straightness', np.nan):.2f}",
                    f"{m.get('mean_angular_velocity_deg_s', np.nan):.1f}",
                    f"{lat:.3f}" if not np.isnan(lat) else "N/A",
                ))

        create_sortable_treeview(self.analysis_summary_frame, file_columns, file_rows,
                                 title="File Averages")

        # --- Per-Fish Details Table ---
        fish_columns = [
            ("file", "File", 130), ("fish", "Fish", 60),
            ("quality", "Valid %", 60),
            ("distance", f"Distance ({u})", 85), ("mean_spd", f"Mean Spd", 75),
            ("max_spd", f"Max Spd", 70), ("freeze_n", "Freeze #", 65),
            ("freeze_pct", "Freeze %", 65), ("burst_n", "Burst #", 60),
            ("burst_rate", "Burst /min", 70), ("erratic", "Erratic /min", 75),
            ("straight", "Straight", 65), ("ang_vel", "Ang Vel", 65),
            ("lateral", "Lateral", 60),
        ]

        create_sortable_treeview(self.analysis_summary_frame, fish_columns, all_fish_rows,
                                 title="Per-Fish Details")

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

        embed_figure_with_toolbar(fig, self.speed_timeseries_frame)

    def _auto_detect_group(self, filename: str) -> str:
        """Infer a group label by stripping trailing numbers from a nickname.

        E.g. 'Control_01' → 'Control', 'Exp2' → 'Exp'.
        Falls back to the full filename if nothing is stripped.
        """
        import re
        result = re.sub(r'[_\-\s]*\d+$', '', filename).strip()
        return result if result else filename

    def _assign_groups_dialog(self):
        """Open a dialog for the user to assign group labels to loaded files."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Assign Group Labels")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Assign each file to a group:",
                 font=("Arial", 10, "bold")).pack(pady=(12, 2), padx=15)
        tk.Label(dialog, text="Files sharing the same group name will be pooled together.",
                 font=("Arial", 8), fg="gray").pack(padx=15)

        grid_frame = tk.Frame(dialog)
        grid_frame.pack(fill="both", expand=True, padx=20, pady=10)

        tk.Label(grid_frame, text="File", font=("Arial", 9, "bold"), anchor="w").grid(
            row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Label(grid_frame, text="Group label", font=("Arial", 9, "bold"), anchor="w").grid(
            row=0, column=1, sticky="w", padx=5, pady=2)

        entries = {}
        for i, filename in enumerate(self.loaded_files.keys(), start=1):
            short = filename if len(filename) <= 22 else filename[:19] + "..."
            tk.Label(grid_frame, text=short, anchor="w", width=24).grid(
                row=i, column=0, padx=5, pady=3, sticky="w")
            current = self.file_groups.get(filename) or self._auto_detect_group(filename)
            var = tk.StringVar(value=current)
            tk.Entry(grid_frame, textvariable=var, width=22).grid(
                row=i, column=1, padx=5, pady=3)
            entries[filename] = var

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(fill="x", padx=20, pady=(4, 12))

        def _apply():
            for fname, var in entries.items():
                label = var.get().strip()
                self.file_groups[fname] = label if label else self._auto_detect_group(fname)
            dialog.destroy()

        tk.Button(btn_frame, text="Apply", command=_apply,
                  bg="lightblue", font=("Arial", 9)).pack(side="right", padx=4)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy,
                  font=("Arial", 9)).pack(side="right")

        dialog.update_idletasks()
        dialog.geometry(f"{dialog.winfo_reqwidth()}x{dialog.winfo_reqheight()}")

    def _build_collapsed_info(self, fish_info, selected_files: List[str], collapse_level: str):
        """Pool fish speeds to per-video or per-group level.

        Returns a list of (label, speed_array, color, unit) tuples.
        """
        from collections import OrderedDict

        if collapse_level == "video":
            file_colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected_files), 1)))
            color_map = {fn: file_colors[i] for i, fn in enumerate(selected_files)}

            video_data: OrderedDict = OrderedDict()
            for filename, fish, speed, _, unit in fish_info:
                if filename not in video_data:
                    video_data[filename] = {'speeds': [], 'color': color_map[filename], 'unit': unit}
                video_data[filename]['speeds'].append(speed)

            return [
                (fn, np.concatenate(d['speeds']), d['color'], d['unit'])
                for fn, d in video_data.items()
            ]

        elif collapse_level == "group":
            # Resolve group for each selected file
            file_to_group = {
                fn: (self.file_groups.get(fn) or self._auto_detect_group(fn))
                for fn in selected_files
            }
            unique_groups = list(dict.fromkeys(file_to_group.values()))
            group_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_groups), 1)))
            group_color_map = {g: group_colors[i] for i, g in enumerate(unique_groups)}

            group_data: OrderedDict = OrderedDict()
            for filename, fish, speed, _, unit in fish_info:
                group = file_to_group.get(filename, filename)
                if group not in group_data:
                    group_data[group] = {'speeds': [], 'color': group_color_map[group], 'unit': unit}
                group_data[group]['speeds'].append(speed)

            return [
                (grp, np.concatenate(d['speeds']), d['color'], d['unit'])
                for grp, d in group_data.items()
            ]

        return []

    def _plot_speed_distribution(self, selected_files: List[str]):
        """Plot speed distribution as histograms or ridge plot."""
        for widget in self.speed_distribution_frame.winfo_children():
            widget.destroy()

        try:
            n_bins = int(self.histogram_bins_var.get())
            if n_bins < 1:
                raise ValueError("Must be at least 1")
        except ValueError:
            n_bins = 30
            self.histogram_bins_var.set("30")

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

        view_mode = self.speed_dist_mode_var.get()
        collapse_level = self.speed_dist_collapse_var.get()

        if collapse_level == "individual":
            if view_mode == "ridge":
                self._plot_speed_ridge(fish_info, n_bins)
            else:
                self._plot_speed_histograms(fish_info, n_bins)
        else:
            collapsed_info = self._build_collapsed_info(fish_info, selected_files, collapse_level)
            if not collapsed_info:
                return
            if view_mode == "ridge":
                self._plot_speed_collapsed_ridge(collapsed_info)
            else:
                self._plot_speed_collapsed_histograms(collapsed_info, n_bins)

    def _plot_speed_histograms(self, fish_info, n_bins):
        """Plot traditional histogram grid."""
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
        embed_figure_with_toolbar(fig, self.speed_distribution_frame)

    def _plot_speed_ridge(self, fish_info, n_bins):
        """Plot overlapping KDE ridge (joy) plot.

        Fish from the same file are grouped tightly together with consistent
        color.  A visual gap separates different files for easy comparison.
        """
        from scipy.stats import gaussian_kde

        all_speeds = np.concatenate([info[2] for info in fish_info])
        x_min = 0
        x_max = np.percentile(all_speeds, 99.5)
        x_grid = np.linspace(x_min, x_max, 200)

        # --- Group fish by file (preserve order) ---
        from collections import OrderedDict
        file_groups = OrderedDict()
        for filename, fish, speed, file_color, unit in fish_info:
            if filename not in file_groups:
                file_groups[filename] = []
            file_groups[filename].append((filename, fish, speed, file_color, unit))

        n_files = len(file_groups)
        n_total = len(fish_info)

        # Spacing: tight within-file, larger gap between files
        within_overlap = 0.45   # vertical step between fish in same file
        between_gap = 0.3       # extra gap between file groups

        # Compute all KDEs first to find global max for normalization
        kde_curves = {}
        for filename, fish, speed, file_color, unit in fish_info:
            key = (filename, fish.fish_id)
            try:
                kde = gaussian_kde(speed, bw_method=0.15)
                density = kde(x_grid)
            except Exception:
                density = np.zeros_like(x_grid)
            kde_curves[key] = density

        max_density = max(d.max() for d in kde_curves.values()) if kde_curves else 1.0

        # Build the layout: bottom to top (reversed so first file is at top)
        ordered_entries = []  # (y_offset, filename, fish, speed, file_color, unit, density)
        y_pos = 0.0
        for file_idx, (filename, group) in enumerate(reversed(list(file_groups.items()))):
            for fish_idx, (fn, fish, speed, file_color, unit) in enumerate(reversed(group)):
                key = (fn, fish.fish_id)
                density = kde_curves[key]
                ordered_entries.append((y_pos, fn, fish, speed, file_color, unit, density))
                y_pos += within_overlap
            # Add extra gap between file groups (but not after the last one)
            if file_idx < n_files - 1:
                y_pos += between_gap

        total_height = y_pos
        fig_height = max(4, total_height * 1.3 + 2)
        fig = Figure(figsize=(10, fig_height), dpi=100)
        ax = fig.add_subplot(111)

        # Track file boundaries for divider lines
        prev_filename = None
        for y_offset, filename, fish, speed, file_color, unit, density in ordered_entries:
            # Normalize density so peaks are comparable
            normalized = density / max_density * within_overlap * 1.5

            ax.fill_between(x_grid, y_offset, y_offset + normalized,
                           color=file_color, alpha=0.6, linewidth=0)
            ax.plot(x_grid, y_offset + normalized, color=file_color,
                   linewidth=1.2, alpha=0.9)

            # Label: show filename only for first fish in each group
            short_name = filename[:15] if len(filename) > 15 else filename
            if prev_filename != filename:
                label = f"{short_name}\n  F{fish.fish_id}"
            else:
                label = f"  F{fish.fish_id}"
            ax.text(-0.02 * x_max, y_offset + within_overlap * 0.2, label,
                   fontsize=8, ha='right', va='center', transform=ax.transData)

            prev_filename = filename

        # Draw thin separator lines between file groups
        if n_files > 1:
            y_pos = 0.0
            for file_idx, (filename, group) in enumerate(reversed(list(file_groups.items()))):
                y_pos += len(group) * within_overlap
                if file_idx < n_files - 1:
                    sep_y = y_pos + between_gap * 0.5
                    ax.axhline(sep_y, color='gray', linewidth=0.5,
                              linestyle=':', alpha=0.5)
                    y_pos += between_gap

        unit = fish_info[0][4] if fish_info else "BL"
        ax.set_xlabel(f'Speed ({unit}/s)', fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.1, total_height + within_overlap)
        ax.set_yticks([])
        ax.set_title('Speed Distribution (Ridge Plot)', fontsize=13, fontweight='bold')
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        embed_figure_with_toolbar(fig, self.speed_distribution_frame)

    def _plot_speed_collapsed_ridge(self, collapsed_info):
        """Ridge plot with one KDE row per video or group.

        collapsed_info: [(label, speed_array, color, unit), ...]
        """
        from scipy.stats import gaussian_kde

        all_speeds = np.concatenate([info[1] for info in collapsed_info])
        x_min = 0.0
        x_max = np.percentile(all_speeds, 99.5)
        if x_max <= x_min:
            x_max = x_min + 1.0
        x_grid = np.linspace(x_min, x_max, 300)

        kde_curves = []
        for label, speed, color, unit in collapsed_info:
            try:
                kde_curves.append(gaussian_kde(speed, bw_method=0.15)(x_grid))
            except Exception:
                kde_curves.append(np.zeros_like(x_grid))

        max_density = max(d.max() for d in kde_curves) if kde_curves else 1.0
        if max_density == 0:
            max_density = 1.0

        within_overlap = 0.55  # slightly more space per row (fewer rows than individual fish)
        n = len(collapsed_info)
        total_height = n * within_overlap

        fig = Figure(figsize=(10, max(4, total_height * 1.5 + 2)), dpi=100)
        ax = fig.add_subplot(111)

        # Draw bottom-to-top so the first entry appears at the top
        for i, ((label, speed, color, unit), density) in enumerate(
                zip(reversed(collapsed_info), reversed(kde_curves))):
            y0 = i * within_overlap
            norm = density / max_density * within_overlap * 1.5
            ax.fill_between(x_grid, y0, y0 + norm, color=color, alpha=0.7, linewidth=0)
            ax.plot(x_grid, y0 + norm, color=color, linewidth=1.5, alpha=0.95)
            ax.text(x_min - 0.02 * x_max, y0 + within_overlap * 0.3,
                    label, fontsize=9, ha='right', va='center')

        unit_label = collapsed_info[0][3] if collapsed_info else "BL"
        ax.set_xlabel(f'Speed ({unit_label}/s)', fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.1, total_height + within_overlap)
        ax.set_yticks([])
        ax.set_title('Speed Distribution (Ridge Plot)', fontsize=13, fontweight='bold')
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        embed_figure_with_toolbar(fig, self.speed_distribution_frame)

    def _plot_speed_collapsed_histograms(self, collapsed_info, n_bins):
        """Histogram grid with one subplot per video or group.

        collapsed_info: [(label, speed_array, color, unit), ...]
        """
        all_speeds = np.concatenate([info[1] for info in collapsed_info])
        x_max = np.percentile(all_speeds, 99.5)
        global_bins = np.linspace(0, x_max, n_bins + 1)

        all_counts = [np.histogram(info[1], bins=global_bins)[0] for info in collapsed_info]
        y_max = max(c.max() for c in all_counts) * 1.1

        n = len(collapsed_info)
        n_cols = min(3, n)
        n_rows = (n + n_cols - 1) // n_cols

        fig = Figure(figsize=(4 * n_cols, 3 * n_rows), dpi=100)

        for plot_idx, (label, speed, color, unit) in enumerate(collapsed_info):
            ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1)
            ax.hist(speed, bins=global_bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            mean_speed = np.mean(speed)
            ax.axvline(mean_speed, color='red', linestyle='--', linewidth=1.5,
                       label=f'Mean: {mean_speed:.2f}')
            ax.set_xlabel(f'Speed ({unit}/s)', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.set_title(label, fontsize=10)
            ax.legend(fontsize=7, loc='upper right')

        fig.tight_layout()
        embed_figure_with_toolbar(fig, self.speed_distribution_frame)

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

        embed_figure_with_toolbar(fig, self.distance_comparison_frame)

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

                # Calculate full arena dimensions
                pixels_to_bl = loaded_file.calibration.scale_factor
                width = loaded_file.metadata.video_width * pixels_to_bl
                height = loaded_file.metadata.video_height * pixels_to_bl

                colors = plt.cm.tab10(np.linspace(0, 1, len(fish_list)))

                for fish, color in zip(fish_list, colors):
                    x = fish.trajectory['x'].values
                    y = fish.trajectory['y'].values
                    ax.plot(x, y, color=color, linewidth=0.8, alpha=0.7, label=f'Fish {fish.fish_id}')

                ax.set_xlim(0, width)
                ax.set_ylim(0, height)
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

                # Calculate full arena dimensions
                pixels_to_bl = loaded_file.calibration.scale_factor
                width = loaded_file.metadata.video_width * pixels_to_bl
                height = loaded_file.metadata.video_height * pixels_to_bl

                x = fish.trajectory['x'].values
                y = fish.trajectory['y'].values

                file_idx = selected_files.index(filename)
                file_color = plt.cm.tab10(file_idx / max(1, len(selected_files) - 1))

                ax.plot(x, y, color=file_color, linewidth=0.6, alpha=0.6)

                ax.set_xlim(0, width)
                ax.set_ylim(0, height)
                short_name = filename[:15] if len(filename) > 15 else filename
                ax.set_title(f'{short_name} - Fish {fish.fish_id}', fontsize=9)
                ax.set_aspect('equal')
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)

            fig.tight_layout()

        embed_figure_with_toolbar(fig, self.trajectory_view_frame)

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
            n_rows = export_individual_metrics_csv(
                analyzed, Path(output_path), file_groups=self.file_groups
            )
            self.set_status(f"Exported {n_rows} fish to {Path(output_path).name}")
            messagebox.showinfo("Export Complete",
                                f"Exported {n_rows} fish rows to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{e}")

    def _export_combined_summary(self):
        """Export combined per-fish summary: trajectory metrics + bout summary + Group."""
        analyzed = {k: v for k, v in self.loaded_files.items() if v.processed_data}
        if not analyzed:
            messagebox.showwarning("No Data",
                                   "No files have been analyzed yet.\n"
                                   "Run 'Run Individual Trajectory Analysis' first.")
            return

        bout_results = getattr(self, 'bout_results', {})
        if not bout_results:
            if not messagebox.askyesno(
                "Bout Data Missing",
                "Bout analysis has not been run.\n\n"
                "The Bout_* columns will be empty (NaN).\n\n"
                "Export trajectory metrics only?"
            ):
                return

        output_path = filedialog.asksaveasfilename(
            title="Export Combined Summary CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="combined_summary.csv"
        )
        if not output_path:
            return

        try:
            n_rows = export_combined_summary_csv(
                analyzed, bout_results, self.file_groups, Path(output_path)
            )
            has_bouts = bool(bout_results)
            msg = (
                f"Exported {n_rows} fish rows to:\n{output_path}\n\n"
                f"Columns: trajectory metrics"
                + (" + bout summary stats." if has_bouts else " only (no bout data).")
            )
            self.set_status(f"Exported combined summary: {n_rows} fish")
            messagebox.showinfo("Export Complete", msg)
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{e}")
