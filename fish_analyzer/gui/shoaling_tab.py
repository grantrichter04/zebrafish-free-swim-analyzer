"""
fish_analyzer/gui/shoaling_tab.py
=================================
Shoaling Tab Mixin - Group behavior analysis.

This mixin provides all methods related to:
- NND (Nearest Neighbor Distance) analysis
- IID (Inter-Individual Distance) analysis
- Convex Hull area analysis
- Time series plotting with smoothing
- CSV export

The interactive frame viewer has been moved to inspector_tab.py.
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
from ..export import export_shoaling_metrics_csv, export_shoaling_summary_csv
from .utils import smooth_time_series, create_sortable_treeview, embed_figure_with_toolbar


class ShoalingTabMixin:
    """
    Mixin providing Shoaling Analysis Tab functionality.

    Expects the following attributes from base class:
    - self.root: tk.Tk
    - self.notebook: ttk.Notebook
    - self.loaded_files: Dict[str, LoadedTrajectoryFile]
    - self.shoaling_params: ShoalingParameters
    """

    def _create_shoaling_tab(self):
        """Create the shoaling analysis tab."""
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
        params_frame = tk.LabelFrame(parent, text="Shoaling Parameters",
                                      font=("Arial", 11, "bold"))
        params_frame.pack(fill="x", padx=10, pady=10)

        interval_frame = tk.Frame(params_frame)
        interval_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(interval_frame, text="Sample every:").pack(side="left")
        self.shoaling_interval_var = tk.StringVar(value="30")
        tk.Entry(interval_frame, textvariable=self.shoaling_interval_var,
                 width=6).pack(side="left", padx=5)
        tk.Label(interval_frame, text="frames").pack(side="left")

        tk.Label(params_frame,
                 text="At 30fps, sampling every 30 frames = 1 sample/second",
                 font=("Arial", 8), fg="gray").pack(anchor="w", padx=10)

        smooth_frame = tk.Frame(params_frame)
        smooth_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(smooth_frame, text="Plot smoothing:").pack(side="left")
        self.shoaling_smooth_var = tk.StringVar(value="5.0")
        tk.Entry(smooth_frame, textvariable=self.shoaling_smooth_var,
                 width=6).pack(side="left", padx=5)
        tk.Label(smooth_frame, text="seconds").pack(side="left")

        # File selection
        file_frame = tk.LabelFrame(parent, text="Select Files to Compare",
                                    font=("Arial", 11, "bold"))
        file_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(file_frame,
                 text="Hold Ctrl/Cmd to select multiple files",
                 font=("Arial", 8), fg="gray").pack(anchor="w", padx=10,
                                                      pady=2)

        list_container = tk.Frame(file_frame)
        list_container.pack(fill="x", padx=10, pady=5)

        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")

        self.shoaling_files_listbox = tk.Listbox(
            list_container, selectmode=tk.EXTENDED, height=5,
            yscrollcommand=scrollbar.set
        )
        self.shoaling_files_listbox.pack(side="left", fill="both",
                                          expand=True)
        scrollbar.config(command=self.shoaling_files_listbox.yview)

        tk.Button(
            parent, text="Run Shoaling Analysis",
            command=self._run_shoaling_analysis,
            bg="lightgreen", font=("Arial", 12, "bold"), height=2
        ).pack(fill="x", padx=10, pady=10)

        tk.Button(
            parent, text="Export Shoaling Results to CSV",
            command=self._export_shoaling_csv,
            font=("Arial", 10)
        ).pack(fill="x", padx=10, pady=(0, 10))

    def _create_shoaling_results(self, parent):
        """Create the results panel for shoaling tab."""
        results_notebook = ttk.Notebook(parent)
        results_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Results Summary tab (Treeview tables)
        text_tab = ttk.Frame(results_notebook)
        results_notebook.add(text_tab, text="Results Summary")

        self.shoaling_summary_frame = tk.Frame(text_tab)
        self.shoaling_summary_frame.pack(fill="both", expand=True)

        tk.Label(self.shoaling_summary_frame,
            text="Shoaling analysis results will appear here.\n\n"
            "Select files and click 'Run Shoaling Analysis'.\n"
            "Metrics: NND (nearest neighbor), IID (all pairs), Hull (group area)",
            font=("Arial", 10), fg="gray", justify=tk.LEFT
        ).pack(padx=20, pady=20)

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

        # Methods tab
        methods_tab = ttk.Frame(results_notebook)
        results_notebook.add(methods_tab, text="Methods")
        methods_scroll = tk.Scrollbar(methods_tab)
        methods_scroll.pack(side="right", fill="y")
        self.shoaling_methods_text = tk.Text(
            methods_tab, wrap=tk.WORD, font=("Courier", 10),
            yscrollcommand=methods_scroll.set
        )
        self.shoaling_methods_text.pack(side="left", fill="both",
                                         expand=True)
        methods_scroll.config(command=self.shoaling_methods_text.yview)
        self.shoaling_methods_text.insert("1.0",
            "Methods text will appear here after running shoaling analysis.\n\n"
            "This provides a draft paragraph describing the shoaling analysis\n"
            "methods and parameters used, suitable for a manuscript methods "
            "section."
        )

    # =========================================================================
    # FILE DROPDOWN UPDATE
    # =========================================================================

    def _update_shoaling_file_dropdown(self):
        """Update all file selection widgets across tabs."""
        # Update shoaling files listbox
        self.shoaling_files_listbox.delete(0, tk.END)
        for nickname in self.loaded_files.keys():
            self.shoaling_files_listbox.insert(tk.END, nickname)

        # Update inspector file dropdown
        if hasattr(self, '_update_inspector_file_dropdown'):
            self._update_inspector_file_dropdown()

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
        """Run shoaling analysis on selected files (defaults to all)."""
        selected_indices = self.shoaling_files_listbox.curselection()
        if not selected_indices:
            # Auto-select all files
            if self.shoaling_files_listbox.size() > 0:
                self.shoaling_files_listbox.select_set(0, tk.END)
                selected_indices = self.shoaling_files_listbox.curselection()
            else:
                messagebox.showerror("No Files",
                                      "Load files in the Data tab first.")
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
            messagebox.showerror("Invalid Parameters",
                                  f"Check parameters:\n{e}")
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
                messagebox.showwarning("Analysis Warning",
                                        f"Failed '{filename}':\n{e}")
                import traceback
                traceback.print_exc()

        if not all_results:
            messagebox.showerror("Analysis Failed",
                                  "Could not analyze any files.")
            return

        self._display_shoaling_comparison(all_results)
        self._plot_nnd_comparison(all_results)
        self._plot_iid_comparison(all_results)
        self._plot_hull_comparison(all_results)
        self._update_shoaling_methods_text(all_results, params)

        # Notify inspector that shoaling data is available
        if hasattr(self, '_inspector_rebuild_needed'):
            self._inspector_rebuild_needed()

        messagebox.showinfo(
            "Analysis Complete",
            f"Shoaling analysis complete for {len(all_results)} file(s).\n\n"
            "Use the Video Inspector tab to visualize overlays."
        )

    def _display_shoaling_comparison(self,
                                      all_results: Dict[str, ShoalingResults]):
        """Display comparison of shoaling results using Treeview tables."""
        for widget in self.shoaling_summary_frame.winfo_children():
            widget.destroy()

        columns = [
            ("file", "File", 160), ("fish", "Fish", 50),
            ("nnd", "Mean NND (BL)", 100), ("nnd_sd", "NND SD", 70),
            ("iid", "Mean IID (BL)", 100), ("iid_sd", "IID SD", 70),
            ("hull", "Mean Hull (BL\u00b2)", 110), ("hull_sd", "Hull SD", 70),
            ("complete", "Complete %", 80),
        ]

        rows = []
        for filename, results in all_results.items():
            rows.append((
                filename, results.n_fish,
                f"{results.mean_nnd:.3f}", f"{results.std_nnd:.3f}",
                f"{results.mean_iid:.3f}", f"{results.std_iid:.3f}",
                f"{results.mean_hull_area:.2f}", f"{results.std_hull_area:.2f}",
                f"{results.completeness_percentage:.1f}",
            ))

        create_sortable_treeview(self.shoaling_summary_frame, columns, rows,
                                 title="Shoaling Comparison")

    def _plot_nnd_comparison(self,
                              all_results: Dict[str, ShoalingResults]):
        """Plot NND time series with smoothing."""
        for widget in self.shoaling_nnd_plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        try:
            smooth_seconds = float(self.shoaling_smooth_var.get())
        except ValueError:
            smooth_seconds = 5.0
            self.shoaling_smooth_var.set("5.0")

        for (filename, results), color in zip(all_results.items(), colors):
            time_minutes = results.timestamps / 60.0
            nnd_data = results.mean_nnd_per_sample.copy()

            if smooth_seconds > 0 and len(nnd_data) > 1:
                effective_fps = 1.0 / (
                    results.sample_interval_frames / results.frame_rate
                )
                nnd_data = smooth_time_series(nnd_data, smooth_seconds,
                                               effective_fps)

            ax.plot(time_minutes, nnd_data, color=color, linewidth=2,
                    label=f'{filename} ({results.mean_nnd:.2f} BL)',
                    alpha=0.9)

        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Mean Nearest Neighbor Distance (BL)', fontsize=12)
        ax.set_title('NND Comparison Over Time', fontsize=14,
                      fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        embed_figure_with_toolbar(fig, self.shoaling_nnd_plot_frame)

    def _plot_iid_comparison(self,
                              all_results: Dict[str, ShoalingResults]):
        """Plot IID time series with smoothing."""
        for widget in self.shoaling_iid_plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        try:
            smooth_seconds = float(self.shoaling_smooth_var.get())
        except ValueError:
            smooth_seconds = 5.0
            self.shoaling_smooth_var.set("5.0")

        for (filename, results), color in zip(all_results.items(), colors):
            time_minutes = results.timestamps / 60.0
            iid_data = results.mean_iid_per_sample.copy()

            if smooth_seconds > 0 and len(iid_data) > 1:
                effective_fps = 1.0 / (
                    results.sample_interval_frames / results.frame_rate
                )
                iid_data = smooth_time_series(iid_data, smooth_seconds,
                                               effective_fps)

            ax.plot(time_minutes, iid_data, color=color, linewidth=2,
                    label=f'{filename} ({results.mean_iid:.2f} BL)',
                    alpha=0.9)

        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Mean Inter-Individual Distance (BL)', fontsize=12)
        ax.set_title('IID (Group Cohesion) Over Time', fontsize=14,
                      fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        embed_figure_with_toolbar(fig, self.shoaling_iid_plot_frame)

    def _plot_hull_comparison(self,
                               all_results: Dict[str, ShoalingResults]):
        """Plot hull area time series with smoothing."""
        for widget in self.shoaling_hull_plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        try:
            smooth_seconds = float(self.shoaling_smooth_var.get())
        except ValueError:
            smooth_seconds = 5.0
            self.shoaling_smooth_var.set("5.0")

        for (filename, results), color in zip(all_results.items(), colors):
            time_minutes = results.timestamps / 60.0
            hull_data = results.convex_hull_area_per_sample.copy()

            if smooth_seconds > 0 and len(hull_data) > 1:
                effective_fps = 1.0 / (
                    results.sample_interval_frames / results.frame_rate
                )
                hull_data = smooth_time_series(hull_data, smooth_seconds,
                                                effective_fps)

            ax.plot(time_minutes, hull_data, color=color, linewidth=2,
                    label=f'{filename} ({results.mean_hull_area:.1f} '
                          f'BL\u00b2)',
                    alpha=0.9)

        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Convex Hull Area (BL\u00b2)', fontsize=12)
        ax.set_title('Group Spread Over Time', fontsize=14,
                      fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        embed_figure_with_toolbar(fig, self.shoaling_hull_plot_frame)

    # =========================================================================
    # METHODS TEXT
    # =========================================================================

    def _update_shoaling_methods_text(self, all_results, params):
        """Generate a methods paragraph for the shoaling analysis."""
        self.shoaling_methods_text.delete("1.0", tk.END)

        first_file = None
        for filename in all_results:
            if filename in self.loaded_files:
                first_file = self.loaded_files[filename]
                break
        if first_file is None:
            return

        cal = first_file.calibration
        fps = cal.frame_rate
        n_files = len(all_results)
        interval_s = params.sample_interval_frames / fps

        text = (
            "METHODS \u2014 Shoaling Analysis\n"
            "============================\n\n"
            "The following is a draft methods paragraph. Edit as needed.\n\n"
            "---\n\n"
            f"Group cohesion was quantified using three shoaling metrics "
            f"computed at {interval_s:.2f}-second intervals "
            f"(every {params.sample_interval_frames} frames at "
            f"{fps:.1f} fps) "
            f"using the Zebrafish Free Swim Analyzer (v2.1). "
            f"All distances were measured in {cal.unit_name}.\n\n"
            f"Nearest Neighbor Distance (NND) was calculated for each fish "
            f"as the Euclidean distance to its closest neighbor using "
            f"scipy.spatial.distance.cdist, then averaged across all fish "
            f"at each time point. "
            f"Inter-Individual Distance (IID) was computed as the mean of "
            f"all pairwise distances between fish (from the condensed "
            f"distance matrix via scipy.spatial.distance.pdist). "
            f"Convex hull area was computed as the area of the smallest "
            f"convex polygon enclosing all fish positions at each time "
            f"point (in {cal.unit_name}\u00b2), requiring at least 3 fish "
            f"with valid positions.\n\n"
            f"Time series were optionally smoothed using a rolling mean "
            f"(window = {params.smoothing_window_seconds} seconds) for "
            f"visualization. Summary statistics (mean \u00b1 SD) were "
            f"computed from the unsmoothed per-sample values. Samples "
            f"where any fish lacked valid position data were flagged as "
            f"incomplete.\n\n"
            f"Analysis included {n_files} file(s).\n\n"
            "---\n\n"
            "KEY PARAMETERS:\n"
            f"  Sample interval:       {params.sample_interval_frames} "
            f"frames ({interval_s:.2f} s)\n"
            f"  Smoothing window:      "
            f"{params.smoothing_window_seconds} s (for plots only)\n"
            f"  Distance unit:         {cal.unit_name}\n"
            f"  Frame rate:            {fps:.1f} fps\n"
            f"  Calibration:           {cal.scale_factor:.6f} "
            f"{cal.unit_name}/px\n"
        )

        self.shoaling_methods_text.insert("1.0", text)

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def _export_shoaling_csv(self):
        """Export shoaling analysis results to CSV files."""
        analyzed = {k: v for k, v in self.loaded_files.items()
                    if getattr(v, 'shoaling_results', None) is not None}
        if not analyzed:
            messagebox.showwarning(
                "No Data",
                "No files have shoaling results yet.\n"
                "Run 'Run Shoaling Analysis' first."
            )
            return

        output_path = filedialog.asksaveasfilename(
            title="Export Shoaling Results CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="shoaling_timeseries.csv"
        )
        if not output_path:
            return

        try:
            out = Path(output_path)
            n_ts = export_shoaling_metrics_csv(analyzed, out)
            summary_path = out.with_name(out.stem + "_summary.csv")
            n_sum = export_shoaling_summary_csv(analyzed, summary_path)

            self.set_status(f"Exported shoaling data to {out.name}")
            messagebox.showinfo(
                "Export Complete",
                f"Exported shoaling data:\n\n"
                f"Time series: {n_ts} rows \u2192 {out.name}\n"
                f"Summary: {n_sum} rows \u2192 {summary_path.name}"
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{e}")
