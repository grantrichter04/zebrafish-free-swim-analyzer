"""
fish_analyzer/gui/bout_tab.py
==============================
Bout Analysis Tab Mixin - Swim bout detection and visualization for larvae.

Detects individual swim bouts from speed traces and provides:
- Configurable detection parameters (threshold, merge gap, min duration)
- Per-fish bout summary statistics
- Distribution plots: bout duration, IBI, peak speed, heading change
- Per-bout laterality analysis
- CSV export of per-bout data

The interactive bout viewer has been moved to inspector_tab.py.
"""

from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from ..bout_analysis import BoutParameters, BoutResults, analyze_bouts_for_file


class BoutTabMixin:
    """
    Mixin providing Bout Analysis Tab functionality.

    Expects GUIBase to provide:
        - self.notebook (ttk.Notebook)
        - self.loaded_files (dict of LoadedTrajectoryFile)
        - self.set_status(msg)
    """

    def _create_bout_tab(self):
        """Create the Bout Analysis tab."""
        bout_tab = ttk.Frame(self.notebook)
        self.notebook.add(bout_tab, text="Bout Analysis")

        # Horizontal split: controls (left) | results (right)
        paned = ttk.PanedWindow(bout_tab, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        # Left panel: controls
        controls_frame = ttk.Frame(paned, width=320)
        paned.add(controls_frame, weight=0)

        # Right panel: results
        results_frame = ttk.Frame(paned)
        paned.add(results_frame, weight=1)

        self._create_bout_controls(controls_frame)
        self._create_bout_results(results_frame)

        # Storage for bout results
        self.bout_results: Dict[str, List[BoutResults]] = {}

    # =========================================================================
    # CONTROLS
    # =========================================================================

    def _create_bout_controls(self, parent):
        """Create the bout analysis control panel."""
        # File selection
        file_frame = tk.LabelFrame(parent, text="Select Files",
                                    font=("Arial", 10, "bold"))
        file_frame.pack(fill="x", padx=10, pady=5)

        self.bout_file_listbox = tk.Listbox(file_frame,
                                             selectmode=tk.EXTENDED,
                                             height=6)
        self.bout_file_listbox.pack(fill="x", padx=10, pady=5)

        tk.Button(file_frame, text="Refresh File List",
                  command=self._refresh_bout_file_list).pack(pady=2)

        # Parameters
        params_frame = tk.LabelFrame(parent,
                                      text="Bout Detection Parameters",
                                      font=("Arial", 10, "bold"))
        params_frame.pack(fill="x", padx=10, pady=5)

        # Speed threshold
        thresh_frame = tk.Frame(params_frame)
        thresh_frame.pack(fill="x", padx=10, pady=3)
        tk.Label(thresh_frame, text="Speed threshold:").pack(side="left")
        self.bout_threshold_var = tk.StringVar(value="0.5")
        tk.Entry(thresh_frame, textvariable=self.bout_threshold_var,
                 width=6).pack(side="left", padx=5)
        tk.Label(thresh_frame, text="BL/s").pack(side="left")

        # Merge gap
        merge_frame = tk.Frame(params_frame)
        merge_frame.pack(fill="x", padx=10, pady=3)
        tk.Label(merge_frame, text="Merge gap:").pack(side="left")
        self.bout_merge_gap_var = tk.StringVar(value="2")
        tk.Entry(merge_frame, textvariable=self.bout_merge_gap_var,
                 width=4).pack(side="left", padx=5)
        tk.Label(merge_frame, text="frames").pack(side="left")

        # Min bout duration
        min_frame = tk.Frame(params_frame)
        min_frame.pack(fill="x", padx=10, pady=3)
        tk.Label(min_frame, text="Min bout length:").pack(side="left")
        self.bout_min_frames_var = tk.StringVar(value="1")
        tk.Entry(min_frame, textvariable=self.bout_min_frames_var,
                 width=4).pack(side="left", padx=5)
        tk.Label(min_frame, text="frames").pack(side="left")

        # Help text
        help_text = (
            "Speed threshold: movement below this is considered rest. "
            "Lower values detect more bouts but may include drift.\n\n"
            "Merge gap: bouts separated by fewer than this many frames "
            "are combined into one (prevents splitting a single dart).\n\n"
            "Min bout length: discard bouts shorter than this."
        )
        tk.Label(params_frame, text=help_text, font=("Arial", 8),
                 fg="gray", wraplength=280,
                 justify=tk.LEFT).pack(padx=10, pady=5)

        # Run button
        tk.Button(parent, text="Run Bout Analysis",
                  font=("Arial", 11, "bold"),
                  bg="lightgreen", command=self._run_bout_analysis,
                  width=25).pack(pady=10)

        # Export button
        tk.Button(parent, text="Export Bout Data to CSV",
                  command=self._export_bout_csv,
                  width=25).pack(pady=2)

    def _create_bout_results(self, parent):
        """Create the bout results display area."""
        results_notebook = ttk.Notebook(parent)
        results_notebook.pack(fill="both", expand=True)

        # Summary text tab
        text_tab = ttk.Frame(results_notebook)
        results_notebook.add(text_tab, text="Summary")

        scrollbar = tk.Scrollbar(text_tab)
        scrollbar.pack(side="right", fill="y")

        self.bout_results_text = tk.Text(
            text_tab, wrap=tk.WORD, font=("Courier", 10),
            yscrollcommand=scrollbar.set
        )
        self.bout_results_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.bout_results_text.yview)

        self.bout_results_text.insert("1.0",
            "Bout analysis detects individual swim bouts (darts) and "
            "measures:\n\n"
            "  - Bout rate: how frequently the fish darts (bouts/min)\n"
            "  - Bout duration: how long each dart lasts\n"
            "  - Inter-bout interval (IBI): rest time between darts\n"
            "  - Peak speed: maximum speed during each dart\n"
            "  - Displacement: how far each dart moves the fish\n"
            "  - Per-bout heading change: turning during each dart\n\n"
            "Load files in the Data tab, then select files here and "
            "click\n'Run Bout Analysis'.\n\n"
            "Tip: For larvae at 30fps, start with 0.5 BL/s threshold.\n"
            "For adults, try 0.3-1.0 BL/s depending on swimming style.\n\n"
            "Use the Video Inspector tab to visualize bout speed traces\n"
            "and heading changes overlaid on fish positions."
        )

        # Distribution plots tab
        plots_tab = ttk.Frame(results_notebook)
        results_notebook.add(plots_tab, text="Distributions")
        self.bout_plots_frame = plots_tab

        # Laterality tab
        lat_tab = ttk.Frame(results_notebook)
        results_notebook.add(lat_tab, text="Laterality")
        self.bout_laterality_frame = lat_tab

        # Methods tab
        methods_tab = ttk.Frame(results_notebook)
        results_notebook.add(methods_tab, text="Methods")
        methods_scroll = tk.Scrollbar(methods_tab)
        methods_scroll.pack(side="right", fill="y")
        self.bout_methods_text = tk.Text(
            methods_tab, wrap=tk.WORD, font=("Courier", 10),
            yscrollcommand=methods_scroll.set
        )
        self.bout_methods_text.pack(side="left", fill="both", expand=True)
        methods_scroll.config(command=self.bout_methods_text.yview)
        self.bout_methods_text.insert("1.0",
            "Methods text will appear here after running bout analysis.\n\n"
            "This provides a draft paragraph describing the bout detection\n"
            "methods and parameters used, suitable for a manuscript methods "
            "section."
        )

    # =========================================================================
    # ACTIONS
    # =========================================================================

    def _refresh_bout_file_list(self):
        """Update the file listbox with currently loaded files."""
        self.bout_file_listbox.delete(0, tk.END)
        for filename in self.loaded_files:
            self.bout_file_listbox.insert(tk.END, filename)
        if self.bout_file_listbox.size() > 0:
            self.bout_file_listbox.select_set(0, tk.END)

    def _get_bout_params(self) -> BoutParameters:
        """Read bout parameters from GUI inputs."""
        try:
            threshold = float(self.bout_threshold_var.get())
        except ValueError:
            messagebox.showwarning(
                "Invalid Input",
                "Speed threshold must be a number. Resetting to 0.5."
            )
            self.bout_threshold_var.set("0.5")
            threshold = 0.5

        try:
            merge_gap = int(self.bout_merge_gap_var.get())
        except ValueError:
            messagebox.showwarning(
                "Invalid Input",
                "Merge gap must be an integer. Resetting to 2."
            )
            self.bout_merge_gap_var.set("2")
            merge_gap = 2

        try:
            min_frames = int(self.bout_min_frames_var.get())
        except ValueError:
            messagebox.showwarning(
                "Invalid Input",
                "Min bout frames must be an integer. Resetting to 1."
            )
            self.bout_min_frames_var.set("1")
            min_frames = 1

        return BoutParameters(
            speed_threshold=threshold,
            merge_gap_frames=merge_gap,
            min_bout_frames=min_frames,
        )

    def _run_bout_analysis(self):
        """Run bout detection on selected files."""
        selected_indices = self.bout_file_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("No Files",
                                "Please select files to analyze.")
            return

        selected_files = [self.bout_file_listbox.get(i)
                          for i in selected_indices]
        params = self._get_bout_params()

        self.set_status("Running bout analysis...")
        self.bout_results.clear()

        for filename in selected_files:
            loaded_file = self.loaded_files[filename]
            print(f"\nBout analysis: {filename}")
            try:
                results = analyze_bouts_for_file(loaded_file, params)
                self.bout_results[filename] = results
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

        if self.bout_results:
            self._display_bout_summary(selected_files)
            self._plot_bout_distributions(selected_files)
            self._plot_bout_laterality(selected_files)
            self._update_bout_methods_text(selected_files, params)

            # Notify inspector that bout data is available
            if hasattr(self, '_inspector_rebuild_needed'):
                self._inspector_rebuild_needed()

            self.set_status(
                f"Bout analysis complete for "
                f"{len(self.bout_results)} file(s)"
            )
        else:
            self.set_status("Bout analysis: no results")

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def _display_bout_summary(self, selected_files: List[str]):
        """Display bout analysis summary as text."""
        self.bout_results_text.delete("1.0", tk.END)
        lines = []

        threshold = self.bout_threshold_var.get()
        lines.append(
            f"BOUT ANALYSIS RESULTS (threshold: {threshold} BL/s)"
        )
        lines.append("=" * 90)
        lines.append("")

        # Comparison table header
        lines.append(
            f"{'File':<22} {'Fish':<5} "
            f"{'Bouts':>7} {'Rate':>8} {'Duration':>10} {'IBI':>10} "
            f"{'PkSpd':>8} {'Disp':>7} {'Lateral':>8}"
        )
        lines.append(
            f"{'':22} {'':5} "
            f"{'':>7} {'(/min)':>8} {'(ms)':>10} {'(ms)':>10} "
            f"{'(BL/s)':>8} {'(BL)':>7} {'(-1..1)':>8}"
        )
        lines.append("-" * 90)

        for filename in selected_files:
            if filename not in self.bout_results:
                continue

            results = self.bout_results[filename]
            n_fish = len(results)
            all_bouts = sum(
                r.summary.get('bout_count', 0) for r in results
            )
            avg_rate = np.mean(
                [r.summary.get('bout_rate_per_min', 0) for r in results]
            )
            avg_dur = np.nanmean(
                [r.summary.get('bout_duration_median_ms', np.nan)
                 for r in results]
            )
            avg_ibi = np.nanmean(
                [r.summary.get('ibi_median_ms', np.nan)
                 for r in results]
            )
            avg_pk = np.nanmean(
                [r.summary.get('bout_peak_speed_median', np.nan)
                 for r in results]
            )
            avg_disp = np.nanmean(
                [r.summary.get('bout_displacement_median', np.nan)
                 for r in results]
            )
            avg_lat = np.nanmean(
                [r.summary.get('bout_laterality_index', np.nan)
                 for r in results]
            )

            short_name = (filename[:21] if len(filename) > 21
                          else filename)
            lines.append(
                f"{short_name:<22} {n_fish:<5} "
                f"{all_bouts:>7} {avg_rate:>8.1f} {avg_dur:>10.0f} "
                f"{avg_ibi:>10.0f} "
                f"{avg_pk:>8.2f} {avg_disp:>7.2f} {avg_lat:>8.3f}"
            )

        lines.append("-" * 90)
        lines.append("")

        # Detailed per-fish results
        for filename in selected_files:
            if filename not in self.bout_results:
                continue

            results = self.bout_results[filename]
            loaded_file = self.loaded_files[filename]
            unit = loaded_file.calibration.unit_name

            lines.append(f"FILE: {filename}")
            lines.append(
                f"Duration: {loaded_file.duration_minutes:.1f} min | "
                f"FPS: {loaded_file.calibration.frame_rate:.1f}"
            )
            lines.append("-" * 70)

            for r in results:
                s = r.summary
                lines.append(
                    f"  Fish {r.fish_id} ({r.identity_label}):"
                )
                lines.append(
                    f"    Bout count:         "
                    f"{s.get('bout_count', 0)}"
                )
                lines.append(
                    f"    Bout rate:          "
                    f"{s.get('bout_rate_per_min', 0):.1f} /min"
                )
                dur_iqr = s.get('bout_duration_iqr_ms', (np.nan, np.nan))
                lines.append(
                    f"    Bout duration:      "
                    f"{s.get('bout_duration_median_ms', np.nan):.0f} ms "
                    f"(IQR: {dur_iqr[0]:.0f}-{dur_iqr[1]:.0f})"
                )
                ibi_iqr = s.get('ibi_iqr_ms', (np.nan, np.nan))
                lines.append(
                    f"    Inter-bout int:     "
                    f"{s.get('ibi_median_ms', np.nan):.0f} ms "
                    f"(IQR: {ibi_iqr[0]:.0f}-{ibi_iqr[1]:.0f})"
                )
                pk_iqr = s.get('bout_peak_speed_iqr', (np.nan, np.nan))
                lines.append(
                    f"    Peak speed:         "
                    f"{s.get('bout_peak_speed_median', np.nan):.2f} "
                    f"{unit}/s "
                    f"(IQR: {pk_iqr[0]:.2f}-{pk_iqr[1]:.2f})"
                )
                lines.append(
                    f"    Displacement:       "
                    f"{s.get('bout_displacement_median', np.nan):.3f} "
                    f"{unit}"
                )
                lines.append(
                    f"    Distance:           "
                    f"{s.get('bout_distance_median', np.nan):.3f} "
                    f"{unit}"
                )
                lines.append(f"    ---")
                lines.append(
                    f"    Mean |heading D|:   "
                    f"{s.get('bout_heading_change_mean_abs_deg', np.nan):.1f}"
                    f" deg"
                )
                lat = s.get('bout_laterality_index', np.nan)
                lat_label = "no bias"
                if not np.isnan(lat):
                    if lat > 0.1:
                        lat_label = "CW (right) bias"
                    elif lat < -0.1:
                        lat_label = "CCW (left) bias"
                lines.append(
                    f"    Laterality:         "
                    f"{lat:.3f} ({lat_label})"
                )
                lines.append(
                    f"    Turns:              "
                    f"{s.get('bout_n_right', 0)} right, "
                    f"{s.get('bout_n_left', 0)} left, "
                    f"{s.get('bout_n_straight', 0)} straight"
                )
                lines.append("")

        self.bout_results_text.insert("1.0", "\n".join(lines))

    def _plot_bout_distributions(self, selected_files: List[str]):
        """Plot bout metric distributions."""
        for widget in self.bout_plots_frame.winfo_children():
            widget.destroy()

        # Collect all bouts across selected files
        all_durations = {}
        all_ibis = {}
        all_peak_speeds = {}
        all_heading_changes = {}

        for filename in selected_files:
            if filename not in self.bout_results:
                continue
            short = (filename[:15] if len(filename) > 15
                     else filename)
            durs = []
            ibis = []
            peaks = []
            headings = []
            for r in self.bout_results[filename]:
                durs.extend([b.duration_s * 1000 for b in r.bouts])
                ibis.extend(r.inter_bout_intervals_s * 1000)
                peaks.extend([b.peak_speed for b in r.bouts])
                headings.extend(
                    [b.heading_change_deg for b in r.bouts]
                )

            all_durations[short] = np.array(durs)
            all_ibis[short] = np.array(ibis)
            all_peak_speeds[short] = np.array(peaks)
            all_heading_changes[short] = np.array(headings)

        if not all_durations:
            return

        fig = Figure(figsize=(12, 8), dpi=100)
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        # 1. Bout duration histogram
        ax1 = fig.add_subplot(2, 2, 1)
        for label, data in all_durations.items():
            if len(data) > 0:
                bins = np.linspace(
                    0, min(np.percentile(data, 99), 500), 30
                )
                ax1.hist(data, bins=bins, alpha=0.6, label=label,
                         edgecolor='black', linewidth=0.5)
        ax1.set_xlabel("Bout Duration (ms)")
        ax1.set_ylabel("Count")
        ax1.set_title("Bout Duration Distribution")
        if len(all_durations) > 1:
            ax1.legend(fontsize=7)

        # 2. Inter-bout interval histogram
        ax2 = fig.add_subplot(2, 2, 2)
        for label, data in all_ibis.items():
            if len(data) > 0:
                bins = np.linspace(
                    0, min(np.percentile(data, 99), 2000), 30
                )
                ax2.hist(data, bins=bins, alpha=0.6, label=label,
                         edgecolor='black', linewidth=0.5)
        ax2.set_xlabel("Inter-Bout Interval (ms)")
        ax2.set_ylabel("Count")
        ax2.set_title("IBI Distribution")
        if len(all_ibis) > 1:
            ax2.legend(fontsize=7)

        # 3. Peak speed histogram
        ax3 = fig.add_subplot(2, 2, 3)
        for label, data in all_peak_speeds.items():
            if len(data) > 0:
                bins = np.linspace(
                    0, min(np.percentile(data, 99), 15), 30
                )
                ax3.hist(data, bins=bins, alpha=0.6, label=label,
                         edgecolor='black', linewidth=0.5)
        ax3.set_xlabel("Peak Bout Speed (BL/s)")
        ax3.set_ylabel("Count")
        ax3.set_title("Bout Peak Speed Distribution")
        if len(all_peak_speeds) > 1:
            ax3.legend(fontsize=7)

        # 4. Heading change histogram
        ax4 = fig.add_subplot(2, 2, 4)
        for label, data in all_heading_changes.items():
            if len(data) > 0:
                bins = np.linspace(-180, 180, 37)
                ax4.hist(data, bins=bins, alpha=0.6, label=label,
                         edgecolor='black', linewidth=0.5)
        ax4.set_xlabel("Bout Heading Change (deg)")
        ax4.set_ylabel("Count")
        ax4.set_title("Per-Bout Turn Angle")
        ax4.axvline(0, color='red', linestyle='--', alpha=0.5)
        if len(all_heading_changes) > 1:
            ax4.legend(fontsize=7)

        canvas = FigureCanvasTkAgg(fig, master=self.bout_plots_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_bout_laterality(self, selected_files: List[str]):
        """Plot per-fish laterality summary."""
        for widget in self.bout_laterality_frame.winfo_children():
            widget.destroy()

        # Collect per-fish laterality data
        fish_labels = []
        lat_values = []
        n_lefts = []
        n_rights = []
        n_straights = []

        for filename in selected_files:
            if filename not in self.bout_results:
                continue
            short = (filename[:12] if len(filename) > 12
                     else filename)
            for r in self.bout_results[filename]:
                s = r.summary
                label = f"{short}\nFish {r.fish_id}"
                fish_labels.append(label)
                lat_values.append(
                    s.get('bout_laterality_index', 0)
                )
                n_lefts.append(s.get('bout_n_left', 0))
                n_rights.append(s.get('bout_n_right', 0))
                n_straights.append(s.get('bout_n_straight', 0))

        if not fish_labels:
            return

        fig = Figure(figsize=(12, 6), dpi=100)

        # Left: laterality index bar chart
        ax1 = fig.add_subplot(1, 2, 1)
        x = np.arange(len(fish_labels))
        colors = [
            '#d32f2f' if v > 0.1
            else '#1976d2' if v < -0.1
            else '#757575'
            for v in lat_values
        ]
        ax1.bar(x, lat_values, color=colors, edgecolor='black',
                linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(fish_labels, fontsize=7, rotation=45,
                             ha='right')
        ax1.set_ylabel("Laterality Index")
        ax1.set_title("Per-Fish Laterality\n(+CW/right, -CCW/left)")
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.axhline(0.1, color='gray', linewidth=0.5, linestyle='--',
                     alpha=0.5)
        ax1.axhline(-0.1, color='gray', linewidth=0.5, linestyle='--',
                     alpha=0.5)
        ax1.set_ylim(-1, 1)

        # Right: stacked bar of left/right/straight turns
        ax2 = fig.add_subplot(1, 2, 2)
        bar_width = 0.6
        ax2.bar(x, n_lefts, bar_width, label='Left (CCW)',
                color='#1976d2')
        ax2.bar(x, n_straights, bar_width, bottom=n_lefts,
                label='Straight', color='#bdbdbd')
        bottoms = [l + s for l, s in zip(n_lefts, n_straights)]
        ax2.bar(x, n_rights, bar_width, bottom=bottoms,
                label='Right (CW)', color='#d32f2f')
        ax2.set_xticks(x)
        ax2.set_xticklabels(fish_labels, fontsize=7, rotation=45,
                             ha='right')
        ax2.set_ylabel("Bout Count")
        ax2.set_title("Turn Direction per Fish")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.bout_laterality_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # =========================================================================
    # METHODS TEXT
    # =========================================================================

    def _update_bout_methods_text(self, selected_files: List[str],
                                   params: BoutParameters):
        """Generate a methods paragraph for the bout analysis."""
        self.bout_methods_text.delete("1.0", tk.END)

        first_file = None
        for f in selected_files:
            if f in self.loaded_files:
                first_file = self.loaded_files[f]
                break
        if first_file is None:
            return

        cal = first_file.calibration
        fps = cal.frame_rate
        n_files = len(selected_files)
        total_fish = sum(
            len(self.bout_results.get(f, []))
            for f in selected_files
        )
        total_bouts = sum(
            sum(r.summary.get('bout_count', 0)
                for r in self.bout_results.get(f, []))
            for f in selected_files
        )

        merge_ms = params.merge_gap_frames / fps * 1000
        min_ms = params.min_bout_frames / fps * 1000

        text = (
            "METHODS \u2014 Bout Analysis\n"
            "========================\n\n"
            "The following is a draft methods paragraph. Edit as "
            "needed.\n\n"
            "---\n\n"
            f"Swim bout detection was performed on raw (unsmoothed) "
            f"trajectory data using the Zebrafish Free Swim Analyzer "
            f"(v2.1). "
            f"Frame-to-frame speed was computed from position "
            f"displacements in {cal.unit_name} at {fps:.1f} fps. "
            f"A bout was defined as a contiguous period where speed "
            f"exceeded {params.speed_threshold} {cal.unit_name}/s. "
            f"Bouts separated by gaps of \u2264"
            f"{params.merge_gap_frames} frames "
            f"({merge_ms:.0f} ms) were merged to prevent single darts "
            f"from being split by transient speed dips. "
            f"Bouts shorter than {params.min_bout_frames} frame(s) "
            f"({min_ms:.0f} ms) were discarded.\n\n"
            f"For each detected bout, the following metrics were "
            f"extracted: "
            f"duration, peak speed, mean speed, displacement "
            f"(straight-line distance from bout start to end), path "
            f"distance (total distance traveled during the bout), and "
            f"heading change (total signed angular displacement during "
            f"the bout, with positive values indicating "
            f"counterclockwise turns and negative values indicating "
            f"clockwise turns). Inter-bout intervals (IBIs) were "
            f"computed as the time between the end of one bout and the "
            f"start of the next.\n\n"
            f"Turning laterality was assessed per fish by classifying "
            f"each bout as a left turn (heading change > +5\u00b0), "
            f"right turn (< \u22125\u00b0), or straight "
            f"(\u22645\u00b0). A laterality index was computed as "
            f"(right \u2212 left) / (right + left), ranging from "
            f"\u22121 (all counterclockwise) to +1 (all "
            f"clockwise).\n\n"
            f"Analysis included {n_files} file(s), {total_fish} fish, "
            f"and {total_bouts} detected bouts.\n\n"
            "---\n\n"
            "KEY PARAMETERS:\n"
            f"  Speed threshold:     {params.speed_threshold} "
            f"{cal.unit_name}/s\n"
            f"  Merge gap:           {params.merge_gap_frames} frames "
            f"({merge_ms:.0f} ms)\n"
            f"  Min bout duration:   {params.min_bout_frames} frame(s) "
            f"({min_ms:.0f} ms)\n"
            f"  Turn dead zone:      \u00b15\u00b0\n"
            f"  Frame rate:          {fps:.1f} fps\n"
            f"  Calibration:         {cal.scale_factor:.6f} "
            f"{cal.unit_name}/px\n"
        )

        self.bout_methods_text.insert("1.0", text)

    # =========================================================================
    # EXPORT
    # =========================================================================

    def _export_bout_csv(self):
        """Export per-bout data to CSV."""
        if not self.bout_results:
            messagebox.showinfo("No Data", "Run bout analysis first.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Bout Data",
            initialfile="bout_analysis.csv"
        )
        if not path:
            return

        import csv

        rows = []
        for filename, results_list in self.bout_results.items():
            loaded_file = self.loaded_files.get(filename)
            fps = (loaded_file.calibration.frame_rate
                   if loaded_file else 30.0)
            unit = (loaded_file.calibration.unit_name
                    if loaded_file else "BL")

            for r in results_list:
                for bout in r.bouts:
                    rows.append({
                        'File': filename,
                        'FishID': r.fish_id,
                        'Label': r.identity_label,
                        'Unit': unit,
                        'BoutStartFrame': bout.start_frame,
                        'BoutEndFrame': bout.end_frame,
                        'BoutStartTime_s': round(
                            bout.start_frame / fps, 3
                        ),
                        'Duration_ms': round(
                            bout.duration_s * 1000, 1
                        ),
                        'PeakSpeed': round(bout.peak_speed, 4),
                        'MeanSpeed': round(bout.mean_speed, 4),
                        'Displacement': round(bout.displacement, 4),
                        'Distance': round(bout.distance, 4),
                        'HeadingChange_deg': round(
                            bout.heading_change_deg, 2
                        ),
                    })

        if not rows:
            messagebox.showinfo("No Data", "No bouts detected.")
            return

        fieldnames = list(rows[0].keys())
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        n_bouts = len(rows)
        messagebox.showinfo(
            "Export Complete",
            f"Exported {n_bouts} bouts to:\n{path}"
        )
        self.set_status(f"Exported {n_bouts} bouts to CSV")
