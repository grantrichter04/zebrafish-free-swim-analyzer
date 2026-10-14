"""
fish_analyzer/gui/results_tab.py
================================
Results Tab - the answer to "is one group more active than another?" on one
screen: a SuperPlot per metric, the table behind it, and one export.

What is shown, and why the session is the unit, is explained in results.py.
"""
import tkinter as tk
from pathlib import Path

import numpy as np
from tkinter import filedialog, messagebox, ttk

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .. import results, results_export
from ..overlay_render import fish_colors
from .utils import embed_figure_with_toolbar


class ResultsTabMixin:
    """
    Mixin providing the Results tab.

    Expects the following attributes from base class:
    - self.notebook: ttk.Notebook
    - self.loaded_files, self.file_groups, self.file_arena_definitions
    """

    RESULT_COLUMNS = (
        ("Group", "Group", 130), ("Session", "Session", 200), ("Fish", "Fish", 60),
        ("Tracked_pct", "Tracked %", 80), ("Jumps", "Jumps", 60),
        ("DistancePerTrackedMin", "Distance per tracked min", 150),
        ("MedianSpeed", "Median speed", 120), ("Speed95", "95th pct speed", 130),
        ("Straightness", "Straightness", 100), ("NearWall", "Near wall %", 100),
        ("RestingTime", "Resting %", 80),
    )

    def _create_results_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Results")
        self.results_tab_frame = tab
        self._results_table = None

        top = tk.Frame(tab)
        top.pack(fill="x", padx=20, pady=(10, 0))
        self.results_note_var = tk.StringVar(
            value="No results yet. Load sessions and press \"Run All Analysis\" "
                  "on the Sessions & Units tab.")
        tk.Label(top, textvariable=self.results_note_var, justify=tk.LEFT,
                 anchor="w", font=("Arial", 9), fg="gray30", wraplength=900
                 ).pack(side="left", fill="x", expand=True)
        self.results_export_button = tk.Button(
            top, text="Export results (CSVs)...", command=self._export_results,
            bg="lightblue", font=("Arial", 10, "bold"), state="disabled")
        self.results_export_button.pack(side="right")

        view_row = tk.Frame(tab)
        view_row.pack(fill="x", padx=20, pady=(6, 0))
        tk.Label(view_row, text="Show:", font=("Arial", 10, "bold")).pack(side="left")
        self.results_view = tk.StringVar(value="comparison")
        for value, text in (("comparison", "Group comparison"),
                            ("minutes", "Minute by minute"),
                            ("distributions", "Speed distributions"),
                            ("density", "Where they swim")):
            tk.Radiobutton(view_row, text=text, value=value,
                           variable=self.results_view,
                           command=self._draw_results_plot).pack(side="left", padx=8)

        # Smoothing is the reader's choice: off until picked, and only the
        # minute-by-minute view has anything to smooth.
        tk.Label(view_row, text="Smoothing:").pack(side="left", padx=(18, 3))
        self.results_smoothing_box = ttk.Combobox(
            view_row, state="disabled", width=10,
            values=[text for text, _ in results.SMOOTHING_CHOICES])
        self.results_smoothing_box.current(0)
        self.results_smoothing_box.pack(side="left")
        self.results_smoothing_box.bind(
            "<<ComboboxSelected>>", lambda e: self._draw_results_plot())
        # Off by default: the session is the unit, and with six fish a tank
        # the fish lines hide the sessions. Kept for looking at one tank.
        self.results_each_fish = tk.BooleanVar(value=False)
        self.results_each_fish_check = tk.Checkbutton(
            view_row, text="Each fish", variable=self.results_each_fish,
            state="disabled", command=self._draw_results_plot)
        self.results_each_fish_check.pack(side="left", padx=(10, 0))
        # Only the speed distributions have ridges.
        tk.Label(view_row, text="Ridges:").pack(side="left", padx=(14, 3))
        self.results_ridge_box = ttk.Combobox(
            view_row, state="disabled", width=7,
            values=[text for text, _ in results.RIDGE_LEVELS])
        self.results_ridge_box.current(0)
        self.results_ridge_box.pack(side="left")
        self.results_ridge_box.bind(
            "<<ComboboxSelected>>", lambda e: self._draw_results_plot())
        # Speeds have a long tail. Blank is automatic: up to where 99.5% of
        # the fastest fish's speeds fall.
        tk.Label(view_row, text="Speed axis up to:").pack(side="left", padx=(14, 3))
        self.results_speed_top_var = tk.StringVar(value="")
        self.results_speed_top_box = tk.Entry(
            view_row, textvariable=self.results_speed_top_var, width=5,
            state="disabled")
        self.results_speed_top_box.pack(side="left")
        for event in ("<Return>", "<FocusOut>"):
            self.results_speed_top_box.bind(event, lambda e: self._draw_results_plot())
        # The heatmaps' grid: the widest side of the video in this many
        # squares, the same size for every session. Fewer reads smoother.
        tk.Label(view_row, text="Squares across:").pack(side="left", padx=(14, 3))
        self.results_squares_var = tk.StringVar(value="48")
        self.results_squares_box = tk.Spinbox(
            view_row, from_=8, to=200, increment=8, width=4, state="disabled",
            textvariable=self.results_squares_var, command=self._draw_results_plot)
        self.results_squares_box.pack(side="left")
        for event in ("<Return>", "<FocusOut>"):
            self.results_squares_box.bind(event, lambda e: self._draw_results_plot())
        tk.Button(view_row, text="What do these measures mean?",
                  command=self._explain_measures).pack(side="right")

        panel_row = tk.Frame(tab)
        panel_row.pack(fill="x", padx=20, pady=(4, 0))
        tk.Label(panel_row, text="Panels:").pack(side="left")
        #: Metric column -> whether it gets a panel in the comparison and the
        #: minute-by-minute views.
        self.results_panel_vars = {}
        for metric in results.METRICS:
            var = tk.BooleanVar(value=True)
            self.results_panel_vars[metric.column] = var
            tk.Checkbutton(panel_row, text=metric.title, variable=var,
                           command=self._draw_results_plot).pack(side="left", padx=4)

        self.results_plot_frame = tk.Frame(tab)
        self.results_plot_frame.pack(fill="both", expand=True, padx=20, pady=5)

        table_frame = tk.Frame(tab)
        table_frame.pack(fill="x", padx=20, pady=(0, 10))
        scroll = tk.Scrollbar(table_frame)
        scroll.pack(side="right", fill="y")
        self.results_tree = ttk.Treeview(
            table_frame, columns=[c[0] for c in self.RESULT_COLUMNS],
            show="headings", height=9, yscrollcommand=scroll.set)
        for column, heading, width in self.RESULT_COLUMNS:
            self.results_tree.heading(column, text=heading)
            self.results_tree.column(column, width=width, anchor="center")
        self.results_tree.pack(side="left", fill="x", expand=True)
        scroll.config(command=self.results_tree.yview)

    def _update_results(self):
        """Redraw the plot and the table from whatever has been analysed."""
        table = results.results_table(self.loaded_files, self.file_groups)
        self._results_table = table

        self.results_tree.delete(*self.results_tree.get_children())

        if table.empty:
            self._draw_results_plot()
            self.results_note_var.set(
                ("Every fish was left out." + self._exclusion_note())
                if results.excluded_fish(self.loaded_files) else
                "No results yet. Load sessions and press \"Run All Analysis\" "
                "on the Sessions & Units tab.")
            self.results_export_button.config(state="disabled")
            return

        sessions = list(dict.fromkeys(table["Session"]))
        groups = list(dict.fromkeys(table["Group"]))
        per_group = table.groupby("Group", sort=False)["Session"].nunique()
        pooled = table["Fish"] == results.POOLED
        fish = f"{int((~pooled).sum())} fish" if not pooled.all() else ""
        tanks = (f"{int(pooled.sum())} tank(s) tracked without identities"
                 if pooled.any() else "")
        self.results_note_var.set(
            f"{' and '.join(t for t in (fish, tanks) if t)} in "
            f"{len(sessions)} session(s), {len(groups)} group(s). Small dots "
            "are fish, large markers are session means, and the line is the "
            "mean of a group's sessions. Fish sharing a tank are not "
            "independent, so compare the large markers."
            + ("" if not pooled.any() else
               "  A session tracked without identities cannot tell its fish "
               "apart, so it has one value for the whole tank.")
            + ("" if per_group.min() > 1 else
               "  A group with one session cannot be tested against another.")
            + self._exclusion_note())
        self.results_export_button.config(state="normal")

        self._draw_results_plot()

        unit = table["Unit"].iloc[0]
        self.results_tree.heading(
            "DistancePerTrackedMin", text=f"Distance per tracked min ({unit})")
        self.results_tree.heading("MedianSpeed", text=f"Median speed ({unit}/s)")
        self.results_tree.heading("Speed95", text=f"95th pct speed ({unit}/s)")
        for _, row in table.iterrows():
            self.results_tree.insert("", "end", values=(
                row["Group"], row["Session"], row["Fish"],
                f"{row['Tracked_pct']:.1f}", row["Jumps"], f"{row['DistancePerTrackedMin']:.1f}",
                f"{row['MedianSpeed']:.2f}", f"{row['Speed95']:.2f}",
                f"{row['Straightness']:.2f}",
                "" if row["NearWall"] != row["NearWall"] else f"{row['NearWall']:.1f}",
                "" if row["RestingTime"] != row["RestingTime"]
                else f"{row['RestingTime']:.1f}"))

    def _exclusion_note(self) -> str:
        left_out = results.excluded_fish(self.loaded_files)
        if not left_out:
            return ""
        shown = "; ".join(left_out[:3]) + ("; ..." if len(left_out) > 3 else "")
        return (f"  {len(left_out)} fish left out ({shown}); the export lists "
                "them all.")

    def _draw_results_plot(self):
        """Draw whichever view is selected, from the current results."""
        for child in self.results_plot_frame.winfo_children():
            child.destroy()
        self.results_smoothing_box.config(
            state="readonly" if self.results_view.get() == "minutes" else "disabled")
        self.results_each_fish_check.config(
            state="normal" if self.results_view.get() == "minutes" else "disabled")
        self.results_ridge_box.config(
            state="readonly" if self.results_view.get() == "distributions"
            else "disabled")
        self.results_speed_top_box.config(
            state="normal" if self.results_view.get() == "distributions"
            else "disabled")
        self.results_squares_box.config(
            state="normal" if self.results_view.get() == "density" else "disabled")
        table = self._results_table
        if table is None or table.empty:
            return

        sessions = list(dict.fromkeys(table["Session"]))
        colors = fish_colors(len(sessions))
        session_colors = {name: tuple(colors[i]) for i, name in enumerate(sessions)}
        figure = Figure(figsize=(11, 3.6), dpi=100)

        if self.results_view.get() == "distributions":
            samples = results.speed_samples(self.loaded_files, self.file_groups)
            if not samples:
                return
            left, right = figure.subplots(1, 2)
            level = dict(results.RIDGE_LEVELS).get(self.results_ridge_box.get(), "fish")
            try:
                top = float(self.results_speed_top_var.get())
            except ValueError:
                top = None                 # blank, or not a number: automatic
            if level == "groups":
                colors = results.group_colors(table)
                results.plot_group_speed_ecdf(left, samples, colors, top)
                results.plot_group_speed_ridges(right, samples, colors, top)
            else:
                results.plot_session_speed_ecdf(left, samples, session_colors, top,
                                                each_fish=level == "fish")
                results.plot_fish_speed_ridges(
                    right, results.tank_samples(samples) if level == "tanks"
                    else samples, session_colors, top)
            figure.tight_layout()
        elif self.results_view.get() == "density":
            figure.set_layout_engine("constrained")
            cell = results.density_cell(self.loaded_files, sessions,
                                        self._results_squares_across())
            maps = [results.position_density(self.loaded_files[name], cell)
                    for name in sessions]
            ceiling = results.shared_density_ceiling([m[0] for m in maps])
            rows, columns = results.panel_grid(len(sessions))
            axes = []
            for index, (name, (density, x_edges, y_edges)) in enumerate(
                    zip(sessions, maps)):
                axes.append(figure.add_subplot(rows, columns, index + 1))
                image = results.plot_position_density(
                    axes[-1], name, self.loaded_files[name], density, x_edges,
                    y_edges, ceiling, self.file_arena_definitions.get(name))
            unit = self.loaded_files[sessions[0]].calibration.unit_name
            figure.colorbar(
                image, ax=axes, shrink=0.85, extend="max",
                label=f"% of time in each {cell:.2g} {unit} square")
        else:
            by_minute = self.results_view.get() == "minutes"
            smooth = dict(results.SMOOTHING_CHOICES).get(
                self.results_smoothing_box.get(), 1) if by_minute else 1
            if by_minute:
                minutes = results.minute_table(self.loaded_files, self.file_groups)
                colors = results.group_colors(minutes)
            metrics = [metric for metric in results.METRICS
                       if self.results_panel_vars[metric.column].get()]
            if not metrics:
                tk.Label(self.results_plot_frame, text="No panels ticked.",
                         fg="gray").pack(pady=40)
                return
            resting = results.resting_title(self._results_params(),
                                            results.shared_unit(table))
            axes = np.atleast_1d(figure.subplots(1, len(metrics)))
            for ax, metric in zip(axes, metrics):
                title = resting if metric.column == "RestingTime" else None
                if by_minute:
                    results.minute_plot(ax, minutes, metric, colors, smooth,
                                        self.results_each_fish.get(), title)
                else:
                    results.superplot(ax, table, metric, session_colors, title)
            if by_minute:
                results.minute_legend(figure, colors)
            else:
                figure.legend(
                    handles=[Line2D([], [], marker="o", linestyle="", markersize=9,
                                    markerfacecolor=session_colors[name],
                                    markeredgecolor="black", label=name)
                             for name in sessions],
                    loc="lower center", ncol=min(len(sessions), 6), frameon=False,
                    fontsize=9, title="Sessions")
            figure.suptitle(results.smoothing_note(smooth), fontsize=9,
                            color="#b35c00")
            figure.tight_layout(rect=(0, 0.13, 1, 0.95 if smooth > 1 else 1))
        embed_figure_with_toolbar(figure, self.results_plot_frame)

    def _results_squares_across(self) -> int:
        """The heatmap grid box; anything that is not a sensible whole
        number is the default, 48."""
        try:
            return min(200, max(4, int(float(self.results_squares_var.get()))))
        except (ValueError, tk.TclError):
            return 48

    def _results_params(self):
        """The settings the results on show were made with, or None. One run
        of the analysis uses one set for every session."""
        for loaded in self.loaded_files.values():
            params = getattr(loaded, "processing_params", None)
            if loaded.processed_data and params is not None:
                return params
        return None

    def _explain_measures(self):
        """Say, in plain words, what each measure is and how it is made."""
        window = tk.Toplevel(self.root)
        window.title("What the measures mean")
        window.geometry("720x520")
        text = tk.Text(window, wrap="word", font=("Arial", 10), padx=14, pady=12)
        text.pack(fill="both", expand=True)
        text.tag_configure("title", font=("Arial", 11, "bold"), spacing1=10)
        for metric in results.METRICS:
            text.insert("end", metric.title + "\n", "title")
            text.insert("end", metric.meaning + "\n")
        text.insert("end", "Jumps\n", "title")
        text.insert("end", results.JUMPS_MEANING + "\n")
        text.insert("end", "Minute by minute\n", "title")
        text.insert("end", results.MINUTE_MEANING + "\n")
        text.insert("end", "Reading the plots\n", "title")
        text.insert(
            "end",
            "Small dots are fish. Large markers are session means. The black "
            "line is the mean of a group's session means. Fish sharing a tank "
            "influence each other, so the session is the unit to compare, and "
            "a group needs several sessions before it can be tested against "
            "another.\n")
        text.insert("end", "Sessions tracked without identities\n", "title")
        text.insert(
            "end",
            "Tracked without identities, idtracker.ai follows fish but cannot "
            "tell them apart: when it loses a fish it may carry the track on "
            "with another fish, so no track is one fish. The track then jumps "
            "across the tank, and those jumps are left out like any other "
            "(see Jumps). All the tracks are pooled into one value for the "
            "tank (Fish: \"all fish\"), and distance is shared out over the "
            "number of fish the setup gave idtracker.ai.\n")
        text.config(state="disabled")
        return window

    def _export_results(self):
        if self._results_table is None or self._results_table.empty:
            return
        parent = filedialog.askdirectory(
            title="Export results: choose where to put the results folder")
        if not parent:
            return
        folder = results_export.export_results_folder(
            self.loaded_files, self.file_groups, Path(parent),
            self._results_squares_across())
        messagebox.showinfo(
            "Exported",
            f"Results saved in:\n{folder}\n\nOne CSV for each view on this tab, "
            "and a README.txt saying what each column is:\n    "
            + "\n    ".join(results_export.FILES))
