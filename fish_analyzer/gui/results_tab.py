"""
fish_analyzer/gui/results_tab.py
================================
Results Tab - the answer to "is one group more active than another?" on one
screen: a SuperPlot per metric, the table behind it, and one export.

What is shown, and why the session is the unit, is explained in results.py.
"""
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .. import results
from ..overlay_render import fish_colors
from .utils import ask_csv_save_path, embed_figure_with_toolbar


class ResultsTabMixin:
    """
    Mixin providing the Results tab.

    Expects the following attributes from base class:
    - self.notebook: ttk.Notebook
    - self.loaded_files, self.file_groups, self.file_arena_definitions
    """

    RESULT_COLUMNS = (
        ("Group", "Group", 130), ("Session", "Session", 200), ("Fish", "Fish", 60),
        ("Tracked_pct", "Tracked %", 80), ("Distance", "Distance", 100),
        ("TypicalSpeed", "Typical speed", 110), ("PeakSpeed", "Peak speed", 100),
        ("Straightness", "Straightness", 100), ("NearWall", "Near wall %", 100),
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
            top, text="Export results (CSV)...", command=self._export_results,
            bg="lightblue", font=("Arial", 10, "bold"), state="disabled")
        self.results_export_button.pack(side="right")

        view_row = tk.Frame(tab)
        view_row.pack(fill="x", padx=20, pady=(6, 0))
        tk.Label(view_row, text="Show:", font=("Arial", 10, "bold")).pack(side="left")
        self.results_view = tk.StringVar(value="comparison")
        for value, text in (("comparison", "Group comparison"),
                            ("distributions", "Speed distributions"),
                            ("paths", "Swim paths"),
                            ("density", "Where they swim")):
            tk.Radiobutton(view_row, text=text, value=value,
                           variable=self.results_view,
                           command=self._draw_results_plot).pack(side="left", padx=8)
        tk.Button(view_row, text="What do these measures mean?",
                  command=self._explain_measures).pack(side="right")

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
                "No results yet. Load sessions and press \"Run All Analysis\" "
                "on the Sessions & Units tab.")
            self.results_export_button.config(state="disabled")
            return

        sessions = list(dict.fromkeys(table["Session"]))
        groups = list(dict.fromkeys(table["Group"]))
        per_group = table.groupby("Group", sort=False)["Session"].nunique()
        self.results_note_var.set(
            f"{len(table)} fish in {len(sessions)} session(s), "
            f"{len(groups)} group(s). Small dots are fish, large markers are "
            "session means, and the line is the mean of a group's sessions. "
            "Fish sharing a tank are not independent, so compare the large "
            "markers."
            + ("" if per_group.min() > 1 else
               "  A group with one session cannot be tested against another."))
        self.results_export_button.config(state="normal")

        self._draw_results_plot()

        unit = table["Unit"].iloc[0]
        self.results_tree.heading("Distance", text=f"Distance ({unit})")
        self.results_tree.heading("TypicalSpeed", text=f"Typical speed ({unit}/s)")
        self.results_tree.heading("PeakSpeed", text=f"Peak speed ({unit}/s)")
        for _, row in table.iterrows():
            self.results_tree.insert("", "end", values=(
                row["Group"], row["Session"], row["Fish"],
                f"{row['Tracked_pct']:.1f}", f"{row['Distance']:.1f}",
                f"{row['TypicalSpeed']:.2f}", f"{row['PeakSpeed']:.2f}",
                f"{row['Straightness']:.2f}",
                "" if row["NearWall"] != row["NearWall"] else f"{row['NearWall']:.1f}"))

    def _draw_results_plot(self):
        """Draw whichever view is selected, from the current results."""
        for child in self.results_plot_frame.winfo_children():
            child.destroy()
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
            results.plot_session_speed_ecdf(left, samples, session_colors)
            results.plot_fish_speed_ridges(right, samples, session_colors)
            figure.tight_layout()
        elif self.results_view.get() == "paths":
            rows, columns = results.path_grid(len(sessions))
            for index, name in enumerate(sessions):
                loaded = self.loaded_files[name]
                ax = figure.add_subplot(rows, columns, index + 1)
                results.plot_swim_paths(
                    ax, name, loaded, fish_colors(len(loaded.processed_data)),
                    self.file_arena_definitions.get(name))
            handles, labels = ax.get_legend_handles_labels()
            figure.legend(handles, labels, loc="center right", frameon=False,
                          fontsize=9, title="Fish")
            figure.tight_layout(rect=(0, 0, 0.93, 1))
        elif self.results_view.get() == "density":
            figure.set_layout_engine("constrained")
            cell = results.density_cell(self.loaded_files, sessions)
            maps = [results.position_density(self.loaded_files[name], cell)
                    for name in sessions]
            ceiling = results.shared_density_ceiling([m[0] for m in maps])
            rows, columns = results.path_grid(len(sessions))
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
            axes = figure.subplots(1, len(results.METRICS))
            for ax, metric in zip(axes, results.METRICS):
                results.superplot(ax, table, metric, session_colors)
            figure.legend(
                handles=[Line2D([], [], marker="o", linestyle="", markersize=9,
                                markerfacecolor=session_colors[name],
                                markeredgecolor="black", label=name)
                         for name in sessions],
                loc="lower center", ncol=min(len(sessions), 6), frameon=False,
                fontsize=9, title="Sessions")
            figure.tight_layout(rect=(0, 0.13, 1, 1))
        embed_figure_with_toolbar(figure, self.results_plot_frame)

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
        text.insert("end", "Reading the plots\n", "title")
        text.insert(
            "end",
            "Small dots are fish. Large markers are session means. The black "
            "line is the mean of a group's session means. Fish sharing a tank "
            "influence each other, so the session is the unit to compare, and "
            "a group needs several sessions before it can be tested against "
            "another.\n")
        text.config(state="disabled")
        return window

    def _export_results(self):
        if self._results_table is None or self._results_table.empty:
            return
        path = ask_csv_save_path("Export results", "free_swim_results.csv")
        if path is None:
            return
        self._results_table.to_csv(path, index=False)
        messagebox.showinfo("Exported", f"Results saved to:\n{path}")
