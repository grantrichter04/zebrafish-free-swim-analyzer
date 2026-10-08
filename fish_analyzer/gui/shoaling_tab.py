"""
fish_analyzer/gui/shoaling_tab.py
=================================
Shoaling Tab - how close the fish keep to each other, on one screen.

The analysis runs with everything else when "Run All Analysis" is pressed on
the Sessions & Units tab. This tab only shows it: the two measures as
SuperPlots against what randomly placed fish would give, the same measures
over the course of the recording, a table, and an export.

What the measures are is explained in shoal_summary.py.
"""
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .. import results, shoal_summary
from ..export import export_shoaling_metrics_csv, export_shoaling_summary_csv
from ..overlay_render import fish_colors
from .utils import ask_csv_save_path, embed_figure_with_toolbar


def _unit(all_results: Dict[str, Any]) -> str:
    """The calibrated unit shared by these results, or a warning label.

    Axis labels used to be hardcoded "BL" while the calculator ignored the
    file's calibration entirely, so a cm-calibrated session was plotted with a
    BL axis (finding B7). Files with different calibrations cannot share an
    axis at all, and saying so is better than picking one.
    """
    units = {r.unit_name for r in all_results.values()}
    return units.pop() if len(units) == 1 else "mixed units"


class ShoalingTabMixin:
    """
    Mixin providing the Shoaling tab.

    Expects the following attributes from base class:
    - self.notebook: ttk.Notebook
    - self.loaded_files, self.file_groups, self.file_arena_definitions
    """

    SHOALING_COLUMNS = (
        ("Group", "Group", 110), ("Session", "Session", 200), ("Fish", "Fish", 50),
        ("NND", "Nearest neighbour", 150), ("RandomNND", "if random", 100),
        ("IID", "Inter-individual", 140), ("RandomIID", "if random", 100),
        ("Hull", "Area covered", 130), ("RandomHull", "if random", 100),
        ("FramesUsed_pct", "Frames used %", 110),
    )
    NO_SHOALING_YET = ("No results yet. Load sessions and press \"Run All "
                       "Analysis\" on the Sessions & Units tab.")

    def _create_shoaling_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Shoaling")
        self.shoaling_tab_frame = tab
        self._shoaling_table = None

        top = tk.Frame(tab)
        top.pack(fill="x", padx=20, pady=(10, 0))
        self.shoaling_note_var = tk.StringVar(value=self.NO_SHOALING_YET)
        tk.Label(top, textvariable=self.shoaling_note_var, justify=tk.LEFT,
                 anchor="w", font=("Arial", 9), fg="gray30", wraplength=900
                 ).pack(side="left", fill="x", expand=True)
        self.shoaling_export_button = tk.Button(
            top, text="Export shoaling (CSV)...", command=self._export_shoaling_csv,
            bg="lightblue", font=("Arial", 10, "bold"), state="disabled")
        self.shoaling_export_button.pack(side="right")

        view_row = tk.Frame(tab)
        view_row.pack(fill="x", padx=20, pady=(6, 0))
        tk.Label(view_row, text="Show:", font=("Arial", 10, "bold")).pack(side="left")
        self.shoaling_view = tk.StringVar(value="comparison")
        for value, text in (("comparison", "Group comparison"),
                            ("time", "Minute by minute")):
            tk.Radiobutton(view_row, text=text, value=value,
                           variable=self.shoaling_view,
                           command=self._draw_shoaling_plot).pack(side="left", padx=8)
        tk.Button(view_row, text="What do these measures mean?",
                  command=self._explain_shoaling).pack(side="right")

        self.shoaling_plot_frame = tk.Frame(tab)
        self.shoaling_plot_frame.pack(fill="both", expand=True, padx=20, pady=5)

        table_frame = tk.Frame(tab)
        table_frame.pack(fill="x", padx=20, pady=(0, 10))
        scroll = tk.Scrollbar(table_frame)
        scroll.pack(side="right", fill="y")
        self.shoaling_tree = ttk.Treeview(
            table_frame, columns=[c[0] for c in self.SHOALING_COLUMNS],
            show="headings", height=6, yscrollcommand=scroll.set)
        for column, heading, width in self.SHOALING_COLUMNS:
            self.shoaling_tree.heading(column, text=heading)
            self.shoaling_tree.column(column, width=width, anchor="center")
        self.shoaling_tree.pack(side="left", fill="x", expand=True)
        scroll.config(command=self.shoaling_tree.yview)

    def _update_shoaling(self):
        """Redraw the plot and the table from whatever has been analysed."""
        table = shoal_summary.shoaling_table(
            self.loaded_files, self.file_groups, self.file_arena_definitions)
        self._shoaling_table = table
        self.shoaling_tree.delete(*self.shoaling_tree.get_children())
        self._draw_shoaling_plot()

        if table.empty:
            self.shoaling_note_var.set(self.NO_SHOALING_YET)
            self.shoaling_export_button.config(state="disabled")
            return

        summary = shoal_summary.session_summary(table)
        self.shoaling_note_var.set(
            f"{len(summary)} session(s), {summary['Group'].nunique()} group(s). "
            "Small dots are fish, large markers are sessions. The dashed line "
            "is what fish placed at random in the tank would give: below it, "
            "they keep together more than chance."
            + ("" if summary["RandomNND"].notna().all() else
               "  Sessions without an arena outline have no random reference."))
        self.shoaling_export_button.config(state="normal")

        unit = results.shared_unit(table)
        self.shoaling_tree.heading("NND", text=f"Nearest neighbour ({unit})")
        self.shoaling_tree.heading("IID", text=f"Inter-individual ({unit})")
        self.shoaling_tree.heading("Hull", text=f"Area covered ({unit}\u00b2)")

        def number(value) -> str:
            return "" if value != value else f"{value:.2f}"

        for _, row in summary.iterrows():
            self.shoaling_tree.insert("", "end", values=(
                row["Group"], row["Session"], row["Fish"],
                number(row["NND"]), number(row["RandomNND"]),
                number(row["IID"]), number(row["RandomIID"]),
                number(row["Hull"]), number(row["RandomHull"]),
                f"{row['FramesUsed_pct']:.1f}"))

    def _draw_shoaling_plot(self):
        """Draw whichever view is selected, from the current results."""
        for child in self.shoaling_plot_frame.winfo_children():
            child.destroy()
        table = self._shoaling_table
        if table is None or table.empty:
            return

        sessions = list(dict.fromkeys(table["Session"]))
        colors = fish_colors(len(sessions))
        session_colors = {name: tuple(colors[i]) for i, name in enumerate(sessions)}
        unit = results.shared_unit(table)
        figure = Figure(figsize=(11, 3.6), dpi=100)
        axes = figure.subplots(1, len(shoal_summary.MEASURES))

        by_minute = self.shoaling_view.get() == "time"
        if by_minute:
            minutes = shoal_summary.minute_table(self.loaded_files, self.file_groups)
        for ax, (column, title, _) in zip(axes, shoal_summary.MEASURES):
            label = shoal_summary.measure_unit(column, unit)
            # The hull is one value for the shoal, so one point per session.
            shown = table.drop_duplicates("Session") if column == "Hull" else table
            few = "Needs at least\nthree fish."
            if by_minute:
                results.draw_minute_lines(ax, minutes, column, title, label,
                                          session_colors, empty_message=few)
            elif not results.draw_superplot(ax, shown, column, title, label,
                                            session_colors, empty_message=few):
                continue
            reference = table[f"Random{column}"].dropna()
            if len(reference):
                results.draw_reference_line(ax, reference.mean(), "if random")
                ax.set_ylim(top=max(ax.get_ylim()[1], reference.mean() * 1.15))

        figure.legend(
            handles=[Line2D([], [], marker="o", linestyle="", markersize=9,
                            markerfacecolor=session_colors[name],
                            markeredgecolor="black", label=name)
                     for name in sessions],
            loc="lower center", ncol=min(len(sessions), 6), frameon=False,
            fontsize=9, title="Sessions")
        figure.tight_layout(rect=(0, 0.13, 1, 1))
        embed_figure_with_toolbar(figure, self.shoaling_plot_frame)

    def _explain_shoaling(self):
        """Say, in plain words, what each measure is and how it is made."""
        window = tk.Toplevel(self.root)
        window.title("What the shoaling measures mean")
        window.geometry("720x620")
        text = tk.Text(window, wrap="word", font=("Arial", 10), padx=14, pady=12)
        text.pack(fill="both", expand=True)
        text.tag_configure("title", font=("Arial", 11, "bold"), spacing1=10)
        for _, title, meaning in shoal_summary.MEASURES:
            text.insert("end", title + "\n", "title")
            text.insert("end", meaning + "\n")
        text.insert("end", "Minute by minute\n", "title")
        text.insert("end", results.MINUTE_MEANING + "\n")
        text.insert("end", "The \"if random\" line\n", "title")
        text.insert("end", shoal_summary.REFERENCE_MEANING + "\n")
        text.insert("end", "Which moments are used\n", "title")
        text.insert("end", shoal_summary.SAMPLING_MEANING + "\n")
        text.config(state="disabled")
        return window

    def _export_shoaling_csv(self):
        """Export the per-second values and the per-session summary."""
        analyzed = {k: v for k, v in self.loaded_files.items()
                    if getattr(v, 'shoaling_results', None) is not None}
        if not analyzed:
            return

        output_path = ask_csv_save_path("Export Shoaling Results CSV", "shoaling_timeseries.csv")
        if not output_path:
            return

        try:
            out = output_path
            n_ts = export_shoaling_metrics_csv(analyzed, out)
            summary_path = out.with_name(out.stem + "_summary.csv")
            n_sum = export_shoaling_summary_csv(analyzed, summary_path)

            self.set_status(f"Exported shoaling data to {out.name}")
            messagebox.showinfo(
                "Export Complete",
                f"Exported shoaling data:\n\n"
                f"Time series: {n_ts} rows → {out.name}\n"
                f"Summary: {n_sum} rows → {summary_path.name}"
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{e}")
