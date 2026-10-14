"""
fish_analyzer/results_export.py
===============================
The Results tab's export: a folder of CSV files, one for each view on the
tab, holding enough to draw that view again in R, Prism or Excel.

    results_per_fish.csv    Group comparison: one row per fish (or tank)
    results_by_minute.csv   Minute by minute: one row per fish per minute
    speed_histogram.csv     Speed distributions: frames in each speed bin
    position_density.csv    Where they swim: share of time in each square
    README.txt              what each file and column is
    prism/                  the same numbers one value per tank, laid out to
                            paste into GraphPad Prism (see PRISM_README)

Every file is long format, one observation per row, and every row says how
its session was tracked (Tracking: "identities" or "no identities"), so the
two kinds cannot be mixed up once the files leave the app.
"""
import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import results

IDENTITIES = "identities"
NO_IDENTITIES = "no identities"


def _tracking(loaded_files: Dict, session: str) -> str:
    return NO_IDENTITIES if results.pooled(loaded_files[session]) else IDENTITIES


def _with_tracking(table: pd.DataFrame, loaded_files: Dict) -> pd.DataFrame:
    """`table` with a Tracking column after Session."""
    table = table.copy()
    table.insert(table.columns.get_loc("Session") + 1, "Tracking",
                 table["Session"].map(lambda s: _tracking(loaded_files, s)))
    return table


def nice_bin_width(top: float, bins: int = 100) -> float:
    """A round bin width (1, 2 or 5 times a power of ten) giving about
    `bins` bins between zero and `top`."""
    if not top > 0:
        return 1.0
    raw = top / bins
    power = 10 ** np.floor(np.log10(raw))
    for step in (1, 2, 5, 10):
        if step * power >= raw:
            return float(step * power)
    return float(10 * power)


def speed_histogram(loaded_files: Dict, file_groups: Optional[Dict[str, str]] = None
                    ) -> pd.DataFrame:
    """Frames in each speed bin, for each fish (or tank). Bins are shared by
    every row and run from zero to past the fastest frame, so the counts
    redraw the cumulative curves exactly to within a bin and the ridges as
    drawn."""
    samples = results.speed_samples(loaded_files, file_groups)
    columns = ["Group", "Session", "Tracking", "Fish", "Unit", "SpeedFrom",
               "SpeedTo", "Frames", "Frames_pct"]
    if not samples:
        return pd.DataFrame(columns=columns)
    width = nice_bin_width(max(np.percentile(s.speeds, 99.5) for s in samples))
    top = max(s.speeds.max() for s in samples)
    edges = np.arange(0.0, top + width, width)
    rows = []
    for sample in samples:
        counts, _ = np.histogram(sample.speeds, bins=edges)
        for low, high, count in zip(edges[:-1], edges[1:], counts):
            rows.append({
                "Group": sample.group, "Session": sample.session,
                "Tracking": _tracking(loaded_files, sample.session),
                "Fish": sample.fish, "Unit": f"{sample.unit}/s",
                "SpeedFrom": round(low, 6), "SpeedTo": round(high, 6),
                "Frames": int(count),
                "Frames_pct": count / len(sample.speeds) * 100})
    return pd.DataFrame(rows, columns=columns)


def position_density_table(loaded_files: Dict,
                           file_groups: Optional[Dict[str, str]] = None,
                           squares_across: int = 48) -> pd.DataFrame:
    """The share of each session's tracked positions in each square cell,
    the numbers behind "Where they swim". y runs up from the bottom of the
    video, as drawn."""
    file_groups = file_groups or {}
    sessions = [name for name, loaded in loaded_files.items() if loaded.processed_data]
    columns = ["Group", "Session", "Tracking", "Unit", "XFrom", "XTo", "YFrom",
               "YTo", "Time_pct"]
    if not sessions:
        return pd.DataFrame(columns=columns)
    cell = results.density_cell(loaded_files, sessions, squares_across)
    rows = []
    for name in sessions:
        loaded = loaded_files[name]
        density, x_edges, y_edges = results.position_density(loaded, cell)
        for row in range(density.shape[0]):
            for column in range(density.shape[1]):
                rows.append({
                    "Group": file_groups.get(name) or results.default_group(name),
                    "Session": name, "Tracking": _tracking(loaded_files, name),
                    "Unit": loaded.calibration.unit_name,
                    "XFrom": round(x_edges[column], 6),
                    "XTo": round(x_edges[column + 1], 6),
                    "YFrom": round(y_edges[row], 6),
                    "YTo": round(y_edges[row + 1], 6),
                    "Time_pct": density[row, column]})
    return pd.DataFrame(rows, columns=columns)


README = """\
Free swim results, exported {when}

Every file has one observation per row (long format). Group, Session and
Tracking are on every row. Tracking is "identities" when idtracker.ai told
the fish apart, "no identities" when it did not; then a session has one row
for the whole tank (Fish: "{pooled}") instead of one per fish.
Do not compare the two kinds in one analysis.

The tank, not the fish, is the experimental unit: fish sharing a tank are
not independent. Average a session's rows first, then compare sessions.

results_per_fish.csv - "Group comparison"
  One row per fish (or tank), then one row per fish left out and why.
  DistancePerTrackedMin  distance swum per minute the fish was seen, in
                     Unit; fair between fish tracked more or less of the time
  Distance           total distance swum, in Unit (lower for a fish tracked
                     less of the time; the app shows the per-minute figure)
  MedianSpeed, Speed95   median and 95th percentile speed, Unit per second
  Straightness       straight-line distance / distance swum, per second of
                     swimming, averaged (0 to 1)
  NearWall           % of time in the zone along the walls; WallZoneArea_pct
                     is that zone's share of the tank (the "even use" line)
  RestingTime        % of tracked time spent resting: slower than
                     Setting_FreezeSpeed for at least Setting_FreezeMin_s
  Tracked_pct        % of frames the fish (or the tank's fish) was seen
  Jumps              frame-to-frame steps too fast to be swimming, left out
  Setting_*          the settings the numbers were made with

results_by_minute.csv - "Minute by minute"
  The same measures worked out on each minute alone (Minute 1 is the
  first). Plot each session as the mean of its fish, and each group as the
  mean of its sessions.

speed_histogram.csv - "Speed distributions"
  Frames whose speed fell in [SpeedFrom, SpeedTo), per fish (or tank).
  Frames_pct is the share of that fish's tracked frames. Cumulative sums of
  Frames_pct give the curves on the left; the bars give the ridges.

position_density.csv - "Where they swim"
  The % of a session's tracked positions in each square, XFrom-XTo by
  YFrom-YTo, in Unit, y measured up from the bottom of the video.

prism/ - the group comparison and minute-by-minute numbers, one value per
  tank, laid out to paste straight into GraphPad Prism. PRISM_README.txt
  there says which table type to make and how.
"""


PRISM_README = """\
Tables for GraphPad Prism, exported {when}

Every number here is a tank (session) value: the mean of its fish, or for a
session tracked without identities the value for the whole tank. That is
the unit to plot and test, so nothing needs averaging first.

group_comparison_<measure>.csv  -> a Column table
  One column per group, one row per session. In Prism: New table, Column,
  "Enter replicate values, stacked into columns". Open the CSV in Excel,
  copy everything including the group names in the first row, and paste
  into the table's first cell. Prism draws the dot plot; a t-test or
  one-way ANOVA from Analyze compares the groups.
  Which session is in which row: sessions_in_each_row.csv.

by_minute_<measure>.csv  -> an XY table
  Minute in the first column, then each group's sessions side by side,
  headed "group: session". Every group has {width} column(s); a group with
  fewer sessions has blank ones. In Prism: New table, XY, X as numbers, Y
  "Enter {width} replicate values in side-by-side subcolumns". Paste the
  numbers (not the header row) into the first X cell, then type the group
  names as the data set titles. Prism draws each group's mean with its
  spread (choose SD, SEM or each replicate under Format graph).

The measures:
{measures}
{note}"""


def prism_tables(loaded_files: Dict, file_groups: Optional[Dict[str, str]],
                 folder: Path) -> List[str]:
    """Write the Prism tables into `folder` and return their names."""
    folder.mkdir(exist_ok=True)
    written = []
    metrics = results.METRICS
    means = results.session_means(results.results_table(loaded_files, file_groups),
                                  [m.column for m in metrics])
    groups = list(dict.fromkeys(means["Group"]))
    sessions = {g: list(means.loc[means["Group"] == g, "Session"]) for g in groups}
    rows = max(len(names) for names in sessions.values())

    order = [{"Row": row + 1, **{g: (sessions[g][row] if row < len(sessions[g]) else "")
                                 for g in groups}} for row in range(rows)]
    pd.DataFrame(order, columns=["Row"] + groups).to_csv(
        folder / "sessions_in_each_row.csv", index=False)
    written.append("sessions_in_each_row.csv")

    minutes = results.minute_table(loaded_files, file_groups)
    unit = results.shared_unit(results.results_table(loaded_files, file_groups))
    lines = []
    for metric in metrics:
        comparison = pd.DataFrame({
            g: pd.Series(means.loc[means["Group"] == g, metric.column].to_list())
            for g in groups})
        name = f"group_comparison_{metric.column}.csv"
        comparison.to_csv(folder / name, index=False, float_format="%.5g")
        written.append(name)

        # A session's minute is the mean of its fish in that minute.
        per_session = (minutes.groupby(["Group", "Session", "Minute"], sort=False)
                       [metric.column].mean().reset_index())
        wide = pd.DataFrame({"Minute": sorted(per_session["Minute"].unique())})
        for g in groups:
            for index in range(rows):
                if index < len(sessions[g]):
                    session = sessions[g][index]
                    values = (per_session[per_session["Session"] == session]
                              .set_index("Minute")[metric.column])
                    wide[f"{g}: {session}"] = wide["Minute"].map(values)
                else:
                    wide[f"{g}: (none)"] = np.nan
        name = f"by_minute_{metric.column}.csv"
        wide.to_csv(folder / name, index=False, float_format="%.5g")
        written.append(name)
        lines.append(f"  {metric.column:22s} {metric.title}, {metric.axis_label(unit)}")

    params = next((getattr(l, "processing_params", None) for l in loaded_files.values()
                   if l.processed_data and getattr(l, "processing_params", None)), None)
    note = ("\nRestingTime counts a fish resting when slower than "
            f"{params.rest_speed_threshold:g} {unit}/s for at least "
            f"{params.min_freeze_seconds:g} s.\n" if params is not None else "")
    (folder / "PRISM_README.txt").write_text(PRISM_README.format(
        when=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), width=rows,
        measures="\n".join(lines), note=note), encoding="utf-8")
    written.append("PRISM_README.txt")
    return written


def export_results_folder(loaded_files: Dict, file_groups: Optional[Dict[str, str]],
                          parent: Path, squares_across: int = 48) -> Path:
    """Write the export into a new folder inside `parent` and return it.

    The folder is named free_swim_results_<date>, with _2, _3... added if
    that exists, so an export never overwrites an earlier one.
    """
    stamp = datetime.date.today().isoformat()
    folder = Path(parent) / f"free_swim_results_{stamp}"
    number = 2
    while folder.exists():
        folder = Path(parent) / f"free_swim_results_{stamp}_{number}"
        number += 1
    folder.mkdir(parents=True)

    _with_tracking(results.results_export_table(loaded_files, file_groups),
                   loaded_files).to_csv(folder / "results_per_fish.csv", index=False)
    _with_tracking(results.minute_table(loaded_files, file_groups),
                   loaded_files).to_csv(folder / "results_by_minute.csv", index=False)
    speed_histogram(loaded_files, file_groups).to_csv(
        folder / "speed_histogram.csv", index=False)
    position_density_table(loaded_files, file_groups, squares_across).to_csv(
        folder / "position_density.csv", index=False)
    (folder / "README.txt").write_text(
        README.format(when=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                      pooled=results.POOLED), encoding="utf-8")
    if not results.results_table(loaded_files, file_groups).empty:
        prism_tables(loaded_files, file_groups, folder / "prism")
    return folder


FILES: List[str] = ["results_per_fish.csv", "results_by_minute.csv",
                    "speed_histogram.csv", "position_density.csv", "README.txt",
                    "prism"]
