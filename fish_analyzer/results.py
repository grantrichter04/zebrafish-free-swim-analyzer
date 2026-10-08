"""
fish_analyzer/results.py
========================
The headline results: one row per fish, a few metrics, and a plot that keeps
the experimental unit in view.

The metrics answer "is one group more active than another?" and none of them
needs a threshold:

    Distance        how much the fish swam
    TypicalSpeed    the median speed: how fast it usually goes
    PeakSpeed       the 99th percentile of speed: how fast its fast moments are
    Straightness    1 = straight paths, lower = more turning

Mean speed is left out on purpose: it is distance divided by time, so beside
distance it says nothing new.

THE UNIT OF OBSERVATION IS THE TANK, NOT THE FISH. Fish filmed together shoal:
each one's speed and position depend on the others', so six fish in a tank
are not six independent measurements. The plot is therefore a SuperPlot (Lord
et al. 2020, J Cell Biol 219:e202001064): every fish is a small dot, each
session's mean is a large marker, and the line for a group is the mean of its
session means. Statistics belong on the large markers.
"""
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metric:
    column: str          # column in the results table and the CSV
    source: str          # key in FishTrajectory.metrics
    title: str           # panel title
    per_time: bool       # a speed (unit/s) rather than a length (unit)
    unitless: bool = False

    def axis_label(self, unit: str) -> str:
        if self.unitless:
            return "1 = straight"
        return f"{unit}/s" if self.per_time else unit


METRICS: List[Metric] = [
    Metric("Distance", "total_distance", "Total distance", per_time=False),
    Metric("TypicalSpeed", "median_speed", "Typical speed (median)", per_time=True),
    Metric("PeakSpeed", "speed_p99", "Peak speed (99th percentile)", per_time=True),
    Metric("Straightness", "mean_path_straightness", "Path straightness",
           per_time=False, unitless=True),
]


def default_group(session_name: str) -> str:
    """A group for a session nobody has grouped: its name without a trailing
    number, so control_1 and control_2 land together."""
    stripped = re.sub(r"[_\-\s]*\d+$", "", session_name).strip()
    return stripped or session_name


def results_table(loaded_files: Dict, file_groups: Optional[Dict[str, str]] = None
                  ) -> pd.DataFrame:
    """One row per analysed fish. Sessions without results are skipped."""
    file_groups = file_groups or {}
    rows = []
    for name, loaded in loaded_files.items():
        if not loaded.processed_data:
            continue
        group = file_groups.get(name) or default_group(name)
        for fish in loaded.processed_data:
            row = {
                "Group": group,
                "Session": name,
                "Fish": fish.identity_label,
                "Tracked_pct": round(fish.valid_percentage * 100, 1),
            }
            for metric in METRICS:
                row[metric.column] = fish.metrics.get(metric.source, np.nan)
            row["Unit"] = loaded.calibration.unit_name
            row["PixelsPerUnit"] = round(loaded.calibration.pixels_per_unit, 4)
            rows.append(row)
    columns = (["Group", "Session", "Fish", "Tracked_pct"]
               + [m.column for m in METRICS] + ["Unit", "PixelsPerUnit"])
    return pd.DataFrame(rows, columns=columns)


def session_means(table: pd.DataFrame) -> pd.DataFrame:
    """One row per session: the mean of its fish. The experimental unit."""
    columns = [m.column for m in METRICS]
    return (table.groupby(["Group", "Session"], sort=False)[columns]
            .mean().reset_index())


def superplot(ax, table: pd.DataFrame, metric: Metric,
              session_colors: Dict[str, tuple]) -> None:
    """Draw one metric: fish as small dots, sessions as large markers, and a
    line per group at the mean of its session means."""
    groups = list(dict.fromkeys(table["Group"]))
    means = session_means(table)

    for position, group in enumerate(groups):
        in_group = table[table["Group"] == group]
        sessions = list(dict.fromkeys(in_group["Session"]))
        # Sessions sit side by side inside the group's slot; fish are spread
        # evenly inside their session's share, so the layout never changes
        # between redraws the way random jitter would.
        slot = 0.7 / max(1, len(sessions))
        for index, session in enumerate(sessions):
            centre = position - 0.35 + slot * (index + 0.5)
            values = in_group.loc[in_group["Session"] == session, metric.column].to_numpy(float)
            values = values[~np.isnan(values)]
            if len(values) == 0:
                continue
            spread = np.linspace(-0.35, 0.35, len(values)) * slot if len(values) > 1 else [0.0]
            color = session_colors[session]
            ax.scatter(centre + np.asarray(spread), values, s=22, color=color,
                       alpha=0.55, linewidths=0, zorder=2)
            ax.scatter([centre], [values.mean()], s=130, color=color,
                       edgecolors="black", linewidths=1.2, zorder=3)

        group_means = means.loc[means["Group"] == group, metric.column].dropna()
        if len(group_means):
            ax.hlines(group_means.mean(), position - 0.4, position + 0.4,
                      colors="black", linewidths=2, zorder=4)

    ax.set_xticks(range(len(groups)))
    crowded = len(groups) > 3 or max(len(str(g)) for g in groups) > 10
    ax.set_xticklabels(groups, rotation=25 if crowded else 0,
                       ha="right" if crowded else "center")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_title(metric.title, fontsize=11, fontweight="bold")
    units = list(dict.fromkeys(table["Unit"]))
    ax.set_ylabel(metric.axis_label(units[0] if len(units) == 1 else "mixed units"))
    if metric.unitless:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
