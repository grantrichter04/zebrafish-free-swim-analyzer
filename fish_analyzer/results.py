"""
fish_analyzer/results.py
========================
The headline results: one row per fish, a few metrics, and a plot that keeps
the experimental unit in view.

The metrics answer "is one group more active than another?" and none of them
needs a threshold:

    Distance        how much the fish swam
    MedianSpeed     the median speed: how fast it usually goes
    Speed99         the 99th percentile of speed: how fast its fast moments are
    Straightness    how direct its path is, second by second (see METRICS)
    NearWall        the share of its time spent in the zone along the walls

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
    column: str            # column in the results table and the CSV
    source: Optional[str]  # key in FishTrajectory.metrics; None = computed elsewhere
    title: str             # panel title
    kind: str              # "length", "speed", "ratio" or "percent"
    meaning: str           # one or two plain sentences: what it is, how it is made

    def axis_label(self, unit: str) -> str:
        return {"length": unit, "speed": f"{unit}/s",
                "ratio": "0 to 1", "percent": "% of time"}[self.kind]


METRICS: List[Metric] = [
    Metric("Distance", "total_distance", "Total distance", "length",
           "How far the fish swam: every frame-to-frame step added up. Frames "
           "where the fish was not tracked add nothing."),
    Metric("MedianSpeed", "median_speed", "Median speed", "speed",
           "The median of its frame-by-frame speed: half the time it was "
           "slower than this, half the time faster."),
    Metric("Speed99", "speed_p99", "99th percentile speed", "speed",
           "The 99th percentile of its speed: it was faster than this for only "
           "1% of the time. A percentile, not the maximum, so a few tracking "
           "errors cannot set it."),
    Metric("Straightness", "mean_path_straightness", "Path straightness", "ratio",
           "For each second of swimming: the straight-line distance from where "
           "the fish started that second to where it ended, divided by the "
           "distance it actually swam. 1 = it went straight; lower = it turned "
           "or doubled back. Averaged over every second of the recording."),
    Metric("NearWall", None, "Time near the wall", "percent",
           "The share of its time spent in the zone along the walls, which is "
           "15% of the tank's shorter side wide. The dashed line is what even "
           "use of the tank would give: the zone's share of the tank's area."),
]

#: Columns that describe a row without being a metric to plot.
WALL_ZONE_SHARE = "WallZoneArea_pct"
NO_OUTLINE = "No tank outline.\nDraw one with\n\"Tank outline...\" on\nSessions & Units."


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
                if metric.source is not None:
                    row[metric.column] = fish.metrics.get(metric.source, np.nan)
            wall = getattr(loaded, "thigmotaxis_results", None)
            row["NearWall"] = (wall.time_in_border_pct[fish.fish_id]
                               if wall is not None else np.nan)
            row[WALL_ZONE_SHARE] = (round(float(wall.border_area_pct), 1)
                                    if wall is not None else np.nan)
            row["Unit"] = loaded.calibration.unit_name
            row["PixelsPerUnit"] = round(loaded.calibration.pixels_per_unit, 4)
            rows.append(row)
    columns = (["Group", "Session", "Fish", "Tracked_pct"]
               + [m.column for m in METRICS]
               + [WALL_ZONE_SHARE, "Unit", "PixelsPerUnit"])
    return pd.DataFrame(rows, columns=columns)


def session_means(table: pd.DataFrame, columns: Optional[List[str]] = None
                  ) -> pd.DataFrame:
    """One row per session: the mean of its fish. The experimental unit."""
    columns = columns or [m.column for m in METRICS]
    return (table.groupby(["Group", "Session"], sort=False)[columns]
            .mean().reset_index())


def draw_superplot(ax, table: pd.DataFrame, column: str, title: str, ylabel: str,
                   session_colors: Dict[str, tuple],
                   empty_message: str = "No data.") -> bool:
    """Draw one column of a per-fish table as a SuperPlot: fish as small dots,
    sessions as large markers, and a line per group at the mean of its
    session means. Returns False if there was nothing to draw.

    `table` needs Group and Session columns and one row per fish.
    """
    ax.set_title(title, fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if table[column].isna().all():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, empty_message, ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        return False

    groups = list(dict.fromkeys(table["Group"]))
    means = session_means(table, [column])
    for position, group in enumerate(groups):
        in_group = table[table["Group"] == group]
        sessions = list(dict.fromkeys(in_group["Session"]))
        # Sessions sit side by side inside the group's slot; fish are spread
        # evenly inside their session's share, so the layout never changes
        # between redraws the way random jitter would.
        slot = 0.7 / max(1, len(sessions))
        for index, session in enumerate(sessions):
            centre = position - 0.35 + slot * (index + 0.5)
            values = in_group.loc[in_group["Session"] == session, column].to_numpy(float)
            values = values[~np.isnan(values)]
            if len(values) == 0:
                continue
            spread = np.linspace(-0.35, 0.35, len(values)) * slot if len(values) > 1 else [0.0]
            color = session_colors[session]
            ax.scatter(centre + np.asarray(spread), values, s=22, color=color,
                       alpha=0.55, linewidths=0, zorder=2)
            ax.scatter([centre], [values.mean()], s=130, color=color,
                       edgecolors="black", linewidths=1.2, zorder=3)

        group_means = means.loc[means["Group"] == group, column].dropna()
        if len(group_means):
            ax.hlines(group_means.mean(), position - 0.4, position + 0.4,
                      colors="black", linewidths=2, zorder=4)

    ax.set_xticks(range(len(groups)))
    crowded = len(groups) > 3 or max(len(str(g)) for g in groups) > 10
    ax.set_xticklabels(groups, rotation=25 if crowded else 0,
                       ha="right" if crowded else "center")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    return True


def draw_reference_line(ax, value: float, text: str) -> None:
    """A dashed level to read a panel against, labelled at its right end."""
    ax.axhline(value, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.text(ax.get_xlim()[1], value, f" {text} ", fontsize=8, color="gray",
            va="bottom", ha="right")


def shared_unit(table: pd.DataFrame) -> str:
    units = list(dict.fromkeys(table["Unit"]))
    return units[0] if len(units) == 1 else "mixed units"


def superplot(ax, table: pd.DataFrame, metric: Metric,
              session_colors: Dict[str, tuple]) -> None:
    """One of the headline metrics as a SuperPlot."""
    drawn = draw_superplot(
        ax, table, metric.column, metric.title,
        metric.axis_label(shared_unit(table)), session_colors,
        # Only the wall measure can be missing wholesale: no arena was found.
        empty_message=NO_OUTLINE)
    if not drawn:
        return
    if metric.kind == "ratio":
        ax.set_ylim(0, 1)
    elif metric.kind == "percent":
        ax.set_ylim(0, 100)
        even_use = table[WALL_ZONE_SHARE].dropna()
        if len(even_use):
            draw_reference_line(ax, even_use.mean(), "even use")


# =============================================================================
# SPEED DISTRIBUTIONS
# =============================================================================
#
# A summary number can hide what a fish actually did: a fish that sits still
# for four minutes and then swims normally has an ordinary-looking mean. The
# whole distribution of its speed shows it at once, as a second peak at zero,
# and needs no threshold to do so.

@dataclass
class SpeedSample:
    group: str
    session: str
    fish: str
    speeds: np.ndarray      # one value per tracked frame, in unit/s
    unit: str


def speed_samples(loaded_files: Dict, file_groups: Optional[Dict[str, str]] = None
                  ) -> List[SpeedSample]:
    """Every analysed fish's frame-by-frame speeds, gaps removed."""
    file_groups = file_groups or {}
    samples = []
    for name, loaded in loaded_files.items():
        if not loaded.processed_data:
            continue
        group = file_groups.get(name) or default_group(name)
        for fish in loaded.processed_data:
            series = fish.metrics.get("speed_time_series")
            if series is None:
                continue
            speeds = np.asarray(series["speed"], dtype=float)
            speeds = speeds[~np.isnan(speeds)]
            if len(speeds):
                samples.append(SpeedSample(group, name, str(fish.identity_label),
                                           speeds, loaded.calibration.unit_name))
    return samples


def _speed_bins(samples: List[SpeedSample], n_bins: int = 70) -> np.ndarray:
    """Shared bins from zero to just past where almost all speeds lie, so the
    rare fastest frames do not squash everything else against the left."""
    upper = max(np.percentile(sample.speeds, 99.5) for sample in samples)
    return np.linspace(0.0, float(upper), n_bins + 1)


def _density(speeds: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(speeds, bins=bins)
    density = counts / max(1, len(speeds)) / np.diff(bins)
    # A light running mean: enough to read as a curve, not enough to move a peak.
    return np.convolve(density, np.ones(3) / 3.0, mode="same")


def plot_session_speed_ecdf(ax, samples: List[SpeedSample],
                            session_colors: Dict[str, tuple]) -> None:
    """One cumulative curve per session, all its fish pooled: the comparison.

    A cumulative curve needs no bins and no smoothing, so nothing about it is
    a choice, and the two speed metrics can be read straight off it: where a
    curve crosses 50% is that session's median speed, where it crosses 99%
    its 99th percentile. A session that is faster overall sits to the right.
    """
    bins = _speed_bins(samples)
    levels = np.linspace(0.0, 1.0, 501)
    sessions = list(dict.fromkeys(sample.session for sample in samples))
    for session in sessions:
        in_session = [sample for sample in samples if sample.session == session]
        pooled = np.concatenate([sample.speeds for sample in in_session])
        ax.plot(np.quantile(pooled, levels), levels * 100,
                color=session_colors[session], linewidth=2,
                label=f"{session}  ({in_session[0].group})")
    for level, text in ((50, "median"), (99, "99th percentile")):
        ax.axhline(level, color="gray", linestyle=":", linewidth=1)
        ax.text(bins[-1], level - 1.5, f"{text} ", fontsize=8, color="gray",
                ha="right", va="top")
    ax.set_title("Each session, all fish together", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Speed ({samples[0].unit}/s)")
    ax.set_ylabel("% of time at or below this speed")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 100.5)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_fish_speed_ridges(ax, samples: List[SpeedSample],
                           session_colors: Dict[str, tuple]) -> None:
    """One ridge per fish, stacked and coloured by session: where a single
    unusual animal shows up."""
    bins = _speed_bins(samples)
    centres = (bins[:-1] + bins[1:]) / 2
    densities = [_density(sample.speeds, bins) for sample in samples]
    overlap = 1.8                       # ridge height, in rows
    labels = []
    for row, (sample, density) in enumerate(zip(samples, densities)):
        base = len(samples) - 1 - row   # first fish at the top
        # Each ridge is scaled to its own peak. The shape is what is being
        # read here; one fish with a tall spike would otherwise flatten the rest.
        height = density / density.max() * overlap
        color = session_colors[sample.session]
        ax.fill_between(centres, base, base + height, color=color, alpha=0.55,
                        zorder=row + 1)
        ax.plot(centres, base + height, color="white", linewidth=0.8, zorder=row + 1)
        labels.append((base, f"{sample.session} \u00b7 {sample.fish}"))
    ax.set_yticks([base for base, _ in labels])
    ax.set_yticklabels([text for _, text in labels], fontsize=8)
    ax.set_title("Each fish", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Speed ({samples[0].unit}/s)")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(-0.2, len(samples) - 1 + overlap + 0.2)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


# =============================================================================
# WHERE THEY SWIM

# =============================================================================
# WHERE THEY SWIM
# =============================================================================
#
# Where the time went. A fish that hangs in one corner, or a tank that keeps
# off the walls, shows here at a glance.

def panel_grid(n_sessions: int) -> tuple:
    """(rows, columns) for one panel per session in a wide, short figure."""
    columns = min(n_sessions, int(np.ceil(np.sqrt(3 * n_sessions))))
    return int(np.ceil(n_sessions / columns)), columns


def density_cell(loaded_files: Dict, sessions: List[str], across: int = 48) -> float:
    """One cell size for every session, so a cell means the same everywhere:
    the widest frame divided into `across` squares."""
    return max(
        max(loaded.metadata.video_width, loaded.metadata.video_height)
        * loaded.calibration.scale_factor
        for loaded in (loaded_files[name] for name in sessions)) / across


def position_density(loaded, cell: float) -> tuple:
    """The share of the session's tracked positions, all fish together, that
    fell in each square cell: (percentages, x_edges, y_edges), drawn the way
    the video shows it."""
    scale = loaded.calibration.scale_factor
    width = loaded.metadata.video_width * scale
    height = loaded.metadata.video_height * scale
    x_edges = np.arange(0.0, width + cell, cell)
    y_edges = np.arange(0.0, height + cell, cell)
    positions = np.asarray(loaded.trajectories, dtype=float).reshape(-1, 2)
    positions = positions[~np.isnan(positions).any(axis=1)]
    counts, _, _ = np.histogram2d(
        positions[:, 0] * scale,
        (loaded.metadata.video_height - positions[:, 1]) * scale,
        bins=[x_edges, y_edges])
    return counts.T / max(1, len(positions)) * 100, x_edges, y_edges


def shared_density_ceiling(densities: List[np.ndarray]) -> float:
    """The top of one colour scale for every session: the 99th percentile of
    the occupied cells, so one crowded cell does not darken everything else."""
    occupied = np.concatenate([d[d > 0] for d in densities])
    return float(np.percentile(occupied, 99)) if len(occupied) else 1.0


def plot_position_density(ax, name: str, loaded, density: np.ndarray,
                          x_edges: np.ndarray, y_edges: np.ndarray,
                          ceiling: float, arena=None):
    """One session's density map. Returns the image, for a shared colour bar."""
    scale = loaded.calibration.scale_factor
    image = ax.imshow(
        density, origin="lower", cmap="magma", vmin=0, vmax=ceiling,
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]))
    if arena is not None:
        outline = np.vstack([arena.vertices_bl, arena.vertices_bl[:1]])
        ax.plot(outline[:, 0], outline[:, 1], color="white", linewidth=1.2)
    ax.set_xlim(0, loaded.metadata.video_width * scale)
    ax.set_ylim(0, loaded.metadata.video_height * scale)
    ax.set_aspect("equal")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel(loaded.calibration.unit_name)
    ax.tick_params(labelsize=8)
    return image



# =============================================================================
# MINUTE BY MINUTE
# =============================================================================
#
# One number for a whole recording hides when things happened: a fish that
# sat still for three minutes, a tank that slowed down as it settled. Here
# every measure is worked out again on each minute by itself. Nothing is
# smoothed, so a point is exactly what happened in that minute and no more.

BIN_SECONDS = 60.0


def bin_count(duration_s: float) -> int:
    """Whole minutes, and a last short one only if it is at least half a minute."""
    return max(1, int(duration_s / BIN_SECONDS + 0.5))


def minute_table(loaded_files: Dict, file_groups: Optional[Dict[str, str]] = None
                 ) -> pd.DataFrame:
    """One row per analysed fish per minute, with the headline metrics for
    that minute alone. Distance is what was swum in the minute; a short last
    minute is scaled up to a full one."""
    from .processing import straightness_windows

    file_groups = file_groups or {}
    rows = []
    for name, loaded in loaded_files.items():
        if not loaded.processed_data:
            continue
        group = file_groups.get(name) or default_group(name)
        fps = loaded.calibration.frame_rate
        duration = loaded.n_frames / fps
        n_bins = bin_count(duration)
        wall = getattr(loaded, "thigmotaxis_results", None)
        wall_bin = (None if wall is None else
                    np.minimum((np.asarray(wall.timestamps) // BIN_SECONDS).astype(int),
                               n_bins - 1))
        window = max(2, int(fps))
        for fish in loaded.processed_data:
            series = fish.metrics.get("speed_time_series")
            if series is None:
                continue
            speed = np.asarray(series["speed"], dtype=float)
            speed_bin = np.minimum(
                (np.asarray(series["time"], dtype=float) // BIN_SECONDS).astype(int),
                n_bins - 1)
            starts, straight = straightness_windows(
                fish.trajectory["x"].to_numpy(), fish.trajectory["y"].to_numpy(), window)
            straight_bin = np.minimum((starts / fps // BIN_SECONDS).astype(int), n_bins - 1)
            for index in range(n_bins):
                length = min(BIN_SECONDS * (index + 1), duration) - BIN_SECONDS * index
                tracked = speed[speed_bin == index]
                tracked = tracked[~np.isnan(tracked)]
                in_bin = straight[straight_bin == index]
                near = (np.nan if wall is None else
                        _nanmean(wall.per_fish_in_border_samples[wall_bin == index,
                                                                 fish.fish_id]) * 100)
                rows.append({
                    "Group": group, "Session": name, "Fish": fish.identity_label,
                    "Minute": index + 1,
                    "Distance": tracked.sum() / fps * (BIN_SECONDS / length),
                    "MedianSpeed": np.median(tracked) if len(tracked) else np.nan,
                    "Speed99": np.percentile(tracked, 99) if len(tracked) else np.nan,
                    "Straightness": in_bin.mean() if len(in_bin) else np.nan,
                    "NearWall": near,
                    "Unit": loaded.calibration.unit_name,
                })
    return pd.DataFrame(rows, columns=["Group", "Session", "Fish", "Minute"]
                        + [m.column for m in METRICS] + ["Unit"])


def _nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else np.nan


def draw_minute_lines(ax, table: pd.DataFrame, column: str, title: str,
                      ylabel: str, session_colors: Dict[str, tuple],
                      empty_message: str = "No data.") -> bool:
    """One column of a per-fish, per-minute table: a thin line per fish and a
    thick one for each session's mean. Returns False if there was nothing to
    draw. `table` needs Session, Fish and Minute columns."""
    ax.set_title(title, fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if table[column].isna().all():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, empty_message, ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        return False
    for session in dict.fromkeys(table["Session"]):
        fish = (table[table["Session"] == session]
                .pivot(index="Minute", columns="Fish", values=column))
        color = session_colors[session]
        ax.plot(fish.index, fish.to_numpy(), color=color, linewidth=0.7, alpha=0.35)
        ax.plot(fish.index, fish.mean(axis=1), color=color, linewidth=2.4,
                marker="o", markersize=4)
    minutes = int(table["Minute"].max())
    # About five labels, whatever the length: every minute crowds a narrow panel.
    step = max(1, int(np.ceil(minutes / 6)))
    ax.set_xticks(range(step, minutes + 1, step))
    ax.set_xlim(0.5, minutes + 0.5)
    ax.set_xlabel("Minute of the recording")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    return True


def minute_plot(ax, table: pd.DataFrame, metric: Metric,
                session_colors: Dict[str, tuple]) -> None:
    """One of the headline metrics, minute by minute."""
    title = "Distance in each minute" if metric.column == "Distance" else metric.title
    drawn = draw_minute_lines(
        ax, table, metric.column, title, metric.axis_label(shared_unit(table)),
        session_colors, empty_message=NO_OUTLINE)
    if not drawn:
        return
    if metric.kind == "ratio":
        ax.set_ylim(0, 1)
    elif metric.kind == "percent":
        ax.set_ylim(0, 100)


MINUTE_MEANING = (
    "\"Minute by minute\" works each measure out again on every minute of the "
    "recording by itself. Thin lines are fish, the thick line is their "
    "session's mean. Nothing is smoothed: a point is what happened in that "
    "minute. Look here before trusting a single number for the whole "
    "recording; one fish that stops for a few minutes moves its tank's mean.")
