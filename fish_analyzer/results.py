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

SESSIONS TRACKED WITHOUT IDENTITIES have no fish to give a row to: each
column is a track that passes from fish to fish (see file_loading.py). Such a
session gets one row, Fish = POOLED, made from every track together: speeds,
straightness windows and wall time pooled over all tracked moments, distance
summed over all tracks and divided by the fish in the tank. These are tank
values, which is what the large markers are anyway.
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
        return {"length": unit, "speed": f"{unit}/s", "rate": f"{unit} per tracked min",
                "ratio": "0 to 1", "percent": "% of time"}[self.kind]


METRICS: List[Metric] = [
    Metric("Distance", "total_distance", "Total distance", "length",
           "How far the fish swam: every frame-to-frame step added up. Frames "
           "where the fish was not tracked add nothing, so a fish tracked less "
           "of the time reads lower; the exported DistancePerTrackedMin "
           "corrects for that. Compare Tracked % between groups."),
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
           "or doubled back. Averaged over the seconds in which the fish was "
           "swimming (faster than the speed set on Sessions & Units, 1 BL/s by "
           "default): in a slow or still second the tracker's small wobble is "
           "most of the path, and the ratio would measure that instead."),
    Metric("NearWall", None, "Time near the wall", "percent",
           "The share of its time spent in the zone along the walls, which is "
           "15% of the tank's shorter side wide. The dashed line is what even "
           "use of the tank would give: the zone's share of the tank's area."),
    Metric("RestingTime", "freeze_fraction_pct", "Time resting", "percent",
           "The share of the time it was tracked that the fish spent resting: "
           "slower than the resting speed for at least the shortest rest, both "
           "set on Sessions & Units and shown in the panel's title. A gap in "
           "tracking of up to half a second does not end a rest if the fish is "
           "in the same place on both sides. Check the setting on the video "
           "first: Video Inspector rings every fish the analysis counted as "
           "resting."),
]


def resting_title(params, unit: str) -> str:
    """"Time resting" with the rule that made it, so the number never
    travels without its definition."""
    if params is None:
        return "Time resting"
    return (f"Time resting\n(<{params.rest_speed_threshold:g} {unit}/s for "
            f"\u2265{params.min_freeze_seconds:g} s)")

JUMPS_MEANING = (
    "How many times the track moved faster than a fish can swim from one "
    "frame to the next (25 BL/s by default, set on Sessions & Units). That is "
    "the tracker, not the fish: an identity swap, or a detection of something "
    "that is not a fish. Those steps are left out of speed and distance. A "
    "fish with many jumps may also have swapped identity with another fish, "
    "so its own numbers partly belong to that fish; check it in Video "
    "Inspector.")

#: Shown in place of Distance whenever a session tracked without identities
#: is among the results. Their tracks miss more of the time than fish tracked
#: with identities (70-87% tracked against 92-99% on the first sessions
#: compared), and total distance falls with every untracked moment; per
#: tracked minute it does not. It is the mean speed, per minute.
DISTANCE_PER_TRACKED_MIN = Metric(
    "DistancePerTrackedMin", "distance_per_tracked_min",
    "Distance per tracked minute", "rate",
    "How far the fish swam in each minute it was tracked: total distance "
    "divided by the time it was seen. Shown instead of total distance when "
    "a session tracked without identities is loaded, because those miss more "
    "of the time and total distance would read lower for that reason alone. "
    "In \"Minute by minute\" (titled \"Distance per minute\") each point is "
    "the distance swum in that minute divided by the part of the minute the "
    "fish were tracked: tracked for 45 of the 60 seconds and swimming 60 BL "
    "in them reads 80 BL, what a whole minute at that pace would give.")


def metrics_for(loaded_files: Dict) -> List[Metric]:
    """The metrics to show: METRICS, with distance per tracked minute in
    place of total distance if any analysed session was tracked without
    identities, so every session is measured the same way."""
    if any(pooled(loaded) and loaded.processed_data
           for loaded in loaded_files.values()):
        return [DISTANCE_PER_TRACKED_MIN if m.column == "Distance" else m
                for m in METRICS]
    return list(METRICS)


#: Columns that describe a row without being a metric to plot.
WALL_ZONE_SHARE = "WallZoneArea_pct"
NO_OUTLINE = "No tank outline.\nDraw one with\n\"Tank outline...\" on\nSessions & Units."


def default_group(session_name: str) -> str:
    """A group for a session nobody has grouped: its name without a trailing
    number, so control_1 and control_2 land together."""
    stripped = re.sub(r"[_\-\s]*\d+$", "", session_name).strip()
    return stripped or session_name


#: The Fish label of the one row a session tracked without identities gets.
POOLED = "all fish (no identities)"


def pooled(loaded) -> bool:
    """Whether a session's results are one row for the tank (it was tracked
    without identities) rather than a row per fish."""
    return not getattr(loaded, "with_identities", True)


def _processing_params(loaded):
    params = getattr(loaded, "processing_params", None)
    if params is None:
        from .processing import ProcessingParameters
        params = ProcessingParameters()
    return params


def _track_speeds(fish) -> tuple:
    """(time, speed) of one analysed fish or track, or None."""
    series = fish.metrics.get("speed_time_series")
    if series is None:
        return None
    return (np.asarray(series["time"], dtype=float),
            np.asarray(series["speed"], dtype=float))


def _track_straightness(loaded, fish) -> tuple:
    """(first frame of each kept window, its straightness) for one track."""
    from .processing import straightness_windows
    params = _processing_params(loaded)
    fps = loaded.calibration.frame_rate
    starts, values, _ = straightness_windows(
        fish.trajectory["x"].to_numpy(), fish.trajectory["y"].to_numpy(),
        params.straightness_window_frames(fps), fps, params.straightness_min_speed)
    return starts, values


def _pooled_row(loaded) -> Dict:
    """The headline metrics for a session tracked without identities, every
    track together. See the module docstring."""
    tracks = loaded.processed_data
    fish_in_tank = loaded.fish_in_tank
    speeds = [s[~np.isnan(s)] for s in
              (pair[1] for pair in map(_track_speeds, tracks) if pair is not None)]
    speeds = np.concatenate(speeds) if speeds else np.array([])
    straight = [_track_straightness(loaded, fish)[1] for fish in tracks]
    straight = np.concatenate(straight) if straight else np.array([])
    distance = float(sum(fish.metrics.get("total_distance", 0.0) for fish in tracks))
    observed_min = sum(fish.metrics.get("observed_duration_s", 0.0)
                       for fish in tracks) / 60.0
    tracked = sum(fish.n_valid_frames for fish in tracks)
    wall = getattr(loaded, "thigmotaxis_results", None)
    near = np.nan
    if wall is not None and wall.in_arena_frames is not None:
        indices = [fish.fish_id for fish in tracks]
        weights = np.asarray(wall.in_arena_frames, dtype=float)[indices]
        pct = np.asarray(wall.time_in_border_pct, dtype=float)[indices]
        usable = weights > 0
        if usable.any():
            near = float(np.sum(pct[usable] * weights[usable]) / weights[usable].sum())
    return {
        "Fish": POOLED,
        "Tracked_pct": round(tracked / (fish_in_tank * loaded.n_frames) * 100, 1),
        "Jumps": int(sum(fish.metrics.get("tracking_jumps", 0) for fish in tracks)),
        "Distance": distance / fish_in_tank,
        "MedianSpeed": float(np.median(speeds)) if len(speeds) else np.nan,
        "Speed99": float(np.percentile(speeds, 99)) if len(speeds) else np.nan,
        "Straightness": float(straight.mean()) if len(straight) else np.nan,
        "NearWall": near,
        "RestingTime": (sum(fish.metrics.get("freeze_total_duration_s", 0.0)
                            for fish in tracks) / (observed_min * 60) * 100
                        if observed_min else np.nan),
        "DistancePerTrackedMin": distance / observed_min if observed_min else np.nan,
    }


def results_table(loaded_files: Dict, file_groups: Optional[Dict[str, str]] = None
                  ) -> pd.DataFrame:
    """One row per analysed fish, or for a session tracked without
    identities one row for the tank. Sessions without results are skipped."""
    file_groups = file_groups or {}
    rows = []
    for name, loaded in loaded_files.items():
        if not loaded.processed_data:
            continue
        group = file_groups.get(name) or default_group(name)
        if pooled(loaded):
            wall = getattr(loaded, "thigmotaxis_results", None)
            rows.append({
                "Group": group, "Session": name, **_pooled_row(loaded),
                WALL_ZONE_SHARE: (round(float(wall.border_area_pct), 1)
                                  if wall is not None else np.nan),
                "Unit": loaded.calibration.unit_name,
                "PixelsPerUnit": round(loaded.calibration.pixels_per_unit, 4),
            })
            continue
        for fish in loaded.processed_data:
            row = {
                "Group": group,
                "Session": name,
                "Fish": fish.identity_label,
                "Tracked_pct": round(fish.valid_percentage * 100, 1),
                "Jumps": int(fish.metrics.get("tracking_jumps", 0)),
            }
            for metric in METRICS:
                if metric.source is not None:
                    row[metric.column] = fish.metrics.get(metric.source, np.nan)
            wall = getattr(loaded, "thigmotaxis_results", None)
            row["NearWall"] = (wall.time_in_border_pct[fish.fish_id]
                               if wall is not None else np.nan)
            row[WALL_ZONE_SHARE] = (round(float(wall.border_area_pct), 1)
                                    if wall is not None else np.nan)
            row["DistancePerTrackedMin"] = fish.metrics.get(
                "distance_per_tracked_min", np.nan)
            row["Unit"] = loaded.calibration.unit_name
            row["PixelsPerUnit"] = round(loaded.calibration.pixels_per_unit, 4)
            rows.append(row)
    columns = (["Group", "Session", "Fish", "Tracked_pct", "Jumps"]
               + [m.column for m in METRICS]
               + [WALL_ZONE_SHARE, "DistancePerTrackedMin", "Unit", "PixelsPerUnit"])
    return pd.DataFrame(rows, columns=columns)


def excluded_fish(loaded_files: Dict) -> List[str]:
    """'session: fish N reason' for every fish left out of the results."""
    out = []
    for name, loaded in loaded_files.items():
        if loaded.processed_data is None:
            continue
        for index, reason in sorted(getattr(loaded, "excluded_fish", {}).items()):
            label = loaded.metadata.identity_labels[index]
            out.append(f"{name}: fish {label} {reason}")
    return out


def results_export_table(loaded_files: Dict,
                         file_groups: Optional[Dict[str, str]] = None
                         ) -> pd.DataFrame:
    """The results table as written to CSV: every analysed fish, then a row
    for each fish left out and why, then the settings the numbers came from."""
    file_groups = file_groups or {}
    table = results_table(loaded_files, file_groups)
    table.insert(3, "Status", "ok")
    extra = []
    for name, loaded in loaded_files.items():
        if loaded.processed_data is None:
            continue
        group = file_groups.get(name) or default_group(name)
        for index, reason in sorted(getattr(loaded, "excluded_fish", {}).items()):
            xy = np.asarray(loaded.trajectories[:, index, :], dtype=float)
            extra.append({
                "Group": group, "Session": name,
                "Fish": loaded.metadata.identity_labels[index],
                "Status": f"excluded: {reason}",
                "Tracked_pct": round(float(np.isfinite(xy).all(axis=1).mean() * 100), 1),
                "Unit": loaded.calibration.unit_name,
                "PixelsPerUnit": round(loaded.calibration.pixels_per_unit, 4),
            })
    if extra:
        table = pd.concat([table, pd.DataFrame(extra)], ignore_index=True)[table.columns]
    settings = {}
    for name, loaded in loaded_files.items():
        params = getattr(loaded, "processing_params", None)
        if params is not None:
            settings[name] = params.export_columns()
    for column in dict.fromkeys(k for v in settings.values() for k in v):
        table[column] = table["Session"].map(
            lambda session: settings.get(session, {}).get(column, np.nan))
    return table


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
              session_colors: Dict[str, tuple], title: Optional[str] = None) -> None:
    """One of the headline metrics as a SuperPlot."""
    drawn = draw_superplot(
        ax, table, metric.column, title or metric.title,
        metric.axis_label(shared_unit(table)), session_colors,
        # Only the wall measure can be missing wholesale: no arena was found.
        empty_message=NO_OUTLINE)
    if not drawn:
        return
    if metric.kind == "ratio":
        ax.set_ylim(0, 1)
    elif metric.column == "NearWall":
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
    """Every analysed fish's frame-by-frame speeds, gaps removed. A session
    tracked without identities gives one sample, all its tracks together."""
    file_groups = file_groups or {}
    samples = []
    for name, loaded in loaded_files.items():
        if not loaded.processed_data:
            continue
        group = file_groups.get(name) or default_group(name)
        if pooled(loaded):
            speeds = [pair[1] for pair in map(_track_speeds, loaded.processed_data)
                      if pair is not None]
            speeds = np.concatenate(speeds) if speeds else np.array([])
            speeds = speeds[~np.isnan(speeds)]
            if len(speeds):
                samples.append(SpeedSample(group, name, POOLED, speeds,
                                           loaded.calibration.unit_name))
            continue
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


def _speed_bins(samples: List[SpeedSample], n_bins: int = 70,
                top: Optional[float] = None) -> np.ndarray:
    """Shared bins from zero to `top`, or by default to just past where
    almost all speeds lie, so the rare fastest frames do not squash
    everything else against the left. Only what is drawn stops at the top:
    every frame still counts in each curve's shares."""
    upper = (top if top is not None and top > 0 else
             max(np.percentile(sample.speeds, 99.5) for sample in samples))
    return np.linspace(0.0, float(upper), n_bins + 1)


def _density(speeds: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(speeds, bins=bins)
    density = counts / max(1, len(speeds)) / np.diff(bins)
    # A light running mean: enough to read as a curve, not enough to move a peak.
    return np.convolve(density, np.ones(3) / 3.0, mode="same")


def plot_session_speed_ecdf(ax, samples: List[SpeedSample],
                            session_colors: Dict[str, tuple],
                            top: Optional[float] = None) -> None:
    """One cumulative curve per session, all its fish pooled: the comparison.

    A cumulative curve needs no bins and no smoothing, so nothing about it is
    a choice, and the two speed metrics can be read straight off it: where a
    curve crosses 50% is that session's median speed, where it crosses 99%
    its 99th percentile. A session that is faster overall sits to the right.
    """
    bins = _speed_bins(samples, top=top)
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
                           session_colors: Dict[str, tuple],
                           top: Optional[float] = None) -> None:
    """One ridge per fish, stacked and coloured by session: where a single
    unusual animal shows up."""
    bins = _speed_bins(samples, top=top)
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
        # The median, the headline speed: a short tick inside the ridge.
        median = float(np.median(sample.speeds))
        ax.plot([median, median], [base, base + overlap * 0.6], color="black",
                linewidth=1.2, zorder=row + 1)
        fish = "all fish" if sample.fish in (POOLED, ALL_FISH) else sample.fish
        labels.append((base, f"{sample.session} \u00b7 {fish}"))
    ax.set_yticks([base for base, _ in labels])
    ax.set_yticklabels([text for _, text in labels], fontsize=8)
    tanks = sum(sample.fish in (POOLED, ALL_FISH) for sample in samples)
    ax.set_title("Each tank" if tanks == len(samples) else
                 "Each fish" if tanks == 0 else
                 "Each fish (each tank where tracked without identities)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Speed ({samples[0].unit}/s); black tick: median")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(-0.2, len(samples) - 1 + overlap + 0.2)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


#: What one ridge stands for: (shown in the list, level).
RIDGE_LEVELS = (("Fish", "fish"), ("Tanks", "tanks"), ("Groups", "groups"))

#: The Fish label of a tank made by putting its fish together.
ALL_FISH = "all fish"


def tank_samples(samples: List[SpeedSample]) -> List[SpeedSample]:
    """One sample per session: every frame of all its fish together, which
    is what "Each session, all fish together" on the left draws."""
    out = []
    for session in dict.fromkeys(sample.session for sample in samples):
        own = [sample for sample in samples if sample.session == session]
        out.append(SpeedSample(own[0].group, session, ALL_FISH,
                               np.concatenate([s.speeds for s in own]), own[0].unit))
    return out


def _by_group(samples: List[SpeedSample]) -> Dict[str, List[SpeedSample]]:
    """The tanks of each group, in the order the groups first appear."""
    groups: Dict[str, List[SpeedSample]] = {}
    for tank in tank_samples(samples):
        groups.setdefault(tank.group, []).append(tank)
    return groups


def plot_group_speed_ecdf(ax, samples: List[SpeedSample],
                          group_colors: Dict[str, tuple],
                          top: Optional[float] = None) -> None:
    """One cumulative curve per group, the mean of its tanks' curves, with
    each tank's own curve thin behind it. Every tank counts the same, however
    long it was tracked: the tank is the unit."""
    bins = _speed_bins(samples, top=top)
    speeds = np.linspace(bins[0], bins[-1], 400)
    for group, tanks in _by_group(samples).items():
        color = group_colors[group]
        curves = np.array([[np.mean(tank.speeds <= s) * 100 for s in speeds]
                           for tank in tanks])
        for curve in curves:
            ax.plot(speeds, curve, color=color, linewidth=1.0, alpha=0.5)
        ax.plot(speeds, curves.mean(axis=0), color=color, linewidth=2.6,
                label=f"{group}  ({len(tanks)} tank{'s' if len(tanks) != 1 else ''})")
    for level, text in ((50, "median"), (99, "99th percentile")):
        ax.axhline(level, color="gray", linestyle=":", linewidth=1)
        ax.text(bins[-1], level - 1.5, f"{text} ", fontsize=8, color="gray",
                ha="right", va="top")
    ax.set_title("Each group (thin lines: its tanks)", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Speed ({samples[0].unit}/s)")
    ax.set_ylabel("% of time at or below this speed")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 100.5)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_group_speed_ridges(ax, samples: List[SpeedSample],
                            group_colors: Dict[str, tuple],
                            top: Optional[float] = None) -> None:
    """One ridge per group: the mean of its tanks' distributions, filled,
    with each tank's own outlined on it, so a tank that does not fit shows.
    The tick is the median of the tank medians."""
    bins = _speed_bins(samples, top=top)
    centres = (bins[:-1] + bins[1:]) / 2
    groups = _by_group(samples)
    overlap = 1.8
    labels = []
    for row, (group, tanks) in enumerate(groups.items()):
        base = len(groups) - 1 - row
        densities = np.array([_density(tank.speeds, bins) for tank in tanks])
        mean = densities.mean(axis=0)
        # One scale for the group and its tanks, so the outlines compare
        # with the fill.
        scale = overlap / max(mean.max(), densities.max())
        color = group_colors[group]
        ax.fill_between(centres, base, base + mean * scale, color=color,
                        alpha=0.45, zorder=row + 1)
        for density in densities:
            ax.plot(centres, base + density * scale, color=color, linewidth=0.9,
                    zorder=row + 1)
        median = float(np.median([np.median(tank.speeds) for tank in tanks]))
        ax.plot([median, median], [base, base + overlap * 0.6], color="black",
                linewidth=1.2, zorder=row + 1)
        labels.append((base, f"{group} ({len(tanks)} tank{'s' if len(tanks) != 1 else ''})"))
    ax.set_yticks([base for base, _ in labels])
    ax.set_yticklabels([text for _, text in labels], fontsize=8)
    ax.set_title("Each group (lines: its tanks)", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Speed ({samples[0].unit}/s); black tick: median of the tanks' medians")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(-0.2, len(groups) - 1 + overlap + 0.2)
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
# every measure is worked out again on each minute by itself. By default
# nothing is smoothed, so a point is exactly what happened in that minute and
# no more. Smoothing is the reader's choice, never ours: it is off until
# chosen, and a plot that has it says so on its face.

BIN_SECONDS = 60.0

#: What the "Smoothing" list offers: (shown in the list, minutes averaged).
SMOOTHING_CHOICES = (("None", 1), ("3 minutes", 3), ("5 minutes", 5))


def smoothing_note(minutes: int) -> str:
    """What to print on a smoothed plot, or "" for an unsmoothed one."""
    if minutes <= 1:
        return ""
    return (f"Smoothed: each point is the mean of the {minutes} minutes centred "
            "on it (fewer at the two ends).")


def bin_count(duration_s: float) -> int:
    """Whole minutes, and a last short one only if it is at least half a minute."""
    return max(1, int(duration_s / BIN_SECONDS + 0.5))


def minute_table(loaded_files: Dict, file_groups: Optional[Dict[str, str]] = None
                 ) -> pd.DataFrame:
    """One row per analysed fish per minute, with the headline metrics for
    that minute alone. Distance is what was swum in the minute; a short last
    minute is scaled up to a full one."""
    from .processing import resting_frames, straightness_windows

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
        params = getattr(loaded, "processing_params", None)
        if params is None:
            from .processing import ProcessingParameters
            params = ProcessingParameters()
        window = params.straightness_window_frames(fps)
        # Without identities every track goes into the one tank row, and the
        # distance swum by all of them is shared out over the fish in the tank.
        together = pooled(loaded)
        per_fish = loaded.fish_in_tank if together else 1
        # label -> minute -> (speeds, straightness values, wall samples,
        # whether each tracked speed was resting)
        collected: Dict[str, List[tuple]] = {}
        for fish in loaded.processed_data:
            series = fish.metrics.get("speed_time_series")
            if series is None:
                continue
            speed = np.asarray(series["speed"], dtype=float)
            speed_bin = np.minimum(
                (np.asarray(series["time"], dtype=float) // BIN_SECONDS).astype(int),
                n_bins - 1)
            starts, straight, _ = straightness_windows(
                fish.trajectory["x"].to_numpy(), fish.trajectory["y"].to_numpy(),
                window, fps, params.straightness_min_speed)
            straight_bin = np.minimum((starts / fps // BIN_SECONDS).astype(int), n_bins - 1)
            resting = resting_frames(fish, params)[:len(speed)]
            label = POOLED if together else fish.identity_label
            minutes = collected.setdefault(label, [([], [], [], []) for _ in range(n_bins)])
            for index in range(n_bins):
                tracked = speed[speed_bin == index]
                minutes[index][0].append(tracked[~np.isnan(tracked)])
                minutes[index][3].append(resting[(speed_bin == index) & ~np.isnan(speed)])
                minutes[index][1].append(straight[straight_bin == index])
                if wall is not None:
                    minutes[index][2].append(
                        wall.per_fish_in_border_samples[wall_bin == index, fish.fish_id])
        for label, minutes in collected.items():
            for index, (speeds, straights, nears, rests) in enumerate(minutes):
                length = min(BIN_SECONDS * (index + 1), duration) - BIN_SECONDS * index
                tracked = np.concatenate(speeds)
                in_bin = np.concatenate(straights)
                rest = np.concatenate(rests)
                near = (np.nan if wall is None else
                        _nanmean(np.concatenate(nears)) * 100)
                rows.append({
                    "Group": group, "Session": name, "Fish": label,
                    "Minute": index + 1,
                    "Distance": (tracked.sum() / fps * (BIN_SECONDS / length)
                                 / per_fish),
                    "DistancePerTrackedMin": (tracked.mean() * 60
                                              if len(tracked) else np.nan),
                    "MedianSpeed": np.median(tracked) if len(tracked) else np.nan,
                    "Speed99": np.percentile(tracked, 99) if len(tracked) else np.nan,
                    "Straightness": in_bin.mean() if len(in_bin) else np.nan,
                    "NearWall": near,
                    "RestingTime": rest.mean() * 100 if len(rest) else np.nan,
                    "Unit": loaded.calibration.unit_name,
                })
    return pd.DataFrame(rows, columns=["Group", "Session", "Fish", "Minute"]
                        + [m.column for m in METRICS]
                        + ["DistancePerTrackedMin", "Unit"])


def _nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else np.nan


#: A group needs this many sessions before its minute-by-minute line gets a
#: confidence band. From two sessions a 95% interval is the mean plus or minus
#: 12.7 standard errors: it fills the panel and says less than the two
#: session lines it is drawn from.
BAND_MIN_SESSIONS = 3


def draw_minute_lines(ax, table: pd.DataFrame, column: str, title: str,
                      ylabel: str, group_colors: Dict[str, tuple],
                      empty_message: str = "No data.",
                      smooth_minutes: int = 1, show_fish: bool = False) -> bool:
    """One column of a per-fish, per-minute table, by group: a thin line for
    each session (the mean of its fish), a thick line for each group (the
    mean of its sessions), and, for a group with BAND_MIN_SESSIONS or more,
    a band for the 95% confidence interval of that mean. The session lines
    are the spread the group line is the mean of; the tank is the unit.
    `show_fish` adds a faint line per fish. Returns False if there was
    nothing to draw. `table` needs Group, Session, Fish and Minute columns.

    With `smooth_minutes` above 1 every fish's line is a running mean over
    that many minutes, centred, using what there is at the two ends, and the
    session and group lines are made from those."""
    from scipy import stats

    ax.set_title(title, fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if table[column].isna().all():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, empty_message, ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        return False
    for group in dict.fromkeys(table["Group"]):
        in_group = table[table["Group"] == group]
        color = group_colors[group]
        sessions = {}
        for session in dict.fromkeys(in_group["Session"]):
            fish = (in_group[in_group["Session"] == session]
                    .pivot(index="Minute", columns="Fish", values=column))
            if smooth_minutes > 1:
                fish = fish.rolling(smooth_minutes, center=True, min_periods=1).mean()
            if show_fish:
                ax.plot(fish.index, fish.to_numpy(), color=color, linewidth=0.5,
                        alpha=0.2)
            sessions[session] = fish.mean(axis=1)
            ax.plot(fish.index, sessions[session], color=color, linewidth=1.0,
                    alpha=0.5)
        sessions = pd.DataFrame(sessions)
        mean = sessions.mean(axis=1)
        n = sessions.notna().sum(axis=1)
        if len(sessions.columns) >= BAND_MIN_SESSIONS:
            usable = n >= 2
            half = (stats.t.ppf(0.975, n[usable] - 1)
                    * sessions[usable].std(axis=1) / np.sqrt(n[usable]))
            ax.fill_between(mean[usable].index, mean[usable] - half,
                            mean[usable] + half, color=color, alpha=0.18,
                            linewidth=0)
        ax.plot(mean.index, mean, color=color, linewidth=2.6, marker="o",
                markersize=4)
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
                group_colors: Dict[str, tuple], smooth_minutes: int = 1,
                show_fish: bool = False, title: Optional[str] = None) -> None:
    """One of the headline metrics, minute by minute, by group."""
    title = title or {"Distance": "Distance in each minute",
                      "DistancePerTrackedMin": "Distance per minute"
                      }.get(metric.column, metric.title)
    drawn = draw_minute_lines(
        ax, table, metric.column, title, metric.axis_label(shared_unit(table)),
        group_colors, empty_message=NO_OUTLINE, smooth_minutes=smooth_minutes,
        show_fish=show_fish)
    if not drawn:
        return
    if metric.kind == "ratio":
        ax.set_ylim(0, 1)
    elif metric.column == "NearWall":
        ax.set_ylim(0, 100)


def group_colors(table: pd.DataFrame) -> Dict[str, tuple]:
    """One colour per group, in the order the groups first appear."""
    from .overlay_render import fish_colors
    groups = list(dict.fromkeys(table["Group"]))
    colors = fish_colors(len(groups))
    return {group: tuple(colors[i]) for i, group in enumerate(groups)}


def minute_legend(figure, colors: Dict[str, tuple]) -> None:
    """The legend under a minute-by-minute figure: groups, and how to read
    the thin lines and the band."""
    from matplotlib.lines import Line2D
    figure.legend(
        handles=[Line2D([], [], color=color, linewidth=2.6, label=group)
                 for group, color in colors.items()],
        loc="lower center", ncol=min(len(colors), 6), frameon=False, fontsize=9,
        title="Groups (thin lines: their sessions; band: 95% CI, "
              f"{BAND_MIN_SESSIONS}+ sessions)", title_fontsize=9)


MINUTE_MEANING = (
    "\"Minute by minute\" works each measure out again on every minute of the "
    "recording by itself. Lines are coloured by group: each thin line is a "
    "session (the mean of its fish), and the thick line is the mean of the "
    "group's sessions. The thin lines are the spread to judge a difference "
    "against. A group with three or more sessions also gets a shaded band, "
    "the 95% confidence interval of its mean; with two sessions that "
    "interval would be wider than the panel. \"Each fish\" adds a faint "
    "line per fish. A point is what happened in that minute and nothing "
    "else, unless you choose a Smoothing: then each point is the mean of "
    "the 3 or 5 minutes around it, which steadies a jumpy line and also "
    "spreads a short event over its neighbours. A smoothed plot says so "
    "along its top. Look here before trusting a single number for the whole "
    "recording; one fish that stops for a few minutes moves its tank's mean.")
