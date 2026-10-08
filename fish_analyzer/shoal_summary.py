"""
fish_analyzer/shoal_summary.py
==============================
The shoaling results as the Shoaling tab shows them: one row per fish, two
measures, and what fish placed at random would give.

    NND   nearest-neighbour distance: how far each fish is from the fish
          closest to it. Small = every fish has company close by.
    IID   inter-individual distance: how far each fish is from all the
          others, on average. Small = the whole shoal is compact.

The two differ when a shoal splits. Two tight pairs at opposite ends of the
tank have a small NND and a large IID.

Both depend on how big the tank is and how many fish are in it, so a number
on its own says little. The reference is fish placed at random inside the
arena outline: below it, the fish keep together more than chance would.

The distances themselves come from shoaling.py, which uses only moments when
every fish was tracked, sampled once a second.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .results import default_group

MEASURES = [
    ("NND", "Nearest-neighbour distance",
     "At each moment, the distance from each fish to the fish closest to it, "
     "averaged over the recording. Small means every fish has company close by."),
    ("IID", "Inter-individual distance",
     "At each moment, the distance from each fish to every other fish, "
     "averaged. Small means the whole shoal is compact. It is larger than the "
     "nearest-neighbour distance, and much larger when the shoal splits into "
     "separate clusters."),
]
REFERENCE_MEANING = (
    "The dashed line is what the same number of fish would give if placed at "
    "random inside the arena outline. Below the line, the fish keep together "
    "more than chance. It needs the arena outline, so it is missing when "
    "there is none.")
SAMPLING_MEANING = (
    "Only moments when every fish was tracked are used, one per second. "
    "\"Frames used\" in the table is the share of the recording that was.")


def random_expectation(arena_vertices: np.ndarray, n_fish: int,
                       n_draws: int = 3000, seed: int = 0
                       ) -> Optional[Tuple[float, float]]:
    """(NND, IID) for `n_fish` points placed uniformly at random inside the
    arena, in the arena's unit. None if it cannot be computed.

    Found by drawing: positions are sampled in the outline's bounding box and
    those outside the outline rejected. The seed is fixed so the reference
    line does not move between redraws.
    """
    if n_fish < 2:
        return None
    try:
        from shapely import contains_xy
        from shapely.geometry import Polygon
        arena = Polygon(np.asarray(arena_vertices, dtype=float))
    except Exception:
        return None
    if not arena.is_valid or arena.area <= 0:
        return None

    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = arena.bounds
    needed = n_draws * n_fish
    points = np.empty((0, 2))
    while len(points) < needed:
        candidates = np.column_stack([
            rng.uniform(minx, maxx, needed * 2), rng.uniform(miny, maxy, needed * 2)])
        inside = contains_xy(arena, candidates[:, 0], candidates[:, 1])
        points = np.vstack([points, candidates[inside]])
    points = points[:needed].reshape(n_draws, n_fish, 2)

    distances = np.linalg.norm(points[:, :, None, :] - points[:, None, :, :], axis=-1)
    off_diagonal = ~np.eye(n_fish, dtype=bool)
    iid = distances[:, off_diagonal].mean()
    distances[:, ~off_diagonal] = np.inf
    nnd = distances.min(axis=2).mean()
    return float(nnd), float(iid)


def shoaling_table(loaded_files: Dict, file_groups: Optional[Dict[str, str]] = None,
                   arenas: Optional[Dict] = None) -> pd.DataFrame:
    """One row per fish, for sessions with shoaling results.

    NND and IID are that fish's own averages. RandomNND, RandomIID and
    FramesUsed_pct describe its session and repeat down the session's rows.
    """
    file_groups = file_groups or {}
    arenas = arenas or {}
    rows = []
    for name, loaded in loaded_files.items():
        shoal = getattr(loaded, "shoaling_results", None)
        if shoal is None:
            continue
        arena = arenas.get(name)
        expected = (random_expectation(arena.vertices_bl, shoal.n_fish)
                    if arena is not None else None)
        own_iid = np.nanmean(shoal.individual_iid_per_sample, axis=0)
        labels = loaded.metadata.identity_labels
        for index in range(shoal.n_fish):
            rows.append({
                "Group": file_groups.get(name) or default_group(name),
                "Session": name,
                "Fish": str(labels[index]) if index < len(labels) else str(index + 1),
                "NND": float(shoal.per_fish_mean_nnd[index]),
                "IID": float(own_iid[index]),
                "RandomNND": expected[0] if expected else np.nan,
                "RandomIID": expected[1] if expected else np.nan,
                "FramesUsed_pct": round(float(shoal.completeness_percentage), 1),
                "Unit": shoal.unit_name,
                "PixelsPerUnit": round(loaded.calibration.pixels_per_unit, 4),
            })
    return pd.DataFrame(rows, columns=[
        "Group", "Session", "Fish", "NND", "IID", "RandomNND", "RandomIID",
        "FramesUsed_pct", "Unit", "PixelsPerUnit"])


def session_summary(table: pd.DataFrame) -> pd.DataFrame:
    """One row per session: what the table under the plot shows."""
    return (table.groupby(["Group", "Session"], sort=False)
            .agg(Fish=("Fish", "count"), NND=("NND", "mean"), IID=("IID", "mean"),
                 RandomNND=("RandomNND", "first"), RandomIID=("RandomIID", "first"),
                 FramesUsed_pct=("FramesUsed_pct", "first"), Unit=("Unit", "first"))
            .reset_index())


def plot_over_time(ax, loaded_files: Dict, measure: str, title: str,
                   session_colors: Dict[str, tuple],
                   smooth_seconds: float = 15.0) -> None:
    """One line per session: the shoal's mean `measure` through the recording,
    as a running mean over `smooth_seconds`."""
    attribute = {"NND": "mean_nnd_per_sample", "IID": "mean_iid_per_sample"}[measure]
    unit = ""
    for name, loaded in loaded_files.items():
        shoal = getattr(loaded, "shoaling_results", None)
        if shoal is None:
            continue
        unit = shoal.unit_name
        values = np.asarray(getattr(shoal, attribute), dtype=float)
        minutes = np.asarray(shoal.timestamps, dtype=float) / 60.0
        if len(minutes) > 1:
            step = float(np.median(np.diff(shoal.timestamps)))
            window = max(1, int(round(smooth_seconds / step))) if step > 0 else 1
            if window > 1 and len(values) >= window:
                values = np.convolve(values, np.ones(window) / window, mode="valid")
                minutes = minutes[window - 1:]
        ax.plot(minutes, values, color=session_colors[name], linewidth=1.8, label=name)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(unit)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
