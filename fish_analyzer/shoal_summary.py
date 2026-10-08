"""
fish_analyzer/shoal_summary.py
==============================
The shoaling results as the Shoaling tab shows them: one row per fish, three
measures, and what fish placed at random would give.

    NND   nearest-neighbour distance: how far each fish is from the fish
          closest to it. Small = every fish has company close by.
    IID   inter-individual distance: how far each fish is from all the
          others, on average. Small = the whole shoal is compact.

    Hull  the area of the smallest convex outline around every fish: how
          much of the tank the shoal covers. One value per moment for the
          whole shoal, not one per fish.

NND and IID differ when a shoal splits. Two tight pairs at opposite ends of
the tank have a small NND and a large IID.

All three depend on how big the tank is and how many fish are in it, so a number
on its own says little. The reference is fish placed at random inside the
arena outline: below it, the fish keep together more than chance would.

The distances themselves come from shoaling.py, which uses only moments when
every fish was tracked, sampled once a second.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .results import BIN_SECONDS, bin_count, default_group

MEASURES = [
    ("NND", "Nearest-neighbour distance",
     "At each moment, the distance from each fish to the fish closest to it, "
     "averaged over the recording. Small means every fish has company close by."),
    ("IID", "Inter-individual distance",
     "At each moment, the distance from each fish to every other fish, "
     "averaged. Small means the whole shoal is compact. It is larger than the "
     "nearest-neighbour distance, and much larger when the shoal splits into "
     "separate clusters."),
    ("Hull", "Area the shoal covers",
     "At each moment, the area of the smallest outline with no dents that "
     "holds every fish (the convex hull), averaged over the recording. It is "
     "one value for the whole shoal, so every fish in a session shows the "
     "same number. One fish straying far makes it large."),
]


def measure_unit(column: str, unit: str) -> str:
    """Hull is an area, the other two are distances."""
    return f"{unit}\u00b2" if column == "Hull" else unit

REFERENCE_MEANING = (
    "The dashed line is what the same number of fish would give if placed at "
    "random inside the arena outline. Below the line, the fish keep together "
    "more than chance. It needs the arena outline, so it is missing when "
    "there is none.")
SAMPLING_MEANING = (
    "Only moments when every fish was tracked are used, one per second. "
    "\"Frames used\" in the table is the share of the recording that was.")


def _remembered(function):
    """Keep the answer for an outline already asked about: the draws take a
    second or two, and the tab asks again on every redraw."""
    answers = {}

    def remembering(arena_vertices, n_fish, *args, **kwargs):
        key = (np.asarray(arena_vertices, dtype=float).tobytes(), n_fish, args,
               tuple(sorted(kwargs.items())))
        if key not in answers:
            answers[key] = function(arena_vertices, n_fish, *args, **kwargs)
        return answers[key]

    remembering.__doc__ = function.__doc__
    return remembering


@_remembered
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


@_remembered
def random_hull_area(arena_vertices: np.ndarray, n_fish: int,
                     n_draws: int = 1000, seed: int = 0) -> Optional[float]:
    """The mean convex-hull area of `n_fish` points placed uniformly at random
    inside the arena, in the arena's unit squared. None if it cannot be
    computed; a hull needs three fish."""
    if n_fish < 3:
        return None
    try:
        from scipy.spatial import ConvexHull
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
    # .volume is the area of a two-dimensional hull.
    return float(np.mean([ConvexHull(draw).volume for draw in points]))


def shoaling_table(loaded_files: Dict, file_groups: Optional[Dict[str, str]] = None,
                   arenas: Optional[Dict] = None) -> pd.DataFrame:
    """One row per fish, for sessions with shoaling results.

    NND and IID are that fish's own averages. Hull, the three Random
    columns and FramesUsed_pct describe its session and repeat down the
    session's rows.
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
        expected_hull = (random_hull_area(arena.vertices_bl, shoal.n_fish)
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
                "Hull": float(shoal.mean_hull_area) if shoal.n_fish >= 3 else np.nan,
                "RandomHull": expected_hull if expected_hull else np.nan,
                "FramesUsed_pct": round(float(shoal.completeness_percentage), 1),
                "Unit": shoal.unit_name,
                "PixelsPerUnit": round(loaded.calibration.pixels_per_unit, 4),
            })
    return pd.DataFrame(rows, columns=[
        "Group", "Session", "Fish", "NND", "IID", "Hull", "RandomNND",
        "RandomIID", "RandomHull", "FramesUsed_pct", "Unit", "PixelsPerUnit"])


def session_summary(table: pd.DataFrame) -> pd.DataFrame:
    """One row per session: what the table under the plot shows."""
    return (table.groupby(["Group", "Session"], sort=False)
            .agg(Fish=("Fish", "count"), NND=("NND", "mean"), IID=("IID", "mean"),
                 Hull=("Hull", "first"), RandomHull=("RandomHull", "first"),
                 RandomNND=("RandomNND", "first"), RandomIID=("RandomIID", "first"),
                 FramesUsed_pct=("FramesUsed_pct", "first"), Unit=("Unit", "first"))
            .reset_index())


def minute_table(loaded_files: Dict, file_groups: Optional[Dict[str, str]] = None
                 ) -> pd.DataFrame:
    """One row per fish per minute: the three measures averaged over the
    moments sampled in that minute. Nothing is smoothed. Hull is the shoal's,
    repeated for each of its fish."""
    file_groups = file_groups or {}
    rows = []
    for name, loaded in loaded_files.items():
        shoal = getattr(loaded, "shoaling_results", None)
        if shoal is None or len(shoal.timestamps) == 0:
            continue
        n_bins = bin_count(loaded.n_frames / loaded.calibration.frame_rate)
        minute = np.minimum((np.asarray(shoal.timestamps) // BIN_SECONDS).astype(int),
                            n_bins - 1)
        labels = loaded.metadata.identity_labels
        for index in range(n_bins):
            here = minute == index
            if not here.any():
                continue                 # no moment in this minute had every fish
            hull = (float(np.mean(shoal.convex_hull_area_per_sample[here]))
                    if shoal.n_fish >= 3 else np.nan)
            nnd = np.nanmean(shoal.individual_nnd_per_sample[here], axis=0)
            iid = np.nanmean(shoal.individual_iid_per_sample[here], axis=0)
            for fish in range(shoal.n_fish):
                rows.append({
                    "Group": file_groups.get(name) or default_group(name),
                    "Session": name,
                    "Fish": str(labels[fish]) if fish < len(labels) else str(fish + 1),
                    "Minute": index + 1,
                    "NND": float(nnd[fish]), "IID": float(iid[fish]), "Hull": hull,
                    "Unit": shoal.unit_name,
                })
    return pd.DataFrame(rows, columns=["Group", "Session", "Fish", "Minute",
                                       "NND", "IID", "Hull", "Unit"])
