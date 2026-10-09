"""
fish_analyzer/export.py
========================
CSV export utilities for analysis results.

Provides functions to export individual trajectory metrics, shoaling metrics,
and spatial/thigmotaxis results to CSV files for use in external statistics
software (R, Prism, SPSS, Excel, etc.).

TRACKING GAPS (Audit B Phase 1, 2026-08-01):
A frame in which idtracker.ai did not locate the fish is unobserved, not
"still" and not "moving". Every rate and fraction here is therefore per unit
of *observed* time, and ObservedDuration_s / LongestGap_s are exported so a
reader can see how much of the recording each number actually rests on.
Episodes cut short by a gap are reported as censored rather than counted, so
FreezeEpisodes_Complete + FreezeEpisodes_Censored together describe what was
seen. See fish_analyzer/segments.py.

WITHDRAWN COLUMNS (Audit B, 2026-08-01):
MeanAngularVelocity_deg_s, ErraticMovementCount, ErraticMovements_per_min,
BurstCount, BurstMeanSpeed, BurstFrequency_per_min, CumulativeHeading_deg and
MeanSignedAngVel_deg_s no longer appear in any export. They measured
idtracker.ai centroid noise rather than fish behaviour and are not recoverable
from centroid data. Do not add them back without real head direction; the
guard is tests/test_metric_correctness.py::test_noise_dominated_columns_are_not_exported.
"""

from pathlib import Path
from typing import Dict, List, Optional
import csv
import numpy as np


def export_combined_summary_csv(
    loaded_files: Dict,
    file_groups: Dict,
    output_path: Path,
) -> int:
    """
    Export a combined per-fish summary CSV: one row per fish, all metrics in one place.

    Columns
    -------
    Group, File, FishID, Label, Unit
    — Trajectory metrics (all from individual analysis)

    Parameters
    ----------
    loaded_files : dict
        nickname -> LoadedTrajectoryFile (must have processed_data)
    file_groups : dict
        nickname -> group label (auto-detected externally if absent)
    output_path : Path

    Returns
    -------
    int : number of fish rows written
    """
    import re

    def _auto_group(name: str) -> str:
        result = re.sub(r'[_\-\s]*\d+$', '', name).strip()
        return result if result else name

    rows = []
    for nickname, loaded_file in loaded_files.items():
        group = file_groups.get(nickname) or _auto_group(nickname)
        rows.extend(_fish_rows(loaded_file, nickname, group))

    if not rows:
        return 0

    fieldnames = _all_columns(rows)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def _fish_rows(loaded_file, nickname: str, group: str) -> List[Dict]:
    """One row per fish in the recording: its metrics if it was analysed, or
    blanks and the reason if it was not.

    A fish that failed or was gated out has no metrics, but leaving it out of
    the CSV entirely means a reader counts five fish in a six-fish recording
    and never knows (finding B10). That holds even when *every* fish in a
    session was gated out, which a strict tracking minimum makes possible.
    """
    if loaded_file.processed_data is None:   # not analysed (or results cleared)
        return []
    nan = float('nan')
    unit = loaded_file.calibration.unit_name
    params = getattr(loaded_file, 'processing_params', None)
    settings = params.export_columns() if params is not None else {}
    fixed = {
        'Group': group, 'File': nickname, 'Unit': unit,
        'PixelsPerUnit': round(loaded_file.calibration.pixels_per_unit, 4),
    }
    rows = []
    for fish in loaded_file.processed_data or []:
        m = fish.metrics
        rows.append({
            **fixed,
            'FishID': fish.fish_id,
            'Label': fish.identity_label,
            'Status': fish.status,
            'ValidFrames_pct': round(fish.valid_percentage * 100, 1),
            'ObservedDuration_s': round(m.get('observed_duration_s', nan), 2),
            'LongestGap_s': round(m.get('longest_gap_s', nan), 2),
            'TotalDistance': round(m.get('total_distance', nan), 3),
            'DistancePerTrackedMin': round(m.get('distance_per_tracked_min', nan), 3),
            'NetDisplacement': round(m.get('net_displacement', nan), 3),
            'MeanSpeed': round(m.get('mean_speed', nan), 4),
            'SpeedP99': round(m.get('speed_p99', nan), 4),
            'MedianSpeed': round(m.get('median_speed', nan), 4),
            'PathStraightness': round(m.get('mean_path_straightness', nan), 4),
            'StraightnessSeconds_pct': round(m.get('straightness_windows_used_pct', nan), 1),
            'FreezeEpisodes_Complete': m.get('freeze_count', 0),
            'FreezeEpisodes_Censored': m.get('freeze_episodes_censored', 0),
            'FreezeTotalDuration_s': round(m.get('freeze_total_duration_s', 0), 2),
            'FreezeMeanDuration_s': round(m.get('freeze_mean_duration_s', 0), 2),
            'FreezeFraction_pct': round(m.get('freeze_fraction_pct', 0), 2),
            'LateralityIndex': round(m.get('laterality_index', nan), 4),
            'RightTurns_CW': m.get('n_right_turns', 0),
            'LeftTurns_CCW': m.get('n_left_turns', 0),
            **settings,
        })
    columns = list(rows[0]) if rows else None
    for fish_idx, reason in sorted(
            getattr(loaded_file, 'excluded_fish', {}).items()):
        row = {k: nan for k in columns} if columns else {}
        row.update({
            **fixed,
            'FishID': fish_idx,
            'Label': _label_for(loaded_file, fish_idx),
            'Status': f'excluded: {reason}',
            'ValidFrames_pct': _tracked_pct(loaded_file, fish_idx),
            **settings,
        })
        rows.append(row)
    return rows


def _all_columns(rows: List[Dict]) -> List[str]:
    """Every column any row has, in first-seen order. A session whose fish
    were all excluded has only the identifying columns, and may come first."""
    return list(dict.fromkeys(k for row in rows for k in row))


def _tracked_pct(loaded_file, fish_idx: int) -> float:
    xy = np.asarray(loaded_file.trajectories[:, fish_idx, :], dtype=float)
    return round(float(np.isfinite(xy).all(axis=1).mean() * 100), 1) if len(xy) else 0.0


def _label_for(loaded_file, fish_idx: int) -> str:
    """Identity label for a fish that has no FishTrajectory to ask."""
    try:
        return loaded_file.metadata.identity_labels[fish_idx]
    except Exception:
        return str(fish_idx)


def export_individual_metrics_csv(
    loaded_files: Dict,
    output_path: Path,
    file_groups: Optional[Dict] = None,
) -> int:
    """
    Export per-fish individual trajectory metrics to CSV.

    Columns: Group, File, FishID, Label, Unit, ValidFrames_pct, TotalDistance,
             DistancePerTrackedMin, ObservedDuration_s, LongestGap_s,
             NetDisplacement, MeanSpeed, SpeedP99, MedianSpeed,
             PathStraightness, StraightnessSeconds_pct,
             FreezeEpisodes_Complete, FreezeEpisodes_Censored,
             FreezeTotalDuration_s, FreezeMeanDuration_s, FreezeFraction_pct,
             LateralityIndex, RightTurns_CW, LeftTurns_CCW, then the
             Setting_* columns the numbers were computed with

    Parameters
    ----------
    loaded_files : dict
        Dictionary of nickname -> LoadedTrajectoryFile (must have processed_data)
    output_path : Path
        Where to save the CSV file
    file_groups : dict, optional
        nickname -> group label; omit to leave Group column blank

    Returns
    -------
    int
        Number of fish rows exported
    """
    import re

    def _auto_group(name: str) -> str:
        result = re.sub(r'[_\-\s]*\d+$', '', name).strip()
        return result if result else name

    rows = []
    for nickname, loaded_file in loaded_files.items():
        if file_groups is not None:
            group = file_groups.get(nickname) or _auto_group(nickname)
        else:
            group = ''
        rows.extend(_fish_rows(loaded_file, nickname, group))

    if not rows:
        return 0

    fieldnames = _all_columns(rows)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def export_shoaling_metrics_csv(loaded_files: Dict, output_path: Path) -> int:
    """
    Export shoaling time series to CSV (one row per time sample).

    Columns: File, Unit, SampleIndex, FrameNumber, Time_s, MeanNND, MeanIID,
             HullArea

    Distances are in whatever the file was calibrated in — the Unit column
    says which. These columns were hardcoded MeanNND_BL / HullArea_BL2 while
    the calculator ignored the calibration entirely (finding B7).

    Parameters
    ----------
    loaded_files : dict
        Dictionary of nickname -> LoadedTrajectoryFile (must have shoaling_results)
    output_path : Path
        Where to save the CSV file

    Returns
    -------
    int
        Number of sample rows exported
    """
    rows = []
    for nickname, loaded_file in loaded_files.items():
        results = getattr(loaded_file, 'shoaling_results', None)
        if results is None:
            continue
        for i in range(results.n_samples_used):
            rows.append({
                'File': nickname,
                'Unit': loaded_file.calibration.unit_name,
                'PixelsPerUnit': round(loaded_file.calibration.pixels_per_unit, 4),
                'SampleIndex': i,
                'FrameNumber': int(results.frame_indices[i]),
                'Time_s': round(float(results.timestamps[i]), 2),
                'MeanNND': round(float(results.mean_nnd_per_sample[i]), 4),
                'MeanIID': round(float(results.mean_iid_per_sample[i]), 4),
                'HullArea': round(float(results.convex_hull_area_per_sample[i]), 3),
            })

    if not rows:
        return 0

    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def export_shoaling_summary_csv(loaded_files: Dict, output_path: Path) -> int:
    """
    Export shoaling summary statistics (one row per file).

    Parameters
    ----------
    loaded_files : dict
        Dictionary of nickname -> LoadedTrajectoryFile (must have shoaling_results)
    output_path : Path
        Where to save the CSV file

    Returns
    -------
    int
        Number of file rows exported
    """
    rows = []
    for nickname, loaded_file in loaded_files.items():
        results = getattr(loaded_file, 'shoaling_results', None)
        if results is None:
            continue
        rows.append({
            'File': nickname,
            'Unit': loaded_file.calibration.unit_name,
            'PixelsPerUnit': round(loaded_file.calibration.pixels_per_unit, 4),
            'NumFish': results.n_fish,
            'NumSamples': results.n_samples_used,
            'MeanNND': round(results.mean_nnd, 4),
            'StdNND': round(results.std_nnd, 4),
            'MeanIID': round(results.mean_iid, 4),
            'StdIID': round(results.std_iid, 4),
            'MeanHullArea': round(results.mean_hull_area, 3),
            'StdHullArea': round(results.std_hull_area, 3),
            'Completeness_pct': round(results.completeness_percentage, 1),
        })

    if not rows:
        return 0

    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def export_thigmotaxis_csv(loaded_files: Dict, output_path: Path) -> int:
    """
    Export thigmotaxis (spatial) summary results to CSV (one row per file).

    Parameters
    ----------
    loaded_files : dict
        Dictionary of nickname -> LoadedTrajectoryFile (must have thigmotaxis_results)
    output_path : Path
        Where to save the CSV file

    Returns
    -------
    int
        Number of file rows exported
    """
    rows = []
    for nickname, loaded_file in loaded_files.items():
        results = getattr(loaded_file, 'thigmotaxis_results', None)
        if results is None:
            continue
        rows.append({
            'File': nickname,
            'NumFish': results.n_fish,
            'NumSamples': results.n_samples,
            'MeanPctInBorder': round(results.mean_pct_in_border, 2),
            'StdPctInBorder': round(results.std_pct_in_border, 2),
            # Border and centre are shares of in-arena time and sum to 100.
            # This is the share of tracked frames that fell outside the arena
            # polygon entirely -- a large value means the arena is misaligned
            # and the two percentages describe only part of the recording.
            'MeanPctOutsideArena': round(float(np.nanmean(results.outside_arena_pct)), 2),
            'MaxPctOutsideArena': round(float(np.nanmax(results.outside_arena_pct)), 2),
        })

    if not rows:
        return 0

    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)
