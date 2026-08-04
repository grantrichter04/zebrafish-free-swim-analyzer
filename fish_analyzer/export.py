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
FreezeEpisodes_Complete + FreezeEpisodes_Censored, and Bout_Count +
Bout_Censored, together describe what was seen. See fish_analyzer/segments.py.

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
    bout_results: Dict,
    file_groups: Dict,
    output_path: Path,
) -> int:
    """
    Export a combined per-fish summary CSV: one row per fish, all metrics in one place.

    Columns
    -------
    Group, File, FishID, Label, Unit
    — Trajectory metrics (all from individual analysis)
    — Bout summary stats (median/IQR per fish; NaN if bout analysis not run)
    — Bout_Analyzed flag so the user knows which fish have bout data

    Parameters
    ----------
    loaded_files : dict
        nickname -> LoadedTrajectoryFile (must have processed_data)
    bout_results : dict
        nickname -> List[BoutResults] (may be empty or missing keys)
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

    nan = float('nan')

    rows = []
    for nickname, loaded_file in loaded_files.items():
        if not loaded_file.processed_data:
            continue

        unit = loaded_file.calibration.unit_name
        group = file_groups.get(nickname) or _auto_group(nickname)

        # Index bout results for this file by fish_id for fast lookup
        fish_bout_map = {}
        for br in bout_results.get(nickname, []):
            fish_bout_map[br.fish_id] = br

        for fish in loaded_file.processed_data:
            m = fish.metrics

            # ---- trajectory metrics ----
            row = {
                'Group':                       group,
                'File':                        nickname,
                'FishID':                      fish.fish_id,
                'Label':                       fish.identity_label,
                'Unit':                        unit,
                'Status':                      fish.status,
                'ValidFrames_pct':             round(fish.valid_percentage * 100, 1),
                'ObservedDuration_s':          round(m.get('observed_duration_s', nan), 2),
                'LongestGap_s':                round(m.get('longest_gap_s', nan), 2),
                'TotalDistance':               round(m.get('total_distance', nan), 3),
                'NetDisplacement':             round(m.get('net_displacement', nan), 3),
                'MeanSpeed':                   round(m.get('mean_speed', nan), 4),
                'SpeedP99':                    round(m.get('speed_p99', nan), 4),
                'MedianSpeed':                 round(m.get('median_speed', nan), 4),
                'PathStraightness':            round(m.get('mean_path_straightness', nan), 4),
                'FreezeEpisodes_Complete':     m.get('freeze_count', 0),
                'FreezeEpisodes_Censored':     m.get('freeze_episodes_censored', 0),
                'FreezeTotalDuration_s':       round(m.get('freeze_total_duration_s', 0), 2),
                'FreezeMeanDuration_s':        round(m.get('freeze_mean_duration_s', 0), 2),
                'FreezeFraction_pct':          round(m.get('freeze_fraction_pct', 0), 2),
                'LateralityIndex':             round(m.get('laterality_index', nan), 4),
                'RightTurns_CW':               m.get('n_right_turns', 0),
                'LeftTurns_CCW':               m.get('n_left_turns', 0),
            }

            # ---- bout summary metrics ----
            br = fish_bout_map.get(fish.fish_id)
            if br is not None:
                s = br.summary
                iqr_dur = s.get('bout_duration_iqr_ms', (nan, nan))
                iqr_ibi = s.get('ibi_iqr_ms', (nan, nan))
                iqr_spd = s.get('bout_peak_speed_iqr', (nan, nan))
                row.update({
                    'Bout_Analyzed':             True,
                    'Bout_Count':                s.get('bout_count', 0),
                    'Bout_Censored':             s.get('bout_censored', 0),
                    'Bout_Rate_per_min':         round(s.get('bout_rate_per_min', nan), 3),
                    'Bout_Duration_Median_ms':   round(s.get('bout_duration_median_ms', nan), 2),
                    'Bout_Duration_Q1_ms':       round(iqr_dur[0] if not _isnan(iqr_dur[0]) else nan, 2),
                    'Bout_Duration_Q3_ms':       round(iqr_dur[1] if not _isnan(iqr_dur[1]) else nan, 2),
                    'IBI_N':                     s.get('ibi_n', 0),
                    'IBI_Median_ms':             round(s.get('ibi_median_ms', nan), 2),
                    'IBI_Q1_ms':                 round(iqr_ibi[0] if not _isnan(iqr_ibi[0]) else nan, 2),
                    'IBI_Q3_ms':                 round(iqr_ibi[1] if not _isnan(iqr_ibi[1]) else nan, 2),
                    'Bout_PeakSpeed_Median':     round(s.get('bout_peak_speed_median', nan), 4),
                    'Bout_PeakSpeed_Q1':         round(iqr_spd[0] if not _isnan(iqr_spd[0]) else nan, 4),
                    'Bout_PeakSpeed_Q3':         round(iqr_spd[1] if not _isnan(iqr_spd[1]) else nan, 4),
                    'Bout_Displacement_Median':  round(s.get('bout_displacement_median', nan), 4),
                    'Bout_Distance_Median':      round(s.get('bout_distance_median', nan), 4),
                    'Bout_MeanAbsTurnAngle_deg': round(s.get('bout_heading_change_mean_abs_deg', nan), 2),
                    'Bout_LateralityIndex':      round(s.get('bout_laterality_index', nan), 4),
                    'Bout_N_Left':               s.get('bout_n_left', 0),
                    'Bout_N_Right':              s.get('bout_n_right', 0),
                    'Bout_N_Straight':           s.get('bout_n_straight', 0),
                    'Bout_N_TurnUnmeasurable':   s.get('bout_n_heading_unmeasurable', 0),
                })
            else:
                row.update({
                    'Bout_Analyzed':             False,
                    'Bout_Count':                nan,
                    'Bout_Censored':             nan,
                    'Bout_Rate_per_min':         nan,
                    'Bout_Duration_Median_ms':   nan,
                    'Bout_Duration_Q1_ms':       nan,
                    'Bout_Duration_Q3_ms':       nan,
                    'IBI_N':                     nan,
                    'IBI_Median_ms':             nan,
                    'IBI_Q1_ms':                 nan,
                    'IBI_Q3_ms':                 nan,
                    'Bout_PeakSpeed_Median':     nan,
                    'Bout_PeakSpeed_Q1':         nan,
                    'Bout_PeakSpeed_Q3':         nan,
                    'Bout_Displacement_Median':  nan,
                    'Bout_Distance_Median':      nan,
                    'Bout_MeanAbsTurnAngle_deg': nan,
                    'Bout_LateralityIndex':      nan,
                    'Bout_N_Left':               nan,
                    'Bout_N_Right':              nan,
                    'Bout_N_Straight':           nan,
                    'Bout_N_TurnUnmeasurable':   nan,
                })

            rows.append(row)

        # A fish that failed or was gated out has no metrics, but leaving it
        # out of the CSV entirely means a reader counts five fish in a
        # six-fish recording and never knows (finding B10).
        for fish_idx, reason in sorted(
                getattr(loaded_file, 'excluded_fish', {}).items()):
            excluded = {k: nan for k in rows[0]} if rows else {}
            excluded.update({
                'Group': group, 'File': nickname, 'FishID': fish_idx,
                'Label': _label_for(loaded_file, fish_idx), 'Unit': unit,
                'Status': f'excluded: {reason}',
            })
            rows.append(excluded)

    if not rows:
        return 0

    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def _label_for(loaded_file, fish_idx: int) -> str:
    """Identity label for a fish that has no FishTrajectory to ask."""
    try:
        return loaded_file.metadata.identity_labels[fish_idx]
    except Exception:
        return str(fish_idx)


def _isnan(v) -> bool:
    """Safe nan check for values that may not be float."""
    try:
        return v != v  # NaN != NaN
    except Exception:
        return False


def export_individual_metrics_csv(
    loaded_files: Dict,
    output_path: Path,
    file_groups: Optional[Dict] = None,
) -> int:
    """
    Export per-fish individual trajectory metrics to CSV.

    Columns: Group, File, FishID, Label, Unit, ValidFrames_pct, TotalDistance,
             ObservedDuration_s, LongestGap_s, NetDisplacement, MeanSpeed,
             SpeedP99, MedianSpeed, PathStraightness,
             FreezeEpisodes_Complete, FreezeEpisodes_Censored,
             FreezeTotalDuration_s, FreezeMeanDuration_s, FreezeFraction_pct,
             LateralityIndex, RightTurns_CW, LeftTurns_CCW

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
        if not loaded_file.processed_data:
            continue
        unit = loaded_file.calibration.unit_name
        if file_groups is not None:
            group = file_groups.get(nickname) or _auto_group(nickname)
        else:
            group = ''
        for fish in loaded_file.processed_data:
            m = fish.metrics
            rows.append({
                'Group': group,
                'File': nickname,
                'FishID': fish.fish_id,
                'Label': fish.identity_label,
                'Unit': unit,
                'Status': fish.status,
                'ValidFrames_pct': round(fish.valid_percentage * 100, 1),
                'ObservedDuration_s': round(m.get('observed_duration_s', float('nan')), 2),
                'LongestGap_s': round(m.get('longest_gap_s', float('nan')), 2),
                'TotalDistance': round(m.get('total_distance', float('nan')), 3),
                'NetDisplacement': round(m.get('net_displacement', float('nan')), 3),
                'MeanSpeed': round(m.get('mean_speed', float('nan')), 4),
                'SpeedP99': round(m.get('speed_p99', float('nan')), 4),
                'MedianSpeed': round(m.get('median_speed', float('nan')), 4),
                'PathStraightness': round(m.get('mean_path_straightness', float('nan')), 4),
                'FreezeEpisodes_Complete': m.get('freeze_count', 0),
                'FreezeEpisodes_Censored': m.get('freeze_episodes_censored', 0),
                'FreezeTotalDuration_s': round(m.get('freeze_total_duration_s', 0), 2),
                'FreezeMeanDuration_s': round(m.get('freeze_mean_duration_s', 0), 2),
                'FreezeFraction_pct': round(m.get('freeze_fraction_pct', 0), 2),
                'LateralityIndex': round(m.get('laterality_index', float('nan')), 4),
                'RightTurns_CW': m.get('n_right_turns', 0),
                'LeftTurns_CCW': m.get('n_left_turns', 0),
            })

    if not rows:
        return 0

    fieldnames = list(rows[0].keys())
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
