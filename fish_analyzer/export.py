"""
fish_analyzer/export.py
========================
CSV export utilities for analysis results.

Provides functions to export individual trajectory metrics, shoaling metrics,
and spatial/thigmotaxis results to CSV files for use in external statistics
software (R, Prism, SPSS, Excel, etc.).
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
                'ValidFrames_pct':             round(fish.valid_percentage * 100, 1),
                'TotalDistance':               round(m.get('total_distance', nan), 3),
                'NetDisplacement':             round(m.get('net_displacement', nan), 3),
                'MeanSpeed':                   round(m.get('mean_speed', nan), 4),
                'MaxSpeed':                    round(m.get('max_speed', nan), 4),
                'MedianSpeed':                 round(m.get('median_speed', nan), 4),
                'PathStraightness':            round(m.get('mean_path_straightness', nan), 4),
                'FreezeCount':                 m.get('freeze_count', 0),
                'FreezeTotalDuration_s':       round(m.get('freeze_total_duration_s', 0), 2),
                'FreezeMeanDuration_s':        round(m.get('freeze_mean_duration_s', 0), 2),
                'FreezeFraction_pct':          round(m.get('freeze_fraction_pct', 0), 2),
                'BurstCount':                  m.get('burst_count', 0),
                'BurstMeanSpeed':              round(m.get('burst_mean_speed', 0), 4),
                'BurstFrequency_per_min':      round(m.get('burst_frequency_per_min', 0), 2),
                'MeanAngularVelocity_deg_s':   round(m.get('mean_angular_velocity_deg_s', nan), 2),
                'ErraticMovementCount':        m.get('erratic_movement_count', 0),
                'ErraticMovements_per_min':    round(m.get('erratic_movements_per_min', 0), 2),
                'LateralityIndex':             round(m.get('laterality_index', nan), 4),
                'RightTurns_CW':               m.get('n_right_turns', 0),
                'LeftTurns_CCW':               m.get('n_left_turns', 0),
                'CumulativeHeading_deg':       round(m.get('cumulative_heading_change_deg', nan), 1),
                'MeanSignedAngVel_deg_s':      round(m.get('mean_signed_angular_velocity_deg_s', nan), 2),
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
                    'Bout_Rate_per_min':         round(s.get('bout_rate_per_min', nan), 3),
                    'Bout_Duration_Median_ms':   round(s.get('bout_duration_median_ms', nan), 2),
                    'Bout_Duration_Q1_ms':       round(iqr_dur[0] if not _isnan(iqr_dur[0]) else nan, 2),
                    'Bout_Duration_Q3_ms':       round(iqr_dur[1] if not _isnan(iqr_dur[1]) else nan, 2),
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
                })
            else:
                row.update({
                    'Bout_Analyzed':             False,
                    'Bout_Count':                nan,
                    'Bout_Rate_per_min':         nan,
                    'Bout_Duration_Median_ms':   nan,
                    'Bout_Duration_Q1_ms':       nan,
                    'Bout_Duration_Q3_ms':       nan,
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
                })

            rows.append(row)

    if not rows:
        return 0

    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


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
             NetDisplacement, MeanSpeed, MaxSpeed, MedianSpeed,
             PathStraightness, FreezeCount, FreezeTotalDuration_s,
             FreezeMeanDuration_s, FreezeFraction_pct, BurstCount,
             BurstMeanSpeed, BurstFrequency_per_min,
             MeanAngularVelocity_deg_s, ErraticMovementCount,
             ErraticMovements_per_min

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
                'ValidFrames_pct': round(fish.valid_percentage * 100, 1),
                'TotalDistance': round(m.get('total_distance', float('nan')), 3),
                'NetDisplacement': round(m.get('net_displacement', float('nan')), 3),
                'MeanSpeed': round(m.get('mean_speed', float('nan')), 4),
                'MaxSpeed': round(m.get('max_speed', float('nan')), 4),
                'MedianSpeed': round(m.get('median_speed', float('nan')), 4),
                'PathStraightness': round(m.get('mean_path_straightness', float('nan')), 4),
                'FreezeCount': m.get('freeze_count', 0),
                'FreezeTotalDuration_s': round(m.get('freeze_total_duration_s', 0), 2),
                'FreezeMeanDuration_s': round(m.get('freeze_mean_duration_s', 0), 2),
                'FreezeFraction_pct': round(m.get('freeze_fraction_pct', 0), 2),
                'BurstCount': m.get('burst_count', 0),
                'BurstMeanSpeed': round(m.get('burst_mean_speed', 0), 4),
                'BurstFrequency_per_min': round(m.get('burst_frequency_per_min', 0), 2),
                'MeanAngularVelocity_deg_s': round(m.get('mean_angular_velocity_deg_s', float('nan')), 2),
                'ErraticMovementCount': m.get('erratic_movement_count', 0),
                'ErraticMovements_per_min': round(m.get('erratic_movements_per_min', 0), 2),
                'LateralityIndex': round(m.get('laterality_index', float('nan')), 4),
                'RightTurns_CW': m.get('n_right_turns', 0),
                'LeftTurns_CCW': m.get('n_left_turns', 0),
                'CumulativeHeading_deg': round(m.get('cumulative_heading_change_deg', float('nan')), 1),
                'MeanSignedAngVel_deg_s': round(m.get('mean_signed_angular_velocity_deg_s', float('nan')), 2),
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

    Columns: File, SampleIndex, FrameNumber, Time_s, MeanNND, MeanIID, HullArea

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
                'SampleIndex': i,
                'FrameNumber': int(results.frame_indices[i]),
                'Time_s': round(float(results.timestamps[i]), 2),
                'MeanNND_BL': round(float(results.mean_nnd_per_sample[i]), 4),
                'MeanIID_BL': round(float(results.mean_iid_per_sample[i]), 4),
                'HullArea_BL2': round(float(results.convex_hull_area_per_sample[i]), 3),
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
            'NumFish': results.n_fish,
            'NumSamples': results.n_samples_used,
            'MeanNND_BL': round(results.mean_nnd, 4),
            'StdNND_BL': round(results.std_nnd, 4),
            'MeanIID_BL': round(results.mean_iid, 4),
            'StdIID_BL': round(results.std_iid, 4),
            'MeanHullArea_BL2': round(results.mean_hull_area, 3),
            'StdHullArea_BL2': round(results.std_hull_area, 3),
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
        })

    if not rows:
        return 0

    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)
