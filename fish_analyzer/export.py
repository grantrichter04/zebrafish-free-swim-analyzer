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


def export_individual_metrics_csv(loaded_files: Dict, output_path: Path) -> int:
    """
    Export per-fish individual trajectory metrics to CSV.

    Columns: File, FishID, Label, Unit, ValidFrames_pct, TotalDistance,
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

    Returns
    -------
    int
        Number of fish rows exported
    """
    rows = []
    for nickname, loaded_file in loaded_files.items():
        if not loaded_file.processed_data:
            continue
        unit = loaded_file.calibration.unit_name
        for fish in loaded_file.processed_data:
            m = fish.metrics
            rows.append({
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
