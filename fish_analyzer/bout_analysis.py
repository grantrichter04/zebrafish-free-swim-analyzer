"""
fish_analyzer/bout_analysis.py
==============================
Bout-based locomotor analysis for zebrafish larvae (and adults).

Zebrafish larvae swim in discrete "bouts" — short darts separated by
pauses (inter-bout intervals). This module detects individual bouts from
the speed trace and computes per-bout metrics.

BOUT DETECTION:
    A bout starts when speed crosses above the threshold and ends when it
    drops back below. Short gaps (< merge_gap frames) between nearby bouts
    are merged to prevent a single dart from being split by frame-rate
    jitter. Bouts shorter than min_bout_frames are discarded as noise.

PER-BOUT METRICS:
    - Duration (seconds)
    - Peak speed (BL/s)
    - Mean speed during bout (BL/s)
    - Displacement (straight-line start→end, in BL)
    - Distance (total path length, in BL)
    - Heading change (signed degrees — positive = CCW/left, negative = CW/right)

SUMMARY METRICS PER FISH:
    - Bout rate (bouts/min)
    - Bout duration (median, IQR)
    - Inter-bout interval (median, IQR)
    - Peak speed distribution
    - Displacement distribution
    - Per-bout laterality (turn bias from bout heading changes)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class BoutParameters:
    """
    Parameters for bout detection.

    speed_threshold: minimum speed (BL/s) to count as active movement.
        Lower values detect more bouts but include drift/noise.
        Typical: 0.3–0.8 BL/s for larvae at 30fps.

    merge_gap_frames: if two bouts are separated by fewer than this many
        frames, merge them into one bout. Prevents a single dart from
        being split by 1–2 frames of jitter dipping below threshold.
        Typical: 2–3 frames at 30fps.

    min_bout_frames: minimum bout duration in frames. Bouts shorter than
        this are discarded as noise. At 30fps, 1 frame = 33ms.
        Typical: 1–2 frames.
    """
    speed_threshold: float = 0.5    # BL/s
    merge_gap_frames: int = 2       # frames
    min_bout_frames: int = 1        # frames

    def validate(self):
        if self.speed_threshold <= 0:
            raise ValueError(f"Speed threshold must be positive, got {self.speed_threshold}")
        if self.merge_gap_frames < 0:
            raise ValueError(f"Merge gap must be non-negative, got {self.merge_gap_frames}")
        if self.min_bout_frames < 1:
            raise ValueError(f"Min bout frames must be >= 1, got {self.min_bout_frames}")


@dataclass
class Bout:
    """A single detected swim bout."""
    start_frame: int
    end_frame: int          # exclusive
    duration_s: float
    peak_speed: float       # BL/s
    mean_speed: float       # BL/s
    displacement: float     # BL (straight line start to end)
    distance: float         # BL (total path)
    heading_change_deg: float  # signed degrees (+ = CCW/left, - = CW/right)


@dataclass
class BoutResults:
    """Complete bout analysis results for one fish."""
    fish_id: int
    identity_label: str
    bouts: List[Bout] = field(default_factory=list)
    inter_bout_intervals_s: np.ndarray = field(default_factory=lambda: np.array([]))
    summary: Dict[str, Any] = field(default_factory=dict)


class BoutDetector:
    """
    Detects swim bouts from trajectory data and computes per-bout metrics.
    """

    def __init__(self, params: BoutParameters, frame_rate: float):
        self.params = params
        self.frame_rate = frame_rate
        params.validate()

    def detect_bouts(self, x: np.ndarray, y: np.ndarray,
                     speed: np.ndarray) -> List[Bout]:
        """
        Detect bouts from position and speed arrays.

        Parameters
        ----------
        x, y : array of positions in calibrated units (BL)
        speed : array of speed in BL/s (same length as x, or len-1)

        Returns
        -------
        List of Bout objects
        """
        threshold = self.params.speed_threshold

        # Speed may be 1 shorter than x/y (from np.diff)
        n = len(speed)

        # Boolean mask: above threshold
        is_active = speed > threshold
        # NaN → not active
        is_active[np.isnan(speed)] = False

        # Find raw bout intervals [start, end)
        intervals = self._find_intervals(is_active)

        # Merge close intervals
        intervals = self._merge_intervals(intervals, self.params.merge_gap_frames)

        # Filter by minimum duration
        intervals = [(s, e) for s, e in intervals
                      if (e - s) >= self.params.min_bout_frames]

        # Compute per-bout metrics
        bouts = []
        for start, end in intervals:
            bout = self._compute_bout_metrics(x, y, speed, start, end)
            if bout is not None:
                bouts.append(bout)

        return bouts

    def _find_intervals(self, mask: np.ndarray) -> List[tuple]:
        """Find contiguous True intervals in a boolean array."""
        intervals = []
        in_interval = False
        start = 0

        for i, val in enumerate(mask):
            if val and not in_interval:
                in_interval = True
                start = i
            elif not val and in_interval:
                in_interval = False
                intervals.append((start, i))

        if in_interval:
            intervals.append((start, len(mask)))

        return intervals

    def _merge_intervals(self, intervals: List[tuple],
                         max_gap: int) -> List[tuple]:
        """Merge intervals separated by fewer than max_gap frames."""
        if not intervals or max_gap <= 0:
            return intervals

        merged = [intervals[0]]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= max_gap:
                # Merge
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        return merged

    def _compute_bout_metrics(self, x: np.ndarray, y: np.ndarray,
                               speed: np.ndarray,
                               start: int, end: int) -> Optional[Bout]:
        """Compute metrics for a single bout."""
        # Clamp to valid ranges
        # x/y may be longer than speed by 1; bout indices are based on speed
        x_start = min(start, len(x) - 1)
        x_end = min(end, len(x) - 1)

        bout_speed = speed[start:end]
        duration_s = (end - start) / self.frame_rate

        peak_speed = float(np.nanmax(bout_speed))
        mean_speed = float(np.nanmean(bout_speed))

        # Displacement: straight line from start to end position
        if x_end > x_start and not (np.isnan(x[x_start]) or np.isnan(x[x_end])):
            displacement = float(np.sqrt(
                (x[x_end] - x[x_start]) ** 2 +
                (y[x_end] - y[x_start]) ** 2
            ))
        else:
            displacement = 0.0

        # Distance: sum of step lengths during bout
        if x_end > x_start:
            bout_x = x[x_start:x_end + 1]
            bout_y = y[x_start:x_end + 1]
            dx = np.diff(bout_x)
            dy = np.diff(bout_y)
            valid = ~(np.isnan(dx) | np.isnan(dy))
            distance = float(np.sum(np.sqrt(dx[valid] ** 2 + dy[valid] ** 2)))
        else:
            distance = 0.0

        # Heading change during bout
        heading_change = self._compute_heading_change(x, y, x_start, x_end)

        return Bout(
            start_frame=start,
            end_frame=end,
            duration_s=duration_s,
            peak_speed=peak_speed,
            mean_speed=mean_speed,
            displacement=displacement,
            distance=distance,
            heading_change_deg=heading_change,
        )

    def _compute_heading_change(self, x: np.ndarray, y: np.ndarray,
                                 start: int, end: int) -> float:
        """
        Compute total signed heading change during a bout.

        For very short bouts (1–2 frames), we compare the heading entering
        the bout to the heading leaving it, rather than summing internal
        frame-to-frame changes (which would be noisy or undefined).
        """
        # We need at least 3 positions to compute a heading change
        # (2 displacement vectors → 1 turn)
        seg_x = x[start:end + 1]
        seg_y = y[start:end + 1]

        if len(seg_x) < 3:
            # For 1–2 frame bouts, use the displacement vector before
            # and after the bout to determine turn direction
            pre_start = max(0, start - 1)
            post_end = min(len(x) - 1, end + 1)

            if pre_start < start and end < post_end:
                # Heading before bout
                dx_pre = x[start] - x[pre_start]
                dy_pre = y[start] - y[pre_start]
                # Heading after bout
                dx_post = x[post_end] - x[end]
                dy_post = y[post_end] - y[end]

                if any(np.isnan([dx_pre, dy_pre, dx_post, dy_post])):
                    return 0.0

                h_pre = np.arctan2(dy_pre, dx_pre)
                h_post = np.arctan2(dy_post, dx_post)
                dh = (h_post - h_pre + np.pi) % (2 * np.pi) - np.pi
                return float(np.degrees(dh))
            return 0.0

        # For longer bouts, sum frame-to-frame heading changes
        dx = np.diff(seg_x)
        dy = np.diff(seg_y)
        valid = ~(np.isnan(dx) | np.isnan(dy))

        if np.sum(valid) < 2:
            return 0.0

        headings = np.arctan2(dy[valid], dx[valid])
        dh = np.diff(headings)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        return float(np.degrees(np.sum(dh)))

    def compute_summary(self, bouts: List[Bout],
                        total_duration_s: float) -> Dict[str, Any]:
        """Compute summary statistics from a list of bouts."""
        if not bouts:
            return {
                'bout_count': 0,
                'bout_rate_per_min': 0.0,
                'bout_duration_median_ms': np.nan,
                'bout_duration_iqr_ms': (np.nan, np.nan),
                'ibi_median_ms': np.nan,
                'ibi_iqr_ms': (np.nan, np.nan),
                'bout_peak_speed_median': np.nan,
                'bout_peak_speed_iqr': (np.nan, np.nan),
                'bout_displacement_median': np.nan,
                'bout_distance_median': np.nan,
                'bout_heading_change_mean_abs_deg': np.nan,
                'bout_laterality_index': np.nan,
                'bout_n_left': 0,
                'bout_n_right': 0,
                'bout_n_straight': 0,
            }

        durations_ms = np.array([b.duration_s * 1000 for b in bouts])
        peak_speeds = np.array([b.peak_speed for b in bouts])
        displacements = np.array([b.displacement for b in bouts])
        distances = np.array([b.distance for b in bouts])
        heading_changes = np.array([b.heading_change_deg for b in bouts])

        # Inter-bout intervals
        if len(bouts) > 1:
            ibis_ms = np.array([
                (bouts[i + 1].start_frame - bouts[i].end_frame)
                / self.frame_rate * 1000
                for i in range(len(bouts) - 1)
            ])
        else:
            ibis_ms = np.array([])

        # Laterality from per-bout heading changes
        # Use a dead zone (5°) to avoid counting noise as a turn
        dead_zone = 5.0
        n_right = int(np.sum(heading_changes < -dead_zone))
        n_left = int(np.sum(heading_changes > dead_zone))
        n_straight = int(np.sum(np.abs(heading_changes) <= dead_zone))
        n_turns = n_right + n_left
        laterality_index = (
            (n_right - n_left) / n_turns if n_turns > 0 else 0.0
        )

        return {
            'bout_count': len(bouts),
            'bout_rate_per_min': len(bouts) / total_duration_s * 60 if total_duration_s > 0 else 0.0,
            'bout_duration_median_ms': float(np.median(durations_ms)),
            'bout_duration_iqr_ms': (
                float(np.percentile(durations_ms, 25)),
                float(np.percentile(durations_ms, 75)),
            ),
            'ibi_median_ms': float(np.median(ibis_ms)) if len(ibis_ms) > 0 else np.nan,
            'ibi_iqr_ms': (
                float(np.percentile(ibis_ms, 25)) if len(ibis_ms) > 0 else np.nan,
                float(np.percentile(ibis_ms, 75)) if len(ibis_ms) > 0 else np.nan,
            ),
            'bout_peak_speed_median': float(np.median(peak_speeds)),
            'bout_peak_speed_iqr': (
                float(np.percentile(peak_speeds, 25)),
                float(np.percentile(peak_speeds, 75)),
            ),
            'bout_displacement_median': float(np.median(displacements)),
            'bout_distance_median': float(np.median(distances)),
            'bout_heading_change_mean_abs_deg': float(np.mean(np.abs(heading_changes))),
            'bout_laterality_index': float(laterality_index),
            'bout_n_left': n_left,
            'bout_n_right': n_right,
            'bout_n_straight': n_straight,
        }


def analyze_bouts_for_file(loaded_file, params: BoutParameters) -> List[BoutResults]:
    """
    Run bout analysis on all fish in a loaded trajectory file.

    Parameters
    ----------
    loaded_file : LoadedTrajectoryFile
    params : BoutParameters

    Returns
    -------
    List of BoutResults, one per fish
    """
    frame_rate = loaded_file.calibration.frame_rate
    scale = loaded_file.calibration.scale_factor
    n_fish = loaded_file.n_fish
    n_frames = loaded_file.n_frames
    total_duration_s = n_frames / frame_rate

    detector = BoutDetector(params, frame_rate)
    all_results = []

    for fish_idx in range(n_fish):
        # Get trajectory in calibrated units
        raw = loaded_file.trajectories[:, fish_idx, :]
        x = raw[:, 0] * scale
        y = (loaded_file.metadata.video_height - raw[:, 1]) * scale

        # Compute speed in calibrated units/s
        dx = np.diff(x)
        dy = np.diff(y)
        step_lengths = np.sqrt(dx ** 2 + dy ** 2)
        speed = step_lengths * frame_rate  # units/s

        # Handle NaN
        speed[np.isnan(speed)] = 0.0

        bouts = detector.detect_bouts(x, y, speed)
        summary = detector.compute_summary(bouts, total_duration_s)

        # Compute IBIs
        if len(bouts) > 1:
            ibis = np.array([
                (bouts[i + 1].start_frame - bouts[i].end_frame) / frame_rate
                for i in range(len(bouts) - 1)
            ])
        else:
            ibis = np.array([])

        label = loaded_file.metadata.identity_labels[fish_idx]
        result = BoutResults(
            fish_id=fish_idx,
            identity_label=label,
            bouts=bouts,
            inter_bout_intervals_s=ibis,
            summary=summary,
        )
        all_results.append(result)

        print(f"  Fish {fish_idx}: {len(bouts)} bouts detected "
              f"({summary['bout_rate_per_min']:.0f}/min)")

    return all_results
