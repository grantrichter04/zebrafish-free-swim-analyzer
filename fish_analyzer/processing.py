"""
fish_analyzer/processing.py
===========================
Layer 2: Individual Trajectory Processing and Metrics Calculation

This module transforms raw pixel data into clean, calibrated trajectories
and calculates individual behavioral metrics.

KEY TRANSFORMATIONS:
1. Flip Y axis (video Y=0 is top, we want Y=0 at bottom)
2. Apply calibration (pixels → body lengths)
3. Smooth trajectories to reduce tracking noise
4. Calculate derivatives (speed, acceleration)
5. Compute behavioral metrics

METRICS CALCULATED:
- Distance: total path length, net displacement
- Speed: mean, max, median, std
- Freezing: freeze count, mean freeze duration, total freeze time
- Bursting: burst count, mean burst speed, mean burst duration
- Angular velocity: mean angular velocity (degrees/second)
- Erratic movements: count of sudden large direction changes
- Path straightness: sliding-window displacement/distance ratio
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import traja
from traja import TrajaDataFrame

# Import from our own package
from .data_structures import LoadedTrajectoryFile


@dataclass
class ProcessingParameters:
    """
    Parameters controlling how trajectories are processed.

    SMOOTHING:
    Raw tracking has small random errors ("jitter"). Savitzky-Golay smoothing
    fits a polynomial to a sliding window of points, reducing noise while
    preserving real movement patterns better than simple averaging.

    FREEZE DETECTION:
    Fish are considered "frozen" (immobile) when their speed drops below
    rest_speed_threshold for at least min_freeze_frames consecutive frames.
    Freezing is a key anxiety-related behavior in zebrafish.

    BURST DETECTION:
    A "burst" is a rapid acceleration event. Detected when acceleration
    exceeds burst_accel_threshold. Consecutive above-threshold frames are
    grouped into single burst events.

    ERRATIC MOVEMENT:
    Sudden large direction changes (heading change > erratic_turn_threshold
    degrees) indicate stress or startle responses. Only counted when the
    fish is actually moving (speed > rest_speed_threshold) to avoid noise
    from stationary heading jitter.

    PATH STRAIGHTNESS:
    Computed over a sliding window (straightness_window_seconds). For each
    window: straightness = net displacement / path distance. Values near 1.0
    indicate straight swimming; values near 0 indicate circling or meandering.
    """
    apply_smoothing: bool = True
    smoothing_window: int = 5
    smoothing_polynomial_order: int = 3
    min_valid_points: int = 10
    min_valid_percentage: float = 0.01
    rest_speed_threshold: float = 0.5       # BL/s below which fish is "frozen"
    min_freeze_frames: int = 5              # Minimum consecutive frames to count as a freeze
    burst_accel_threshold: float = 2.0      # BL/s² acceleration to count as a burst
    erratic_turn_threshold: float = 90.0    # Degrees — heading change above this = erratic
    straightness_window_seconds: float = 1.0  # Window for path straightness calculation

    def validate(self):
        """Check that all parameters are valid. Raises ValueError if not."""
        if self.smoothing_window % 2 == 0:
            raise ValueError(
                f"Smoothing window must be odd, got {self.smoothing_window}. "
                f"Try {self.smoothing_window + 1}."
            )
        if self.smoothing_window < 3:
            raise ValueError(f"Smoothing window must be at least 3, got {self.smoothing_window}")
        if not 0 < self.min_valid_percentage <= 1.0:
            raise ValueError(f"Valid percentage must be between 0 and 1, got {self.min_valid_percentage}")
        if self.rest_speed_threshold < 0:
            raise ValueError(f"Rest threshold must be non-negative, got {self.rest_speed_threshold}")
        if self.burst_accel_threshold <= 0:
            raise ValueError(f"Burst acceleration threshold must be positive, got {self.burst_accel_threshold}")
        if not 0 < self.erratic_turn_threshold <= 180:
            raise ValueError(f"Erratic turn threshold must be between 0 and 180, got {self.erratic_turn_threshold}")
        if self.straightness_window_seconds <= 0:
            raise ValueError(f"Straightness window must be positive, got {self.straightness_window_seconds}")

    @classmethod
    def default_for_fish(cls) -> 'ProcessingParameters':
        """Get default parameters that work well for fish tracking."""
        return cls(
            apply_smoothing=True,
            smoothing_window=5,
            smoothing_polynomial_order=3,
            min_valid_points=10,
            min_valid_percentage=0.01,
            rest_speed_threshold=0.5,
            min_freeze_frames=5,
            burst_accel_threshold=2.0,
            erratic_turn_threshold=90.0,
            straightness_window_seconds=1.0,
        )


@dataclass
class FishTrajectory:
    """
    Complete information about one fish's trajectory and calculated metrics.

    This combines the processed spatial data (in the TrajaDataFrame) with
    all the derived metrics like speed, distance, freezing, bursting, etc.
    """
    fish_id: int
    identity_label: str
    trajectory: TrajaDataFrame
    metrics: Dict[str, Any] = field(default_factory=dict)
    n_valid_frames: int = 0
    n_total_frames: int = 0

    @property
    def valid_percentage(self) -> float:
        """What fraction of frames have valid (non-NaN) positions?"""
        if self.n_total_frames == 0:
            return 0.0
        return self.n_valid_frames / self.n_total_frames

    def summary(self) -> str:
        """Generate a human-readable summary of this fish's data."""
        lines = [
            f"Fish {self.fish_id} (ID: {self.identity_label})",
            f"Valid frames: {self.n_valid_frames}/{self.n_total_frames} ({self.valid_percentage:.1%})",
        ]
        if self.metrics:
            lines.append("\nMetrics:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                elif not isinstance(value, (dict, np.ndarray)):
                    lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class TrajectoryProcessor:
    """
    Converts raw idtracker.ai trajectory data into processed TrajaDataFrames.

    This class handles the coordinate transformations and smoothing. It creates
    one FishTrajectory object for each fish in the file.
    """

    def __init__(self, loaded_file: LoadedTrajectoryFile, params: ProcessingParameters):
        self.file = loaded_file
        self.params = params
        self.params.validate()

    def process_all_fish(self) -> List[FishTrajectory]:
        """
        Process trajectories for all fish in the file.

        Returns
        -------
        List[FishTrajectory]
            Processed trajectory for each fish that had sufficient valid data
        """
        processed_fish = []
        print(f"\nProcessing {self.file.n_fish} fish from {self.file.nickname}...")
        print(f"Using smoothing window: {self.params.smoothing_window} frames")

        for fish_idx in range(self.file.n_fish):
            try:
                fish_traj = self._process_single_fish(fish_idx)
                if fish_traj is not None:
                    processed_fish.append(fish_traj)
                    print(f"  Fish {fish_idx}: ✓ ({fish_traj.valid_percentage:.1%} valid data)")
                else:
                    print(f"  Fish {fish_idx}: ✗ (insufficient valid data)")
            except Exception as e:
                print(f"  Fish {fish_idx}: ✗ (error: {e})")
                continue

        print(f"Successfully processed {len(processed_fish)}/{self.file.n_fish} fish\n")
        return processed_fish

    def _process_single_fish(self, fish_idx: int) -> Optional[FishTrajectory]:
        """Process trajectory for a single fish."""
        raw_coords = self.file.trajectories[:, fish_idx, :]
        transformed_coords = self._transform_coordinates(raw_coords)
        df = self._create_dataframe_with_time(transformed_coords)

        valid_mask = ~(df['x'].isna() | df['y'].isna())
        n_valid = valid_mask.sum()
        n_total = len(df)
        valid_pct = n_valid / n_total if n_total > 0 else 0

        if n_valid < self.params.min_valid_points:
            return None
        if valid_pct < self.params.min_valid_percentage:
            return None

        trj = TrajaDataFrame(df)
        trj.fps = self.file.calibration.frame_rate
        trj.spatial_units = self.file.calibration.unit_name
        trj.time_units = "s"

        if self.params.apply_smoothing:
            try:
                trj = traja.smooth_sg(
                    trj,
                    w=self.params.smoothing_window,
                    p=self.params.smoothing_polynomial_order
                )
            except Exception:
                print(f"    Note: Smoothing failed for fish {fish_idx}, using raw trajectory")

        return FishTrajectory(
            fish_id=fish_idx,
            identity_label=self.file.metadata.identity_labels[fish_idx],
            trajectory=trj,
            n_valid_frames=n_valid,
            n_total_frames=n_total
        )

    def _transform_coordinates(self, raw_coords: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from pixel space to calibrated space.

        1. Flips Y axis (video convention is Y=0 at top, science is Y=0 at bottom)
        2. Scales by calibration factor (pixels → body lengths or other units)
        """
        transformed = raw_coords.copy()
        transformed[:, 1] = self.file.metadata.video_height - transformed[:, 1]
        transformed = transformed * self.file.calibration.scale_factor
        return transformed

    def _create_dataframe_with_time(self, coords: np.ndarray) -> pd.DataFrame:
        """Create a DataFrame with x, y, and time columns."""
        n_frames = len(coords)
        frame_numbers = np.arange(n_frames)
        time_seconds = frame_numbers / self.file.calibration.frame_rate
        return pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'time': time_seconds
        })


class MetricsCalculator:
    """
    Calculate behavioral metrics from processed trajectories.

    Metrics computed:
    - Total distance traveled and net displacement
    - Speed statistics (mean, max, median, std)
    - Freeze analysis (count, duration, total time)
    - Burst analysis (count, speed, duration)
    - Angular velocity (mean rate of turning, degrees/second)
    - Erratic movement count (sudden large direction changes)
    - Path straightness (sliding-window displacement/distance ratio)
    """

    def __init__(self, params: ProcessingParameters):
        self.params = params

    def calculate_all_metrics(self, fish: FishTrajectory) -> FishTrajectory:
        """
        Calculate all metrics for a fish and store them in fish.metrics.

        Computes derivatives once and reuses them across all metric calculations.
        """
        trj = fish.trajectory
        frame_rate = trj.fps if hasattr(trj, 'fps') and trj.fps else 30.0

        # Compute derivatives ONCE and reuse
        try:
            derivs = trj.traja.get_derivatives()
            speed_series = derivs['speed'].values
        except Exception as e:
            print(f"Warning: Could not compute derivatives for fish {fish.fish_id}: {e}")
            speed_series = np.full(len(trj), np.nan)

        # Distance metrics
        fish.metrics['total_distance'] = self._calc_total_distance(trj)
        fish.metrics['net_displacement'] = self._calc_net_displacement(trj)

        # Speed metrics (from pre-computed derivatives)
        fish.metrics.update(self._calc_speed_metrics(speed_series))

        # Speed time series for plotting
        time = trj['time'].values
        min_len = min(len(speed_series), len(time))
        fish.metrics['speed_time_series'] = {
            'time': time[:min_len],
            'speed': speed_series[:min_len]
        }

        # Freeze analysis
        fish.metrics.update(
            self._calc_freeze_metrics(speed_series, frame_rate)
        )

        # Burst analysis (needs acceleration)
        fish.metrics.update(
            self._calc_burst_metrics(speed_series, frame_rate)
        )

        # Angular velocity and erratic movement
        fish.metrics.update(
            self._calc_movement_direction_metrics(trj, speed_series, frame_rate)
        )

        # Path straightness (sliding window)
        fish.metrics.update(
            self._calc_path_straightness(trj, frame_rate)
        )

        return fish

    # =========================================================================
    # DISTANCE
    # =========================================================================

    def _calc_total_distance(self, trj: TrajaDataFrame) -> float:
        """Calculate total path length (sum of all step lengths)."""
        return traja.length(trj)

    def _calc_net_displacement(self, trj: TrajaDataFrame) -> float:
        """Calculate straight-line distance from start to end."""
        return traja.distance(trj)

    # =========================================================================
    # SPEED
    # =========================================================================

    def _calc_speed_metrics(self, speed_series: np.ndarray) -> Dict[str, float]:
        """Calculate summary statistics for speed from pre-computed derivatives."""
        speed = speed_series[~np.isnan(speed_series)]

        if len(speed) == 0:
            return {
                'mean_speed': np.nan,
                'max_speed': np.nan,
                'std_speed': np.nan,
                'median_speed': np.nan,
            }

        return {
            'mean_speed': float(np.mean(speed)),
            'max_speed': float(np.max(speed)),
            'std_speed': float(np.std(speed)),
            'median_speed': float(np.median(speed)),
        }

    # =========================================================================
    # FREEZE ANALYSIS
    # =========================================================================

    def _calc_freeze_metrics(self, speed_series: np.ndarray,
                              frame_rate: float) -> Dict[str, Any]:
        """
        Detect freezing episodes and compute freeze metrics.

        A freeze is a consecutive run of frames where speed < rest_speed_threshold
        lasting at least min_freeze_frames frames.

        Returns
        -------
        dict with keys:
            freeze_count: number of freeze episodes
            freeze_total_duration_s: total time spent frozen (seconds)
            freeze_mean_duration_s: average freeze duration (seconds)
            freeze_fraction_pct: % of time spent frozen
        """
        threshold = self.params.rest_speed_threshold
        min_frames = self.params.min_freeze_frames

        # Boolean mask: True where fish is below threshold
        is_slow = speed_series < threshold
        # Treat NaN as not-frozen
        is_slow[np.isnan(speed_series)] = False

        # Find consecutive runs of True (frozen)
        freeze_durations = []  # in frames
        run_length = 0

        for val in is_slow:
            if val:
                run_length += 1
            else:
                if run_length >= min_frames:
                    freeze_durations.append(run_length)
                run_length = 0
        # Check final run
        if run_length >= min_frames:
            freeze_durations.append(run_length)

        freeze_count = len(freeze_durations)
        total_freeze_frames = sum(freeze_durations)
        n_valid = np.sum(~np.isnan(speed_series))

        return {
            'freeze_count': freeze_count,
            'freeze_total_duration_s': total_freeze_frames / frame_rate,
            'freeze_mean_duration_s': (
                (total_freeze_frames / freeze_count / frame_rate)
                if freeze_count > 0 else 0.0
            ),
            'freeze_fraction_pct': (
                (total_freeze_frames / n_valid * 100) if n_valid > 0 else 0.0
            ),
        }

    # =========================================================================
    # BURST ANALYSIS
    # =========================================================================

    def _calc_burst_metrics(self, speed_series: np.ndarray,
                             frame_rate: float) -> Dict[str, Any]:
        """
        Detect burst events based on acceleration threshold.

        A burst is a consecutive run of frames where acceleration exceeds
        burst_accel_threshold. For each burst, we record the peak speed
        and duration.

        Returns
        -------
        dict with keys:
            burst_count: number of burst events
            burst_mean_speed: average peak speed during bursts
            burst_mean_duration_s: average burst duration (seconds)
            burst_frequency_per_min: bursts per minute
        """
        threshold = self.params.burst_accel_threshold

        # Compute acceleration (change in speed per frame, scaled to per second)
        speed_clean = speed_series.copy()
        acceleration = np.diff(speed_clean) * frame_rate  # units/s²

        # Treat NaN acceleration as non-burst
        accel_valid = np.where(np.isnan(acceleration), 0.0, acceleration)

        is_bursting = accel_valid > threshold

        # Find burst episodes
        burst_peak_speeds = []
        burst_durations = []  # in frames
        in_burst = False
        burst_start = 0

        for i, val in enumerate(is_bursting):
            if val and not in_burst:
                in_burst = True
                burst_start = i
            elif not val and in_burst:
                in_burst = False
                burst_end = i
                duration = burst_end - burst_start
                # Peak speed during this burst (index offset by 1 for speed vs accel)
                speed_segment = speed_clean[burst_start:burst_end + 1]
                peak = np.nanmax(speed_segment)
                burst_peak_speeds.append(peak)
                burst_durations.append(duration)

        # Handle burst continuing to end
        if in_burst:
            duration = len(is_bursting) - burst_start
            speed_segment = speed_clean[burst_start:]
            peak = np.nanmax(speed_segment) if len(speed_segment) > 0 else np.nan
            burst_peak_speeds.append(peak)
            burst_durations.append(duration)

        burst_count = len(burst_peak_speeds)
        total_time_s = len(speed_series) / frame_rate

        return {
            'burst_count': burst_count,
            'burst_mean_speed': (
                float(np.nanmean(burst_peak_speeds)) if burst_count > 0 else 0.0
            ),
            'burst_mean_duration_s': (
                float(np.mean(burst_durations) / frame_rate) if burst_count > 0 else 0.0
            ),
            'burst_frequency_per_min': (
                (burst_count / total_time_s * 60) if total_time_s > 0 else 0.0
            ),
        }

    # =========================================================================
    # ANGULAR VELOCITY & ERRATIC MOVEMENT
    # =========================================================================

    def _calc_movement_direction_metrics(self, trj: TrajaDataFrame,
                                          speed_series: np.ndarray,
                                          frame_rate: float) -> Dict[str, float]:
        """
        Calculate angular velocity and erratic movement count.

        Angular velocity: mean absolute rate of heading change (degrees/second).
        Computed only when the fish is moving (speed > threshold) to avoid
        noise from stationary heading jitter.

        Erratic movement count: number of frames where the heading change
        exceeds erratic_turn_threshold degrees AND the fish is moving.

        Heading is computed frame-to-frame (diff of 1) for consistency.
        """
        try:
            x = trj['x'].values
            y = trj['y'].values

            # Frame-to-frame displacement
            dx = np.diff(x)
            dy = np.diff(y)

            # Heading at each step (radians)
            headings = np.arctan2(dy, dx)

            # Angular change between consecutive headings
            dheading = np.diff(headings)

            # Wrap to [-pi, pi]
            dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
            dheading_deg = np.degrees(dheading)

            # Angular velocity in degrees/second
            angular_velocity = np.abs(dheading_deg) * frame_rate

            # Only consider frames where the fish is actually moving
            # speed_series is len(trj), dheading_deg is len(trj)-2
            # Align: dheading[i] corresponds to the turn between step i and step i+1,
            # which happens around frame i+1. Use speed at frame i+1.
            min_len = min(len(dheading_deg), len(speed_series) - 1)
            if min_len <= 0:
                return self._empty_direction_metrics()

            angular_vel_aligned = angular_velocity[:min_len]
            dheading_aligned = np.abs(dheading_deg[:min_len])
            speed_aligned = speed_series[1:min_len + 1]

            # Mask: fish must be moving and values must be valid
            moving_mask = (
                (speed_aligned > self.params.rest_speed_threshold) &
                ~np.isnan(speed_aligned) &
                ~np.isnan(angular_vel_aligned)
            )

            if np.sum(moving_mask) == 0:
                return self._empty_direction_metrics()

            mean_angular_velocity = float(np.mean(angular_vel_aligned[moving_mask]))

            # Erratic movements: large turns while moving
            erratic_mask = (
                moving_mask & (dheading_aligned > self.params.erratic_turn_threshold)
            )
            erratic_count = int(np.sum(erratic_mask))
            total_time_s = len(speed_series) / frame_rate
            erratic_per_min = (erratic_count / total_time_s * 60) if total_time_s > 0 else 0.0

            return {
                'mean_angular_velocity_deg_s': mean_angular_velocity,
                'erratic_movement_count': erratic_count,
                'erratic_movements_per_min': erratic_per_min,
            }

        except Exception as e:
            print(f"Warning: Direction metrics failed: {e}")
            return self._empty_direction_metrics()

    def _empty_direction_metrics(self) -> Dict[str, float]:
        """Return NaN/zero direction metrics when calculation fails."""
        return {
            'mean_angular_velocity_deg_s': np.nan,
            'erratic_movement_count': 0,
            'erratic_movements_per_min': 0.0,
        }

    # =========================================================================
    # PATH STRAIGHTNESS
    # =========================================================================

    def _calc_path_straightness(self, trj: TrajaDataFrame,
                                 frame_rate: float) -> Dict[str, float]:
        """
        Calculate path straightness over a sliding window.

        For each window of straightness_window_seconds:
            straightness = net_displacement / path_distance

        Values near 1.0 = straight swimming
        Values near 0.0 = circling, meandering, or stationary

        Returns mean straightness across all windows.
        """
        try:
            x = trj['x'].values
            y = trj['y'].values
            window_frames = max(2, int(self.params.straightness_window_seconds * frame_rate))

            straightness_values = []

            for start in range(0, len(x) - window_frames + 1, window_frames // 2):
                end = start + window_frames
                x_win = x[start:end]
                y_win = y[start:end]

                # Skip windows with NaN
                if np.any(np.isnan(x_win)) or np.any(np.isnan(y_win)):
                    continue

                # Net displacement (straight-line start to end)
                net_disp = np.sqrt(
                    (x_win[-1] - x_win[0]) ** 2 +
                    (y_win[-1] - y_win[0]) ** 2
                )

                # Path distance (sum of step lengths)
                dx = np.diff(x_win)
                dy = np.diff(y_win)
                step_lengths = np.sqrt(dx ** 2 + dy ** 2)
                path_dist = np.sum(step_lengths)

                if path_dist > 0:
                    straightness_values.append(net_disp / path_dist)

            if len(straightness_values) == 0:
                return {'mean_path_straightness': np.nan}

            return {
                'mean_path_straightness': float(np.mean(straightness_values)),
            }

        except Exception as e:
            print(f"Warning: Path straightness calculation failed: {e}")
            return {'mean_path_straightness': np.nan}


def process_and_analyze_file(
    loaded_file: LoadedTrajectoryFile,
    params: Optional[ProcessingParameters] = None
) -> List[FishTrajectory]:
    """
    Complete pipeline: process trajectories and calculate all metrics.

    This is a convenience function that combines TrajectoryProcessor and
    MetricsCalculator into a single call.

    Parameters
    ----------
    loaded_file : LoadedTrajectoryFile
        The loaded trajectory data
    params : ProcessingParameters, optional
        Processing settings. Uses defaults if not provided.

    Returns
    -------
    List[FishTrajectory]
        Processed trajectories with metrics for all fish
    """
    if params is None:
        params = ProcessingParameters.default_for_fish()

    processor = TrajectoryProcessor(loaded_file, params)
    fish_list = processor.process_all_fish()

    calculator = MetricsCalculator(params)
    for fish in fish_list:
        calculator.calculate_all_metrics(fish)

    return fish_list
