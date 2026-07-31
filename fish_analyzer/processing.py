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
- Path straightness: sliding-window displacement/distance ratio
- Turning bias: laterality index, left/right turn counts

WITHDRAWN 2026-08-01 (Audit B):
Angular velocity, erratic-movement counts, all three burst metrics, cumulative
heading change and mean signed angular velocity have been removed. All six
derive from the direction of frame-to-frame centroid displacement, which at
this tracker's noise level carries no usable signal: a *perfectly straight*
synthetic swimmer at the observed median speed, with 0.5-1.0 px of centroid
noise, reproduced the entire range those columns reported across four real
sessions. They were removed rather than repaired because no amount of
thresholding recovers a signal that is not there -- recovering them needs real
head direction (see head_detection/), not better arithmetic.

Laterality survives because it counts turn *directions*, which stay balanced
under symmetric noise, rather than turn *rates*, which do not. See
AUDIT_B_CORRECTNESS.md B1, B3 and B15, and tests/test_metric_correctness.py.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import traja
from traja import TrajaDataFrame

# Import from our own package
from .data_structures import LoadedTrajectoryFile
from .segments import (backward_speed_slices, contiguous_tracked_segments,
                       first_and_last_tracked, longest_gap_frames, run_lengths,
                       tracked_frame_count)


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

    TURNING BIAS:
    rest_speed_threshold doubles as the gate on turn counting: a heading
    change is only counted when the fish is actually moving, so a stationary
    fish contributes nothing.

    PATH STRAIGHTNESS:
    Computed over a sliding window (straightness_window_seconds). For each
    window: straightness = net displacement / path distance. Values near 1.0
    indicate straight swimming; values near 0 indicate circling or meandering.
    """
    apply_smoothing: bool = False
    smoothing_window: int = 5
    smoothing_polynomial_order: int = 3
    min_valid_points: int = 10
    min_valid_percentage: float = 0.01
    rest_speed_threshold: float = 0.5       # BL/s below which fish is "frozen"
    min_freeze_frames: int = 5              # Minimum consecutive frames to count as a freeze
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
        if self.straightness_window_seconds <= 0:
            raise ValueError(f"Straightness window must be positive, got {self.straightness_window_seconds}")

    @classmethod
    def default_for_fish(cls) -> 'ProcessingParameters':
        """Get default parameters that work well for fish tracking."""
        return cls(
            apply_smoothing=False,
            smoothing_window=5,
            smoothing_polynomial_order=3,
            min_valid_points=10,
            min_valid_percentage=0.01,
            rest_speed_threshold=0.5,
            min_freeze_frames=5,
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
    smoothing_failed: bool = False
    #: Names of metric groups whose computation raised. A NaN in the export is
    #: ambiguous on its own — it can mean "this fish never turned" as easily as
    #: "the calculation crashed" — so the exporter reads this to tell them
    #: apart (finding B10).
    failed_metrics: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """'ok' or 'partial: <what failed>', for the CSV's Status column."""
        if not self.failed_metrics:
            return "ok"
        return "partial: " + ", ".join(sorted(self.failed_metrics))

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
        #: fish_idx -> why it is missing from the returned list. A fish that
        #: failed or was gated out used to vanish silently, leaving nothing in
        #: the export to say the file had more fish in it (finding B10).
        self.excluded: Dict[int, str] = {}

        print(f"\nProcessing {self.file.n_fish} fish from {self.file.nickname}...")
        print(f"Using smoothing window: {self.params.smoothing_window} frames")

        for fish_idx in range(self.file.n_fish):
            try:
                fish_traj = self._process_single_fish(fish_idx)
                if fish_traj is not None:
                    processed_fish.append(fish_traj)
                    print(f"  Fish {fish_idx}: [ok] ({fish_traj.valid_percentage:.1%} valid data)")
                else:
                    self.excluded[fish_idx] = "insufficient valid data"
                    print(f"  Fish {fish_idx}: [skip] (insufficient valid data)")
            except Exception as e:
                self.excluded[fish_idx] = f"failed: {e}"
                print(f"  Fish {fish_idx}: [FAILED] (error: {e})")
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

        smoothing_failed = False
        if self.params.apply_smoothing:
            try:
                # Interpolate NaN gaps before smoothing to prevent
                # traja.smooth_sg from zero-filling them (which creates
                # large artificial spikes that corrupt neighboring frames).
                nan_mask = trj['x'].isna() | trj['y'].isna()
                if nan_mask.any():
                    valid_idx = np.where(~nan_mask)[0]
                    nan_idx = np.where(nan_mask)[0]
                    trj.loc[nan_mask, 'x'] = np.interp(
                        nan_idx, valid_idx, trj['x'].values[valid_idx])
                    trj.loc[nan_mask, 'y'] = np.interp(
                        nan_idx, valid_idx, trj['y'].values[valid_idx])

                trj = traja.smooth_sg(
                    trj,
                    w=self.params.smoothing_window,
                    p=self.params.smoothing_polynomial_order
                )

                # Restore NaN positions (smoothed values at gaps are
                # interpolation artifacts, not real data)
                if nan_mask.any():
                    trj.loc[nan_mask, 'x'] = np.nan
                    trj.loc[nan_mask, 'y'] = np.nan
            except Exception:
                print(f"    Note: Smoothing failed for fish {fish_idx}, using raw trajectory")
                smoothing_failed = True

        return FishTrajectory(
            fish_id=fish_idx,
            identity_label=self.file.metadata.identity_labels[fish_idx],
            trajectory=trj,
            n_valid_frames=n_valid,
            n_total_frames=n_total,
            smoothing_failed=smoothing_failed,
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
    - Turning bias (laterality index, left/right turn counts)
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
            fish.failed_metrics.append("derivatives")
            speed_series = np.full(len(trj), np.nan)

        # Where the tracker actually had this fish. Every run-length metric
        # below is computed inside these spans and never across them.
        x, y = trj['x'].values, trj['y'].values
        segments = contiguous_tracked_segments(x, y)

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

        # Tracking quality — the context every metric below has to be read in
        fish.metrics.update(self._calc_tracking_quality(x, y, frame_rate))

        # Freeze analysis
        fish.metrics.update(
            self._calc_freeze_metrics(speed_series, frame_rate, segments)
        )

        # Turning bias and path straightness both catch their own exceptions
        # and return NaN. They flag it with a private '_failed' key, which is
        # stripped here into fish.failed_metrics so the export can distinguish
        # "this fish never turned" from "the calculation raised" (B10).
        for name, result in (
            ("turning", self._calc_movement_direction_metrics(
                trj, speed_series, frame_rate)),
            ("straightness", self._calc_path_straightness(trj, frame_rate)),
        ):
            if result.pop('_failed', False):
                fish.failed_metrics.append(name)
            fish.metrics.update(result)

        return fish

    # =========================================================================
    # DISTANCE
    # =========================================================================

    def _calc_total_distance(self, trj: TrajaDataFrame) -> float:
        """Total path length (sum of all step lengths).

        Steps that span a tracking gap are NaN and drop out of the sum, so this
        is distance *observed* rather than distance travelled — it under-counts
        by however far the fish moved while untracked, and never invents a
        straight-line jump across the gap. Read it next to tracked_fraction.
        """
        return traja.length(trj)

    def _calc_net_displacement(self, trj: TrajaDataFrame) -> float:
        """Straight-line distance from the first tracked point to the last.

        Not traja.distance(), which reads row 0 and row -1 unconditionally and
        so returned NaN for any fish whose recording began or ended in a
        tracking gap -- 12 of the 24 fish in the supplied sessions (B16).
        """
        x, y = trj['x'].values, trj['y'].values
        first, last = first_and_last_tracked(x, y)
        if first < 0 or first == last:
            return np.nan
        return float(np.hypot(x[last] - x[first], y[last] - y[first]))

    # =========================================================================
    # SPEED
    # =========================================================================

    def _calc_speed_metrics(self, speed_series: np.ndarray) -> Dict[str, float]:
        """Summary statistics for speed from pre-computed derivatives.

        TOP SPEED IS A PERCENTILE, NOT A MAXIMUM.
        The old `max_speed` was a raw np.max, which on real recordings reports
        the worst tracking glitch rather than the fastest swim: identity swaps
        and re-acquisitions teleport a fish across the arena in one frame. The
        supplied sessions gave 32-321 BL/s against a physiological ceiling near
        25 BL/s, 2.4-7.9x each fish's own 99.9th percentile (finding B17).

        `speed_p99` answers the same question -- how fast does this fish swim
        when it is going flat out -- without being settable by a single bad
        frame. The count of implausible samples is reported separately as a
        data-quality signal rather than being silently folded into a
        behavioural number.
        """
        speed = speed_series[np.isfinite(speed_series)]

        if len(speed) == 0:
            return {
                'mean_speed': np.nan,
                'speed_p99': np.nan,
                'std_speed': np.nan,
                'median_speed': np.nan,
                'speed_max_raw': np.nan,
            }

        return {
            'mean_speed': float(np.mean(speed)),
            'speed_p99': float(np.percentile(speed, 99)),
            'std_speed': float(np.std(speed)),
            'median_speed': float(np.median(speed)),
            # Kept out of the export: useful for spotting a bad recording,
            # meaningless as a behavioural readout.
            'speed_max_raw': float(np.max(speed)),
        }

    # =========================================================================
    # TRACKING QUALITY
    # =========================================================================

    def _calc_tracking_quality(self, x: np.ndarray, y: np.ndarray,
                               frame_rate: float) -> Dict[str, Any]:
        """How much of this fish was actually observed.

        Every duration and rate below is over *observed* time, so these two
        numbers are what make them interpretable — and comparable between fish
        that were tracked 99% and 87% of the time.
        """
        n_total = len(x)
        n_tracked = tracked_frame_count(x, y)
        return {
            'tracked_fraction': (n_tracked / n_total) if n_total else 0.0,
            'longest_gap_s': (longest_gap_frames(x, y) / frame_rate
                              if frame_rate else np.nan),
        }

    # =========================================================================
    # FREEZE ANALYSIS
    # =========================================================================

    def _calc_freeze_metrics(self, speed_series: np.ndarray, frame_rate: float,
                             segments: List[tuple]) -> Dict[str, Any]:
        """
        Detect freezing episodes and compute freeze metrics.

        A freeze is a run of at least min_freeze_frames consecutive frames,
        *within one stretch of continuous tracking*, where speed stays below
        rest_speed_threshold.

        WHY SEGMENTS (finding B5)
        -------------------------
        This used to run over the whole speed array with NaN forced to
        "not frozen", which meant every tracking gap severed a freeze run. One
        motionless fish with 5% scattered dropout reported 22 freeze episodes
        instead of 1. Forcing NaN the other way is no better -- then the gap
        itself becomes a freeze. A gap is neither, so runs are now found inside
        segments and gap frames enter no count at all.

        COMPLETE VS CENSORED EPISODES
        -----------------------------
        Once tracking fragments, "how many freezes were there" stops being
        answerable. A freeze run that ends because the tracker lost the fish
        might be one episode or the first half of a longer one -- there is no
        way to tell. Counting such runs as episodes is what produced 22 from a
        fish that froze once; counting them as nothing throws away real data.

        So episodes are split in two:

        - **complete** -- the run begins and ends with an observed transition
          (or at the recording boundary, which is conventional). Its duration
          is a measurement. Only these are counted and averaged.
        - **censored** -- the run touches a tracking gap, so its true extent is
          unknown and its measured duration is only a lower bound. Its frames
          still count toward the time-frozen totals, but it is not counted as
          an episode.

        A fish that froze once through 5% scattered dropout now reports 0
        complete episodes, 22 censored, and ~100% of observed time frozen --
        which is exactly what is known about it.

        DENOMINATORS (finding B6)
        -------------------------
        freeze_fraction_pct used to divide by valid frames while
        freeze_total_duration_s divided by the whole recording, so the two
        disagreed by the dropout fraction with nothing to reconcile them.
        Both are now over *observed* time, and the identity

            freeze_total_duration_s / observed_duration_s * 100
                == freeze_fraction_pct

        holds exactly, because observed_duration_s is returned from here rather
        than recomputed -- it counts the same speed samples the numerator does.
        A reader who wants wall-clock can convert; the honest default is that
        we cannot claim a fish was frozen during frames we could not see.
        """
        threshold = self.params.rest_speed_threshold
        min_frames = self.params.min_freeze_frames
        n_frames = len(speed_series)

        complete: List[int] = []
        censored: List[int] = []
        n_observed = 0

        for seg_start, seg_stop in segments:
            # traja's speed[i] spans frames [i-1, i), so a segment's first
            # frame carries no speed — its predecessor is the gap.
            start, stop = seg_start + 1, seg_stop
            if stop <= start:
                continue

            speeds = speed_series[start:stop]
            # A finite speed here means both endpoint frames were tracked.
            usable = np.isfinite(speeds)
            n_observed += int(np.count_nonzero(usable))

            # Running out of recording is not the same as running into a gap:
            # the first and last episodes of a recording are conventionally
            # counted, an episode interrupted by lost tracking is not.
            opens_at_recording_start = seg_start == 0
            closes_at_recording_end = seg_stop == n_frames

            is_slow = usable & (speeds < threshold)
            for a, b in run_lengths(is_slow):
                if b - a < min_frames:
                    continue
                touches_gap = (
                    (a == 0 and not opens_at_recording_start)
                    or (b == stop - start and not closes_at_recording_end)
                )
                (censored if touches_gap else complete).append(b - a)

        total_freeze_frames = sum(complete) + sum(censored)

        return {
            'freeze_count': len(complete),
            'freeze_episodes_censored': len(censored),
            'freeze_total_duration_s': total_freeze_frames / frame_rate,
            'freeze_mean_duration_s': (
                (sum(complete) / len(complete) / frame_rate) if complete else 0.0
            ),
            'freeze_fraction_pct': (
                (total_freeze_frames / n_observed * 100) if n_observed > 0 else 0.0
            ),
            # Returned from here, not recomputed elsewhere, so the identity
            # above cannot drift.
            'observed_duration_s': n_observed / frame_rate if frame_rate else np.nan,
        }

    # =========================================================================
    # TURNING BIAS
    # =========================================================================

    def _calc_movement_direction_metrics(self, trj: TrajaDataFrame,
                                          speed_series: np.ndarray,
                                          frame_rate: float) -> Dict[str, float]:
        """
        Calculate turning bias (laterality) from frame-to-frame heading.

        - Laterality index: (right turns - left turns) / total turns.
          Ranges from -1 (all left/CCW) to +1 (all right/CW). Near 0 = no bias.
        - n_right_turns / n_left_turns: the raw counts behind that ratio.

        Turns are only counted where the fish is actively moving (speed >
        rest_speed_threshold), which makes this work for both adult continuous
        swimming and larval bout-based locomotion (dart-glide).

        WHY ONLY DIRECTIONS, NOT RATES:
        This function used to also return mean angular velocity, erratic
        movement counts, cumulative heading change and mean signed angular
        velocity. Audit B removed all four. They are magnitudes derived from
        the same frame-to-frame arctan2, and at this tracker's noise level the
        magnitude is noise: a straight-line swimmer with 0.5 px of centroid
        jitter reported 271 deg/s, and cumulative heading flipped sign for 4 of
        24 real fish when smoothing was toggled.

        The counts survive that noise because it is symmetric -- it adds turns
        to the left and right in equal measure, so the ratio stays near zero
        instead of running away. Verified against left- and right-biased
        synthetic circlers in tests/test_metric_correctness.py.

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

            # Only consider frames where the fish is actually moving
            # speed_series is len(trj), dheading_deg is len(trj)-2
            # Align: dheading[i] corresponds to the turn between step i and step i+1,
            # which happens around frame i+1. Use speed at frame i+1.
            min_len = min(len(dheading_deg), len(speed_series) - 1)
            if min_len <= 0:
                return self._empty_direction_metrics()

            dheading_signed = dheading_deg[:min_len]
            speed_aligned = speed_series[1:min_len + 1]

            # Mask: fish must be moving and values must be valid
            moving_mask = (
                (speed_aligned > self.params.rest_speed_threshold) &
                ~np.isnan(speed_aligned) &
                ~np.isnan(dheading_signed)
            )

            if int(np.sum(moving_mask)) == 0:
                return self._empty_direction_metrics()

            # --- Turning bias / laterality ---
            # Convention: positive dheading = counterclockwise (left turn in standard coords)
            #             negative dheading = clockwise (right turn)
            # We define laterality index as (right - left) / total,
            # so CW-biased fish → positive, CCW-biased → negative.
            #
            # This holds only because the camera looks down at the tank. A
            # bottom-mounted camera or a mirror rig inverts it, and nothing in
            # the idtracker.ai file records the viewing geometry.
            signed_turns = dheading_signed[moving_mask]
            n_right = int(np.sum(signed_turns < 0))  # CW = right
            n_left = int(np.sum(signed_turns > 0))    # CCW = left
            n_turns = n_right + n_left
            laterality_index = (
                (n_right - n_left) / n_turns if n_turns > 0 else 0.0
            )

            return {
                'laterality_index': float(laterality_index),
                'n_right_turns': n_right,
                'n_left_turns': n_left,
            }

        except Exception as e:
            print(f"Warning: Direction metrics failed: {e}")
            return self._empty_direction_metrics(failed=True)

    def _empty_direction_metrics(self, failed: bool = False) -> Dict[str, Any]:
        """Direction metrics when there is nothing to measure.

        ``failed=True`` marks the difference between a fish that never moved --
        a real result, correctly zero -- and a computation that raised. Without
        it both look identical in the CSV (finding B10).
        """
        return {
            'laterality_index': np.nan,
            'n_right_turns': np.nan if failed else 0,
            'n_left_turns': np.nan if failed else 0,
            '_failed': failed,
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
            return {'mean_path_straightness': np.nan, '_failed': True}


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

    # Carry the exclusions onto the file so the exporter can emit a row for
    # every fish the recording contained, not only the ones that survived.
    loaded_file.excluded_fish = dict(processor.excluded)

    calculator = MetricsCalculator(params)
    for fish in fish_list:
        calculator.calculate_all_metrics(fish)

    return fish_list
