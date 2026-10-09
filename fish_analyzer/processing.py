"""
fish_analyzer/processing.py
===========================
Layer 2: Individual Trajectory Processing and Metrics Calculation

This module transforms raw pixel data into clean, calibrated trajectories
and calculates individual behavioral metrics.

KEY TRANSFORMATIONS:
1. Flip Y axis (video Y=0 is top, we want Y=0 at bottom)
2. Apply calibration (pixels → body lengths)
3. Calculate derivatives (speed, acceleration)
4. Compute behavioral metrics

LIGHT SMOOTHING (2026-10-09):
Positions get a centred running mean over smoothing_seconds (0.17 s, five
frames at 30 fps) before any metric is computed, inside each stretch of
continuous tracking only. An earlier version dropped smoothing because it
moved the metrics by about 1%; that was measured on fish swimming at normal
speed, where the tracker's ~1 px centroid wobble barely matters. It matters
for slow and still fish. Simulated at a 75 px body length, unsmoothed, 0.5-1
px of wobble makes a motionless fish read 0.3-0.7 BL/s -- right at the freeze
threshold, so it was frozen 3-64% of the time instead of 100% -- inflates the
distance of a fish swimming 0.5 BL/s by 17-67%, and drops path straightness
towards 0. Five frames removes nearly all of that; a fish swimming 2.5 BL/s
keeps its median speed, and its distance moves by about 2% (sharp turns are
cut slightly short).

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
head direction (see extras/head_detection/), not better arithmetic.

Laterality survives because it counts turn *directions*, which stay balanced
under symmetric noise, rather than turn *rates*, which do not. See
docs/audit/AUDIT_B_CORRECTNESS.md B1, B3 and B15, and tests/test_metric_correctness.py.
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
                       smooth_within_segments, tracked_frame_count)


@dataclass
class ProcessingParameters:
    """
    Parameters controlling how trajectories are processed.

    Speeds are in the session's unit per second (BL/s or cm/s); the GUI
    converts them when the unit changes.

    QUALITY GATE:
    A fish tracked in fewer than min_valid_percentage of frames is excluded,
    and listed in the export with its tracked share, rather than averaged in
    beside fish that were seen all the time.

    SMOOTHING:
    smoothing_seconds of centred running mean on positions, never across a
    tracking gap. 0 turns it off. See the module docstring for why.

    FREEZE DETECTION:
    Fish are considered "frozen" (immobile) when their speed stays below
    rest_speed_threshold for at least min_freeze_seconds. 1 s is the usual
    adult criterion; anything much shorter counts slow swimming as freezing.
    A tracking gap of up to freeze_bridge_seconds does not end a freeze if
    the fish is in the same place on both sides of it.

    TURNING BIAS:
    rest_speed_threshold doubles as the gate on turn counting: a heading
    change is only counted when the fish is actually moving, so a stationary
    fish contributes nothing.

    PATH STRAIGHTNESS:
    Computed over a sliding window (straightness_window_seconds). For each
    window: straightness = net displacement / path distance. Values near 1.0
    indicate straight swimming; values near 0 indicate circling or meandering.
    Only windows in which the fish averaged at least straightness_min_speed
    count: in a slow or still window, what is left of the tracker's wobble
    dominates the path and straightness says nothing about how it swims.
    """
    min_valid_points: int = 10
    min_valid_percentage: float = 0.80
    rest_speed_threshold: float = 0.5       # unit/s below which fish is "frozen"
    min_freeze_seconds: float = 1.0         # Shortest stillness that counts as a freeze
    freeze_bridge_seconds: float = 0.5      # Longest tracking gap a freeze may span
    straightness_window_seconds: float = 1.0  # Window for path straightness calculation
    straightness_min_speed: float = 1.0     # unit/s a window must average to count
    smoothing_seconds: float = 0.17         # Running mean on positions; 0 = off

    def validate(self):
        """Check that all parameters are valid. Raises ValueError if not."""
        if not 0 < self.min_valid_percentage <= 1.0:
            raise ValueError(f"Valid percentage must be between 0 and 1, got {self.min_valid_percentage}")
        if self.rest_speed_threshold < 0:
            raise ValueError(f"Rest threshold must be non-negative, got {self.rest_speed_threshold}")
        if self.min_freeze_seconds <= 0:
            raise ValueError(f"Minimum freeze length must be positive, got {self.min_freeze_seconds}")
        if self.freeze_bridge_seconds < 0:
            raise ValueError(f"Freeze gap bridging must be zero or more, got {self.freeze_bridge_seconds}")
        if self.straightness_window_seconds <= 0:
            raise ValueError(f"Straightness window must be positive, got {self.straightness_window_seconds}")
        if self.straightness_min_speed < 0:
            raise ValueError(f"Straightness speed must be non-negative, got {self.straightness_min_speed}")
        if self.smoothing_seconds < 0:
            raise ValueError(f"Smoothing must be zero or more, got {self.smoothing_seconds}")

    def freeze_frames(self, frame_rate: float) -> int:
        """min_freeze_seconds as a whole number of frames, at least 1."""
        return max(1, int(round(self.min_freeze_seconds * frame_rate)))

    def smoothing_frames(self, frame_rate: float) -> int:
        """smoothing_seconds as an odd number of frames; 1 means off."""
        frames = int(round(self.smoothing_seconds * frame_rate))
        return frames + 1 if frames % 2 == 0 else frames

    def straightness_window_frames(self, frame_rate: float) -> int:
        return max(2, int(self.straightness_window_seconds * frame_rate))

    def export_columns(self) -> Dict[str, float]:
        """The settings every export carries, so a number in a CSV can be
        traced back to how it was made. Speeds are in the row's Unit/s."""
        return {
            'Setting_MinTracked_pct': round(self.min_valid_percentage * 100, 1),
            'Setting_Smoothing_s': self.smoothing_seconds,
            'Setting_FreezeSpeed': self.rest_speed_threshold,
            'Setting_FreezeMin_s': self.min_freeze_seconds,
            'Setting_FreezeBridgesGapsUpTo_s': self.freeze_bridge_seconds,
            'Setting_StraightnessMinSpeed': self.straightness_min_speed,
        }

    @classmethod
    def default_for_fish(cls) -> 'ProcessingParameters':
        """Get default parameters that work well for fish tracking."""
        return cls()


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

    This class handles the coordinate transformations. It creates
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

        for fish_idx in range(self.file.n_fish):
            try:
                fish_traj = self._process_single_fish(fish_idx)
                if fish_traj is not None:
                    processed_fish.append(fish_traj)
                    print(f"  Fish {fish_idx}: [ok] ({fish_traj.valid_percentage:.1%} valid data)")
                else:
                    self.excluded[fish_idx] = self._last_exclusion
                    print(f"  Fish {fish_idx}: [skip] ({self._last_exclusion})")
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
        transformed_coords[:, 0], transformed_coords[:, 1] = smooth_within_segments(
            transformed_coords[:, 0], transformed_coords[:, 1],
            self.params.smoothing_frames(self.file.calibration.frame_rate))
        df = self._create_dataframe_with_time(transformed_coords)

        valid_mask = ~(df['x'].isna() | df['y'].isna())
        n_valid = valid_mask.sum()
        n_total = len(df)
        valid_pct = n_valid / n_total if n_total > 0 else 0

        if n_valid < self.params.min_valid_points:
            self._last_exclusion = f"only {n_valid} tracked frames"
            return None
        if valid_pct < self.params.min_valid_percentage:
            self._last_exclusion = (
                f"tracked {valid_pct:.0%} of frames, below the "
                f"{self.params.min_valid_percentage:.0%} minimum")
            return None

        trj = TrajaDataFrame(df)
        trj.fps = self.file.calibration.frame_rate
        trj.spatial_units = self.file.calibration.unit_name
        trj.time_units = "s"

        return FishTrajectory(
            fish_id=fish_idx,
            identity_label=self.file.metadata.identity_labels[fish_idx],
            trajectory=trj,
            n_valid_frames=n_valid,
            n_total_frames=n_total,
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
            self._calc_freeze_metrics(speed_series, frame_rate, segments, x, y)
        )

        # Distance as a rate over the time the fish was actually seen, so a
        # fish tracked 85% of the time is not read as 15% less active than
        # the same fish tracked throughout. Same denominator as the freeze
        # fraction: the frames that carry a speed.
        observed_min = fish.metrics['observed_duration_s'] / 60.0
        fish.metrics['distance_per_tracked_min'] = (
            fish.metrics['total_distance'] / observed_min
            if observed_min > 0 else np.nan)

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
                             segments: List[tuple], x: np.ndarray,
                             y: np.ndarray) -> Dict[str, Any]:
        """
        Detect freezing episodes and compute freeze metrics.

        A freeze is a run of at least min_freeze_seconds of consecutive frames,
        *within one stretch of continuous tracking* (short still gaps aside,
        see below), where speed stays below rest_speed_threshold.

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

        SHORT GAPS INSIDE A FREEZE (2026-10-09)
        ---------------------------------------
        With a 1 s minimum, a freeze found strictly inside segments cannot
        survive scattered dropout: a still fish losing one frame in twenty has
        no 1 s stretch of continuous tracking, and read as ~50% frozen instead
        of ~100%. A poorly tracked group would then look like it freezes less.

        So a gap of up to freeze_bridge_seconds is bridged when the fish is in
        the same place on both sides (moving slower than the freeze threshold
        across it): it cannot have swum away and back in that time. A bridged
        gap joins the runs on either side into one episode. Its frames count
        toward the episode's *length* -- the episode did last that long -- but
        not toward time frozen, which stays over observed frames only so the
        identity below still holds. A longer gap, or one the fish moved
        across, still ends the run and censors it.

        A fish that froze once through 5% scattered dropout now reports 1
        complete episode and ~100% of observed time frozen.

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
        min_frames = self.params.freeze_frames(frame_rate)
        max_bridge = int(round(self.params.freeze_bridge_seconds * frame_rate))
        n_frames = len(speed_series)

        # Join segments across a short gap when the fish is in the same place
        # on both sides: it cannot have swum off and back in that time.
        chains: List[List[tuple]] = []
        for seg in segments:
            if chains:
                prev_stop, start = chains[-1][-1][1], seg[0]
                gap = start - prev_stop
                last, first = prev_stop - 1, start
                still = (np.hypot(x[first] - x[last], y[first] - y[last])
                         / ((first - last) / frame_rate)) < threshold
                if gap <= max_bridge and still:
                    chains[-1].append(seg)
                    continue
            chains.append([seg])

        complete: List[int] = []
        censored: List[int] = []
        total_freeze_frames = 0
        n_observed = 0

        for chain in chains:
            seg_start, seg_stop = chain[0][0], chain[-1][1]
            # traja's speed[i] spans frames [i-1, i), so a segment's first
            # frame carries no speed — its predecessor is the gap.
            start, stop = seg_start + 1, seg_stop
            if stop <= start:
                continue

            speeds = speed_series[start:stop]
            # A finite speed here means both endpoint frames were tracked.
            usable = np.isfinite(speeds)
            n_observed += int(np.count_nonzero(usable))
            # The speed samples that fall in a bridged gap: neither slow nor
            # fast, but they do not break a run either.
            bridged = np.zeros(len(speeds), dtype=bool)
            for (_, gap_from), (gap_to, _) in zip(chain[:-1], chain[1:]):
                bridged[gap_from - start:gap_to - start + 1] = True

            # Running out of recording is not the same as running into a gap:
            # the first and last episodes of a recording are conventionally
            # counted, an episode interrupted by lost tracking is not.
            opens_at_recording_start = seg_start == 0
            closes_at_recording_end = seg_stop == n_frames

            is_slow = usable & (speeds < threshold)
            for a, b in run_lengths(is_slow | bridged):
                frozen = int(np.count_nonzero(is_slow[a:b]))
                if b - a < min_frames or frozen == 0:
                    continue
                touches_gap = (
                    (a == 0 and not opens_at_recording_start)
                    or (b == stop - start and not closes_at_recording_end)
                )
                total_freeze_frames += frozen
                (censored if touches_gap else complete).append(b - a)

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
            window_frames = self.params.straightness_window_frames(frame_rate)

            _, straightness_values, n_tracked = straightness_windows(
                x, y, window_frames, frame_rate, self.params.straightness_min_speed)

            used_pct = (len(straightness_values) / n_tracked * 100
                        if n_tracked else np.nan)
            if len(straightness_values) == 0:
                return {'mean_path_straightness': np.nan,
                        'straightness_windows_used_pct': used_pct}

            return {
                'mean_path_straightness': float(np.mean(straightness_values)),
                'straightness_windows_used_pct': used_pct,
            }

        except Exception as e:
            print(f"Warning: Path straightness calculation failed: {e}")
            return {'mean_path_straightness': np.nan,
                    'straightness_windows_used_pct': np.nan, '_failed': True}


def straightness_windows(x: np.ndarray, y: np.ndarray, window_frames: int,
                         frame_rate: float, min_speed: float = 0.0):
    """Path straightness in each half-overlapping window of `window_frames`.

    Returns (index of each kept window's first frame, its net displacement /
    distance swum, how many windows were fully tracked).

    A window with an untracked frame has no straightness and is left out. So
    is one in which the fish averaged less than `min_speed` (unit/s): there
    the tracker's residual wobble is a large part of the path, and the ratio
    measures that rather than the fish. The third value is the denominator
    for reporting how many tracked windows were swimming fast enough to use.
    """
    starts, values = [], []
    n_tracked = 0
    min_path = min_speed * (window_frames - 1) / frame_rate
    for start in range(0, len(x) - window_frames + 1, max(1, window_frames // 2)):
        x_win = x[start:start + window_frames]
        y_win = y[start:start + window_frames]
        if np.any(np.isnan(x_win)) or np.any(np.isnan(y_win)):
            continue
        n_tracked += 1
        net_disp = np.hypot(x_win[-1] - x_win[0], y_win[-1] - y_win[0])
        path_dist = np.sum(np.hypot(np.diff(x_win), np.diff(y_win)))
        if path_dist > 0 and path_dist >= min_path:
            starts.append(start)
            values.append(net_disp / path_dist)
    return np.asarray(starts, dtype=int), np.asarray(values, dtype=float), n_tracked


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
    # And the settings, so every export can say how its numbers were made.
    loaded_file.processing_params = params

    calculator = MetricsCalculator(params)
    for fish in fish_list:
        calculator.calculate_all_metrics(fish)

    return fish_list
