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

from .segments import (contiguous_tracked_segments, run_lengths,
                       tracked_frame_count)


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

    heading_lookback_frames: number of frames to accumulate before/after a
        bout when estimating the approach and departure heading. A larger
        value gives a more stable direction estimate by averaging over more
        displacement, at the cost of capturing turns that happen close
        together. At 30fps, 5 frames = ~167ms, 10 frames = ~333ms.
        Typical: 5–10 frames.
    """
    speed_threshold: float = 0.5    # BL/s
    merge_gap_frames: int = 2       # frames
    min_bout_frames: int = 1        # frames
    heading_lookback_frames: int = 5  # frames to accumulate for pre/post heading

    def validate(self):
        if self.speed_threshold <= 0:
            raise ValueError(f"Speed threshold must be positive, got {self.speed_threshold}")
        if self.merge_gap_frames < 0:
            raise ValueError(f"Merge gap must be non-negative, got {self.merge_gap_frames}")
        if self.min_bout_frames < 1:
            raise ValueError(f"Min bout frames must be >= 1, got {self.min_bout_frames}")
        if self.heading_lookback_frames < 1:
            raise ValueError(f"Heading look-back must be >= 1, got {self.heading_lookback_frames}")


@dataclass
class Bout:
    """A single detected swim bout.

    ``censored`` marks a bout that ran into a tracking gap rather than ending
    because the fish slowed down. Its measured duration and displacement are
    lower bounds, not measurements, so it is excluded from the duration and
    interval statistics — see BoutDetector.detect_bouts.
    """
    start_frame: int
    end_frame: int          # exclusive
    duration_s: float
    peak_speed: float       # BL/s
    mean_speed: float       # BL/s
    displacement: float     # BL (straight line start to end)
    distance: float         # BL (total path)
    heading_change_deg: float  # signed degrees (+ = CCW/left, - = CW/right)
    censored: bool = False
    segment_index: int = 0  # which continuously-tracked stretch it came from


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

        Bouts are found **inside stretches of continuous tracking** and never
        across them. This module used to set NaN speed to 0.0, which made a
        tracking gap read as though the fish had stopped: it split one bout in
        two and inserted a fake pause between them. On the four supplied
        recordings, 18-86% of exported inter-bout intervals were gaps rather
        than behaviour (finding B4).

        A bout that begins or ends at a gap is flagged ``censored`` — the fish
        may well have kept swimming while untracked, so its duration is a lower
        bound. Bouts at the very start or end of the recording are *not*
        censored; running out of recording is a different thing from losing the
        fish, and counting the first and last episodes is conventional.

        Parameters
        ----------
        x, y : array of positions in calibrated units (BL). NaN marks an
            untracked frame; this is what the segment boundaries come from.
        speed : array of speed in BL/s, forward-difference indexed so that
            ``speed[i]`` covers frames ``[i, i+1)``. Length ``len(x) - 1``.

        Returns
        -------
        List of Bout objects, in frame order.
        """
        threshold = self.params.speed_threshold
        n_frames = len(x)
        bouts: List[Bout] = []

        segments = contiguous_tracked_segments(x, y)
        for seg_index, (seg_start, seg_stop) in enumerate(segments):
            # np.diff's speed[i] spans frames [i, i+1), so a segment's last
            # frame carries no speed — its successor is the gap.
            lo, hi = seg_start, seg_stop - 1
            if hi <= lo:
                continue

            seg_speed = speed[lo:hi]
            is_active = np.isfinite(seg_speed) & (seg_speed > threshold)

            intervals = run_lengths(is_active)
            intervals = self._merge_intervals(intervals,
                                              self.params.merge_gap_frames)
            intervals = [(s, e) for s, e in intervals
                         if (e - s) >= self.params.min_bout_frames]

            opens_at_recording_start = seg_start == 0
            closes_at_recording_end = seg_stop == n_frames

            for s, e in intervals:
                censored = (
                    (s == 0 and not opens_at_recording_start)
                    or (e == hi - lo and not closes_at_recording_end)
                )
                bout = self._compute_bout_metrics(x, y, speed, lo + s, lo + e)
                if bout is not None:
                    bout.censored = censored
                    bout.segment_index = seg_index
                    bouts.append(bout)

        return bouts

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
        Compute total signed heading change during a bout, or NaN.

        For 1-frame bouts (where the bout itself has only two position samples,
        giving a single displacement vector with no internal turn to measure),
        compare the cumulative displacement over params.heading_lookback_frames
        before the bout to the equivalent window after it, giving a stable
        heading estimate even when adjacent frames are noisy.

        For longer bouts, sum frame-to-frame heading changes, skipping any
        step whose displacement is below _MIN_DISP (tracking noise).

        RETURNS NaN WHEN THE TURN CANNOT BE MEASURED (finding B8).
        There are five ways that happens: no room for the look-back window, a
        tracking gap inside it, or every step too short to carry a direction.
        All five used to return a literal 0.0, which then landed inside the
        +/-5 degree "straight" dead zone in compute_summary -- so an
        unmeasurable bout was reported as a measured straight one. On the four
        supplied recordings that was 490 of 2,866 bouts: 77% of everything
        counted as straight was a guard return rather than a measurement.
        """
        lookback = self.params.heading_lookback_frames
        _MIN_DISP = 0.05       # BL — below this, displacement direction is unreliable

        seg_x = x[start:end + 1]
        seg_y = y[start:end + 1]

        if len(seg_x) < 3:
            # 1-frame bouts: compare approach heading to departure heading.
            # Accumulate displacement over up to lookback frames for stability.
            pre = max(0, start - lookback)
            post = min(len(x) - 1, end + lookback)

            if pre == start or post == end:
                return float('nan')

            dx_pre = x[start] - x[pre]
            dy_pre = y[start] - y[pre]
            dx_post = x[post] - x[end]
            dy_post = y[post] - y[end]

            if any(np.isnan([dx_pre, dy_pre, dx_post, dy_post])):
                return float('nan')
            if (np.hypot(dx_pre, dy_pre) < _MIN_DISP
                    or np.hypot(dx_post, dy_post) < _MIN_DISP):
                return float('nan')

            h_pre = np.arctan2(dy_pre, dx_pre)
            h_post = np.arctan2(dy_post, dx_post)
            dh = (h_post - h_pre + np.pi) % (2 * np.pi) - np.pi
            return float(np.degrees(dh))

        # For longer bouts, sum frame-to-frame heading changes.
        # Filter out near-zero displacement steps — their heading is dominated
        # by tracking noise and would corrupt the accumulated turn angle.
        dx = np.diff(seg_x)
        dy = np.diff(seg_y)
        disp = np.hypot(dx, dy)
        valid = (~np.isnan(dx)) & (~np.isnan(dy)) & (disp >= _MIN_DISP)

        if np.sum(valid) < 2:
            # Not enough clean steps to sum heading changes — fall back to the
            # lookback approach used for 1-frame bouts: compare total approach
            # and departure vectors around the bout.
            pre = max(0, start - lookback)
            post = min(len(x) - 1, end + lookback)
            if pre == start or post == end:
                return float('nan')
            dx_pre = x[start] - x[pre]
            dy_pre = y[start] - y[pre]
            dx_post = x[post] - x[end]
            dy_post = y[post] - y[end]
            if any(np.isnan([dx_pre, dy_pre, dx_post, dy_post])):
                return float('nan')
            if (np.hypot(dx_pre, dy_pre) < _MIN_DISP
                    or np.hypot(dx_post, dy_post) < _MIN_DISP):
                return float('nan')
            h_pre = np.arctan2(dy_pre, dx_pre)
            h_post = np.arctan2(dy_post, dx_post)
            dh = (h_post - h_pre + np.pi) % (2 * np.pi) - np.pi
            return float(np.degrees(dh))

        headings = np.arctan2(dy[valid], dx[valid])
        dh = np.diff(headings)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        return float(np.degrees(np.sum(dh)))

    def inter_bout_intervals_s(self, bouts: List[Bout]) -> np.ndarray:
        """Gaps between consecutive bouts, in seconds.

        Only pairs that sit in the **same** continuously-tracked stretch count.
        An interval spanning a tracking gap is not an inter-bout interval — it
        is the tracker blinking — and including them is what made 18-86% of
        this column tracking artefact (finding B4).
        """
        return np.array([
            (nxt.start_frame - cur.end_frame) / self.frame_rate
            for cur, nxt in zip(bouts, bouts[1:])
            if cur.segment_index == nxt.segment_index
        ])

    def compute_summary(self, bouts: List[Bout],
                        observed_duration_s: float) -> Dict[str, Any]:
        """Summary statistics from a list of bouts.

        Censored bouts (those cut short by a tracking gap) are excluded from
        every duration, speed and shape statistic, because their measured
        extent is a lower bound rather than a measurement. They are reported
        as their own count so the reader can see how much was set aside.

        ``observed_duration_s`` is tracked time, not wall-clock: a bout rate
        must be per unit of time the fish was actually visible, or it silently
        reports worse-tracked recordings as less active.
        """
        measurable = [b for b in bouts if not b.censored]
        n_censored = len(bouts) - len(measurable)

        if not measurable:
            return {
                'bout_count': 0,
                'bout_censored': n_censored,
                'bout_rate_per_min': 0.0,
                'bout_duration_median_ms': np.nan,
                'bout_duration_iqr_ms': (np.nan, np.nan),
                'ibi_n': 0,
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
                'bout_n_heading_unmeasurable': 0,
            }

        durations_ms = np.array([b.duration_s * 1000 for b in measurable])
        peak_speeds = np.array([b.peak_speed for b in measurable])
        displacements = np.array([b.displacement for b in measurable])
        distances = np.array([b.distance for b in measurable])
        heading_changes = np.array([b.heading_change_deg for b in measurable])

        # Inter-bout intervals, from the full list so a censored bout can still
        # bound an interval — but never across a tracking gap.
        ibis_ms = self.inter_bout_intervals_s(bouts) * 1000

        # Laterality from per-bout heading changes.
        # A 5-degree dead zone keeps noise from counting as a turn. NaN means
        # the turn could not be measured at all, which is neither a turn nor a
        # straight line, so those bouts are counted separately rather than
        # being swept into n_straight (finding B8).
        dead_zone = 5.0
        measured = heading_changes[np.isfinite(heading_changes)]
        n_unmeasurable = int(len(heading_changes) - len(measured))
        n_right = int(np.sum(measured < -dead_zone))
        n_left = int(np.sum(measured > dead_zone))
        n_straight = int(np.sum(np.abs(measured) <= dead_zone))
        n_turns = n_right + n_left
        laterality_index = (
            (n_right - n_left) / n_turns if n_turns > 0 else 0.0
        )

        return {
            'bout_count': len(measurable),
            'bout_censored': n_censored,
            'bout_rate_per_min': (len(measurable) / observed_duration_s * 60
                                  if observed_duration_s > 0 else 0.0),
            'bout_duration_median_ms': float(np.median(durations_ms)),
            'bout_duration_iqr_ms': (
                float(np.percentile(durations_ms, 25)),
                float(np.percentile(durations_ms, 75)),
            ),
            'ibi_n': int(len(ibis_ms)),
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
            'bout_heading_change_mean_abs_deg': (
                float(np.mean(np.abs(measured))) if len(measured) else np.nan
            ),
            'bout_laterality_index': float(laterality_index),
            'bout_n_left': n_left,
            'bout_n_right': n_right,
            'bout_n_straight': n_straight,
            'bout_n_heading_unmeasurable': n_unmeasurable,
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

        # NaN speed stays NaN. It used to be zeroed here, which told the
        # detector the fish had stopped when in fact the tracker had lost it
        # (finding B4); detect_bouts now excludes those frames by segment.
        observed_duration_s = tracked_frame_count(x, y) / frame_rate

        bouts = detector.detect_bouts(x, y, speed)
        summary = detector.compute_summary(bouts, observed_duration_s)
        ibis = detector.inter_bout_intervals_s(bouts)

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
