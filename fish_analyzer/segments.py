"""
fish_analyzer/segments.py
=========================
Tracking gaps: one definition, shared by every metric that walks the timeline.

WHY THIS MODULE EXISTS
----------------------
idtracker.ai writes NaN for a frame in which it could not place a fish. That
is not a measurement of the fish — it is an absence of one. Audit B found the
package answering the question three different ways:

- ``processing.py`` set NaN speed to *not frozen*, so a gap broke a freeze run
  into two. One continuously motionless fish with 5% scattered dropout
  reported 22 freeze episodes instead of 1.
- ``bout_analysis.py`` set NaN speed to *0.0*, so a gap read as a pause. On the
  four supplied recordings, 18-86% of exported inter-bout intervals were
  actually tracking gaps.
- Neither excluded gap frames from the denominators, so ``freeze_fraction_pct``
  and ``freeze_total_duration_s`` disagreed by the dropout fraction.

Both conventions are wrong in the same way: they force a gap to mean *some*
behaviour. The only defensible reading is that nothing is known about those
frames, so every run-length metric must be computed *within* stretches of
continuous tracking and never across them.

Fixing that in each module separately is how the two conventions diverged in
the first place, so the rule lives here and both call it.

THE OFF-BY-ONE
--------------
The package derives speed two ways, and they index differently:

- ``traja.get_derivatives()`` gives a **backward** difference: ``speed[i]``
  covers frames ``[i-1, i)``, so the first frame of a segment has no speed.
- ``np.diff`` on positions gives a **forward** difference: ``speed[i]`` covers
  frames ``[i, i+1)``, so the *last* frame of a segment has no speed.

Both are correct; mixing them up shifts every event by a frame.
``backward_speed_slices`` and ``forward_speed_slices`` do that mapping so no
caller has to get it right by hand.

See AUDIT_B_CORRECTNESS.md findings B4, B5 and B6.
"""

from typing import List, Sequence, Tuple

import numpy as np

Span = Tuple[int, int]  # half-open [start, stop)


def _tracked_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True where both coordinates are real numbers.

    Infinities count as untracked: they are not positions, and letting one
    through would poison every distance computed from it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.isfinite(x) & np.isfinite(y)


def run_lengths(mask: np.ndarray) -> List[Span]:
    """Find every maximal run of True as half-open ``[start, stop)`` spans.

    Replaces the three hand-written accumulate-and-reset loops the package had
    (freeze, burst and bout interval detection), which is one reason they could
    drift apart.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    # Pad with False at both ends so every run has a rising and falling edge.
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def contiguous_tracked_segments(x: np.ndarray, y: np.ndarray,
                                min_length: int = 1) -> List[Span]:
    """Split a trajectory into maximal stretches of continuously tracked frames.

    Parameters
    ----------
    x, y : array
        Positions in any units. A frame counts as tracked only if *both* are
        finite.
    min_length : int
        Drop segments shorter than this many frames. Pass 2 when you need at
        least one step (speed, heading); the default keeps everything.

    Returns
    -------
    list of (start, stop)
        Half-open frame spans, in order. Empty if nothing is tracked.
    """
    spans = run_lengths(_tracked_mask(x, y))
    if min_length > 1:
        spans = [(a, b) for a, b in spans if b - a >= min_length]
    return spans


def backward_speed_slices(segments: Sequence[Span]) -> List[Span]:
    """Map frame segments onto a backward-difference speed array.

    ``speed[i]`` spans frames ``[i-1, i)`` (traja's convention), so a segment's
    first frame contributes no speed sample — its predecessor is the gap.
    One-frame segments drop out entirely.
    """
    return [(a + 1, b) for a, b in segments if b > a + 1]


def forward_speed_slices(segments: Sequence[Span]) -> List[Span]:
    """Map frame segments onto a forward-difference speed array.

    ``speed[i]`` spans frames ``[i, i+1)`` (``np.diff``'s convention), so a
    segment's *last* frame contributes no speed sample — its successor is the
    gap. One-frame segments drop out entirely.
    """
    return [(a, b - 1) for a, b in segments if b - 1 > a]


def tracked_frame_count(x: np.ndarray, y: np.ndarray) -> int:
    """How many frames actually hold a position."""
    return int(np.count_nonzero(_tracked_mask(x, y)))


def longest_gap_frames(x: np.ndarray, y: np.ndarray) -> int:
    """Length of the longest untracked run, in frames.

    The headline data-quality number: the supplied recordings reach 188 frames
    (6.3 s), which is long enough that anything interpolated across it is
    invention rather than measurement.
    """
    gaps = run_lengths(~_tracked_mask(x, y))
    return max((b - a for a, b in gaps), default=0)


def first_and_last_tracked(x: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
    """Indices of the first and last tracked frames.

    Returns ``(-1, -1)`` if nothing is tracked. Exists because
    ``traja.distance()`` reads row 0 and row -1 unconditionally, which made
    ``NetDisplacement`` NaN for 12 of the 24 fish in the supplied sessions
    (finding B16).
    """
    tracked = np.flatnonzero(_tracked_mask(x, y))
    if tracked.size == 0:
        return -1, -1
    return int(tracked[0]), int(tracked[-1])
