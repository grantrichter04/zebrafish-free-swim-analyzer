"""Tests for fish_analyzer.segments — the shared tracking-gap boundary.

Audit B found the same defect in three places: processing.py treated a
tracking gap as "moving", bout_analysis.py treated it as "still", and neither
excluded it from a denominator. This module is the single answer both now
call, so these tests are the specification for all three.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fish_analyzer.segments import (  # noqa: E402
    backward_speed_slices, contiguous_tracked_segments, forward_speed_slices,
    longest_gap_frames, run_lengths, tracked_frame_count)

NAN = np.nan


def xy(pattern):
    """'..X..' -> coordinate arrays where X marks an untracked frame."""
    x = np.array([NAN if c == "X" else float(i)
                  for i, c in enumerate(pattern)])
    return x, x.copy()


# -----------------------------------------------------------------------------
# contiguous_tracked_segments
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("pattern,expected", [
    (".....", [(0, 5)]),                       # nothing missing
    ("XXXXX", []),                             # nothing tracked
    ("..X..", [(0, 2), (3, 5)]),               # one interior gap
    ("X....", [(1, 5)]),                       # gap at the start
    ("....X", [(0, 4)]),                       # gap at the end
    ("X.X.X", [(1, 2), (3, 4)]),               # alternating
    ("..XX..", [(0, 2), (4, 6)]),              # multi-frame gap
])
def test_segment_boundaries(pattern, expected):
    assert contiguous_tracked_segments(*xy(pattern)) == expected


def test_a_gap_in_either_coordinate_breaks_the_segment():
    """A frame is tracked only if both x and y are finite."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, NAN, 3.0])
    assert contiguous_tracked_segments(x, y) == [(0, 2), (3, 4)]


def test_min_length_filters_out_isolated_frames():
    """A one-frame island yields no usable step, so callers can drop it."""
    assert contiguous_tracked_segments(*xy("X.X..."), min_length=2) == [(3, 6)]


def test_infinities_count_as_untracked():
    x = np.array([0.0, np.inf, 2.0])
    assert contiguous_tracked_segments(x, x.copy()) == [(0, 1), (2, 3)]


def test_empty_input():
    empty = np.array([])
    assert contiguous_tracked_segments(empty, empty) == []


# -----------------------------------------------------------------------------
# Speed-index mapping — the off-by-one this module exists to centralise
# -----------------------------------------------------------------------------

def test_backward_speed_slices_drop_the_first_frame_of_each_segment():
    """traja's speed[i] spans frames [i-1, i), so the first frame of a segment
    has no speed — its predecessor is the gap."""
    assert backward_speed_slices([(0, 5), (10, 14)]) == [(1, 5), (11, 14)]


def test_forward_speed_slices_drop_the_last_frame_of_each_segment():
    """np.diff's speed[i] spans frames [i, i+1), so the last frame of a
    segment has no speed — its successor is the gap."""
    assert forward_speed_slices([(0, 5), (10, 14)]) == [(0, 4), (10, 13)]


@pytest.mark.parametrize("mapper", [backward_speed_slices, forward_speed_slices])
def test_a_one_frame_segment_yields_no_speed_samples(mapper):
    assert mapper([(3, 4)]) == []


@pytest.mark.parametrize("mapper", [backward_speed_slices, forward_speed_slices])
def test_both_mappings_preserve_the_number_of_steps(mapper):
    """A segment of n frames contains exactly n-1 steps either way."""
    for start, stop in [(0, 5), (10, 14), (100, 101)]:
        got = mapper([(start, stop)])
        assert sum(b - a for a, b in got) == max(stop - start - 1, 0)


# -----------------------------------------------------------------------------
# run_lengths
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("mask,expected", [
    ([], []),
    ([False, False], []),
    ([True, True, True], [(0, 3)]),
    ([False, True, True, False], [(1, 3)]),
    ([True, False, True], [(0, 1), (2, 3)]),
    ([False, True], [(1, 2)]),
])
def test_run_lengths_finds_true_runs(mask, expected):
    assert run_lengths(np.array(mask, dtype=bool)) == expected


def test_run_lengths_matches_a_naive_loop():
    rng = np.random.default_rng(0)
    mask = rng.random(500) < 0.4
    naive, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            naive.append((start, i))
            start = None
    if start is not None:
        naive.append((start, len(mask)))
    assert run_lengths(mask) == naive


# -----------------------------------------------------------------------------
# Quality summaries
# -----------------------------------------------------------------------------

def test_tracked_frame_count():
    assert tracked_frame_count(*xy("..X..")) == 4
    assert tracked_frame_count(*xy("XXXXX")) == 0


@pytest.mark.parametrize("pattern,expected", [
    (".....", 0),
    ("..X..", 1),
    (".XXX.", 3),
    ("XX...", 2),
    ("...XX", 2),
    ("XXXXX", 5),
])
def test_longest_gap_frames(pattern, expected):
    assert longest_gap_frames(*xy(pattern)) == expected
