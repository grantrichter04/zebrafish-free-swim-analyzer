"""Ground-truth tests for every exported behavioural metric (Audit Pass B).

Each test builds a trajectory whose answer is known analytically and checks the
metric returns it. Tests marked ``xfail(strict=True)`` state the value the
metric *should* return; they will start failing — loudly — the moment the
underlying defect is fixed, which is the signal to delete the marker.

Everything is synthetic. No real session data, no network, no disk writes.
See docs/audit/AUDIT_B_CORRECTNESS.md for what each xfail means scientifically.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_tracks import (  # noqa: E402
    BODY_LENGTH_PX, FPS, VIDEO_SIZE, circler,
    jittered_straight_line, make_file, single_swim_event, stationary,
    straight_line, three_fish_fixed_geometry, with_dropout,
    with_scattered_dropout)

from fish_analyzer.data_structures import CalibrationSettings  # noqa: E402
from fish_analyzer.processing import (  # noqa: E402
    ProcessingParameters, process_and_analyze_file)
from fish_analyzer.shoaling import ShoalingCalculator, ShoalingParameters  # noqa: E402
from fish_analyzer.spatial import ArenaDefinition, ThigmotaxisCalculator  # noqa: E402

OMEGA = 0.05  # rad/frame for the circler cases


def metrics_for(traj, params=None, **kwargs):
    """Run the individual-metrics pipeline over one synthetic track."""
    loaded = make_file(traj, **kwargs)
    fish = process_and_analyze_file(
        loaded, params or ProcessingParameters.default_for_fish())
    return fish[0].metrics


# =============================================================================
# Distance, displacement and speed — a straight-line swimmer
# =============================================================================

def test_total_distance_matches_path_length():
    """599 steps of 2 px at 50 px/BL is exactly 23.96 BL of path."""
    m = metrics_for(straight_line(n=600, step_px=2.0))
    assert m["total_distance"] == pytest.approx(599 * 2.0 / BODY_LENGTH_PX)


def test_net_displacement_equals_distance_when_path_is_straight():
    m = metrics_for(straight_line(n=600, step_px=2.0))
    assert m["net_displacement"] == pytest.approx(m["total_distance"])


def test_net_displacement_survives_a_dropout_at_the_ends():
    """B16: traja.distance() read row 0 and row -1 unconditionally, so a gap at
    either end returned NaN — true for 12 of the 24 real fish. It now measures
    between the first and last *tracked* positions, here frames 1 and 598."""
    traj = straight_line(n=600, step_px=2.0)
    traj[0, 0, :] = np.nan
    traj[-1, 0, :] = np.nan
    assert metrics_for(traj)["net_displacement"] == pytest.approx(
        597 * 2.0 / BODY_LENGTH_PX)


def test_net_displacement_is_nan_when_nothing_can_be_measured():
    """One tracked frame gives no displacement to report — and NaN is the
    honest answer, not 0.0, which would read as 'the fish went nowhere'."""
    traj = straight_line(n=600, step_px=2.0)
    traj[1:, 0, :] = np.nan
    params = ProcessingParameters.default_for_fish()
    params.min_valid_points = 1
    params.min_valid_percentage = 0.001   # let the quality gate pass it through
    assert np.isnan(metrics_for(traj, params)["net_displacement"])


def test_top_speed_is_not_set_by_a_single_teleport():
    """B17: a raw max reports the worst tracking glitch in the recording —
    32-321 BL/s on real sessions against a ceiling near 25. speed_p99 answers
    the same question without being settable by one bad frame."""
    traj = straight_line(n=600, step_px=2.0)
    traj[300:, 0, 0] += 900.0  # identity swap across the arena
    m = metrics_for(traj)
    assert m["speed_p99"] == pytest.approx(2.0 / BODY_LENGTH_PX * FPS, rel=0.1)
    # The raw max is kept out of the export but retained for quality checks.
    assert m["speed_max_raw"] > 100


def test_speed_statistics_on_constant_velocity_swimmer():
    """2 px/frame at 50 px/BL and 30 fps is 1.2 BL/s, with no spread."""
    m = metrics_for(straight_line(n=600, step_px=2.0))
    expected = 2.0 / BODY_LENGTH_PX * FPS
    assert m["mean_speed"] == pytest.approx(expected)
    assert m["median_speed"] == pytest.approx(expected)
    assert m["speed_p99"] == pytest.approx(expected)
    assert m["std_speed"] == pytest.approx(0.0, abs=1e-9)


def test_speed_follows_the_calibration_unit():
    """Individual metrics honour calibration.scale_factor. Contrast with the
    group metrics, which do not — see test_group_metrics_ignore_calibration."""
    traj = straight_line(n=600, step_px=2.0)
    cm = CalibrationSettings.from_physical_measurement(10.0, "cm", FPS)
    m = metrics_for(traj, calibration=cm)
    assert m["mean_speed"] == pytest.approx(2.0 / 10.0 * FPS)
    assert m["total_distance"] == pytest.approx(599 * 2.0 / 10.0)


# =============================================================================
# Path straightness
# =============================================================================

def test_path_straightness_is_one_for_a_straight_line():
    m = metrics_for(straight_line(n=600, step_px=2.0))
    assert m["mean_path_straightness"] == pytest.approx(1.0)


def test_path_straightness_matches_the_chord_over_arc_ratio():
    """A 1 s window on a constant-rate circler holds 30 samples = 29 steps,
    so straightness is sin(29w/2) / (29 sin(w/2)). Smoothing off: this pins
    the arithmetic, and smoothing shrinks the circle at its two ends."""
    m = metrics_for(circler(n=600, omega=OMEGA),
                    ProcessingParameters(smoothing_seconds=0))
    expected = np.sin(29 * OMEGA / 2) / (29 * np.sin(OMEGA / 2))
    assert m["mean_path_straightness"] == pytest.approx(expected, rel=1e-9)


# =============================================================================
# Angular velocity and turn direction
# =============================================================================

def test_clockwise_on_screen_reads_as_a_right_turn():
    """Camera above the tank: clockwise on screen is the fish turning right.

    Pins the sign convention asserted in _calc_movement_direction_metrics but
    never tested before this suite.
    """
    m = metrics_for(circler(n=600, omega=OMEGA, clockwise_on_screen=True))
    assert m["laterality_index"] == pytest.approx(1.0)
    assert m["n_right_turns"] > 0
    assert m["n_left_turns"] == 0


def test_counter_clockwise_on_screen_reads_as_a_left_turn():
    m = metrics_for(circler(n=600, omega=OMEGA, clockwise_on_screen=False))
    assert m["laterality_index"] == pytest.approx(-1.0)
    assert m["n_left_turns"] > 0
    assert m["n_right_turns"] == 0


def test_laterality_index_is_unbiased_under_symmetric_jitter():
    """Turn *counts* stay balanced under noise even though the turn *rate*
    does not — which is why laterality survives and angular velocity does not."""
    m = metrics_for(jittered_straight_line(n=600, jitter_px=0.3))
    assert abs(m["laterality_index"]) < 0.1


# -----------------------------------------------------------------------------
# Withdrawn metrics
#
# Audit B found eight columns to be readouts of idtracker.ai centroid noise
# rather than of fish behaviour, and none of them is recoverable from centroid
# data — see docs/audit/AUDIT_B_CORRECTNESS.md B1, B3 and B15. They were removed rather
# than repaired. These two tests exist so they cannot drift back in unnoticed:
# reintroducing any of them needs real head direction from extras/head_detection/,
# plus a deliberate edit here.
# -----------------------------------------------------------------------------

WITHDRAWN_METRIC_KEYS = (
    "mean_angular_velocity_deg_s",
    "erratic_movement_count",
    "erratic_movements_per_min",
    "burst_count",
    "burst_mean_speed",
    "burst_mean_duration_s",
    "burst_frequency_per_min",
    "cumulative_heading_change_deg",
    "mean_signed_angular_velocity_deg_s",
)

WITHDRAWN_CSV_COLUMNS = (
    "MeanAngularVelocity_deg_s",
    "ErraticMovementCount",
    "ErraticMovements_per_min",
    "BurstCount",
    "BurstMeanSpeed",
    "BurstFrequency_per_min",
    "CumulativeHeading_deg",
    "MeanSignedAngVel_deg_s",
)


@pytest.mark.parametrize("key", WITHDRAWN_METRIC_KEYS)
def test_noise_dominated_metrics_are_not_computed(key):
    """A jittering fish that never leaves its spot must not produce turn or
    burst statistics. Before Audit B it reported 4,000 deg/s and 783 erratic
    movements per minute."""
    assert key not in metrics_for(stationary(n=600, jitter_px=0.5))


@pytest.mark.parametrize("exporter", ["combined", "individual"])
def test_noise_dominated_columns_are_not_exported(exporter, tmp_path):
    from fish_analyzer.export import (export_combined_summary_csv,
                                      export_individual_metrics_csv)

    loaded = make_file(jittered_straight_line(n=600, jitter_px=0.5))
    loaded.processed_data = process_and_analyze_file(
        loaded, ProcessingParameters.default_for_fish())

    out = tmp_path / "metrics.csv"
    if exporter == "combined":
        assert export_combined_summary_csv({"f": loaded}, {}, out) == 1
    else:
        assert export_individual_metrics_csv({"f": loaded}, out) == 1

    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert not set(header) & set(WITHDRAWN_CSV_COLUMNS)
    # The columns that survived the audit must still be there.
    for kept in ("TotalDistance", "MeanSpeed", "PathStraightness",
                 "LateralityIndex", "RightTurns_CW", "LeftTurns_CCW"):
        assert kept in header


# =============================================================================
# Freezing
# =============================================================================

def test_a_motionless_fish_is_one_continuous_freeze():
    """Running out of recording is not the same as running into a gap, so the
    single episode is complete rather than censored."""
    m = metrics_for(stationary(n=600, jitter_px=0.0))
    assert m["freeze_count"] == 1
    assert m["freeze_episodes_censored"] == 0
    assert m["freeze_fraction_pct"] == pytest.approx(100.0)


def test_a_swimming_fish_never_freezes():
    m = metrics_for(straight_line(n=600, step_px=2.0))
    assert m["freeze_count"] == 0
    assert m["freeze_total_duration_s"] == pytest.approx(0.0)


def test_three_clean_freezes_are_counted_and_measured():
    """200 frames swimming then 100 frozen, three times over."""
    n = 900
    step = np.zeros(n)
    for k in range(3):
        step[k * 300:k * 300 + 200] = 4.0
    traj = np.stack([100.0 + np.cumsum(step), np.full(n, 500.0)],
                    axis=1)[:, None, :]
    m = metrics_for(traj)
    assert m["freeze_count"] == 3
    assert m["freeze_episodes_censored"] == 0
    assert m["freeze_mean_duration_s"] == pytest.approx(100 / FPS, rel=0.02)


def test_dropout_does_not_inflate_the_freeze_episode_count():
    """B5: NaN speed used to count as not-frozen, so every gap severed the run
    and one motionless fish reported 22 episodes.

    A one-frame gap the fish sits still across cannot hide a swim, so a short
    still gap is bridged and the fish reads as what it did: one freeze.
    """
    traj = with_scattered_dropout(stationary(n=600, jitter_px=0.0), fraction=0.05)
    m = metrics_for(traj)
    assert m["freeze_count"] == 1
    assert m["freeze_episodes_censored"] == 0
    assert m["freeze_fraction_pct"] > 95.0


def test_a_long_gap_still_censors_a_freeze():
    """Past freeze_bridge_seconds nothing is known about the fish, so the runs
    either side are censored rather than joined."""
    traj = with_dropout(stationary(n=600, jitter_px=0.0), 200, 260)   # 2 s
    m = metrics_for(traj)
    assert m["freeze_count"] == 0
    assert m["freeze_episodes_censored"] == 2


def test_a_gap_the_fish_moved_across_is_not_bridged():
    """Still, a 3-frame gap during which it moved 1 BL, still again: two runs,
    each censored at the gap -- not one freeze through a move."""
    traj = stationary(n=600, jitter_px=0.0)
    traj[300:, 0, 0] += BODY_LENGTH_PX
    traj = with_dropout(traj, 299, 302)
    m = metrics_for(traj)
    assert m["freeze_count"] == 0
    assert m["freeze_episodes_censored"] == 2


def test_a_short_stop_is_not_a_freeze():
    """Half a second below the threshold is a pause, not a freeze, at the
    1 s minimum."""
    n = 600
    step = np.full(n, 4.0)
    step[300:315] = 0.0
    traj = np.stack([100.0 + np.cumsum(step), np.full(n, 500.0)], axis=1)[:, None, :]
    assert metrics_for(traj)["freeze_count"] == 0
    longer = ProcessingParameters(min_freeze_seconds=0.3)
    assert metrics_for(traj, longer)["freeze_count"] == 1


def test_smoothing_lets_a_still_fish_with_tracker_wobble_freeze():
    """Unsmoothed, 1 px of centroid wobble reads as ~0.7 BL/s at this body
    length and a motionless fish never drops below 0.5 BL/s for a second."""
    traj = stationary(n=900, jitter_px=1.0)
    raw = metrics_for(traj, ProcessingParameters(smoothing_seconds=0))
    smoothed = metrics_for(traj)
    assert raw["freeze_fraction_pct"] < 20.0
    assert smoothed["freeze_fraction_pct"] > 90.0


def test_smoothing_leaves_steady_swimming_alone():
    traj = straight_line(n=600, step_px=2.0)
    m = metrics_for(traj)
    assert m["total_distance"] == pytest.approx(599 * 2.0 / BODY_LENGTH_PX)
    assert m["median_speed"] == pytest.approx(2.0 * FPS / BODY_LENGTH_PX)


def test_smoothing_does_not_reach_across_a_gap():
    """A fish that vanishes and reappears 2 BL away: the frames beside the gap
    must stay where they were, not be pulled toward the other side."""
    from fish_analyzer.segments import smooth_within_segments
    x = np.r_[np.zeros(20), np.nan, np.full(20, 2.0)]
    y = np.zeros_like(x)
    sx, _ = smooth_within_segments(x, y, 5)
    assert np.isnan(sx[20])
    assert np.allclose(sx[:20], 0.0) and np.allclose(sx[21:], 2.0)


def test_straightness_ignores_seconds_the_fish_was_still():
    """Swims straight for 10 s, then sits for 10 s with tracker wobble. The
    still seconds' wobble used to drag straightness towards 0."""
    moving = straight_line(n=300, step_px=4.0)
    still = np.repeat(moving[-1:], 300, axis=0) + np.random.default_rng(3).normal(
        0, 1.0, (300, 1, 2))
    m = metrics_for(np.concatenate([moving, still]))
    assert m["mean_path_straightness"] == pytest.approx(1.0, abs=0.01)
    assert 40.0 < m["straightness_windows_used_pct"] < 60.0


def test_distance_per_tracked_minute_does_not_penalise_dropout():
    full = straight_line(n=1800, step_px=2.0)
    gappy = with_dropout(full.copy(), 600, 780)      # 10% untracked
    a, b = metrics_for(full), metrics_for(gappy)
    assert b["total_distance"] < 0.92 * a["total_distance"]
    assert b["distance_per_tracked_min"] == pytest.approx(
        a["distance_per_tracked_min"], rel=0.01)


def test_freeze_fraction_and_freeze_duration_reconcile_exactly():
    """B6: the two used different denominators with nothing relating them —
    fraction over valid frames, duration over the whole recording. Both are now
    over observed time, and observed_duration_s is exported so the identity is
    checkable by the reader, not just by this test."""
    for traj in (stationary(n=600, jitter_px=0.0),
                 with_dropout(stationary(n=600, jitter_px=0.0), 200, 230),
                 with_scattered_dropout(stationary(n=600, jitter_px=0.0), 0.05)):
        m = metrics_for(traj)
        implied = m["freeze_total_duration_s"] / m["observed_duration_s"] * 100
        assert implied == pytest.approx(m["freeze_fraction_pct"], abs=1e-9)


def test_observed_duration_excludes_tracking_gaps():
    """observed_duration_s counts speed *samples*, not tracked frames.

    570 frames survive the 30-frame gap, but they sit in two segments and a
    segment's first frame carries no speed (its predecessor is the gap), so
    568 steps are measurable. That one-frame-per-segment difference is exactly
    why this value is returned from the freeze calculation rather than
    recomputed from the frame count — otherwise the B6 identity drifts.
    """
    m = metrics_for(with_dropout(stationary(n=600, jitter_px=0.0), 200, 230))
    assert m["observed_duration_s"] == pytest.approx(568 / FPS)
    assert m["longest_gap_s"] == pytest.approx(30 / FPS)
    assert m["tracked_fraction"] == pytest.approx(570 / 600)


# =============================================================================
# Bursting
# =============================================================================

# =============================================================================
# Shoaling — NND, IID, convex hull
# =============================================================================

def test_nnd_iid_and_hull_on_a_known_triangle():
    """Right triangle with legs of 150 px at 50 px/BL: every fish's nearest
    neighbour is 3 BL away, mean pairwise distance is (2 + sqrt(2))/3 * 3 BL,
    and the hull is half of a 3x3 BL square."""
    loaded = make_file(three_fish_fixed_geometry(n=300, sep_px=150.0))
    r = ShoalingCalculator(loaded, ShoalingParameters(30)).calculate()
    leg_bl = 150.0 / BODY_LENGTH_PX
    assert r.mean_nnd == pytest.approx(leg_bl)
    assert r.mean_iid == pytest.approx((2 + np.sqrt(2)) / 3 * leg_bl)
    assert r.mean_hull_area == pytest.approx(0.5 * leg_bl ** 2)


def test_nnd_never_exceeds_iid():
    loaded = make_file(three_fish_fixed_geometry(n=300, sep_px=150.0))
    r = ShoalingCalculator(loaded, ShoalingParameters(30)).calculate()
    assert r.mean_nnd <= r.mean_iid


def test_group_metrics_follow_the_calibration():
    """B7: NND/IID/hull hardcoded 1/body_length, so they stayed in body lengths
    whatever the user calibrated in — and body length varies 15% across the
    four supplied sessions, making "BL" a different physical distance per file
    with no way to override it."""
    traj = three_fish_fixed_geometry(n=300, sep_px=150.0)
    cm = CalibrationSettings.from_physical_measurement(10.0, "cm", FPS)
    r = ShoalingCalculator(make_file(traj, calibration=cm),
                           ShoalingParameters(30)).calculate()
    assert r.unit_name == "cm"
    assert r.mean_nnd == pytest.approx(150.0 / 10.0)          # 15 cm
    assert r.mean_iid == pytest.approx((2 + np.sqrt(2)) / 3 * 15.0)
    assert r.mean_hull_area == pytest.approx(0.5 * 15.0 ** 2)  # cm^2


def test_the_same_geometry_reads_the_same_in_cm_whatever_the_body_length():
    """The point of calibrating in cm: two recordings of fish of different
    sizes must report the same physical separation as the same number. In body
    lengths they differ by 15%, which is what made cross-file NND comparison
    unsound."""
    traj = three_fish_fixed_geometry(n=300, sep_px=150.0)
    cm = CalibrationSettings.from_physical_measurement(10.0, "cm", FPS)

    in_cm, in_bl = [], []
    for body_length in (71.3, 82.2):        # the real spread across sessions
        in_cm.append(ShoalingCalculator(
            make_file(traj, body_length=body_length, calibration=cm),
            ShoalingParameters(30)).calculate().mean_nnd)
        in_bl.append(ShoalingCalculator(
            make_file(traj, body_length=body_length),
            ShoalingParameters(30)).calculate().mean_nnd)

    assert in_cm[0] == pytest.approx(in_cm[1])          # identical in cm
    assert in_bl[0] != pytest.approx(in_bl[1])          # 15% apart in BL


def test_thigmotaxis_zone_assignment_is_unchanged_by_the_calibration_unit():
    """The arena polygon and the fish positions must be scaled by the *same*
    factor, or the zones move relative to the fish."""
    traj = stationary(n=300, x0=250.0, y0=500.0)   # 50 px inside the wall
    cm = CalibrationSettings.from_physical_measurement(10.0, "cm", FPS)

    bl_arena = _square_arena()
    cm_arena = ArenaDefinition(
        vertices_pixels=bl_arena.vertices_pixels.copy(),
        vertices_bl=np.stack([
            bl_arena.vertices_pixels[:, 0] * cm.scale_factor,
            (VIDEO_SIZE - bl_arena.vertices_pixels[:, 1]) * cm.scale_factor,
        ], axis=1))

    for loaded, arena in ((make_file(traj), bl_arena),
                          (make_file(traj, calibration=cm), cm_arena)):
        r = ThigmotaxisCalculator(loaded, arena, 0.15, 30).calculate()
        assert r.time_in_border_pct[0] == pytest.approx(100.0)


# =============================================================================
# Thigmotaxis
# =============================================================================

def _square_arena(inset_px=200.0):
    verts_px = np.array([
        [inset_px, inset_px], [VIDEO_SIZE - inset_px, inset_px],
        [VIDEO_SIZE - inset_px, VIDEO_SIZE - inset_px],
        [inset_px, VIDEO_SIZE - inset_px]])
    verts_bl = np.stack([verts_px[:, 0] / BODY_LENGTH_PX,
                         (VIDEO_SIZE - verts_px[:, 1]) / BODY_LENGTH_PX], axis=1)
    return ArenaDefinition(vertices_pixels=verts_px, vertices_bl=verts_bl)


def test_a_fish_in_the_middle_of_the_arena_is_all_centre():
    traj = stationary(n=300, x0=500.0, y0=500.0)
    r = ThigmotaxisCalculator(make_file(traj), _square_arena(), 0.15, 30).calculate()
    assert r.time_in_center_pct[0] == pytest.approx(100.0)
    assert r.time_in_border_pct[0] == pytest.approx(0.0)


def test_a_fish_hugging_the_wall_is_all_border():
    traj = stationary(n=300, x0=250.0, y0=500.0)  # 50 px inside a 600 px arena
    r = ThigmotaxisCalculator(make_file(traj), _square_arena(), 0.15, 30).calculate()
    assert r.time_in_border_pct[0] == pytest.approx(100.0)
    assert r.time_in_center_pct[0] == pytest.approx(0.0)


def test_border_and_centre_percentages_sum_to_one_hundred():
    """B9: out-of-arena positions sat in the denominator only, deflating both
    percentages with nothing in the export to say so. They are now shares of
    in-arena time, and the outside share is reported separately."""
    traj = stationary(n=200, x0=500.0, y0=500.0)
    traj[100:, 0] = [100.0, 100.0]  # outside the polygon for half the recording
    r = ThigmotaxisCalculator(make_file(traj), _square_arena(), 0.15, 30).calculate()
    assert r.time_in_border_pct[0] + r.time_in_center_pct[0] == pytest.approx(100.0)


# =============================================================================
# Arena coordinate round trip
# =============================================================================

def test_normalized_arena_vertices_round_trip():
    """from_normalized flips Y on vertices_pixels and get_normalized_vertices
    does not — check that is deliberate asymmetry, not a broken round trip."""
    norm = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
    arena = ArenaDefinition.from_normalized(
        norm, VIDEO_SIZE, VIDEO_SIZE, BODY_LENGTH_PX)
    back = arena.get_normalized_vertices(VIDEO_SIZE, VIDEO_SIZE, BODY_LENGTH_PX)
    assert back == pytest.approx(norm)


def test_arena_pixel_and_body_length_vertices_are_y_mirrors():
    """vertices_pixels is image space (Y down), vertices_bl is plot space
    (Y up). to_shapely(use_bl=False) therefore returns a mirrored polygon."""
    norm = np.array([[0.2, 0.3], [0.8, 0.3], [0.8, 0.9], [0.2, 0.9]])
    arena = ArenaDefinition.from_normalized(
        norm, VIDEO_SIZE, VIDEO_SIZE, BODY_LENGTH_PX)
    assert arena.vertices_pixels[:, 1] == pytest.approx(
        VIDEO_SIZE - arena.vertices_bl[:, 1] * BODY_LENGTH_PX)


# =============================================================================
# Quality gates and failure reporting
# =============================================================================

def test_a_poorly_tracked_fish_is_excluded_and_says_why():
    """B11: a fish tracked 1% of the time used to be averaged in beside fish
    tracked throughout. Below 80% it is now left out, with its share."""
    traj = straight_line(n=1000, step_px=2.0)
    traj[700:, 0, :] = np.nan
    loaded = make_file(traj)
    fish = process_and_analyze_file(loaded, ProcessingParameters.default_for_fish())
    assert fish == []
    assert "70%" in loaded.excluded_fish[0] and "80%" in loaded.excluded_fish[0]
    lenient = ProcessingParameters(min_valid_percentage=0.5)
    assert len(process_and_analyze_file(make_file(traj), lenient)) == 1


def test_a_failed_direction_calculation_is_distinguishable_from_a_real_zero():
    """B10: the except blocks returned the same zeros a genuine measurement
    returns, so the CSV could not tell 'this fish never turned' from 'the
    computation raised'."""
    from fish_analyzer.processing import MetricsCalculator

    class Exploding:
        def __getitem__(self, key):
            raise RuntimeError("synthetic failure")

    calc = MetricsCalculator(ProcessingParameters.default_for_fish())
    failed = calc._calc_movement_direction_metrics(Exploding(), np.zeros(10), FPS)
    # A real "fish never turned" result also yields 0 for both of these.
    assert not all(failed[k] == 0 for k in ("n_right_turns", "n_left_turns"))


def test_out_of_arena_share_is_exported(tmp_path):
    """B9: a reader seeing 0% border must be able to find out that half the
    recording fell outside the polygon entirely."""
    from fish_analyzer.export import export_thigmotaxis_csv

    traj = stationary(n=200, x0=500.0, y0=500.0)
    traj[100:, 0] = [100.0, 100.0]
    loaded = make_file(traj)
    loaded.thigmotaxis_results = ThigmotaxisCalculator(
        loaded, _square_arena(), 0.15, 30).calculate()

    out = tmp_path / "thigmo.csv"
    assert export_thigmotaxis_csv({"f": loaded}, out) == 1
    header, row = out.read_text(encoding="utf-8").splitlines()[:2]
    fields = dict(zip(header.split(","), row.split(",")))
    assert float(fields["MeanPctOutsideArena"]) == pytest.approx(50.0)


def test_the_arena_misalignment_warning_does_not_understate_itself():
    """The old denominator double-counted out-of-arena frames, reporting a
    fish 50% outside the polygon as 33.3% and making the 5% trigger harder to
    trip than intended."""
    traj = stationary(n=200, x0=500.0, y0=500.0)
    traj[100:, 0] = [100.0, 100.0]
    loaded = make_file(traj)
    with pytest.warns(UserWarning, match=r"50\.0% of valid fish positions"):
        ThigmotaxisCalculator(loaded, _square_arena(), 0.15, 30).calculate()


def test_a_fish_excluded_by_the_quality_gate_still_appears_in_the_csv(tmp_path):
    """B10: a fish that failed or was gated out simply vanished, so a reader
    counted five fish in a six-fish recording and never knew."""
    from fish_analyzer.export import export_combined_summary_csv

    traj = np.concatenate([straight_line(n=600, step_px=2.0),
                           straight_line(n=600, step_px=2.0)], axis=1)
    traj[5:, 1, :] = np.nan          # fish 1 fails the quality gate
    loaded = make_file(traj)
    loaded.processed_data = process_and_analyze_file(
        loaded, ProcessingParameters.default_for_fish())
    assert len(loaded.processed_data) == 1

    out = tmp_path / "combined.csv"
    assert export_combined_summary_csv({"f": loaded}, {}, out) == 2

    lines = out.read_text(encoding="utf-8").splitlines()
    fields = [dict(zip(lines[0].split(","), r.split(","))) for r in lines[1:]]
    by_id = {f["FishID"]: f for f in fields}
    assert by_id["0"]["Status"] == "ok"
    assert by_id["1"]["Status"].startswith("excluded:")


def test_status_names_which_calculation_failed(tmp_path):
    """A NaN with Status='ok' is a real result; a NaN with Status naming the
    calculation is a crash. The CSV has to carry the difference."""
    from fish_analyzer.export import export_combined_summary_csv

    loaded = make_file(straight_line(n=600, step_px=2.0))
    loaded.processed_data = process_and_analyze_file(
        loaded, ProcessingParameters.default_for_fish())
    assert loaded.processed_data[0].status == "ok"

    loaded.processed_data[0].failed_metrics.append("turning")
    out = tmp_path / "combined.csv"
    export_combined_summary_csv({"f": loaded}, {}, out)
    assert "partial: turning" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The nearest-neighbour helper shared with the Video Inspector overlay
# ---------------------------------------------------------------------------

def test_nearest_neighbour_distances_matches_calculator_method():
    """The shared helper and ShoalingCalculator must not drift apart."""
    from fish_analyzer.shoaling import nearest_neighbour_distances

    positions = np.array([[0.0, 0.0], [3.0, 4.0], [100.0, 100.0]])
    nnd, nn_idx = nearest_neighbour_distances(positions)

    assert nnd[0] == pytest.approx(5.0)
    assert nnd[1] == pytest.approx(5.0)
    assert nn_idx[0] == 1
    assert nn_idx[1] == 0


def test_nearest_neighbour_distances_ignores_untracked_fish():
    """A NaN fish is neither a source nor a candidate neighbour."""
    from fish_analyzer.shoaling import nearest_neighbour_distances

    positions = np.array([[0.0, 0.0], [np.nan, np.nan], [3.0, 4.0]])
    nnd, nn_idx = nearest_neighbour_distances(positions)

    assert np.isnan(nnd[1])
    assert nn_idx[1] == -1
    assert nnd[0] == pytest.approx(5.0)
    assert nn_idx[0] == 2


def test_nearest_neighbour_distances_single_fish_has_no_neighbour():
    from fish_analyzer.shoaling import nearest_neighbour_distances

    nnd, nn_idx = nearest_neighbour_distances(np.array([[10.0, 10.0]]))

    assert np.isnan(nnd[0])
    assert nn_idx[0] == -1


# ---------------------------------------------------------------------------
# Per-fish IID, which the Video Inspector's IID panel plots
# ---------------------------------------------------------------------------

def test_individual_iid_is_each_fish_mean_distance_to_the_others():
    """Distinct from the all-pairs mean, which is one number for the group."""
    from fish_analyzer.shoaling import ShoalingCalculator

    # Fish 0 and 1 are 10 apart; fish 2 sits 100 away from both.
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 100.0]])
    per_fish = ShoalingCalculator._calculate_individual_iid_at_frame(
        None, positions)

    assert per_fish[0] == pytest.approx((10.0 + 100.0) / 2)
    assert per_fish[1] == pytest.approx((10.0 + np.hypot(10.0, 100.0)) / 2)
    assert per_fish[2] == pytest.approx((100.0 + np.hypot(10.0, 100.0)) / 2)


def test_individual_iid_averages_to_the_all_pairs_mean():
    """The group value is recoverable, so nothing is lost by storing per-fish."""
    from fish_analyzer.shoaling import ShoalingCalculator
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(3)
    positions = rng.normal(0, 50, size=(6, 2))

    per_fish = ShoalingCalculator._calculate_individual_iid_at_frame(
        None, positions)

    assert per_fish.mean() == pytest.approx(pdist(positions).mean())


def test_individual_iid_is_nan_for_an_untracked_fish():
    from fish_analyzer.shoaling import ShoalingCalculator

    positions = np.array([[0.0, 0.0], [np.nan, np.nan], [3.0, 4.0]])
    per_fish = ShoalingCalculator._calculate_individual_iid_at_frame(
        None, positions)

    assert np.isnan(per_fish[1])
    assert per_fish[0] == pytest.approx(5.0)


def test_individual_iid_can_differ_sharply_from_the_group_mean():
    """The reason the inspector must not caption one fish with the other.

    One outlier drags its own mean far above the all-pairs mean.
    """
    from fish_analyzer.shoaling import ShoalingCalculator
    from scipy.spatial.distance import pdist

    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [500.0, 500.0]])
    per_fish = ShoalingCalculator._calculate_individual_iid_at_frame(
        None, positions)

    group = pdist(positions).mean()
    assert per_fish[3] > group * 1.5
    assert per_fish[0] < group
