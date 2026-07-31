# Audit B — Numerical and scientific correctness

Run 2026-07-31 against `60c6c2e` on branch `audit/passes-a-h-d`, following
[`audit/PASS_B_correctness.md`](audit/PASS_B_correctness.md).

The audit itself changed nothing. It left behind
[`tests/synthetic_tracks.py`](tests/synthetic_tracks.py) and
[`tests/test_metric_correctness.py`](tests/test_metric_correctness.py) — one
test per metric, `xfail(strict=True)` on each confirmed defect.

> ### Acted on 2026-08-01 — eight columns withdrawn
> Following this report, **eight columns were removed from the exports, the
> GUI and the computation** rather than repaired. They are listed in
> [B1](#b1), [B3](#b3) and [B15](#b15). See
> [What changed on 2026-08-01](#what-changed-on-2026-08-01) at the end for the
> full diff summary and what was deliberately left alone.
>
> Three findings below were **corrected after the fact** by checking every
> "verified correct" claim against the real sessions rather than only against
> synthetic input — [B15](#b15), [B16](#b16) and [B17](#b17). Two of them
> downgrade columns this report originally called trustworthy.

Environment: a throwaway venv built from Audit A's recipe (Python 3.12.5,
numpy 2.5.1, pandas 3.0.5, traja 25.0.1, scipy 1.18.0, shapely 2.1.2). All four
supplied `wtTAB_6mo/Freeswim` sessions were used read-only for the real-data
figures; nothing was written outside the scratchpad.

---

## Verdict up front

Of the 26 columns `export_combined_summary_csv` wrote when this audit started,
**10 are trustworthy, 5 are trustworthy but mislabelled or not comparable
across files, and 11 were not measuring fish behaviour.** Eight of those 11
have since been removed; the other three are fixable and remain.

The headline: **`MeanAngularVelocity_deg_s`, `ErraticMovementCount`,
`BurstCount`, `BurstMeanSpeed` and `BurstFrequency_per_min` are readouts of
idtracker.ai centroid noise.** A *perfectly straight* synthetic swimmer moving
at the real median speed, with 0.5–1.0 px of Gaussian centroid noise added,
reproduces the entire observed range of all five columns across the four real
sessions. No turning and no bursting need be invoked to explain them.

Second: `IBI_Median_ms` is substantially a measurement of tracking dropout.
Across the four sessions, **18–86% of exported inter-bout intervals coincide
with a frame gap** — the fish did not pause, the tracker lost it.

Third, found only by running the exporter end to end on real sessions rather
than on synthetic input: **`NetDisplacement` is NaN for half the real fish**
(B16) and **`MaxSpeed` reports tracking teleports of up to 321 BL/s** (B17).
Both are exact on clean input, which is why the first pass called them
trustworthy.

`TotalDistance`, `MeanSpeed`, `MedianSpeed`, `PathStraightness`,
`LateralityIndex`, NND, IID and hull area check out against analytic ground
truth *and* survive the real-data check. Those you can publish.

---

# (a) Findings, ordered by scientific impact

## B1 — Angular velocity and erratic-movement counts measure tracking noise `CONFIRMED` <a id="b1"></a>

**Would it change a published number?** It *is* the published number. There is
no signal underneath to recover.

`_calc_movement_direction_metrics` ([processing.py:558-572](fish_analyzer/processing.py#L558))
takes `arctan2` of raw frame-to-frame displacement with no displacement floor,
while `bout_analysis.py:255` guards the identical computation with
`_MIN_DISP = 0.05` BL. `apply_smoothing` defaults to `False`.

| Synthetic case | Expected | Actual |
|---|---|---|
| straight swimmer, no noise | 0 deg/s | **0.000** ✓ |
| circler, 0.05 rad/frame @ 30 fps | 85.944 deg/s | **85.944** ✓ |
| straight swimmer + 0.1 px jitter | 0 deg/s | **164.6 deg/s** |
| straight swimmer + 0.3 px jitter | 0 deg/s | **496.6 deg/s** |
| straight swimmer + 0.5 px jitter | 0 deg/s | **843.1 deg/s** |
| stationary fish + 0.5 px jitter | 0 deg/s, 0 erratic | **4000.5 deg/s, 783 erratic/min** |

The metric is exact on a clean signal and catastrophic on a noisy one. **0.1 px
of jitter produces twice the angular velocity of a genuine, hard 86 deg/s
circle.**

**The noise floor accounts for the real data completely.** A straight-line
swimmer at the real median speed (6.2 px/frame, body length 75 px, 18,000
frames):

| centroid noise | `MeanAngularVelocity_deg_s` (smoothing off) | (smoothing on) | `BurstCount` |
|---|---|---|---|
| 0.00 px | 0.0 | 0.0 | 0 |
| 0.25 px | 134.9 | 48.9 | 6,128 |
| 0.50 px | 271.0 | 97.8 | 6,451 |
| 0.75 px | 409.1 | 146.8 | 6,499 |
| 1.00 px | 550.9 | 196.2 | 6,529 |
| **observed, 4 real sessions, 24 fish** | **227 – 504** | **154 – 377** | **4,138 – 5,290** |

Corroborating evidence from the real sessions:

- Turning on the smoothing checkbox lowers `MeanAngularVelocity_deg_s` by
  **36.8% on average**, and the Spearman rank correlation between the two
  settings is only **0.866** — a display option reorders the fish.
- `ErraticMovements_per_min` runs 20.6–81.3 with smoothing off and 5.6–44.2
  with it on: **a 3× change from a checkbox.**

**Consequence.** Any group difference reported in these two columns is a
difference in tracking quality between the groups. Smoothing does not rescue
them: at 0.3 px jitter, smoothing brings 496.6 deg/s down to 177.7 deg/s, still
double a real hard circle.

This is the empirical confirmation of the limit `AUDIT_H_APPROACH.md` predicted
from first principles — centroid geometry cannot support heading, angular
velocity, laterality or erratic-movement counts. B adds the number: the noise
contribution is on the order of 200–550 deg/s, and the exported values are
227–504 deg/s.

Tests: these three metrics were withdrawn on 2026-08-01, so the `xfail` tests
that described their defects were deleted with them. What remains is
`test_noise_dominated_metrics_are_not_computed`, which fails if any of them
reappears.

---

## B2 — Burst detection fires on 39% of all frames `CONFIRMED` <a id="b2"></a>

**Would it change a published number?** `BurstCount`, `BurstFrequency_per_min`
and `BurstMeanSpeed` are unusable as published. The real sessions report
**4,138–5,290 bursts per 10-minute recording — 7 to 9 "bursts" per second.**

`_calc_burst_metrics` differentiates the already-noisy speed trace, so it
inherits B1's noise and squares it. Measured directly on the pooled real speed
traces (407,588 samples): **39.12% of frames exceed the default
`burst_accel_threshold = 2.0` BL/s².** An event detector that fires on four
frames in ten is not detecting events.

| Synthetic case | Expected | Actual |
|---|---|---|
| straight swimmer, no noise | 0 bursts | **0** ✓ |
| straight swimmer + 0.1 px jitter | 0 bursts | **179 (537/min)** |
| straight swimmer + 0.3 px jitter | 0 bursts | **215 (645/min)** |
| straight swimmer + 0.5 px jitter | 0 bursts | **221 (663/min)** |

Test: withdrawn 2026-08-01; guarded by `test_noise_dominated_metrics_are_not_computed`.

---

## B3 — Burst duration measures the acceleration ramp, and peak speed can be missed `CONFIRMED` <a id="b3"></a>

Two separate defects in the same function, independent of B2.

**Duration.** [processing.py:491](fish_analyzer/processing.py#L491) counts frames
where *acceleration* exceeds threshold and reports that span as
`burst_mean_duration_s`. The docstring promises "For each burst, we record the
peak speed and duration".

> Synthetic case: one swim event — ramp up over 10 frames, hold top speed for
> 60 frames, ramp down over 10. One burst by any reading, 80 frames = **2.67 s**.
> **Actual: `burst_mean_duration_s = 0.300 s`** — the 9-frame acceleration ramp.
> The 60-frame plateau contributes nothing.

`burst_mean_duration_s` answers "how long did the fish spend accelerating",
under a column name that says otherwise. That is a definitional mislabel, not a
rounding error: the two quantities differ by 9× on this case.

**Peak speed.** The `speed_segment` slice at
[processing.py:493](fish_analyzer/processing.py#L493) ends where acceleration
falls below threshold, which is not where speed peaks.

> Synthetic case: speed rises as √t (decaying acceleration) to a plateau of
> 3.6 BL/s. **Expected `burst_mean_speed` 3.60 BL/s; actual 1.2151 BL/s** — a
> 3× understatement.

The slice happens to capture the peak when acceleration stays above threshold
right up to top speed, which is why the first case above got
`burst_mean_speed` exactly right. It is correct by coincidence of the test
case, not by construction.

Tests: withdrawn 2026-08-01. `synthetic_tracks.single_swim_event` is retained
unused — it encodes the ground truth a replacement burst detector must return.

---

## B4 — Inter-bout interval largely measures tracking dropout `CONFIRMED`

**Would it change a published number?** Yes, and unpredictably — the bias
tracks tracking quality, which differs between recordings and therefore between
experimental groups.

[bout_analysis.py:432](fish_analyzer/bout_analysis.py#L432) sets NaN speed to
`0.0`, so a tracking gap reads as "the fish stopped". A gap in the middle of
continuous swimming splits one bout into two and inserts an inter-bout interval
equal to the gap length.

> Synthetic case: a swimmer moving in a perfectly straight line at constant
> speed for 600 frames, with frames 200–229 blanked.
> **Expected: 1 bout, no IBI. Actual: 2 bouts, `ibi_median_ms = 1033.3`** —
> exactly the 31-frame gap.

On the four real sessions, per fish:

| Session | share of exported IBIs that coincide with a tracking gap |
|---|---|
| G604_Freeswim | 62% – 86% |
| G604_Freeswim_group2 | 44% – 86% |
| H604_Freeswim | 34% – 83% |
| H604_Freeswim_group2 | **18% – 61%** |

The best-tracked session is the one with the fewest contaminated intervals, and
the worst-tracked session is the one with the most. `IBI_Median_ms` and
`Bout_Rate_per_min` therefore correlate with tracking quality by construction.

The exported medians (133–333 ms) are also implausible on their face for adult
free swimming, which is close to continuous rather than bout-structured. See
B13.

Test: `test_a_tracking_gap_does_not_become_an_inter_bout_interval` (`xfail`).

---

## B5 — Freeze count is set by the dropout policy, not by the fish `CONFIRMED`

[processing.py:413](fish_analyzer/processing.py#L413) sets NaN speed to
*not-frozen* — the opposite of `bout_analysis.py`'s convention. A gap therefore
severs a freeze run into two.

> Synthetic case: a fish that never moves for 600 frames, with 5% of frames
> blanked as isolated single-frame dropouts.
> **Expected `freeze_count = 1`. Actual: 22.**
> With the same 5% blanked as one contiguous run: `freeze_count = 2`.

On the real sessions, recomputing with the *opposite* convention (a gap
continues a freeze rather than breaking it):

| | as exported | if a gap did not sever the run |
|---|---|---|
| range across 24 fish | **1 – 55** | **15 – 170** |
| example: G604 fish 0 | 13 | 143 |
| example: H604_group2 fish 4 | 30 | 40 |

A binary policy choice inside the function moves `FreezeCount` by **2× to 11×**,
and the size of the move depends on how badly that particular fish was tracked.
Neither convention is right — a gap is neither frozen nor moving, and freeze
runs should be detected only within contiguous tracked segments, with gap
frames excluded from the denominator.

Note the two modules disagree in opposite directions on the *same* dropout:
`processing.py` calls it moving, `bout_analysis.py` calls it still. Both
columns appear in the same CSV row.

Test: `test_dropout_does_not_split_one_freeze_into_many` (`xfail`).

---

## B6 — `FreezeFraction_pct` and `FreezeTotalDuration_s` use different denominators `CONFIRMED`

[processing.py:434-443](fish_analyzer/processing.py#L434): `freeze_fraction_pct`
divides frozen frames by `n_valid`; `freeze_total_duration_s` divides by frame
rate over the whole recording.

> Synthetic case: a motionless fish, 600 frames at 30 fps, frames 200–229
> blanked. **`freeze_fraction_pct = 100.00`** while
> **`freeze_total_duration_s = 18.933 s` of a 20.000 s recording** — 94.67%.

Both are defensible on their own; together in one row they are contradictory.
A reader computing `FreezeTotalDuration_s / recording_length` gets a different
answer from `FreezeFraction_pct`, and the gap is exactly the dropout fraction —
1.3% to 9.7% on the supplied sessions.

Test: `test_freeze_fraction_and_freeze_duration_use_the_same_denominator` (`xfail`).

---

## B7 — Group and spatial metrics ignore calibration; "BL" is a different unit per file `CONFIRMED`

[shoaling.py:204](fish_analyzer/shoaling.py#L204) and
[spatial.py:344](fish_analyzer/spatial.py#L344) compute
`pixels_to_bl = 1.0 / metadata.body_length` instead of using
`calibration.scale_factor`.

**The numbers are not wrong and the column names are honest** — `MeanNND_BL`,
`MeanIID_BL`, `HullArea_BL2` really are body lengths. The defect is that the
unit is *unchangeable*, and body length is a per-file quantity.

> Synthetic case: three motionless fish in a fixed 150 px triangle.
> Calibrated in BL (50 px/BL): `MeanNND_BL = 3.0000`. Correct.
> Calibrated in cm (10 px/cm): **still `3.0000`** — the true answer is 15 cm.
> Meanwhile the same file's `TotalDistance` and `MeanSpeed` *did* switch to cm.

So one CSV row can carry `Unit = cm` for the individual columns while the
shoaling export for the same file is in body lengths. Heatmap axes are labelled
"X (BL)" and the shoaling plots "Mean NND (BL)" regardless of calibration
(`shoaling_tab.py:292-428`, `spatial_tab.py:411`).

**The scientific cost is cross-file comparability.** Across the four supplied
sessions `body_length` runs 71.31 – 82.22 px. The same physical separation
therefore exports as:

| `body_length` | `MeanNND_BL` for an identical 150 px separation |
|---|---|
| 71.31 px | 2.1038 |
| 82.22 px | 1.8248 |
| | **15.3% apart** |

Calibrating every file in cm is exactly the control a researcher would reach
for to fix this, and it has no effect. The observed session means
(1.047, 1.428, 1.183, 1.169 BL) cannot be compared like-for-like.

Thigmotaxis is *not* affected in the same way: the arena polygon's `vertices_bl`
is built with `1/body_length` in the GUI and the fish positions are converted
with `1/body_length` in the calculator, so zone assignment is self-consistent
under any calibration. Verified — see (d).

Test: `test_group_metrics_ignore_calibration` (`xfail`);
`test_speed_follows_the_calibration_unit` (passes — the individual side is fine).

---

## B8 — 77% of "straight" bouts are guard returns, not measurements `CONFIRMED`

`_compute_heading_change` ([bout_analysis.py:240](fish_analyzer/bout_analysis.py#L240))
returns a literal `0.0` from five separate guard paths: no room for the
look-back window (`pre == start or post == end`), NaN in the look-back vectors,
sub-`_MIN_DISP` displacement, and two more in the `np.sum(valid) < 2` fallback.
`compute_summary` then classifies `|heading_change| <= 5°` as straight
([bout_analysis.py:363](fish_analyzer/bout_analysis.py#L363)), and `0.0` sits
in the middle of that dead zone.

> Synthetic case: a 1-frame bout at array index 0, immediately followed by an
> unambiguous 90° turn. `pre == start`, so the guard fires.
> **Expected: a measurement or a NaN. Actual: `heading_change_deg = 0.00`,
> classified STRAIGHT.** Same for a bout at the last index, and for a bout
> preceded by a tracking gap.

Across all 2,866 bouts detected in the four real sessions:

- **490 bouts (17.1%) have `heading_change_deg` exactly `0.0`** — the guard
  signature, since a measured value hitting exact zero is vanishingly unlikely.
- 637 bouts (22.2%) are classified straight.
- **77% of the "straight" classification is guard returns.**

`Bout_N_Straight` is therefore mostly a count of bouts whose heading could not
be measured. `Bout_LateralityIndex` is unaffected (it uses only `n_left` and
`n_right`), but `Bout_MeanAbsTurnAngle_deg` is diluted toward zero by the same
490 bouts.

Test: `test_an_unmeasurable_heading_change_is_not_reported_as_straight` (`xfail`).

---

## B9 — Thigmotaxis percentages need not sum to 100, and the warning understates the problem `CONFIRMED`

[spatial.py:411-434](fish_analyzer/spatial.py#L411): a position outside the
arena polygon increments `frames_valid` but neither `frames_in_border` nor
`frames_in_center`. Both exported percentages divide by `frames_valid`.

> Synthetic case: a fish in the exact centre of the arena for 100 frames, then
> outside the polygon entirely for 100 frames.
> **Expected: the export tells the reader half the data was unclassifiable.**
> **Actual: `border = 0.0%`, `centre = 50.0%`, sum 50%.**

A reader of `MeanPctInBorder = 0.0` concludes the fish never approached a wall.
The exported CSV (`export_thigmotaxis_csv`, five columns) carries no trace of
the missing 50%.

**Judgement: bug, not a defensible choice.** Excluding out-of-arena positions
from *both* numerator and denominator would be defensible; excluding them from
the numerators only, while keeping them in the denominator, is not a policy —
it silently deflates both percentages by the out-of-arena fraction.

**Sub-finding.** The warning at
[spatial.py:444-457](fish_analyzer/spatial.py#L444) computes
`total_outside / (total_valid + total_outside)`, but `frames_outside_arena` is
*already* counted inside `frames_valid`. On the case above it printed
**33.3%** where the true out-of-arena share is **50%**. The warning
systematically understates the misalignment it exists to catch, and the
5% trigger threshold is correspondingly harder to trip than intended.

Test: `test_border_and_centre_percentages_sum_to_one_hundred` (`xfail`);
`test_a_fish_in_the_middle_of_the_arena_is_all_centre` and
`test_a_fish_hugging_the_wall_is_all_border` (both pass).

---

## B10 — A reader of the CSV cannot tell a measurement from a crash `CONFIRMED`

`_calc_movement_direction_metrics` and `_calc_path_straightness` catch
`Exception` and return a mixture of NaN *and zeros*:

```
{'mean_angular_velocity_deg_s': nan, 'erratic_movement_count': 0,
 'erratic_movements_per_min': 0.0, 'laterality_index': nan,
 'cumulative_heading_change_deg': nan,
 'mean_signed_angular_velocity_deg_s': nan,
 'n_right_turns': 0, 'n_left_turns': 0}
```

`ErraticMovementCount = 0`, `ErraticMovements_per_min = 0.0`,
`RightTurns_CW = 0` and `LeftTurns_CCW = 0` are **exactly what a genuine
"this fish never turned" result produces**. Four columns of a crashed
computation are indistinguishable from a real measurement. The NaN columns at
least signal *something*, but they too are indistinguishable from the legitimate
NaN a never-moving fish produces (verified: a motionless fish with zero jitter
returns NaN for all direction metrics via the `n_moving == 0` path).

`process_all_fish` ([processing.py:190](fish_analyzer/processing.py#L190))
additionally swallows per-fish failures with a `print` and `continue`, so a
fish that failed entirely is simply **absent** from the CSV rather than present
with a failure marker. Combined with Audit D's finding that the GUI log's
history is unreachable, a batch run can lose a fish with no durable record.

**This is a data-integrity finding, not a style one.** Nothing in the export
distinguishes the three states, and there is no `Status` or `Error` column to
add one to.

Test: `test_a_failed_direction_calculation_is_distinguishable_from_a_real_zero`
(`xfail`).

---

## B11 — `min_valid_percentage = 0.01` lets a 1%-tracked fish into the export `CONFIRMED`

[processing.py:73](fish_analyzer/processing.py#L73) and
[processing.py:210](fish_analyzer/processing.py#L210).

> Synthetic case: a fish tracked in 10 of 1,000 frames.
> **Actual: exported, with `ValidFrames_pct = 1.0`, `MeanSpeed = 1.2000`
> computed from nine real steps, `FreezeCount = 0`, `LateralityIndex = 0.0`** —
> formatted identically to a fish measured over 18,000 frames.

**Assessment: not defensible as a default.** `ValidFrames_pct` is present, so a
careful reader *can* filter, but nothing in the pipeline, the GUI or the CSV
signals that they must, and a mean over nine steps carries no uncertainty
marker to distinguish it. The supplied sessions are all 87–99% tracked, so this
did not bite here; it is a trap waiting for a bad recording. A defensible
default is 0.5–0.8 with an explicit override, plus a `Quality` flag column.

Test: `test_a_fish_tracked_in_one_percent_of_frames_is_still_exported`
(passes — it pins the *current* policy, so changing the default will fail it
deliberately).

---

## B12 — Shoaling samples are not evenly spaced in time `CONFIRMED`, minor

[shoaling.py:232](fish_analyzer/shoaling.py#L232) does
`complete_frame_indices[::sample_interval_frames]` — every 30th *complete*
frame, not every 30th frame. The `ShoalingParameters` docstring says "we sample
at regular intervals (e.g., every 30 frames = 1/sec)".

| Session | completeness (all 6 tracked) | samples used | median spacing | **worst gap** |
|---|---|---|---|---|
| G604_Freeswim | 61.0% | 367 of 600 | 1.03 s | **9.4 s** |
| G604_group2 | 71.1% | 427 of 600 | 1.00 s | 8.5 s |
| H604_Freeswim | 87.6% | 526 of 600 | 1.00 s | 6.2 s |
| H604_group2 | 93.8% | 563 of 600 | 1.00 s | 2.8 s |

The `timestamps` array is correct, so time-series plots are drawn at the right
x-positions — but up to 39% of the recording is silently absent, and the
summary statistics are unweighted means over irregularly spaced samples.

**I expected this to bias NND upward** (idtracker.ai loses identities when fish
overlap, so complete frames should over-represent separated fish). **It does
not.** Comparing mean NND on complete frames only against mean NND computed
from whoever happened to be tracked, at a true fixed 30-frame interval:
**+0.1%, −0.3%, +1.4%, +1.0%.** The all-fish-tracked requirement costs coverage,
not accuracy. Reported as a caveat, not a correctness defect.

---

## B15 — Cumulative heading and signed angular velocity flip sign under smoothing `CONFIRMED` <a id="b15"></a>

**This corrects this report's own earlier claim.** Section (c) originally said
of `CumulativeHeading_deg` and `MeanSignedAngVel_deg_s` that "magnitudes
inherit B1, signs are sound". The first half was right; **the second half was
wrong**, and it mattered enough to change what got removed.

The reasoning behind "signs are sound" was that symmetric noise cancels in a
signed sum, and the noise floor does confirm that — a straight swimmer at real
speed with 1.0 px of jitter accumulates only −8.2° of spurious heading against
real values in the thousands. But I never checked sign *stability on real
trajectories*, where the turns are not symmetric noise.

| | smoothing OFF | smoothing ON | rank ρ | sign flips |
|---|---|---|---|---|
| `CumulativeHeading_deg` | −12648 … +4296 | −8160 … +4768 | 0.678 | **4 / 24** |
| `MeanSignedAngVel_deg_s` | −24.8 … +8.8 | −14.5 … +9.7 | 0.685 | **4 / 24** |
| `LateralityIndex` | −0.02 … +0.05 | −0.02 … +0.05 | 0.941 | 2 / 24 (all on values < 0.05, i.e. noise around a true zero) |

The flips are not marginal. `H604_Freeswim` fish 4 goes from **−5133.9° to
+1836.5°** — fourteen net clockwise rotations becoming five counter-clockwise —
purely from ticking the smoothing checkbox. `G604_group2` fish 0 goes
+1703.7 → −3388.7.

Both columns were withdrawn alongside the six in B1 and B3, bringing the total
to eight. `LateralityIndex` and its two turn counts survive: their sign flips
occur only where the value is indistinguishable from zero anyway, and the rank
correlation holds at 0.941.

---

## B16 — `NetDisplacement` is NaN for half the real fish `CONFIRMED` <a id="b16"></a>

**Also a correction.** Section (c) listed `NetDisplacement` as trustworthy on
the strength of an exact synthetic result (23.96 BL, correct to machine
precision). It is exact — *when it returns a number at all*.

`traja.distance()` computes the straight line between the **first and last
rows** of the frame. If either is a tracking dropout, the result is NaN for
that fish regardless of how well-tracked the other 17,998 frames are.

> **12 of the 24 fish in the four supplied sessions export
> `NetDisplacement = nan`**, while `TotalDistance` for the same fish is a
> perfectly good 1711.9 BL.

The fix is to use the first and last *valid* positions. This was missed in the
first pass because every synthetic track began and ended with a tracked frame —
a gap in the test design, not in the reasoning. Now pinned by
`test_net_displacement_survives_a_dropout_at_the_ends`.

---

## B17 — `MaxSpeed` reports the worst tracking glitch, not the fastest swim `CONFIRMED` <a id="b17"></a>

**The third correction**, and the same root cause as B16: a metric that is
exact on clean synthetic input and meaningless on real input.

`max_speed` is a raw `np.max` over the speed distribution, which on real
recordings contains identity swaps and re-acquisitions that teleport a fish
across the arena in one frame.

| | across the 24 real fish |
|---|---|
| exported `MaxSpeed` | **32.0 – 321.0 BL/s** |
| 99.9th percentile of the same speed traces | 10.4 – 60.3 BL/s |
| ratio | **2.4× – 7.9×** |

Adult zebrafish burst swimming is on the order of 15–25 BL/s. A reported
**321 BL/s is roughly 17 m/s** — not a fish. For one representative fish, only
43 of 15,983 speed samples exceed 25 BL/s (0.27%), and `MaxSpeed` is set
entirely by those.

`MeanSpeed` and `MedianSpeed` are unaffected — a 0.27% contamination moves a
median not at all and a mean negligibly. **`MaxSpeed` should be a high
percentile** (P99 or P99.9), or the speed trace should be filtered against a
physiological ceiling before the max is taken. Pinned by
`test_max_speed_is_not_set_by_a_single_teleport`.

**The general lesson**, which applies to how the rest of this report should be
read: a metric verified only against clean synthetic input is verified against
arithmetic, not against data. B16 and B17 were both invisible until the
exporter was run end to end on the real sessions. Every remaining
"trustworthy" verdict in section (c) has now had that check applied —
`TotalDistance`, `MeanSpeed`, `MedianSpeed`, `PathStraightness`,
`LateralityIndex`, NND, IID and hull area all survive it.

---

## B13 — Thresholds and definitions drift from the zebrafish behaviour literature `PLAUSIBLE`

Labelled `PLAUSIBLE`: the code behaviour below is `CONFIRMED` by execution, but
the comparisons to standard definitions are from background knowledge and were
**not** re-verified against primary sources in this session. Treat the specific
attributions as leads to check, not as citations.

| Metric | Standard definition | This implementation | Verdict |
|---|---|---|---|
| **NND** | distance to nearest conspecific | identical | matches |
| **IID** | mean over all unique pairs | identical (`pdist`) | matches |
| **Convex hull area** | area of the hull over all individuals | identical | matches |
| **Path straightness** | net displacement ÷ path length (Batschelet's straightness index) | identical, over a sliding 1 s window | matches, but **window-dependent** — the same circler scores 0.9148 at a 1 s window and would score very differently at 2 s. The window is not exported. |
| **Laterality index** | (R − L)/(R + L) | identical | matches; sign convention verified in (d) |
| **Freezing** | speed below threshold for a *minimum duration*, conventionally ≥1 s in adults | `min_freeze_frames = 5` = **167 ms** at 30 fps | **too short.** 167 ms of slow swimming is not a freezing episode under any published adult criterion. Combined with B5 this is why `FreezeCount` reaches 55 in a 10-minute recording. |
| **Freeze speed threshold** | typically a small fraction of cruising speed | `0.5` BL/s | plausible: only **4.81%** of real frames fall below it, and the real median is 2.47 BL/s. |
| **Burst / high-speed swim** | a discrete escape or startle event, expected order 0–10 per minute | acceleration > `2.0` BL/s², which **39.12% of real frames exceed** | **wrong by orders of magnitude.** See B2. |
| **Bout rate / IBI** | a *larval* construct — larvae swim in discrete darts; adults swim near-continuously | applied to 6-month adults | **the model does not fit the assay.** Median IBIs of 133–333 ms are threshold-crossing artefacts (B4), not behavioural pauses. |
| **Thigmotaxis** | outer zone defined either by area (e.g. an outer annulus of 50% of arena area) or by a fixed distance from the wall (often 1–2 body lengths) | 15% of the **shorter bounding-box dimension**, buffered uniformly inward | a third convention. Defensible, but **not exported** — `export_thigmotaxis_csv` writes no `border_zone_pct`, so the number is not reproducible from the CSV alone. |
| **Angular velocity** | rate of change of *body axis* heading | rate of change of centroid displacement direction | a proxy, and B1 shows it fails at this noise level. |

**The bout module deserves a scoping decision, not a fix.** It is documented as
"Bout-based locomotor analysis for zebrafish larvae (and adults)" and its
defaults (`merge_gap_frames = 2`, `min_bout_frames = 1`) are larval. Applied to
adult free swim, `Bout_Count` (39–230 per 10 min) and `IBI_Median_ms` are
threshold-crossing statistics of a continuous swimmer. Fixing B4 and B8 would
make the numbers self-consistent without making them meaningful for this assay.

---

## B14 — Docstrings that promise something the code does not do `CONFIRMED`

| Location | Promise | Reality |
|---|---|---|
| `processing.py:452-465` | "For each burst, we record the peak speed and duration" | duration is the acceleration ramp (B3); peak can be missed (B3) |
| `processing.py:399-405` | `freeze_total_duration_s` = "total time spent frozen", `freeze_fraction_pct` = "% of time spent frozen" | different denominators, so the two cannot both be true (B6) |
| `processing.py:532-535` | angular velocity "Computed only when the fish is moving … to avoid noise from stationary heading jitter" | the speed gate does not remove heading jitter at swimming speeds — 6.2 px/frame with 0.5 px noise still yields 271 deg/s (B1) |
| `processing.py:575-577` | comment asserting the speed/heading alignment | **correct** — verified explicitly, see (d) |
| `shoaling.py:42-45` | "we sample at regular intervals (e.g., every 30 frames = 1/sec)" | every 30th *complete* frame; worst real gap 9.4 s (B12) |
| `spatial.py:12-14` | "Create 'center zone' by shrinking arena inward by X%" | X% of the **shorter bounding-box dimension**, applied uniformly — for a non-square arena the ring is not X% of each dimension |
| `export.py:183-189` | column list for `export_individual_metrics_csv` | omitted the turning columns the function actually wrote — **fixed 2026-08-01** |
| `bout_analysis.py:22`, `processing.py:609-615` | "+ = CCW/left, − = CW/right" | **correct and mutually consistent** — verified, see (d) |

---

# (b) Seed suspicions: confirmed or refuted

| # | Suspicion | Verdict |
|---|---|---|
| 1 | Calibration bypass in `shoaling.py`/`spatial.py` | **CONFIRMED** → B7. Numbers self-consistent, unit unchangeable, 15.3% cross-file spread |
| 2 | Two independent speed pipelines disagree | **REFUTED.** Same formula, same values. See (d) |
| 3 | Opposite NaN conventions | **CONFIRMED** → B4, B5. Quantified: freeze count 2–11×, 18–86% of IBIs |
| 4 | Turn metrics on unsmoothed positions | **CONFIRMED, worst finding** → B1 |
| 5 | Mixed freeze denominators | **CONFIRMED** → B6 |
| 6 | Burst duration measures the wrong interval | **CONFIRMED** → B3. The `speed_segment` indexing is also wrong, but only when acceleration decays before the speed peak |
| 7 | Y-flip in six-plus places; `from_normalized` round trip | **REFUTED, both halves.** See (d) |
| 8 | Laterality sign asserted, never tested | **REFUTED.** Convention is correct and the two modules agree. See (d) |
| 9 | Thigmotaxis percentages need not sum to 100 | **CONFIRMED** → B9, plus a new sub-finding in the warning denominator |
| 10 | Silent NaN | **CONFIRMED** → B10 |
| 11 | Bout metric edge cases | **SPLIT.** The `x_end` clamp is **REFUTED** (see (d)); the `_compute_heading_change` fallbacks are **CONFIRMED and much worse than suspected** → B8 |
| 12 | `min_valid_percentage = 0.01` | **CONFIRMED** → B11 |

---

# (c) Metrics verified correct — what you can trust

Every item below was checked against an analytically known answer and matched.
All are covered by passing tests in
[`tests/test_metric_correctness.py`](tests/test_metric_correctness.py).

### Exact on synthetic ground truth

> **Read this table with B16 and B17 in mind.** Two rows below are exact on
> synthetic input and unusable on real input: `NetDisplacement` (NaN for half
> the real fish) and `MaxSpeed` (set by tracking teleports). They are left in
> the table because the arithmetic *is* correct — the defect is in what the
> arithmetic is applied to.

| Metric | Case | Expected | Actual |
|---|---|---|---|
| `TotalDistance` | 599 steps × 2 px, 50 px/BL | 23.96 BL | **23.96** |
| `NetDisplacement` ⚠️ B16 | same, straight line | 23.96 BL | **23.96** |
| `MeanSpeed` / `MedianSpeed` | 2 px/frame, 30 fps | 1.2 BL/s | **1.2** |
| `MaxSpeed` ⚠️ B17 | same | 1.2 BL/s | **1.2** |
| `StdSpeed` | constant velocity | 0 | **3.8e-15** |
| all four, under cm calibration | 10 px/cm | 59.8 cm, 6.0 cm/s | **exact** |
| `PathStraightness` | straight line | 1.0 | **1.0** |
| `PathStraightness` | circler, 1 s window = 29 steps: sin(29ω/2)/(29 sin(ω/2)) | 0.9147648611 | **0.9147648611** (16 s.f.) |
| `MeanAngularVelocity_deg_s` | circler at 0.05 rad/frame | 85.9436692696 | **85.9436692696** |
| `LateralityIndex` | pure right-circler / left-circler | +1.0 / −1.0 | **+1.0 / −1.0** |
| `MeanNND_BL` | 3 fish, 150 px triangle legs | 3.0 BL | **3.0** |
| `MeanIID_BL` | same, (2+√2)/3 × 3 BL | 3.41421 BL | **3.41421** |
| `HullArea_BL2` | same, ½ × 3² | 4.5 BL² | **4.5** |
| `Bout_Count` | 10 engineered darts | 10 | **10** |
| `Bout_Duration_Median_ms` | 5 frames at 30 fps | 166.67 ms | **166.67** |
| `IBI_Median_ms` | 25-frame pauses, **no dropout** | 833.33 ms | **833.33** |
| `Bout_Displacement_Median` / `Bout_Distance_Median` | 5 × 6 px darts | 0.6 BL | **0.6** |
| thigmotaxis zone assignment | fish at arena centre / 50 px from wall | 100% centre / 100% border | **exact** |

### Robust as well as correct

- **`PathStraightness` is the one turning-adjacent metric that survives noise.**
  On the real sessions it moves from 0.658–0.852 (smoothing off) to 0.669–0.868
  (on) — a **1.4% change** where angular velocity moves 36.8%. Trustworthy.
- **`LateralityIndex` survives symmetric noise.** At 0.3 px jitter on a straight
  swimmer it reports −0.0100 (truth 0); at 0.1 px, +0.0201. Noise dilutes a real
  bias toward zero (it adds turn counts to both sides) but does not manufacture
  a false bias. On the real sessions it runs −0.025 to +0.047, i.e. no fish
  shows a bias. Safe to report as a null; treat any *positive* finding with
  suspicion given the dilution.

### Suspicions that turned out fine

- **The two speed pipelines agree.** `traja.get_derivatives()['speed']` and
  `np.diff`-on-positions produce **identical values** — both are step length ×
  frame rate in calibrated units. They differ only in convention: traja returns
  length `n` with `speed[0] = NaN` and `speed[i]` covering the interval
  *ending* at frame `i` (backward difference); `np.diff` returns length `n−1`
  with `speed[i]` covering the interval *starting* at frame `i` (forward).
  Verified with a single-jump track: the jump appears at traja index 10 and
  `np.diff` index 9. Consequence: bout frame numbers are offset by one frame
  from freeze frame numbers. Harmless for every summary statistic; worth knowing
  if you ever cross-reference frame indices between the two.
- **The speed/heading alignment comment at `processing.py:575-577` is correct.**
  `dheading[i]` is the turn at frame `i+1`, and `speed_series[1:min_len+1]`
  supplies the speed at frame `i+1`. Verified index by index.
- **The Y-flip is consistent across all six sites.** `processing.py:265`,
  `bout_analysis.py:423`, `spatial.py:160/417/559/600`, `shoaling.py:415` and
  `spatial_tab.py:548/623` all implement `video_height − y`. No disagreement.
- **`ArenaDefinition` round-trips.** `from_normalized` → `get_normalized_vertices`
  returns the input to machine precision. The asymmetry the prompt flagged is
  deliberate and correct: `vertices_pixels` is image space (Y down) and
  `vertices_bl` is plot space (Y up), so the two are Y-mirrors of each other by
  design. `to_shapely(use_bl=False)` would return the mirrored polygon, but it
  has **zero call sites** — `spatial.py:365` is the only caller and passes
  `use_bl=True`.
- **The laterality sign convention is right, and both modules agree.** A fish
  circling clockwise *as seen on screen* — which, with the camera above the tank
  looking down, is the fish turning to its own **right** — exports
  `LateralityIndex = +1.0`, `RightTurns_CW = 598`, `LeftTurns_CCW = 0`,
  `CumulativeHeading_deg = −1713.1`. The mirror image gives the exact negation.
  `Bout_LateralityIndex` agrees in sign in both directions. **This holds only
  because the camera looks down at the tank**; a mirrored optical path would
  invert it, and nothing in the file format or the code records the viewing
  geometry.
- **`_compute_bout_metrics`'s `x_end = min(end, len(x) - 1)` clamp does not
  collapse.** In the only call path, `len(speed) == len(x) − 1`, so `end` never
  exceeds `len(x) − 1` and the clamp is a no-op. A terminal bout returns the
  right displacement and distance (verified: 3.0 BL for 25 × 6 px). The clamp is
  fragile — it would silently zero a bout if `speed` were ever passed at the
  same length as `x`, which is exactly what `traja`'s pipeline produces — but it
  is not a live defect. Pinned by
  `test_a_bout_running_to_the_end_of_the_array_keeps_its_displacement`.
- **The all-fish-tracked requirement in shoaling does not bias NND.** Costs
  coverage (B12), not accuracy: ≤1.4% on all four sessions.

---

# (d) Column-by-column verdict for `export_combined_summary_csv`

| Column | Verdict |
|---|---|
| `Group`, `File`, `FishID`, `Label` | fine (grouping is regex-inferred — Audit D/G territory) |
| `Unit` | fine for this file's columns; does **not** describe the shoaling or spatial exports (B7) |
| `ValidFrames_pct` | correct, and the only defence against B11 |
| `TotalDistance` | **trustworthy** |
| `NetDisplacement` | correct when it returns a number — **NaN for 12 of 24 real fish** (B16) |
| `MeanSpeed`, `MedianSpeed` | **trustworthy** |
| `MaxSpeed` | **do not publish as-is** — set by tracking teleports, 32–321 BL/s observed (B17) |
| `PathStraightness` | **trustworthy**, window-dependent and the window is not exported (B13) |
| `FreezeCount` | **do not publish** — set by the dropout policy (B5) |
| `FreezeTotalDuration_s` | wall-clock denominator (B6); inherits B5's run-splitting |
| `FreezeMeanDuration_s` | inherits B5 |
| `FreezeFraction_pct` | valid-frame denominator (B6); the most defensible of the four |
| ~~`BurstCount`, `BurstFrequency_per_min`~~ | **REMOVED 2026-08-01** — noise, 39% duty cycle (B2) |
| ~~`BurstMeanSpeed`~~ | **REMOVED 2026-08-01** (B2, B3) |
| ~~`MeanAngularVelocity_deg_s`~~ | **REMOVED 2026-08-01** — measured centroid noise (B1) |
| ~~`ErraticMovementCount`, `ErraticMovements_per_min`~~ | **REMOVED 2026-08-01** (B1); also indistinguishable from a crash (B10) |
| `LateralityIndex`, `RightTurns_CW`, `LeftTurns_CCW` | **trustworthy for a null**; diluted toward zero by noise, so a positive result needs corroboration |
| ~~`CumulativeHeading_deg`, `MeanSignedAngVel_deg_s`~~ | **REMOVED 2026-08-01** — magnitudes inherit B1 and the *sign* flips for 4 of 24 real fish under smoothing (B15) |
| `Bout_Analyzed`, `Bout_Count`, `Bout_Rate_per_min` | count is dropout-inflated (B4); the construct also does not fit adult swim (B13) |
| `Bout_Duration_*` | correct on clean input; contaminated by B4's split bouts |
| `IBI_Median_ms`, `IBI_Q1_ms`, `IBI_Q3_ms` | **do not publish** — 18–86% of intervals are tracking gaps (B4) |
| `Bout_PeakSpeed_*`, `Bout_Displacement_Median`, `Bout_Distance_Median` | **trustworthy per bout**, conditional on the bout boundaries being real |
| `Bout_MeanAbsTurnAngle_deg` | diluted by 490 guard-zero bouts (B8) |
| `Bout_LateralityIndex`, `Bout_N_Left`, `Bout_N_Right` | **trustworthy**; sign verified |
| `Bout_N_Straight` | **do not publish** — 77% guard returns (B8) |

Shoaling and spatial exports: `MeanNND_BL`, `MeanIID_BL`, `HullArea_BL2`,
`MeanHullArea_BL2` are **numerically exact** but in a per-file unit that
calibration cannot override (B7). `MeanPctInBorder` / `StdPctInBorder` are
correct for in-arena positions and silently deflated by any out-of-arena share,
which is not exported (B9).

---

# (e) The regression suite

Two new files, both self-contained — no real session data, no network, no
writes outside pytest's tmp dirs. Full suite runs in 5.2 s.

```bash
pytest -q
```

```
47 passed, 13 xfailed in 5.19s
```

- **[`tests/synthetic_tracks.py`](tests/synthetic_tracks.py)** — the trajectory
  builders, each documenting its ground truth: `straight_line`, `circler`
  (with an explicit on-screen rotation direction so the turn-sign tests mean
  something), `stationary`, `jittered_straight_line`, `discrete_bouts`,
  `single_swim_event`, `three_fish_fixed_geometry`, `with_dropout`,
  `with_scattered_dropout`, plus `make_file` to wrap an array in a
  `LoadedTrajectoryFile` without touching disk.
- **[`tests/test_metric_correctness.py`](tests/test_metric_correctness.py)** —
  24 passing tests asserting the verified-correct values in (c), and 13
  `xfail(strict=True)` tests asserting the value each broken metric *should*
  return. `strict=True` means an unexpected pass is a failure, so fixing a
  defect fails its test until the marker is removed. Each marker names its
  finding and quotes the observed magnitude.

The 13 xfails map to findings as: B1 ×3, B2 ×1, B3 ×2, B4 ×1, B5 ×1, B6 ×1,
B7 ×1, B8 ×1, B9 ×1, B10 ×1.

**One pre-existing flake, noticed in passing and left alone.** Over five full
runs, `test_smoke.py::test_gui_constructs` passed three times, skipped once
("no display available"), and once failed with
`_tkinter.TclError: Can't find a usable init.tcl`. It builds its own `Tk()`
root rather than using the shared `_gui_app` fixture that `conftest.py` added
precisely to avoid Tcl refusing repeated interpreters — its `except tk.TclError`
guard covers the probe root but not the `EnhancedFishAnalyzer()` construction
that follows. Nothing in this pass touches tkinter, and the new tests are
deterministic across all five runs. Flagging it as Audit D territory, not
fixing it here.

---

# What I could not settle

- **The camera's viewing geometry is not recorded anywhere.** The laterality
  sign convention is correct for a camera above the tank looking down. Nothing
  in the `.npy`, the code, or the docs states this, and a mirrored optical path
  (bottom-mounted camera, or a mirror rig) would invert `LateralityIndex`
  without any signal. What would settle it: one frame of source video plus a
  note on the rig. The source video is not on this machine — Audit A's addendum
  records that the path stored in each `.npy`'s unused `video_paths` key points
  at a different user's OneDrive.
- **The true centroid noise level of idtracker.ai on this rig.** I bounded it
  indirectly: 0.4–0.9 px reproduces the observed `MeanAngularVelocity_deg_s`
  and `BurstCount`. A direct measurement needs a stationary reference object
  tracked through a real recording, or the `list_of_blobs.pickle` centroids
  compared against a hand-labelled subset. `list_of_blobs.pickle` *is* present
  (41 MB per session) but reading it needs `idtrackerai` importable, which
  Audit A found lives in a separate Python 3.13 conda environment.
- **Whether `id_probabilities` would fix B5 and B4.** The `(n_frames, n_fish, 1)`
  per-frame tracking confidence noted in `AUDIT_PLAN.md` is present in every
  `.npy` and read by nothing. It is the obvious input for distinguishing
  "absent" from "tracked unreliably", which is precisely the distinction the
  NaN conventions fail to make. I did not test it because using it would be a
  fix, and this pass reports rather than fixes.

---

# If you fix in one order

Not a plan — an observation about how these interact, since the prompt says
you want to choose.

1. **B1 and B2 are the same defect** (no noise floor on differentiated
   centroid positions) and one displacement threshold fixes both. Until then,
   five columns should not leave the building.
2. **B4 and B5 are the same defect** (NaN policy) in two modules with opposite
   signs. Fixing them separately guarantees they drift again; the durable fix
   is one shared "contiguous tracked segments" helper both call, which also
   makes B6's denominator question answer itself.
3. **B7 is orthogonal** and is a one-line change per site (`1.0 / body_length`
   → `calibration.scale_factor`) plus unfreezing the `_BL` column names — but
   it touches the ~10 duplicated sites `AUDIT_PLAN.md` inventoried, which is
   why `AUDIT_H_APPROACH.md` Step 4 (`to_tidy()` + one exporter) is the right
   place for it rather than a spot fix.
4. **B3, B8, B9 and B10 are independent** and each is local.
5. **B11 and B13 are policy questions for you**, not bugs to fix.

The regression suite is written so that each fix flips a named `xfail` to a
pass. If a fix flips one that it should not have touched, `strict=True` makes
that loud.

---

# What changed on 2026-08-01 <a id="what-changed-on-2026-08-01"></a>

Phase 0 of the remediation: **stop exporting what cannot be defended.** The
decision was to remove rather than flag or hide — fewer metrics that are
correct beats more metrics that need a footnote.

## Eight columns withdrawn

| Column | Finding | Why removal rather than repair |
|---|---|---|
| `MeanAngularVelocity_deg_s` | B1 | Noise-limited at the centroid level. No threshold recovers a signal that is not there. |
| `ErraticMovementCount` | B1 | Same heading, same noise. |
| `ErraticMovements_per_min` | B1 | " |
| `BurstCount` | B2 | Detector fires on 39% of real frames. |
| `BurstFrequency_per_min` | B2 | " |
| `BurstMeanSpeed` | B2, B3 | Noise-driven, and the peak-speed slice misses the peak. |
| `CumulativeHeading_deg` | B15 | Sign flips for 4 of 24 real fish under a display toggle. |
| `MeanSignedAngVel_deg_s` | B15 | " |

The three burst columns are the one group that *could* have been repaired —
the defects are a missing noise floor, a wrong duration definition and a larval
threshold. Repairing all three amounts to writing a new detector, so they were
removed with the rest and left as a clean slate. Peak swim speed **is**
recoverable from centroid data; a replacement should start from
`synthetic_tracks.single_swim_event`, which is retained unused for exactly that
purpose and encodes the ground truth a burst duration should return.

## Files touched

| File | Change |
|---|---|
| `fish_analyzer/processing.py` | `_calc_burst_metrics` deleted; `_calc_movement_direction_metrics` reduced to laterality + turn counts; `burst_accel_threshold` and `erratic_turn_threshold` removed from `ProcessingParameters` and its validation; docstrings rewritten to say what is *not* computed and why |
| `fish_analyzer/export.py` | eight columns dropped from both exporters; stale column list in the docstring corrected (B14) |
| `fish_analyzer/gui/analysis_tab.py` | columns dropped from the File Averages and Per-Fish Details tables; behavioural comparison went from a 2×3 grid with an empty cell to a 2×2 of distance / freezing / straightness / **laterality** (previously table-only); methods panel now states the omission and its reason instead of describing the withdrawn metrics |
| `fish_analyzer/gui/data_tab.py`, `fish_analyzer/__init__.py` | metric lists in help text |
| `README.md` | current-state descriptions updated; the 2.1.0 changelog left intact but struck through with a withdrawal note, so release history is not rewritten |

`burst_accel_threshold` and `erratic_turn_threshold` were safe to delete: they
were never bound to a GUI widget, only printed in the methods panel.

## Left alone deliberately

- **Bout analysis stays**, per the scoping decision — it is correct for larvae
  and the fixes for B4 and B8 are worth making. It should carry a UI note that
  it targets larval locomotion.
- **B4, B5, B6, B7, B8, B9, B10, B16, B17 are unfixed.** All are live findings
  with `xfail` tests waiting.
- **The GUI still computes and plots nothing withdrawn**, but the Bout tab's
  own laterality and heading-change plots are untouched.

## Verification

```
pytest -q
57 passed, 9 xfailed in 4.65s
```

The two `xfail` counts moved for opposite reasons: six markers were **deleted**
along with the metrics they described (a withdrawn metric has no defect left to
track), and two were **added** for B16 and B17. What replaced the six is a pair
of parametrised guards —
`test_noise_dominated_metrics_are_not_computed` (9 keys) and
`test_noise_dominated_columns_are_not_exported` (both exporters) — which fail
if any withdrawn quantity reappears in `fish.metrics` or in a CSV header.

End to end on all four real sessions: 24 rows, 38 columns, none of the
withdrawn eight present, every retained column populated.

## Next

Phase 1 is the shared `contiguous_tracked_segments` helper, which closes B4,
B5 and B6 together and is a prerequisite for trusting `FreezeCount` or any
bout timing. B16 and B17 are small enough to ride along with it.

---

# Phase 1, 2026-08-01 — one definition of a tracking gap

Phase 0 removed what could not be defended. Phase 1 fixes what could: the
findings that all traced back to the same question being answered three
different ways.

## The change

A new module, [`fish_analyzer/segments.py`](fish_analyzer/segments.py), owns
one rule: **a frame in which idtracker.ai did not locate the fish is
unobserved — not "still", not "moving".** Every run-length metric now works
inside stretches of continuous tracking and never across them, and gap frames
enter no numerator and no denominator.

It also owns the off-by-one that made this dangerous to fix piecemeal. The
package derives speed two ways — `traja` gives a backward difference where
`speed[i]` spans frames `[i-1, i)`, `np.diff` gives a forward one where
`speed[i]` spans `[i, i+1)` — so a segment loses its *first* frame under one
convention and its *last* under the other. `backward_speed_slices` and
`forward_speed_slices` do that mapping so no caller has to get it right by
hand. 31 tests in [`tests/test_segments.py`](tests/test_segments.py) pin it.

## Complete vs censored episodes

The interesting problem was B5. Once tracking fragments, *"how many freezes
were there"* stops being answerable: a run that ends because the tracker
blinked might be one episode or the first half of a longer one. The old code
counted every fragment, which turned one motionless fish into 22 episodes.
Counting none throws away real data.

So episodes are now split. **Complete** ones begin and end with an observed
transition (or at the recording boundary, which is conventional) — their
duration is a measurement. **Censored** ones touch a gap — their frames still
count toward time-frozen totals, but they are not counted or averaged. The
same split applies to bouts.

| Synthetic case | Before | After |
|---|---|---|
| motionless fish, clean | 1 episode | **1 complete, 0 censored** ✓ |
| motionless fish, 5% scattered dropout | **22 episodes** | **0 complete, 22 censored, 96.7% of observed time frozen** |
| three clean 100-frame freezes | 3 episodes | **3 complete, mean 3.333 s** (truth 3.333) ✓ |
| straight swimmer through a 30-frame gap | 2 bouts, **1033 ms IBI** | **0 complete, 2 censored, no IBI reported** |

## Findings closed

| | Fix |
|---|---|
| **B4** — 18–86% of IBIs were tracking gaps | Intervals are only measured between bouts in the same segment. `IBI_N` is exported so a reader knows whether a median rests on 11 intervals or 129. |
| **B5** — freeze count set by the dropout policy | Complete/censored split, above. |
| **B6** — freeze denominators disagreed | Both are over observed time, and `observed_duration_s` is *returned from the freeze calculation* rather than recomputed, so `freeze_total_duration_s / observed_duration_s * 100 == freeze_fraction_pct` holds by construction. Verified exact on all 24 real fish. |
| **B16** — `NetDisplacement` NaN for half the fish | Measured between the first and last *tracked* positions. **Now populated for all 24** (was 12). |
| **B17** — `MaxSpeed` reported teleports | Replaced by `SpeedP99`. Real range is now **6.3–12.8 BL/s** against the old **32–321**. The raw max is retained in-memory as `speed_max_raw` for quality checks but is not exported. |

## What the real sessions look like now

All 24 fish, four sessions:

| | before Phase 1 | after |
|---|---|---|
| `NetDisplacement` populated | 12 / 24 | **24 / 24** |
| top speed | 32 – 321 BL/s | **6.3 – 12.8 BL/s** |
| B6 identity holds | never | **24 / 24** |
| freeze episodes | 1 – 55, dropout-driven | **1 – 54 complete, 0 – 7 censored** |
| IBI median | 133 – 333 ms, 18–86% gaps | **100 – 183 ms, 0% gaps** |

**Freezing came through well** — censoring is 0–7 episodes per fish against
1–54 complete, so freeze episodes on this data are largely measurable.

**Bouts did not, and that is the finding.** Complete bouts run 6–101 per fish
while *censored* bouts run 26–252 — for one fish, 11 complete against 252
censored. An adult swimming near-continuously through a recording with ~10%
dropout has almost every "bout" interrupted. This is B13 arriving from a new
direction: the numbers are now honest, and what they honestly say is that the
bout model does not fit adult free swim. The remaining 100–183 ms intervals
are genuine threshold crossings of a continuous swimmer, not gaps — B4 is
fixed, and what is left is a scoping decision, not a bug.

## Export changes

Added: `ObservedDuration_s`, `LongestGap_s`, `FreezeEpisodes_Censored`,
`Bout_Censored`, `IBI_N`.
Renamed: `MaxSpeed` → `SpeedP99`, `FreezeCount` → `FreezeEpisodes_Complete`
(the old names carried the old semantics, and a silently redefined column is
worse than a renamed one).
43 columns, up from 38.

## Verification

```
pytest -q
98 passed, 1 skipped, 4 xfailed in 4.51s
```

Four `xfail` remain, one per open finding: B7 (calibration bypass), B8 (bout
heading guards), B9 (thigmotaxis zone sum), B10 (failures indistinguishable
from measurements). End to end on all four real sessions: 24 rows, 43 columns,
every retained column populated, B6 identity exact for every fish.

## Still open

**B7, B8, B9, B10** — none share a root cause with Phase 1, and each is local.
B7 is the largest and belongs with `AUDIT_H_APPROACH.md` Step 4 rather than as
a spot fix across ten duplicated sites.

**B11 and B13 are decisions, not defects.** `min_valid_percentage = 0.01` and
`min_freeze_frames = 5` (167 ms) are both indefensible as defaults but are
yours to set. The bout censoring rates above are the strongest argument yet
for scoping the bout tab to larvae explicitly.

---

# Phase 2, 2026-08-01 — the last four findings

B7, B8, B9 and B10 are closed. **Every finding in this report is now either
fixed or a decision you have taken.** The suite has no `xfail` markers left:
113 tests, all passing.

## B7 — calibration is no longer bypassed

`shoaling.py` and `spatial.py` hardcoded `1.0 / metadata.body_length`, so NND,
IID, hull area, thigmotaxis zones and heatmap axes stayed in body lengths
however the file was calibrated. Twelve sites in all, counting the GUI.

All of them now use `calibration.scale_factor`. The proof on the real
sessions — the cm/BL ratio comes out at exactly each file's body length,
which is what "the calibration is being applied" looks like:

| session | body length | `MeanNND` in BL | in cm | ratio |
|---|---|---|---|---|
| G604_Freeswim | 71.31 px | 1.047 | 7.467 | **71.31 / 10** |
| G604_group2 | 73.01 px | 1.428 | 10.427 | **73.01 / 10** |
| H604_Freeswim | 82.22 px | 1.183 | 9.725 | **82.22 / 10** |
| H604_group2 | 79.63 px | 1.169 | 9.307 | **79.63 / 10** |

The BL column is not comparable across those four rows — that is the 15%
body-length spread showing up as a 15% difference in a supposedly fixed
quantity. The cm column is comparable, and now reachable.

Three follow-on changes were needed to make it honest rather than merely
correct:

- **`ShoalingResults` carries `unit_name`.** Displays and exports read it
  instead of assuming. Files with different calibrations plot as
  `"mixed units"` rather than silently adopting one of them.
- **Export columns lost their hardcoded suffixes.** `MeanNND_BL` → `MeanNND`,
  `HullArea_BL2` → `HullArea`, plus a `Unit` column, matching how the combined
  summary already worked.
- **`ArenaDefinition`'s `body_length` parameter is now `pixels_per_unit`**,
  which is what it always was — body length only under the default
  calibration. Worth knowing: `from_normalized` and `get_normalized_vertices`
  have **no callers** anywhere in the package. They are covered by tests and
  left in place, but they are unused API.

## B8 — unmeasurable turns are no longer "straight"

The five guard paths in `_compute_heading_change` return `NaN` instead of
`0.0`, and `compute_summary` counts them as `bout_n_heading_unmeasurable`
rather than letting `0.0` land inside the ±5° straight dead zone.
`bout_heading_change_mean_abs_deg` uses only measured turns.

On the real sessions, of 718 measurable bouts: **34 straight, 300 with an
unmeasurable turn.** Before, 637 bouts were reported straight, of which 490
were guard returns. The new `Bout_N_TurnUnmeasurable` column makes the
distinction visible instead of burying it.

## B9 — thigmotaxis percentages sum to 100

Border and centre are now shares of *in-arena* time, so they sum to 100 by
construction, and the out-of-arena share is exported as `MeanPctOutsideArena`
/ `MaxPctOutsideArena` rather than silently deflating both.

The warning's denominator is also fixed: `frames_valid` already contained the
out-of-arena frames, so dividing by `valid + outside` double-counted them and
reported a fish 50% outside the polygon as 33.3%. That understated the exact
problem the 5% trigger exists to catch.

## B10 — a NaN now says which kind it is

- Every row carries a **`Status`** column: `ok`, `partial: turning`,
  `partial: straightness`, or `excluded: <reason>`. A NaN with `Status = ok`
  is a measurement; a NaN with `Status` naming a calculation is a crash.
- `_empty_direction_metrics` distinguishes the two callers: a fish that never
  moved returns real zeros, an exception returns NaN **and** sets the flag.
  They used to be byte-identical.
- **Fish that never make it into `processed_data` now get a row.** A fish
  failing the quality gate or raising simply vanished, so a reader counted
  five fish in a six-fish recording with nothing to say otherwise.
  `LoadedTrajectoryFile.excluded_fish` carries the reason through to the CSV.

All 24 real fish report `Status = ok`, which is the answer you want from that
column — and now you can tell.

## Verification

```
pytest -q
113 passed in 4.94s
```

**No `xfail` markers remain.** Every one has been either promoted to a passing
test or deleted along with the metric it described. End to end on all four
real sessions: 24 rows, 45 columns, all `Status = ok`.

## Where this leaves the export

45 columns, from 26 at the start. Not by adding metrics — eight were removed —
but by making what was implicit explicit: the unit, the observed duration, the
longest gap, how many episodes were censored, how many turns were
unmeasurable, and whether anything failed. **The old CSV was shorter because
it was quieter about what it did not know.**

## What is left

Nothing in this report. The two open items are decisions:

- **B11** — `min_valid_percentage = 0.01` and `min_freeze_frames = 5` (167 ms)
  are both indefensible as defaults, but they are yours to set.
- **B13** — the bout model does not fit adult free swim. Phase 1's censoring
  rates (26–252 censored against 6–101 complete per fish) are the strongest
  evidence, and B8's 300-of-718 unmeasurable turns add to it.

Beyond this pass: Pass E (head detection, which is what would bring back the
withdrawn turning metrics), and the two performance tickets in
`AUDIT_PLAN.md` — of which the `VideoFrameReader` thread race is a correctness
bug, not a performance one.
