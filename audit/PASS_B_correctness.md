# Pass B — Numerical and scientific correctness

**Shared rules for every audit pass:**
- **Report, don't fix.** This pass ends in a findings file, not a diff. Several of these findings interact and I want the whole picture before choosing.
- **Verify, don't inherit.** The seed suspicions below came from reading the code, not running it. Confirm or refute each one with evidence.
- **Label every finding** `CONFIRMED` (you executed it, or cited an authoritative doc/reference) or `PLAUSIBLE` (read-only reasoning). Never blur the two.
- **Report what's fine too.** I need to know which metrics I can trust, not just which are broken.
- Read `AUDIT_PLAN.md` in the repo root first for the stock-take context.

---

## Goal

Establish whether every behavioral metric this package exports is numerically correct and means what its column name says.

This is the highest-stakes pass. These numbers go into zebrafish neurobehavior results for the Morsch lab. A silently wrong metric is worse than a crash, because a crash gets noticed.

## Scope

`fish_analyzer/processing.py`, `bout_analysis.py`, `shoaling.py`, `spatial.py`, `export.py`. Read the GUI only where it *computes* a number rather than displaying one.

## Method

Build synthetic trajectories with analytically known ground truth and check each metric returns the right answer:

- a straight-line swimmer at constant speed
- a pure circler at a known angular rate and radius
- a stationary fish (with and without sub-pixel jitter)
- a fish with an exact known number of discrete bouts separated by known pauses
- a fish with an engineered NaN dropout of known length and position
- a known left-biased turner (and its mirror image, a right-biased turner)
- two fish at a fixed known separation (for NND/IID/hull)

**A metric that cannot be pinned down by any synthetic case is itself a finding** — report it as such.

Prerequisite: a working environment. If `AUDIT_A_REPRODUCIBILITY.md` exists, use its recipe. Otherwise build a throwaway venv first — note that `import fish_analyzer` currently fails because `traja` is missing.

## Seed suspicions — confirm or refute each with a runnable case

1. **Calibration bypass.** `shoaling.py:204` and `spatial.py:344` compute `pixels_to_bl = 1.0 / metadata.body_length` instead of using `calibration.scale_factor`. Test: calibrate a file in cm and check whether NND/IID/hull/thigmotaxis change units at all. Note that `export.py` hardcodes the column names `MeanNND_BL`, `MeanIID_BL`, `HullArea_BL2`. The same hardcoding is replicated ~6 times in `gui/spatial_tab.py` and `gui/inspector_tab.py`.
2. **Two independent speed pipelines.** `processing.py:309` takes speed from `traja.get_derivatives()`; `bout_analysis.py:428` recomputes it from `np.diff` on positions. The same default threshold `0.5` is applied to both. Quantify the disagreement on identical input — do they even have the same length and alignment?
3. **Opposite NaN conventions.** `bout_analysis.py:432` sets NaN speed to `0.0`, so a tracking dropout reads as "still". `processing.py:413` sets NaN to not-frozen, so a dropout reads as "moving". Quantify the effect of a 5% dropout on `freeze_count`, `freeze_fraction_pct`, `bout_count`, and IBI.
4. **Turn metrics on unsmoothed positions.** `apply_smoothing` defaults to `False`, and `_calc_movement_direction_metrics` (`processing.py:553`) takes frame-to-frame `arctan2` of raw positions with no displacement floor — while `bout_analysis.py:255` guards the same computation with `_MIN_DISP = 0.05` BL. Inject sub-pixel noise into a straight-line swimmer and report what `mean_angular_velocity_deg_s` and `laterality_index` return. If they're measuring jitter rather than behavior, say so unambiguously.
5. **Mixed freeze denominators.** `freeze_fraction_pct` divides freeze frames by `n_valid`; `freeze_total_duration_s` divides frames by frame rate over the whole recording (`processing.py:434-443`).
6. **Burst duration measures the wrong interval.** `_calc_burst_metrics` (`processing.py:491`) counts frames where *acceleration* exceeds threshold and reports that span as the burst duration. Also check the `speed_segment` indexing at line 493 — the comment claims an offset that the slice may not implement.
7. **Y-flip applied independently in six-plus places.** `processing.py:265`, `bout_analysis.py:423`, `spatial.py:160`, `spatial.py:417`, `spatial.py:559`, `spatial.py:600`, `shoaling.py:415`, plus GUI sites. Check they agree. Separately: `ArenaDefinition.from_normalized` flips Y on `vertices_pixels` (`spatial.py:160`) but `get_normalized_vertices` does not — test the round trip explicitly.
8. **Laterality sign convention is asserted, never tested.** `processing.py:609-615` declares CW = right = negative `dheading`, in coordinates that have *already* been Y-flipped. Establish with a synthetic known-direction turner what the exported `LateralityIndex` actually means in the real world, and whether `processing.py:614` and `bout_analysis.py:361` agree on sign. If they disagree, the two laterality columns in the combined CSV contradict each other.
9. **Thigmotaxis percentages need not sum to 100.** Positions outside the arena polygon increment `frames_valid` but neither border nor center (`spatial.py:411-434`). Confirm, then judge: defensible choice or bug? Either way it's undocumented in the export.
10. **Silent NaN.** Five `except Exception` blocks in `processing.py` return NaN or empty metric dicts; `process_all_fish` (`processing.py:190`) swallows per-fish errors with a `print`. Determine whether a reader of an exported CSV can distinguish "genuinely NaN" from "the computation raised". If they can't, that's a data-integrity finding, not a style one.
11. **Bout metric edge cases.** `_compute_bout_metrics` (`bout_analysis.py:197`) clamps `x_end = min(end, len(x) - 1)`, so `displacement` and `distance` for a bout ending at the array boundary may silently collapse to `0.0`. Also check the `len(seg_x) < 3` fallback path in `_compute_heading_change` — it returns `0.0` in several conditions, which then counts as "straight" in the laterality dead-zone at `bout_analysis.py:363`, inflating `bout_n_straight`.
12. **`min_valid_percentage` default is 0.01.** A fish tracked in 1% of frames passes the quality gate and gets full metrics exported (`processing.py:73`, `210`). Assess whether that default is defensible, and what the exported CSV tells a reader about it (`ValidFrames_pct` is present — is that enough?).

## Also required

For each metric, state whether the implementation matches the standard definition used in the zebrafish behavior literature — NND, IID, thigmotaxis, bout rate, inter-bout interval, laterality index, path straightness, angular velocity. Cite a reference where the definition is contested. Flag every place a docstring promises something different from what the code does.

## Deliverable

`AUDIT_B_CORRECTNESS.md` in the repo root, containing:

- **(a)** Findings ordered by scientific impact — for each, answer "would this change a published number, and by roughly how much".
- **(b)** For each finding: the synthetic case, the expected value, the actual value.
- **(c)** `CONFIRMED`/`PLAUSIBLE` label on every finding.
- **(d)** A separate section listing metrics you **verified as correct**. This is not optional — I need to know what I can trust.
- **(e)** The synthetic test cases as runnable files under `tests/`. These become the regression suite the repo currently lacks, so write them to outlive this audit: clear names, asserted expected values, no dependency on private data.

## Constraints

- Do not fix anything. Several of these findings interact (the calibration bypass, the two speed pipelines, and the NaN conventions all touch the same exports) and I want to choose the fix order myself.
- If a finding depends on a real `trajectories.npy` you don't have, say so and specify exactly what sample would settle it.
