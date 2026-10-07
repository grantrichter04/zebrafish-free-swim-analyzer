# Pass E — Where head_detection and the posture analyzer belong

**Shared rules for every audit pass:**
- **Report, don't fix.** This pass ends in a findings file, not a diff.
- **Verify, don't inherit.** The seed findings below came from reading, not running. Confirm or refute each.
- **Label every finding** `CONFIRMED` / `PLAUSIBLE`. Be explicit where you could not verify an idtrackerai API without the package installed.
- Read `AUDIT_PLAN.md` in the repo root first for the stock-take context.

---

## AMENDMENT — 2026-07-31, before you start

Three things have changed since this prompt was written. Read them first;
they change what this pass is for.

**1. Check `polavieja_lab/midline` before deciding where anything belongs.**
`AUDIT_H_APPROACH.md` found that the idtracker.ai authors publish `midline`,
which already extracts nose/tail/midline from idtracker.ai sessions — and that
`fish_posture_analyzer.py`'s own docstring says its midline code is "adapted
from the idtrackerai sample code". So **"vendor the upstream implementation and
keep only our head/tail disambiguation" is a live answer**, and it changes what
you are finding a home for. Evaluate that before answering the placement
question. This is now the most valuable thing this pass can settle.

**2. Data status.** Four real sessions are available — see the data-kit section
of `AUDIT_PLAN.md` for the path. They have `list_of_blobs.pickle` (41 MB) and
`session.json`, so the head-detection half is unblocked. They have **no
`individual_videos/`**, so the posture-analyzer half cannot be run. Report what
you could not execute rather than reasoning about it as if you had.

**3. Pass A's A10 is yours.** `head_detection/` still has hardcoded personal
paths at module level, no `__main__` guards, and a tkinter dialog at import
time. Pass A deliberately left these to you because they are entangled with the
placement decision.

## Goal

Decide whether the two standalone modules should be folded into the `fish_analyzer` package, kept as documented sibling tools, or rewritten against a shared core — and give me one recommendation.

## Scope

`head_detection/` (4 scripts, 915 lines) and `fish_posture_analyzer.py` (391 lines), plus their relationship to `fish_analyzer/`.

## Context

These are the newest work in the repo — the last two commits — and the direction the science is going: real head direction and body posture, rather than turn angles inferred from centroid movement. Right now neither is reachable from the GUI, and neither shares the package's calibration, Y-flip, or unit conventions.

## Assess

1. **Scientific overlap.** `head_detection/validation_video.py` does turn analysis from actual head direction. `processing.py` and `bout_analysis.py` infer turns from centroid displacement. Are these measuring the same construct? If the head-based measure is better, say so plainly, and say what it means for the centroid-based metrics — do they get superseded, or kept as a cheap fallback for when only centroids are available? This is the question that decides the whole pass; answer it first.

2. **Input dependencies — the real constraint on integration.**
   - `head_detection/*` needs `preprocessing/list_of_blobs.pickle` **and** the `idtrackerai` package importable.
   - `fish_posture_analyzer.py` needs `individual_*.avi` crops.
   - `fish_analyzer/` needs only `trajectories/trajectories.npy`.
   Establish what each pipeline actually requires from an idtracker.ai session, and whether those artifacts are always produced and retained. A package that hard-depends on `idtrackerai` is a much heavier install than one that reads a `.npy` — quantify that cost. Consider whether an optional-dependency pattern (like the existing `SHAPELY_AVAILABLE` / `CV2_AVAILABLE` guards) is the right shape here.

3. **Convention mismatches.** Check whether these scripts agree with the package on Y-axis direction, pixel→BL scaling, and turn sign. Specifically:
   - `head_detection_test.py:160`: `x_min, y_min = bc.bottom, bc.left`. Assigning `bottom`→x and `left`→y reads like an axis swap. **Verify against the real idtrackerai `bbox_corners` API** and report what the attribute names actually mean. If it is a swap, determine whether the downstream head positions are wrong or whether a second swap cancels it out.
   - `head_detection_test.py:85`: `v = np.array([velocity_xy[1], velocity_xy[0]])` reorders (vx,vy)→(v_row,v_col) against **un-flipped** full-frame coordinates, while the package Y-flips everywhere. Check the sign consistency of the resulting head choice.
   - `fish_posture_analyzer.py:164`: the head is chosen as the *second*-highest curvature peak, on the stated assumption that the tail is the highest. Is that assumption sound across body postures, and does it agree with head_detection's distance-transform approach (`head_detection_test.py:68`, where *higher* DT = wider body = head)? Two different heuristics for the same decision — do they ever disagree?

4. **Script hygiene.**
   - `generate_mask_video.py`, `generate_individual_mask_videos.py`, and `head_detection_test.py` hardcode my OneDrive path and a specific session folder at module level.
   - `head_detection_test.py` executes its entire pipeline at import time — there's no `main()` guard.
   - `generate_mask_video.py` and `generate_individual_mask_videos.py` look like earlier iterations of the same idea. Determine whether `validation_video.py` supersedes them.
   - `head_detection/.gitignore` ignores `session_*/`, `*.avi`, and the output dirs — check whether any large artifact has slipped into git history regardless.

5. **Duplication** with the package and with each other: skeletonization, endpoint detection (the 3×3 `kernel == 11` trick appears in at least two files), temporal median filtering, mask cropping, NaN interpolation. Cross-reference `AUDIT_C_CLEANLINESS.md` if it exists.

## Deliverable

`AUDIT_E_INTEGRATION.md` in the repo root, containing:

- **(a)** **One recommendation** with reasoning: fold in / keep separate as a documented sibling tool / rewrite against a shared core. Not a menu.
- **(b)** If folding in: the minimum interface change needed, and the effect on install footprint.
- **(c)** A scientific-validity section on head-based vs centroid-based turn measurement, including the axis-swap verification and the two-competing-head-heuristics question.
- **(d)** Disposition for each of the four `head_detection` scripts — keep, merge, or delete.
- **(e)** The hardcoded-path and import-time-side-effect punch list, with `file:line`.

## Constraints

- Report only, no restructuring.
- You will likely not be able to install `idtrackerai`. Where an API question can't be settled, say so explicitly and state what would settle it — a doc link, a version number, or a session folder to inspect.
