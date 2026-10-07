# Pass C — One source of truth, and code cleanliness

**Shared rules for every audit pass:**
- **Report, don't fix.** This pass ends in a findings file, not a diff. Fixes are a separate decision.
- **Verify, don't inherit.** The seed findings below came from reading, not running. Confirm or refute each, and extend the lists — my counts are floors, not ceilings.
- **Label every finding** `CONFIRMED` / `PLAUSIBLE`.
- **Report what's fine too.** Say which modules are clean; I want to know where not to spend effort.
- Read `AUDIT_PLAN.md` in the repo root first for the stock-take context.

---

## Goal

Two related things:

1. **One source of truth.** The same physical quantity is computed several different ways in several places, free to drift apart. Map every instance and say which implementation should be canonical.
2. **Cleanliness.** Make this codebase pleasant to change. It's 11,787 lines written across 15 commits with no tests, and I'm the only maintainer.

Best run after `AUDIT_B_CORRECTNESS.md` exists — Pass B decides which competing implementation is *right*, this pass decides where it should *live*. If B hasn't run, flag any consolidation whose correct target is still unknown rather than guessing.

## Scope

Whole repo, including the GUI.

---

## Part 1 — Duplicated and divergent logic

Verify each, and map exhaustively — don't stop at my list.

1. **Speed.** `traja.get_derivatives()` in `processing.py:309` vs `np.diff`-based in `bout_analysis.py:428`. Find every other site where speed is computed, including in the GUI (`inspector_tab.py` computes its own — check around lines 1144 and 1306).
2. **Pixel→unit conversion.** `calibration.scale_factor` vs hardcoded `1.0 / metadata.body_length`. I count ~10 sites across `shoaling.py:204`, `spatial.py:344`, `spatial.py:531`, `gui/spatial_tab.py` (5 sites), `gui/inspector_tab.py:993`, `gui/analysis_tab.py:937/980`. Get the real list and mark which are correct. This one has correctness consequences — cross-reference Pass B finding 1.
3. **Y-axis flip.** At least six independent implementations. Map all of them; check they agree.
4. **Heading / turn-angle computation.** Three implementations of "which way did the fish turn": `processing.py::_calc_movement_direction_metrics`, `bout_analysis.py::_compute_heading_change`, and whatever `head_detection/validation_video.py` does. Note whether they share a sign convention.
5. **Smoothing.** At least four implementations: `traja.smooth_sg` (Savitzky-Golay) in `processing.py:233`; `uniform_filter1d` (boxcar) in `gui/utils.py::smooth_time_series`; and two more inside `spatial.py` — `get_smoothed_group_timeseries` and `get_smoothed_fish_timeseries`. Different filter families used interchangeably on the same kinds of series.
6. **Run-length / interval detection.** The same "find consecutive True runs" loop is hand-written three times: `processing.py::_calc_freeze_metrics` (line 419), `processing.py::_calc_burst_metrics` (line 484), `bout_analysis.py::_find_intervals` (line 161). All three are Python loops over every frame — cross-reference Pass F.
7. **Skeleton endpoint detection.** The identical 3×3 `kernel == 11` trick appears in `fish_posture_analyzer.py:207` and `head_detection/head_detection_test.py:62` (and likely `validation_video.py`).
8. **Export row construction.** `export_combined_summary_csv` and `export_individual_metrics_csv` build overlapping row dicts by hand — ~25 duplicated keys. Assess drift risk. Also note `export_combined_summary_csv` is **not** re-exported in `fish_analyzer/__init__.py` while its sibling is, and `gui/analysis_tab.py:23` imports it from the submodule directly.

For each concept, produce a table: every site (`file:line`), what it computes, whether the sites agree numerically, and which should be canonical.

---

## Part 2 — Cleanliness

- **Analysis logic living in the GUI.** Anything under `fish_analyzer/gui/` that computes a number rather than drawing one belongs in the package. Find it all. This is the highest-value cleanliness finding because it's also what makes the analysis untestable.
- **Dead code.** `fish_analyzer/backup/gui.py.backup` (3,318 lines). `GUILogRedirector.get_log()` (`gui/base.py:61`) is defined and never called anywhere — the 100-line log history it maintains is unreachable. Find the rest: unused imports, unreachable branches, superseded helpers, dataclass fields never read.
- **Error-handling idioms.** ~37 `except Exception` blocks repo-wide. Classify each: legitimate boundary (optional dependency, user-facing guard) vs. swallowed bug. Flag every one that returns a plausible-looking value on failure — those are the dangerous ones.
- **Magic numbers.** `_MIN_DISP = 0.05` (`bout_analysis.py:255`), the 5° laterality dead zone (`bout_analysis.py:360`), `CROP_SIZE = 158` (head_detection), `min_valid_percentage = 0.01`, the 15% border default, `MEDIAN_KERNEL = 5`. Which are principled, which are tuned-to-one-dataset, and which should be surfaced as parameters?
- **Type hints and docstrings.** Coverage is uneven — some modules are thoroughly annotated, others not. Report the gaps that actually matter (public API surface, dataclass fields, anything crossing a module boundary), not a blanket "add type hints everywhere". Flag docstrings that describe behavior the code doesn't have (the `# ENHANCED (v2.0)` comments throughout `spatial.py` are archaeology, not documentation).
- **Naming.** Is `EnhancedFishAnalyzer` still a meaningful name? Is `fish_analyzer` vs `fish_posture_analyzer` vs `head_detection` a coherent scheme? Inconsistent `BL` suffixing on variables that may not be in BL.
- **Import-time side effects.** `print()` on missing shapely (`spatial.py:49`) and missing cv2 (`video_utils.py:26`); `matplotlib.use('TkAgg')` at `gui/base.py:17`; tkinter dialogs at module level in the standalone scripts; `head_detection_test.py` runs its entire pipeline on import. Enumerate all of them.
- **Whitespace/formatting consistency.** Note whether a formatter (black/ruff) would produce a large diff, and whether adopting one is worth the one-time churn. Recommend, don't apply.

---

## Deliverable

`AUDIT_C_CLEANLINESS.md` in the repo root, containing:

- **(a)** One table per duplicated concept, as specified above.
- **(b)** A proposed consolidation order: smallest-risk / highest-payoff first, with the blast radius of each (how many call sites move).
- **(c)** Explicit flagging of any consolidation that **would change exported numbers**. Those need my sign-off, not a refactor — call them out in their own section.
- **(d)** The cleanliness findings, grouped, with a rough effort estimate per group.
- **(e)** A short list of what's already clean and should be left alone.

## Constraints

- No refactoring in this pass. Report only.
- Prefer "delete this" over "abstract this" where both are available — this is a one-maintainer lab tool, not a framework. Be skeptical of proposing new abstraction layers.
