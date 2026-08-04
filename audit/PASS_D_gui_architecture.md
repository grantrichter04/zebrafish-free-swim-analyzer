# Pass D — GUI architecture and failure visibility

**Shared rules for every audit pass:**
- **Report, don't fix.** This pass ends in a findings file, not a diff.
- **Verify, don't inherit.** The seed findings below came from reading, not running. Confirm or refute each.
- **Label every finding** `CONFIRMED` / `PLAUSIBLE`.
- **Report what's fine too.**
- Read `AUDIT_PLAN.md` in the repo root first for the stock-take context.

---

## Goal

Understand and improve the structure of the GUI layer. It's 6,698 lines — 57% of the codebase — across 9 files, and it's where I spend the most time making changes and the most time confused.

This pass is about *code structure and failure handling*. The separate Pass G covers what the researcher experiences. Don't do G's job here; if you find a UX issue, note it in one line and move on.

## Scope

`fish_analyzer/gui/` entirely, plus `run_analyzer.py`.

## Specific questions

1. **The mixin design.** `EnhancedFishAnalyzer` (`gui/__init__.py:29`) combines `GUIBase` plus six tab mixins, all sharing `self` with no declared interface. Map which attributes each mixin reads and writes. **Every attribute touched by more than one mixin is the real, undocumented contract** — produce that set explicitly, as a table. Then judge: is this holding up, or is it the source of the bug clusters visible in the git log? Commit `9ad0345` is literally "Fix 4 bugs: IBI display, slow-fish heading, inspector status, smoothing warning" — check whether those were cross-mixin state bugs.

2. **Oversized files.** `inspector_tab.py` 1,682 lines, `spatial_tab.py` 1,299, `analysis_tab.py` 1,163. For each, propose a split along real seams — state management / plotting / export / event handling — not arbitrary line counts. If a file is long but coherent, say so and leave it.

3. **Failure visibility. Treat this as the priority finding of the pass.**
   `GUILogRedirector` (`gui/base.py:159`) replaces `sys.stdout` process-wide for the lifetime of the app, and all analysis progress is `print()`. The status bar is a **single line where the last message wins**. `get_log()` maintains 100 lines of history and is never called from anywhere — verify that.
   Trace at least two real error paths end to end and report exactly what the user sees:
   - an exception raised inside an analysis called from a button callback
   - an exception raised inside a matplotlib event handler (e.g. arena-drawing clicks in `spatial_tab.py`)
   Then answer plainly: **can a researcher run an analysis, get a silently degraded result, and not know?** In a research tool that's the failure mode that matters — a swallowed exception means someone publishes a NaN.

4. **Blocking the UI thread.** There is no threading anywhere in the GUI. Long analyses run directly in tkinter callbacks, and `data_tab.py:601-622` calls `self.root.update()` inside the processing loop to keep the window alive. Assess:
   - the reentrancy hazard — `root.update()` processes pending events, so a user can click buttons (including "Process" again) mid-computation and re-enter the handler
   - `set_status()` calling `update_idletasks()` on every status change (`base.py:164`)
   - what the window does during a long thigmotaxis run (no progress bar exists for it — the only `Progressbar` in the repo is in `data_tab.py`, for batch processing)
   - whether moving analyses to a worker thread is feasible given matplotlib/tkinter constraints, or whether a cooperative-chunking approach is the realistic option. Give one recommendation.

5. **Stale derived state.** Look for paths where the display and the underlying numbers can disagree:
   - user changes calibration *after* running an analysis — do cached `processed_data`, `shoaling_results`, `thigmotaxis_results` get invalidated?
   - re-running processing while bout results from a previous run are still held
   - loading a second file mid-analysis
   - arena defined for file A then applied to file B with different video dimensions
   For each, say whether the user gets stale numbers, an error, or correct invalidation.

6. **Duplicated widget scaffolding.** CSV export buttons, figure setup, file-selector listboxes, and methods-text panels are repeated across the six tabs. `gui/utils.py` is only 165 lines, so most of this is inline. Quantify the duplication and propose the specific shared helpers — with the line count each would remove.

## Deliverable

`AUDIT_D_GUI.md` in the repo root, containing:

- **(a)** Failure-visibility findings **first**, with the traced error paths written out step by step.
- **(b)** The cross-mixin shared-attribute table — the implicit contract.
- **(c)** Stale-state findings, each as a concrete reproducible sequence of user actions.
- **(d)** A decomposition proposal per oversized file.
- **(e)** An honest verdict on the mixin approach: keep and tighten, or restructure? **Give one recommendation with reasoning, not a menu of options.** If restructuring, sketch the target and estimate the effort.

## Constraints

- Report only, no refactoring.
- Be realistic about scope. This is a working lab tool with one maintainer and no tests — a proposal that requires rewriting the GUI before anything else can ship is not useful. Sequence your recommendations so the highest-value change is also independently shippable.
