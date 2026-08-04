# Pass G — User experience and interface polish

**Shared rules for every audit pass:**
- **Report, don't fix.** This pass ends in a findings file, not a diff.
- **Verify, don't inherit.** The seed findings below came from reading, not running. Confirm or refute each.
- **Label every finding** `CONFIRMED` / `PLAUSIBLE`.
- **Report what's good too** — and here that matters more than usual. Some of this UI is thoughtfully designed and I don't want a redesign that throws it away.
- Read `AUDIT_PLAN.md` in the repo root first for the stock-take context.

---

## Goal

Make this feel like a tool, not a script with buttons.

The bar to aim at: **a lab member who is not me can sit down with a folder of idtracker.ai sessions and produce a correct, exported analysis without me next to them** — and enjoy doing it.

This pass is about what the researcher experiences. Pass D covers the GUI's internal code structure; don't duplicate its work. Where a UX problem has an architectural cause, name the cause in one line and cross-reference.

## Scope

The whole interactive surface: all six tabs, the arena-drawing interaction, the inspector, the export flow, error and empty states, and the two standalone scripts' tkinter dialogs.

## Method

Walk the actual workflow end to end, in order, as a new user would — load → calibrate → set parameters → process → analysis tab → bout tab → shoaling → spatial/arena → inspector → export. Report friction at each step, with the specific widget and `file:line`. If you can run the GUI, do, and include screenshots. If you can't, say so and work from the code — but say which findings are unverified as a result.

## Seed findings — verify each

### Feedback and trust

1. **The status bar is a single line where the last message wins.** All analysis progress goes through `print()`, redirected by `GUILogRedirector` (`gui/base.py:159`). It keeps 100 lines of history in `_log_lines`, and `get_log()` is never called from anywhere — that history is unreachable. So a user who looks away during a run cannot recover what happened. Verify, then propose the fix: a real scrollable log panel, a per-run summary, or both.
2. **The window freezes during analysis.** No threading anywhere in the GUI; `data_tab.py:601-622` calls `root.update()` inside the processing loop to keep the window from going grey. Report what the user sees during a long run — especially thigmotaxis, which has **no progress bar at all** (the repo's only `Progressbar` is in `data_tab.py`). A frozen window with no progress indication reads as "crashed" to a user, which is a UX failure even when the computation is fine.
3. **76 modal `messagebox` calls**, 26 in `spatial_tab.py` and 25 in `data_tab.py`. Assess how many are genuine decisions requiring a modal versus things that should be inline validation, a status message, or simply prevented. Modal-dialog density is a good proxy for "the UI lets you do the wrong thing then scolds you".
4. **Error message quality.** Sample a dozen error paths. Does the message tell the user what to *do*, or just what went wrong? Does any error leave the app in a half-configured state?

### Workflow

5. **The six tabs have an implicit required order** (you must load and calibrate before analysis, define an arena before thigmotaxis) but nothing in the UI communicates or enforces it. What happens when a user clicks straight into the Bout tab on a fresh launch? Report the actual behaviour for each tab, and propose how to make the sequence legible — disabled tabs, a checklist, inline prerequisites.
6. **Nothing persists between sessions.** There is no settings/config save anywhere in the repo. Calibration, arena definitions (`file_arena_definitions`), group assignments (`file_groups`), and all processing/bout/shoaling parameters die when the window closes. For someone processing a batch of videos across several days, re-drawing arenas by clicking vertices every session is the single largest time cost in the tool. Confirm, and propose the smallest thing that fixes it — per-session sidecar files, a project file, or a global settings file. This is likely the highest-value UX finding; treat it accordingly.
7. **No recent-files list, no drag-and-drop, no "reopen last session".** Assess the file-loading flow (`data_tab.py`) against what a repeat user needs.
8. **Arena definition is the most interaction-heavy surface in the app** — clicking polygon vertices on a background image. Evaluate it properly: can you undo a misplaced vertex? Reposition one? See the resulting border zone before committing? Copy an arena to another file with different video dimensions (`ArenaDefinition.copy()` and `from_normalized()` exist — is either reachable from the UI)?
9. **Silent experimental grouping.** `export.py:50` infers group membership by regex-stripping trailing digits from the filename. The user is never shown what grouping was inferred before the CSV is written. Verify, and propose how to surface it.
10. **Export flow.** CSV export buttons are scattered per tab. Can a user tell what's already been exported, where it went, and with which parameters? Note that exported CSVs record `Unit` but not the thresholds or smoothing settings used — a provenance gap that's simultaneously a UX problem ("which settings produced this file?") and a reproducibility one.

### Polish

11. **Visual consistency.** Fonts and colours are set inline throughout (`("Arial", 9)`, `fg="gray40"`, etc.) with no shared style definition. Report the inconsistencies a user would notice: mismatched fonts, padding, button sizes, alignment across tabs.
12. **Window and layout behaviour.** `base.py:118-123` caps the window at 1200×850. Test on a small laptop screen and a large monitor. Does anything clip, or fail to expand? `data_tab.py` wraps content in a scrollable canvas — do the others need it?
13. **Table and plot ergonomics.** `create_sortable_treeview` (`gui/utils.py:15`) sorts numerically by stripping `%` and `,` — check how it handles NaN and mixed content. Are plots readable, labelled with units, and exportable at publication quality?
14. **Standalone scripts' dialogs.** `fish_posture_analyzer.py:340` asks "select a session folder?" via a yes/no `askquestion` to decide *which kind of dialog to open next* — a confusing pattern. Note it, since these scripts may end up in the GUI (see Pass E).

## What to preserve

Call out explicitly what already works well so a redesign doesn't discard it. From my read, at least these are genuine strengths:

- The plain-language "methods text" panels explaining each analysis (`analysis_tab.py:217`, `bout_tab.py:217`) — unusually good for a research tool, and worth extending rather than replacing.
- Graceful degradation on missing optional dependencies (`SHAPELY_AVAILABLE`, `CV2_AVAILABLE`) — features disable instead of crashing.
- The trajectory-trail slider on the frame viewer.
- Per-file calibration rather than one global setting.

Add to this list anything else you find. I want the report to make the good parts as visible as the bad ones.

## Deliverable

`AUDIT_G_UX.md` in the repo root, containing:

- **(a)** A walkthrough of the full workflow with friction points in order, each tagged with severity (blocks work / slows work / feels unfinished).
- **(b)** Findings ranked by **time saved per week for a real user**, not by visual appeal. Persistence and progress feedback almost certainly outrank fonts.
- **(c)** A "preserve this" section.
- **(d)** Concrete proposals: for each significant finding, what the fixed version looks like — described specifically enough to implement, with a sketch or mockup where it helps.
- **(e)** A quick-wins list: changes under ~20 lines that a user would notice immediately.

## Constraints

- Report only, no changes.
- **This must stay a tkinter app.** Do not propose migrating to Qt, a web front-end, or Electron. It has to run in a conda env on lab Windows machines with no build step. Work within tkinter/ttk and matplotlib.
- Respect that the users are researchers, not software users by trade. Favour explicitness and hard-to-misuse flows over cleverness or density.
