# Audit D — GUI architecture and failure visibility

> **Status (2026-07-31): items 1–8 of the sequenced recommendations are applied
> and verified.** See the *Fix status* section at the end. The findings text
> below is the pre-fix snapshot, left unedited so the evidence stays readable.
> **Still open: items 9–11**, which are gated on `AnalysisConfig` (H Step 1)
> and Pass B respectively.

Pass D of [AUDIT_PLAN.md](AUDIT_PLAN.md). Scope: `fish_analyzer/gui/` (6,698 lines across 9 files) plus `run_analyzer.py`. Run 2026-07-31 against the working tree at `17fe96d`.

## How this was verified

The GUI cannot be launched here — `import fish_analyzer` still fails on the ambient interpreter (Pass A's blocker). So:

- **Static reading** of all 9 GUI files end to end.
- **AST analysis** of every `self.<attr>` load/store per mixin, to produce the shared-attribute table in §B rather than eyeballing it.
- **Library-level verification** of the two exception-swallowing mechanisms in §A2 and §A3, against the actually-installed matplotlib 3.10.8 and Python 3.11 tkinter. These are the versions the tool would run on, and the swallowing behaviour is a property of those libraries, not of a guess.

Anything whose *consequence* needs a live window (flicker, freeze duration, event-loop reentrancy in practice) is labelled `PLAUSIBLE`. Anything that is a determinate property of the code or the libraries is `CONFIRMED`.

**Headline:** the answer to "can a researcher run an analysis, get a silently degraded result, and not know?" is **yes, by at least six distinct routes**, and one of them (§C1) puts a wrong unit label on an exported CSV. The mixin architecture, by contrast, is in better shape than the plan assumed — see §E.

---

# (a) Failure visibility

## A1. The status bar is a write-only channel, and the history is unreachable — `CONFIRMED`

`GUILogRedirector` ([base.py:25-63](fish_analyzer/gui/base.py:25)) keeps the last 100 lines in `self._log_lines` and exposes them via `get_log()`. **`get_log()` is defined once and called from nowhere** — a repo-wide grep for the name returns exactly one hit, the definition itself. The history is dead code; the only surface is `status_label.config(text=...)`, one line, last writer wins.

Because `set_status()` and every `print()` write to the same single label, **any message survives only until the next one**. In a loop over files that is microseconds.

## A2. Exceptions in Tk button callbacks vanish to stderr — `CONFIRMED`

Verified against the installed Python 3.11 tkinter:

```
tkinter.CallWrapper.__call__  →  except: self.widget._report_exception()
Misc._report_exception        →  root.report_callback_exception(exc, val, tb)
Tk.report_callback_exception  →  print("Exception in Tkinter callback", file=sys.stderr)
                                 traceback.print_exception(...)   # → stderr
```

`_setup_log_redirect()` ([base.py:155-159](fish_analyzer/gui/base.py:155)) replaces **`sys.stdout` only**. `sys.stderr` is untouched. So an unhandled exception in any button command prints to a terminal the researcher probably does not have (double-click / `pythonw` launch = no console at all), the button appears to do nothing, and the app keeps running.

## A3. Exceptions in matplotlib event handlers vanish the same way — `CONFIRMED`

The arena editor's vertex-placing handler `_on_arena_click` ([spatial_tab.py:432](fish_analyzer/gui/spatial_tab.py:432)) is wired with `mpl_connect`. Verified against matplotlib 3.10.8:

```
Figure.__init__            →  self._canvas_callbacks = cbook.CallbackRegistry(signals=...)
                              # no exception_handler passed → default _exception_printer
CallbackRegistry.process   →  except Exception as exc: self.exception_handler(exc)
_exception_printer(exc)    →  if _get_running_interactive_framework() in ["headless", None]:
                                  raise exc
                              else:
                                  traceback.print_exc()      # → stderr, swallowed
```

`_get_running_interactive_framework()` walks live stack frames for `tkinter.Misc.mainloop.__code__`; under `app.run()` that is always on the stack, so it returns `"tk"` and the `raise` branch is never taken. **Every exception inside an arena or ROI click is printed to stderr and discarded.** The user clicks, no vertex appears, no error is shown.

## A4. Two traced error paths, end to end

### Path 1 — an analysis raises during batch processing

`Run All Analysis` → `_run_analysis()` ([data_tab.py:582-641](fish_analyzer/gui/data_tab.py:582)). Six files loaded, `process_and_analyze_file` raises on file 3:

| Step | What happens | What the user sees |
|---|---|---|
| 1 | Files 1–2 processed, `loaded_file.processed_data` assigned | Status line flickers through per-file `[OK]` messages |
| 2 | File 3 raises | — |
| 3 | `except Exception` at [:634](fish_analyzer/gui/data_tab.py:634) sets status to `Error: ...`, then **immediately overwrites it** with `Analysis failed: ...` at [:636](fish_analyzer/gui/data_tab.py:636) | One grey status line, 9pt, at the bottom of the window |
| 4 | `traceback.print_exc()` → **stderr** → nowhere (§A2) | nothing |
| 5 | `finally:` hides the progress bar | Progress bar disappears at 2/6, same as a normal finish |
| 6 | `_update_analysis_files_listbox()` and `_update_analysis_visualizations()` are **inside the `try`** and are skipped | Plots still show the *previous* run |
| 7 | **No `messagebox`.** The success path has one ("Processed N file(s)"); the failure path has none | Absence of a dialog is the only signal |

Files 4–6 keep `processed_data` from an earlier run, or `None`. `_export_individual_csv` filters on `if v.processed_data` — so a stale-but-truthy file **is exported**, mixed in with fresh ones, indistinguishable in the CSV. `CONFIRMED`.

Nested inside this: `TrajectoryProcessor.process_all_fish` ([processing.py:182-192](fish_analyzer/processing.py:182)) catches per-fish exceptions and `print`s `Fish N: [FAILED]`, then `continue`s. That print lands on the status bar for microseconds before the next fish overwrites it. The surviving signal is the count in `[OK] {nickname}: {len(fish_list)}/{n_fish} fish analyzed` at [data_tab.py:620](fish_analyzer/gui/data_tab.py:620) — **which is itself overwritten by the next file's line**. With 6 files, only file 6's fish count is ever readable. `CONFIRMED`.

### Path 2 — an exception inside the arena click handler

Spatial Analysis → select file → click on the arena canvas → `_on_arena_click` raises (e.g. `self.roi_status_label` destroyed by an intervening `_setup_arena_canvas_for_file`, or `arena_canvas` is `None`):

| Step | What happens | What the user sees |
|---|---|---|
| 1 | matplotlib catches it, calls `_exception_printer` | — |
| 2 | Framework is `"tk"` → `traceback.print_exc()` → stderr | nothing |
| 3 | Handler returns; `arena_vertices` is left **partially mutated** (the append at [:456](fish_analyzer/gui/spatial_tab.py:456) happens *before* the redraw that could fail) | no vertex marker drawn |
| 4 | User clicks again | a second invisible vertex is appended |
| 5 | User gives up and clicks `Complete` | An arena is built from vertices the user never saw placed |

`CONFIRMED` for the swallowing mechanism; the specific triggering conditions are `PLAUSIBLE`.

## A5. The animation loop swallows *everything*, including to stderr — `CONFIRMED`

```python
except Exception:
    pass  # Never let a render error break the animation chain
```
[inspector_tab.py:740-741](fish_analyzer/gui/inspector_tab.py:740)

This wraps `_update_inspector_info()` **and** `_inspector_update_fast()` — i.e. the entire render path including all overlay drawing. During playback, a render error produces **no output on any channel**: the frame counter keeps advancing, the image freezes on the last good frame, and there is no way to distinguish that from a video that happens to be still. This is the only `except: pass` in the codebase with no logging at all, and it sits on the most-watched code path.

## A6. Ranked inventory of silent-degradation routes

| # | Route | Location | Visible to user? |
|---|---|---|---|
| 1 | Calibration change silently mislabels exported units | §C1 | **No** |
| 2 | Per-fish processing failure → fish silently missing from export | [processing.py:190](fish_analyzer/processing.py:190) | Count on a status line that is then overwritten |
| 3 | Per-file bout failure → file silently missing from `bout_results` | [bout_tab.py:313](fish_analyzer/gui/bout_tab.py:313) | `print` overwritten; final count says "N file(s)" without saying N of what |
| 4 | Batch analysis aborts mid-way → mixed fresh/stale state | §A4 path 1 | One status line, no dialog |
| 5 | Render error during playback | [inspector_tab.py:740](fish_analyzer/gui/inspector_tab.py:740) | **No** |
| 6 | Any matplotlib event-handler error | §A3 | **No** |
| 7 | `except Exception: density = np.zeros_like(x_grid)` in KDE plotting | [analysis_tab.py:659](fish_analyzer/gui/analysis_tab.py:659), [:747](fish_analyzer/gui/analysis_tab.py:747), [bout_tab.py:667](fish_analyzer/gui/bout_tab.py:667) | A flat ridge line reads as "this fish barely moved", not as "the KDE failed" |

Route 7 deserves a note: a `gaussian_kde` failure (singular covariance — every value identical, e.g. a fish that never moved) is rendered as a **flat density curve indistinguishable from real data**. That is a plot a researcher could put in a figure.

## A7. Failure visibility — what is already right

- The bout panel's **staleness banner** ([inspector_tab.py:1160-1176, 1240-1249](fish_analyzer/gui/inspector_tab.py:1160)) compares the current UI parameter values against `_bout_run_params` from the run that produced the displayed bouts, and turns the title firebrick red with `⚠ params changed — re-run analysis`. This is the correct pattern, correctly implemented. **It is the only place in the app that does this**, and §E recommends generalising exactly it.
- `_export_bout_csv` writes the four detection parameters into every row ([bout_tab.py:948-960](fish_analyzer/gui/bout_tab.py:948)) — the only export with provenance.
- `smoothing_failed` is tracked per fish and surfaced in the status bar ([data_tab.py:613-618](fish_analyzer/gui/data_tab.py:613)) — a deliberate fix for exactly this class of bug.
- Shoaling reports per-file failures with a **modal** `showwarning` ([shoaling_tab.py:246](fish_analyzer/gui/shoaling_tab.py:246)) rather than a print. Spatial reports a `N/M files` success count ([spatial_tab.py:743](fish_analyzer/gui/spatial_tab.py:743)). Both are better than the bout and individual paths; they are the in-repo precedent to standardise on.
- Optional dependencies degrade gracefully with actionable messages (`Install with: pip install shapely`, `pip install opencv-python`), not tracebacks.

---

# (b) The implicit cross-mixin contract

Extracted by AST over all six mixins plus `GUIBase`: **302 distinct `self.*` attributes, of which only 26 are touched by more than one module.** That is a far weaker coupling than the plan assumed.

### Shared attributes (the real contract)

| Attribute | Written by | Read by | Note |
|---|---|---|---|
| `loaded_files` | base (+ mutated in place by data) | **all 6 mixins** | The genuine hub. See caveat below. |
| `notebook` | base | all 6 | Widget parent only |
| `root` | base | analysis, base, data, inspector | Widget parent + `update()` |
| `set_status` (method) | — | all 6 | The single output channel |
| `active_file` | base, data | data only | **Not** cross-mixin despite the name |
| `file_groups` | base, analysis | analysis (+ `export.py`) | |
| `processing_params` | base **only** | analysis only | Never updated from the GUI — see §C5 |
| `shoaling_params` | base, shoaling | *nothing* | Write-only; dead |
| `video_readers` | base | inspector | |
| `animation_running`, `animation_after_id` | base, inspector | inspector | |
| `arena_vertices`, `arena_definition`, `arena_fig`, `arena_ax`, `arena_canvas`, `current_arena_file`, `file_arena_definitions`, `_arena_width_bl`, `_arena_height_bl` | base, spatial | spatial only | 9 attributes declared in base, used by exactly one mixin |
| `analysis_files_listbox` | analysis | analysis, data | |
| `bout_results`, `_bout_run_params`, `_bout_selected_files` | bout | bout, **analysis, inspector** (via `getattr`) | Invisible to AST — see below |

**Caveat on the "written by" column:** `self.loaded_files[nickname] = x` is a subscript-store on an attribute *load*, so the table under-reports mutation. `loaded_files` is mutated by `data_tab` (add/remove) and its per-file *contents* are mutated by four mixins: `processed_data` (data), `shoaling_results` (shoaling), `thigmotaxis_results` (spatial), `roi_analysis_done` (spatial, injected dynamically and never read anywhere). **The real shared mutable state is not the mixin attributes — it is the result slots hanging off `LoadedTrajectoryFile`.** That is Pass H's question, and every stale-state finding in §C traces back to it.

### Cross-mixin *method* calls — the actual coupling

Eight calls cross mixin boundaries, five of them guarded by `hasattr`:

| Caller → callee | Guard |
|---|---|
| data → `_refresh_bout_file_list`, `_run_bout_analysis` | `hasattr` ([data_tab.py:574](fish_analyzer/gui/data_tab.py:574)) |
| data → `_update_analysis_files_listbox`, `_update_analysis_visualizations` | unguarded |
| data, shoaling → `_update_shoaling_file_dropdown` | unguarded |
| shoaling → `_update_inspector_file_dropdown`, `_update_analysis_files_listbox`, `_update_spatial_file_dropdown` | `hasattr` ([shoaling_tab.py:192-201](fish_analyzer/gui/shoaling_tab.py:192)) |
| bout, shoaling → `_inspector_rebuild_needed` | `hasattr` |

`_update_shoaling_file_dropdown` is misnamed: it is the fan-out that refreshes **every** tab's file widget, and it lives in the shoaling mixin. That is the one genuine layering violation.

The `hasattr` guards are defensive against a class composition that is fixed at import time in `gui/__init__.py:29` — all six mixins are always present. They can never be false. `spatial_tab` goes further and `hasattr`-guards **its own** attributes ([:319](fish_analyzer/gui/spatial_tab.py:319), [:688](fish_analyzer/gui/spatial_tab.py:688), [:820](fish_analyzer/gui/spatial_tab.py:820), [:897](fish_analyzer/gui/spatial_tab.py:897), [:1021](fish_analyzer/gui/spatial_tab.py:1021)) — all created unconditionally in `_create_spatial_controls`. This is cargo-cult defensiveness that makes real bugs (an attribute genuinely missing because construction order changed) fail silently into a default instead of loudly.

### Was commit `9ad0345` a cluster of cross-mixin state bugs?

**No — the plan's hypothesis is refuted.** Reading the commit:

| Fix | Actual category |
|---|---|
| IBI shows `—` instead of garbling when NaN | Display formatting, one file |
| Slow-fish heading falls back to look-back | Core algorithm, `bout_analysis.py` — not GUI at all |
| Inspector status "Rebuilding inspector…" | **Missing progress feedback** (§D) |
| `smoothing_failed` surfaced in status bar | **Failure visibility** (§A) |

Three of four are the subject of *this* section, not of mixin aliasing. The one GUI-structural fix (#3) added the `set_status` + `update_idletasks` at [inspector_tab.py:936-937](fish_analyzer/gui/inspector_tab.py:936) that, because of §D1, now fires on *every single frame*. The bug clusters in this repo come from invisible failure and stale derived state, not from shared `self`.

---

# (c) Stale derived state

Each of these is a sequence a researcher can actually perform. **None of the invalidation paths asked about in the prompt exist** — there is no invalidation logic anywhere in the GUI.

## C1. Change calibration after analysis → CSV with the wrong unit label — `CONFIRMED` — **highest severity in this audit**

```
1. Data tab → load session → Run All Analysis        (unit = BL)
2. Data tab → "Use custom calibration" → 24 px = 1 cm → Apply Calibration to Active File
3. Individual Analysis tab → Export Results to CSV
```

- `_apply_calibration` ([data_tab.py:519](fish_analyzer/gui/data_tab.py:519)) assigns `loaded_file.calibration = calibration` and touches nothing else. `processed_data` is untouched.
- Scaling is applied **at processing time**: `transformed = transformed * self.file.calibration.scale_factor` ([processing.py:266](fish_analyzer/processing.py:266)). The stored metrics are still in BL.
- The unit label is read **at export time**: `unit = loaded_file.calibration.unit_name` ([export.py:60](fish_analyzer/export.py:60), [:215](fish_analyzer/export.py:215)).

**Result: a CSV whose `Unit` column says `cm` over numbers in body lengths.** No warning, no dialog, no status message. The same mismatch appears in the summary tables (`Distance (cm)` header over BL values, [analysis_tab.py:268](fish_analyzer/gui/analysis_tab.py:268)) and in the Methods panel (`Scale factor: 0.041667 cm/px`, describing a scale factor that was never used).

The trajectory plot is the *only* place the user could notice: `_plot_trajectory_view` sets `xlim` from the **new** `scale_factor` ([analysis_tab.py:937-939](fish_analyzer/gui/analysis_tab.py:937)) while plotting **old** coordinates, so the trajectories collapse into a corner of the axes. That visual glitch is the sole hint that anything is wrong.

## C2. Remove a file, then export bouts → data from the removed file is exported — `CONFIRMED`

```
1. Load files A and B → Run Bout Analysis on both
2. Data tab → select A → Remove → confirm
3. Bout tab → Export Bout Data to CSV
```

`_remove_file` ([data_tab.py:434-448](fish_analyzer/gui/data_tab.py:434)) deletes from `loaded_files` **only**. It leaves behind: `bout_results['A']`, `file_arena_definitions['A']`, `file_roi_definitions['A']`, `file_groups['A']`, and `video_readers['A']` (an open `cv2.VideoCapture` plus its frame cache — never `.close()`d; the only `close()` call in the repo is [inspector_tab.py:865](fish_analyzer/gui/inspector_tab.py:865), on video *replacement*).

`_export_bout_csv` iterates `self.bout_results.items()` directly ([bout_tab.py:919](fish_analyzer/gui/bout_tab.py:919)), so A's bouts are written out. Worse, `loaded_file = self.loaded_files.get(filename)` returns `None`, and the code falls back to `fps = 30.0` / `unit = "BL"` ([bout_tab.py:921-924](fish_analyzer/gui/bout_tab.py:921)). **If the recording was not 30 fps, every `BoutStartTime_s` and `Duration_ms` for that file is silently wrong** — the fallback exists precisely to avoid crashing, and so guarantees a wrong number instead of an error.

## C3. Reload a session under an existing nickname → the old video is kept — `CONFIRMED`

```
1. Load session as "Control_01"; inspector auto-loads its video
2. Load a different session, name it "Control_01" too → "Nickname Exists → Replace it?" → Yes
3. Video Inspector → select Control_01
```

`_load_session_folder` ([data_tab.py:336](fish_analyzer/gui/data_tab.py:336)) replaces the `loaded_files` entry. `video_readers['Control_01']` still holds the **old** reader, and `_inspector_try_auto_load_video` short-circuits on `if selected in self.video_readers: return True` ([inspector_tab.py:794](fish_analyzer/gui/inspector_tab.py:794)). The inspector then draws the new session's trajectory dots on the old session's video frames. `file_arena_definitions`, `file_groups` and `bout_results` are inherited the same way.

## C4. Arena drawn on file A, applied to file B of different dimensions — `CONFIRMED`

`_apply_arena_to_selected` ([spatial_tab.py:633-636](fish_analyzer/gui/spatial_tab.py:633)) does `self.file_arena_definitions[filename] = self.arena_definition.copy()` with **no check** on `video_width`, `video_height` or `body_length`.

Arena vertices are stored in body lengths, derived from A's `body_length` and Y-flipped with A's `video_height` ([spatial_tab.py:535-539](fish_analyzer/gui/spatial_tab.py:535)). Applied to B, the polygon lands somewhere else entirely — and if B's fish are larger (smaller BL-space arena), the polygon may not even enclose the trajectories, in which case thigmotaxis reports a plausible-looking percentage computed against a wrong boundary. The dialog reports success: `Arena copied to N additional file(s)`.

The same silent cross-file copy happens for ROIs, without even a button: `_run_roi_analysis` ([spatial_tab.py:756-761](fish_analyzer/gui/spatial_tab.py:756)) applies `self.roi_definition` to every selected file that lacks one.

## C5. The Methods paragraph reports default parameters, not the ones used — `CONFIRMED`

`_run_analysis` builds parameters from the GUI into a **local** variable and passes them to the processor ([data_tab.py:589, 610](fish_analyzer/gui/data_tab.py:589)). It never assigns them to `self.processing_params`. AST confirms `processing_params` has exactly one store in the entire package — `ProcessingParameters.default_for_fish()` at [base.py:88](fish_analyzer/gui/base.py:88) — and one load, at [analysis_tab.py:1022](fish_analyzer/gui/analysis_tab.py:1022), which is the Methods text generator.

So: tick **Apply Savitzky-Golay smoothing**, set the rest threshold to 1.0, run the analysis — and the Methods tab, whose stated purpose is *"a draft paragraph … suitable for a manuscript methods section"*, reports `Smoothing: OFF` and `Rest speed threshold: 0.5`. The numbers in the table are correct; the description of how they were produced is not.

This is a one-line fix (`self.processing_params = params`) and it should be the first thing anyone changes in this file.

## C6. Partial re-runs leave mismatched result sets across tabs — `CONFIRMED`

- `_run_bout_analysis` calls `self.bout_results.clear()` ([bout_tab.py:305](fish_analyzer/gui/bout_tab.py:305)) then repopulates **only the selected files**. Run bouts on A only, then `Export Combined Summary (per fish)`: the "Bout data missing" prompt at [analysis_tab.py:1132](fish_analyzer/gui/analysis_tab.py:1132) only fires when `bout_results` is *entirely* empty. With A present, B..N are exported with blank `Bout_*` columns and **no warning at all**.
- `_export_shoaling_csv` and `_export_thigmotaxis_csv` export **every** file that has a result attached ([shoaling_tab.py:496](fish_analyzer/gui/shoaling_tab.py:496), [spatial_tab.py:1276](fish_analyzer/gui/spatial_tab.py:1276)), regardless of which files were selected for the last run. Change the sample interval, re-run on a subset, export → one CSV containing rows computed under two different parameter sets, with no column distinguishing them.
- Re-running `Run All Analysis` does not clear `shoaling_results` or `thigmotaxis_results`, which were computed from the previous trajectories.

## C7. Analysis units and displayed units can disagree between tabs — `CONFIRMED` (cross-reference to Pass B)

The inspector's NND/IID overlay labels use `loaded.calibration.scale_factor` ([inspector_tab.py:1306, 1525, 1566](fish_analyzer/gui/inspector_tab.py:1306)), while `ShoalingCalculator` computes `1.0 / metadata.body_length` ([shoaling.py:204](fish_analyzer/shoaling.py:204)) and `ThigmotaxisCalculator` does the same ([spatial.py:344](fish_analyzer/spatial.py:344)). Calibrate in cm and **the number drawn on the video frame and the number in the shoaling table are in different units**, while the table header and the plot axis both say `BL` (hardcoded at [shoaling_tab.py:280-282, 331, 373, 416](fish_analyzer/gui/shoaling_tab.py:280)). The GUI replicates the `1.0 / body_length` bypass in six further places in `spatial_tab.py` ([:385](fish_analyzer/gui/spatial_tab.py:385), [:535](fish_analyzer/gui/spatial_tab.py:535), [:783](fish_analyzer/gui/spatial_tab.py:783), [:1049](fish_analyzer/gui/spatial_tab.py:1049), [:1104](fish_analyzer/gui/spatial_tab.py:1104)) and once in `inspector_tab.py` ([:993](fish_analyzer/gui/inspector_tab.py:993)).

The fix belongs to Pass B/C; recorded here because the *user-visible symptom* is two contradictory numbers on screen at once.

---

# (d) Blocking, progress, and file decomposition

## D1. The inspector rebuilds its entire widget tree on every frame in the default configuration — `CONFIRMED`

`_inspector_update_fast` decides whether to rebuild from `self._insp_fig is None` ([inspector_tab.py:926-933](fish_analyzer/gui/inspector_tab.py:926)). But `_insp_fig` is set to `None` at the top of `_inspector_rebuild_figure` ([:974](fish_analyzer/gui/inspector_tab.py:974)) and **only reassigned inside `if show_time:`** ([:1030](fish_analyzer/gui/inspector_tab.py:1030)).

`show_time = time_mode != "none"`, and the Time Panel radio defaults to `"none"` ([:407](fish_analyzer/gui/inspector_tab.py:407)). **So with default settings `_insp_fig` is permanently `None`, and every slider tick, every playback step, and every trail-slider drag destroys and rebuilds the whole display panel** — `winfo_children()` destroy, new `tk.Canvas`, new `<Configure>` binding, re-read of the background PNG from disk via `plt.imread` ([:1096](fish_analyzer/gui/inspector_tab.py:1096)).

It also means the `set_status("Rebuilding inspector...")` + `root.update_idletasks()` added by commit `9ad0345` ([:936-937](fish_analyzer/gui/inspector_tab.py:936)) executes on every frame — a synchronous event-loop pump inside the render path.

Consequences that need a live run to quantify (`PLAUSIBLE`): visible flicker during playback, background image re-decoded per frame, and a possible **rebuild feedback loop** — `_on_inspector_resize` schedules `_resume` → `_inspector_update_fast` → rebuild → destroys the canvas that fired `<Configure>` → new canvas → new `<Configure>` → `_on_inspector_resize` again ([:1025, :704](fish_analyzer/gui/inspector_tab.py:1025)).

One-line fix: assign `self._insp_fig` unconditionally, or track rebuild-needed in a dedicated `self._insp_needs_rebuild` flag rather than overloading a figure handle as a sentinel.

## D2. `root.update()` inside the processing loop is a genuine reentrancy hazard — `CONFIRMED` (mechanism) / `PLAUSIBLE` (consequence)

[data_tab.py:601, 608, 622](fish_analyzer/gui/data_tab.py:601) call `self.root.update()` — not `update_idletasks()`. `update()` processes the **full** event queue including user input. The `Run All Analysis` button is never disabled, so a second click during processing re-enters `_run_analysis_and_switch_tab` from inside the first call's loop. Both invocations then write `loaded_file.processed_data` for the same objects and both run the `finally: pack_forget()`. The inner call also fires `_run_bout_analysis`, which calls `self.bout_results.clear()` while the outer call is still iterating.

Every other button in the app is live during this window too — including `Remove` (mutating `loaded_files` while `_run_analysis` iterates it → `RuntimeError: dictionary changed size during iteration`, which lands in §A4 path 1).

The minimal fix is not threading: disable the button (`config(state=DISABLED)`) for the duration and switch `update()` → `update_idletasks()`. That is ~6 lines and removes the whole class.

## D3. Thigmotaxis has no progress feedback of any kind — `CONFIRMED`

The only `ttk.Progressbar` in the repo is `analysis_progress` in `data_tab.py`. The thigmotaxis batch loop ([spatial_tab.py:715-732](fish_analyzer/gui/spatial_tab.py:715)) contains **no `set_status`, no `update()`, no progress bar** — nothing between "user clicks Run Analysis" and the completion dialog. Given Pass F's finding that thigmotaxis is a Python double loop constructing a shapely `Point` per fish per frame (~360k `contains()` calls for a 6-fish 30k-frame recording), the window will be marked *Not Responding* by Windows for the whole run. The user's only feedback is a frozen window.

## D4. Threading vs cooperative chunking — one recommendation

**Recommendation: do not thread the analyses. Add a cooperative-chunking progress helper and disable the triggering controls.**

Reasoning:

- Tkinter widget calls are not thread-safe; every analysis path in this app ends in a widget update (tables, figures, status). A worker thread would need a result queue plus `root.after` polling for marshalling — real complexity, in a repo with **no tests**, maintained by one person.
- The analyses are pure CPU over numpy/shapely. Except for the numpy-internal GIL releases, a worker thread buys responsiveness, not speed.
- The batch loops are already structured as `for file in selected_files:` with a natural yield point per iteration. Two of the three (`_run_analysis`, `_run_bout_analysis`) already pump the event loop; they just do it with the wrong call and without disabling input.
- Pass F wants the thigmotaxis inner loop vectorised. A vectorised thigmotaxis will run in seconds, at which point the responsiveness problem is mostly gone and a threading rewrite would have been wasted work. **Fix the algorithm before adding concurrency.**

Concretely: one `_with_progress(items, label)` generator in `gui/utils.py` that disables a given button, shows the progress bar, yields each item, calls `set_status` + `update_idletasks()` between items, and restores state in a `finally`. Applied to the four batch loops (individual, bout, shoaling, spatial), that is ~30 lines of helper replacing ~40 lines of ad-hoc pumping, and it fixes D2 and D3 together.

The one place threading is genuinely warranted is video frame reading — where it already exists, and where Pass F reports an unsynchronised `cv2.VideoCapture` shared between `_preload_worker` and the main thread ([video_utils.py:227-230](fish_analyzer/video_utils.py:227)). That is a correctness bug in the inspector, and it is the reason to be conservative about adding more threads here.

## D5. Decomposition proposals

### `inspector_tab.py` — 1,682 lines → split, one seam matters much more than the others

| Proposed module | Lines | Content | Why |
|---|---|---|---|
| **`inspector_render.py`** | ~250 | `_inspector_draw_cv2`, `_inspector_draw_numpy`, `_inspector_update_zoom`, `_rgba_to_bgr`, `_rgba_to_rgb_uint8` ([:1466-1683](fish_analyzer/gui/inspector_tab.py:1466)) | **Do this one.** These are pure functions over numpy arrays — they mutate `display` in place and return nothing. Their *only* contact with `self` is reading tk `BooleanVar`s. Hoist those into an `OverlayOptions` dataclass at the call site and the whole block becomes GUI-free and **unit-testable** (assert a dot lands at the right pixel, assert NND label text). It is the one part of the inspector with real logic, and today none of it can be tested. |
| `inspector_controls.py` | ~420 | `_make_collapsible`, `_create_inspector_controls`, `_create_inspector_display` ([:49-466](fish_analyzer/gui/inspector_tab.py:49)) | Pure widget construction, zero logic. Splitting it is cosmetic but it is 25% of the file and it is what you scroll past to reach anything interesting. |
| `inspector_playback.py` | ~220 | Frame index, jump, step, play/pause, resize debounce ([:529-750](fish_analyzer/gui/inspector_tab.py:529)) | A self-contained state machine over `(frame_var, step, after_id)`. Also where D1's rebuild-flag bug lives. |
| remains in `inspector_tab.py` | ~700 | File selection, video source management, `_inspector_update_fast`, `_rebuild_figure`, `_build_bout_panels`, `_update_dynamic` | Coherent: this is "own the figure lifecycle". |

### `spatial_tab.py` — 1,299 lines → one clean lift, one merge

| Proposed module | Lines | Content | Why |
|---|---|---|---|
| **`spatial_heatmaps.py`** | ~145 | `_generate_comparison_heatmaps`, `_plot_combined_heatmaps`, `_plot_individual_fish_heatmaps` ([:1003-1147](fish_analyzer/gui/spatial_tab.py:1003)) | Zero dependency on arena, thigmotaxis or ROI state — needs only `(loaded_file, grid_size, mode, shared_scale)`. Clean lift, no untangling. |
| `arena_editor.py` | ~290 | `_setup_arena_canvas_for_file` … `_draw_rectangle_arena` ([:372-661](fish_analyzer/gui/spatial_tab.py:372)) | A self-contained interactive polygon editor. Worth isolating **because it is the thing that most needs to be reused** — the plan identifies re-drawing arenas every session as the single largest time cost in the tool, so persistence will be built on top of this. |
| **merge, don't split** | ~150 | ROI mode ([:1163-1209](fish_analyzer/gui/spatial_tab.py:1163), [:746-815](fish_analyzer/gui/spatial_tab.py:746)) | The ROI code is a **second, parallel implementation** of polygon drawing and point-in-polygon analysis, bolted onto the arena canvas via a `_roi_drawing_mode` flag checked inside `_on_arena_click`. Splitting it out would preserve the duplication. Fold it into `arena_editor.py` as "draw a polygon, tagged arena or ROI". |

Also delete the two orphan widgets at [:78-79](fish_analyzer/gui/spatial_tab.py:78) — `spatial_compare_listbox = tk.Listbox(tk.Frame())` is parented to a throwaway frame, never packed, never read. Same for `roi_analysis_done` ([:809](fish_analyzer/gui/spatial_tab.py:809)), an attribute injected onto `LoadedTrajectoryFile` and read nowhere.

### `analysis_tab.py` — 1,163 lines → **one lift, then leave it alone**

| Proposed module | Lines | Content |
|---|---|---|
| `distribution_plots.py` | ~285 | `_plot_speed_histograms`, `_plot_speed_ridge`, `_plot_speed_collapsed_ridge`, `_plot_speed_collapsed_histograms`, `_build_collapsed_info` ([:487-816](fish_analyzer/gui/analysis_tab.py:487)) — a ridge/histogram library over `[(label, values, color, unit)]`, already written to that shape. `bout_tab.py`'s `_plot_bout_ridge` ([:569-716](fish_analyzer/gui/bout_tab.py:569)) is a fourth copy of the same layout maths and should call it. |

The remaining ~880 lines (controls, summary tables, behavioural comparison, trajectory view, methods text, export) are **long but coherent** — one tab, one data type, one flow. Leave them. Splitting on line count alone would just add import ceremony.

## D6. Duplicated widget scaffolding — quantified

| Duplicated pattern | Sites | Lines now | Proposed helper | Lines removed |
|---|---|---|---|---|
| Numeric entry parse + silent reset to default (`try: float(var.get()) except ValueError: var.set("30")`) | 23 | ~115 | `read_number(var, default, label) -> float` — **and make it report** rather than silently substituting | ~80 |
| CSV export flow (`asksaveasfilename` → try/except → `set_status` → `showinfo`/`showerror`) | 5 | ~110 | `run_csv_export(title, initialfile, export_fn)` | ~80 |
| Methods-text panel (Scrollbar + Text + placeholder insert) | 4 | ~52 | `create_methods_panel(notebook, placeholder) -> tk.Text` | ~37 |
| File-selector listbox + scrollbar + container | 5 | ~50 | `create_file_listbox(parent, multi=True) -> tk.Listbox` | ~35 |
| Clear-frame → `Figure(figsize=…)` → `tight_layout()` → `embed_figure_with_toolbar` | 17 | ~68 | `plot_into(frame, figsize)` context manager | ~40 |
| `plt.cm.tab10(np.linspace(0, 1, n))` colour assignment | 14 | ~14 | `file_colors(names) -> dict` (also fixes inconsistent colour maps between tabs) | ~10 |
| **Total** | | **~409** | | **~280 (≈4% of the GUI)** |

`gui/utils.py` is only 165 lines and already has the two right ideas (`create_sortable_treeview`, `embed_figure_with_toolbar`), which are used 8 and 17 times respectively. Extending it is low-risk.

But be honest about the value: **280 lines is not the problem in this codebase.** The real payoff is behavioural consistency — right now the five export paths have five slightly different error handlings, and the 23 numeric-parse sites have three different behaviours (silent reset, `showwarning` then reset, reset without touching the var). One helper makes all of them behave the same, which is what §A actually needs.

---

# (e) Verdict on the mixin architecture

## Keep the mixins. Do not restructure.

The plan proposed that the mixin design is the source of the bug clusters. **The evidence does not support that.** Specifically:

1. **Coupling is low.** 26 shared attributes out of 302 (8.6%). Nine of the 26 are arena state that base declares and only `spatial_tab` ever touches. Two are dead (`shoaling_params` write-only; `processing_params` written once and read once). `active_file` is used by exactly one mixin despite living in base.
2. **No mixin overrides another's method.** The AST scan found **zero** method names defined in more than one module. There is no MRO ambiguity — the classic mixin failure mode is entirely absent.
3. **The one commit cited as evidence isn't.** `9ad0345` is three failure-visibility/display fixes and one core-algorithm fix (§B).
4. **Every stale-state bug in §C is orthogonal to the mixins.** C1 is `calibration` vs `processed_data` on `LoadedTrajectoryFile`. C2/C3 are per-file side dictionaries not being cleaned up. C5 is a missing assignment. C4 is a missing dimension check. Converting the mixins to composed panel objects would leave all six bugs exactly as they are, because the shared state lives on the data object, not on `self`.

A restructure would cost weeks against 6,698 untested lines and fix none of the findings above. That is the wrong trade for a one-maintainer lab tool.

## What to tighten instead (cheap, no restructure)

- **Delete the impossible `hasattr` guards** (5 cross-mixin + 5 self-guards). They can never be false and they convert real construction-order bugs into silent defaults.
- **Move `_update_shoaling_file_dropdown` into `GUIBase` and rename it `_refresh_all_file_widgets`.** It is the fan-out for every tab; it should not live in the shoaling mixin.
- **Declare the contract where it already half-exists.** Every mixin docstring already lists "Expects the following attributes from base class" — and every one of those lists is incomplete (`DataTabMixin` omits `file_groups`, `analysis_files_listbox`, `analysis_progress`; `BoutTabMixin` omits `bout_results`, which it creates itself). Bring them in line with §B's table and they become the documented interface, for free.
- **Initialise `bout_results` in `GUIBase.__init__`**, not partway through `_create_bout_tab` ([bout_tab.py:60](fish_analyzer/gui/bout_tab.py:60)). That removes all four `getattr(self, 'bout_results', {})` defensive reads.

## Sequenced recommendations

Ordered by (value to a researcher) ÷ (risk), each independently shippable.

| # | Change | Fixes | Size |
|---|---|---|---|
| **1** | `self.processing_params = params` in `_run_analysis` | C5 — Methods paragraph currently misreports the parameters used | **1 line** |
| **2** | Invalidate on calibration change: in `_apply_calibration` / `_apply_calibration_to_all`, clear `processed_data`, `shoaling_results`, `thigmotaxis_results` for affected files and tell the user "calibration changed — re-run analysis" | **C1**, the one finding that can put a wrong unit on a published number | ~15 lines |
| **3** | Install a global error reporter: `root.report_callback_exception = self._report_error` and `fig.canvas.callbacks.exception_handler = self._report_error`, where `_report_error` shows a modal with the exception and appends the traceback to the log | A2, A3, A4, A5 — closes the two channels through which *every* unhandled exception currently disappears | ~25 lines |
| **4** | Disable the triggering button + `update_idletasks()` instead of `update()` in the four batch loops, via one `_with_progress()` helper | D2, D3 | ~30 lines net negative |
| **5** | Give the log a home: a "Log" tab (or a `Show Log` button) rendering `get_log()`, and route `sys.stderr` through the redirector too | A1 — makes the 100 lines of history reachable and gives 3 somewhere to write | ~20 lines |
| **6** | Assign `self._insp_fig` unconditionally / use a dedicated rebuild flag | D1 — inspector stops rebuilding every frame in the default configuration | 2 lines |
| **7** | Clean up all per-file side dictionaries in `_remove_file` and on nickname replacement; call `video_readers[n].close()` | C2, C3, plus the `cv2.VideoCapture` leak | ~10 lines |
| **8** | Refuse to copy an arena between files whose `video_width`/`video_height`/`body_length` differ — or warn and offer to rescale | C4 | ~10 lines |
| **9** | Generalise the bout tab's staleness banner: stamp every result object with the parameters that produced it, and mark any display whose current UI values differ | C6, and the general class | ~40 lines |
| **10** | Extract `inspector_render.py`; add the first unit tests for overlay drawing | D5 — the only untestable logic in the GUI becomes testable | ~250 lines moved |
| **11** | Extract `spatial_heatmaps.py`, `distribution_plots.py`; consolidate scaffolding into `gui/utils.py` | D5, D6 | ~700 lines moved, ~280 deleted |

**Items 1–8 total roughly 110 lines of change and close every high-severity finding in this audit.** They require no restructure, no test harness, and no working environment beyond what Pass A is already fixing. Items 9–11 are the structural work, and they are worth doing only after Pass H confirms this pipeline is the one to invest in.

---

## What's fine — an honest list

Not everything here is broken, and a report of only problems would misrepresent the codebase.

- **The mixin split is real and sensible.** Six tabs, six mixins, one file each, no method collisions, no MRO tricks. Someone thought about this.
- **`gui/utils.py` has the right two helpers.** `create_sortable_treeview` (8 uses) and `embed_figure_with_toolbar` (17 uses) are genuinely shared and genuinely reduce duplication. The problem is that there aren't more of them, not that they're wrong.
- **The bout staleness banner** (§A7) is a better solution to the stale-derived-state problem than most production applications ship.
- **Optional dependencies degrade gracefully** — `SHAPELY_AVAILABLE`, `CV2_AVAILABLE`, `_PIL_AVAILABLE` are checked at every use site with actionable install instructions, not tracebacks.
- **The Methods panels** are a genuinely good idea, well executed — plain-language, parameter-complete, copy-pasteable into a manuscript. (They just need C5 fixed to be trustworthy.)
- **The inspector's PIL/ImageTk render path** with cv2 compositing is a considered, correct optimisation, with comments explaining *why* (`replace reference BEFORE using canvas` — the PhotoImage refcount trap is a real and non-obvious Tk bug, and it's handled).
- **The resize debounce** (`_on_inspector_resize`, 200 ms) shows awareness of exactly the kind of event-storm problem this GUI is prone to.
- **Per-file calibration and per-file arenas** are the right data model — a batch of recordings genuinely does need per-file settings, and this supports it.
- **Progress feedback where it exists** (`analysis_progress`, per-file status lines) is well-formed; it just needs to exist in three more places.

---

# Fix status — applied 2026-07-31

Items 1–8 of the sequenced recommendations, verified in a fresh venv
(Python 3.12.5, traja 25.0.1) and against four real idtracker.ai sessions.

| # | Finding | Fix | Where |
|---|---|---|---|
| 1 | C5 — Methods paragraph reported constructor defaults | `self.processing_params = params` in `_run_analysis` | `data_tab.py` |
| 2 | **C1 — calibration change exported old numbers under the new unit** | `_invalidate_results_for()` clears `processed_data`, `shoaling_results`, `thigmotaxis_results` and `bout_results`, and `_invalidation_notice()` tells the user why | `data_tab.py` |
| 3 | A2/A3 — Tk and matplotlib callback exceptions went to stderr and vanished | `_setup_error_reporting()` installs `_report_uncaught` on `root.report_callback_exception` and, via `utils.set_figure_error_handler`, on every embedded canvas | `base.py`, `utils.py`, `spatial_tab.py` |
| 3b | A5 — the playback loop's `except Exception: pass` hid render errors on every channel | playback stops and the error is reported once, instead of the counter advancing over a frozen image | `inspector_tab.py` |
| 4 | D2/D3 — `root.update()` allowed re-entry; thigmotaxis had no feedback | `_with_progress()` disables the triggering button, uses `update_idletasks()`, restores state in `finally`; applied to all four batch loops | `base.py` + 4 tabs |
| 4b | A4/A6 — partial batch failures were invisible | `_report_batch_outcome()` names every file that failed or degraded | `base.py` + 4 tabs |
| 5 | A1 — `get_log()` had no caller; stderr was never captured | stdout **and** stderr share one 500-line buffer; a "Show Log" button opens it | `base.py` |
| 6 | D1 — inspector rebuilt its widget tree every frame by default | dedicated `_insp_needs_rebuild` flag replaces the `_insp_fig is None` sentinel | `inspector_tab.py` |
| 7 | C2/C3 — removed/replaced files left arena, group, bout and video state behind | `_purge_file_state()` clears every side dictionary and releases the `cv2.VideoCapture` | `data_tab.py` |
| 8 | C4 — arenas copied between files landed in the wrong place | `_rescale_arena_for()` re-expresses the polygon in the target's body-length units; different frame sizes are refused | `spatial_tab.py` |

557 insertions, 118 deletions across 7 files. `root.update()` no longer appears
anywhere in the package.

## Verification

**16 regression tests added** in `tests/test_gui_regressions.py`, with shared
fixtures moved to `tests/conftest.py`. Suite: **23 passed**, stable across
repeated runs, 3.8 s.

The only two `except Exception: pass` blocks left in the GUI are both
legitimate — writing to a console stream that may not exist under `pythonw`
(`base.py:75`) and a matplotlib blit-cache miss (`inspector_tab.py:1120`).

Two problems surfaced while writing them, both fixed:

- The source-level guard against `root.update()` returned a false positive on a
  docstring that *mentions* the call. Now matches only statement lines.
- Constructing one `EnhancedFishAnalyzer` per test creates a Tk root per test,
  and Tcl starts refusing new interpreters after a handful — surfacing as
  `tk.Tk()` failing in a *different* test on every run, reported as a skip. A
  silently-skipping test protects nothing. The app is now session-scoped with
  per-test state reset, which also cut suite time from 11 s to 3.8 s.

**Against real data** (four sessions, 6 fish × 18,000 frames): the full
pipeline runs, and each fix was exercised directly — 14/14 checks. Two results
worth keeping:

- **C4 was live, not hypothetical.** `body_length` differs across the supplied
  sessions (71.3–82.2 px), so an arena drawn on G604 and copied to H604
  rendered **15.3% too large relative to the fish**, shifting every
  border-zone number. The rescale round-trips to the same pixels exactly
  (error 0.00e+00).
- All four sessions share one frame size (1288×964), so all of them are
  rescalable rather than refused.

## Not done — and why

- **Item 9 (stamp parameters onto every result)** is H's Step 2, which sits on
  top of H's Step 1 `AnalysisConfig`. Doing it ad hoc in the GUI means writing
  it twice; H's amendment (1) explicitly routes provenance work there.
- **Items 10–11 (extract `inspector_render.py`, `spatial_heatmaps.py`,
  `distribution_plots.py`; consolidate scaffolding)** wait for Pass B, which
  decides which implementations are canonical. Moving code before that would
  move the wrong code.
- **C7 (inspector labels vs shoaling table in different units)** is unfixed by
  design — it is the `scale_factor` vs `1/body_length` split, and which one is
  correct is Pass B's call, not a GUI decision.
