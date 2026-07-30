# Audit Plan — zebrafish-free-swim-analyzer

Stock-take written 2026-07-30 from a full read of the repo at `17fe96d`.
The audit prompts live in [`audit/`](audit/) — one file per pass, each self-contained and ready to start a session with.

---

## How to run a pass

Open a fresh session with this repo as the working directory and say:

```
Follow audit/PASS_A_reproducibility.md
```

Every prompt carries the same shared rules: **report don't fix**, **verify don't inherit my reading**, **label every finding CONFIRMED or PLAUSIBLE**, and **report what's already fine** — a list of only problems says nothing about what's trustworthy. Each pass writes one findings file to the repo root.

| Pass | Prompt | Focus | Status |
|---|---|---|---|
| **A** | [PASS_A_reproducibility.md](audit/PASS_A_reproducibility.md) | Environment, packaging, doc truth | ✅ **done** → [AUDIT_A_REPRODUCIBILITY.md](AUDIT_A_REPRODUCIBILITY.md), fixes applied, real-data addendum added |
| **H** | [PASS_H_approach.md](audit/PASS_H_approach.md) | Is this the right approach at all | ✅ **done** → [AUDIT_H_APPROACH.md](AUDIT_H_APPROACH.md), verdict: build, don't rewrite |
| **D** | [PASS_D_gui_architecture.md](audit/PASS_D_gui_architecture.md) | GUI structure, failure visibility | ✅ **done** → [AUDIT_D_GUI.md](AUDIT_D_GUI.md), 8 fixes applied + 15 regression tests |
| **B** | [PASS_B_correctness.md](audit/PASS_B_correctness.md) | Are the exported numbers right | ⏭ **next — run in full** |
| **E** | [PASS_E_integration.md](audit/PASS_E_integration.md) | Where head_detection / posture belong | ⏳ re-scoped — see amendment in the prompt |
| **C** | [PASS_C_cleanliness.md](audit/PASS_C_cleanliness.md) | One source of truth + code cleanliness | 🔀 **folded** — see below |
| **F** | [PASS_F_performance.md](audit/PASS_F_performance.md) | Speed, memory, hot paths | 🔀 **folded** — see below |
| **G** | [PASS_G_ux.md](audit/PASS_G_ux.md) | What the researcher experiences | 🔀 **folded** — see below |

## Revision, 2026-07-31 — the remaining plan is compressed

After A, H and D, three of the eight passes are no longer worth running as full
passes. Their headline findings are already inventoried, and continuing to
audit would cost more than acting on what has been found.

**Run in full: B.** Nobody has yet checked whether a single exported number is
correct. A verified the environment, D verified failure *visibility*, H
verified direction — all scaffolding. B is the pass that decides whether the
tool's output is trustworthy, and real data is now available for it.

**Re-scope: E.** Per H's amendment (4), check `polavieja_lab/midline` *before*
deciding where the posture modules belong — "vendor upstream and keep only our
head/tail disambiguation" is a live answer that changes the question. Note the
supplied sessions have **no `individual_videos/`**, so the posture half is
blocked on data regardless.

**Fold C, F and G into H's staged path** rather than running them:

| Was | Fold into | Because |
|---|---|---|
| **C** — one source of truth | H Step 4 (`to_tidy()` + one exporter) | The duplication inventory below is already the finding. C's remaining judgement call — which implementation is canonical — is B's to make, not a separate pass's. |
| **F** — performance | Two concrete tickets | The hot paths are already identified: the thigmotaxis double loop (`spatial.py:403-434`) and the unsynchronised `cv2.VideoCapture` shared between `_preload_worker` and the main thread (`video_utils.py:227-230`). The second is a *correctness* bug and should be fixed on that basis, not profiled first. |
| **G** — UX | H Steps 1–2 (`AnalysisConfig`, provenance) | H's own amendment (1) already ruled that G's biggest findings — no settings persistence, arenas redrawn every session, exports without provenance — are config-architecture work, not widget work. And G is "nearly worthless without a real session" *including video*, which is still missing. |

The failure mode being avoided: auditing is cheaper and more legible than
fixing, so the document count outgrows the diff. Three more large findings
files would tip it.

> **These sequencing notes are historical — H has landed and resolved them.**
> Kept for the reasoning; see the revision above for what is actually left.

~~**Run H early.**~~ Done. H did *not* answer "freeze the centroid pipeline", so
the provisional flag on C, D, F and G is **lifted** — per H's own amendment (5).
H's actual verdicts: keep building (no packaged tool covers this assay); the
partial overlap is with `trajectorytools` and `midline`, not ZebraZoom; and
centroid is sufficient for distance/speed/freezing/bouts/shoaling/thigmotaxis
but *not* for heading, angular velocity, laterality or erratic-movement counts.

~~**A → B → C is a hard ordering.**~~ A is done and the environment is
reproducible, so **B is unblocked**. C is folded (see revision).

~~**D, E, and G are independent.**~~ D is done. G is folded.

## Data kit — supplied 2026-07-31

Four real sessions are now available (idtracker.ai 6.0.8, 6 fish × 18,000
frames at 30 fps, 10 min each):

```
C:\Users\grich\Macquarie University\Morsch Group - Documents\04_People\Grant R\
  02_PROJECTS\TRACKING\Tracking Adult Fish\02_Processed_Data\wtTAB_6mo\Freeswim\
    session_G604_Freeswim_2025-09-19-133933-0000
    session_G604_Freeswim_group2_2025-09-19-141050-0000
    session_H604_Freeswim_2025-09-19-151812-0000
    session_H604_Freeswim_group2_2025-09-19-153149-0000
```

| Needed | Present | Unblocks |
|---|---|---|
| `trajectories/trajectories.npy` | ✅ | **B**, F (numpy side) |
| `preprocessing/background.png` | ✅ | G (arena drawing) |
| `preprocessing/list_of_blobs.pickle` | ✅ 41 MB | E |
| `session.json` | ✅ | E |
| `individual_videos/individual_*.avi` | ❌ | E (posture) — still blocked |
| source video | ❌ not on this machine | F (video reader), G (inspector) |

**The source video path is recorded inside each `.npy`** under the unused
`video_paths` key — here pointing at `MQ10002204`'s OneDrive, i.e. a different
user's account. `TrajectoryFileLoader._find_video_file()` searches the
filesystem instead of reading that key; wiring it up would auto-locate videos
wherever the folder is mounted.

Three properties of this data that any remaining pass should assume:

- **`body_length` varies 71.3–82.2 px across the four sessions** (15%), so "BL"
  is a different physical unit per file and cross-file BL comparisons are not
  like-for-like.
- **Dropout is 1.3–9.7%, in long runs** — the worst single gap is 188 frames
  (6.3 s). Anything that interpolates across gaps is inventing trajectory.
- **`id_probabilities` (n_frames, n_fish, 1) is present and unread** — the
  per-frame tracking confidence that would let the tool distinguish "absent"
  from "tracked unreliably".

<details>
<summary>Original data-kit request (superseded)</summary>

To unblock all of them at once, put **one representative idtracker.ai session** somewhere accessible and point the prompts at it:

```
<some_session_folder>/
├── trajectories/trajectories.npy          ← A, B, F (thigmotaxis profiling), G
├── preprocessing/background.png           ← G (arena drawing)
├── preprocessing/list_of_blobs.pickle     ← E
├── individual_videos/individual_*.avi     ← E (posture analyzer)
├── session.json                           ← E (validation_video.py reads this)
└── ../<source_video>.avi                  ← F (video reader), G (inspector)
```

Notes:

- **E is the hungriest** and the only one with a hard blocker beyond files: it also needs `idtrackerai` importable, which is a heavy install. If that's not practical, E can still deliver the scientific-overlap and hygiene findings — the prompt tells it to flag API questions it couldn't settle rather than guess.
- **G is nearly worthless without a real session.** Walking the workflow means actually drawing an arena on a real background image and scrubbing real video in the inspector.
- **F splits in two.** The numpy-side findings (thigmotaxis loops, run-length loops, heatmaps) can be profiled on synthesized arrays at realistic scale. The video-reader findings — including the probable `VideoFrameCapture` thread race — need a real video file.
- **B is designed around synthetic trajectories** with known ground truth, so it gets most of the way with none of this. One real `.npy` is worth having to confirm the loader and the end-to-end export, and to test the calibrate-in-cm case.
- **`.gitignore` excludes `*.npy`, `*.avi`, and `session_*/`** — don't commit any of this. Keep it outside the repo (or in an ignored `sample_data/` dir) and pass the path to each session.

If you only want to hunt down one thing: a session folder plus its source video unblocks A, B, F, and G. E additionally needs the blobs pickle and the individual crops.

</details>

---

# Where the repo actually stands

## Shape

| Area | Files | Lines | Share |
|---|---|---|---|
| `fish_analyzer/gui/` | 9 | 6,698 | 57% |
| `fish_analyzer/` core (analysis) | 8 | 3,611 | 31% |
| `head_detection/` | 4 | 915 | 8% |
| `fish_posture_analyzer.py` | 1 | 391 | 3% |
| `run_analyzer.py` | 1 | 81 | <1% |
| **Total** | **23** | **11,787** | |

Plus `fish_analyzer/backup/gui.py.backup` — 3,318 lines of dead pre-refactor GUI, tracked in git since the very first commit. **(Deleted by Pass A.)**

15 commits, all on `main`, working tree clean. No tests, no CI, no `pyproject.toml`/`setup.py`, no license. **(Pass A added 7 smoke tests + `pytest.ini`; Pass D's fixes added 15 GUI regression tests. Still no CI, packaging, or licence.)**

## Three codebases that don't talk to each other

1. **`fish_analyzer/`** — the real package. Consumes idtracker.ai `trajectories.npy` (centroids only), computes locomotor + group + spatial metrics, GUI + API.
2. **`fish_posture_analyzer.py`** — standalone script. Midline/skeleton extraction from `individual_*.avi` crops. Own tkinter dialog, own output folder, no import path into the package.
3. **`head_detection/`** — standalone scripts. Reads `list_of_blobs.pickle` for masks, does head-vs-tail disambiguation and turn analysis. Three of four scripts hardcode a OneDrive path at module level.

2 and 3 are the newest work and the direction the science is heading — real head direction and body posture rather than centroid geometry. Neither is reachable from the GUI, and neither shares the package's calibration, Y-flip, or unit conventions. → **Pass E**

## What blocks verification today

- `import fish_analyzer` fails on ambient Python 3.11 — `traja` not installed. Nothing here can be run or checked without reconstructing an environment by hand.
- `requirements.txt` is not a valid pip requirements file: lines 13–15 are pasted conda shell commands, so `pip install -r requirements.txt` errors out.
- Those pasted conda lines pin `pandas=1.5.3`; the pip section above them says `pandas>=2.0`. Direct contradiction in one file.
- Python version claimed four ways: README "3.8+", a commit fixing 3.9 compat, conda line "python=3.10", ambient 3.11.
- `scikit-image` and `idtrackerai` are imported but appear nowhere in `requirements.txt`.
- README's API example calls `ShoalingCalculator.calculate(fish_list, params)` as a static method; the real signature is `ShoalingCalculator(loaded_file, params).calculate()`. The documented example probably doesn't run.

→ **Pass A**

## Correctness questions I'd want answered before trusting an export

Unverified suspicions from reading, not confirmed bugs. Full list in [PASS_B](audit/PASS_B_correctness.md); the ones that would change a published number:

- **Calibration is bypassed in two analysis modules.** `shoaling.py:204` and `spatial.py:344` compute `1.0 / metadata.body_length` directly instead of using `calibration.scale_factor`. Calibrate in cm and individual metrics switch to cm while NND/IID/hull/thigmotaxis silently stay in body lengths — with export columns hardcoded `MeanNND_BL`, `HullArea_BL2`. Replicated ~6 more times in the GUI.
- **Two independent speed pipelines.** `processing.py:309` uses `traja.get_derivatives()`; `bout_analysis.py:428` uses `np.diff`. Same `0.5` threshold applied to both.
- **Opposite NaN conventions.** `bout_analysis.py:432` treats a dropout as "still"; `processing.py:413` treats it as "moving".
- **Turn metrics run on unsmoothed positions.** `apply_smoothing` defaults `False`, and `processing.py` has no displacement noise floor where `bout_analysis.py` uses `_MIN_DISP = 0.05` BL. Angular velocity and laterality may be measuring jitter.
- **Laterality sign is asserted, never tested** (`processing.py:614`), in already-Y-flipped coordinates — and `bout_analysis.py:361` may not agree with it.
- **Y-flip implemented independently in six-plus places.** `ArenaDefinition.from_normalized` flips Y on `vertices_pixels`; `get_normalized_vertices` doesn't — the pair doesn't round-trip.
- **Burst duration counts frames where *acceleration* exceeds threshold**, then reports that as the burst duration.
- **Mixed freeze denominators** — `freeze_fraction_pct` over valid frames, `freeze_total_duration_s` over all frames.
- **Failures become NaN silently.** ~5 `except Exception` blocks in `processing.py` return NaN; `process_all_fish` swallows per-fish errors with a `print`. Nothing in the exported CSV distinguishes "genuinely NaN" from "crashed".
- **`min_valid_percentage` defaults to 0.01** — a fish tracked in 1% of frames passes the gate and gets full metrics exported.

## Performance and resource use

- **Thigmotaxis is the worst offender**: `spatial.py:403-434` is a Python double loop constructing a shapely `Point` and calling `contains()` twice per fish per frame — ~360k calls for a 30k-frame 6-fish recording.
- **`VideoFrameReader` shares one `cv2.VideoCapture` across threads with no lock.** `_preload_worker` (`video_utils.py:230`) seeks and reads the same capture the main thread uses, without updating `last_frame_read` — so the main thread's "sequential read" fast path can read the wrong frame and cache it under the wrong number. If confirmed this is a correctness bug in the inspector, not just a perf issue.
- **Cache LRU is O(n) per access** (`access_order.remove()`), and every frame is `.copy()`d twice — ~600 MB resident at 1080p with the default 100-frame cache.
- **Three hand-written run-length loops** over every frame (freeze, burst, bout intervals) that all vectorize to two numpy calls.
- **`get_smoothed_fish_timeseries` is O(n²)** (`spatial.py:276`) — per-NaN distance to all valid indices.
- **Heatmaps build million-element Python lists** via `.extend()` before `histogram2d`.

→ **Pass F**

## GUI and UX

- **No threading anywhere.** Analyses run in tkinter callbacks; `data_tab.py:601-622` calls `root.update()` inside the processing loop to keep the window alive — a reentrancy hazard, since the user can re-click Process mid-run. Thigmotaxis has no progress bar at all.
- **Feedback is one status line where the last message wins.** `GUILogRedirector` replaces `sys.stdout` process-wide; it keeps 100 lines of history in `get_log()`, which **is never called anywhere** — the history is unreachable.
- **Nothing persists between sessions.** No config save anywhere in the repo. Calibration, arena definitions, group assignments, and all parameters die with the window. Re-drawing arenas by clicking vertices every session is probably the single largest time cost in the tool.
- **76 modal `messagebox` calls** — 26 in `spatial_tab.py`, 25 in `data_tab.py`.
- **Six tabs with an implicit required order** that nothing communicates or enforces.
- **Mixin architecture with no declared contract.** `EnhancedFishAnalyzer` combines `GUIBase` + 6 tab mixins all sharing `self`; any mixin can read or write any attribute. `inspector_tab.py` alone is 1,682 lines. Commit `9ad0345` ("Fix 4 bugs: IBI display, slow-fish heading, inspector status, smoothing warning") looks like the symptom.
- **Silent experimental grouping.** `export.py:50` infers group membership by regex-stripping trailing digits off filenames, never showing the user what it inferred.
- **Exports carry no provenance** — `Unit` is recorded, but not the thresholds, smoothing settings, fps, or scale factor that produced the numbers.

Worth preserving: the plain-language methods panels, graceful optional-dependency degradation, per-file calibration, the trajectory-trail slider.

→ **Passes D and G**

## Duplication inventory

Same quantity, multiple implementations, free to drift:

| Concept | Implementations |
|---|---|
| Speed | 2+ (traja derivatives, np.diff, possibly inspector) |
| Pixel→unit scaling | ~10 sites, mixing `scale_factor` and `1/body_length` |
| Y-axis flip | 6+ |
| Heading / turn angle | 3 (processing, bout_analysis, head_detection) |
| Smoothing | 4 (Savitzky-Golay, boxcar, 2× in spatial.py) |
| Run-length detection | 3 (freeze, burst, bout intervals) |
| Skeleton endpoint detection | 2 (posture analyzer, head_detection) |
| CSV row construction | 2 overlapping builders, ~25 shared keys |

→ **Pass C**

## Docs drift

README's Project Structure omits `bout_analysis.py`, `bout_tab.py`, `inspector_tab.py`. `run_analyzer.py`'s docstring lists a `gui.py` that no longer exists, and `main()` prints "v2.0" while `__version__` is `2.1.0`. License section is `[Add license here]`.

→ **Pass A**
