# Audit H — Is this the right approach at all?

Desk research + full read of the repo at `17fe96d`. Written 2026-07-30.
All external facts checked against live sources on 2026-07-30; links and dates in [Sources](#sources).

**Labels:** `CONFIRMED` = cited source or a check I executed. `PLAUSIBLE` = reasoning from what I read.

---

## The short version

**You have not reinvented ZebraZoom.** The assay this tool serves — N adult zebrafish, identities maintained, one shared arena, group metrics — is not what ZebraZoom is built for, and no packaged tool covers it end to end. Keep building.

**You have partly reinvented `trajectorytools` and `midline`** — two packages written by the idtracker.ai authors, for idtracker.ai output, which the idtracker.ai docs point you at from the page describing the very files you load. `trajectorytools` already does NaN interpolation, smoothing, body-length and seconds unit conversion, velocity/acceleration, bout detection and inter-individual distances. `midline` already extracts nose/tail/midline from idtracker.ai sessions. `fish_posture_analyzer.py`'s own docstring says its midline code is "adapted from the idtrackerai sample code" — so half of this is known; the trajectory half appears not to be. This is the single most consequential finding in this pass.

**The architecture verdict is not "rewrite".** It is: extract configuration into a file, and give the analysis a headless entry point. The analysis core is already GUI-free — `tkinter` appears in `fish_analyzer/gui/` and nowhere else (CONFIRMED, `grep` over the package). That one fact makes every recommendation here cheap, and it is the best decision in the codebase.

**Centroid vs posture is not an either/or, and the answer is not the one AUDIT_PLAN.md braced for.** Centroid is the right and sufficient foundation for distance, speed, freezing, bursts, bout timing, shoaling, and thigmotaxis — the bulk of the tool. It is *not* an adequate foundation for heading, angular velocity, laterality, or erratic-movement counts, which are the metrics the audit's own correctness suspicions cluster around. Freeze half the pipeline, not the whole thing.

**Consequence for the plan: Passes C, D, F and G are un-blocked.** AUDIT_PLAN.md marks them provisional pending this pass. My answer does not invalidate them. Run them — with the two scoping amendments in [section (f)](#f--where-i-contradict-the-other-prompts).

---

## (a) The landscape

### What each tool actually is

| Tool | Covers of this repo's scope | Does not cover | Licence | Latest activity | Install cost, lab Windows + conda | Verdict |
|---|---|---|---|---|---|---|
| **[trajectorytools](https://gitlab.com/polavieja_lab/trajectorytools)** | **~50–60% of the analysis core.** Native idtracker.ai loading, NaN interpolation, smoothing, length unit → BL, time unit → s, positions/velocity/acceleration (`traj.s/.v/.a`), curvature, bout detection, inter-individual distance, polarisation, group border | Arena polygon + thigmotaxis, freeze/burst episode metrics as defined here, laterality, CSV export shaped for R/Prism, any GUI | GPL-3.0 | 0.4.2, **2025-04-28**; maintained by Jordi Torrents (2024–), also the current idtracker.ai maintainer | `pip install trajectorytools`; deps scipy/sklearn/matplotlib/h5py/miniballcpp — all conda-friendly. Requires Python ≥3.10 | **Adopt — as an oracle first, a dependency second** |
| **[midline (“Fish Midline”)](https://gitlab.com/polavieja_lab/midline)** | **~80–90% of `fish_posture_analyzer.py` and the skeleton half of `head_detection/`.** Nose, tail and midline for fish tracked with idtracker.ai; linked from idtracker.ai's own Data Analysis page | Turn analysis, the DT+velocity head/tail disambiguation, integration with your metrics | GPL-3.0 | Repo created 2018-09-19, 32 commits; low volume | Script, not a package — read and vendor | **Adopt as reference before writing more posture code** |
| **[ZebraZoom](https://github.com/oliviermirat/ZebraZoom)** | Bout kinematics (40+ parameters/bout), tail-beat frequency, population comparison GUI, unsupervised bout clustering, batch over many videos | **The assay itself.** Tracks from raw video in a well/grid layout, docs assume "only one animal is contained in each well"; no identity-preserving multi-animal tracking in a shared arena, no NND/IID/hull, no idtracker.ai input | **AGPL-3.0** | Last push 2025-11-01, repo updated 2026-06-23, 0 open issues, PyPI 1.34.96 | `pip install zebrazoom`, Python 3.9–3.12 | **Do not adopt.** Wrong assay shape. Worth reading for its bout-parameter list |
| **[TRex](https://github.com/mooch443/trex)** | Would replace idtracker.ai *and* deliver posture: tracking + markerless ID + 2D posture and visual fields in one pass | Everything downstream — no shoaling summary, no thigmotaxis, no export layer | NOASSERTION (unclear — check before relying) | Last push **2026-07-28**, 124★, 33 open issues | C++; heaviest install of anything here | **Evaluate later, do not switch now** |
| **[idtracker.ai](https://idtracker.ai/) itself** | Tracking, validation GUI, per-animal video generator, blob/contour data | **Nothing downstream.** Docs: idtracker.ai's job "ends when the trajectory files are validated" | GPL-3.0 | 6.0.14, Jan 2025; eLife paper 2025 | Already installed | **You are recomputing nothing it provides.** Correct boundary |
| **[SLEAP](https://sleap.ai/)** | Head/tail/spine keypoints on the `individual_*.avi` crops idtracker.ai already emits — the "AI posture" path idtracker.ai's own FAQ recommends | All analysis | BSD-3-Clause-Clear | 1.6.4, **2026-07-28** — most actively released tool in this table | Annotation effort + GPU for training; heavy but well-documented | **The fallback if classical midline extraction stalls** |
| **[DeepLabCut](https://deeplabcut.github.io/)** | Same as SLEAP | All analysis | LGPL-3.0+ | 3.0.x, PyTorch backend | Heavier than SLEAP on Windows | Same role; pick one, not both |
| **[SimBA](https://github.com/sgoldenlab/simba)** | Supervised behaviour classification downstream of pose | Everything you compute | BSD-3 modified, **academic/non-commercial only** | 5.4.6, 2025-01-19 | Notoriously fiddly on Windows | Not applicable — you want named hypothesis-driven metrics, not learned classifiers |
| **[B-SOiD](https://github.com/YttriLab/B-SOID)** | Unsupervised behaviour clusters from pose | Everything you compute | GPL-3.0 | **Last push 2024-02-09** — ~2.5 years stale, 52 open issues | — | Not applicable, and I would not build on it |
| **[VAME](https://github.com/EthoML/VAME)** | Unsupervised motif discovery from pose | Everything you compute | GPL-3.0 | Last push **2026-07-21** (EthoML fork; original repo is the stale one) | — | Not applicable now. Interesting only if the science moves to unsupervised ethograms |
| **[keypoint-MoSeq](https://github.com/dattalab/keypoint-moseq)** | Unsupervised syllable segmentation from pose | Everything you compute | 0.6.8, **2026-02-26** | — | — | Same |
| **[traja](https://github.com/traja-team/traja)** (current dep) | Used at exactly 4 sites, all in `processing.py`: `smooth_sg`, `length`, `distance`, `.traja.get_derivatives()` (CONFIRMED, grep) | — | MIT | 25.0.1, **2025-10-23** after a 3-year gap (22.0.0 → 25.0.0) | Light: matplotlib/pandas/numpy/shapely/scipy/tzlocal. `torch` is an *optional* extra | See the correction below |
| **[movingpandas](https://github.com/movingpandas/movingpandas)** | Trajectory data structures on GeoPandas | Animal-behaviour semantics entirely | BSD | 0.22.4, 2025-07-16 | Drags in the GeoPandas/GDAL stack | **Do not adopt.** Built for GPS/CRS movement data; wrong domain, heavy dependency for no gain |

### A correction to AUDIT_PLAN.md on traja

AUDIT_PLAN.md says traja is "the reason the package won't import." That is true only in the trivial sense that it isn't installed in your ambient Python. Two checks change the picture:

- `CONFIRMED` — **there has never been a traja 0.6.** Full release list from the PyPI API: `… 0.2.7, 0.2.8, 22.0.0, 25.0.0, 25.0.1`. `requirements.txt` pins `traja>=0.6`, which therefore resolves to **25.0.1** — a release from October 2025 after a three-year gap, almost certainly not the version this code was written against.
- `CONFIRMED` — all four APIs you use still exist on traja `master` (`smooth_sg` at `trajectory.py:68`, `distance:154`, `length:181`, accessor `get_derivatives` at `accessor.py:426`). So it will probably import and run. Whether it *computes the same thing* as the 2021-era version is unverified and belongs to Pass B.

So the real traja finding isn't installability, it's that a meaningless pin silently selects a major-version jump. `PLAUSIBLE`: if the numbers in `AUDIT_PLAN.md`'s "two independent speed pipelines" disagree, a version skew inside traja is a candidate cause worth ruling out before blaming the code.

### Verdict: adopt-in-part, and keep building

**Keep building, and here is which needs justify it.** Not one packaged tool ingests idtracker.ai trajectories for N identity-tracked adult fish in a shared arena and returns per-fish locomotor metrics *plus* group cohesion *plus* user-defined-polygon thigmotaxis *plus* a CSV shaped for the lab's stats workflow. ZebraZoom is the closest thing and it is well-based single-animal tail tracking from raw video — adopting it would mean abandoning idtracker.ai, losing identity across a shoal, and losing every group metric. That is not a trade, it is a different experiment.

The specific needs that justify a bespoke tool, in order of how load-bearing they are:

1. **Identity-preserved group assays.** NND, IID, convex hull only exist because idtracker.ai keeps identities. This is the reason the tool exists.
2. **Arbitrary arena polygons drawn on the real background image.** Every packaged alternative assumes wells, circles, or rectangles. `PLAUSIBLE`: this is also why the GUI can't be deleted.
3. **Group-aware export shaped for how this lab actually does stats** (commit `dcf88e1`).
4. **Plain-language methods panels.** No package ships this, and for a lab tool used by people who are not software users it is worth more than most of the code.

**Adopt-in-part, concretely:**

- **`trajectorytools` as a validation oracle, immediately.** You have no tests and Pass B has no ground truth. An independent implementation from the tracker's own authors, reading the same file, is the cheapest correctness evidence available to a single maintainer. Load one session both ways and compare speed, acceleration, and inter-individual distance. Agreement is real assurance; disagreement localises a bug fast. This costs a notebook, not a migration.
- **`midline` before another line of `head_detection/`.** You are building head/tail disambiguation from scratch. The idtracker.ai authors published theirs. Read it, then decide whether yours adds anything (the DT + velocity-agreement scoring in `head_detection_test.py:101` plausibly does — the temporal median filter is a nice touch) or duplicates it.
- **Licence note, not a blocker:** `trajectorytools` and `midline` are GPL-3.0. Importing them makes a distributed combined work GPL-3.0. For an internal lab tool, irrelevant. If this ever ships with a methods paper, you inherit GPL — decide with open eyes. ZebraZoom's AGPL-3.0 is stricter still, which is a second reason not to vendor from it. Running idtracker.ai as a separate program creates no such obligation.

---

## (b) The recommended direction

> **Keep the tkinter application. Extract configuration into a file, and give the analysis a headless entry point that reads it. Do not rewrite the GUI, do not move to the web, do not go notebook-only.**

### Why the GUI is 6,698 lines — and it isn't the dual purpose

The prompt's hypothesis is that serving batch and interactive needs through one interface is what bloated the GUI. I think that is half right, and the diagnosis matters because it changes the fix.

`CONFIRMED` — the package is properly layered. `tkinter` is imported in seven files, all under `fish_analyzer/gui/`. `data_structures.py`, `processing.py`, `shoaling.py`, `spatial.py`, `bout_analysis.py`, `export.py` and `file_loading.py` import no GUI code. A batch pipeline could be written against the existing API today.

So the GUI is not large because analysis leaked into it wholesale. It is large because of three specific things:

1. **`PLAUSIBLE` — the widget *is* the parameter store.** There is no configuration object between the UI and the analysis, so every parameter needs bespoke widget code, bespoke validation, bespoke read-back, and bespoke error handling at its call site. `CONFIRMED`: no JSON, YAML, TOML, configparser or pickle write exists anywhere in `fish_analyzer/` (grep). The only `json` import in the entire repo is `head_detection/validation_video.py:8`, reading idtracker.ai's `session.json`.
2. **`CONFIRMED` — plotting lives in the tabs.** `np.` appears 56× in `analysis_tab.py`, 38× in `bout_tab.py`, 36× in `inspector_tab.py`. Some is legitimate plot preparation; some is analysis that escaped the core — `AUDIT_PLAN.md` flags the inspector computing its own speed, and the `1.0 / body_length` calibration bypass replicated in the GUI.
3. **Genuinely interactive work** — arena drawing, frame scrubbing with overlays — which is irreducible and *should* be a GUI.

Only (2) and (3) are really "GUI". (1) is a missing architectural layer showing up as UI code.

### Why not the alternatives

- **Library + notebooks.** Fails the users. Arena polygons need a canvas; a researcher who is not a software user by trade will not maintain a notebook per experiment. It also makes reproducibility worse, not better — notebooks drift silently.
- **Streamlit/Dash.** Adds a server, a browser, a port, and a dependency stack to an offline lab Windows machine, and delivers nothing tkinter can't for this workload. It would also require rewriting all 6,698 lines. `PLAUSIBLE`: the appeal here is aesthetic, and the prompt asked me not to reason that way.
- **Full rewrite in Qt/anything.** One maintainer, no tests, a tool people already use. This is the classic failed rewrite. Off the table.

### What "extract configuration" means concretely

One dataclass — `AnalysisConfig` — holding what today lives in widget state: calibration mode and value, fps, `ProcessingParameters`, `BoutParameters`, `ShoalingParameters`, arena vertices in normalised coordinates, border percentage, group assignments, `min_valid_percentage`. It serialises to JSON. Three consequences, each independently valuable:

- **Arena definitions and calibration survive the session.** `AUDIT_PLAN.md` calls re-drawing arenas "probably the single largest time cost in the tool". This deletes that cost. It is a config feature, not a UX feature.
- **Every export can stamp provenance** — thresholds, smoothing, fps, scale factor, tool version, config hash — because there is finally one object that holds all of it.
- **A headless entry point becomes ~50 lines**: read config, glob sessions, run, write CSVs. Batch and interactive stop competing, because they share an artifact instead of a code path.

Note the ordering: the GUI keeps owning the interactive work it is good at, and stops being the only place parameters can exist.

---

## (c) Centroid vs posture

### The answer

**Split the pipeline along a line that already exists in the code but isn't acknowledged.**

**Centroid is correct and sufficient — keep, harden, do not freeze:**
`total_distance`, `net_displacement`, speed statistics, freeze episodes, burst episodes, bout detection/rate/IBI/peak speed/displacement, path straightness, NND, IID, convex hull, thigmotaxis, heatmaps.

These are *scalar* or *positional* quantities. Posture would not make them more accurate — a midline gives you a better-defined body point, but centroid noise is not the limiting factor on "how far did this fish swim". Roughly two-thirds of the exported columns in `export.py` fall here. `PLAUSIBLE`, but the reasoning is straightforward: none of these quantities is a function of body orientation.

**Centroid is inadequate — freeze, document, re-found on posture:**
`mean_angular_velocity_deg_s`, `laterality_index`, `n_right_turns` / `n_left_turns`, `cumulative_heading_change_deg`, `mean_signed_angular_velocity_deg_s`, `erratic_movement_count`, and the bout-level `Bout_MeanAbsTurnAngle_deg` / `Bout_LateralityIndex`.

These all reduce to *heading*, and centroid heading is velocity direction, which:

- is **undefined at low speed** — and a freezing-heavy anxiety assay is full of low speed;
- is **noise-dominated near the displacement floor** — `AUDIT_PLAN.md` notes `processing.py` has no minimum-displacement guard where `bout_analysis.py` uses `_MIN_DISP = 0.05` BL, and that `apply_smoothing` defaults to `False`;
- carries a **sign convention asserted rather than tested** (`processing.py:614`) in an already-Y-flipped frame.

`PLAUSIBLE` but I hold it with high confidence: it is not a coincidence that almost every correctness suspicion in `AUDIT_PLAN.md`'s "would change a published number" list attaches to a direction-derived metric. Those are not eight separate bugs. They are one measurement standing on a foundation that cannot support it, and the bugs are what that looks like from the inside.

Laterality is the sharpest case. It is a headline claim of this tool, it is the kind of result that goes in a paper, and it is currently computed from the direction of centroid displacement between frames in a coordinate frame whose flip is implemented six-plus separate ways. A real head vector from a midline makes turn direction a direct measurement instead of an inference — that is a difference in kind, not in precision.

**So: posture is the future for direction. Centroid is the present and the future for everything else.** The two newest modules are pointed at exactly the right target. They are not a replacement for `processing.py`; they are the replacement for one function inside it — `_calc_movement_direction_metrics` (`processing.py:526–658`).

### What this means for the other passes

`AUDIT_PLAN.md` says: if H answers "freeze the centroid pipeline and build on posture", then C, D, F and G are polishing doomed code. **It does not, so they are not.** Concretely:

- **Pass C — run it, with one carve-out.** It is told to unify duplicate implementations into one source of truth. That is right for speed, scaling, Y-flip, smoothing, run-length detection and CSV construction. It is **wrong for heading**: do not spend a day merging three centroid-heading implementations into one canonical centroid heading. Define the single seam (`heading(fish, frame) -> angle`) and note that its implementation will be swapped for a posture-backed one with centroid fallback. Unify the interface, not the algorithm.
- **Pass D — run it, unchanged.** Mixins, stale state and failure visibility are orthogonal to which pipeline computes the numbers.
- **Pass F — run it, with a priority note.** Thigmotaxis (`spatial.py:403-434`) and the `VideoFrameReader` thread race are on the keep side of the line and are worth every minute. Deprioritise micro-optimising the direction-metric code paths.
- **Pass G — run it.** Its findings stand, but see (f): several of them are config problems wearing UX clothing.
- **Pass E — run it, but ask a prior question first.** E asks *where* `head_detection/` and `fish_posture_analyzer.py` should live. Before choosing a home, check them against `polavieja_lab/midline` and against what TRex would give for free. The right answer might be "vendor midline, keep only the head/tail disambiguation on top" — which changes what E is placing.

---

## (d) What I would change nothing about

Six things here are genuinely well chosen, and I would push back on any refactor that damaged them.

1. **The scope boundary: consume tracking, don't do tracking.** `CONFIRMED` from idtracker.ai's docs — its "job ends when the trajectory files are validated", and it explicitly hands users to downstream packages. You are filling a gap its authors acknowledge, not duplicating them. Never build a tracker.
2. **The GUI/analysis separation.** `CONFIRMED`: zero tkinter imports outside `gui/`. This is better discipline than most research code and it is why nothing in this report requires a rewrite. Every proposal here is affordable *because of* this decision.
3. **Optional-dependency degradation.** `SHAPELY_AVAILABLE`, `CV2_AVAILABLE`, `GUI_AVAILABLE` with a warning rather than a crash (`__init__.py:67-74`). For lab machines with inconsistent environments this is exactly right. Keep it.
4. **`CalibrationSettings` as an explicit object** with a named unit, validation in `__post_init__`, a `scale_factor` property and three named constructors (`from_body_lengths`, `from_physical_measurement`, `no_calibration`). The abstraction is correct. `CONFIRMED` that two modules bypass it (`shoaling.py:203-205` and `spatial.py:343-344` both compute `1.0 / loaded_file.metadata.body_length` directly) — but that is a bug against a good design, not evidence of a bad one. Fix the callers; do not touch the class.
5. **Stopping at the CSV.** Export tidy numbers, do statistics in R/Prism. Building group comparisons into the tool would mean shipping untested inferential statistics from a codebase with no tests, and researchers cannot audit what they cannot see. Hold this line even as the group-aware exports grow. Provenance columns are the right way to serve that need.
6. **The plain-language methods panels, per-file calibration, and the trajectory-trail slider.** These are the parts a researcher would miss. They are also the parts a refactor is most likely to break silently.

Also worth keeping, more mildly: dataclass results with validating `__post_init__`, and `LoadedTrajectoryFile._find_background_image` quietly locating idtracker.ai's conventional layout.

---

## (e) The staged path

Each step is independently valuable and shippable. Nothing here requires finishing the next step to pay off.

**Step 0 — make it runnable.** (Pass A.) One valid `requirements.txt`, one Python version, a resolvable traja pin. `CONFIRMED`: `requirements.txt` lines 15–17 are pasted conda shell commands, so `pip install -r requirements.txt` fails outright — that is the first thing a new lab member hits. Nothing else in this list can start until this is done. Half a day.

**Step 1 — `AnalysisConfig`, saved and loaded.** The dataclass from (b), JSON on disk, a Save/Load Settings button. Pays for itself the first time an arena is not redrawn. Does not require touching any analysis code. Two to three days.

**Step 2 — stamp provenance into exports.** Once Step 1 exists this is nearly free: append the config hash, tool version, fps, scale factor and thresholds to every CSV, or write a sidecar `<export>_params.json`. This is the step that makes a published number defensible six months later.

**Step 3 — cross-validate against `trajectorytools`.** One notebook, one real session, compare speed / acceleration / inter-individual distance. Turns "no tests" into "independently corroborated", and hands Pass B its ground truth. `PLAUSIBLE`: this is the highest correctness-confidence-per-hour action available to you right now, and it needs no test infrastructure.

**Step 4 — `to_tidy()` and one exporter.** Each results dataclass gains a method returning long-form rows (`file, fish, metric, value, unit`). The two overlapping CSV builders in `export.py` (~25 duplicated keys, `CONFIRMED` by reading both) collapse into one writer over a metric registry. Adding a metric then touches one file instead of the four or five it touches today.

**Step 5 — headless entry point.** `python -m fish_analyzer.run --config params.json --sessions ...`. ~50 lines on top of Steps 1 and 4. This is where 100-file batches and cross-batch comparisons become possible.

**Step 6 — posture, deliberately.** Read `polavieja_lab/midline`; evaluate whether TRex's built-in posture would obviate the whole line of work; then integrate a single `heading()` seam behind which the direction metrics are recomputed. Only after Step 0 gives you an environment to verify in.

**Deliberately not on this list:** rewriting the GUI framework, adding statistics, adopting SimBA/B-SOiD/VAME, migrating to movingpandas, switching trackers.

On sequencing and honesty about inertia: this codebase is five months old (first commit 2026-03-02, latest 2026-04-03 — `CONFIRMED` from git log), 15 commits, one maintainer, no external users I can see. The usual argument against change — "a working tool a lab actually uses has real value" — applies with much less force at five months than at five years. The window for cheap structural change is open now and will not stay open. That said, none of the six steps above is a rewrite, precisely because item (d)(2) already did the hard part.

---

## (f) Where I contradict the other prompts

1. **Pass G's tkinter constraint — I agree with it, but reject its framing.** Staying in tkinter is right. But several findings G will surface as UX problems are configuration-architecture problems: no settings persistence, arenas redrawn every session, exports without provenance. Fixing those in the GUI would be treating symptoms. **Amendment:** G should tag each finding "GUI" or "config", and route the config ones to Step 1 rather than to widget work.
2. **Pass C's "one source of truth" mandate — carved out for heading.** As in (c): unify the *interface*, not the algorithm, for anything direction-derived. Merging three centroid-heading implementations into one canonical centroid heading is consolidating a measure I am recommending you replace.
3. **Pass A's traja question — I have pre-empted its verdict, partially.** A is told to independently recommend keep-or-drop. My checks (no 0.6 release; all four APIs present on master; light dependency footprint; MIT) narrow that. **My recommendation: keep traja for now with a real pin (`traja==25.0.1`), and let Pass B's numeric comparison decide whether to drop it** — installability is not the argument for dropping it, and A's prompt currently implies it is.
4. **Pass E's question order.** E asks where the posture modules belong. **Amendment:** first check them against `polavieja_lab/midline`, because "vendor the upstream implementation and keep only our head/tail disambiguation" is a live answer that changes what E is finding a home for.
5. **AUDIT_PLAN.md's provisional flag on C, D, F, G — lift it.** The plan hedged on this pass answering "freeze the centroid pipeline". It does not. Those four passes are worth their full effort, subject to amendments 1, 2 and the Pass F priority note in (c).
6. **AUDIT_PLAN.md's characterisation of traja as "the reason the package won't import".** Accurate but misleading; see the correction in (a).

---

## Question 6 — does it scale

Answered directly, since it feeds the staging above.

**100 files.** `PLAUSIBLE` — the GUI holds `loaded_files: nickname -> LoadedTrajectoryFile`, each carrying a full `(n_frames, n_fish, 2)` array plus post-hoc result objects, all resident simultaneously. At 30k frames × 6 fish that is small per file, but the processing loop calls `root.update()` inside itself (`data_tab.py:601-622`) and there is no threading anywhere, so a 100-file run is one long unresponsive window with a reentrancy hazard if the user re-clicks. Step 5 (headless) is the answer, not a better progress bar.

**Multiple experimenters.** Today, nothing is shared: parameters live in each person's widget state and die with the window, so two people cannot demonstrably run the same analysis. A checked-in config file makes "we used these parameters" a fact rather than a memory. This is Step 1.

**Comparing batches recorded months apart.** Currently unanswerable — the exports do not record what produced them. Step 2.

**Adding one metric.** Today: `processing.py` (compute) + both builders in `export.py` + the analysis tab's table + the comparison table ≈ 4–5 files, with the two export builders sharing ~25 keys that must be kept in sync by hand. Should be one: register a metric, and computation, export and display follow. Step 4.

**Handing it to a new lab member.** The first thing they hit is `pip install -r requirements.txt` failing on line 15. `CONFIRMED` — that file is not a valid requirements file. The second thing is six tabs with an implicit required order that nothing communicates. Step 0 fixes the first; the second is Pass G's, and is real.

---

## Data model (Question 3), answered

**Mutable result slots — yes, they are a cause, not just a smell.** `LoadedTrajectoryFile` (`data_structures.py:145-175`) holds raw arrays *and* `processed_data`, `shoaling_results`, `thigmotaxis_results`, all `None` at construction and assigned post-hoc by whichever code ran an analysis. `PLAUSIBLE` with high confidence: this is precisely the shape that produces stale-state bugs — change the calibration and `processed_data` is silently the old calibration's; re-run one analysis and the others are quietly inconsistent. Pass D will find those bugs; they are consequences of this design, and Pass D should be told so.

The fix is not a rewrite: keep `LoadedTrajectoryFile` as an immutable *dataset* (raw arrays, metadata, calibration) and let analyses be pure functions returning results into a separate results container keyed by (file, analysis, config-hash). Then "is this result stale?" becomes a comparison instead of an assumption.

**Tidy long-form — yes, as the export and pooling representation; no, as a replacement for the dataclasses.** Keep `ShoalingResults` and friends: named fields are good at computation time, they document themselves, and their validation is real. Add `to_tidy()`. What it buys: pooling across files becomes a concat; adding a metric stops touching the exporter; provenance columns attach naturally; and the two hand-transcribed CSV builders collapse. What it costs: one method per results type and a one-time change to downstream R scripts, which will mostly get shorter.

**The statistics boundary — leave it where it is.** See (d)(5).

---

## Sources

All accessed 2026-07-30.

- idtracker.ai — [Data analysis](https://idtracker.ai/latest/user_guide/data_analysis.html) · [Output structure](https://idtracker.ai/latest/user_guide/output_structure.html) · [FAQs](https://idtracker.ai/latest/user_guide/FAQs.html) · [PyPI 6.0.14, Jan 2025](https://pypi.org/project/idtrackerai/) · [eLife 2025 reviewed preprint](https://elifesciences.org/reviewed-preprints/107602)
- trajectorytools — [GitLab](https://gitlab.com/polavieja_lab/trajectorytools) · [PyPI 0.4.2, 2025-04-28](https://pypi.org/project/trajectorytools/) (README/API read from PyPI metadata) · [idtracker.ai docs page](https://idtracker.ai/latest/user_guide/trajectorytools.html)
- midline — [gitlab.com/polavieja_lab/midline](https://gitlab.com/polavieja_lab/midline)
- ZebraZoom — [GitHub](https://github.com/oliviermirat/ZebraZoom) (pushed 2025-11-01, updated 2026-06-23, AGPL-3.0) · [PyPI 1.34.96](https://pypi.org/project/zebrazoom/) · [Freely swimming tracking docs](https://zebrazoom.org/documentation/docs/softwareTutorial/freelySwim/) · [Behavior analysis GUI docs](https://zebrazoom.org/documentation/docs/behaviorAnalysis/behaviorAnalysisGUI/) · [Frontiers 2013 paper](https://www.frontiersin.org/articles/10.3389/fncir.2013.00107)
- Stytra — [GitHub](https://github.com/portugueslab/stytra) (GPL-3.0, last push 2024-11-08, 37 open issues — stale; excluded from the table above as it is an acquisition/stimulation framework, not an analysis package)
- TRex — [GitHub](https://github.com/mooch443/trex) (pushed 2026-07-28, 124★, 33 open issues)
- SLEAP — [PyPI 1.6.4, 2026-07-28](https://pypi.org/project/sleap/) · DeepLabCut — [PyPI 3.0.x](https://pypi.org/project/deeplabcut/)
- SimBA — [PyPI 5.4.6, 2025-01-19](https://pypi.org/project/simba-uw-tf-dev/) · B-SOiD — [GitHub, last push 2024-02-09](https://github.com/YttriLab/B-SOID) · VAME — [GitHub EthoML fork, pushed 2026-07-21](https://github.com/EthoML/VAME) · keypoint-MoSeq — [PyPI 0.6.8, 2026-02-26](https://pypi.org/project/keypoint-moseq/)
- traja — [GitHub, MIT, pushed 2025-10-23](https://github.com/traja-team/traja) · [PyPI release list](https://pypi.org/pypi/traja/json) (queried directly; no 0.6 exists) · API presence checked against `master`: `traja/trajectory.py`, `traja/accessor.py`
- movingpandas — [PyPI 0.22.4, 2025-07-16](https://pypi.org/project/movingpandas/)

Repo checks executed for this pass: tkinter import survey across `fish_analyzer/`; config-persistence grep (`json|yaml|toml|configparser|pickle`); traja call-site grep; `np.` density per GUI file; `shoaling.py:198-205`; `spatial.py:338-344`; `export.py` builder comparison; `requirements.txt`; `git log`.
