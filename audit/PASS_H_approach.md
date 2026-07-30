# Pass H — Is this the right approach at all?

**Shared rules for every audit pass:**
- **Report, don't fix.** This pass ends in a written recommendation, not a diff.
- **Verify, don't inherit.** Check claims about other tools against their actual docs and repos, with links and dates. Don't recall from memory.
- **Label every finding** `CONFIRMED` (cited source or executed check) / `PLAUSIBLE` (reasoning).
- Read `AUDIT_PLAN.md` in the repo root first for the stock-take context.

---

## Goal

Step back and question the premise. Every other pass improves this thing as it is. This pass asks whether "as it is" is the right shape, and whether something better already exists.

**This pass is explicitly permitted to contradict the constraints written into the other prompts.** Pass G, for instance, is told to stay in tkinter. If you think that's the wrong call, say so here and say what changing it would cost. The other prompts optimise within the current design; this one interrogates the design.

**Run this early.** If the answer is "freeze the centroid pipeline and build on posture instead", then Passes C, D, F, and G would otherwise spend real effort polishing code that shouldn't be invested in. Treat those passes as provisional until this one lands.

## Scope

The whole project, at the level of decisions rather than lines. No data required — this pass is desk research plus reading.

---

## Question 1 — Is this already solved?

Survey the existing landscape for zebrafish / animal behavior analysis and report honestly what overlaps this repo's scope. At minimum look at:

- **ZebraZoom** — closest known overlap; bout detection, tail tracking, kinematic parameters, GUI
- **Stytra** — zebrafish behavior acquisition/analysis
- **TRex** — high-throughput tracking with posture
- **idtracker.ai's own** analysis and validation tooling — what does it already give me that I'm recomputing?
- **DeepLabCut / SLEAP** for pose, plus downstream behavior classification: **SimBA**, **B-SOiD**, **VAME**, **keypoint-MoSeq**
- **traja** and **movingpandas** as trajectory-analysis libraries (traja is already a dependency, and the reason the package won't import)

For each: what fraction of this repo's scope it covers, what it doesn't, licence, maintenance status (last release, open issue trends), install cost on a lab Windows machine, and whether its output could feed or replace parts of this pipeline.

Then give a verdict: **adopt, adopt-in-part, or justified to keep building.** If the honest answer is "this substantially reinvents ZebraZoom", say that plainly — I would rather hear it now than after another year of commits. If the honest answer is "no, the Morsch lab's needs are specific enough that a bespoke tool is right", say that just as plainly and say which needs those are.

## Question 2 — Is a desktop GUI the right shape?

The app currently serves two quite different needs through one interface:

- **Batch reproducible processing** — load N sessions, apply consistent parameters, export CSVs for stats. Wants determinism, version-controlled parameters, provenance, no clicking.
- **Interactive visual work** — drawing arena polygons, inspecting video frames with overlays, eyeballing whether tracking is sane. Genuinely needs a GUI.

Assess whether trying to do both in one tkinter app is *why* the GUI is 6,698 lines (57% of the codebase). Consider alternatives against real criteria, not aesthetics:

- library + Jupyter notebooks
- CLI + YAML/TOML config for batch, plus a thin viewer for only the interactive parts
- local web app (Streamlit/Dash) — but weigh the install and offline constraints honestly
- keep the single tkinter app, better organised

Constraint to respect in your reasoning: this must run in a conda environment on lab Windows machines, offline, with no build step, for researchers who are not software users by trade. Recommend one direction.

## Question 3 — Is the data model right?

- `LoadedTrajectoryFile` (`data_structures.py:146`) holds raw arrays *and* mutable result slots — `processed_data`, `shoaling_results`, `thigmotaxis_results` are `None` at construction and assigned post-hoc by whoever runs an analysis. Assess whether analyses should instead be pure functions over an immutable dataset returning results. Tie this to the stale-state problems Pass D is looking for: are those bugs a *consequence* of this design?
- Results are dataclasses with dozens of named fields (`ShoalingResults` has ~30), which then get hand-transcribed into CSV rows in `export.py`. Everything ultimately lands in a table for R/Prism. Would a tidy long-form DataFrame (`file, fish, metric, value, unit, parameters`) be a better central representation? What would it cost, and what would it make easy that's currently hard — pooling across files, adding a metric without touching the exporter, provenance?
- Where should the boundary with statistics sit? Currently: export CSV, do stats elsewhere. Is that right, or should the tool do the group comparisons it's clearly heading toward (the group-aware exports in commit `dcf88e1`)?

## Question 4 — Is centroid-based analysis the right foundation?

The core package infers behavior from centroid trajectories. The two newest modules (`head_detection/`, `fish_posture_analyzer.py`) work from masks and midlines — real head direction and body posture.

If posture-based measurement is strictly better for the turn/laterality questions this project cares about, then a large part of `processing.py` and `bout_analysis.py` is a legacy layer that should be **frozen and documented rather than refactored**. That's a very different instruction than "clean it up".

Answer directly: which pipeline is the future, what should happen to the other, and what does that imply for how much effort Passes C, D, and F deserve? Pass E asks the tactical version of this question ("where do these modules live?"); answer the strategic version.

## Question 5 — Is reproducibility an architecture problem here?

For a research tool the artifact that matters is a *rerunnable analysis*. Two observed symptoms:

- no settings persistence anywhere in the repo — parameters live in tkinter widget state and die with the window
- exported CSVs record `Unit` but not the thresholds, smoothing settings, fps, or scale factor that produced them

These read as symptoms of a design where configuration is UI state rather than a first-class artifact. Assess whether a config-driven pipeline — parameters in a file, one command, deterministic output, provenance stamped into every export — would serve the science better than any amount of GUI polish. If so, that reframes Pass G's findings as treating symptoms.

## Question 6 — Does it scale to where this is going?

What happens at 100 files? Multiple experimenters in the lab using it? Comparing across batches recorded months apart? Adding a new metric — how many files must change today, and what should that number be? Would you hand this to a new lab member, and what's the first thing they'd hit?

---

## Deliverable

`AUDIT_H_APPROACH.md` in the repo root, containing:

- **(a)** The landscape survey as a table, with links and dates, and a clear adopt / adopt-in-part / keep-building verdict.
- **(b)** One recommended direction for the architecture, with the reasoning that got you there — not a menu of options.
- **(c)** An explicit answer on centroid vs posture as the foundation, and what it means for the other passes' priority.
- **(d)** A "what I'd change nothing about" section. If parts of the current design are genuinely well-chosen, I need that as clearly as the criticism.
- **(e)** A staged path, if you're recommending change: what's the smallest first step that's independently valuable and doesn't require finishing the whole migration to pay off?
- **(f)** Any place where your recommendation contradicts a constraint in another pass prompt, called out explicitly.

## Constraints

- Report only. No code changes, no restructuring.
- **Be honest, including about inertia.** Rewrites usually fail, and a working tool a lab actually uses has real value that a better-designed unfinished one doesn't. "Keep going as-is, here's why" is a legitimate answer and should be treated as a live option, not a strawman you knock down.
- Weigh recommendations against the real situation: **one maintainer, who is a researcher first, with no tests and no CI.** A proposal requiring a software team is not a proposal. If your recommendation only works with a test suite in place first, say that — it's useful information.
- Don't hedge into a list of considerations. I want a verdict I can disagree with.
