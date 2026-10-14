# Audit I — Publication readiness

Executed 2026-10-14 against `polish/ra-ready` at `73680c7` (pushed; CI green
on that commit, run 38516830434), working tree clean apart from an untracked
`idtrackerai.log`.

**The question:** could a paper's numbers be traced, months later, to the
exact code, tracking and settings that made them; could a reader cite and
reuse the tool; and does a user have what they need to write the methods
and choose the statistics?

**Labels:** `CONFIRMED` = I ran it, or read it out of the code, the data or
the repository. `PLAUSIBLE` = reasoning from reading only.

**Severity**, for publication:
**High** = a reviewer or a later re-analysis would be stuck.
**Medium** = it costs the author time or invites an error.
**Low** = tidiness.

Nothing in the repository was changed while auditing. The fixes are
proposed here and will be applied one at a time, after the priorities are
agreed, each recorded in the *Fix status* table at the end and in
`CHANGELOG.md`.

---

## How it was checked

| Question | How |
|---|---|
| Version tracing | Read `pyproject.toml`, `fish_analyzer/__init__.py`, `CHANGELOG.md` headings, `git tag`, `git log main..HEAD`; searched the export code for any version or commit. |
| Settings travelling with results | Read `results_export.py` end to end; read the keys of a real `session.json` (`freeswim-test-data/GRNA_mpegcontrol_V2/session_GRNA_3_6m`); traced the tank-outline setter in `gui/data_tab.py`. |
| README | Read it against the code (`results.METRICS`, `processing.py` defaults, the package layout). Ran its Python example and the developer check script it names on a real session. |
| Methods and citations | Searched code and docs for citations and methods text; read the git history of the removed Methods panels. |
| Citable release | `gh repo view` (visibility, licence); `git tag`; looked for `LICENSE`, `CITATION.cff`; read the licences of idtracker.ai and traja from their installed metadata. |
| Validation | Read the CHANGELOG's own evidence, the handover note and the earlier audits; ran the test suite (347 passed, 34 s). |
| Statistics | Read the export READMEs (`README.txt`, `PRISM_README.txt`) and `results.py`'s statement of the experimental unit. |

---

## The short version

The analysis itself is in good shape: the measures are defined, tested and
named for what they are, and the export already says the tank is the unit.
**Most of what is missing is provenance and paperwork.**

- **An export cannot say which code made it.** Two sets of changes that
  moved the numbers (denoising, the 80% rule, 1 s freezes, jump removal,
  95th percentile) sit under "Unreleased" while the app still calls itself
  2.2.0, and no export records a version or commit. `CONFIRMED`
- **The tracking settings are left behind.** idtracker.ai's version,
  thresholds, region of interest and number of animals are all in each
  `session.json`, but no export file carries them. The file the README
  sends you to for statistics also lacks the analysis `Setting_` columns.
  `CONFIRMED`
- **The repository is public with no licence**, so nobody may legally reuse
  it, and the app imports and patches idtracker.ai, which is GPLv3+. The
  licence choice is the owner's. `CONFIRMED`
- **No methods text, no citation list, no release, no DOI.** `CONFIRMED`
- **The README is out of date in several places**, and the real-data check
  script it points developers to crashes on the current code. `CONFIRMED`
- **No single statement of what has and has not been validated.** The
  evidence exists, scattered across the CHANGELOG and audits; the main gaps
  are no ground-truth check of positions and speeds, and the resting
  threshold not yet checked against this rig's noise floor. `CONFIRMED`
- **The statistics guidance covers the group comparison only.** It says
  nothing about minute-by-minute (repeated measures), about choosing one
  primary measure among six, or about mixed models, which the README rules
  out entirely. `CONFIRMED`

---

## Findings

### I1 — No licence on a public repository. **High.** `CONFIRMED`

`gh repo view` reports `"visibility":"PUBLIC"` and `"licenseInfo":null`.
There is no `LICENSE` file, and the README ends with
`[Add license here]`. Without a licence, others may read the code but not
legally run, change or share it, so a paper cannot point readers to it as
reusable software. Journals and Zenodo both expect one.

**The constraint.** The app imports idtracker.ai and patches its classes at
run time (`fish_analyzer/idtrackerai_setup_window.py`,
`fish_analyzer/idtrackerai_track.py`). idtracker.ai's installed metadata:

```
idtrackerai  License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)
traja        MIT
```

Code that imports and extends a GPL library is commonly treated as a
derivative work, so GPLv3 for this project is the choice that raises no
question. A permissive licence (MIT, BSD) for the analysis package alone is
arguable but needs care. `PLAUSIBLE`: this is a reading of common practice,
not legal advice, and the institution may have its own policy.

**Proposed fix.** The owner picks the licence (GPLv3 recommended for the
reason above). Then: a `LICENSE` file, the licence in `pyproject.toml`, and
the README's last section.

### I2 — An export cannot be traced to the code that made it. **High.** `CONFIRMED`

- `fish_analyzer/__init__.py:81` says `__version__ = "2.2.0"`.
- `CHANGELOG.md` has two `## Unreleased` sections above `## 2.2.0`
  (lines 3 and 50). Between them they change what is measured: positions
  denoised over 0.1 s, fish tracked under 80% left out, freezes of at
  least 1 s, tracking jumps removed, top speed moved from the 99th to the
  95th percentile. **A 2.2.0 export from August and one from today report
  the same version and different numbers.**
- No file the Results export writes (`results_export.py`) contains a
  version or commit. Its README records only the export time.
- `git tag` prints nothing. `main` is at `e484a4b` (2026-08-05);
  `git log main..HEAD` counts 46 commits not merged.

**Proposed fix.**
1. Bump the version when numbers change (the next is 2.3.0) and give the
   Unreleased sections that number.
2. Write the analyzer version, and the git commit when the app runs from a
   checkout (with "+modified" if the working tree has changes), into every
   export: a line in `README.txt` and an `AnalyzerVersion` column in
   `results_per_tank.csv`.
3. Release from `main` after merging (see I7).

### I3 — Tracking settings do not travel with the results. **High.** `CONFIRMED`

A real `session.json` holds everything needed to describe the tracking:

```
version                      '6.0.14'
intensity_ths                [36, 255]
area_ths                     [250.0, inf]
use_bkg                      True
background_subtraction_stat  'median'
resolution_reduction         0.8
number_of_animals            6
roi_list                     ['+ Polygon [[1101.0, 86.0], ...
track_wo_identities          False
tracking_intervals           [[0, 18000]]
```

`results_per_tank` (`results_export.py`) takes only `median_body_length`,
`estimated_accuracy` and the video name from it. Also not recorded:

- **Whether identity training was finished early** with "Finish identity
  training now" (`idtrackerai_track.py` prints it to the tracking log only).
- **Whether the session was reviewed and corrected** in idtracker.ai's
  validator. The app shows a "Reviewed" tick (`tracking.session_reviewed_on`)
  but does not export it.
- **The rolling background**, when chosen. It is an option this app added,
  so `session.json` alone may not show it. `PLAUSIBLE`: not traced.

Separately, **`results_per_tank.csv`, which the export README calls the
file for statistics, has no `Setting_` columns.** `results_per_fish.csv`
has them. A tank-level table copied on its own loses the analysis
settings.

**Proposed fix.** A `tracking_settings.csv` in the export, one row per
session, with the keys above plus early finish, reviewed date and
background. Add the `Setting_` columns to `results_per_tank.csv`.

### I4 — Choices made in the app are not saved, so an export cannot be regenerated. **Medium.** `CONFIRMED`

`_set_tank_outline` (`gui/data_tab.py:837`) keeps a hand-drawn outline in
`self.file_arena_definitions` only; nothing writes it to disk. The same
holds for session names, groups and the scale: closing the app loses them.
Audit H recorded this in July ("no settings persistence, arenas redrawn
every session, exports without provenance"); provenance is I2–I3 here, and
the rest is still open.

For a paper this matters most for **time near the wall**: when the outline
was drawn by hand, neither the outline nor its source (idtracker.ai's
region of interest or a drawing) is in the export. Groups and scale can be
read back from the export's `Group`, `Unit` and `PixelsPerUnit` columns.

**Proposed fix.** Smallest version: export each session's outline corners
in pixels and their source, so the number can be recomputed. Full
persistence of names, groups and outlines (for example a file beside each
session) is a larger change. That is a separate decision, and probably not
needed for publication.

### I5 — The README is out of date. **Medium.** `CONFIRMED`

Checked against the code:

| README says | Code does |
|---|---|
| Opening line: measures "freezing, turning bias" | Results shows six measures (`results.METRICS`); turning bias is computed in `processing.py` but shown nowhere; "freezing" is now **Time resting**. |
| "What is measured" table: Individual = "Speed, distance, freezing, path straightness, turning bias" | Same mismatch. |
| Smoothing changes "the 99th-percentile speed by about 6%" | Top speed is the 95th percentile (`Speed95`). |
| "one cut short by lost tracking is reported as *censored* ... rates are per unit of observed time" | No freeze rate or censored count is shown or in the Results export. |
| "Frozen while slower than 0.5 BL/s for at least 1 s" | The box and the measure are labelled resting. |
| Sessions without identities | Not mentioned, though the app now handles them and the export marks every row. |
| Layout lists `export.py # CSV exports` | The Results export is `results_export.py`. `results.py`, `shoal_summary.py`, `fish_outlines.py` and `idtrackerai_rolling.py` are missing from the list. |
| "`python scripts/verify_on_session.py path/to/session_folder`" | **Crashes:** `KeyError: 'speed_p99'` at `scripts/verify_on_session.py:109`, after two checks pass. The real-data re-checks of Audit B cannot currently run. |
| No "Citing" section | See I6. |
| `[Add license here]` | See I1. |

The Python API example **runs** (checked on `session_GRNA_3_6m`). Note that
it reports body lengths from each session's own body length, while the app
uses one scale for every session. That is fine for an example, but a reader
should not compare its numbers across sessions.

Also stale: `docs/handover-2026-10-08.md` says the branch is local only and
CI has not run (both no longer true); a comment at `gui/data_tab.py:879`
refers to a "Methods tab" removed in `9a7d9d6`.

**Proposed fix.** Rewrite the "What is measured" section from
`results.METRICS`; fix the lines above; repair `verify_on_session.py` for the
current measure names; add "Citing" and "Licence" sections; update the
handover and the comment.

### I6 — No methods text and no citations. **High.** `CONFIRMED`

Nothing in the app, the export or the docs gives the author a methods
paragraph. The per-tab "Methods" panels that once existed were removed with
their tabs (for example `9a7d9d6`, the Spatial tab's). The only citation in
the code is SuperPlots (Lord et al. 2020, `results.py:21`), in a docstring.

A methods section would need, and cannot currently get from one place:

- **Tracking:** idtracker.ai, its version and the settings in I3.
- **Processing:** denoising window, the 80% tracking rule, jump threshold,
  resting rule, straightness speed rule, wall-zone width, scale and how it
  was measured.
- **Measures:** one-line definitions, which `results.METRICS` already holds.
- **Unit of analysis and statistics** (see I9).
- **Software to cite:** idtracker.ai (the paper its documentation asks
  users to cite; check that page for the one matching 6.x), SuperPlots
  (Lord et al. 2020, J Cell Biol 219:e202001064), traja (still imported,
  `processing.py:56`), and this tool itself (see I7).

**Choices with no source.** These are stated in the app but cite neither
literature nor a validation:

| Choice | Default | Basis on record |
|---|---|---|
| Wall zone width | 15% of the shorter side | None found (`spatial.py:335`). |
| Resting speed | 0.5 BL/s | Simulation; real-rig noise-floor check still pending (handover). |
| Shortest rest | 1 s | Reasoning in CHANGELOG; no literature cited. |
| Tracking jump | 25 BL/s | Reasoning in CHANGELOG. |
| Tracking minimum | 80% | Owner's decision, 2026-10-09. |
| Denoising | 0.1 s | Simulation plus four real sessions (CHANGELOG). |

These can all be defended, but a reviewer will ask, and the answer should
be written down once.

**Proposed fix.** A `METHODS.txt` in every export, filled in with that
export's actual versions, settings, scale and n, written as a draft
paragraph to edit. A `docs/METHODS.md` giving the reasoning and sources for
each default (the table above), and a "Citing" section in the README.

### I7 — No citable release. **Medium.** `CONFIRMED`

No tag, no GitHub release, no `CITATION.cff`, no DOI. The repository is
public, so Zenodo's GitHub integration would mint a DOI for a release.
This needs I1 (a licence) first, and is best cut from `main` after the
branch is merged.

**Proposed fix.** Add a `CITATION.cff` (author list and ORCIDs from the
owner). Merge to `main`, tag `v2.3.0`, create the GitHub release. Enabling
Zenodo and creating the release use the owner's GitHub and Zenodo
accounts, **so I will ask before each of those steps.**

### I8 — What has and has not been validated is not stated anywhere. **High.** `CONFIRMED`

Gathered from the CHANGELOG, the handover and Audits A and B:

**Checked**

| What | Evidence |
|---|---|
| Measure arithmetic | 347 tests on synthetic tracks with known answers (`tests/test_metric_correctness.py` and others); CI green. |
| Scale conversion | Audit A: cm and BL runs agree to 8.9e-16. |
| Effect of tracker wobble and dropout | Simulated fish (CHANGELOG, "measures that tracker wobble..."). |
| Denoising window | Four real sessions: a resting fish recovered 4.0 of 10 min resting against 0.7 unsmoothed. |
| Top-speed percentile | Three sessions tracked with and without identities from one setup. |
| Time near the wall | Two sessions: outline and positions drawn on a frame and checked by eye, and recomputed in pixels. |
| A near-still fish is real | Owner checked in the validator. |

**Not checked**

| What | Why it matters |
|---|---|
| Positions, speed and distance against ground truth (hand-marked fish positions on sampled frames) | Every measure rests on idtracker.ai's positions; only its self-estimated accuracy is on record. |
| Resting threshold against this rig's noise floor (film an object that cannot move) | Time resting depends on it; already on the handover's to-do list. |
| The new camera and lighting | Segmentation on the badly lit videos is unresolved (handover). |
| Pooled (no identities) measures against per-fish measures beyond top speed | Both kinds are exported side by side. |
| Real-data re-checks of Audit B | `verify_on_session.py` crashes (I5). |
| Any biological result | So far one tank per group: observations, not a test. |

**Proposed fix.** A `docs/VALIDATION.md` holding these two tables, linked
from the README and referenced from `METHODS.txt`. Running the two missing
checks (ground truth, still object) is lab work for the owner. I can write
the scoring script for the ground-truth check if wanted.

### I9 — Statistics guidance stops at the group comparison. **Medium.** `CONFIRMED`

What is there is right. `README.txt` says the tank is the unit and sends the
reader to `results_per_tank.csv`; the Prism tables are tank values;
`PRISM_README.txt` suggests a t-test or one-way ANOVA. The gaps:

1. **Minute by minute is repeated measures**, the same tank measured every
   minute. `PRISM_README.txt` explains how to draw it but not how to test
   it. In Prism that is a two-way repeated-measures ANOVA, or its
   mixed-effects model when minutes are missing. For jamovi or R there is
   no long table of tank values by minute: `results_by_minute.csv` is per
   fish, and the per-tank values exist only in Prism's wide layout.
2. **Six measures, no primary one.** Testing all six at 0.05 makes a false
   positive likely. The project's question ("is one group more active?")
   points to distance per tracked minute as the primary measure, with the
   rest secondary or descriptive. That should be said once, in the export
   README.
3. **Mixed models are ruled out without a reason a statistician would
   accept.** `README.txt` says "Do not run tests on the fish rows." A mixed
   model on the fish rows is a valid alternative to tank means: in jamovi,
   the GAMLj module's Mixed Model, with the measure as dependent, Group as
   factor and Session as the cluster (random intercept). The guidance
   should explain when to use it:
   - With equal fish per tank and few tanks, it gives essentially the same
     group test as a t-test on tank means, so it adds little. `PLAUSIBLE`
     (standard result for balanced nested designs).
   - With two or three tanks per group the tank variance is poorly
     estimated and often comes out at zero (a "singular fit"). The model
     then treats fish as independent, which is the error to avoid.
   - It helps when tanks hold different numbers of analysed fish (fish left
     out under the 80% rule), or with a fish-level covariate.
   - It cannot be used on sessions tracked without identities, which have
     one row per tank.
   - A tank-level covariate, such as the sex ratio noted for the gRNA
     experiment, goes in either model as a column per tank.
4. **No guidance on n.** "A group needs several sessions to be tested"
   (app README). A number, or a pointer to a power calculation from the
   pilot's tank-to-tank variation, would be more useful.

**Proposed fix.** A `STATISTICS.txt` in the export (or a section of
`README.txt`) covering the four points. Add a long tank-by-minute table
(`results_by_minute_per_tank.csv`) for jamovi and R. Mirror the guidance
in `docs/METHODS.md`.

### I10 — Small items. **Low.** `CONFIRMED`

- Stale comment at `gui/data_tab.py:879` ("the Methods tab reads these").
- The handover note's "local only, CI has not run" (I5).
- `idtrackerai.log` sits untracked in the repository root. It probably
  belongs in `.gitignore`.

---

## Proposed order

Grouped so that each step is useful on its own:

| # | Fix | Findings | Needs from the owner |
|---|---|---|---|
| 1 | Version and commit in every export; version bump to 2.3.0 for the Unreleased changes | I2 | — |
| 2 | Tracking settings, early finish, reviewed date and `Setting_` columns in the export | I3 | — |
| 3 | Tank outline corners and their source in the export | I4 (small part) | — |
| 4 | `METHODS.txt` in the export; `docs/METHODS.md` with sources for each default | I6 | Confirm the idtracker.ai paper to cite |
| 5 | Statistics guidance and a long tank-by-minute table | I9 | Agree distance per tracked minute as the primary measure |
| 6 | `docs/VALIDATION.md` | I8 | — |
| 7 | README rewrite of the stale parts; repair `verify_on_session.py`; small items | I5, I10 | — |
| 8 | Licence | I1 | **Choose the licence** |
| 9 | `CITATION.cff`, merge to `main`, tag, GitHub release, Zenodo DOI | I7 | Authors and ORCIDs; **approve each account action** |

Out of scope here, and left as decisions: full persistence of names, groups
and outlines (I4); the ground-truth and still-object checks (I8, lab work).

---

## Fix status

| Finding | Status | Commit | Notes |
|---|---|---|---|
| I1 | open | | awaiting licence choice |
| I2 | fixed (release is I7) | see `git log --grep "Audit I, I2"` | Version 2.3.0; `fish_analyzer/provenance.py` gives the version with the commit (and "modified") from a checkout, with or without git installed. Written to `results_per_tank.csv` and the shoaling summary (`AnalyzerVersion`) and to both export READMEs. Tests: `tests/test_provenance.py`, including that the CHANGELOG's top entry is the package version. |
| I3 | fixed | see `git log --grep "Audit I, I3"` | `tracking_settings.csv` and `setups/` in the Results export; `freeswim_tracking.json` written by `tracking.run_tracking` after a successful run (app version, rolling background, setup and its text); early finish read from the session's `idtrackerai.log` ("Training interrupted by user"), so it is known for older sessions too. `Setting_` columns added to `results_per_tank.csv`. Checked on two real sessions: thresholds, ROI, intervals read correctly; both show "rolling not recorded", which is right, since they predate the record. Tests: `tests/test_tracking_settings.py`. |
| I4 | export part fixed; persistence left as a decision | see `git log --grep "Audit I, I4"` | `tank_outlines.csv`: corners in video pixels, source ("idtracker.ai region of interest" / "drawn in Free Swim Analyzer"), wall zone width. `ArenaDefinition.source` and `ThigmotaxisResults.arena` carry it. Checked on two real sessions: 22 and 19 corners, matching `session.json`. Saving names, groups and drawn outlines between runs of the app is not done. Test: `tests/test_tracking_settings.py`. |
| I5 | open | | |
| I6 | fixed; three defaults still lack a published source | see `git log --grep "Audit I, I6"` | `fish_analyzer/methods.py` writes `METHODS.txt` into every Results export. `docs/METHODS.md` gives the basis for each default. References checked against Crossref 2026-10-14; idtracker.ai's site lists the 2026 eLife paper (10.7554/eLife.107602) as its latest and the 2019 Nature Methods paper before it. Proofread on two real sessions, which exposed a false "one scale for every session" when scales differed; now flagged instead. Still unsourced, and said so in `docs/METHODS.md`: 15% wall zone, 25 BL/s jumps, 1 s rests. Tests: `tests/test_methods.py`. Also found: the 95th percentile's explanation said 1% (fixed, `0f84c62`). |
| I7 | open | | |
| I8 | open | | |
| I9 | open | | |
| I10 | open | | |
