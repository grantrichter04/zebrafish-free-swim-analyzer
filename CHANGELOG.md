# Changelog

## 2.3.0 - numbers that resist tracker noise, and exports that say where they came from

Not yet tagged. Everything below changes exported numbers relative to
2.2.0, so it carries a new version: before this, exports made with this
code still said 2.2.0.

### Exports say which code made them

Audit I ([docs/audit/AUDIT_I_PUBLICATION.md](docs/audit/AUDIT_I_PUBLICATION.md))
found that an export could not be traced to the code that made it.

- **The analyzer version is in every export**: an `AnalyzerVersion` column in
  `results_per_tank.csv` and the shoaling summary, and a line at the top of
  both READMEs. When the app runs from a git checkout, as on the lab laptop,
  the version names the commit, and says "modified" if the code differed
  from it: `2.3.0 (commit 6e84edf, modified)`. git need not be installed.
  (Audit I, finding I2.)
- **How each session was tracked is in the export**: `tracking_settings.csv`,
  one row per session, with idtracker.ai's version, the number of animals,
  intensity and area thresholds, background, resolution reduction,
  tracking intervals, region of interest, whether identity training was
  finished early, when the session was last reviewed, and which version of
  this app tracked it. A `setups` folder holds each session's setup file as
  it was when tracked. Tracking now leaves `freeswim_tracking.json` in each
  new session folder for this, since idtracker.ai's own record cannot show a
  rolling background, and the setup beside the videos may change later.
  Sessions tracked earlier say "not recorded" where that is unknown.
  `results_per_tank.csv` now carries the `Setting_` columns too.
  (Audit I, finding I3.)
- **The tank outline is in the export**: `tank_outlines.csv` gives, for
  each session, the corners of the outline time near the wall was measured
  against, in video pixels, whether it was idtracker.ai's region of
  interest or drawn in this app, and the wall zone's width. An outline
  drawn with "Tank outline..." is still not saved between runs of the app,
  but the export now keeps it, so the measure can be recomputed.
  (Audit I, finding I4.)
- **A draft methods paragraph is in the export**: `METHODS.txt`, filled in
  with the export's own idtracker.ai and analyzer versions, frame rate,
  recording length, fish per tank, scale, settings, fish left out, whether
  identity training was finished early or sessions were reviewed, and n per
  group, followed by the references (idtracker.ai 2026 and 2019, SuperPlots,
  traja, each checked against Crossref). What it cannot know, such as the
  statistical test, is left in [square brackets]. Sessions on different
  scales are flagged for checking, not described as one scale.
  `docs/METHODS.md` gives the basis for every default, and says where no
  published source has been found yet (the 15% wall zone, 25 BL/s jumps,
  1 s rests). The README gains a "Citing" section. (Audit I, finding I6.)
- **Guidance on the statistics is in the export**: `STATISTICS.txt` covers
  the tank as the unit (Welch's t-test or ANOVA on tank values, with a
  warning for any group of one tank), distance per tracked minute as the
  one primary measure with the other five corrected (Holm) or descriptive,
  minute by minute as repeated measures (in Prism and in jamovi), and when a
  mixed model on the fish rows in jamovi (GAMLj, Session as the cluster)
  helps and when it misleads: a singular fit with few tanks, and no fish rows
  for sessions without identities. `results_by_minute_per_tank.csv` gives
  the tank values by minute in long form for jamovi or R, and the Prism
  README says how to test minute by minute. (Audit I, finding I9.)
- **`docs/VALIDATION.md`** says in one place what has been checked and what
  has not, with where the evidence is. Not yet checked: idtracker.ai's
  positions against hand-marked frames, and the resting speed against this
  rig's noise floor. (Audit I, finding I8.)
- **README brought up to date**: the measures as Results shows them (no
  more turning bias, freeze rates or freezing), tracking jumps and sessions
  without identities explained, the Export results folder described, and
  the file layout current. `scripts/verify_on_session.py`, which crashed
  on the renamed top speed, is repaired, checks the Results export rather
  than the old per-fish one, and is now run by the tests.
  (Audit I, findings I5 and I10.)
- The explanation of 95th percentile speed said "faster than this for only
  1% of the time"; it is 5%.

### Small display changes

- **Time to track each video** is shown in a new column in "1. Videos" on
  the Tracking tab, and the batch summary gives the total. It is kept in
  `freeswim_tracking.json` as `tracking_seconds`, so it survives a restart;
  videos tracked before this show nothing there.
- **Identity numbers on the inspector's dots are larger**, with a black edge,
  and sit just beside the dot so the dot's colour stays visible. Inside the
  dot they were too small to read at the default dot size.
- **Results are coloured by group.** Each group has a colour of its own and
  its tanks are shades of it, on every Results and Shoaling plot. Two groups
  are no longer neighbouring blues on the minute-by-minute plot, and the
  group comparison reads by condition before tank.
- **Speed ridges keep each group together**, under its name, with a gap
  between groups. "Where they swim" puts each group on a row of its own,
  named down the left.
- **The minute-by-minute band is one standard error of the group mean**,
  not a 95% confidence interval, and is off until "Band" is ticked. From
  three tanks that interval is the mean plus or minus 4.3 standard errors,
  and two groups' intervals covered the panel.
- **Each minute-by-minute layer can be switched off**: "Group mean",
  "Tanks" and "Band" sit beside "Each fish", and the legend names only what
  is on show.
- **Error bars on the group comparison**, beside each group's mean: a 95%
  confidence interval (the default, from three tanks), the SD, or none,
  chosen in "Error bars". Both are worked out from the tank means, never
  the fish.
- **A session's group is remembered.** "Set group..." keeps it in
  `freeswim_session.json` in the session folder, and it comes back when the
  session is loaded again. The dialog and the Sessions note say that
  several rows can be selected (Shift- or Ctrl-click) to group them at once.
- **The note above the Results plot describes the view on show.** It
  described the group comparison whatever was showing.

### Sessions without identities, resting time, and checking it on the video

- **Sessions tracked without identities are recognised** from their
  session.json and reported per tank, not per fish: their columns are tracks
  that pass from fish to fish, so every track is pooled into one row
  ("all fish (no identities)"). The fish count is read from the setup
  printed in idtracker.ai's log, because without identities idtracker.ai
  overwrites it with the most blobs it saw in a frame. Shoaling leaves them out and says why: fish go
  missing mostly while touching (median 0.6-0.7 BL from another fish in the
  frame before, against 1.1-1.3 BL overall), so close contact is lost.
  Sessions & Units shows "no identities" in place of idtracker.ai's
  accuracy, which it reports as 100% for identities it never assigned.
- **Time resting** is a measure on Results: the share of tracked time slower
  than the resting speed for at least the shortest rest, with that rule in
  the panel title.
- **Video Inspector, "Check the analysis"**: rings round fish the analysis
  counted as resting (with the settings that analysis ran with) and round
  frames dropped as tracking jumps, with buttons to step between jumps.
  Trails are cut at gaps and jumps, so they show only movement that was
  measured. Without identities, positions are unnumbered and one colour.
- **Minute by minute is drawn by group**: a thin line per session and a
  thick line for the group's mean; a 95% confidence band for groups with
  three or more sessions. "Each fish" adds the fish back.
- **Speed distributions** can show ridges per fish, tank or group (a group's
  ridge is the mean of its tanks', each tank counting the same), have a
  median tick, and a box to set where the speed axis ends. The
  cumulative curves follow the same fish / tank / group choice.
- **Panels** on Group comparison and Minute by minute can be ticked on and off.
- **Distance is shown per tracked minute** for every session, in place of
  total distance, which reads lower for a fish tracked less of the time.
  Total distance stays in the export.
- **Top speed is the 95th percentile** (was the 99th). On three sessions
  tracked with and without identities from one setup, the artifacts left
  after jump removal moved the 99th up 2.2-4.4% and the 95th 0.7-1.8%;
  both tell fish in a tank apart equally. Column `Speed95`.
- **Where they swim** has a "Squares across" box (48 by default) for the
  heatmap grid, also used by the export.
- **Export results** writes a folder: a CSV for each view on Results, a
  `prism` folder of tank-level tables laid out to paste into GraphPad Prism
  (a Column table per measure, an XY table per measure by minute), a
  Tracking column on every row, and a README saying what each column is.
  `results_per_tank.csv` leads it: one row per session with every measure
  for the tank and what the session was (fish in the tank, minutes, frame
  rate, body length, idtracker.ai accuracy, video), and the README opens
  by naming it as the file for statistics and giving n per group.
- **Tracking tab** draws idtracker.ai's identity training as it runs.

### Measures that tracker wobble and dropout cannot move

A review against simulated fish found that the tracker's ~1 px frame-to-frame
wobble and gaps in tracking could bias the measures whenever groups differ in
activity or in how well they were tracked. These are now accounted for; every
setting is a box on Sessions & Units and is written into every export as a
`Setting_` column.

- **Positions are smoothed over 0.1 s** (three frames at 30 fps) before any
  measure is computed, within stretches of continuous tracking only (0 turns
  it off). Checked on four real sessions, this is the shortest window that
  recovers a resting fish's freezing (4.0 of its 10 minutes, against 0.7
  unsmoothed); 0.17 s and 0.3 s recover no more and take 11% and 18% off the
  99th-percentile speed, where 0.1 s takes 6% and under 2% off distance.
  In simulation 0.1 s holds up to a wobble of about 0.01 body lengths (the
  recordings measured 0.003-0.005); beyond that use 0.17 s. Unsmoothed,
  a motionless fish read 0.3-0.7 BL/s and was frozen 3-64% of the time
  instead of 100%, and a fish swimming 0.5 BL/s read 17-67% too far. Normal
  swimming changes by about 2%. This is denoising the positions, not the
  display smoothing on Minute by minute, which stays off unless chosen.
- **Fish tracked less than 80% of the time are left out** (was 1%), and named
  with their tracked share in the run report, on Results and in the exports.
  A session whose fish were all left out still appears in the exports.
- **A freeze must last 1 s** (was 5 frames, about 0.17 s), and a gap of up to
  0.5 s does not break one if the fish is in the same place either side, so
  dropout cannot make a group look like it freezes less.
- **Path straightness uses only seconds faster than 1 BL/s.** Still seconds
  scored near 0 from wobble alone, so a fish that rested a lot looked like
  it swam tortuously. `StraightnessSeconds_pct` is exported.
- **`DistancePerTrackedMin`** is exported beside total distance.
- **Tracking jumps are left out.** A frame-to-frame step faster than 25 BL/s
  is the tracker moving to another fish (an identity swap) or to something
  that is not a fish, not swimming. The frame it lands on is treated as
  untracked, so the jump adds nothing to speed or distance. Each fish's count
  is in a new Jumps column on Results and exported as `TrackingJumps`.

## 2.2.0 - one install, video to results

**Tracking is part of the app.** A new Tracking tab takes a folder of videos
through idtracker.ai:

- Choose the experiment folder; each video shows as not tracked, tracked,
  incomplete, running or failed.
- "Configure new setup..." opens idtracker.ai's own window on a video. That
  window now has one action, "Save setup and close", which writes a named
  setup (.toml) beside the videos. It cannot start tracking, so a setup is
  never tracked one video at a time by accident.
- After saving, the app offers to open the setup on the next video for a
  quick check, since one setup is used for every video in the folder.
- "Track all untracked videos" runs each video in turn. One failure does not
  stop the rest, "Stop" ends the run, and a folder can be resumed later.
- "Load tracked sessions for analysis" loads every tracked session at once.

**One environment, one installer.**

- `install.bat` builds a single conda environment holding idtracker.ai and the
  analyzer, from pinned versions, and adds a desktop shortcut. On a computer
  without an NVIDIA GPU it installs the CPU build of PyTorch instead of
  stopping, so a computer used only for analysis can be set up too.
- "Check Setup" in the status bar, and `fish-analyzer --check`, report whether
  the machine is ready: Python, analyzer, OpenCV, idtracker.ai, PyTorch, GPU.
- The shortcut shows a "Starting" window while the app loads.
- OpenCV is now `opencv-python-headless`, the build idtracker.ai uses, so the
  two can share an environment.
- Python 3.10 or newer is required. `requirements.txt` is gone;
  `pyproject.toml` is the only dependency list.

**Sessions & Units replaces Data Setup & Calibration.**

- One table lists every loaded session: fish, length, tracking quality,
  idtracker.ai accuracy, body length, frame rate and the scale in use.
- "Add sessions..." takes one session folder or a folder containing several.
- One scale applies to every session. Centimetres can be measured by clicking
  two points on a video frame. Body lengths use one value for the experiment
  instead of each video's own, because idtracker.ai's body length moves with
  lighting and threshold (73.2 and 79.7 px for two videos from one rig).
- The "active file", per-file calibration, the editable frame rate and the
  pixels-only option are gone.
- Exports gained a `PixelsPerUnit` column beside `Unit`.
- The freeze threshold is converted when the unit changes, so it keeps the
  same physical speed.

**Results tab.** Total distance, median speed, 99th percentile speed and path
straightness as SuperPlots: fish as small dots,
session means as large markers, and a group line at the mean of its sessions,
because the tank is the experimental unit. One table, one export.

**Results also shows** time near the wall, computed from the arena outline
drawn in idtracker.ai's setup window and read against the level even use of
the tank would give; a speed-distribution view (a cumulative curve per
session, a ridge per fish); and a plain description of every measure.

**Minute by minute.** Results and Shoaling each have a view that works every
measure out again on each minute of the recording by itself: a thin line per
fish and a thick one per session. Nothing is smoothed unless the reader picks
a Smoothing (a 3 or 5 minute running mean), and a smoothed plot says so along
its top. The plot under the video in the Video Inspector has the same choice,
in seconds, under "More options".

**Shoaling tab rebuilt, and run with everything else.** Nearest-neighbour and
inter-individual distance as SuperPlots against what randomly placed fish
would give, with the area the shoal covers (convex hull) beside them, one
table, one export. It no longer has its own run button, file list or
parameters: it samples once a second and runs from "Run All Analysis".

**Names, groups and review on the sessions table.** Rename a session, put
several in a group, and open idtracker.ai's validator on one; a session saved
from the validator is ticked as Reviewed and reloaded.

**Position smoothing removed.** The Savitzky-Golay option changed distance and
speed by about 1% and path straightness by about 1%; the metrics it mattered
for were withdrawn in 2.1.0.

**Video Inspector reorganised.** The left column is three short groups: the
session, what to show, and export. The everyday choices are in view: fish
positions, lines to the nearest neighbour, a trail of any length in seconds,
and which shoaling measure to plot under the video. The less used ones (an
outline around the shoal, lines from one fish to all the others, dot size)
are under "More options". The video loads with the session, and "Find
video..." appears only when it is not found. Fish are numbered as in Results
and Shoaling. The scrubber and the step buttons move one frame at a time, and
playback faster than 1x skips frames.

New in the inspector: "Fish outlines, as idtracker.ai saw them" redraws the
blobs idtracker.ai segmented, from the thresholds, background and region of
interest saved in the session folder. Scrolling on the video zooms about the
pointer, dragging moves the view and a double-click resets it; exports are
always the whole frame. Distance lines are dark with a white edge, so they
can be read on a brightly lit tank.

Removed from the inspector: the trail opacity and width sliders, the
step-size and jump-to boxes, and the video quality setting. Fixed: a second,
stationary cursor could appear on the time panel and in exported clips.

**Spatial Analysis folded into Results.** "Where they swim" on Results is the
position heatmap: one map per session, square cells of one size and one
colour scale for all sessions, with the tank outline. "Tank outline..." on
Sessions & Units draws an outline on a video frame for sessions that have
none from idtracker.ai, or a wrong one, and time near the wall follows it.
The tab is gone, and with it the custom region of interest, the wall-time
time series, the per-fish heatmap grid, the border-width, sampling and
smoothing settings, and the separate thigmotaxis export button.

**Individual Analysis removed.** Results replaces it. Its two CSV export
buttons went with it; the Results export is the one per-fish file.

**Bout Analysis removed.** The bout model was built for larval dart-and-glide
swimming and does not fit adults, which swim continuously. The tab, the
`Bout_*` and `IBI_*` export columns, and the Video Inspector's bout overlay,
bout time panel and fish zoom are gone.

**Repository layout.** The audit trail moved to `docs/audit/`, the standalone
scripts to `extras/`, and `run_analyzer.py` was removed in favour of the
shortcut and `python -m fish_analyzer`.

## 2.1.0 - analysis overhaul and UX improvements

**New Bout Analysis tab:**
- Detects individual swim bouts (darts) from speed traces — works for both larvae and adults
- Per-bout metrics: duration, peak speed, displacement, distance, heading change
- Summary stats: bout rate, inter-bout interval, duration/speed distributions
- Per-bout laterality analysis with turn direction counts and laterality index
- Distribution plots (bout duration, IBI, peak speed, heading change histograms)
- Per-fish laterality bar charts
- CSV export of every detected bout with all metrics

**New behavioral metrics** replacing sinuosity and turn angles:
- **Freeze analysis** — episode count, mean duration, total time frozen (anxiety indicator)
- ~~**Burst analysis** — burst count, peak speed, frequency per minute (locomotor vigor)~~ **withdrawn, see below**
- ~~**Angular velocity** — mean turning rate in degrees/second~~ **withdrawn**
- ~~**Erratic movements** — count of sudden large direction changes per minute (startle/stress)~~ **withdrawn**
- **Path straightness** — sliding-window displacement/distance ratio (0 = circling, 1 = straight)
- **Turning bias** — laterality index and left/right turn counts (~~cumulative heading change, signed angular velocity~~ **withdrawn**)

> **Withdrawn 2026-08-01 — read this before using any 2.1.0 export.**
> Audit B tested every metric against synthetic trajectories with known
> answers and found eight columns to be measuring idtracker.ai centroid noise
> rather than fish behaviour: `MeanAngularVelocity_deg_s`,
> `ErraticMovementCount`, `ErraticMovements_per_min`, `BurstCount`,
> `BurstMeanSpeed`, `BurstFrequency_per_min`, `CumulativeHeading_deg` and
> `MeanSignedAngVel_deg_s`. A *perfectly straight* synthetic swimmer with
> 0.5–1.0 px of tracking noise reproduced the entire range those columns
> reported across four real recordings. They have been removed from the
> exports and the GUI rather than repaired, because they are not recoverable
> from centroid positions — restoring them needs head-direction tracking.
> Distance, speed, path straightness, laterality, NND, IID and hull area were
> verified exact against analytic ground truth and are unaffected.
> Full detail: [AUDIT_B_CORRECTNESS.md](docs/audit/AUDIT_B_CORRECTNESS.md).
>
> **Also changed 2026-08-01 (Phase 1).** A frame in which idtracker.ai did not
> locate the fish is now treated as *unobserved* rather than as "still"
> (bout analysis) or "moving" (freeze analysis) — the two modules previously
> disagreed. Episode metrics are computed within stretches of continuous
> tracking and never across a gap, and every rate and fraction is per unit of
> observed time. New columns: `ObservedDuration_s`, `LongestGap_s`,
> `FreezeEpisodes_Censored`, `Bout_Censored`, `IBI_N`. Renamed: `MaxSpeed` →
> `SpeedP99` (the raw max reported tracking teleports up to 321 BL/s),
> `FreezeCount` → `FreezeEpisodes_Complete`. `NetDisplacement` is now
> populated for every fish rather than NaN for half of them.
>
> **And Phase 2.** Calibration is no longer bypassed: NND, IID, hull area,
> thigmotaxis zones and heatmap axes all follow `calibration.scale_factor`,
> so calibrating in cm finally makes group metrics comparable across
> recordings of different-sized fish. Shoaling columns lost their hardcoded
> `_BL` suffixes and gained a `Unit` column. Every row now carries a `Status`
> column, and a fish excluded by the quality gate appears in the CSV with the
> reason rather than vanishing. Bout turns that cannot be measured are counted
> as `Bout_N_TurnUnmeasurable` instead of being reported as straight.

**Bug fixes:**
- Thigmotaxis border percentage now correctly handles missing fish data
- Arena misalignment warning when >5% of positions fall outside the defined boundary
- NND calculation optimized with scipy cdist
- Minimum valid data threshold raised to prevent metrics from near-empty trajectories

**UX improvements:**
- Trajectory trail slider on the frame viewer — see where each fish has been
- CSV export buttons on all analysis tabs
- Progress bar during batch processing
- Clearer error messages with auto-reset to defaults
- Units displayed in comparison tables
- Rewritten help text explaining each analysis in plain language

## 2.0.0

Refactored from monolithic scripts into a modular package with GUI and API layers.
