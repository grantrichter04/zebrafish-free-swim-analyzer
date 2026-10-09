# Changelog

## Unreleased - measures that tracker wobble and dropout cannot move

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
  analyzer, from pinned versions, and adds a desktop shortcut.
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
