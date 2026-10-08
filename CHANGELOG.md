# Changelog

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

**Results tab.** Total distance, typical (median) speed, peak (99th
percentile) speed and path straightness as SuperPlots: fish as small dots,
session means as large markers, and a group line at the mean of its sessions,
because the tank is the experimental unit. One table, one export.

**Names, groups and review on the sessions table.** Rename a session, put
several in a group, and open idtracker.ai's validator on one; a session saved
from the validator is ticked as Reviewed and reloaded.

**Position smoothing removed.** The Savitzky-Golay option changed distance and
speed by about 1% and path straightness by about 1%; the metrics it mattered
for were withdrawn in 2.1.0. Plot smoothing on the Individual and Shoaling
tabs is unchanged and does not affect exported numbers.

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
