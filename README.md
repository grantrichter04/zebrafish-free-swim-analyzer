# Zebrafish Free Swim Analyzer

A desktop application for analyzing zebrafish locomotor behavior from free-swimming assays. Built for researchers using [idtracker.ai](https://idtrackerai.readthedocs.io/) to track multiple fish in open-field arenas.

The tool takes raw trajectory data (x, y positions per frame per fish) and produces calibrated behavioral metrics — swimming speed, distance traveled, freezing, path straightness, turning bias, group cohesion (shoaling), and anxiety-related wall-hugging (thigmotaxis) — through an interactive GUI or a scriptable Python API.

Designed for the Morsch lab at Macquarie University to support zebrafish neurobehavioral research.

> **Status:** Under active development. Core analyses are functional; refinements ongoing.

---

## Features

- **Individual behavior metrics** — speed, distance, freezing (count/duration), path straightness, turning bias (laterality index)
- **Shoaling analysis** — nearest neighbor distance (NND), inter-individual distance (IID), convex hull area
- **Spatial analysis** — thigmotaxis (wall-hugging behavior), position heatmaps
- **Calibration** — converts raw pixel coordinates to real-world units (body lengths, cm, etc.)
- **GUI** — interactive tkinter + matplotlib interface with tabs for each analysis type
- **Programmable API** — use individual components directly in your own scripts

---

## Project Structure

```
fish_analyzer/
├── __init__.py          # Package exports and version
├── data_structures.py   # Core data classes (metadata, calibration, loaded files)
├── file_loading.py      # Load .npy trajectory files from idtracker.ai
├── processing.py        # Trajectory processing and individual metrics
├── bout_analysis.py     # Swim bout detection, per-bout metrics, laterality
├── shoaling.py          # Group behavior analysis (NND, IID, convex hull)
├── spatial.py           # Thigmotaxis and heatmap generation
├── export.py            # CSV export utilities for all analysis results
├── video_utils.py       # Optional video frame reading (OpenCV)
└── gui/
    ├── __init__.py      # GUI package and main application class
    ├── base.py          # Shared GUI base class, status bar, log redirect
    ├── data_tab.py      # GUI tab: data loading, calibration, processing
    ├── analysis_tab.py  # GUI tab: individual trajectory metrics
    ├── bout_tab.py      # GUI tab: bout detection and laterality
    ├── shoaling_tab.py  # GUI tab: shoaling metrics and frame viewer
    ├── spatial_tab.py   # GUI tab: thigmotaxis and heatmaps
    ├── inspector_tab.py # GUI tab: frame-by-frame trajectory inspector
    └── utils.py         # Shared GUI utility functions

fish_posture_analyzer.py # Standalone: midline/skeleton extraction from crops
head_detection/          # Standalone: head-vs-tail and turn analysis
```

> `fish_posture_analyzer.py` and `head_detection/` are standalone scripts. They
> are not reachable from the GUI and have their own dependencies — see
> [Installation](#installation).

---

## Installation

### Quick setup with pip (recommended)

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Alternative: conda

```bash
conda create -n fishanalyzer python=3.12 numpy pandas matplotlib scipy shapely scikit-learn opencv -c conda-forge
conda activate fishanalyzer
pip install traja==25.0.1
```

### Requirements

- **Python 3.9 or newer.** Verified end to end on 3.9, 3.12 and 3.13; 3.12 is the
  recommended default. Note that `trajectorytools` and idtracker.ai 6.x both
  require 3.10+, so pick 3.12 if you expect to use them alongside this tool.
- Everything the package needs is in [`requirements.txt`](requirements.txt).
  `scikit-learn` is listed there and is **not** optional — `traja` imports it
  without declaring it, so `import traja` fails if it is missing.
- `shapely` (thigmotaxis) and `opencv-python` (video frame reading) are
  functionally optional: without them those features are disabled rather than
  crashing. They are installed by default because most workflows use them.

The standalone scripts need more:

```bash
pip install scikit-image      # fish_posture_analyzer.py and head_detection/
# head_detection/ additionally needs idtrackerai — install per idtracker.ai's docs
```

---

## Usage

### GUI (recommended)

```bash
python run_analyzer.py
```

### Programmatic API

Run this from the repository root — `fish_analyzer` is not installed as a
package, so it is only importable from there.

```python
from pathlib import Path
from fish_analyzer import (
    TrajectoryFileLoader,
    process_and_analyze_file,
    ShoalingCalculator,
    ShoalingParameters,
)

# Load a trajectory file exported from idtracker.ai
loaded_file = TrajectoryFileLoader.load_file(Path("trajectories.npy"))

# Process trajectories and compute individual metrics
fish_list = process_and_analyze_file(loaded_file)
for fish in fish_list:
    print(f"Fish {fish.fish_id}: {fish.metrics['total_distance']:.1f} BL traveled")

# Compute shoaling metrics — note this takes the loaded file, not the fish list
params = ShoalingParameters()
results = ShoalingCalculator(loaded_file, params).calculate()
print(f"Mean NND: {results.mean_nnd:.2f}")
```

### Verifying against your own recordings

The test suite (`pytest -q`) is entirely synthetic so it runs for anyone who
clones the repo. Two defects were invisible to synthetic input and only showed
up on real recordings, so there is a second check that takes real sessions:

```bash
python scripts/verify_on_session.py path/to/session_folder
```

Point it at one session or at a folder containing several. It re-verifies
every claim in [AUDIT_B_CORRECTNESS.md](AUDIT_B_CORRECTNESS.md) that depends
on real data — the freeze denominators reconciling, net displacement surviving
tracking gaps, top speed staying physiologically plausible, no inter-bout
interval spanning a gap, and group metrics following the calibration. It exits
non-zero if any check fails, and contains no data or paths of its own.

---

## Input Data

This tool expects `.npy` trajectory files in the format exported by **idtracker.ai** — a deep-learning-based system for tracking multiple animals in video. Each file contains per-frame (x, y) coordinates for each tracked individual.

---

## Analysis Methods

| Module | Key Metrics |
|---|---|
| `processing.py` | Speed, distance, freezing, path straightness, turning bias |
| `shoaling.py` | NND, IID, convex hull area (group cohesion) |
| `spatial.py` | Thigmotaxis % (border vs center), position heatmaps |
| `segments.py` | Tracking-gap boundaries, shared by every episode metric |

All distances and areas are in the unit you calibrated in — the `Unit` column
in each export says which. Metrics that count episodes (freezes, bouts) are
computed within stretches of continuous tracking and never across a gap; an
episode cut short by lost tracking is reported as *censored* rather than
counted, and rates are per unit of observed time. See
[AUDIT_B_CORRECTNESS.md](AUDIT_B_CORRECTNESS.md).

### Thigmotaxis
Quantifies anxiety-like wall-hugging behavior. The arena boundary is defined by the user; a configurable inner zone (default 15% inward) separates the border region from the center. High thigmotaxis (more time near walls) typically indicates stress or novelty response.

### Shoaling Metrics
- **NND** (Nearest Neighbor Distance): distance from each fish to its closest neighbor — sensitive to tight schooling
- **IID** (Inter-Individual Distance): mean pairwise distance across all fish pairs — less sensitive to outliers
- **Convex Hull**: area of the polygon enclosing all fish — proxy for group spread

---

## Version

`2.1.0` — analysis overhaul and UX improvements.

### What's New in 2.1

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
> Full detail: [AUDIT_B_CORRECTNESS.md](AUDIT_B_CORRECTNESS.md).
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

`2.0.0` — refactored from monolithic scripts into a modular package with GUI and API layers.

---

## License

[Add license here]
