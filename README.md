# Zebrafish Free Swim Analyzer

A desktop application for analyzing zebrafish locomotor behavior from free-swimming assays. Built for researchers using [idtracker.ai](https://idtrackerai.readthedocs.io/) to track multiple fish in open-field arenas.

The tool takes raw trajectory data (x, y positions per frame per fish) and produces calibrated behavioral metrics — swimming speed, distance traveled, freezing, bursting, erratic movements, path straightness, group cohesion (shoaling), and anxiety-related wall-hugging (thigmotaxis) — through an interactive GUI or a scriptable Python API.

Designed for the Morsch lab at Macquarie University to support zebrafish neurobehavioral research.

> **Status:** Under active development. Core analyses are functional; refinements ongoing.

---

## Features

- **Individual behavior metrics** — speed, distance, freezing (count/duration), bursting (count/speed), angular velocity, erratic movements, path straightness
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
├── shoaling.py          # Group behavior analysis (NND, IID, convex hull)
├── spatial.py           # Thigmotaxis and heatmap generation
├── export.py            # CSV export utilities for all analysis results
├── video_utils.py       # Optional video frame reading (OpenCV)
└── gui/
    ├── __init__.py      # GUI package and main application class
    ├── base.py          # Shared GUI base class, status bar, log redirect
    ├── data_tab.py      # GUI tab: data loading, calibration, processing
    ├── analysis_tab.py  # GUI tab: individual trajectory metrics
    ├── shoaling_tab.py  # GUI tab: shoaling metrics and frame viewer
    ├── spatial_tab.py   # GUI tab: thigmotaxis and heatmaps
    └── utils.py         # Shared GUI utility functions
```

---

## Installation

### Quick setup with conda (recommended)

```bash
# Create a new environment with all dependencies
conda create -n fishanalyzer python=3.10 numpy pandas matplotlib scipy shapely scikit-learn -c conda-forge
conda activate fishanalyzer

# Install remaining packages via pip (not available on conda-forge)
pip install traja opencv-python
```

### Requirements

- Python 3.8+
- Core dependencies:
  ```
  pip install numpy pandas matplotlib scipy traja
  ```
- Optional dependencies:
  ```
  pip install shapely       # Enables thigmotaxis analysis
  pip install opencv-python # Enables video frame reading
  ```

> The application handles missing optional dependencies gracefully — features that require them will be disabled rather than crashing.

---

## Usage

### GUI (recommended)

```bash
python run_analyzer.py
```

### Programmatic API

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

# Compute shoaling metrics
params = ShoalingParameters()
results = ShoalingCalculator.calculate(fish_list, params)
print(f"Mean NND: {results.mean_nnd:.2f}")
```

---

## Input Data

This tool expects `.npy` trajectory files in the format exported by **idtracker.ai** — a deep-learning-based system for tracking multiple animals in video. Each file contains per-frame (x, y) coordinates for each tracked individual.

---

## Analysis Methods

| Module | Key Metrics |
|---|---|
| `processing.py` | Speed, distance, freezing, bursting, angular velocity, erratic movements, path straightness |
| `shoaling.py` | NND, IID, convex hull area (group cohesion) |
| `spatial.py` | Thigmotaxis % (border vs center), position heatmaps |

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
- **Burst analysis** — burst count, peak speed, frequency per minute (locomotor vigor)
- **Angular velocity** — mean turning rate in degrees/second
- **Erratic movements** — count of sudden large direction changes per minute (startle/stress)
- **Path straightness** — sliding-window displacement/distance ratio (0 = circling, 1 = straight)
- **Turning bias** — laterality index, cumulative heading change, signed angular velocity

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
