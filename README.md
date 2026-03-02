# Fish Trajectory Analyzer

A modular Python application for analyzing fish trajectory data exported from [idtracker.ai](https://idtrackerai.readthedocs.io/). Provides both a graphical user interface and a programmable API for quantifying individual and group behavior.

---

## Features

- **Individual behavior metrics** — speed, distance traveled, sinuosity, turn angles, acceleration
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
├── run_analyzer.py      # Entry point — launches the GUI
├── data_structures.py   # Core data classes (metadata, calibration, loaded files)
├── file_loading.py      # Load .npy trajectory files from idtracker.ai
├── processing.py        # Trajectory processing and individual metrics
├── shoaling.py          # Group behavior analysis (NND, IID, convex hull)
├── spatial.py           # Thigmotaxis and heatmap generation
├── video_utils.py       # Optional video frame reading (OpenCV)
├── base.py              # Shared GUI base classes
├── analysis_tab.py      # GUI tab: individual metrics
├── data_tab.py          # GUI tab: data loading and preview
├── shoaling_tab.py      # GUI tab: shoaling metrics
├── spatial_tab.py       # GUI tab: spatial analysis
└── utils.py             # Shared utility functions
```

---

## Installation

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
| `processing.py` | Speed, distance, sinuosity, turn angles, acceleration |
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

`2.0.0` — refactored from monolithic scripts into a modular package with GUI and API layers.

---

## License

[Add license here]
