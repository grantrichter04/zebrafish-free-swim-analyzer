# Zebrafish Free Swim Analyzer

A desktop app that takes free-swim videos of zebrafish from recording to
results: it tracks the fish with [idtracker.ai](https://idtracker.ai/), then
measures swimming speed, distance, freezing, path straightness, turning bias,
shoaling and thigmotaxis.

Built for the Morsch lab at Macquarie University.

---

## Installing on the lab laptop

This is done once, by whoever looks after the laptop.

1. Install [Miniconda](https://www.anaconda.com/download/success) and a current
   NVIDIA driver, if they are not already there.
2. Download or clone this repository to a folder that will not move, for
   example `C:\FreeSwimAnalyzer`.
3. Double-click **`install.bat`**.

It builds one conda environment, `freeswim`, holding both idtracker.ai and the
analyzer, checks it, and puts a **Free Swim Analyzer** shortcut on the desktop.
It downloads about 5 GB and is safe to run again.

The shortcut and the environment belong to the Windows account that ran
`install.bat`, and the shortcut points into the repository folder, so do not
move or rename that folder afterwards.

---

## Running an experiment

Open **Free Swim Analyzer** from the desktop. A "Starting" window appears
first; the app follows in a few seconds.

### 1. Track the videos (Tracking tab)

1. Put the experiment's videos in one folder, and press **Choose folder...**.
   Each video is listed as *not tracked*, *tracked* or *incomplete*.
2. Make a setup. Select a video, press **Configure new setup...**, and give it
   a name. idtracker.ai opens on that video: set the number of animals, the
   thresholds and the arena, then press **Save setup and close**. The setup is
   saved beside the videos and is used for every video in the folder.
3. Check the setup. The app offers to open it on the next video. Look that
   every fish is detected, then close the window. If you adjust anything,
   press **Save setup and close**; the change applies to all videos.
4. Press **Track all untracked videos**. Videos are tracked one after another.
   The tab shows which one is running and for how long, with idtracker.ai's
   output below. Keep the laptop on and plugged in.
5. When it finishes, press **Load tracked sessions for analysis**.

A video that fails does not stop the others; the summary at the end says which
failed and why. **Stop** ends the run, and the folder can be opened again later
to track whatever is left.

Each tracked video gets a `session_<video name>` folder beside it. Keep these
with the videos: they are the tracking results.

### 2. Analyse (the other tabs)

1. **Sessions & Units** lists every loaded session with its fish count,
   length, tracking quality and body length. Choose the units, then press
   **Run All Analysis**.
   - **Centimetres** is best. Press **Measure on a video frame...**, click the
     two ends of something whose length you know (a ruler, or the inside width
     of the tank), and type that length.
   - **Body lengths** uses one value for the whole experiment, suggested from
     the sessions. It is approximate: idtracker.ai's body length depends on
     lighting and threshold.

   One scale is used for every session, so results can be compared between
   them, and it is written into every export.
   - **Rename...** and **Set group...** give sessions short names and put them
     in experimental groups; these label every plot and export.
   - **Open in validator...** opens idtracker.ai's own tool on the selected
     session, to check that fish were not swapped and correct any that were.
     Save there (Ctrl+S) and the session is ticked as **Reviewed** and
     reloaded with the corrections.
2. **Results** shows the headline measures: total distance, median speed,
   99th percentile speed, path straightness and time near the wall. Small dots are fish,
   large markers are session means. Fish sharing a tank are not independent,
   so the session is the unit to compare, and a group needs several sessions
   to be tested. **Minute by minute** shows each measure on every minute of
   the recording by itself, unsmoothed; **Speed distributions** shows the
   speeds in full; **Where they swim** shows where each session's fish spent
   their time; and **What do these measures mean?** explains each measure.
   - Time near the wall uses the tank outline drawn in idtracker.ai's setup
     window, so nothing needs redrawing. If a session has none, or it is
     wrong, **Tank outline...** on **Sessions & Units** draws one on a video
     frame.
3. **Shoaling** shows how close the fish keep to each other: nearest-neighbour
   distance, inter-individual distance and the area the shoal covers, against
   what fish placed at random in the tank would give, and minute by minute.
4. **Video Inspector** plays the video with the tracking drawn on top, and
   exports frames and clips.

Already have tracked sessions from before? Skip the Tracking tab and press
**Add sessions...** on **Sessions & Units**. It takes one session folder, or a
folder containing several and loads them all.

### If something looks wrong

- Press **Check Setup** at the bottom right. It says whether the analyzer,
  idtracker.ai and the GPU are working.
- Press **Show Log** beside it for the full message history.
- If the app does not open at all, run `install.bat` again.

---

## What is measured

| Analysis | Metrics |
|---|---|
| Individual | Speed, distance, freezing, path straightness, turning bias |
| Shoaling | Nearest neighbour distance (NND), inter-individual distance (IID), convex hull area |
| Position | Time near the walls, where the fish spend their time |

All distances and areas are in the unit you chose; the `Unit` and
`PixelsPerUnit` columns in each export say which and at what scale, so results
can be converted later.

Frames where idtracker.ai lost a fish are treated as *unobserved*, not as the
fish being still. Freezes are counted within stretches of continuous
tracking and never across a gap; one cut short by lost tracking is reported as
*censored* rather than counted, and rates are per unit of observed time.

**Time near the wall** (thigmotaxis) is the share of time spent in a zone
along the walls, a measure of anxiety-like behaviour. The zone is 15% of the
tank's shorter side wide, measured in from the tank outline.

**Shoaling:**

- **NND**: distance from each fish to its closest neighbour.
- **IID**: mean distance over all pairs of fish.
- **Convex hull**: area of the polygon enclosing all fish.

Some turning and burst metrics from version 2.1.0 were withdrawn because they
measured tracking noise rather than behaviour. See
[CHANGELOG.md](CHANGELOG.md) and
[docs/audit/AUDIT_B_CORRECTNESS.md](docs/audit/AUDIT_B_CORRECTNESS.md).

---

## Exporting figures and clips

The Video Inspector draws the tracking on the video: fish positions, numbered
as in the results, the outlines idtracker.ai segmented, trails, and lines to
each fish's nearest neighbour. "More options" adds an outline around the shoal
and lines from one fish to all the others. Scroll on the video to zoom, drag to
move, double-click to see the whole frame again.

- **Save frame (PNG)** writes the current frame with its overlays at full video
  resolution.
- **Export clip** writes a marked range as an MP4, or as a numbered PNG
  sequence for lossless frames. Mark the range with **Set In** / **Set Out**
  beside the frame slider. With neither set, the whole recording is exported,
  and the dialog says so first.

Both export what the tab is showing, at the whole frame whatever the zoom,
including the plot under the video. If the session has not been analysed yet, the export
refuses rather than writing a clip with a "run the analysis first" message in
it.

In an exported clip the time panel covers only the exported range, at one
sample a second.

Expect roughly 28 ms per frame at 1288×964 with a time panel, about 8 seconds
for a 10-second clip.

The source video is found automatically when it sits beside the session folder,
which is where the Tracking tab leaves it. Otherwise **Find video...** appears
under the session name.

---

## For developers

### Analysis only, on any machine

```bash
conda env create -f environment.yml
conda activate freeswim
pip install -e ".[dev]"
pytest -q
```

This skips idtracker.ai and PyTorch. Every analysis tab works; the Tracking tab
says tracking is unavailable.

### Requirements

- **Python 3.10 or newer**; 3.12 is what `install.bat` builds and what CI tests.
- Dependencies are declared once, in [`pyproject.toml`](pyproject.toml).
  [`constraints-win-cu128.txt`](constraints-win-cu128.txt) pins the versions
  the lab laptop install was tested with.
- OpenCV is installed as `opencv-python-headless`, the same build idtracker.ai
  uses. Do not also install `opencv-python` into the same environment.

### Commands

```bash
python -m fish_analyzer           # open the app
python -m fish_analyzer --check   # report whether this machine is set up
pytest -q                         # run the tests (synthetic data, a few seconds)
```

The tests need no real data. A second check takes real sessions, because two
past defects only showed up on real recordings:

```bash
python scripts/verify_on_session.py path/to/session_folder
```

### Using the analysis from Python

```python
from pathlib import Path
from fish_analyzer import (
    TrajectoryFileLoader,
    process_and_analyze_file,
    ShoalingCalculator,
    ShoalingParameters,
)

# Load a session tracked by idtracker.ai
loaded_file = TrajectoryFileLoader.load_from_session_folder(Path("session_fish_A"))

# Process trajectories and compute individual metrics
fish_list = process_and_analyze_file(loaded_file)
for fish in fish_list:
    print(f"Fish {fish.fish_id}: {fish.metrics['total_distance']:.1f} BL traveled")

# Shoaling takes the loaded file, not the fish list
results = ShoalingCalculator(loaded_file, ShoalingParameters()).calculate()
print(f"Mean NND: {results.mean_nnd:.2f}")
```

### Layout

```
fish_analyzer/
├── tracking.py                  # run idtracker.ai on a folder of videos
├── idtrackerai_setup_window.py  # idtracker.ai's window, limited to saving a setup
├── selfcheck.py                 # the Check Setup report
├── file_loading.py              # load idtracker.ai session folders
├── processing.py                # individual metrics
├── segments.py                  # tracking-gap boundaries, shared by episode metrics
├── shoaling.py                  # NND, IID, convex hull
├── spatial.py                   # tank outline and time near the wall
├── export.py                    # CSV exports
├── video_utils.py, overlay_render.py, media_export.py   # video and clip export
└── gui/                         # one file per tab
install.bat                      # one-time laptop setup
launch.pyw                       # what the desktop shortcut runs
docs/audit/                      # the correctness audit and its findings
docs/superpowers/                # design specs and implementation plans
extras/                          # standalone scripts, not part of the app
```

Version history is in [CHANGELOG.md](CHANGELOG.md).

---

## License

[Add license here]
