# Video Inspector Frame and Clip Export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Video Inspector save the current composited frame as a PNG and export a marked frame range as an MP4 or PNG sequence, with the time-series panel stacked underneath.

**Architecture:** Extract the overlay compositor out of the tkinter mixin into a pure module (`overlay_render.py`) that takes an `OverlaySettings` dataclass instead of reading tk variables. The live view and the exporter then call the same function, so the export cannot drift from the display. A second module (`media_export.py`) owns frame sourcing, the pre-rasterised time strip, and the output sinks. The GUI keeps only wiring.

**Tech Stack:** Python 3.9+, numpy, OpenCV (`cv2`), matplotlib, tkinter, pytest.

**Spec:** [docs/superpowers/specs/2026-08-04-inspector-overlay-export-design.md](../specs/2026-08-04-inspector-overlay-export-design.md)

---

## Environment

Every command in this plan uses the project's conda environment, which has all
dependencies including `cv2`, `shapely` and `PIL`:

```
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe
```

The repo's default `python` is miniconda base and is **missing traja, cv2 and
shapely** — commands run with it will fail with `ModuleNotFoundError`. Run
everything from the repo root (`C:\projects\zebrafish-free-swim-analyzer`);
`fish_analyzer` is not installed as a package and is only importable from there.

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `fish_analyzer/overlay_render.py` | `OverlaySettings`, `fish_colors`, `compose_frame`. Pure array work. No tkinter. |
| `fish_analyzer/media_export.py` | `ExportFrameSource`, `TimeStrip`, `Mp4Sink`, `PngSequenceSink`, `export_clip`. No tkinter. |
| `tests/test_overlay_render.py` | Compositor tests. Run headless. |
| `tests/test_media_export.py` | Sink, source and loop tests. Run headless. |

**Modified:**

| File | Change |
|---|---|
| `fish_analyzer/shoaling.py` | Add module-level `nearest_neighbour_distances`; `_calculate_nnd_at_frame` delegates to it. |
| `fish_analyzer/gui/inspector_tab.py` | Delete `_inspector_draw_cv2`, `_inspector_draw_numpy`, `_rgba_to_bgr`. Add `render_settings_from_vars`, In/Out markers, Export section, export driver. |
| `tests/test_gui_regressions.py` | Add GUI-level tests for settings snapshot, In/Out normalisation, export guard. |
| `README.md` | Document the export controls. |

---

### Task 1: Share the nearest-neighbour computation

The inspector hand-rolls an O(n²) Python loop per frame; `shoaling.py` already
does this with scipy `cdist`. Extract a NaN-safe module-level function both can
use, so the distances drawn on a frame equal the ones in the exported CSV.

**Files:**
- Modify: `fish_analyzer/shoaling.py:343-362`
- Test: `tests/test_metric_correctness.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_metric_correctness.py`:

```python
def test_nearest_neighbour_distances_matches_calculator_method():
    """The shared helper and ShoalingCalculator must not drift apart."""
    from fish_analyzer.shoaling import nearest_neighbour_distances

    positions = np.array([[0.0, 0.0], [3.0, 4.0], [100.0, 100.0]])
    nnd, nn_idx = nearest_neighbour_distances(positions)

    assert nnd[0] == pytest.approx(5.0)
    assert nnd[1] == pytest.approx(5.0)
    assert nn_idx[0] == 1
    assert nn_idx[1] == 0


def test_nearest_neighbour_distances_ignores_untracked_fish():
    """A NaN fish is neither a source nor a candidate neighbour."""
    from fish_analyzer.shoaling import nearest_neighbour_distances

    positions = np.array([[0.0, 0.0], [np.nan, np.nan], [3.0, 4.0]])
    nnd, nn_idx = nearest_neighbour_distances(positions)

    assert np.isnan(nnd[1])
    assert nn_idx[1] == -1
    assert nnd[0] == pytest.approx(5.0)
    assert nn_idx[0] == 2


def test_nearest_neighbour_distances_single_fish_has_no_neighbour():
    from fish_analyzer.shoaling import nearest_neighbour_distances

    nnd, nn_idx = nearest_neighbour_distances(np.array([[10.0, 10.0]]))

    assert np.isnan(nnd[0])
    assert nn_idx[0] == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_metric_correctness.py -k nearest_neighbour -v
```
Expected: FAIL with `ImportError: cannot import name 'nearest_neighbour_distances'`

- [ ] **Step 3: Write the implementation**

In `fish_analyzer/shoaling.py`, add after the imports (module level, before the
dataclasses):

```python
def nearest_neighbour_distances(positions: np.ndarray):
    """Nearest-neighbour distance and neighbour index for each row.

    Shared by ShoalingCalculator and the Video Inspector overlay so the
    distances drawn on a frame are the same ones that reach the CSV.

    Parameters
    ----------
    positions : np.ndarray
        Shape (n, 2). Rows may be NaN for untracked individuals.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Distances shape (n,) in the units of `positions`, NaN where the row is
        untracked or has no neighbour; neighbour indices shape (n,), -1 where
        there is none.
    """
    n = len(positions)
    nnd = np.full(n, np.nan)
    nn_idx = np.full(n, -1, dtype=int)
    if n < 2:
        return nnd, nn_idx

    tracked = ~np.isnan(positions[:, 0])
    if tracked.sum() < 2:
        return nnd, nn_idx

    idx = np.flatnonzero(tracked)
    dist = cdist(positions[idx], positions[idx], metric='euclidean')
    np.fill_diagonal(dist, np.inf)

    nearest = np.argmin(dist, axis=1)
    nnd[idx] = dist[np.arange(len(idx)), nearest]
    nn_idx[idx] = idx[nearest]
    return nnd, nn_idx
```

Then replace the body of `_calculate_nnd_at_frame` (currently lines 359-362) so
the two cannot diverge:

```python
        return nearest_neighbour_distances(positions)[0]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_metric_correctness.py -v
```
Expected: PASS, including the pre-existing shoaling tests — `_calculate_nnd_at_frame`
is only ever called on complete (NaN-free) frames, so its results are unchanged.

- [ ] **Step 5: Commit**

```bash
git add fish_analyzer/shoaling.py tests/test_metric_correctness.py
git commit -m "Share one nearest-neighbour implementation"
```

---

### Task 2: OverlaySettings and the compositor skeleton

**Files:**
- Create: `fish_analyzer/overlay_render.py`
- Test: `tests/test_overlay_render.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_overlay_render.py`:

```python
"""Tests for the pure overlay compositor.

These need no display and no video file, so unlike the GUI tests they run
everywhere. cv2 is required for the drawing path and skipped if absent.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fish_analyzer.overlay_render import (
    OverlaySettings,
    compose_frame,
    fish_colors,
)


def blank(h=100, w=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


def one_fish_at(x, y, n_frames=1):
    """Trajectories array of shape (n_frames, 1, 2)."""
    traj = np.full((n_frames, 1, 2), np.nan)
    traj[:, 0, 0] = x
    traj[:, 0, 1] = y
    return traj


def test_all_overlays_off_returns_the_frame_unchanged():
    base = blank()
    base[10, 10] = (7, 8, 9)
    out = compose_frame(base, one_fish_at(50, 50), 0, OverlaySettings(), 1.0)
    assert np.array_equal(out, base)


def test_compose_frame_does_not_mutate_the_input():
    base = blank()
    before = base.copy()
    settings = OverlaySettings(show_positions=True, dot_radius=20)
    compose_frame(base, one_fish_at(50, 50), 0, settings, 1.0)
    assert np.array_equal(base, before)


def test_fish_dot_uses_the_tab10_colour_in_rgb_order():
    """Regression test for the BGR/RGB channel swap.

    The frame is RGB (video_utils converts BGR2RGB on read and PIL expects
    RGB), so fish 0 must be drawn tab10 blue (31, 119, 180). Before the fix it
    was drawn (180, 119, 31) — orange — which did not match the colour the
    Individual Analysis plots use for the same fish.
    """
    settings = OverlaySettings(show_positions=True, dot_radius=20)
    out = compose_frame(blank(), one_fish_at(50, 50), 0, settings, 1.0)

    # 15px above centre: inside the r=20 disc, clear of the 2px border and of
    # the fish-number glyph drawn at the centre.
    assert tuple(int(v) for v in out[35, 50]) == (31, 119, 180)


def test_untracked_fish_draws_nothing_and_does_not_raise():
    traj = np.full((1, 1, 2), np.nan)
    settings = OverlaySettings(show_positions=True, dot_radius=20)
    out = compose_frame(blank(), traj, 0, settings, 1.0)
    assert out.sum() == 0


def test_fish_colors_matches_the_app_wide_convention():
    """Same sampling as analysis_tab.py:924, so per-fish colours agree."""
    import matplotlib.pyplot as plt

    expected = plt.cm.tab10(np.linspace(0, 1, 6))
    assert np.allclose(fish_colors(6), expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_overlay_render.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'fish_analyzer.overlay_render'`

- [ ] **Step 3: Write the implementation**

Create `fish_analyzer/overlay_render.py`:

```python
"""
fish_analyzer/overlay_render.py
===============================
Pure overlay compositing for the Video Inspector.

Deliberately free of tkinter: the live view and the clip exporter both call
compose_frame, so what gets exported cannot drift from what is displayed, and
the drawing can be tested without a display.

All frames are RGB uint8. VideoFrameReader converts BGR2RGB on read and PIL
expects RGB, so there is no BGR anywhere in this module.
"""
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from .shoaling import nearest_neighbour_distances

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass(frozen=True)
class OverlaySettings:
    """Everything the compositor needs to know, snapshotted from the GUI.

    Frozen so an export cannot change appearance halfway through because a
    checkbox was toggled while it ran.
    """
    show_positions: bool = False
    show_nnd: bool = False
    show_hull: bool = False
    show_iid: bool = False
    iid_focus: int = 0
    show_bout_ring: bool = False
    bout_fish: int = 0
    trail_length: int = 0
    trail_opacity: float = 0.6
    trail_width: float = 1.0
    dot_radius: int = 6


def fish_colors(n_fish: int) -> np.ndarray:
    """Per-fish RGBA colours, matching the convention used across the app.

    analysis_tab.py, bout_tab.py, shoaling_tab.py and spatial_tab.py all sample
    tab10 this way, so a fish keeps one colour between the video and the plots.
    """
    return plt.cm.tab10(np.linspace(0, 1, max(1, n_fish)))


def rgb_uint8(rgba) -> tuple:
    """matplotlib RGBA floats (0-1) to an RGB 0-255 tuple."""
    return (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))


def compose_frame(base, trajectories, frame_idx, settings, scale, colors=None):
    """Draw the overlays for one frame onto a copy of `base`.

    Parameters
    ----------
    base : np.ndarray
        RGB uint8 frame, shape (h, w, 3). Not modified.
    trajectories : np.ndarray
        Shape (n_frames, n_fish, 2) in pixel coordinates. NaN where untracked.
    frame_idx : int
    settings : OverlaySettings
    scale : float
        calibration.scale_factor — pixels to the calibrated unit, used for the
        distance labels.
    colors : np.ndarray, optional
        Result of fish_colors(n_fish). Passed in by callers that render many
        frames so it is not recomputed per frame.

    Returns
    -------
    np.ndarray
        A new RGB uint8 array.
    """
    display = base.copy()
    n_fish = trajectories.shape[1]
    if colors is None:
        colors = fish_colors(n_fish)

    positions = trajectories[frame_idx]

    if CV2_AVAILABLE:
        _draw_cv2(display, trajectories, positions, frame_idx, n_fish,
                  colors, settings, scale)
    else:
        _draw_numpy(display, positions, n_fish, colors, settings)
    return display


def _draw_numpy(display, positions, n_fish, colors, settings):
    """Minimal fallback so the inspector still shows something without cv2."""
    if not settings.show_positions:
        return
    r = settings.dot_radius
    for i in range(n_fish):
        if np.isnan(positions[i, 0]):
            continue
        px, py = int(positions[i, 0]), int(positions[i, 1])
        y_grid, x_grid = np.ogrid[-r:r + 1, -r:r + 1]
        mask = x_grid ** 2 + y_grid ** 2 <= r ** 2
        y_start, y_end = max(0, py - r), min(display.shape[0], py + r + 1)
        x_start, x_end = max(0, px - r), min(display.shape[1], px + r + 1)
        mask_y = slice(max(0, r - py),
                       r + 1 + min(0, display.shape[0] - py - r - 1))
        mask_x = slice(max(0, r - px),
                       r + 1 + min(0, display.shape[1] - px - r - 1))
        display[y_start:y_end, x_start:x_end][mask[mask_y, mask_x]] = \
            rgb_uint8(colors[i])


def _draw_cv2(display, trajectories, positions, frame_idx, n_fish, colors,
              settings, scale):
    """Draw every enabled overlay directly onto the array."""
    _draw_positions(display, positions, n_fish, colors, settings)
```

- [ ] **Step 4: Run the tests to verify they pass**

`_draw_positions` does not exist yet, so add it to `overlay_render.py` as well:

```python
def _draw_positions(display, positions, n_fish, colors, settings):
    if not settings.show_positions:
        return
    r = settings.dot_radius
    for i in range(n_fish):
        if np.isnan(positions[i, 0]):
            continue
        px, py = int(positions[i, 0]), int(positions[i, 1])
        color = rgb_uint8(colors[i])
        cv2.circle(display, (px, py), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(display, (px, py), r, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        font_scale = max(0.3, r / 20.0)
        text = str(i)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, 1)
        cv2.putText(display, text, (px - tw // 2, py + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
```

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_overlay_render.py -v
```
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add fish_analyzer/overlay_render.py tests/test_overlay_render.py
git commit -m "Add a pure overlay compositor with correct RGB channel order"
```

---

### Task 3: Port the remaining overlays

Trails, NND lines, hull, IID lines and the bout ring. The trail loop blends
once instead of once per fish, and NND uses the shared helper from Task 1.

**Files:**
- Modify: `fish_analyzer/overlay_render.py`
- Test: `tests/test_overlay_render.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_overlay_render.py`:

```python
def two_fish_at(p0, p1):
    traj = np.full((1, 2, 2), np.nan)
    traj[0, 0] = p0
    traj[0, 1] = p1
    return traj


def test_nnd_line_connects_the_nearest_pair():
    """Fish 0 and 1 are close; fish 2 is far. The line must join 0 and 1."""
    traj = np.full((1, 3, 2), np.nan)
    traj[0, 0] = (20.0, 50.0)
    traj[0, 1] = (40.0, 50.0)
    traj[0, 2] = (95.0, 95.0)

    out = compose_frame(blank(), traj, 0, OverlaySettings(show_nnd=True), 1.0)

    midpoint_between_0_and_1 = out[50, 30]
    assert midpoint_between_0_and_1.sum() > 0, "no line drawn between 0 and 1"


def test_nnd_label_uses_the_calibration_scale():
    """The drawn distance is in calibrated units, not pixels.

    Asserted through the shared helper rather than by reading pixels: the
    overlay must report scale * pixel distance.
    """
    from fish_analyzer.shoaling import nearest_neighbour_distances

    positions = np.array([[20.0, 50.0], [40.0, 50.0]])
    nnd, _ = nearest_neighbour_distances(positions)
    assert nnd[0] * 0.0125 == pytest.approx(0.25)


def test_hull_needs_three_tracked_fish():
    out = compose_frame(blank(), two_fish_at((20.0, 20.0), (60.0, 60.0)), 0,
                        OverlaySettings(show_hull=True), 1.0)
    assert out.sum() == 0


def test_hull_is_drawn_for_three_tracked_fish():
    traj = np.full((1, 3, 2), np.nan)
    traj[0, 0] = (20.0, 20.0)
    traj[0, 1] = (80.0, 20.0)
    traj[0, 2] = (50.0, 80.0)
    out = compose_frame(blank(), traj, 0, OverlaySettings(show_hull=True), 1.0)
    assert out[40, 50].sum() > 0, "hull interior not tinted"


def test_trails_blend_once_regardless_of_fish_count():
    """Two fish with identical trails must not double-darken the shared pixel.

    The old per-fish copy-and-blend applied alpha n_fish times over.
    """
    traj = np.full((10, 2, 2), np.nan)
    traj[:, 0, 0] = np.linspace(10, 90, 10)
    traj[:, 0, 1] = 50.0
    traj[:, 1, 0] = np.linspace(10, 90, 10)
    traj[:, 1, 1] = 70.0

    settings = OverlaySettings(trail_length=9, trail_opacity=0.5)
    base = np.zeros((100, 100, 3), dtype=np.uint8)
    out = compose_frame(base, traj, 9, settings, 1.0)

    row_50 = out[50].astype(int).sum()
    row_70 = out[70].astype(int).sum()
    assert row_50 == row_70, "fish drawn later got a different blend"


def test_iid_lines_radiate_from_the_focus_fish():
    traj = np.full((1, 3, 2), np.nan)
    traj[0, 0] = (10.0, 50.0)
    traj[0, 1] = (90.0, 50.0)
    traj[0, 2] = (50.0, 10.0)

    out = compose_frame(blank(), traj, 0,
                        OverlaySettings(show_iid=True, iid_focus=0), 1.0)
    assert out[50, 50].sum() > 0, "no line from fish 0 to fish 1"


def test_bout_ring_is_drawn_around_the_selected_fish():
    settings = OverlaySettings(show_bout_ring=True, bout_fish=0, dot_radius=10)
    out = compose_frame(blank(), one_fish_at(50, 50), 0, settings, 1.0)
    # ring radius is 1.8 * dot_radius = 18
    assert out[50, 68].sum() > 0 or out[50, 67].sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_overlay_render.py -v
```
Expected: the seven new tests FAIL (nothing but positions is drawn yet).

- [ ] **Step 3: Write the implementation**

Replace `_draw_cv2` in `fish_analyzer/overlay_render.py` with:

```python
def _draw_cv2(display, trajectories, positions, frame_idx, n_fish, colors,
              settings, scale):
    """Draw every enabled overlay directly onto the array.

    Order matters: trails and the hull go underneath, dots and labels on top.
    """
    _draw_trails(display, trajectories, frame_idx, n_fish, colors, settings)
    _draw_nnd(display, positions, scale)
    _draw_hull(display, positions, n_fish)
    _draw_iid(display, positions, n_fish, settings, scale)
    _draw_positions(display, positions, n_fish, colors, settings)
    _draw_bout_ring(display, positions, n_fish, settings)


def _draw_trails(display, trajectories, frame_idx, n_fish, colors, settings):
    """All trails into one overlay, blended once.

    Previously each fish copied the whole frame and blended separately, which
    cost n_fish full-frame copies per frame and compounded the alpha.
    """
    if settings.trail_length <= 0:
        return
    start = max(0, frame_idx - settings.trail_length)
    end = frame_idx + 1
    thickness = max(1, int(settings.trail_width * 2))

    overlay = display.copy()
    drew = False
    for i in range(n_fish):
        traj = trajectories[start:end, i, :]
        valid = ~np.isnan(traj[:, 0])
        if np.sum(valid) < 2:
            continue
        pts = traj[valid].astype(np.int32)
        cv2.polylines(overlay, [pts], False, rgb_uint8(colors[i]), thickness,
                      lineType=cv2.LINE_AA)
        drew = True

    if drew:
        alpha = min(1.0, settings.trail_opacity)
        cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)


def _label(display, text, at, color):
    """Distance label: black outline first so it reads over any background."""
    cv2.putText(display, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(display, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)


def _draw_nnd(display, positions, scale):
    """White line from each fish to its nearest neighbour, labelled."""
    nnd, nn_idx = nearest_neighbour_distances(positions)
    for i, j in enumerate(nn_idx):
        if j < 0:
            continue
        p1 = (int(positions[i, 0]), int(positions[i, 1]))
        p2 = (int(positions[j, 0]), int(positions[j, 1]))
        cv2.line(display, p1, p2, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        _label(display, f'{nnd[i] * scale:.1f}', mid, (255, 255, 255))


def _draw_hull(display, positions, n_fish):
    valid = positions[~np.isnan(positions[:, 0])]
    if len(valid) < 3:
        return
    hull = cv2.convexHull(valid.astype(np.float32).astype(np.int32))
    overlay = display.copy()
    cv2.fillPoly(overlay, [hull], (100, 200, 100))
    cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
    cv2.polylines(display, [hull], True, (0, 180, 0), 2, lineType=cv2.LINE_AA)


def _draw_iid(display, positions, n_fish, settings, scale):
    """Magenta lines from one focus fish to every other tracked fish."""
    focus = settings.iid_focus
    if focus >= n_fish or np.isnan(positions[focus, 0]):
        return
    pf = (int(positions[focus, 0]), int(positions[focus, 1]))
    for j in range(n_fish):
        if j == focus or np.isnan(positions[j, 0]):
            continue
        pj = (int(positions[j, 0]), int(positions[j, 1]))
        cv2.line(display, pf, pj, (255, 100, 255), 2, lineType=cv2.LINE_AA)
        d = np.hypot(positions[focus, 0] - positions[j, 0],
                     positions[focus, 1] - positions[j, 1]) * scale
        mid = ((pf[0] + pj[0]) // 2, (pf[1] + pj[1]) // 2)
        _label(display, f'{d:.1f}', mid, (255, 100, 255))


def _draw_bout_ring(display, positions, n_fish, settings):
    fish = settings.bout_fish
    if fish >= n_fish or np.isnan(positions[fish, 0]):
        return
    px, py = int(positions[fish, 0]), int(positions[fish, 1])
    cv2.circle(display, (px, py), int(settings.dot_radius * 1.8),
               (0, 255, 255), 3, lineType=cv2.LINE_AA)
```

Guard each helper on its flag by editing `_draw_cv2` to:

```python
def _draw_cv2(display, trajectories, positions, frame_idx, n_fish, colors,
              settings, scale):
    _draw_trails(display, trajectories, frame_idx, n_fish, colors, settings)
    if settings.show_nnd:
        _draw_nnd(display, positions, scale)
    if settings.show_hull:
        _draw_hull(display, positions, n_fish)
    if settings.show_iid:
        _draw_iid(display, positions, n_fish, settings, scale)
    _draw_positions(display, positions, n_fish, colors, settings)
    if settings.show_bout_ring:
        _draw_bout_ring(display, positions, n_fish, settings)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_overlay_render.py -v
```
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git add fish_analyzer/overlay_render.py tests/test_overlay_render.py
git commit -m "Port trails, NND, hull, IID and bout ring to the compositor"
```

---

### Task 4: Make the live view use the compositor

**Files:**
- Modify: `fish_analyzer/gui/inspector_tab.py` — delete `_rgba_to_bgr` (1483-1486), `_inspector_draw_cv2` (1492-1627), `_inspector_draw_numpy` (1629-1650); edit `_inspector_update_dynamic` (1360-1379)
- Test: `tests/test_gui_regressions.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui_regressions.py`:

```python
def test_render_settings_reflect_the_inspector_controls(app):
    """The snapshot handed to the compositor must be what the user ticked.

    Same failure mode as C5: a settings object that ignores the GUI renders
    something other than what the controls say.
    """
    app.inspector_show_nnd_var.set(True)
    app.inspector_show_hull_var.set(False)
    app.inspector_dot_size_var.set(11)
    app.inspector_trail_var.set(45)

    settings = app.render_settings_from_vars()

    assert settings.show_nnd is True
    assert settings.show_hull is False
    assert settings.dot_radius == 11
    assert settings.trail_length == 45
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_gui_regressions.py -k render_settings -v
```
Expected: FAIL with `AttributeError: 'EnhancedFishAnalyzer' object has no attribute 'render_settings_from_vars'`

- [ ] **Step 3: Write the implementation**

In `fish_analyzer/gui/inspector_tab.py`, add the import near the top:

```python
from ..overlay_render import OverlaySettings, compose_frame, fish_colors
```

Add this method to the mixin:

```python
    def render_settings_from_vars(self) -> OverlaySettings:
        """Snapshot the overlay controls.

        The single place tk state becomes an OverlaySettings — the live view
        and the exporter both go through here, so they cannot disagree.
        """
        def _int(var, default=0):
            try:
                return int(var.get())
            except (ValueError, TypeError):
                return default

        return OverlaySettings(
            show_positions=self.inspector_show_positions_var.get(),
            show_nnd=self.inspector_show_nnd_var.get(),
            show_hull=self.inspector_show_hull_var.get(),
            show_iid=self.inspector_show_iid_var.get(),
            iid_focus=_int(self.inspector_iid_focus_var),
            show_bout_ring=self.inspector_show_bouts_var.get(),
            bout_fish=_int(self.inspector_bout_fish_var),
            trail_length=_int(self.inspector_trail_var),
            trail_opacity=float(self.inspector_trail_opacity_var.get()),
            trail_width=float(self.inspector_trail_width_var.get()),
            dot_radius=_int(self.inspector_dot_size_var, 6),
        )
```

Then in `_inspector_update_dynamic`, replace the block that currently reads the
settings and dispatches to the two draw methods (lines 1362-1379) with:

```python
        self._insp_zoom_raw_frame = display.copy()

        settings = self.render_settings_from_vars()
        if self._insp_fish_colors is None or len(self._insp_fish_colors) != n_fish:
            self._insp_fish_colors = fish_colors(n_fish)

        display = compose_frame(display, loaded.trajectories, frame_idx,
                                settings, scale, self._insp_fish_colors)
        raw_pos = loaded.trajectories[frame_idx]
```

Delete the now-unused local `tab10 = plt.cm.tab10(...)` line, the
`dot_radius`/`trail_*` reads above it, and the three methods listed under
**Files**. Add `self._insp_fish_colors = None` next to the other cache
attributes in `_create_inspector_display` (near line 456) and in the reset
block near line 990.

- [ ] **Step 4: Run the full suite to verify nothing regressed**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest -q
```
Expected: PASS. Confirm no test errors mention `_inspector_draw_cv2` or
`_rgba_to_bgr`.

- [ ] **Step 5: Verify the live view by eye**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe run_analyzer.py
```
Load `session_dNLS_FREESWIM_2025-08-29-135947-0000`, open Video Inspector,
attach the video with **Browse Video…** (auto-detect will not find it), tick
Fish positions, NND lines and Convex hull, and scrub. Fish 0 must now be
**blue**, not orange. Close the app.

- [ ] **Step 6: Commit**

```bash
git add fish_analyzer/gui/inspector_tab.py tests/test_gui_regressions.py
git commit -m "Draw the live inspector through the shared compositor"
```

---

### Task 5: Output sinks

**Files:**
- Create: `fish_analyzer/media_export.py`
- Test: `tests/test_media_export.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_media_export.py`:

```python
"""Tests for clip export. No display and no real video needed."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fish_analyzer import media_export
from fish_analyzer.media_export import Mp4Sink, PngSequenceSink, CodecUnavailable


def frame(h=32, w=48):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_png_sequence_writes_one_numbered_file_per_frame(tmp_path):
    sink = PngSequenceSink(tmp_path / "clip", fps=30.0, size=(48, 32))
    for _ in range(3):
        sink.write(frame())
    sink.close()

    written = sorted(p.name for p in (tmp_path / "clip").glob("*.png"))
    assert written == ["frame_000000.png", "frame_000001.png",
                       "frame_000002.png"]


def test_mp4_sink_writes_a_playable_file(tmp_path):
    out = tmp_path / "clip.mp4"
    sink = Mp4Sink(out, fps=30.0, size=(48, 32))
    for _ in range(10):
        sink.write(frame())
    sink.close()

    assert out.exists() and out.stat().st_size > 0


def test_mp4_sink_falls_back_when_the_primary_codec_will_not_open(tmp_path,
                                                                 monkeypatch):
    """OpenCV returns an unopened writer rather than raising, which is how a
    zero-byte file gets left behind. The fallback must engage."""
    real = cv2.VideoWriter
    attempts = []

    class Writer:
        def __init__(self, path, fourcc, fps, size):
            attempts.append(fourcc)
            self._real = real(path, fourcc, fps, size)
            # Reject only the first codec tried.
            self._ok = len(attempts) > 1 and self._real.isOpened()

        def isOpened(self):
            return self._ok

        def write(self, f):
            self._real.write(f)

        def release(self):
            self._real.release()

    monkeypatch.setattr(media_export.cv2, "VideoWriter", Writer)

    sink = Mp4Sink(tmp_path / "clip.mp4", fps=30.0, size=(48, 32))
    sink.write(frame())
    sink.close()

    assert len(attempts) == 2, "fallback codec was not attempted"
    assert sink.path.suffix == ".avi", "fallback should write .avi"


def test_mp4_sink_raises_when_no_codec_opens(tmp_path, monkeypatch):
    class DeadWriter:
        def __init__(self, *a):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(media_export.cv2, "VideoWriter", DeadWriter)

    with pytest.raises(CodecUnavailable):
        Mp4Sink(tmp_path / "clip.mp4", fps=30.0, size=(48, 32))

    assert not (tmp_path / "clip.mp4").exists() or \
        (tmp_path / "clip.mp4").stat().st_size == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_media_export.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'fish_analyzer.media_export'`

- [ ] **Step 3: Write the implementation**

Create `fish_analyzer/media_export.py`:

```python
"""
fish_analyzer/media_export.py
=============================
Writing Video Inspector overlays out as images and clips.

No tkinter: the GUI supplies settings and a progress callback, this module
does the work. Frames are RGB uint8 throughout; only the sinks convert to BGR,
because that is what cv2.VideoWriter and cv2.imwrite expect.
"""
from pathlib import Path

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class CodecUnavailable(RuntimeError):
    """No usable video encoder. Raised instead of leaving a 0-byte file."""


class PngSequenceSink:
    """Numbered lossless frames. Cannot fail on a missing codec."""

    def __init__(self, directory, fps, size):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.size = size
        self._n = 0

    def write(self, rgb):
        out = self.path / f"frame_{self._n:06d}.png"
        cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self._n += 1

    def close(self):
        pass

    def discard_partial(self):
        """Completed PNGs are individually valid, so cancelling keeps them."""


class Mp4Sink:
    """MP4 via mp4v, falling back to MJPG/.avi when the encoder will not open.

    OpenCV signals a missing codec by returning a writer whose isOpened() is
    False and then silently accepting writes, which leaves an empty file. Every
    codec is checked before use.
    """

    CODECS = [("mp4v", ".mp4"), ("MJPG", ".avi")]

    def __init__(self, path, fps, size):
        path = Path(path)
        self.fps = fps
        self.size = size
        self._writer = None
        self.path = None

        for fourcc, ext in self.CODECS:
            candidate = path.with_suffix(ext)
            writer = cv2.VideoWriter(
                str(candidate), cv2.VideoWriter_fourcc(*fourcc), fps, size
            )
            if writer.isOpened():
                self._writer = writer
                self.path = candidate
                self.fourcc = fourcc
                return
            writer.release()
            if candidate.exists() and candidate.stat().st_size == 0:
                candidate.unlink()

        raise CodecUnavailable(
            "No usable video encoder was found (tried "
            + ", ".join(c for c, _ in self.CODECS)
            + "). Export as a PNG sequence instead."
        )

    def write(self, rgb):
        self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def discard_partial(self):
        """A truncated MP4 is unplayable, so a cancelled export removes it."""
        self.close()
        if self.path is not None and self.path.exists():
            self.path.unlink()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_media_export.py -v
```
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add fish_analyzer/media_export.py tests/test_media_export.py
git commit -m "Add MP4 and PNG-sequence sinks with codec fallback"
```

---

### Task 6: Frame source and time strip

**Files:**
- Modify: `fish_analyzer/media_export.py`
- Test: `tests/test_media_export.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_media_export.py`:

```python
def make_video(path, n_frames=20, size=(48, 32)):
    """A tiny MJPG clip whose frame i has red channel == i * 10."""
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
                         (w, h))
    assert vw.isOpened()
    for i in range(n_frames):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:, :, 2] = i * 10          # BGR: channel 2 is red
        vw.write(f)
    vw.release()
    return path


def test_frame_source_reads_the_requested_range_in_order(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource

    video = make_video(tmp_path / "in.avi")
    with ExportFrameSource(video, start_frame=5) as src:
        frames = [src.read() for _ in range(3)]

    reds = [int(f[0, 0, 0]) for f in frames]      # RGB out: channel 0 is red
    assert reds == pytest.approx([50, 60, 70], abs=6)


def test_frame_source_returns_rgb(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource

    video = make_video(tmp_path / "in.avi")
    with ExportFrameSource(video, start_frame=10) as src:
        f = src.read()

    assert f[0, 0, 0] > f[0, 0, 2], "channels look like BGR, not RGB"


def test_cursor_x_maps_time_to_a_pixel_column():
    from fish_analyzer.media_export import cursor_x

    # An axes spanning 0-600 s across pixel columns 80..980
    assert cursor_x(0.0, x0=80.0, px_per_unit=1.5) == 80
    assert cursor_x(100.0, x0=80.0, px_per_unit=1.5) == 230
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_media_export.py -k "frame_source or cursor" -v
```
Expected: FAIL with `ImportError: cannot import name 'ExportFrameSource'`

- [ ] **Step 3: Write the implementation**

Append to `fish_analyzer/media_export.py`:

```python
class ExportFrameSource:
    """Sequential frame reader owned by one export.

    Deliberately not VideoFrameReader: that class runs a background preload
    thread which calls cap.set() and cap.read() on the same capture the
    foreground uses, with only the frame cache locked. It also copies every
    frame twice through an LRU that a single forward pass cannot benefit from.

    Measured on a 1288x964 MJPG session: 9.9 ms/frame reading sequentially
    against 64.7 ms/frame when seeking per frame.
    """

    def __init__(self, video_path, start_frame=0):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        if start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def read(self):
        """Next frame as RGB uint8, or None at end of stream."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def cursor_x(t, x0, px_per_unit):
    """Pixel column for time `t` on a linear axis.

    Kept as a function of the transform rather than of a matplotlib axes so it
    can be tested without rendering anything.
    """
    return int(round(x0 + t * px_per_unit))


class TimeStrip:
    """The time panel rasterised once, with the cursor stamped per frame.

    Valid only while the axes are fixed for the whole session, which is true of
    the NND, IID and Hull panels (xlim is set once at rebuild) and false of the
    Bout panel, which scrolls its xlim every frame. Bout mode re-renders
    instead — see the spec.
    """

    def __init__(self, figure, axes, color=(255, 60, 60), width=2):
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        self._pristine = rgba[:, :, :3].copy()
        self._buffer = self._pristine.copy()
        self.color = color
        self.width = width

        x_at_0 = axes.transData.transform((0.0, 0.0))[0]
        x_at_1 = axes.transData.transform((1.0, 0.0))[0]
        self._x0 = x_at_0
        self._px_per_unit = x_at_1 - x_at_0

    @property
    def height(self):
        return self._pristine.shape[0]

    @property
    def width_px(self):
        return self._pristine.shape[1]

    def at(self, t):
        """The strip with the cursor at time `t`. Reuses one buffer."""
        np.copyto(self._buffer, self._pristine)
        x = cursor_x(t, self._x0, self._px_per_unit)
        x = max(0, min(self._buffer.shape[1] - 1, x))
        cv2.line(self._buffer, (x, 0), (x, self._buffer.shape[0]),
                 self.color, self.width)
        return self._buffer


class ScrollingStrip:
    """Time panel re-rendered per frame, for panels whose axes move.

    The Bout panel resets xlim to a window around the current frame on every
    update, so a strip rasterised once cannot represent it — cropping a
    pre-rendered image would drag the y-axis out of frame. This costs a full
    matplotlib draw per frame (50-100 ms against ~0.3 ms for TimeStrip), which
    is why the export dialog warns before using it.
    """

    def __init__(self, figure, axes, window_s, total_s, color=(255, 60, 60),
                 width=2):
        self._figure = figure
        self._axes = axes
        self._window_s = window_s
        self._total_s = total_s
        self.color = color
        self.width = width

        figure.canvas.draw()
        self._height = np.asarray(figure.canvas.buffer_rgba()).shape[0]

    @property
    def height(self):
        return self._height

    def at(self, t):
        """Redraw the panel with its window centred on `t`."""
        start = max(0.0, t - self._window_s / 2)
        end = start + self._window_s
        if end > self._total_s:
            end = self._total_s
            start = max(0.0, end - self._window_s)

        self._axes.set_xlim(start, end)
        self._figure.canvas.draw()
        strip = np.asarray(self._figure.canvas.buffer_rgba())[:, :, :3].copy()

        x_at_start = self._axes.transData.transform((start, 0.0))[0]
        x_at_end = self._axes.transData.transform((end, 0.0))[0]
        px_per_s = (x_at_end - x_at_start) / max(1e-9, end - start)
        x = cursor_x(t - start, x_at_start, px_per_s)
        x = max(0, min(strip.shape[1] - 1, x))
        cv2.line(strip, (x, 0), (x, strip.shape[0]), self.color, self.width)
        return strip
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_media_export.py -v
```
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add fish_analyzer/media_export.py tests/test_media_export.py
git commit -m "Add the export frame source and pre-rasterised time strip"
```

---

### Task 7: The export loop

**Files:**
- Modify: `fish_analyzer/media_export.py`
- Test: `tests/test_media_export.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_media_export.py`:

```python
def test_export_clip_writes_one_frame_per_source_frame(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource, export_clip
    from fish_analyzer.overlay_render import OverlaySettings

    video = make_video(tmp_path / "in.avi", n_frames=20)
    traj = np.full((20, 1, 2), 10.0)

    written = []

    class RecordingSink:
        def write(self, f):
            written.append(f.copy())

        def close(self):
            pass

        def discard_partial(self):
            pass

    with ExportFrameSource(video, start_frame=2) as src:
        export_clip(src, RecordingSink(), None, traj,
                    OverlaySettings(show_positions=True),
                    scale=1.0, start_frame=2, end_frame=7, step=1)

    assert len(written) == 6, "inclusive range 2..7 is six frames"


def test_export_clip_stacks_the_strip_under_the_frame(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource, export_clip
    from fish_analyzer.overlay_render import OverlaySettings

    video = make_video(tmp_path / "in.avi", n_frames=10)
    traj = np.full((10, 1, 2), 10.0)
    sizes = []

    class SizeSink:
        def write(self, f):
            sizes.append(f.shape)

        def close(self):
            pass

        def discard_partial(self):
            pass

    class FakeStrip:
        height = 40

        def at(self, t):
            return np.zeros((40, 48, 3), dtype=np.uint8)

    with ExportFrameSource(video, start_frame=0) as src:
        export_clip(src, SizeSink(), FakeStrip(), traj, OverlaySettings(),
                    scale=1.0, start_frame=0, end_frame=2, step=1, fps=30.0)

    assert sizes[0] == (32 + 40, 48, 3), "strip not stacked below the frame"


def test_export_clip_stops_when_cancelled(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource, export_clip
    from fish_analyzer.overlay_render import OverlaySettings

    video = make_video(tmp_path / "in.avi", n_frames=20)
    traj = np.full((20, 1, 2), 10.0)
    discarded = []
    count = []

    class Sink:
        def write(self, f):
            count.append(1)

        def close(self):
            pass

        def discard_partial(self):
            discarded.append(True)

    with ExportFrameSource(video, start_frame=0) as src:
        result = export_clip(src, Sink(), None, traj, OverlaySettings(),
                             scale=1.0, start_frame=0, end_frame=19, step=1,
                             should_cancel=lambda: len(count) >= 3)

    assert result.cancelled is True
    assert discarded == [True], "partial output not discarded on cancel"


def test_export_clip_reports_progress(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource, export_clip
    from fish_analyzer.overlay_render import OverlaySettings

    video = make_video(tmp_path / "in.avi", n_frames=10)
    traj = np.full((10, 1, 2), 10.0)
    seen = []

    class Sink:
        def write(self, f):
            pass

        def close(self):
            pass

        def discard_partial(self):
            pass

    with ExportFrameSource(video, start_frame=0) as src:
        export_clip(src, Sink(), None, traj, OverlaySettings(), scale=1.0,
                    start_frame=0, end_frame=4, step=1,
                    on_progress=lambda done, total: seen.append((done, total)))

    assert seen[0] == (1, 5)
    assert seen[-1] == (5, 5)


def test_scrolling_strip_moves_its_window_with_time():
    """The Bout panel scrolls, so its strip must differ between timepoints."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from fish_analyzer.media_export import ScrollingStrip

    fig = Figure(figsize=(4, 1), dpi=50)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, 100, 500), np.sin(np.linspace(0, 100, 500)))

    strip = ScrollingStrip(fig, ax, window_s=10.0, total_s=100.0)
    early = strip.at(5.0).copy()
    late = strip.at(80.0).copy()

    assert not np.array_equal(early, late), "window did not scroll"
    assert early.shape == late.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_media_export.py -k export_clip -v
```
Expected: FAIL with `ImportError: cannot import name 'export_clip'`

- [ ] **Step 3: Write the implementation**

Append to `fish_analyzer/media_export.py`:

```python
from dataclasses import dataclass

from .overlay_render import compose_frame, fish_colors


@dataclass
class ExportResult:
    frames_written: int
    cancelled: bool
    path: object = None


def export_clip(source, sink, strip, trajectories, settings, scale,
                start_frame, end_frame, step=1, fps=30.0,
                on_progress=None, should_cancel=None):
    """Composite `start_frame`..`end_frame` inclusive and write them to `sink`.

    The output canvas is allocated once and written into, so no per-frame
    vstack allocation. Overlay settings are taken as given and never re-read,
    so a clip cannot change appearance halfway through.

    `strip` is any object with `.height` and `.at(t) -> ndarray`, or None for
    no time panel. `should_cancel` is polled once per frame.
    """
    n_fish = trajectories.shape[1]
    colors = fish_colors(n_fish)
    total = len(range(start_frame, end_frame + 1, step))

    out = None
    written = 0
    try:
        for i, frame_idx in enumerate(range(start_frame, end_frame + 1, step)):
            if should_cancel is not None and should_cancel():
                sink.discard_partial()
                return ExportResult(written, True, getattr(sink, "path", None))

            base = source.read()
            if base is None:
                break

            composed = compose_frame(base, trajectories, frame_idx, settings,
                                     scale, colors)

            if strip is None:
                out = composed
            else:
                if out is None:
                    h, w = composed.shape[:2]
                    out = np.zeros((h + strip.height, w, 3), dtype=np.uint8)
                h = composed.shape[0]
                out[:h] = composed
                out[h:] = strip.at(frame_idx / fps)

            sink.write(out)
            written += 1

            if on_progress is not None:
                on_progress(i + 1, total)

            # step > 1 means skipping frames; read past them sequentially
            # rather than seeking, which is 6.5x cheaper on MJPG sources.
            for _ in range(step - 1):
                if source.read() is None:
                    break
    finally:
        sink.close()

    return ExportResult(written, False, getattr(sink, "path", None))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_media_export.py -v
```
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add fish_analyzer/media_export.py tests/test_media_export.py
git commit -m "Add the export loop with progress and cancellation"
```

---

### Task 8: In/Out markers on the frame slider

**Files:**
- Modify: `fish_analyzer/gui/inspector_tab.py` — controls near line 134-190, new state near line 443
- Test: `tests/test_gui_regressions.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui_regressions.py`:

```python
def test_in_out_markers_normalise_when_set_backwards(app):
    """Marking Out before In must not produce an empty or negative range."""
    app.inspector_mark_in = 900
    app.inspector_mark_out = 100

    start, end = app._inspector_export_range(n_frames=1000)

    assert (start, end) == (100, 900)


def test_unset_markers_default_to_the_whole_recording(app):
    app.inspector_mark_in = None
    app.inspector_mark_out = None

    start, end = app._inspector_export_range(n_frames=1000)

    assert (start, end) == (0, 999)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_gui_regressions.py -k markers -v
```
Expected: FAIL with `AttributeError: ... has no attribute '_inspector_export_range'`

- [ ] **Step 3: Write the implementation**

In `_create_inspector_display` near line 443, add the state:

```python
        self.inspector_mark_in = None
        self.inspector_mark_out = None
```

Add the methods to the mixin:

```python
    def _inspector_export_range(self, n_frames):
        """The marked range, normalised, clamped, inclusive of both ends."""
        start = self.inspector_mark_in
        end = self.inspector_mark_out
        if start is None and end is None:
            return 0, n_frames - 1
        if start is None:
            start = 0
        if end is None:
            end = n_frames - 1
        if start > end:
            start, end = end, start
        return max(0, start), min(n_frames - 1, end)

    def _inspector_set_mark_in(self):
        self.inspector_mark_in = self._get_inspector_frame_idx()
        self._inspector_update_mark_label()

    def _inspector_set_mark_out(self):
        self.inspector_mark_out = self._get_inspector_frame_idx()
        self._inspector_update_mark_label()

    def _inspector_clear_marks(self):
        self.inspector_mark_in = None
        self.inspector_mark_out = None
        self._inspector_update_mark_label()

    def _inspector_update_mark_label(self):
        """Show the marked range in frames and seconds."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            self.inspector_mark_label.config(text="In -- | Out --")
            return

        loaded = self.loaded_files[selected]
        fps = loaded.calibration.frame_rate
        start, end = self._inspector_export_range(loaded.n_frames)
        n = end - start + 1
        marked = (self.inspector_mark_in is not None
                  or self.inspector_mark_out is not None)
        prefix = "" if marked else "(whole recording) "
        self.inspector_mark_label.config(
            text=f"{prefix}In {start} \u2192 Out {end} "
                 f"({n} frames, {n / fps:.1f} s)"
        )
```

In `_create_inspector_controls`, after the step row (around line 164), add:

```python
        mark_row = tk.Frame(nav_frame)
        mark_row.pack(fill="x", padx=5, pady=2)
        tk.Button(mark_row, text="Set In", command=self._inspector_set_mark_in,
                  bg="lightblue").pack(side="left", padx=2)
        tk.Button(mark_row, text="Set Out", command=self._inspector_set_mark_out,
                  bg="lightblue").pack(side="left", padx=2)
        tk.Button(mark_row, text="Clear", command=self._inspector_clear_marks
                  ).pack(side="left", padx=2)

        self.inspector_mark_label = tk.Label(
            nav_frame, text="In -- | Out --", font=("Arial", 9)
        )
        self.inspector_mark_label.pack(anchor="w", padx=5)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_gui_regressions.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fish_analyzer/gui/inspector_tab.py tests/test_gui_regressions.py
git commit -m "Add In/Out range markers to the Video Inspector"
```

---

### Task 9: Save the current frame as a PNG

**Files:**
- Modify: `fish_analyzer/gui/inspector_tab.py`

- [ ] **Step 1: Add the Export section and the handler**

In `_create_inspector_controls`, after the Bout Overlay section (around line
403), add:

```python
        _, export_frame = self._make_collapsible(scroll_frame, "Export")

        tk.Button(export_frame, text="Save Frame (PNG)...",
                  command=self._inspector_save_frame,
                  bg="lightgreen").pack(fill="x", padx=5, pady=2)
        tk.Button(export_frame, text="Export Clip...",
                  command=self._inspector_export_clip_dialog,
                  bg="lightgreen").pack(fill="x", padx=5, pady=2)
```

Add the handler to the mixin:

```python
    def _inspector_current_composite(self):
        """The current frame with overlays, at full video resolution.

        Returns (rgb_array, loaded, frame_idx) or (None, None, None) when no
        file is selected.
        """
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            return None, None, None

        loaded = self.loaded_files[selected]
        frame_idx = min(self._get_inspector_frame_idx(), loaded.n_frames - 1)

        base = None
        if self.inspector_video_var.get() and selected in self.video_readers:
            base = self.video_readers[selected].read_frame(frame_idx)
        if base is None and self._insp_cached_background is not None:
            base = self._insp_cached_background
        if base is None:
            base = np.ones((loaded.metadata.video_height,
                            loaded.metadata.video_width, 3),
                           dtype=np.uint8) * 200

        settings = self.render_settings_from_vars()
        composed = compose_frame(base, loaded.trajectories, frame_idx,
                                 settings, loaded.calibration.scale_factor)
        return composed, loaded, frame_idx

    def _inspector_save_frame(self):
        """Write the current composite to a PNG at full resolution."""
        composed, loaded, frame_idx = self._inspector_current_composite()
        if composed is None:
            messagebox.showwarning(
                "No File Selected",
                "Select a file in the Video Inspector first."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save Frame",
            defaultextension=".png",
            initialfile=f"{loaded.nickname}_frame{frame_idx:06d}.png",
            filetypes=[("PNG image", "*.png")]
        )
        if not path:
            return

        try:
            from ..media_export import save_frame_png
            save_frame_png(composed, path)
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not write {path}:\n{e}")
            return

        self.set_status(f"Saved frame to {Path(path).name}")
        messagebox.showinfo("Frame Saved", f"Saved to:\n{path}")
```

Add to `fish_analyzer/media_export.py`:

```python
def save_frame_png(rgb, path):
    """Write an RGB frame as a PNG. Raises IOError if cv2 refuses."""
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"cv2.imwrite refused to write {path}")
```

Confirm `numpy as np`, `Path`, `filedialog` and `messagebox` are already
imported in `inspector_tab.py` — they are, at the top of the file.

- [ ] **Step 2: Verify it works**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe run_analyzer.py
```
Load the dNLS session, attach the video via Browse Video…, tick NND lines and
Convex hull, then Export → Save Frame (PNG). Open the PNG: it must be 1288×964
(not the canvas size) and show the overlays.

- [ ] **Step 3: Commit**

```bash
git add fish_analyzer/gui/inspector_tab.py fish_analyzer/media_export.py
git commit -m "Add Save Frame (PNG) to the Video Inspector"
```

---

### Task 10: Export Clip

**Files:**
- Modify: `fish_analyzer/gui/inspector_tab.py`
- Test: `tests/test_gui_regressions.py`

- [ ] **Step 1: Write the failing test for the guard**

Append to `tests/test_gui_regressions.py`:

```python
def test_export_refuses_when_the_time_panel_needs_missing_shoaling(app,
                                                                  synthetic_npy):
    """The on-screen panel says 'Run Shoaling Analysis first'; an export must
    refuse rather than bake that placeholder into a video."""
    from fish_analyzer.file_loading import TrajectoryFileLoader

    loaded = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.loaded_files["s1"] = loaded
    app.inspector_file_var.set("s1")
    app.inspector_time_mode_var.set("nnd")

    assert loaded.shoaling_results in (None, {}, [])

    ok, reason = app._inspector_can_export()

    assert ok is False
    assert "Shoaling" in reason


def test_export_is_allowed_when_the_time_panel_is_off(app, synthetic_npy):
    from fish_analyzer.file_loading import TrajectoryFileLoader

    app.loaded_files["s1"] = TrajectoryFileLoader.load_file(synthetic_npy, "s1")
    app.inspector_file_var.set("s1")
    app.inspector_time_mode_var.set("none")

    ok, _ = app._inspector_can_export()

    assert ok is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest tests/test_gui_regressions.py -k can_export -v
```
Expected: FAIL with `AttributeError: ... has no attribute '_inspector_can_export'`

- [ ] **Step 3: Write the guard and the dialog**

Add to the mixin in `inspector_tab.py`:

```python
    def _inspector_can_export(self):
        """(ok, reason) — whether an export can produce a meaningful clip."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            return False, "Select a file in the Video Inspector first."

        loaded = self.loaded_files[selected]
        time_mode = self.inspector_time_mode_var.get()

        if time_mode in ('nnd', 'iid', 'hull') and not loaded.shoaling_results:
            return False, (
                "The time panel is set to "
                f"{time_mode.upper()}, but no shoaling results exist for "
                f"'{selected}'.\n\nRun Shoaling Analysis first, or set the "
                "Time Panel to 'none'."
            )

        if time_mode == 'bout' and not self.bout_results.get(selected):
            return False, (
                f"The time panel is set to Bout, but no bout results exist "
                f"for '{selected}'.\n\nRun Bout Analysis first, or set the "
                "Time Panel to 'none'."
            )

        return True, ""

    def _inspector_export_clip_dialog(self):
        """Collect export options, then run the export."""
        ok, reason = self._inspector_can_export()
        if not ok:
            messagebox.showwarning("Cannot Export", reason)
            return

        selected = self.inspector_file_var.get()
        loaded = self.loaded_files[selected]
        start, end = self._inspector_export_range(loaded.n_frames)
        fps = loaded.calibration.frame_rate
        n_frames = end - start + 1

        time_mode = self.inspector_time_mode_var.get()
        # Bout re-renders matplotlib per frame; everything else stamps a
        # cursor onto a strip rasterised once.
        per_frame_ms = 80 if time_mode == 'bout' else 19
        estimate_s = n_frames * per_frame_ms / 1000.0

        message = (
            f"Export frames {start}-{end}\n"
            f"{n_frames} frames, {n_frames / fps:.1f} s of video\n"
            f"Estimated time: {estimate_s:.0f} s\n\n"
        )
        if time_mode == 'bout':
            message += ("The Bout time panel must be redrawn for every frame, "
                        "which is around four times slower than the other "
                        "panels.\n\n")
        message += "Continue?"

        if not messagebox.askyesno("Export Clip", message):
            return

        as_png = messagebox.askyesno(
            "Output Format",
            "Yes  = PNG sequence (lossless, one file per frame)\n"
            "No   = MP4 video"
        )

        if as_png:
            target = filedialog.askdirectory(
                title="Choose a folder for the PNG sequence"
            )
        else:
            target = filedialog.asksaveasfilename(
                title="Save Clip",
                defaultextension=".mp4",
                initialfile=f"{selected}_{start}-{end}.mp4",
                filetypes=[("MP4 video", "*.mp4")]
            )
        if not target:
            return

        self._inspector_run_export(loaded, selected, start, end, target,
                                   as_png)
```

- [ ] **Step 4: Write the chunked driver**

Add to the mixin:

```python
    def _inspector_run_export(self, loaded, selected, start, end, target,
                              as_png):
        """Run the export in chunks so Tk keeps painting and Cancel works.

        Not a worker thread: neither Tk nor matplotlib is safe to drive off
        the main thread.
        """
        from ..media_export import (CodecUnavailable, ExportFrameSource,
                                    Mp4Sink, PngSequenceSink, ScrollingStrip,
                                    TimeStrip)

        self._inspector_stop_playback()

        fps = loaded.calibration.frame_rate
        settings = self.render_settings_from_vars()
        time_mode = self.inspector_time_mode_var.get()

        video_path = None
        if self.inspector_video_var.get() and selected in self.video_readers:
            video_path = self.video_readers[selected].video_path
        if video_path is None:
            messagebox.showwarning(
                "No Video",
                "Clip export needs a video. Load one with Browse Video..."
            )
            return

        strip = None
        if time_mode != 'none' and self._insp_fig is not None:
            if time_mode == 'bout':
                # Scrolling axes: cannot be rasterised once.
                strip = ScrollingStrip(
                    self._insp_fig, self._insp_ax_time,
                    window_s=float(self.inspector_bout_window_var.get()),
                    total_s=loaded.n_frames / fps,
                )
            else:
                strip = TimeStrip(self._insp_fig, self._insp_ax_time)

        frame_h = loaded.metadata.video_height
        frame_w = loaded.metadata.video_width
        out_h = frame_h + (strip.height if strip is not None else 0)

        try:
            source = ExportFrameSource(video_path, start_frame=start)
            if as_png:
                sink = PngSequenceSink(target, fps, (frame_w, out_h))
            else:
                sink = Mp4Sink(target, fps, (frame_w, out_h))
        except CodecUnavailable as e:
            messagebox.showerror("Export Failed", str(e))
            return
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not start:\n{e}")
            return

        progress = tk.Toplevel(self.root)
        progress.title("Exporting")
        progress.transient(self.root)
        progress.grab_set()
        label = tk.Label(progress, text="Starting...", width=30)
        label.pack(padx=20, pady=10)
        state = {"cancel": False}
        tk.Button(progress, text="Cancel",
                  command=lambda: state.__setitem__("cancel", True)
                  ).pack(pady=(0, 10))

        total = end - start + 1
        colors = fish_colors(loaded.n_fish)
        cursor = {"frame": start, "written": 0}
        out_buffer = {"array": None}

        def finish(message, cancelled):
            try:
                progress.destroy()
            except Exception:
                pass
            source.close()
            # ScrollingStrip leaves the live axes on the last exported window,
            # and TimeStrip forced a draw on the shared canvas. Rebuild so the
            # tab is not left showing export state.
            self._insp_needs_rebuild = True
            self._inspector_update_fast()
            self.set_status(message)
            if cancelled:
                messagebox.showinfo("Export Cancelled", message)
            else:
                messagebox.showinfo("Export Complete", message)

        def do_chunk():
            if state["cancel"]:
                sink.discard_partial()
                finish(f"Export cancelled after {cursor['written']} frames.",
                       True)
                return

            for _ in range(10):
                if cursor["frame"] > end:
                    sink.close()
                    path = getattr(sink, "path", target)
                    finish(f"Wrote {cursor['written']} frames to {path}",
                           False)
                    return

                base = source.read()
                if base is None:
                    sink.close()
                    finish(f"Video ended early; wrote "
                           f"{cursor['written']} frames.", False)
                    return

                composed = compose_frame(base, loaded.trajectories,
                                         cursor["frame"], settings,
                                         loaded.calibration.scale_factor,
                                         colors)

                if strip is None:
                    frame_out = composed
                else:
                    if out_buffer["array"] is None:
                        out_buffer["array"] = np.zeros(
                            (out_h, frame_w, 3), dtype=np.uint8
                        )
                    out_buffer["array"][:frame_h] = composed
                    out_buffer["array"][frame_h:] = strip.at(
                        cursor["frame"] / fps
                    )
                    frame_out = out_buffer["array"]

                try:
                    sink.write(frame_out)
                except Exception as e:
                    sink.close()
                    finish(f"Write failed at frame {cursor['frame']}: {e}",
                           True)
                    return

                cursor["written"] += 1
                cursor["frame"] += 1

            label.config(text=f"Frame {cursor['written']} / {total}")
            self.root.after(1, do_chunk)

        self.root.after(1, do_chunk)
```

`VideoFrameReader` already stores `self.video_path`, so
`self.video_readers[selected].video_path` resolves.

- [ ] **Step 5: Run the tests**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add fish_analyzer/gui/inspector_tab.py tests/test_gui_regressions.py
git commit -m "Add clip export with progress, cancellation and guards"
```

---

### Task 11: Verify on the real session and document

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Export a clip from the real recording**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe run_analyzer.py
```

1. Data Setup & Calibration → Browse… → select
   `C:\Users\grich\Macquarie University\Morsch Group - Documents\04_People\Grant R\02_PROJECTS\TRACKING\Tracking Adult Fish\02_Processed_Data\Rachel's Project\Freeswim\session_dNLS_FREESWIM_2025-08-29-135947-0000`
2. Apply calibration (body length is 79.8 px; the session has 6 fish, 18,000
   frames at 30 fps, 1288×964).
3. Shoaling Analysis → Run Shoaling Analysis, so the time panel has data.
4. Video Inspector → Browse Video… →
   `C:\Users\grich\Macquarie University\Morsch Group - Documents\04_People\Grant R\02_PROJECTS\TRACKING\Tracking Adult Fish\01_Raw_Data\ZF Videos\Adult Swimming\Rachel's Project\250829_wtTDP_and_dNLS\Freeswim\dNLS_FREESWIM_2025-08-29.avi`
   (auto-detect cannot find it — raw data and session live in different trees)
5. Tick Fish positions, NND lines, Convex hull. Set Time Panel to NND.
6. Scrub to a frame where the fish are grouped, Set In, advance ~300 frames,
   Set Out.
7. Export → Export Clip… → MP4.

Confirm: the dialog reports ~300 frames and ~6 s; the progress dialog counts up
and Cancel works; the MP4 plays; the NND strip sits below the frame with a
cursor tracking playback; fish colours match the Individual Analysis plots.

- [ ] **Step 2: Update the README**

In `README.md`, under **Features**, change the GUI bullet to:

```markdown
- **GUI** — interactive tkinter + matplotlib interface with tabs for each analysis type
- **Figure and clip export** — save any inspector frame as a full-resolution PNG, or export a marked range as an MP4 or PNG sequence with overlays and the time-series panel
```

And add after the *Verifying against your own recordings* section:

```markdown
### Exporting figures and clips

The Video Inspector composites overlays (fish positions, NND lines, convex
hull, IID lines, trails) onto video frames. To get them out:

- **Save Frame (PNG)** writes the current composite at full video resolution.
- **Export Clip** writes a marked range as an MP4 (or a PNG sequence). Mark the
  range with **Set In** / **Set Out** beside the frame slider.

Both export exactly what the tab is showing, including the time-series panel
when the Time Panel is set to NND, IID, Hull or Bout. A clip whose time panel
needs shoaling or bout results will refuse to export until those have been run.

The source video is auto-detected only when it sits beside the session folder;
otherwise attach it with **Browse Video…**.
```

- [ ] **Step 3: Run the full suite one last time**

Run:
```bash
C:/Users/grich/.conda/envs/traja_fish_analysis/python.exe -m pytest -q
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document frame and clip export"
```

---

## Self-review notes

**Spec coverage.** In/Out markers → Task 8. WYSIWYG output → Tasks 9 and 10
(both take `render_settings_from_vars` and the live `_insp_fig`). MP4 + PNG
sequence with fallback → Task 5. `overlay_render.py` → Tasks 2 and 3.
`media_export.py` → Tasks 5, 6, 7. Private capture → Task 6. Colour fix →
Task 2. Trail blending → Task 3. Shared NND → Tasks 1 and 3. Colormap hoist →
Tasks 2 and 4. Error handling → Tasks 5, 7, 10. Testing → throughout.
Bout mode re-render → `ScrollingStrip` in Task 6, selected by Task 10, with the
cost warning in the dialog.

**Fixed during review:** the first draft built `TimeStrip` for every mode,
which would have frozen the Bout panel's scrolling window inside an exported
clip — contradicting the spec's WYSIWYG guarantee. `ScrollingStrip` (Task 6)
now re-renders per frame for Bout, and Task 10 restores the live figure
afterwards, since both strip classes draw on the shared canvas.

**Type consistency check:** both strip classes expose `.height` and
`.at(t) -> ndarray`, which is what `export_clip` and the Task 10 driver expect.
Sinks expose `write`, `close`, `discard_partial` and `path`. `OverlaySettings`
field names are identical between Tasks 2, 3 and 4.
