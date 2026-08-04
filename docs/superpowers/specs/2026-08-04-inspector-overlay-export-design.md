# Video Inspector: frame and clip export

**Date:** 2026-08-04
**Status:** approved, not yet implemented

## Problem

The Video Inspector composites overlays (fish positions, NND lines, convex
hull, IID lines, trails, bout ring) onto video frames and can pair them with a
time-series panel underneath. There is no way to get any of it out of the
application. Producing a figure or showing a colleague a shoaling moment means
taking a screenshot of a canvas-downscaled image.

Two capabilities are missing:

1. Save the current composited frame as an image.
2. Export a marked range of frames as a video clip, with the time panel
   stacked below the frame.

## Scope

In scope:

- In/Out markers on the Video Inspector's frame slider.
- `Save Frame (PNG)…` — the current composite at full video resolution.
- `Export Clip…` — MP4, or a numbered PNG sequence.
- Extraction of the overlay compositor out of the GUI into a pure module.
- The correctness and performance fixes listed under *Fixes folded in*, all of
  which lie on the code path being extracted.

Out of scope:

- A headless CLI script. The compositor is being made GUI-free so this stays a
  thin `argparse` wrapper, but it is not built now.
- Any change to metrics, exports, or analysis results.
- The app-wide `plt.cm.tab10(np.linspace(0, 1, n))` colour convention, under
  which per-fish colours shift when a session has a different fish count. Real,
  but unrelated to this work.

## Decisions

| Question | Decision |
|---|---|
| Clip range selection | In/Out markers on the existing frame slider, editable in the export dialog |
| Output contents | WYSIWYG — whatever overlays are ticked, plus the time panel when it is not "none" |
| Clip format | MP4 by default, automatic MJPG/`.avi` fallback, PNG sequence as an option |
| Headless script | Deferred; GUI only |
| Colour channel fix | In scope |

## Architecture

Three units, each independently testable.

### `fish_analyzer/overlay_render.py` (new)

No `tkinter` import. Pure functions over arrays.

- `OverlaySettings` — frozen dataclass: `show_positions`, `show_nnd`,
  `show_hull`, `show_iid`, `iid_focus`, `show_bout_ring`, `bout_fish`,
  `trail_length`, `trail_opacity`, `trail_width`, `dot_radius`.
  Replaces seven inline `.get()` calls on tk variables and the fourteen
  positional parameters `_inspector_draw_cv2` currently takes.
- `compose_frame(base, trajectories, frame_idx, settings, scale) -> ndarray`
  — returns a new RGB `uint8` array. Never mutates `base`.
- `fish_colors(n_fish) -> ndarray` — the tab10 sampling, hoisted so it is
  computed once rather than per frame.

`base` is RGB throughout, matching what `VideoFrameReader` produces
(`video_utils.py:196` converts `BGR2RGB` on read) and what PIL expects.

### `fish_analyzer/media_export.py` (new)

- `ExportFrameSource` — opens its **own** `cv2.VideoCapture`, seeks once to the
  start frame, then reads sequentially. Deliberately does not reuse
  `self.video_readers[...]`; see *Why a private capture*.
- `TimeStrip` — rasterises the matplotlib time panel **once** via
  `fig.canvas.buffer_rgba()`, exposes `cursor_x(t)` derived from
  `ax.transData`, and stamps the cursor per frame with `cv2.line` onto a
  restored copy of the pristine strip. Valid only for panels whose axes are
  fixed for the whole session — see *Bout mode*.
- `Mp4Sink` — `mp4v`, falling back to MJPG/`.avi` when
  `VideoWriter.isOpened()` is false. Raises if both fail; never leaves a
  zero-byte file.
- `PngSequenceSink` — numbered lossless frames in a chosen directory.
- `export_clip(source, sink, strip, settings, frame_range, on_progress,
  should_cancel)` — the loop. Output canvas allocated once as
  `(vid_h + strip_h, vid_w, 3)`; the frame is written into `out[:vid_h]` and
  the strip into `out[vid_h:]`, so no per-frame `vstack` allocation.

### `fish_analyzer/gui/inspector_tab.py` (reduced)

Keeps tk only: controls, In/Out marker state, the export dialog, and driving
the loop. `render_settings_from_vars()` is the single place tk state becomes an
`OverlaySettings`. The live view calls the same `compose_frame` as the export,
so what is exported cannot drift from what is displayed.

At 1,698 lines this is the largest module in the repo; the extraction moves
roughly 200 lines of drawing out of it.

## Data flow

```
Save Frame:
  tk vars -> OverlaySettings -> compose_frame(current frame) -> cv2.imwrite

Export Clip:
  tk vars -> OverlaySettings          (read once, at export start)
  In/Out  -> frame_range
  time panel figure -> TimeStrip (rasterised once; NND/IID/Hull only)
  for each frame:
      ExportFrameSource.read()  ->  compose_frame  ->  out[:vid_h]
      TimeStrip.at(t)           ->                     out[vid_h:]
        (bout mode: re-render the panel for this frame instead)
      sink.write(out)
```

## Why a private capture

`VideoFrameReader._preload_worker` calls `self.cap.set()` and `self.cap.read()`
on the same `VideoCapture` the foreground thread reads from
(`video_utils.py:245`). Only the frame cache is lock-protected; the capture is
not. A preload thread seeking mid-export would break sequential-read detection
and yield wrong frames.

The reader's LRU also copies every frame twice (`put` and `get`), which earns
its keep for scrubbing and is pure overhead for a single forward pass.

Measured on `dNLS_FREESWIM_2025-08-29.avi` (MJPG, 1288×964, 1.5 GB):

| Access pattern | Cost |
|---|---|
| Sequential `read()` | 9.9 ms/frame (101 fps) |
| `set(POS_FRAMES)` + `read()` per frame | 64.7 ms/frame |

A private capture also leaves the viewer's reader position untouched, so the
displayed frame does not jump when an export finishes.

## Performance

Measured on the same machine, 1288×964:

| Stage | Cost |
|---|---|
| Sequential decode | 9.9 ms/frame |
| `mp4v` encode | 6.3 ms/frame |
| MJPG fallback encode | 35 ms/frame, ~9× file size |
| Composite | ~2–3 ms/frame (estimated) |

≈19 ms/frame end to end: a 10 s clip in about 6 seconds, a full 18,000-frame
session in about 6 minutes. The dialog states the frame count before starting
so a full-session export is never accidental.

Rendering the time panel per frame with matplotlib would cost 50–100 ms/frame
and dominate everything else. Rasterising once and stamping the cursor reduces
that to a memcpy plus a line — the same reasoning behind the existing blitting
in the live view (`inspector_tab.py:1450`).

### Bout mode

The rasterise-once trick holds only while the panel's axes are fixed. For the
NND, IID and Hull panels they are: `xlim` is set once over the whole session at
rebuild (`inspector_tab.py:1082`) and thereafter only the cursor moves, which is
why the live view can blit them (`inspector_tab.py:1450`).

The Bout panel is different — it scrolls, resetting `xlim` to a window around
the current frame on every update (`inspector_tab.py:1473`), and the live view
consequently falls back to `draw_idle()` (`inspector_tab.py:1476`). Cropping a
pre-rendered strip would scroll the y-axis and its labels out of frame, so it is
not a valid shortcut here.

Decision: Bout mode re-renders the panel per frame. It stays available and
WYSIWYG, but the export dialog states that it is markedly slower — roughly
50–100 ms/frame instead of ~19, so a 10 s clip takes around 30 seconds rather
than 6. NND, IID and Hull use the fast path.

## Fixes folded in

All on the extracted path.

1. **Colour channels.** `_rgba_to_bgr` (`inspector_tab.py:1483`) emits B,G,R
   tuples that are drawn onto an RGB array, so every fish renders with red and
   blue exchanged — fish 0 is tab10 blue in the analysis plots and orange on
   the video. The numpy fallback path uses `_rgba_to_rgb_uint8` and is correct,
   so the two paths disagree; this is a slip, not a choice. Drop
   `_rgba_to_bgr`, use the RGB converter at all four call sites. Affects only
   what colour things are painted — no metric, CSV, or previously exported
   number changes.

2. **Trail blending.** The trail loop copies the whole frame once per fish
   (`inspector_tab.py:1503`) to alpha-blend. All trails share one alpha, so
   they can be drawn into a single overlay and blended once: with 6 fish at
   1288×964 that is 6 full-frame copies per frame reduced to 1.

3. **NND.** The inspector hand-rolls an O(n²) Python nearest-neighbour search
   per frame (`inspector_tab.py:1528`) while `shoaling.py` already does it with
   scipy `cdist`. Share the helper: faster, and the distances drawn on the
   frame agree with the exported CSV by construction.

4. **Colormap sampling** is recomputed every frame; hoist to `fish_colors()`.

## Error handling

Following the no-silent-failure principle established by Audit D.

| Condition | Behaviour |
|---|---|
| Time panel set to NND/IID/Hull, `shoaling_results` empty | Refuse with "Run Shoaling Analysis first" — never bake the placeholder text into a video |
| `VideoWriter.isOpened()` false | Fall back to MJPG/`.avi`; if that also fails, report the codec error and write nothing |
| No video loaded | Allowed — composites onto the background image or blank arena, as the viewer already does |
| Write error mid-export | Abort, close the sink, report the file and the OS error |
| In/Out set backwards | Normalised (swapped) at read time |
| In/Out never set | Dialog defaults to the whole recording and states the frame count and estimated duration |
| Cancel during export | Close sink, delete the partial MP4 (a truncated MP4 is unplayable); PNG sequences keep completed frames, which are individually valid |

Playback stops before an export begins. The loop runs in chunks of ~10 frames
through `root.after` rather than a worker thread — Tk and matplotlib are both
unsafe to drive off-thread — so the UI keeps painting and Cancel stays live.
Overlay settings are read once at export start, so toggling a control mid-run
cannot produce a clip that changes appearance halfway through.

## Testing

Most of this runs without a display, so it executes rather than skipping behind
the `display_available` fixture.

**`overlay_render`** — synthetic positions from the existing `conftest.py`
fixtures, no video file needed:

- All settings off returns the input frame unchanged (purity).
- A fish at a known pixel produces a dot at that pixel in tab10 blue. Fails on
  current code; pins fix 1.
- A `NaN` position draws nothing and does not raise. The fixtures already
  inject 2% dropouts, and the real session carries 0.93%.
- Fewer than three tracked fish draws no hull.
- NND lines connect the genuinely nearest pair.
- **NND drawn on a frame equals `ShoalingCalculator`'s value for that frame** —
  pins the "agree by construction" claim from fix 3.

**`media_export`:**

- `PngSequenceSink` writes the expected count and names under `tmp_path`.
- `Mp4Sink` with a monkeypatched `VideoWriter` reporting `isOpened() == False`
  takes the MJPG fallback; when both fail it raises rather than leaving an
  empty file.
- `cursor_x(t)` is tested as a function — time in, pixel out — rather than by
  inspecting rendered pixels, which is brittle across matplotlib versions.

**Behind `display_available`**, following `test_processing_params_reflect_the_gui`:

- `render_settings_from_vars` reflects what is actually ticked.
- In/Out normalise when set backwards.
- The export guard refuses when the time panel needs absent shoaling results.

**Manual, on real data.** Automated tests cannot judge whether output *looks*
right. Final check: load
`session_dNLS_FREESWIM_2025-08-29-135947-0000`, attach
`dNLS_FREESWIM_2025-08-29.avi` via Browse Video (auto-detect does not find it —
the recording and the session live in different trees), export a few seconds
with NND lines and the hull on, confirm it plays, and confirm fish colours match
the Individual Analysis plots.

## Risks

The extraction threads a settings object through the *live* render path, not
just the new export path. A mistake there degrades the viewer. Mitigated by
populating the settings object in one place, by the existing GUI regression
tests, and by the fact that the live view and the export share one compositor —
if the viewer looks right, the export is right.
