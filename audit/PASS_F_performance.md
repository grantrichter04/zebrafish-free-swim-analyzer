# Pass F — Performance, efficiency, and resource use

**Shared rules for every audit pass:**
- **Report, don't fix.** This pass ends in a findings file, not a diff.
- **Verify, don't inherit.** The seed findings below came from reading, not running. Confirm or refute each.
- **Label every finding** `CONFIRMED` (you measured it) or `PLAUSIBLE` (read-only reasoning). For a performance pass, `CONFIRMED` means **you have a timing number**.
- Read `AUDIT_PLAN.md` in the repo root first for the stock-take context.

---

## Goal

Make this fast enough that analysis isn't a coffee break, and make it not eat all the RAM. Typical workload: several 10–20 minute recordings at 30 fps with 6 fish — roughly 20k–40k frames per file, batch-processed.

**Measure before you claim.** Profile with real or synthetic data at realistic scale (30k frames × 6 fish minimum). A finding without a number attached is a hypothesis, and should be labelled as one. Report both wall-clock time and peak memory.

Prerequisite: a working environment — `import fish_analyzer` currently fails because `traja` is missing. See `AUDIT_A_REPRODUCIBILITY.md` if it exists.

## Scope

Whole repo. Prioritise by what the user actually waits for.

## Seed findings — verify and measure each

### Almost certainly the worst offender

1. **Thigmotaxis is a Python double loop with a shapely call per point.** `spatial.py:403-434` loops over every frame × every fish and calls `arena_poly.contains(Point(...))` plus `center_poly.contains(...)`. At 30k frames × 6 fish that's ~360k point-in-polygon calls through the shapely Python layer, constructing a `Point` object each time. Measure it, then evaluate vectorized alternatives — `shapely.contains_xy` (Shapely 2.0+, which `requirements.txt` already requires), `matplotlib.path.Path.contains_points`, or a prepared geometry. Report the speedup you actually achieve on a prototype.

### Video reading — likely a correctness bug as well as a performance one

2. **`VideoFrameReader` shares one `cv2.VideoCapture` across threads with no lock.** `_preload_worker` (`video_utils.py:230`) calls `self.cap.set()` and `self.cap.read()` on the same capture object the main thread uses. `cv2.VideoCapture` is not thread-safe. Worse: the worker moves the underlying stream position without updating `self.last_frame_read`, so the main thread's `is_sequential` fast path (`video_utils.py:180`) can read a frame from the wrong position and **cache it under the wrong frame number**. Verify this — it would mean the inspector can display frame N's image labelled as frame M. If confirmed, this belongs in the correctness findings too, not just performance.
3. **`VideoFrameCache` LRU is O(n) per access.** `access_order.remove(frame_num)` (`video_utils.py:53`, `62`) is a linear list scan on every hit and every put. Use `collections.OrderedDict` (`move_to_end`) or a dict + deque.
4. **Every frame is copied twice.** `.copy()` on both `put` (line 71) and `get` (line 55). With a 100-frame cache at 1080p RGB that's ~600 MB resident plus a fresh ~6 MB allocation on every single frame access during playback. Measure peak RSS during a playback session and assess whether the defensive copies are actually needed.
5. **`time.sleep(0.001)` per frame in the preload worker** (line 259) to "not hog CPU" — assess whether this helps or just caps preload throughput at 1000 fps.

### Vectorizable inner loops

6. **Three hand-written run-length loops over every frame:** `processing.py::_calc_freeze_metrics` (line 419), `processing.py::_calc_burst_metrics` (line 484), `bout_analysis.py::_find_intervals` (line 161). All are `for` loops over the full frame count, per fish. All three are the same operation and all three vectorize to a couple of `np.diff`/`np.flatnonzero` calls.
7. **`_calc_path_straightness`** (`processing.py:679`) loops over sliding windows in Python with a fresh `np.diff` per window.
8. **`spatial.py::get_smoothed_fish_timeseries`** (line 276) is O(n²): for every NaN sample it computes distances to *all* valid indices to find the nearest. Replace with a forward/backward fill or `np.searchsorted`.
9. **`HeatmapGenerator.generate_combined_heatmap`** (`spatial.py:551-562`) builds Python lists via `.extend()` over every fish's positions — potentially millions of Python floats — before passing them to `histogram2d`. Use `np.concatenate`.
10. **`generate_all_individual_heatmaps`** (`spatial.py:631`) re-derives bin edges per fish and discards all but the first.
11. **`shoaling.py::get_all_pairwise_distances_at_frame`** (line 462) is a Python double loop over fish pairs; `_calculate_nnd_at_frame` builds a full n×n `cdist` matrix per sampled frame in a Python loop over samples — assess whether batching across samples is worth it for realistic fish counts (6), or whether this is already fine.
12. **`head_detection_test.py:49-55`** interpolates NaN gaps with a nested Python loop over fish × coordinate. Minor, but check the same pattern isn't repeated at scale elsewhere.

### Whole-pipeline questions

13. **Are trajectories loaded once and shared, or re-read?** Trace whether `trajectories` arrays get copied per analysis. `_transform_coordinates` (`processing.py:264`) does `raw_coords.copy()` then a full-array multiply — check for redundant copies across the pipeline, and whether float64 is needed throughout or float32 would halve the footprint.
14. **Batch processing over N files** (`data_tab.py:596-641`): does memory grow monotonically as files accumulate in `self.loaded_files`? Every `LoadedTrajectoryFile` retains its full trajectory array plus `processed_data` plus a `speed_time_series` per fish. Measure peak RSS for 10 loaded files and say whether that's a practical ceiling.
15. **Matplotlib figure lifecycle.** Check whether figures and canvases are reused or recreated per redraw, and whether any are created without ever being closed — a classic slow memory leak in long tkinter sessions.

## Deliverable

`AUDIT_F_PERFORMANCE.md` in the repo root, containing:

- **(a)** A profile table for a realistic workload: what the user waits for, ranked by measured wall-clock cost. Include your test setup (frames, fish, files, machine) so the numbers are reproducible.
- **(b)** Findings ranked by *measured* impact, not by how ugly the code looks. Each labelled `CONFIRMED` (with a number) or `PLAUSIBLE`.
- **(c)** Peak memory findings separately — for a lab machine running a batch, RAM ceiling may matter more than seconds.
- **(d)** For each proposed optimisation: expected speedup, risk of changing numerical output, and effort. **Anything that could change an exported number goes in its own section** — I need to know that before it's touched.
- **(e)** An explicit "leave this alone" list: code that looks inefficient but isn't on any hot path. I don't want effort spent micro-optimising a function called once per session.

## Constraints

- Report only. No optimisation in this pass. A prototype used purely to measure a potential speedup is fine — put it under `scratch/` or a temp dir, not in the package, and say so.
- Correctness beats speed every time. If an optimisation would change a metric's value even slightly, that's a Pass B question, not a Pass F decision.
