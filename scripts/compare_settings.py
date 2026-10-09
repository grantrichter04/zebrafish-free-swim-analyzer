#!/usr/bin/env python
"""Compare the old and new analysis settings on real idtracker.ai sessions.

The 2026-10-09 settings (CHANGELOG "Unreleased") were chosen from simulated
fish. This script is the check against real recordings before they are
merged: it analyses each session twice and reports, per fish, what moved.

    python scripts/compare_settings.py <session_folder> [<session_folder> ...]
    python scripts/compare_settings.py <folder_of_sessions> --csv out.csv

OLD = the settings before the change: no smoothing, 1% tracking minimum,
      5-frame freezes, every second counted for straightness.
NEW = the defaults now.

It also estimates the tracker's wobble on this rig, from stretches where a
fish sits still: the number the 0.17 s smoothing and the 0.5 BL/s freeze
threshold were chosen against (simulation assumed 0.5-1 px). Body lengths
are idtracker.ai's own, one per session, unless --body-length is given.
"""
import argparse
import contextlib
import csv
import io
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fish_analyzer.data_structures import CalibrationSettings  # noqa: E402
from fish_analyzer.file_loading import TrajectoryFileLoader  # noqa: E402
from fish_analyzer.processing import (  # noqa: E402
    ProcessingParameters, process_and_analyze_file)
from fish_analyzer.segments import contiguous_tracked_segments, smooth_within_segments  # noqa: E402

OLD = ProcessingParameters(min_valid_percentage=0.01, min_freeze_seconds=5 / 30,
                           freeze_bridge_seconds=0.0, straightness_min_speed=0.0,
                           smoothing_seconds=0.0)
NEW = ProcessingParameters()

COLUMNS = [("total_distance", "Dist"), ("median_speed", "Med"),
           ("speed_p99", "P99"), ("mean_path_straightness", "Straight"),
           ("freeze_fraction_pct", "Frz%"), ("freeze_count", "FrzN")]


def sessions_from(paths):
    for p in map(Path, paths):
        if (p / "trajectories" / "trajectories.npy").exists():
            yield p
        else:
            yield from sorted(q.parent.parent for q in p.glob("*/trajectories/trajectories.npy"))


def analyse(folder, params, body_length):
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = TrajectoryFileLoader.load_from_session_folder(folder)
        if body_length:
            loaded.calibration = CalibrationSettings.from_body_lengths(
                body_length, loaded.metadata.frames_per_second)
        fish = process_and_analyze_file(loaded, params)
    return loaded, {f.fish_id: f for f in fish}


def wobble_px(loaded, fish_idx):
    """Median frame-to-frame step, in pixels, while the fish is still.

    "Still" = 1 s or more in which the smoothed position stays within 0.1 BL
    of its mean. There a step is wobble alone; for round Gaussian noise of
    s.d. sigma per axis the median step is 2 * sigma * sqrt(ln 2), so
    sigma ~ median step / 1.665.
    Returns (sigma estimate in px, seconds of stillness it rests on).
    """
    xy = np.asarray(loaded.trajectories[:, fish_idx, :], dtype=float)
    fps = loaded.metadata.frames_per_second
    bl = loaded.metadata.body_length
    win = int(round(fps))
    sx, sy = smooth_within_segments(xy[:, 0], xy[:, 1], 5)
    steps = []
    for a, b in contiguous_tracked_segments(xy[:, 0], xy[:, 1], min_length=win):
        for s in range(a, b - win + 1, win):
            wx, wy = sx[s:s + win], sy[s:s + win]
            if np.max(np.hypot(wx - wx.mean(), wy - wy.mean())) < 0.1 * bl:
                steps.append(np.hypot(*np.diff(xy[s:s + win], axis=0).T))
    if not steps:
        return np.nan, 0.0
    return (float(np.median(np.concatenate(steps)) / (2 * np.sqrt(np.log(2)))),
            len(steps) * win / fps)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--body-length", type=float, help="pixels per BL for every session")
    ap.add_argument("--csv", help="also write every number to this CSV")
    args = ap.parse_args()

    rows = []
    for folder in sessions_from(args.paths):
        old_file, old = analyse(folder, OLD, args.body_length)
        new_file, new = analyse(folder, NEW, args.body_length)
        print(f"\n=== {folder.name}  ({old_file.n_fish} fish, "
              f"{old_file.duration_minutes:.1f} min, "
              f"{old_file.calibration.pixels_per_unit:.1f} px/BL)")
        print(f"{'fish':>4} {'trk%':>5} {'wob px':>6}  "
              + "  ".join(f"{h:>13}" for _, h in COLUMNS) + "   (old -> new)")
        for i in range(old_file.n_fish):
            label = old_file.metadata.identity_labels[i]
            sigma, still_s = wobble_px(old_file, i)
            o = old[i].metrics if i in old else {}
            n = new[i].metrics if i in new else {}
            trk = (old[i].valid_percentage * 100) if i in old else np.nan
            cells = "  ".join(f"{o.get(k, np.nan):6.2f}>{n.get(k, np.nan):6.2f}"
                              for k, _ in COLUMNS)
            print(f"{label:>4} {trk:5.1f} {sigma:6.2f}  {cells}"
                  + ("" if i in new else f"   NEW: excluded, {new_file.excluded_fish.get(i)}"))
            row = {"Session": folder.name, "Fish": label, "Tracked_pct": round(trk, 1),
                   "Wobble_px": round(sigma, 3), "StillSeconds": round(still_s, 1),
                   "PixelsPerBL": round(old_file.calibration.pixels_per_unit, 2),
                   "NewStatus": "ok" if i in new else f"excluded: {new_file.excluded_fish.get(i)}",
                   "New_StraightnessSeconds_pct": n.get("straightness_windows_used_pct", np.nan),
                   "New_DistancePerTrackedMin": n.get("distance_per_tracked_min", np.nan)}
            for k, h in COLUMNS:
                row[f"Old_{h}"], row[f"New_{h}"] = o.get(k, np.nan), n.get(k, np.nan)
            rows.append(row)
        shoal_ok = np.isfinite(old_file.trajectories).all(axis=(1, 2)).mean() * 100
        print(f"     frames with every fish tracked (shoaling uses only these): {shoal_ok:.1f}%")

    if not rows:
        sys.exit("No idtracker.ai sessions found under those paths.")
    sigmas = [r["Wobble_px"] for r in rows if np.isfinite(r["Wobble_px"])]
    if sigmas:
        print(f"\nTracker wobble across fish with still stretches: median "
              f"{np.median(sigmas):.2f} px (range {min(sigmas):.2f}-{max(sigmas):.2f}). "
              "The new defaults were chosen for 0.5-1.5 px.")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
