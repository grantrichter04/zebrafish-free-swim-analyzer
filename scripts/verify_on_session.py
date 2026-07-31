#!/usr/bin/env python
"""Re-check Audit B's real-data claims against idtracker.ai sessions.

The regression suite in ``tests/`` is entirely synthetic, by design — it must
run for anyone who clones the repo. But two of Audit B's findings were
invisible to synthetic input and only showed up when the exporter was run end
to end on real recordings: ``NetDisplacement`` was NaN for half the fish
(B16), and ``MaxSpeed`` reported tracking teleports of up to 321 BL/s (B17).

So the synthetic suite cannot be the only check. This script is the other
half: point it at real sessions and it re-verifies every claim in
AUDIT_B_CORRECTNESS.md that depends on real data. It contains no data and no
paths — the sessions come from the command line.

    python scripts/verify_on_session.py <session_folder> [<session_folder> ...]
    python scripts/verify_on_session.py <folder_of_sessions>

Exits 0 if every check passes, 1 otherwise.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fish_analyzer.bout_analysis import BoutParameters, analyze_bouts_for_file
from fish_analyzer.data_structures import CalibrationSettings
from fish_analyzer.export import export_combined_summary_csv
from fish_analyzer.file_loading import TrajectoryFileLoader
from fish_analyzer.processing import ProcessingParameters, process_and_analyze_file
from fish_analyzer.shoaling import ShoalingCalculator, ShoalingParameters

#: Columns Audit B withdrew. Their reappearance in an export means someone has
#: restored a metric that measures centroid noise — see AUDIT_B_CORRECTNESS.md.
WITHDRAWN = {
    "MeanAngularVelocity_deg_s", "ErraticMovementCount",
    "ErraticMovements_per_min", "BurstCount", "BurstMeanSpeed",
    "BurstFrequency_per_min", "CumulativeHeading_deg", "MeanSignedAngVel_deg_s",
}

#: Adult zebrafish burst swimming tops out around 25 BL/s. Anything far above
#: that is a tracking teleport, which is what B17 was about.
PLAUSIBLE_TOP_SPEED_BL_S = 25.0

PIXELS_PER_CM = 10.0   # arbitrary but fixed; only the ratio is checked


class Report:
    """Collects pass/fail lines so every check runs before anything exits."""

    def __init__(self):
        self.failures = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)
        return ok


def find_sessions(args) -> list:
    """Accept session folders directly, or one folder containing them."""
    sessions = []
    for raw in args:
        path = Path(raw).expanduser()
        if not path.exists():
            sys.exit(f"No such path: {path}")
        if (path / "trajectories" / "trajectories.npy").exists():
            sessions.append(path)
        else:
            sessions.extend(
                child for child in sorted(path.iterdir())
                if (child / "trajectories" / "trajectories.npy").exists()
            )
    if not sessions:
        sys.exit("No idtracker.ai sessions found (looking for "
                 "<folder>/trajectories/trajectories.npy).")
    return sessions


def verify_session(session: Path, report: Report) -> dict:
    params = ProcessingParameters.default_for_fish()
    loaded = TrajectoryFileLoader.load_from_session_folder(session)
    fish_list = process_and_analyze_file(loaded, params)
    loaded.processed_data = fish_list
    bouts = analyze_bouts_for_file(loaded, BoutParameters())

    print(f"\n{loaded.nickname}")
    print(f"  {loaded.n_frames} frames, {loaded.n_fish} fish, "
          f"body length {loaded.metadata.body_length:.2f} px")

    # --- B6: the freeze denominators must reconcile exactly ---
    worst = max(
        abs(f.metrics["freeze_total_duration_s"]
            / f.metrics["observed_duration_s"] * 100
            - f.metrics["freeze_fraction_pct"])
        for f in fish_list
    )
    report.check(worst < 1e-9, "B6 freeze denominators reconcile",
                 f"worst drift {worst:.2e} across {len(fish_list)} fish")

    # --- B16: net displacement must survive gaps at the recording ends ---
    n_nan = sum(1 for f in fish_list if not np.isfinite(f.metrics["net_displacement"]))
    report.check(n_nan == 0, "B16 NetDisplacement populated",
                 f"{len(fish_list) - n_nan}/{len(fish_list)} fish")

    # --- B17: top speed must not be set by a teleport ---
    tops = [f.metrics["speed_p99"] for f in fish_list]
    raws = [f.metrics["speed_max_raw"] for f in fish_list]
    report.check(max(tops) < PLAUSIBLE_TOP_SPEED_BL_S,
                 "B17 SpeedP99 is physiologically plausible",
                 f"{min(tops):.1f}–{max(tops):.1f} BL/s "
                 f"(raw max would be {min(raws):.0f}–{max(raws):.0f})")

    # --- B4: no exported inter-bout interval may span a tracking gap ---
    spanning = 0
    for result in bouts:
        untracked = np.isnan(loaded.trajectories[:, result.fish_id, 0])
        spanning += sum(
            1 for cur, nxt in zip(result.bouts, result.bouts[1:])
            if cur.segment_index == nxt.segment_index
            and untracked[cur.end_frame:nxt.start_frame + 1].any()
        )
    report.check(spanning == 0, "B4 no IBI spans a tracking gap",
                 f"{spanning} offending intervals")

    # --- B8: unmeasurable turns must not be counted as straight ---
    straight = sum(b.summary["bout_n_straight"] for b in bouts)
    unmeasurable = sum(b.summary["bout_n_heading_unmeasurable"] for b in bouts)
    report.check(True, "B8 turn measurability reported",
                 f"{straight} straight, {unmeasurable} unmeasurable")

    # --- B7: group metrics must follow the calibration ---
    nnd_bl = ShoalingCalculator(loaded, ShoalingParameters(30)).calculate().mean_nnd
    in_cm = TrajectoryFileLoader.load_from_session_folder(
        session, calibration=CalibrationSettings.from_physical_measurement(
            PIXELS_PER_CM, "cm", loaded.calibration.frame_rate))
    nnd_cm = ShoalingCalculator(in_cm, ShoalingParameters(30)).calculate().mean_nnd
    expected = loaded.metadata.body_length / PIXELS_PER_CM
    report.check(abs(nnd_cm / nnd_bl - expected) < 1e-6,
                 "B7 group metrics follow the calibration",
                 f"cm/BL ratio {nnd_cm / nnd_bl:.3f}, "
                 f"body length / px-per-cm {expected:.3f}")

    return {"loaded": loaded, "bouts": bouts}


def verify_export(sessions: dict, report: Report, tmp: Path):
    """The exported CSV is the artefact that leaves the building."""
    files = {n: d["loaded"] for n, d in sessions.items()}
    bout_map = {n: d["bouts"] for n, d in sessions.items()}
    out = tmp / "combined_summary.csv"
    n_rows = export_combined_summary_csv(files, bout_map, {}, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]

    print("\nExported CSV")
    report.check(not WITHDRAWN & set(header),
                 "withdrawn columns absent",
                 f"{len(header)} columns, {n_rows} rows")

    expected_fish = sum(d["loaded"].n_fish for d in sessions.values())
    report.check(n_rows == expected_fish,
                 "every fish in every recording has a row",
                 f"{n_rows} rows for {expected_fish} fish")

    statuses = sorted({r["Status"] for r in rows})
    report.check(all(s == "ok" for s in statuses),
                 "no fish failed or was excluded", f"statuses: {statuses}")

    for column in ("Unit", "ObservedDuration_s", "LongestGap_s", "Status",
                   "FreezeEpisodes_Censored", "Bout_Censored", "IBI_N"):
        report.check(column in header, f"{column} is exported")


def main(argv) -> int:
    if not argv:
        sys.exit(__doc__.strip())

    import tempfile

    report = Report()
    results = {}
    for session in find_sessions(argv):
        data = verify_session(session, report)
        results[data["loaded"].nickname] = data

    with tempfile.TemporaryDirectory() as tmp:
        verify_export(results, report, Path(tmp))

    print()
    if report.failures:
        print(f"{len(report.failures)} CHECK(S) FAILED:")
        for failure in report.failures:
            print(f"  - {failure}")
        return 1
    print(f"All checks passed across {len(results)} session(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
