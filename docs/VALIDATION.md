# What has been validated, and what has not

Written 2026-10-14 for version 2.3.0 (Audit I, finding I8). Update it when a
check is added or a measure changes. Each line says where its evidence is,
so it can be re-checked.

Every measure rests on idtracker.ai's position for each fish in each frame.
The checks below test what the analyzer does with those positions. **No
check yet compares those positions with where the fish really were.**

## Checked

| What | How | Evidence |
|---|---|---|
| The arithmetic of every measure | Tracks with known answers (straight lines, circles, still fish, gaps, jumps), run through the analysis in the test suite: 378 tests, also run on GitHub for every push | `tests/`, especially `test_metric_correctness.py` and `test_resting_and_jumps.py` |
| Scale conversion | The same sessions run in centimetres and in body lengths: every distance in proportion, to 8.9e-16 | [Audit A](audit/AUDIT_A_REPRODUCIBILITY.md), real-data addendum |
| What tracker wobble and gaps do to each measure | Simulated fish with wobble and dropout added | CHANGELOG 2.3.0, "Measures that tracker wobble and dropout cannot move" |
| The smoothing window | Four real sessions: a resting fish's resting time recovered (4.0 of 10 min, against 0.7 unsmoothed); distance changed under 2% | CHANGELOG 2.3.0; `scripts/compare_settings.py` re-runs it |
| Top speed as the 95th percentile | Three real sessions tracked with and without identities from one setup | CHANGELOG 2.3.0 |
| Time near the wall | Two real sessions: outline and positions drawn on a video frame and checked by eye, and the share recomputed in raw pixels | Handover note, 2026-10-08 |
| A fish that barely moves is real, not a tracking fault | Checked by the owner in idtracker.ai's validator | Handover note |
| Withdrawn measures | Turning, burst and bout measures shown to measure tracking noise, and removed | [Audit B](audit/AUDIT_B_CORRECTNESS.md) |

## Not checked

| What | Why it matters | How it could be checked |
|---|---|---|
| **Positions against ground truth** | Speed, distance, straightness and wall time all inherit any error in idtracker.ai's positions. Only idtracker.ai's own estimate of its identity accuracy is on record. | Mark the fish by hand on a few hundred frames sampled across sessions and compare with idtracker.ai's positions: position error, and identity swaps. |
| **The resting speed against this rig's noise floor** | Time resting depends on 0.5 BL/s sitting above the apparent speed of a fish that is not moving. | Film an object that cannot move, track it, and read its apparent speed. Until then, check on the video: the Video Inspector rings every fish counted as resting. |
| **The new camera and lighting** | Segmentation of the badly lit test videos is unresolved: on the worst-lit one, no method tried shows the right number of separate fish in more than 43% of sampled frames (and the count of six fish is itself unconfirmed). | `scripts/score_segmentation.py` scores a lighting set-up before tracking. |
| **Pooled measures (sessions without identities) against per-fish measures** | Both kinds appear in one export. Only top speed has been compared between them. | Track the same videos both ways and compare every measure per tank. |
| **The real-data re-checks of Audit B** | `scripts/verify_on_session.py` re-checks Audit B's findings on real sessions, but crashes on the current measure names. | Repair is Audit I, finding I5. |
| **Any difference between groups** | So far one tank per group: observations, not results. | More tanks; see each export's `STATISTICS.txt`. |
