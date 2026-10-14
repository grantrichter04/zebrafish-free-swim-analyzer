# Methods: what the analyzer does, and why each setting has its value

Every Results export holds a `METHODS.txt`: a draft methods paragraph filled
in with that export's versions, settings, scale and n. This page is the
reasoning behind it, for the author writing the paper and for a reviewer who
asks "why that number?".

Each default below says where its value came from. **Where no published
source has been found, it says so.** Those are the places a reviewer is
most likely to push, and the lab may want to add a citation or its own check.

## Tracking

Fish are tracked with idtracker.ai (Torrents et al. 2026; Romero-Ferrero
et al. 2019), which follows every fish in a group and, when trained to,
keeps each fish's identity across the recording. Everything the analysis
uses comes from idtracker.ai's positions: one point per fish per frame.
`tracking_settings.csv` in each export records how every session was
tracked (thresholds, background, region of interest, whether identity
training was finished early, whether the session was checked in the
validator), and `setups/` keeps each session's setup file as it was used.

## Scale

Distances are in centimetres when a known length was measured on a video
frame, otherwise in body lengths (BL). **One scale is used for every
session.** idtracker.ai's own body length estimate moves with lighting and
threshold (73.2 and 79.7 px for two videos from one rig), so a per-session
body length would make "1 BL" a different distance in each session.

## Settings and their basis

| Setting | Default | Why | Basis |
|---|---|---|---|
| Position smoothing (running mean, within continuous tracking only) | 0.1 s (3 frames at 30 fps) | idtracker.ai's position for a fish wobbles by about a pixel from frame to frame, and that wobble reads as movement. | **Simulation and real data.** In simulation, unsmoothed, a motionless fish read 0.3-0.7 BL/s and was counted as resting 3-64% of the time instead of 100%, and a fish swimming at 0.5 BL/s read 17-67% too far. On four real sessions, 0.1 s is the shortest window that recovers a resting fish's resting time (4.0 of 10 min, against 0.7 unsmoothed). Longer windows recover no more and cut the top speed further. 0.1 s holds while the wobble is under about 0.01 BL; this lab's recordings measured 0.003-0.005 BL. Fish that are small in the frame need 0.17 s. (CHANGELOG 2.3.0) |
| Tracking jumps | steps faster than 25 BL/s are dropped | A faster step is the tracker moving to another fish or to something that is not a fish. | **Reasoning, with a check.** Audit B found exported maximum speeds of 32-321 BL/s from such steps (321 BL/s is about 17 m/s), against adult burst swimming of the order of 15-25 BL/s. *No published source for 25 BL/s is cited yet.* |
| Minimum tracking | fish tracked in under 80% of frames are left out | A fish seen only part of the time, averaged in beside fish seen throughout, shifts the tank mean towards whatever behaviour was easiest to track. | **Lab decision** (2026-10-09). The previous 1% let almost any fish through (Audit B, finding B11). Left-out fish are listed in the export with their tracked share. |
| Top speed | 95th percentile of frame-to-frame speed | A maximum is set by a single tracking error. | **Real data.** On three sessions tracked with and without identities from one setup, the errors left after jump removal moved the 99th percentile up 2.2-4.4% and the 95th 0.7-1.8%; both told the fish in a tank apart equally. (CHANGELOG 2.3.0) |
| Distance | per minute tracked, not total | A total falls with every moment a fish was not tracked, so a group that was harder to track would look less active. Total distance is still exported. | **Reasoning.** |
| Resting speed | 0.5 BL/s | Below this, a fish is counted as resting. | **Partly checked.** Audit B found 4.8% of real frames below it, against a median of 2.47 BL/s. *Not yet checked against this rig's noise floor:* filming an object that cannot move and reading its apparent speed is on the to-do list. Until then, check it on the video: the Video Inspector rings every fish counted as resting. |
| Shortest rest | 1 s, bridging tracking gaps of up to 0.5 s | Shorter still periods are pauses in swimming. A gap in tracking should not split one rest into two, or make a group that is tracked less well look as though it rests less. | **Convention.** Audit B describes about 1 s as the usual minimum for freezing in adults; the previous 0.17 s matched no published criterion. *No paper is cited yet.* |
| Straightness | over 1 s windows, using only seconds faster than 1 BL/s | In a slow second the remaining wobble is most of the path, so a fish that rested a lot would look as though it swam tortuously. | **Simulation and reasoning** (CHANGELOG 2.3.0). The 1 s window is the usual straightness index (net displacement / path length) over a sliding window; the result depends on the window, which is fixed. |
| Wall zone | a band 15% of the tank's shorter side wide, inside the outline | Time near the wall (thigmotaxis) is a common measure of anxiety-like behaviour. | **No published source found.** Audit B notes that published zones are defined either by area (for example an outer annulus of half the arena) or by a distance from the wall (often 1-2 body lengths); 15% of the shorter side is a third convention. Each export gives the zone's share of the tank's area (`WallZoneArea_pct`), the value even use of the tank would give, and `tank_outlines.csv` gives the outline. |

## Unit of analysis

Fish sharing a tank are not independent: each one's speed and position
depend on the others'. **The tank is the experimental unit.** Every measure
is averaged over a tank's analysed fish, and groups are compared on these
tank means. Plots are SuperPlots (Lord et al. 2020): each fish a small point,
each tank's mean a large one, so the spread between fish is shown without
being counted as replication. Statistical guidance, including
minute-by-minute data and mixed models, is in each export's `STATISTICS.txt`.

## What has been validated

See [VALIDATION.md](VALIDATION.md): what has been checked, and what has not.

## References

- Lord SJ, Velle KB, Mullins RD, Fritz-Laylin LK (2020). SuperPlots:
  Communicating reproducibility and variability in cell biology. *J Cell
  Biol* 219(6):e202001064. https://doi.org/10.1083/jcb.202001064
- Romero-Ferrero F, Bergomi MG, Hinz RC, Heras FJH, de Polavieja GG (2019).
  idtracker.ai: tracking all individuals in small or large collectives of
  unmarked animals. *Nat Methods* 16(2):179-182.
  https://doi.org/10.1038/s41592-018-0295-5
- Shenk J, Byttner W, Nambusubramaniyan S, Zoeller A (2021). Traja: A Python
  toolbox for animal trajectory analysis. *J Open Source Softw* 6(63):3202.
  https://doi.org/10.21105/joss.03202
- Torrents J, Costa T, de Polavieja GG (2026). New idtracker.ai rethinks
  multi-animal tracking as a representation learning problem to increase
  accuracy and reduce tracking time. *eLife* 14:RP107602.
  https://doi.org/10.7554/eLife.107602

idtracker.ai's site lists the 2026 eLife paper as its latest publication and
the 2019 Nature Methods paper before it; both are cited here. All four
entries were checked against Crossref on 2026-10-14.
