"""
fish_analyzer/methods.py
========================
METHODS.txt and STATISTICS.txt in the Results export.

METHODS.txt is a draft methods paragraph and its references, filled in with
that export's own versions, settings, scale and n, so the text in a paper
matches the numbers it describes. STATISTICS.txt is guidance on testing
them, with the export's n per group.

It states only what the export knows. Anything else (the statistical test,
the recording set-up) is left in [square brackets] for the author, and
tracking details from sessions loaded without their session folder are left
as placeholders, not guessed.
"""
import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from . import provenance

REFERENCES = [
    "Lord SJ, Velle KB, Mullins RD, Fritz-Laylin LK (2020). SuperPlots: "
    "Communicating reproducibility and variability in cell biology. "
    "J Cell Biol 219(6):e202001064. https://doi.org/10.1083/jcb.202001064",
    "Romero-Ferrero F, Bergomi MG, Hinz RC, Heras FJH, de Polavieja GG (2019). "
    "idtracker.ai: tracking all individuals in small or large collectives of "
    "unmarked animals. Nat Methods 16(2):179-182. "
    "https://doi.org/10.1038/s41592-018-0295-5",
    "Shenk J, Byttner W, Nambusubramaniyan S, Zoeller A (2021). Traja: A "
    "Python toolbox for animal trajectory analysis. J Open Source Softw "
    "6(63):3202. https://doi.org/10.21105/joss.03202",
    "Torrents J, Costa T, de Polavieja GG (2026). New idtracker.ai rethinks "
    "multi-animal tracking as a representation learning problem to increase "
    "accuracy and reduce tracking time. eLife 14:RP107602. "
    "https://doi.org/10.7554/eLife.107602",
]

HEADER = """\
Draft methods text, exported {when}
by Free Swim Analyzer {version}

Edit before use. It describes the sessions and settings in this export;
words in [square brackets] are for you to fill in. Every number here is
also in the CSV files beside it. docs/METHODS.md in the software's
repository explains why each setting has the value it has.
"""


def _join(values) -> str:
    """Distinct values, as written: "6" or "5, 6"."""
    return ", ".join(f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v)
                     for v in dict.fromkeys(values))


def _sessions(count: int, total: int) -> str:
    return f"{count} of {total} session{'s' if total != 1 else ''}"


def _tracking(per_tank: pd.DataFrame, tracking: pd.DataFrame) -> str:
    known = tracking.dropna(subset=["IdtrackerVersion"]) if (
        "IdtrackerVersion" in tracking) else tracking.iloc[0:0]
    version = (_join(known["IdtrackerVersion"]) if not known.empty
               else "[idtracker.ai version]")
    fps = _join(per_tank["FrameRate"]) if not per_tank.empty else "[frame rate]"
    fish = _join(per_tank["FishInTank"]) if not per_tank.empty else "[n]"
    minutes = _join(per_tank["Minutes"]) if not per_tank.empty else "[length]"
    text = [
        f"Fish were filmed in groups of {fish} per tank at {fps} frames per "
        f"second, in recordings of {minutes} min [camera, lighting, tank "
        f"size, acclimation and time of day]. Each fish was tracked with idtracker.ai {version} (Torrents "
        "et al., 2026; Romero-Ferrero et al., 2019)."]
    pooled = int((per_tank["Tracking"] == "no identities").sum()) if not per_tank.empty else 0
    if pooled:
        text.append(
            f"{_sessions(pooled, len(per_tank))} were tracked without "
            "identities; their measures were made from all tracks together, "
            "as values for the tank.")
    if not known.empty:
        with_ids = known[known["IdentityTraining"] != "none (no identities)"]
        early = int((with_ids["IdentityTraining"] == "finished early").sum())
        if early:
            text.append(
                f"Identity training was finished early for "
                f"{_sessions(early, len(with_ids))}, keeping the best model "
                "found so far.")
        reviewed = int(known["ReviewedOn"].notna().sum())
        text.append(
            f"{_sessions(reviewed, len(known))} were checked, and identity "
            "errors corrected, in idtracker.ai's validator." if reviewed else
            "No session was checked in idtracker.ai's validator.")
    text.append("The thresholds, background and region of interest used for "
                "each session are in tracking_settings.csv.")
    return " ".join(text)


def _scale(per_tank: pd.DataFrame) -> str:
    if per_tank.empty:
        return "[Scale.]"
    unit = _join(per_tank["Unit"])
    pixels = _join(per_tank["PixelsPerUnit"])
    if per_tank["PixelsPerUnit"].nunique() > 1 or per_tank["Unit"].nunique() > 1:
        # The app puts every session on one scale unless told otherwise, so
        # this is worth a second look, not a sentence that hides it.
        return (f"[Check: sessions were measured on different scales "
                f"({pixels} pixels per {unit}), so their distances and speeds "
                "are not directly comparable.]")
    if unit == "BL":
        return ("Distances are in body lengths (BL), using one body length of "
                f"{pixels} pixels for every session, from idtracker.ai's "
                "estimate of the fishes' median body length.")
    if unit == "cm":
        return ("Distances are in centimetres, at a scale of "
                f"{pixels} pixels per cm measured on a video frame "
                "[from what: ruler, tank width].")
    return f"Distances are in {unit}, at {pixels} pixels per {unit}."


def _processing(params, unit: str, left_out: int) -> str:
    if params is None:
        return "[Processing settings.]"
    text = []
    if params.smoothing_seconds:
        text.append(
            "Positions were smoothed with a running mean over "
            f"{params.smoothing_seconds:g} s, within stretches of continuous "
            "tracking only, to remove the tracker's frame-to-frame jitter.")
    if params.max_plausible_speed:
        text.append(
            f"A frame-to-frame step faster than {params.max_plausible_speed:g} "
            f"{unit}/s was taken as a tracking error (an identity swap or a jump "
            "to another object), and the frame it landed on was treated as "
            "untracked.")
    text.append(
        "Fish tracked in less than "
        f"{params.min_valid_percentage * 100:g}% of frames were left out "
        f"({left_out} fish in this export).")
    return " ".join(text)


def _measures(params, unit: str, wall_pct: float) -> str:
    if params is None:
        return "[Measures.]"
    return (
        "For each fish we measured the distance swum per minute it was "
        "tracked; the median and 95th percentile of its frame-to-frame speed; "
        "path straightness, the straight-line distance over each one-second "
        "window divided by the distance swum in it, averaged over the seconds "
        f"in which the fish swam faster than {params.straightness_min_speed:g} "
        f"{unit}/s; time near the wall, the share of its time within a zone "
        f"along the walls {wall_pct:g}% of the tank's shorter side wide; and "
        "time resting, the share of tracked time it spent slower than "
        f"{params.rest_speed_threshold:g} {unit}/s for at least "
        f"{params.min_freeze_seconds:g} s (a gap in tracking of up to "
        f"{params.freeze_bridge_seconds:g} s does not end a rest when the fish "
        "is in the same place on both sides).")


def _statistics(per_tank: pd.DataFrame) -> str:
    counts = (per_tank.groupby("Group", sort=False).size() if not per_tank.empty
              else pd.Series(dtype=int))
    n = ", ".join(f"{group} {count}" for group, count in counts.items())
    return (
        "Fish sharing a tank are not independent, so the tank was the "
        "experimental unit: each measure was averaged over the analysed fish "
        "in a tank, and groups were compared on these tank means (tanks per "
        f"group: {n}) [test, and correction for the number of measures "
        "tested; STATISTICS.txt has guidance]. Results are plotted as SuperPlots (Lord et al., 2020), each "
        "fish a small point and each tank's mean a large one.")


STATISTICS = """\
Statistics for these results, exported {when}
by Free Swim Analyzer {version}

Guidance, not a prescription: check it with a statistician for your design.

1. THE UNIT IS THE TANK
   Fish sharing a tank are not independent: each one's speed and position
   depend on the others'. Test on results_per_tank.csv, one value per tank.
   {n_line}{one_tank}
   Two groups: Welch's t-test on the tank values. More groups: Welch's
   one-way ANOVA. Show every tank (the SuperPlot does), since with few tanks
   a reader should see them all.
   Never run a plain t-test or ANOVA on the fish rows: that counts each fish
   as an independent replicate, which they are not.

2. CHOOSE ONE PRIMARY MEASURE BEFORE LOOKING
   Six measures tested at p < 0.05 make a false positive likely. These
   measures were chosen to ask "is one group more active than another?",
   which DistancePerTrackedMin answers: make it the primary outcome.
   Treat MedianSpeed, Speed95, Straightness, NearWall and RestingTime as
   secondary: correct them for multiple testing (Holm's method is simple and
   not too strict) or report them descriptively.

3. MINUTE BY MINUTE IS REPEATED MEASURES
   Each tank is measured in every minute, so its minutes are
   repeated measures of one tank, not independent values.
   results_by_minute_per_tank.csv has one row per tank per minute,
   each value the mean of the tank's fish in that minute.
   - Prism: the by_minute tables in prism/ -> two-way repeated-measures
     ANOVA, or Prism's mixed-effects model if some minutes are missing.
   - jamovi or R: the long file with GAMLj's Mixed Model (below):
     dependent = the measure, factors = Group and Minute, cluster = Session.
   The usual question is the Group x Minute interaction: whether the groups
   change differently over the recording, for example as they settle.

4. MIXED MODELS ON THE FISH ROWS
   A mixed model on results_per_fish.csv with a random intercept for each
   tank is a valid alternative to tank means. In jamovi: install GAMLj from
   the jamovi library, then Linear Models > Mixed Model; dependent = the
   measure, factor = Group, cluster = Session, random effect = Intercept
   | Session. Use the rows with Status "ok" only.
   It helps when:
   - tanks hold different numbers of analysed fish (FishAnalysed in
     results_per_tank.csv: fish tracked under the minimum are left out), or
   - you want a covariate measured on each fish.
   It does not help, or misleads, when:
   - every tank has the same number of fish: it then gives essentially the
     same group test as the t-test on tank means;
   - there are only two or three tanks per group: the variation between
     tanks is poorly estimated and often comes out as zero (a "singular fit"
     warning). The model then treats fish as independent, the very mistake
     tank means avoid. Report the tank-means test instead;
   - sessions were tracked without identities: they have one row per tank,
     so there are no fish to model.
   A covariate that belongs to the tank (sex ratio, day, rig) works in either
   analysis as one value per tank.

5. HOW MANY TANKS
   More fish in a tank do not add replication; more tanks do. To size an
   experiment, take the standard deviation of the tank means of the primary
   measure from a pilot and run a power calculation on it (G*Power, or the
   power tools in Prism or jamovi).
"""


def statistics_text(per_tank: pd.DataFrame) -> str:
    """The text of STATISTICS.txt."""
    counts = (per_tank.groupby("Group", sort=False).size() if not per_tank.empty
              else pd.Series(dtype=int))
    n_line = ("Tanks per group: " + ", ".join(
        f"{group} {count}" for group, count in counts.items()) + "."
        if len(counts) else "No tanks were analysed.")
    single = [str(group) for group, count in counts.items() if count < 2]
    one_tank = ("".join(f"\n   {group} has 1 tank." for group in single)
                + "\n   With one tank a group can be described, not tested."
                if single else "")
    return STATISTICS.format(
        when=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        version=provenance.analyzer_version(), n_line=n_line, one_tank=one_tank)


def methods_text(loaded_files: Dict, per_tank: pd.DataFrame,
                 tracking: pd.DataFrame, wall_zone_pct: float = 15.0) -> str:
    """The text of METHODS.txt."""
    params = next((getattr(l, "processing_params", None) for l in loaded_files.values()
                   if l.processed_data and getattr(l, "processing_params", None)), None)
    unit = _join(per_tank["Unit"]) if not per_tank.empty else "[unit]"
    left_out = int(per_tank["FishLeftOut"].sum()) if not per_tank.empty else 0
    version = provenance.analyzer_version()
    sections: List[str] = [
        HEADER.format(when=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                      version=version).rstrip(),
        "Tracking\n" + _tracking(per_tank, tracking),
        "Processing\nTrajectories were analysed with Free Swim Analyzer "
        f"{version} [citation or DOI], written in Python and using traja "
        f"(Shenk et al., 2021). {_scale(per_tank)} "
        + _processing(params, unit, left_out),
        "Measures\n" + _measures(params, unit, wall_zone_pct),
        "Statistics\n" + _statistics(per_tank),
        "References\n" + "\n".join(REFERENCES),
    ]
    return "\n\n".join(sections) + "\n"
