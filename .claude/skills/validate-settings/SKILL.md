---
name: validate-settings
description: Check the 2026-10-09 analysis settings (smoothing, 80% tracking minimum, 1 s freezes, straightness speed gate) on real idtracker.ai sessions before branch ccr-922dbe18-vx83fk is merged. Use when the owner asks to validate or check the new settings on real data.
argument-hint: "<folder of session_ folders>"
---

Validate the new analysis settings on real recordings, following
`docs/validate-2026-10-09.md` exactly. The sessions are in: $ARGUMENTS

If no folder was given, ask for it before doing anything else.

1. Make sure branch `ccr-922dbe18-vx83fk` is checked out
   (`git fetch origin ccr-922dbe18-vx83fk`, then `git checkout ccr-922dbe18-vx83fk`).
   If there are uncommitted changes, stop and ask.
2. Use the `freeswim` conda environment. Run `python -m pytest -q` first.
3. Run `python scripts/compare_settings.py "<folder>" --csv compare.csv` and
   `python scripts/verify_on_session.py "<folder>"`. Ask the owner which
   sessions belong to which group, and whether they analyse in BL with one
   body length (then pass `--body-length <px>`) or in cm.
4. Work through every check in the doc, using the numbers, not impressions.
   For fish 6 of the treated tank in the first experiment, look at the
   still stretch in the video if one is available.
5. Report as the doc describes: a table per session, excluded fish, the fish
   6 result, and per setting "keep" or "change to X because Y".

Do not push, merge, or change the analysis code. Leave `compare.csv`
uncommitted. If a setting looks wrong, say what you would change and wait
for the owner.
