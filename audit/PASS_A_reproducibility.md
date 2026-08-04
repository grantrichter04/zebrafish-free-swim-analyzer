# Pass A — Reproducibility, environment, and doc truth

**Shared rules for every audit pass:**
- **Report, don't fix.** This pass ends in a findings file, not a diff. Fixes are a separate decision I make after reading it.
- **Verify, don't inherit.** The seed suspicions below came from reading the code, not running it. Confirm or refute each one with evidence.
- **Label every finding** `CONFIRMED` (you executed it, or cited an authoritative doc/API reference) or `PLAUSIBLE` (read-only reasoning). Never blur the two.
- **Report what's fine too.** A list of only problems tells me nothing about what I can trust.
- Read `AUDIT_PLAN.md` in the repo root first for the stock-take context.

---

## Goal

A person who is not me can clone this repo, build the environment from one documented command, and run both the GUI and the README's API example — and every factual claim in the docs is true.

This pass is a prerequisite for Pass B: right now nothing in the repo can be executed, so no numerical finding can be confirmed.

## Scope

`requirements.txt`, `README.md`, `run_analyzer.py`, `.gitignore`, packaging (currently absent), and the full third-party import graph.

## Seed findings — verify each

1. `import fish_analyzer` fails on ambient Python 3.11: `ModuleNotFoundError: No module named 'traja'`.
2. `requirements.txt` lines 13–15 are pasted conda shell commands, not requirement specifiers, so `pip install -r requirements.txt` errors. Confirm the exact failure mode and message.
3. Those pasted conda lines pin `pandas=1.5.3` and `numpy=1.26.4`; the pip section above says `pandas>=2.0` and `numpy>=1.24`. Direct contradiction in one file. Determine which the code actually needs — check every pandas and numpy API used, especially anything traja passes through.
4. The Python version is claimed four ways: README says "3.8+", commit `ebb825c` fixes 3.9 compatibility (`Path | None` → `Optional[Path]`), the conda line says `python=3.10`, ambient is 3.11. Find the real floor by checking syntax and API usage across all 23 modules, and the real ceiling (does anything break on 3.12/3.13?).
5. `scikit-image` is imported by `fish_posture_analyzer.py` and `head_detection/*`; `idtrackerai` is imported by three `head_detection` scripts. Neither appears in `requirements.txt`. Map **every** third-party import in the repo against declared dependencies, in both directions — including declared-but-unused.
6. Doc drift to verify claim by claim:
   - README's Project Structure block omits `bout_analysis.py`, `bout_tab.py`, and `inspector_tab.py`.
   - `run_analyzer.py`'s docstring lists a `gui.py` that no longer exists.
   - `run_analyzer.py:66` prints "v2.0" while `__version__` is `2.1.0`.
   - README license section is the literal placeholder `[Add license here]`.
   - **Does the README's API usage example actually run?** It calls `ShoalingCalculator.calculate(fish_list, params)` as if it were a static method taking a fish list; the real signature is `ShoalingCalculator(loaded_file, params).calculate()`. Check it, and check the `process_and_analyze_file` example too.

## Also assess

- **traja.** It's the single reason the package won't import. Only four entry points are used: `traja.length`, `traja.distance`, `TrajaDataFrame.traja.get_derivatives`, `traja.smooth_sg`. Report its last release date, maintenance status, its own dependency footprint, and what replacing those four calls with direct numpy/scipy would take. Give one recommendation — keep or drop — with reasoning. **Do not act on it.**
- **Headless testability.** Can the GUI be smoke-tested without a display? If not, what's the cheapest path to an automated "imports cleanly and constructs" test? The repo has zero tests; I want to know the minimum viable first test.
- **`fish_analyzer/backup/gui.py.backup`** — 3,318 lines of dead pre-refactor GUI, tracked since the first commit. Confirm nothing imports or references it. Recommend disposition.
- **Packaging.** No `pyproject.toml`/`setup.py`, so `fish_analyzer` is importable only from the repo root (hence the `sys.path` insertion in `run_analyzer.py:56`). Assess whether proper packaging is worth it for a lab tool, and what the minimum would be.

## Deliverable

`AUDIT_A_REPRODUCIBILITY.md` in the repo root, containing:

- **(a)** Findings, most severe first, each labelled `CONFIRMED`/`PLAUSIBLE`.
- **(b)** An exact, **tested** environment recipe — the literal commands you ran and verified, including the Python version. Not a plausible-looking recipe; one you executed.
- **(c)** A punch list of false doc claims with `file:line` for each.
- **(d)** A recommendation on traja, and on the minimum first test.

## Constraints

- Do not modify code or docs in this pass.
- A throwaway venv is fine without asking. **Ask me before installing anything into a global or conda base environment.**
- If you cannot verify something without data I haven't provided (e.g. a real `trajectories.npy`), say so explicitly rather than guessing — and tell me exactly what sample file you'd need.
