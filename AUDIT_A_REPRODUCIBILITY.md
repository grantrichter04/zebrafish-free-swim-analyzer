# Audit A — Reproducibility, environment, and doc truth

Executed 2026-07-30/31 against the repo at `17fe96d`, working tree clean apart from the audit files.

**Labels:** `CONFIRMED` = I executed it or read it out of package metadata / source. `PLAUSIBLE` = reasoning from reading only.

> **Status update (2026-07-31): most of this has since been fixed.** After the pass was
> delivered, findings A1–A6, A9, A12 and punch-list items 1–14 were applied and verified
> — see the *Fix status* table at the end. The findings text below is the pre-fix snapshot
> and is left unedited so the evidence stays readable. **Still open: A7's real-data caveat,
> A8's decision, A10, A11, and the licence.**

Everything below was run in throwaway venvs under the session scratchpad. **Nothing was installed into your conda base or into any existing conda environment**, and no file in the repo was modified. The one existing environment I touched — `traja_fish_analysis` — was used read-only, to run scripts.

---

## The short version

The environment problem is real but smaller than `AUDIT_PLAN.md` assumed, and one previously-unknown blocker is bigger.

- `pip install -r requirements.txt` fails outright. `CONFIRMED`, exact message below.
- Even after fixing that, `pip install traja` **still leaves you with a broken `import traja`** — traja 25.0.1 has an undeclared hard dependency on `scikit-learn`. `CONFIRMED`. This is not in your requirements file and not in the README's pip list. It is why a "fixed" requirements.txt would still fail for the next person.
- The documented README API example does not run. `CONFIRMED`, with the exact `TypeError`.
- `process_and_analyze_file` **crashes** — not degrades, crashes — the moment stdout is a pipe or a redirect under a non-UTF-8 locale, because `processing.py:187` prints `✓`. The `except` handler at `:191` prints `✗` and raises a *second* `UnicodeEncodeError`, destroying the original traceback. `CONFIRMED`.
- The version-skew fear is **unfounded**. I ran the full pipeline on traja 22.0.0 / pandas 1.5.3 / numpy 1.26.4 / Python 3.9 and on traja 25.0.1 / pandas 3.0.5 / numpy 2.5.1 / Python 3.12 and diffed every scalar metric and the exported CSV. **Zero differences.** `CONFIRMED`.
- The real Python range is much wider than any of the four claims: it runs on **3.9, 3.12 and 3.13**, all verified end to end. `CONFIRMED`.
- traja is **replaceable in about 15 lines**, and I verified the replacement is numerically exact. `CONFIRMED`.

---

## (a) Findings

### A1 — `pip install -r requirements.txt` fails. `CONFIRMED`

Lines 15–17 of `requirements.txt` are pasted conda shell commands. (`AUDIT_PLAN.md` and `PASS_A` both say lines 13–15; the actual lines are **15–17** — 13 and 14 are blank.)

```
$ pip install -r requirements.txt
Usage: __main__.py [options]

ERROR: Invalid requirement: conda create -n fishanalyzer python=3.10 "numpy=1.26.4" "pandas=1.5.3" matplotlib scipy shapely scikit-learn opencv -c conda-forge
__main__.py: error: no such option: -n
```

pip parses the whole file before installing anything, so this fails clean — no partial environment is left behind. Exit code 1.

### A2 — `pip install traja` produces a broken `import traja`. `CONFIRMED`. New finding.

This is the one that would still bite after fixing A1.

```
>>> import traja
  File ".../traja/dataset/dataset.py", line 21, in <module>
    import sklearn
ModuleNotFoundError: No module named 'sklearn'
```

`traja/__init__.py:3` does `from traja import dataset, models` unconditionally, and `dataset.py:21` imports `sklearn` unconditionally — but traja 25.0.1's `Requires-Dist` is only `matplotlib, pandas, numpy, shapely, scipy, tzlocal`. `scikit-learn` appears nowhere. This is a packaging bug in traja, not in your code, but it lands on you: **`scikit-learn` is a hard requirement of this project and is not declared anywhere in `requirements.txt`.**

It is present in the conda line (15) — which is presumably why this has never been noticed. Anyone following the pip path in the README (lines 66, 58) gets a package that cannot be imported.

Related, cosmetic: every `import traja` emits `UserWarning: PyTorch not installed. Deep learning models not available.` from `traja/models/__init__.py:13`. Harmless, but it is the first thing printed on every run and it will appear in the GUI status bar.

### A3 — the analysis core crashes on a redirected stdout. `CONFIRMED`. New finding.

```
  File "fish_analyzer/processing.py", line 187, in process_all_fish
    print(f"  Fish {fish_idx}: \u2713 ({fish_traj.valid_percentage:.1%} valid data)")
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 10

During handling of the above exception, another exception occurred:
  File "fish_analyzer/processing.py", line 191, in process_all_fish
    print(f"  Fish {fish_idx}: \u2717 (error: {e})")
UnicodeEncodeError: 'charmap' codec can't encode character '\u2717' in position 10
```

Reproduced on Python 3.12 with `sys.stdout.encoding == 'cp1252'`, which is what you get whenever stdout is a pipe or a file redirect on this machine — `python run_analyzer.py > log.txt`, a batch script, an IDE run window, or anything that captures output. In an interactive console with `chcp 65001` it does not fire. Setting `PYTHONIOENCODING=utf-8` also suppresses it, which is how I got past it for every other test in this report.

Two things make this worse than a cosmetic annoyance:

1. **The error handler is the same bug.** The `except Exception` at `:190` prints `✗`, which raises again. A genuine per-fish failure would therefore be reported as a `UnicodeEncodeError` from the handler, with the real cause discarded. `AUDIT_PLAN.md`'s "failures become NaN silently" concern is worse than stated: in this path, failures become a misleading traceback.
2. **The GUI is not immune.** `GUILogRedirector.write` (`gui/base.py:38-41`) forwards every write to `self.original_stdout`, so a GUI launched with redirected output hits the same exception inside a tkinter callback. `PLAUSIBLE` — I confirmed the forwarding by reading, and confirmed the encode failure independently, but did not run the GUI under a redirect.

Scope: **12 `print()` lines contain non-ASCII characters, across 6 files** — `processing.py` (3), `spatial.py` (3), `fish_posture_analyzer.py` (3), `shoaling.py` (1), `gui/base.py` (1), `head_detection/validation_video.py` (1). Characters involved include `✓ ✗ ⚠ ± → ²`. `CONFIRMED` by script.

### A4 — the README's API example does not run. `CONFIRMED`

I ran the README block (lines 88–107) verbatim against a synthetic idtracker.ai-shaped file. Everything up to the last two lines works. The documented shoaling call does not:

```
  File "readme_example.py", line 29, in <module>
    results = ShoalingCalculator.calculate(fish_list, params)
TypeError: ShoalingCalculator.calculate() takes 1 positional argument but 2 were given
```

`ShoalingCalculator.calculate` is an instance method taking only `self`; the constructor takes `(loaded_file, params)`. The correct form is `ShoalingCalculator(loaded_file, params).calculate()` — which is exactly what `run_analyzer.py`'s docstring (lines 44–47) already shows. **The docstring is right and the README is wrong**, and they disagree with each other.

`process_and_analyze_file(loaded_file)` in the same example is correct and works. `CONFIRMED`.

Note the shape of the error: because `ShoalingCalculator` is a normal class, `fish_list` silently binds to `self` and only the arity check catches it. A one-argument call would have failed much later and more confusingly.

### A5 — `requirements.txt` contradicts itself, and the README contradicts both. `CONFIRMED`

| Claim | Where |
|---|---|
| `pandas>=2.0`, `numpy>=1.24` | `requirements.txt:2-3` |
| `"pandas=1.5.3"`, `"numpy=1.26.4"` | `requirements.txt:15` |
| conda set **includes** opencv, pins numpy/pandas | `requirements.txt:15` |
| conda set **excludes** opencv, pins nothing, installs opencv by pip | `README.md:54-58` |
| `scikit-learn` present | `requirements.txt:15`, `README.md:54` |
| `scikit-learn` absent | `requirements.txt` pip section, `README.md:66`, `run_analyzer.py:22` |

**Which does the code need?** Neither pin is required. `CONFIRMED` by running the full pipeline on both `pandas 1.5.3 / numpy 1.26.4` and `pandas 3.0.5 / numpy 2.5.1` — identical results (see A7). The `pandas=1.5.3` pin is the more misleading of the two: it is two major versions behind, and nothing in the repo or in traja needs it.

`PLAUSIBLE`: the conda block is a transcription of how the `traja_fish_analysis` environment on this machine was actually built (Python 3.9.23, numpy 1.26.4, pandas 1.5.3, traja 22.0.0 — I read its package list). That is the environment the code was developed against. It works. It is just not what the pip section describes.

### A6 — `traja>=0.6` is a meaningless pin that selects a major-version jump. `CONFIRMED`

Verified against the live index, not the docs:

```
$ pip index versions traja
Available versions: 25.0.1, 25.0.0, 22.0.0, 0.2.8, 0.2.7, 0.2.6, ... 0.0.1

$ pip install "traja==0.6"
ERROR: No matching distribution found for traja==0.6
```

**There has never been a traja 0.6.** The highest 0.x release is 0.2.8. So `traja>=0.6` cannot be satisfied by any 0.x release and resolves to **25.0.1** — released 2025-10-23 after a three-year gap. Your working environment has 22.0.0.

This confirms the corresponding claim in `AUDIT_H_APPROACH.md` §(a), first-hand.

### A7 — the version skew does not change any number. `CONFIRMED`. This is the reassuring finding.

I ran the entire pipeline — load → process → all individual metrics → shoaling → bout detection → thigmotaxis → CSV export — in two environments and diffed the results as JSON at 1e-9 tolerance.

| | Environment 1 | Environment 2 |
|---|---|---|
| Python | 3.9.23 | 3.12.5 |
| traja | 22.0.0 | 25.0.1 |
| pandas | 1.5.3 | 3.0.5 |
| numpy | 1.26.4 | 2.5.1 |
| scipy | 1.13.1 | 1.18.0 |

**Result: 0 differences across every scalar metric, and the exported CSVs are byte-identical.**

Repeated with `apply_smoothing=True, smoothing_window=11` — which is the only path that exercises `traja.smooth_sg` — plus direct comparison of `traja.length`, `traja.distance` and `get_derivatives` output columns. **Again 0 differences**, and `get_derivatives` returns the same six columns in the same order in both versions.

Caveat, stated plainly: this is a synthetic 3000-frame × 4-fish random walk with 2% dropouts, not your data. It exercises the code paths but not every numeric edge case. `PLAUSIBLE` that a real session behaves the same; the traja functions involved are simple enough (see A12) that I would be surprised otherwise. **This retires `AUDIT_H_APPROACH.md`'s hypothesis that traja version skew is a candidate cause for Pass B's "two independent speed pipelines" disagreement.** If those two pipelines disagree, it is your code, not traja.

### A8 — the real Python range is 3.9 – 3.13, wider than any of the four claims. `CONFIRMED`

| Claim | Source | Verdict |
|---|---|---|
| "Python 3.8+" | `README.md:63` | Unverifiable-as-stated; see below |
| 3.9 | commit `ebb825c` | Works — full pipeline verified |
| `python=3.10` | `requirements.txt:15`, `README.md:54` | Not tested directly; nothing suggests it fails |
| 3.11 ambient | your conda base | Not tested (traja absent, and I did not install into base) |
| — | 3.12.5 | **Works** — full pipeline verified |
| — | 3.13.5 | **Works** — full pipeline verified |

- **Syntax floor is genuinely 3.8.** All 24 `.py` files parse with `ast.parse(feature_version=(3,8))`. `CONFIRMED`. Commit `ebb825c` did its job: there is no remaining PEP 604 (`X | None`) or PEP 585 (`list[int]`) usage anywhere. `CONFIRMED` by grep.
- **The practical floor is set by dependencies, not by your code.** Current wheels of numpy (≥3.12), scipy (≥3.12), pandas (≥3.11), matplotlib (≥3.11) and scikit-learn (≥3.11) all require newer Pythons; on 3.9 you get the older pinned wheels, which is what `traja_fish_analysis` has and it works fine. `CONFIRMED` from wheel metadata.
- **No ceiling found.** 3.13 works. `PLAUSIBLE` that 3.14 does too; untested.

The README's "3.8+" is not *false* about the code, but it is not actionable either — nobody can build this on 3.8 today.

### A9 — `.gitignore` does not protect against committing data. `CONFIRMED`

`AUDIT_PLAN.md:57` states that `.gitignore` excludes `*.npy`, `*.avi`, and `session_*/`. **It excludes only `*.npy`.** Verified:

```
$ git check-ignore -v foo.npy
.gitignore:15:*.npy     foo.npy

$ git check-ignore -v test.avi session_foo/x.txt
(no match, exit 1)
```

So a session folder dropped into the repo — the natural thing to do — shows up in `git status` with its `.avi` files, its `list_of_blobs.pickle`, and its individual crops. Nothing stops `git add -A` from committing gigabytes. `.gitignore` also does not ignore `venv/`, `.venv/`, or `env/`.

Minor: the file begins with a UTF-8 BOM. Git 2.53 handles it correctly (`__pycache__/` on line 1 does match), so this is cosmetic, not broken. `CONFIRMED`.

### A10 — the `head_detection/` scripts cannot be imported at all. `CONFIRMED`

This matters for Pass A because it means they can never be covered by any automated test, and it is a hard blocker for Pass E.

- Three of four hardcode a personal absolute path at module level: `C:\Users\grich\Macquarie University\Morsch Group - Documents\Grant R\Fish videos\...` in `generate_mask_video.py:6-10`, `generate_individual_mask_videos.py:6-10`, `head_detection_test.py:17-21`. Confirms `AUDIT_PLAN.md`'s claim.
- **All four execute their work at import time.** No `if __name__ == "__main__":` guard in any of them. `CONFIRMED` — the only two files in the repo with a main guard are `run_analyzer.py:80` and `fish_posture_analyzer.py:390`.
- `head_detection_test.py:29` calls `os.makedirs()` at module scope — importing it creates a directory as a side effect.
- `validation_video.py:23-26` **opens a tkinter directory-picker dialog at module scope**. Importing this file blocks on a GUI dialog.
- All four `import idtrackerai`, which is not in `requirements.txt` and is a heavy install. You have it at `~/.conda/envs/idtrackerai` (idtrackerai 6.0.10, Python 3.13.5) — a *different* environment from `traja_fish_analysis`. `CONFIRMED`. Today, no single environment in this repo can run both halves of the project.

### A11 — packaging: `fish_analyzer` is importable only from the repo root. `CONFIRMED`

```
$ cd /some/other/dir && python -c "import fish_analyzer"
ModuleNotFoundError: No module named 'fish_analyzer'
```

`run_analyzer.py:55-57` works around this with a `sys.path.insert`, and it does work — I ran `run_analyzer.py` from a foreign working directory with `mainloop` stubbed, and it reached the mainloop cleanly. `CONFIRMED`. But the README's API example (which has no such insert) only runs with the repo root as cwd, and that is not stated anywhere.

### A12 — `fish_analyzer/backup/gui.py.backup` is genuinely dead. `CONFIRMED`

Tracked by git (`git ls-files` confirms), 3,318 lines, and **zero references** anywhere in the repo — no import, no string mention, no doc reference. `CONFIRMED` by grep across `*.py`, `*.md`, `*.txt`. The `.backup` extension also means it is invisible to every Python tool: linters, `compileall`, and `grep --include=*.py` all skip it, so it cannot even rot loudly.

---

## (b) Tested environment recipe

These are the literal commands I ran, in this order, in a scratch directory. The resulting environment ran the full pipeline plus a GUI construction test.

```bash
"$LOCALAPPDATA/Programs/Python/Python312/python.exe" -m venv venv312
venv312/Scripts/python.exe -m pip install --upgrade pip
venv312/Scripts/python.exe -m pip install "numpy>=1.24" "pandas>=2.0" "matplotlib>=3.7" "scipy>=1.10" "traja>=0.6" "shapely>=2.0" "opencv-python>=4.8" scikit-learn
```

(`py -3.12 -m venv venv312` resolves to the same interpreter here — `py -0p` lists 3.12 as the only registered launcher entry.)

Resolved to: Python 3.12.5, numpy 2.5.1, pandas 3.0.5, matplotlib 3.11.1, scipy 1.18.0, traja 25.0.1, shapely 2.1.2, opencv-python 5.0.0.93, scikit-learn 1.9.0, pillow 12.3.0 (transitively, via matplotlib).

Two things to note about that command:

1. **`scikit-learn` is not optional** and must be added — without it `import traja` fails (A2).
2. Set `PYTHONIOENCODING=utf-8` before running anything that captures output, or you hit A3.

The same command against `python 3.13.5 -m venv` also succeeded and produced identical results, so the recipe is not 3.12-specific.

Verified in this environment:

| Check | Result |
|---|---|
| `import fish_analyzer` | OK, `__version__` 2.1.0, `GUI_AVAILABLE=True` |
| `SHAPELY_AVAILABLE` / `CV2_AVAILABLE` | both `True` |
| load → `process_and_analyze_file` | 4/4 fish, all metrics finite |
| `ShoalingCalculator(...).calculate()` | NND / IID / hull all computed |
| `analyze_bouts_for_file` | bouts detected per fish |
| `ThigmotaxisCalculator(...).calculate()` | ran, incl. the >5% out-of-arena warning |
| `export_individual_metrics_csv` | wrote a valid CSV |
| `EnhancedFishAnalyzer()` construct + destroy | OK (see (d)) |
| `run_analyzer.py` from a foreign cwd | reached mainloop |

**What I could not verify without your data:** everything above used a synthetic 3000×4 idtracker.ai-shaped `.npy` I generated. To close this out I need **one real `trajectories.npy`** — that is the whole ask for Pass A. It would confirm the real metadata keys (I assumed `width`, `height`, `frames_per_second`, `body_length`, `estimated_accuracy`, `fraction_identified`, `identities_labels`, `version` from `data_structures.py:46-54`), the actual dropout patterns, and the calibrate-in-cm path. Nothing else in this report depends on it.

**A note on the conda recipe:** I did **not** test `conda create -n fishanalyzer ...` from `README.md:54` or `requirements.txt:15`, because that would create an environment outside the scratchpad and the prompt told me to ask first. I can run it if you want. `PLAUSIBLE` that the `requirements.txt:15` variant works, since it is a transcript of how `traja_fish_analysis` was built; the `README.md:54` variant omits opencv from conda and adds it by pip, which should also work.

---

## (c) False doc claims — punch list

| # | Location | Claim | Reality |
|---|---|---|---|
| 1 | `README.md:105` | `ShoalingCalculator.calculate(fish_list, params)` | `TypeError`. Correct form is `ShoalingCalculator(loaded_file, params).calculate()` — as `run_analyzer.py:44-47` already shows |
| 2 | `README.md:66` · `README.md:58` · `run_analyzer.py:22` | pip lists omit `scikit-learn` | Hard requirement — `import traja` fails without it |
| 3 | `requirements.txt:6` | `traja>=0.6` | No 0.6 exists; resolves to 25.0.1 |
| 4 | `requirements.txt:2-3` vs `:15` | `pandas>=2.0` / `numpy>=1.24` vs `pandas=1.5.3` / `numpy=1.26.4` | Direct contradiction; neither pin is needed |
| 5 | `requirements.txt:15-17` | conda shell commands in a pip requirements file | Makes `pip install -r` fail |
| 6 | `README.md:63` | "Python 3.8+" | Syntactically true, practically unbuildable. Verified working: 3.9, 3.12, 3.13 |
| 7 | `README.md:26-44` | Project Structure block | Omits `bout_analysis.py`, `gui/bout_tab.py`, `gui/inspector_tab.py`, and `backup/` |
| 8 | `run_analyzer.py:19` | lists `gui.py` in the package structure | Does not exist; it is the `gui/` subpackage |
| 9 | `run_analyzer.py:66` | prints `v2.0` | `__version__` is `2.1.0` |
| 10 | `README.md:178` | `[Add license here]` | Placeholder. Matters more than it looks — `AUDIT_H_APPROACH.md` recommends adopting GPL-3.0 packages, and you cannot reason about that without knowing your own licence |
| 11 | `README.md:88-107` | API example given without context | Only runs with cwd = repo root (A11) |
| 12 | `README.md:54` vs `requirements.txt:15` | two different conda recipes | Different package sets, different pinning |
| 13 | `AUDIT_PLAN.md:57` | `.gitignore` excludes `*.npy`, `*.avi`, `session_*/` | Only `*.npy` |
| 14 | `AUDIT_PLAN.md:91` · `PASS_A:25` | "lines 13–15" | The conda lines are 15–17 |

Items 13 and 14 are in the audit documents rather than the repo docs, but they are false claims that would mislead the other passes, so they belong on the list.

---

## (d) Recommendations

### traja: **drop it**, but not urgently, and not for the reason the plan assumed

The case for dropping is not installability and not numerical risk — A7 shows the version jump changes nothing. It is that the dependency is doing almost no work and costs more than it returns.

**What you actually use** (`CONFIRMED`, 4 sites, all in `processing.py`):

| Call | Site | What traja does |
|---|---|---|
| `traja.smooth_sg(trj, w, p)` | `:233` | 4 lines: `scipy.signal.savgol_filter` on `x` and `y`, then `fill_in_traj` |
| `trj.traja.get_derivatives()` | `:309` | ~10 lines: `d/dt` of step length, then `d/dt` again |
| `traja.length(trj)` | `:358` | `sum` of per-step euclidean displacement |
| `traja.distance(trj)` | `:362` | `norm(last − first)` |

**I wrote the replacement and verified it is exact.** ~15 lines of numpy/scipy, compared against traja on all four fish:

```
fish     length   distance                  speed                  accel    smooth_sg
   0      EXACT      EXACT                  EXACT                  EXACT        EXACT
   1      EXACT      EXACT                  EXACT                  EXACT        EXACT
   2      EXACT      EXACT                  EXACT                  EXACT        EXACT
   3      EXACT      EXACT                  EXACT                  EXACT        EXACT
```

(`EXACT` = every element agrees to <1e-9, with NaN positions matching. `CONFIRMED`.)

**What dropping buys:**

- Deletes the `scikit-learn` requirement (A2) and the PyTorch warning on every startup.
- Deletes `TrajaDataFrame`, which is the reason `pandas` is in the dependency list at all — `processing.py` is the only module that imports pandas, and it does so only to build the frame traja wants. Dropping traja plausibly drops pandas too, taking the dependency set down to numpy + scipy + matplotlib (+ shapely, opencv optional).
- Makes the speed pipeline visible. Pass B is going to have to read `get_derivatives` anyway to compare it against `bout_analysis.py`'s `np.diff`; owning those ten lines makes that comparison a code review instead of an archaeology exercise.

**Two things dropping would expose, which is an argument *for* doing it:**

- `traja.distance` uses `trj.iloc[0]` and `trj.iloc[-1]` raw. If the first or last frame is a tracking dropout, `net_displacement` is silently `NaN`. `CONFIRMED` by reading `trajectory.py:169-177`. Real sessions frequently start or end with a dropout.
- `smooth_sg` calls `savgol_filter` directly, which cannot handle NaN. `processing.py:224-243` already works around this with interpolate-smooth-restore, and the comment there shows someone found this the hard way. That workaround is the interesting code; the traja call in the middle is not.

**Sequencing:** do this *after* Pass B, not before. Right now traja 22.0.0 is your only independent implementation of these four quantities, and A7 makes it a usable reference. Replace the calls, keep traja installed as a dev-only dependency, and assert exactness in a test — then remove it. Meanwhile, pin it: `traja==25.0.1` (or `==22.0.0` if you want to match `traja_fish_analysis` exactly). `traja>=0.6` should not survive the week.

### Minimum first test: **seven tests, 3.7 seconds, and two of them fail today**

Headless testability is better than expected. `EnhancedFishAnalyzer()` constructs and destroys cleanly without entering `mainloop` — the whole mixin stack, all six tabs, every matplotlib canvas — in about a second:

```
CONSTRUCTED OK: EnhancedFishAnalyzer root: Tk
   child: Notebook
   child: Frame
   child: Frame
SMOKE PASS
```

`CONFIRMED`. Caveat: this was on Windows, which always has a window station. On a truly headless CI runner `tk.Tk()` raises `TclError`, so the test needs a skip guard — included below. `PLAUSIBLE` that this is the only obstacle to running it in CI.

I wrote and ran a candidate suite. It is at `<scratchpad>/test_smoke.py` — I did not add it to the repo, per the no-modification constraint. Result:

```
2 failed, 5 passed, 1 warning in 3.68s
```

Both failures are real bugs, not test bugs:

- `test_version_matches_banner` — catches finding #9 (`v2.0` vs `2.1.0`)
- `test_console_output_is_ascii_safe` — catches A3, and reports all three offending lines with line numbers

The five that pass are the ones worth having as a regression net: package imports; the shoaling call signature (asserted as *not* static, so it fails loudly if someone "fixes" the README by changing the code instead); load-and-process on a generated fixture; shoaling with the invariant `mean_nnd <= mean_iid`; and GUI construction.

The fixture is the load-bearing part: ~15 lines building a synthetic idtracker.ai-shaped `.npy` in `tmp_path`. It needs no data from you, commits nothing, and every future numerical test in Pass B can build on it. If you adopt one thing from this section, adopt the fixture.

### Packaging: yes, and it is smaller than you think

A `pyproject.toml` with `[project]` name/version/dependencies and `[tool.setuptools] packages = ["fish_analyzer", "fish_analyzer.gui"]` is roughly 20 lines and buys three things:

- `pip install -e .` makes `import fish_analyzer` work from anywhere, deleting the `sys.path` hack at `run_analyzer.py:55-57` and making the README example true as written.
- Dependencies get declared in **one** place, ending the requirements.txt-vs-README-vs-docstring drift entirely (findings 2, 3, 4, 5, 12 are all the same finding wearing different hats).
- `__version__` can be read from package metadata, so finding #9 becomes structurally impossible.

It does not require abandoning `python run_analyzer.py`; that keeps working. `PLAUSIBLE`, but this is well-trodden ground — for a single-maintainer lab tool the cost is an afternoon and it is the cheapest of the fixes here.

### Everything else, in the order I would do it

1. **Rewrite `requirements.txt`** as a valid pip file: add `scikit-learn`, pin traja, drop the conda lines (move them to the README as a clearly-labelled alternative), and reconcile the pandas/numpy contradiction downward to `pandas>=1.5`, `numpy>=1.24`. Half an hour, unblocks everyone.
2. **Strip non-ASCII from `print()`** — 12 lines, 6 files, mechanical. `[ok]` / `[fail]` / `[warn]`. Fixes A3 including the poisoned error handler.
3. **Fix `README.md:105`** to match `run_analyzer.py:44-47`.
4. **Add `*.avi`, `*.mp4`, `session_*/`, `venv/`, `.venv/` to `.gitignore`** before someone commits a session folder.
5. **Delete `fish_analyzer/backup/gui.py.backup`.** It is in git history at `b5c9725`; `git show b5c9725:fish_analyzer/backup/gui.py.backup` retrieves it forever. A file that no tool can see and no code can reach is not a backup, it is 3,318 lines of noise in every future `grep`.
6. **Add the licence.** `AUDIT_H_APPROACH.md` recommends adopting two GPL-3.0 packages; that decision is unanswerable while `README.md:178` is a placeholder.

---

## What is fine, and what I would not touch

Worth saying explicitly, because a list of only problems tells you nothing about what to trust.

- **The layering holds up under execution, not just under reading.** I imported and ran every analysis module without a GUI, in three Python versions. `AUDIT_H_APPROACH.md`'s claim that a headless entry point is cheap is `CONFIRMED` — I effectively wrote one, four times, as test scripts.
- **The optional-dependency degradation works as designed.** `SHAPELY_AVAILABLE`, `CV2_AVAILABLE`, `GUI_AVAILABLE` all resolve correctly, and `__init__.py:67-74` warns rather than crashing. `PIL` in `inspector_tab.py:32` has the same guard, and in practice always resolves because matplotlib pulls in pillow.
- **CSV export is written correctly.** All six `open()` calls for writing use `newline='' , encoding='utf-8'` (`export.py:159, 256, 302, 348, 389`; `gui/bout_tab.py:968`). Given A3, this is a notable piece of care — the export path is *more* robust than the console path.
- **`ebb825c` did its job completely.** No PEP 604 or PEP 585 syntax remains anywhere. `CONFIRMED`.
- **`run_analyzer.py`'s `sys.path` insertion works**, and its docstring API example is correct where the README's is not.
- **The code survives numpy 1.26 → 2.5 and pandas 1.5 → 3.0 unchanged**, which is not typical of research code of this age and is worth knowing before anyone proposes pinning it down harder.
- **The pandas dependency is confined to one file.** `processing.py` is the only module that imports pandas, with two `pd.` call sites (`CONFIRMED` by grep) — both building the frame `TrajaDataFrame` wraps. That is why dropping traja plausibly drops pandas with it.

---

## What I ran

Throwaway venvs at `<scratchpad>/venv312` (Python 3.12.5) and `<scratchpad>/venv313` (Python 3.13.5, created from the `idtrackerai` conda env's interpreter without modifying it). The existing `traja_fish_analysis` conda env (Python 3.9.23) was used read-only as a comparison environment.

Scripts, all in the scratchpad: `make_synth.py` (fixture generator), `readme_example.py` (README block verbatim), `smoke_full.py` (full pipeline → JSON, run in all three envs), `smoke_smooth.py` (smoothing path + raw traja outputs), `traja_replacement.py` (numpy equivalence proof), `gui_smoke.py` (headless GUI construction), `test_smoke.py` (candidate pytest suite).

Read-only checks: `ast.parse` feature-version sweep over all 24 modules; import-graph extraction; wheel metadata for traja / trajectorytools / idtrackerai / numpy / pandas / scipy / matplotlib / shapely / opencv / scikit-learn; `pip index versions traja`; `git check-ignore`; `git ls-files`; `git log`; non-ASCII `print()` census.

---

## Appendix — three claims from `AUDIT_H_APPROACH.md`, checked independently

Checked first-hand at your request, against the live index and installed package metadata rather than against docs.

**1. "There is no traja 0.6, so `traja>=0.6` resolves to 25.0.1."** — `CONFIRMED`, both halves. `pip index versions traja` lists 31 releases, highest 0.x is 0.2.8; `pip install "traja==0.6"` errors with no matching distribution; and `pip install "traja>=0.6"` in a clean venv installed **25.0.1**. See A6.

**2. "`smooth_sg`, `length`, `distance`, `accessor.get_derivatives` all still exist on traja master."** — `CONFIRMED`, with one precision note. I verified against the **installed 25.0.1 wheel**, not against the git master branch — I had no network access to the GitLab/GitHub tree, and for your purposes the released wheel is the thing that matters anyway. The line numbers cited in `AUDIT_H_APPROACH.md` match exactly:

| API | Cited | Found |
|---|---|---|
| `smooth_sg` | `trajectory.py:68` | `trajectory.py:68` ✓ |
| `distance` | `trajectory.py:154` | `trajectory.py:154` ✓ |
| `length` | `trajectory.py:181` | `trajectory.py:181` ✓ |
| `get_derivatives` | `accessor.py:426` | `accessor.py:426` ✓ |

And they don't merely exist — all four **run and return identical values to traja 22.0.0** (A7). `AUDIT_H_APPROACH.md` flagged "whether it computes the same thing is unverified and belongs to Pass B"; that question is now answered, at least for synthetic data.

**3. "trajectorytools and idtracker.ai 6.x both require Python ≥3.10."** — `CONFIRMED`, from package metadata rather than docs:

- `trajectorytools 0.4.2` wheel metadata: `Requires-Python: >=3.10`. Its `Requires-Dist` is `scipy, scikit-learn, matplotlib, h5py, miniballcpp` — five packages, all conda-friendly, no compiled exotica. The install-cost claim in `AUDIT_H_APPROACH.md` holds.
- `idtrackerai 6.0.10` (installed on this machine): `Requires-Python: >=3.10`.

**Consequence `AUDIT_H_APPROACH.md` did not draw:** this makes the `python=3.10` line in `requirements.txt:15` the only one of the four Python claims that is forward-compatible with the recommended direction. The Step 3 cross-validation against `trajectorytools` cannot happen in `traja_fish_analysis` (Python 3.9.23) — that environment is below the floor. Either that notebook gets its own environment, or the project moves to ≥3.10 first. Given that 3.12 and 3.13 are both verified working (A8), moving is the cheaper option, and it should be folded into fix #1 above.

One more thing worth knowing for Pass E: your `idtrackerai` env is Python **3.13.5** and your `traja_fish_analysis` env is Python **3.9.23**. `head_detection/` needs the former, `fish_analyzer/` currently runs in the latter. Standardising on one ≥3.10 environment is what would let those two halves of the project be in the same room.

---

## Fix status — applied 2026-07-31

Applied and verified. Every check below was re-run **without** `PYTHONIOENCODING=utf-8`, i.e. under the cp1252 stdout that used to crash the run.

| Finding | Fix | Verified by |
|---|---|---|
| A1 conda lines in requirements | `requirements.txt` rewritten as a valid pip file | fresh venv, `pip install -r requirements.txt` → exit 0 |
| A2 undeclared scikit-learn | `scikit-learn>=1.0` added, with the reason in a comment | `import fish_analyzer` OK in that fresh venv |
| A3 non-ASCII `print()` | 11 statements across 5 files → `[ok]` / `[skip]` / `[FAILED]` / `[warn]` / `+/-` / `^2` / `->` | 0 remaining; full pipeline runs on cp1252 stdout |
| A4 README API example | corrected to `ShoalingCalculator(loaded_file, params).calculate()` | pinned by `test_readme_api_example_signature` |
| A5 pandas/numpy contradiction | reconciled to `pandas>=1.5`, `numpy>=1.24`; conda recipe deduplicated into README | single source in `requirements.txt` |
| A6 `traja>=0.6` | pinned `traja==25.0.1` | resolves deterministically |
| A9 `.gitignore` | added `*.avi *.mp4 *.mov *.mkv *.pickle session_*/ venv/ .venv/ env/` | `git check-ignore -v` on each |
| A12 dead backup | `fish_analyzer/backup/gui.py.backup` deleted (recoverable: `git show f3a634c:fish_analyzer/backup/gui.py.backup`) | zero references confirmed first |
| #7 README structure | added `bout_analysis.py`, `bout_tab.py`, `inspector_tab.py`, standalone scripts | — |
| #8 `run_analyzer.py:19` | `gui.py` → `gui/` subpackage, module list completed | — |
| #9 `v2.0` banner | now interpolates `__version__` | prints `v2.1.0` |
| #6 "Python 3.8+" | → "3.9 or newer", with the 3.10+ note for trajectorytools/idtracker.ai | — |
| #11 cwd requirement | stated in the README API section | — |
| — | **7 smoke tests added** at `tests/test_smoke.py` | `7 passed in 3.36s` |
| — | **`pytest.ini` added** | see below |

**One new finding, discovered while wiring up the tests:** bare `pytest` at the repo root tried to import `head_detection/head_detection_test.py`, because it matches pytest's default `*_test.py` pattern. That file is a script, not a test module — importing it executes its work at module scope and calls `os.makedirs`. `pytest.ini` now restricts collection to `tests/`. This is A10 biting in practice.

**Numeric regression check:** the full pipeline was re-run after all edits and diffed against the pre-fix run. **0 differences, and the exported CSV is byte-identical.** The fixes changed no numbers.

### Still open

- **Licence** (#10) — needs your decision, not a fix. Blocks reasoning about the GPL-3.0 adoptions in `AUDIT_H_APPROACH.md`.
- **Packaging / `pyproject.toml`** (A11) — deliberately not done unilaterally; it changes how the tool is installed.
- **A10** — `head_detection/` still has hardcoded personal paths, no main guards, and a tkinter dialog at module scope. That is Pass E's territory.
- ~~**A7's caveat** — the traja-version equivalence is proven on synthetic data only. One real `trajectories.npy` closes it.~~ **Closed 2026-07-31, see below.**
- **Dropping traja** — recommended, but deliberately after Pass B, since traja 22.0.0 is currently the only independent reference implementation.

---

## Addendum — verified against real data, 2026-07-31

Four real idtracker.ai sessions were supplied after this pass was written
(`wtTAB_6mo/Freeswim/`, 6 fish × 18,000 frames each, idtracker.ai 6.0.8).
Everything this report could only check on a synthetic array has now been
re-checked on them.

### A7 closed — traja is not doing anything special with real dropouts

The two-venv version comparison could **not** be repeated on real data:
`traja==22.0.0` does `traja/__init__.py → traja.dataset → import torch`
unconditionally, and torch's `c10.dll` fails to initialise in this
environment (`OSError: [WinError 1114]`). So the old stack cannot be stood up
here at all.

Instead the underlying concern was tested directly — *does traja treat real
NaN gaps differently from the 15-line replacement this report proposed?*

| Check | Result |
|---|---|
| `trj.traja.get_derivatives()['speed']` vs manual `np.diff`-based speed | max relative difference **2.53e-12** across all 6 fish |
| NaN masks of the two | agree on **18,000 / 18,000** frames, every fish |

Floating-point identical, on trajectories with real dropout structure. This
also hands Pass B a result: the two speed pipelines it is asked to reconcile
do **not** differ in the derivative itself, so any divergence between
`processing.py` and `bout_analysis.py` comes from thresholds or NaN policy,
not from traja.

**A new argument for the `traja==25.0.1` pin:** 25.0.1 makes torch optional
and degrades with a `UserWarning`; 22.0.0 hard-imports it. The pin is not just
about reproducibility, it avoids a multi-GB dependency.

### Metadata keys — the assumption was correct, and there is more in the file

All eight assumed keys are present. But real files carry **17** keys, and
eight are read by nothing in the codebase:

| Unused key | Type | Why it matters |
|---|---|---|
| `video_paths` | list of str | **The source video path is recorded in the file.** `_find_video_file()` (`file_loading.py:203`) searches the filesystem instead of reading this. In these sessions it points at a different user's OneDrive (`MQ10002204`), which is why no video is found locally. |
| `id_probabilities` | ndarray (18000, 6, 1) | Per-frame, per-fish identification confidence. This is the missing signal for telling "genuinely absent" from "tracked but unreliable" — directly relevant to Pass D's finding that nothing distinguishes a real NaN from a failure. |
| `areas` | dict | Per-blob areas; relevant to the posture/head work in Pass E. |
| `silhouette_score` | float (0.854 here) | A whole-session tracking-quality number that could be surfaced at load time. |
| `length_unit` | `None` here | idtracker.ai *can* carry a unit. It is unset in these files, so the tool's own calibration is the only source of truth — which rules out a competing one for Pass B. |
| `setup_points`, `identities_groups`, `fragment_connectivity` | dict/dict/float | Unexamined. |

### Calibrate-in-cm path — works, exactly

Could not be exercised before. Loading a real session, setting
`from_physical_measurement(pixels_per_unit=24.0, unit_name="cm")` and
re-processing:

| Check | Result |
|---|---|
| `total_distance` ratio cm-run / BL-run, per fish | `2.9712` for all six |
| Expected ratio (`body_length_px / 24`) | `2.9712` |
| Max deviation | **8.88e-16** |
| Exported `Unit` column | `'cm'` |

The scaling itself is correct. (What is *not* safe is changing calibration
**after** an analysis has run — that was Pass D finding C1, now fixed.)

### Real dropout is far worse than the synthetic fixture assumed

The smoke-test fixture uses 2% uniformly-random NaNs. Real sessions:

| Session | NaN | Longest single gap | body_length |
|---|---|---|---|
| G604_Freeswim | 9.65% | 99 frames (3.3 s) | 71.31 px |
| G604_Freeswim_group2 | 6.79% | **188 frames (6.3 s)** | 73.01 px |
| H604_Freeswim | 2.69% | 108 frames (3.6 s) | 82.22 px |
| H604_Freeswim_group2 | 1.26% | 38 frames (1.3 s) | 79.63 px |

Two consequences worth handing to Pass B:

1. **Gaps are long and clustered, not sprinkled.** Anything that interpolates
   across them — `smooth_time_series`, `get_smoothed_fish_timeseries`,
   heatmap binning — is inventing up to 6 seconds of trajectory.
2. **`body_length` differs by up to 15% across sessions**, so "BL" is a
   different physical unit per file and cross-file comparisons in body lengths
   are not like-for-like.

### One API sharp edge found while writing these checks

`TrajectoryProcessor(...).process_all_fish()` returns `FishTrajectory` objects
with an **empty `metrics` dict** — metrics are added by a later stage. Only
`process_and_analyze_file()` returns fully-populated fish. The README's API
example uses the correct one, but the class is public and the failure mode is
a bare `KeyError: 'total_distance'`.
