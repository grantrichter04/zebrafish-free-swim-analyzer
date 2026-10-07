# RA-ready pipeline: one install, video to results

Date: 2026-10-08. Branch: `polish/ra-ready`.

## Goal

A research assistant with no command-line experience can sit at the shared lab
laptop, open one app, and take a folder of free-swim videos through
idtracker.ai tracking and into the existing analysis tabs. Whoever sets the
laptop up does so once, by double-clicking an installer.

## Decisions already made

| Question | Decision |
|---|---|
| Where it runs | One shared Windows laptop with an NVIDIA GPU, set up once |
| Environments | One conda env holding both idtracker.ai and the analyzer |
| Wrapper form | A Tracking tab inside the existing app, not a CLI or a second app |
| Batch parameters | One setup covers every video in a folder; no per-video overrides yet |
| Where setups live | Beside the videos, as `.toml` files in the experiment folder |
| Repo strategy | Work on a branch of this repo; no rewrite |

## Evidence the single env works

Built 2026-10-08 as conda env `freeswim` on this machine (RTX 4070 Ti SUPER,
driver 595.71):

- Python 3.12, torch 2.11.0+cu128, torchvision 0.26.0+cu128, idtrackerai 6.0.14,
  opencv-python-headless 5.0.0.93, numpy 2.5.2, pandas 3.0.6.
- `idtrackerai_test` passed in 3 min 4 s on the CUDA backend.
- The analyzer's suite passed (188 tests).
- The analyzer loaded the self-test's `session_test` folder and processed all
  8 fish.

The only workaround was installing the analyzer with `--no-deps`, because
`pyproject.toml` asks for `opencv-python` while idtracker.ai installs
`opencv-python-headless`; both own the same `cv2` directory.

## Phase 1: installation

1. `pyproject.toml` depends on `opencv-python-headless>=4.10` instead of
   `opencv-python`. Nothing in the repo opens an OpenCV window.
2. New optional extra `tracking = ["idtrackerai>=6.0.14,<6.1"]`. A plain install
   still gives analysis only.
3. `requires-python = ">=3.10"`. CI runs 3.12 only.
4. Delete `requirements.txt`. `environment.yml` shrinks to Python 3.12 and pip.
5. `constraints-win-cu128.txt` pins the versions listed above, so a reinstall
   reproduces the tested environment.
6. `install.bat`, double-clickable:
   - stops with a plain message if conda or an NVIDIA driver is missing;
   - creates (or reuses) the `freeswim` env;
   - installs torch and torchvision from the cu128 index;
   - runs `pip install -e ".[tracking]" -c constraints-win-cu128.txt`;
   - runs the self-check and prints the result;
   - creates a "Free Swim Analyzer" desktop shortcut that starts the app with
     no console window.
7. Self-check, available as `fish-analyzer --check` and from the app's Help
   menu. Reports Python version, analyzer version, idtrackerai version or
   "not installed", torch version, and whether CUDA is available with the GPU
   name. Exit code is non-zero if the analyzer cannot import; a missing
   idtrackerai or GPU is reported as a warning, since analysis still works.

`fish_analyzer/selfcheck.py` holds the check as a function returning structured
results, so the CLI flag, the menu item and the tests all use the same code.

## Phase 2: the Tracking tab

### What the RA sees

The tab is first in the notebook. Top to bottom:

1. **Videos.** "Choose folder…" and a list of the videos found, each with a
   status: *not tracked*, *tracked*, *incomplete*, *running*, *failed*.
2. **Setup.** A dropdown of the `.toml` files in that folder and a
   "Configure new setup…" button. The button asks which video to use, opens
   idtracker.ai's own window on it, and tells the RA to save the parameters
   into the experiment folder. When that window closes the dropdown refreshes.
   "Edit setup…" reopens idtracker.ai with the selected setup loaded.
3. **Run.** "Track all untracked" and "Stop". Shows which video is running,
   elapsed time, and idtracker.ai's log streaming into a pane. No percentage:
   idtracker.ai does not expose one reliably.
4. **Done.** A summary of successes and failures, and "Load tracked sessions",
   which passes the session folders to the existing session loader.

### How it works

`fish_analyzer/tracking.py`, with no GUI imports:

- `find_videos(folder)`: files with extensions `.avi .mp4 .mov .mkv .mpg .mpeg`,
  top level only, sorted by name.
- `session_folder_for(video)`: `<folder>/session_<video stem>`.
- `tracking_status(video)`: *tracked* if
  `session_<stem>/trajectories/trajectories.npy` exists, *incomplete* if the
  session folder exists without it, otherwise *not tracked*. Status comes from
  the filesystem alone, so it is correct after a restart and a batch resumes
  where it stopped.
- `find_setups(folder)`: `.toml` files in the folder.
- `build_track_command(video, setup)`: runs
  `<this python> -m idtrackerai.start --load <setup> --video_paths <video>
  --name <stem> --track`. Using the current interpreter avoids any dependence
  on PATH.
- `build_configure_command(video, setup=None)`: the same without `--track`,
  which opens idtracker.ai's window.
- `run_tracking(video, setup, on_line, should_stop)`: starts the process,
  feeds each output line to `on_line`, ends the process if `should_stop`
  returns true, and returns a result with success, duration and the last
  error lines.
- `idtrackerai_available()`: whether the package can be found, without
  importing torch.

`fish_analyzer/gui/tracking_tab.py` is a mixin like the other tabs. The batch
runs on a worker thread and reports back to tkinter through `after`. Videos run
one at a time. idtracker.ai is always a separate process, so its Qt window and
our tkinter app never share an interpreter, and a tracking crash cannot take
the analyzer down.

### Failure handling

- One failed video does not stop the batch; the summary lists it with the last
  lines of its log.
- A setup that names a different number of animals than the video contains
  fails inside idtracker.ai; that message is surfaced as-is.
- If idtracker.ai is not installed, the tab shows one sentence saying so and
  disables its buttons. The rest of the app works.
- "Stop" ends the running process and leaves that video *incomplete*.
  Re-running an *incomplete* video asks before overwriting its session folder.
- Closing the app while tracking is running asks for confirmation, then stops
  the process.

### To verify during implementation

- That command-line `--video_paths` and `--name` override the values stored in
  a setup `.toml` saved from idtracker.ai's window. If they do not, the tab
  writes a per-video copy of the setup into a temp folder with those two
  fields replaced.
- That idtracker.ai's window accepts `--video_paths` without `--track` and
  opens with the video loaded.

### Known risk: the data lives in a synced folder

The test folder is under `<lab share>`, a
OneDrive/SharePoint sync root. Tracking writes a session folder with large
intermediate files beside each video, and the sync client will upload them and
may lock files mid-write. Phase 2 keeps sessions beside the videos, because
that is where the analyzer already looks for the session and its video. If
sync causes failures in testing, the fallback is an optional "working folder"
setting for session output. That is not built unless testing shows it is
needed.

## Phase 3: docs and tidy-up

- `README.md` rewritten for RAs: what it is, install, open the app, track a
  folder, analyse, export, where to get help. About one screen, plus a short
  section for developers.
- `CHANGELOG.md` takes the version history currently in the README.
- `AUDIT_PLAN.md`, the four `AUDIT_*.md` files and `audit/` move to
  `docs/audit/`. Links between them are updated. Nothing is deleted.
- `head_detection/` and `fish_posture_analyzer.py` move to `extras/` with a
  short README saying they are standalone and not part of the app.
- `run_analyzer.py` is removed; the shortcut and `python -m fish_analyzer`
  replace it.
- `docs/running-an-experiment.md`: a one-page guide with screenshots of the
  Tracking tab, written after Phase 2.
- Version becomes 2.2.0.

## Testing

- `tests/test_tracking.py`: video discovery, status from folder layouts built
  in a temp dir, command construction, and `run_tracking` driven by a stand-in
  script that prints lines and exits with a chosen code. None of it needs
  idtracker.ai or a GPU, so it runs in CI.
- `tests/test_selfcheck.py`: the check returns sensible results with and
  without idtrackerai importable.
- GUI regression tests for the Tracking tab follow the existing pattern in
  `tests/test_gui_regressions.py`.
- Manual acceptance on real data: the two 10-minute, 1288×964, 30 fps MJPG
  videos in `<first experiment>`. Configure a setup on one, batch
  both, load both sessions, run the analysis tabs.
- Final acceptance: delete the `freeswim` env, run `install.bat`, and repeat
  the manual run from the desktop shortcut.

## Out of scope

- Per-video parameter overrides.
- Tracking videos in parallel.
- Mac or Linux installers, and CPU-only tracking.
- Reimplementing any idtracker.ai controls.
- The two open analysis decisions from Audit B (the `min_valid_percentage` and
  `min_freeze_frames` defaults, and scoping the bout tab to larvae).
- Choosing a licence; the README placeholder stays until the lab decides.

## Order of work

Phase 1, then Phase 2, then Phase 3. Each phase gets its own implementation
plan and its own PR into `polish/ra-ready`.
