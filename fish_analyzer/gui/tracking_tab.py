"""
fish_analyzer/gui/tracking_tab.py
=================================
Tracking Tab - choose a folder of videos, make an idtracker.ai setup for it,
track every video, and hand the results to the analysis tabs.

The work itself is in fish_analyzer/tracking.py. This file is the widgets, and
the plumbing that lets a separate idtracker.ai process report back to tkinter:
the process runs on a worker thread, its output goes into a queue, and the
main thread drains that queue on a timer.
"""
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Callable, Dict, List, Optional, Set

from .. import tracking


class TrackingTabMixin:
    """
    Mixin providing Tracking Tab functionality.

    Expects the following attributes from base class:
    - self.root: tk.Tk
    - self.notebook: ttk.Notebook
    - self.loaded_files, and DataTabMixin's _add_session
    """

    LOG_LINES_KEPT = 2000

    def _create_tracking_tab(self):
        """Create the tracking tab: videos, setup, tracking, and the output."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Tracking")

        self._tracking_folder: Optional[Path] = None
        self._tracking_videos: List[Path] = []
        self._tracking_setups: List[Path] = []
        self._tracking_busy = False
        self._tracking_batch_running = False
        self._tracking_stop = False
        self._tracking_thread: Optional[threading.Thread] = None
        # Setup file name -> the videos it has been opened on this session.
        self._tracking_checked: Dict[str, Set[Path]] = {}
        # What this run knows that the folder cannot say: running, failed,
        # stopped. Anything else is read from the folder.
        self._tracking_live: Dict[Path, str] = {}
        self._tracking_progress: Optional[tuple] = None
        self._tracking_queue: "queue.Queue" = queue.Queue()

        tk.Label(
            tab, justify=tk.LEFT, anchor="w", font=("Arial", 9), fg="gray30",
            text="Start here with new videos: idtracker.ai turns each video "
                 "into a tracked session.\nAlready have tracked sessions? Go "
                 "straight to \"Data Setup & Calibration\"."
        ).pack(fill="x", padx=20, pady=(10, 0))

        self._create_tracking_videos_section(tab)
        self._create_tracking_setup_section(tab)
        self._create_tracking_run_section(tab)
        self._create_tracking_log_section(tab)

        if not tracking.idtrackerai_available():
            self.tracking_hint_var.set(
                "idtracker.ai is not installed in this environment, so "
                "tracking is unavailable. Run install.bat to add it.")
        self._tracking_update_buttons()
        self.root.protocol("WM_DELETE_WINDOW", self._tracking_on_close)

    # =========================================================================
    # LAYOUT
    # =========================================================================

    def _create_tracking_videos_section(self, parent):
        frame = tk.LabelFrame(parent, text="1. Videos", font=("Arial", 12, "bold"))
        frame.pack(fill="x", padx=20, pady=(10, 5))

        controls = tk.Frame(frame)
        controls.pack(fill="x", pady=8, padx=10)
        tk.Label(controls, text="Experiment folder:").pack(side="left", padx=5)
        self.tracking_folder_var = tk.StringVar()
        tk.Entry(controls, textvariable=self.tracking_folder_var, width=70,
                 state="readonly").pack(side="left", padx=5)
        tk.Button(controls, text="Choose folder...",
                  command=self._tracking_choose_folder,
                  bg="lightblue", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(controls, text="Refresh",
                  command=self._tracking_refresh).pack(side="left", padx=2)

        self.tracking_videos_tree = ttk.Treeview(
            frame, columns=("video", "status"), show="headings", height=5,
            selectmode="browse")
        self.tracking_videos_tree.heading("video", text="Video")
        self.tracking_videos_tree.heading("status", text="Status")
        self.tracking_videos_tree.column("video", width=600, anchor="w")
        self.tracking_videos_tree.column("status", width=140, anchor="w")
        self.tracking_videos_tree.pack(fill="x", padx=10, pady=(0, 8))

    def _create_tracking_setup_section(self, parent):
        frame = tk.LabelFrame(parent, text="2. Setup", font=("Arial", 12, "bold"))
        frame.pack(fill="x", padx=20, pady=5)

        tk.Label(
            frame, justify=tk.LEFT, font=("Arial", 9), fg="gray", wraplength=900,
            text="A setup holds the idtracker.ai settings for this experiment: "
                 "the thresholds, the number of fish and the arena. It is saved "
                 "as a .toml file beside the videos and used for every video "
                 "in the folder."
        ).pack(anchor="w", padx=10, pady=(6, 0))

        controls = tk.Frame(frame)
        controls.pack(fill="x", pady=8, padx=10)
        tk.Label(controls, text="Setup:").pack(side="left", padx=5)
        self.tracking_setup_var = tk.StringVar()
        self.tracking_setup_combo = ttk.Combobox(
            controls, textvariable=self.tracking_setup_var, state="readonly",
            width=40)
        self.tracking_setup_combo.pack(side="left", padx=5)
        self.tracking_setup_combo.bind(
            "<<ComboboxSelected>>", lambda e: self._tracking_update_buttons())
        self.tracking_configure_button = tk.Button(
            controls, text="Configure new setup...",
            command=lambda: self._tracking_configure(edit=False),
            bg="lightblue", font=("Arial", 10, "bold"))
        self.tracking_configure_button.pack(side="left", padx=5)
        self.tracking_edit_button = tk.Button(
            controls, text="Check setup on selected video...",
            command=lambda: self._tracking_configure(edit=True))
        self.tracking_edit_button.pack(side="left", padx=2)

        self.tracking_hint_var = tk.StringVar()
        tk.Label(frame, textvariable=self.tracking_hint_var, justify=tk.LEFT,
                 font=("Arial", 10, "bold"), fg="#8a4b00", wraplength=900
                 ).pack(anchor="w", padx=10, pady=(0, 6))

    def _create_tracking_run_section(self, parent):
        frame = tk.LabelFrame(parent, text="3. Track", font=("Arial", 12, "bold"))
        frame.pack(fill="x", padx=20, pady=5)

        controls = tk.Frame(frame)
        controls.pack(fill="x", pady=8, padx=10)
        self.tracking_track_button = tk.Button(
            controls, text="Track all untracked videos",
            command=self._tracking_track_all,
            bg="lightgreen", font=("Arial", 11, "bold"))
        self.tracking_track_button.pack(side="left", padx=5)
        self.tracking_stop_button = tk.Button(
            controls, text="Stop", command=self._tracking_request_stop)
        self.tracking_stop_button.pack(side="left", padx=5)
        self.tracking_load_button = tk.Button(
            controls, text="Load tracked sessions for analysis",
            command=self._tracking_load_sessions,
            bg="lightblue", font=("Arial", 10, "bold"))
        self.tracking_load_button.pack(side="right", padx=5)

        self.tracking_progress_var = tk.StringVar()
        tk.Label(frame, textvariable=self.tracking_progress_var, anchor="w",
                 font=("Arial", 10, "bold"), fg="#1f4e79"
                 ).pack(fill="x", padx=15, pady=(0, 6))

    def _create_tracking_log_section(self, parent):
        frame = tk.LabelFrame(parent, text="idtracker.ai output",
                              font=("Arial", 12, "bold"))
        frame.pack(fill="both", expand=True, padx=20, pady=(5, 10))

        scroll = tk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        self.tracking_log_text = tk.Text(
            frame, height=8, wrap="none", font=("Courier", 9),
            state="disabled", yscrollcommand=scroll.set)
        self.tracking_log_text.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.tracking_log_text.yview)

    # =========================================================================
    # FOLDER, VIDEOS AND SETUPS
    # =========================================================================

    def _tracking_choose_folder(self):
        folder = filedialog.askdirectory(
            title="Choose the folder holding this experiment's videos")
        if folder:
            self._tracking_set_folder(Path(folder))

    def _tracking_set_folder(self, folder: Path):
        self._tracking_folder = Path(folder)
        self._tracking_live.clear()
        self.tracking_folder_var.set(str(self._tracking_folder))
        self._tracking_refresh()

    def _tracking_status(self, video: Path) -> str:
        """The folder's answer, unless this run knows better."""
        status = tracking.tracking_status(video)
        if status == tracking.TRACKED:
            return status
        return self._tracking_live.get(video, status)

    def _tracking_refresh(self, select_setup: Optional[Path] = None):
        """Re-read the folder: which videos, their status, which setups."""
        folder = self._tracking_folder
        tree = self.tracking_videos_tree
        selected_video = self._tracking_selected_video()
        tree.delete(*tree.get_children())
        if folder is None or not folder.is_dir():
            self._tracking_videos, self._tracking_setups = [], []
            self.tracking_setup_combo["values"] = []
            self.tracking_setup_var.set("")
            self._tracking_update_buttons()
            return

        self._tracking_videos = tracking.find_videos(folder)
        for index, video in enumerate(self._tracking_videos):
            tree.insert("", "end", iid=str(index),
                        values=(video.name, self._tracking_status(video)))
        if self._tracking_videos:
            keep = (self._tracking_videos.index(selected_video)
                    if selected_video in self._tracking_videos else 0)
            tree.selection_set(str(keep))

        previous = self.tracking_setup_var.get()
        self._tracking_setups = tracking.find_setups(folder)
        names = [s.name for s in self._tracking_setups]
        self.tracking_setup_combo["values"] = names
        if select_setup is not None and select_setup.name in names:
            self.tracking_setup_var.set(select_setup.name)
        elif previous not in names:
            self.tracking_setup_var.set(names[0] if names else "")

        if tracking.idtrackerai_available() and not self._tracking_busy:
            if not self._tracking_videos:
                self.tracking_hint_var.set("No videos found in this folder.")
            elif not names:
                self.tracking_hint_var.set(
                    "This folder has no setup yet. Select a video and press "
                    "\"Configure new setup...\" to make one in idtracker.ai.")
            elif "no setup yet" in self.tracking_hint_var.get():
                self.tracking_hint_var.set("")
        self._tracking_update_buttons()

    def _tracking_selected_video(self) -> Optional[Path]:
        selection = self.tracking_videos_tree.selection()
        if selection and int(selection[0]) < len(self._tracking_videos):
            return self._tracking_videos[int(selection[0])]
        return self._tracking_videos[0] if self._tracking_videos else None

    def _tracking_selected_setup(self) -> Optional[Path]:
        name = self.tracking_setup_var.get()
        return next((s for s in self._tracking_setups if s.name == name), None)

    def _tracking_untracked(self) -> List[Path]:
        return [v for v in self._tracking_videos
                if tracking.tracking_status(v) != tracking.TRACKED]

    def _tracking_tracked(self) -> List[Path]:
        return [v for v in self._tracking_videos
                if tracking.tracking_status(v) == tracking.TRACKED]

    def _tracking_update_buttons(self):
        def state(enabled) -> str:
            return "normal" if enabled else "disabled"

        idle = not self._tracking_busy
        ready = (tracking.idtrackerai_available() and idle
                 and bool(self._tracking_videos))
        has_setup = self._tracking_selected_setup() is not None
        self.tracking_configure_button.config(state=state(ready))
        self.tracking_edit_button.config(state=state(ready and has_setup))
        self.tracking_track_button.config(
            state=state(ready and has_setup and self._tracking_untracked()))
        self.tracking_stop_button.config(
            state=state(self._tracking_batch_running and not self._tracking_stop))
        self.tracking_load_button.config(
            state=state(idle and self._tracking_tracked()))

    # =========================================================================
    # MAKING AND CHECKING A SETUP IN IDTRACKER.AI
    # =========================================================================

    def _tracking_configure(self, edit: bool):
        """Make a new setup, or check an existing one, on the selected video.

        Either way idtracker.ai's own window opens on that video. It cannot
        start tracking from here (see idtrackerai_setup_window.py): tracking is
        for the whole folder, afterwards.
        """
        video = self._tracking_selected_video()
        if video is None:
            messagebox.showinfo(
                "Choose a folder first",
                "Choose a folder that contains videos, then try again.")
            return

        if edit:
            setup = self._tracking_selected_setup()
            if setup is None:
                messagebox.showinfo("No setup selected",
                                    "Pick a setup from the list first.")
                return
            self._tracking_open_setup_window(video, setup, load=setup)
            return

        setup = self._tracking_ask_new_setup_name(video.parent)
        if setup is None:
            return
        if not messagebox.askokcancel(
                "Configure setup in idtracker.ai",
                f"idtracker.ai will open on:\n    {video.name}\n\n"
                "  1. Set the number of animals, the thresholds and the arena.\n"
                "  2. Press \"Save setup and close\" at the bottom left.\n\n"
                "Nothing is tracked yet. Tracking is started from this tab "
                "afterwards, for every video in the folder."):
            return
        self._tracking_open_setup_window(video, setup, load=None)

    def _tracking_ask_new_setup_name(self, folder: Path) -> Optional[Path]:
        """Ask what to call a new setup; None if the user backs out."""
        while True:
            name = simpledialog.askstring(
                "Name this setup",
                "A short name for these settings, for example the rig or "
                "the experiment:",
                initialvalue="setup", parent=self.root)
            if name is None:
                return None
            setup = tracking.setup_path_for(folder, name)
            if setup is None:
                messagebox.showwarning(
                    "That name cannot be used",
                    "Use letters, numbers, spaces, dashes or underscores.")
            elif setup.exists():
                messagebox.showwarning(
                    "That name is taken",
                    f"{setup.name} already exists in this folder. Choose "
                    "another name, or select it and use \"Check setup on "
                    "selected video...\".")
            else:
                return setup

    def _tracking_open_setup_window(self, video: Path, setup: Path,
                                    load: Optional[Path]):
        """Run idtracker.ai's window on `video`, then report what happened and
        offer the next video the setup has not been looked at on."""
        checking = load is not None
        before = setup.stat().st_mtime_ns if setup.exists() else None
        command = tracking.build_configure_command(video, save_to=setup, load=load)

        def work(log):
            log(f"> {' '.join(command)}")
            return tracking.run_process(command, video.parent, log,
                                        lambda: self._tracking_stop)

        def finished(exit_code):
            after = setup.stat().st_mtime_ns if setup.exists() else None
            saved = after is not None and after != before
            if after is not None:
                self._tracking_checked.setdefault(setup.name, set()).add(video)
            self._tracking_refresh(select_setup=setup if after else None)

            if not saved and not checking:
                self.tracking_hint_var.set("")
                messagebox.showwarning(
                    "No setup was saved",
                    "idtracker.ai was closed without pressing \"Save setup "
                    "and close\", so there is no setup yet."
                    + ("" if exit_code == 0 else
                       "\n\nidtracker.ai reported an error - see the output "
                       "box on the Tracking tab."))
                return
            self._tracking_offer_next_check(setup, just_saved=saved)

        self.tracking_hint_var.set(
            f"idtracker.ai is open on {video.name}. "
            + ("If it looks right, just close it. If you change anything, "
               "press \"Save setup and close\"." if checking else
               "When it looks right, press \"Save setup and close\"."))
        self._tracking_background(work, finished)

    def _tracking_offer_next_check(self, setup: Path, just_saved: bool):
        """One setup serves every video, so offer to look at it on the next
        video it has not been opened on. Thresholds that suit one recording
        can miss fish in a darker or brighter one."""
        checked = self._tracking_checked.get(setup.name, set())
        remaining = [v for v in self._tracking_videos if v not in checked]
        done = len(self._tracking_videos) - len(remaining)
        status = (f"Setup saved: {setup.name}." if just_saved
                  else f"Setup unchanged: {setup.name}.")
        if not remaining:
            self.tracking_hint_var.set(
                f"{status} Checked on all {done} video(s).")
            return

        self.tracking_hint_var.set(
            f"{status} Checked on {done} of {len(self._tracking_videos)} "
            "video(s).")
        next_video = remaining[0]
        if messagebox.askyesno(
                "Check the setup on the next video?",
                f"{status}\n\nThe same setup is used for every video, so it "
                "is worth a quick look at each one.\n\n"
                f"Open it on:\n    {next_video.name}\n\n"
                "Check the fish are all detected. If it looks right, just "
                "close the window. If you adjust anything, press \"Save "
                "setup and close\" - the change then applies to all videos."):
            index = self._tracking_videos.index(next_video)
            self.tracking_videos_tree.selection_set(str(index))
            self._tracking_open_setup_window(next_video, setup, load=setup)

    # =========================================================================
    # TRACKING EVERY VIDEO IN THE FOLDER
    # =========================================================================

    def _tracking_track_all(self):
        """Track each untracked video in turn with the selected setup."""
        setup = self._tracking_selected_setup()
        todo = self._tracking_untracked()
        if setup is None or not todo:
            return
        restarting = [v for v in todo
                      if tracking.tracking_status(v) == tracking.INCOMPLETE]
        listing = "\n".join(f"    {v.name}" for v in todo[:10])
        if len(todo) > 10:
            listing += f"\n    ... and {len(todo) - 10} more"
        if not messagebox.askokcancel(
                "Track videos",
                f"Track {len(todo)} video(s) with the setup \"{setup.name}\"?"
                f"\n\n{listing}\n\n"
                "Videos are tracked one after another and each can take a "
                "long time. Keep the laptop on and plugged in. You can leave "
                "this window open and come back."
                + (f"\n\n{len(restarting)} of these were started before and "
                   "did not finish; they start again from the beginning."
                   if restarting else "")):
            return

        self._tracking_live.clear()
        self._tracking_batch_running = True
        self.tracking_hint_var.set("")

        def work(log):
            outcomes = []
            for index, video in enumerate(todo):
                if self._tracking_stop:
                    break
                self._tracking_post(
                    lambda v=video, i=index: self._tracking_video_started(
                        v, i, len(todo)))
                log("")
                log(f"========== {video.name} ==========")
                outcome = tracking.run_tracking(
                    video, setup, log, lambda: self._tracking_stop)
                outcomes.append(outcome)
                self._tracking_post(
                    lambda o=outcome: self._tracking_video_finished(o))
            return outcomes

        self._tracking_background(
            work, lambda outcomes: self._tracking_batch_done(todo, outcomes))
        self.root.after(1000, self._tracking_tick)

    def _tracking_video_started(self, video: Path, index: int, total: int):
        self._tracking_live[video] = "running"
        self._tracking_progress = (index, total, video, time.monotonic())
        self._tracking_refresh()
        self._tracking_show_progress()

    def _tracking_video_finished(self, outcome: "tracking.TrackOutcome"):
        if outcome.ok:
            self._tracking_live.pop(outcome.video, None)
        else:
            self._tracking_live[outcome.video] = (
                "stopped" if outcome.stopped else "failed")
        self._tracking_refresh()

    def _tracking_show_progress(self):
        if self._tracking_progress is None:
            return
        index, total, video, started = self._tracking_progress
        minutes, seconds = divmod(int(time.monotonic() - started), 60)
        self.tracking_progress_var.set(
            ("Stopping... " if self._tracking_stop else "")
            + f"Tracking {index + 1} of {total}: {video.name}  -  "
              f"{minutes}:{seconds:02d} elapsed")

    def _tracking_tick(self):
        if self._tracking_batch_running:
            self._tracking_show_progress()
            self.root.after(1000, self._tracking_tick)

    def _tracking_request_stop(self):
        if not self._tracking_batch_running:
            return
        if messagebox.askyesno(
                "Stop tracking?",
                "The video being tracked now will have to start again from "
                "the beginning next time. Videos already finished are kept."
                "\n\nStop tracking?"):
            self._tracking_stop = True
            self._tracking_show_progress()
            self._tracking_update_buttons()

    def _tracking_batch_done(self, todo: List[Path],
                             outcomes: Optional[List["tracking.TrackOutcome"]]):
        outcomes = outcomes or []
        stopped = self._tracking_stop
        self._tracking_batch_running = False
        self._tracking_stop = False
        self._tracking_progress = None
        self._tracking_refresh()

        succeeded = [o.video.name for o in outcomes if o.ok]
        failed = [f"{o.video.name}: {o.reason()}"
                  for o in outcomes if not o.ok and not o.stopped]
        self.tracking_progress_var.set(
            f"{len(succeeded)} of {len(todo)} video(s) tracked."
            + (" Stopped before the rest." if stopped else ""))
        if stopped and not failed:
            return
        self._report_batch_outcome("Tracking", len(todo), succeeded, failed, [])

    # =========================================================================
    # HANDING TRACKED SESSIONS TO THE ANALYSIS TABS
    # =========================================================================

    def _tracking_load_sessions(self):
        """Load every tracked session in the folder, then show the data tab."""
        loaded, already, failed = [], [], []
        for video in self._tracking_tracked():
            nickname = video.stem
            if nickname in self.loaded_files:
                already.append(nickname)
                continue
            try:
                self._add_session(tracking.session_folder_for(video), nickname)
                loaded.append(nickname)
            except Exception as exc:
                failed.append(f"{nickname}: {exc}")

        text = f"Loaded {len(loaded)} session(s)."
        if already:
            text += self._format_outcome_list("Already loaded:", already)
        if failed:
            text += self._format_outcome_list("Could not be loaded:", failed)
            messagebox.showerror("Some sessions could not be loaded", text)
        else:
            messagebox.showinfo(
                "Sessions loaded",
                text + "\n\nNext: set the calibration, then press "
                       "\"Run All Analysis\".")
        if loaded or already:
            self.notebook.select(self.data_tab_frame)

    # =========================================================================
    # RUNNING IDTRACKER.AI WITHOUT FREEZING THE WINDOW
    # =========================================================================

    def _tracking_post(self, action: Callable[[], None]):
        """From the worker thread: run `action` on the main thread."""
        self._tracking_queue.put(("call", action))

    def _tracking_background(self, work: Callable[[Callable[[str], None]], Any],
                             on_done: Callable[[Any], None]):
        """Run `work(log)` on a worker thread, then `on_done(result)` on the
        main thread. `log(line)` adds a line to the output box."""
        self._tracking_busy = True
        self._tracking_stop = False
        self._tracking_update_buttons()

        def run():
            try:
                result = work(self._tracking_queue.put)
            except Exception as exc:
                self._tracking_queue.put(f"Could not run idtracker.ai: {exc!r}")
                result = None
            self._tracking_queue.put(("finish", lambda: on_done(result)))

        self._tracking_thread = threading.Thread(target=run, daemon=True)
        self._tracking_thread.start()
        self.root.after(200, self._tracking_poll)

    def _tracking_poll(self):
        """Move queued output into the log box; finish when the work has."""
        try:
            while True:
                item = self._tracking_queue.get_nowait()
                if isinstance(item, str):
                    self._tracking_log(item)
                    continue
                kind, action = item
                if kind == "finish":
                    self._tracking_busy = False
                    self._tracking_update_buttons()
                    action()
                    return
                action()
        except queue.Empty:
            pass
        self.root.after(200, self._tracking_poll)

    def _tracking_log(self, line: str):
        text = self.tracking_log_text
        text.config(state="normal")
        text.insert("end", line + "\n")
        excess = int(text.index("end-1c").split(".")[0]) - self.LOG_LINES_KEPT
        if excess > 0:
            text.delete("1.0", f"{excess + 1}.0")
        text.see("end")
        text.config(state="disabled")

    def _tracking_on_close(self):
        """Closing the app must not leave idtracker.ai running unseen, still
        holding the GPU, with nothing left to stop it."""
        if self._tracking_busy:
            if not messagebox.askyesno(
                    "idtracker.ai is still running",
                    "Closing now stops idtracker.ai. A video being tracked "
                    "will have to start again next time.\n\nClose anyway?"):
                return
            self._tracking_stop = True
            if self._tracking_thread is not None:
                self._tracking_thread.join(timeout=10)
        self.root.destroy()
