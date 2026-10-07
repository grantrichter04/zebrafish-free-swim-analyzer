"""
fish_analyzer/gui/tracking_tab.py
=================================
Tracking Tab - choose a folder of videos and make an idtracker.ai setup for it.

The work itself is in fish_analyzer/tracking.py. This file is the widgets, and
the plumbing that lets a separate idtracker.ai process report back to tkinter:
the process runs on a worker thread, its output goes into a queue, and the
main thread drains that queue on a timer.
"""
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable, Dict, List, Optional, Set

from .. import tracking


class TrackingTabMixin:
    """
    Mixin providing Tracking Tab functionality.

    Expects the following attributes from base class:
    - self.root: tk.Tk
    - self.notebook: ttk.Notebook
    """

    LOG_LINES_KEPT = 2000

    def _create_tracking_tab(self):
        """Create the tracking tab: videos, setup, and idtracker.ai's output."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Tracking")

        self._tracking_folder: Optional[Path] = None
        self._tracking_videos: List[Path] = []
        self._tracking_setups: List[Path] = []
        self._tracking_busy = False
        # Setup file name -> the videos it has been opened on this session.
        self._tracking_checked: Dict[str, Set[Path]] = {}
        self._tracking_queue: "queue.Queue" = queue.Queue()

        tk.Label(
            tab, justify=tk.LEFT, anchor="w", font=("Arial", 9), fg="gray30",
            text="Start here with new videos: idtracker.ai turns each video "
                 "into a tracked session.\nAlready have tracked sessions? Go "
                 "straight to \"Data Setup & Calibration\"."
        ).pack(fill="x", padx=20, pady=(10, 0))

        self._create_tracking_videos_section(tab)
        self._create_tracking_setup_section(tab)
        self._create_tracking_log_section(tab)

        if not tracking.idtrackerai_available():
            self.tracking_hint_var.set(
                "idtracker.ai is not installed in this environment, so "
                "tracking is unavailable. Run install.bat to add it.")
        self._tracking_update_buttons()

    # =========================================================================
    # LAYOUT
    # =========================================================================

    def _create_tracking_videos_section(self, parent):
        frame = tk.LabelFrame(parent, text="1. Videos", font=("Arial", 12, "bold"))
        frame.pack(fill="x", padx=20, pady=10)

        controls = tk.Frame(frame)
        controls.pack(fill="x", pady=10, padx=10)
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
            frame, columns=("video", "status"), show="headings", height=6,
            selectmode="browse")
        self.tracking_videos_tree.heading("video", text="Video")
        self.tracking_videos_tree.heading("status", text="Status")
        self.tracking_videos_tree.column("video", width=600, anchor="w")
        self.tracking_videos_tree.column("status", width=140, anchor="w")
        self.tracking_videos_tree.pack(fill="x", padx=10, pady=(0, 10))

    def _create_tracking_setup_section(self, parent):
        frame = tk.LabelFrame(parent, text="2. Setup", font=("Arial", 12, "bold"))
        frame.pack(fill="x", padx=20, pady=10)

        tk.Label(
            frame, justify=tk.LEFT, font=("Arial", 9), fg="gray", wraplength=900,
            text="A setup holds the idtracker.ai settings for this experiment: "
                 "the thresholds, the number of fish and the arena. It is saved "
                 "as a .toml file beside the videos and used for every video "
                 "in the folder."
        ).pack(anchor="w", padx=10, pady=(8, 0))

        controls = tk.Frame(frame)
        controls.pack(fill="x", pady=10, padx=10)
        tk.Label(controls, text="Setup:").pack(side="left", padx=5)
        self.tracking_setup_var = tk.StringVar()
        self.tracking_setup_combo = ttk.Combobox(
            controls, textvariable=self.tracking_setup_var, state="readonly",
            width=40)
        self.tracking_setup_combo.pack(side="left", padx=5)
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
                 ).pack(anchor="w", padx=10, pady=(0, 8))

    def _create_tracking_log_section(self, parent):
        frame = tk.LabelFrame(parent, text="idtracker.ai output",
                              font=("Arial", 12, "bold"))
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        scroll = tk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        self.tracking_log_text = tk.Text(
            frame, height=10, wrap="none", font=("Courier", 9),
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
        self.tracking_folder_var.set(str(self._tracking_folder))
        self._tracking_refresh()

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
                        values=(video.name, tracking.tracking_status(video)))
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
            else:
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

    def _tracking_update_buttons(self):
        ready = (tracking.idtrackerai_available() and not self._tracking_busy
                 and bool(self._tracking_videos))
        self.tracking_configure_button.config(
            state="normal" if ready else "disabled")
        self.tracking_edit_button.config(
            state="normal" if ready and self._tracking_setups else "disabled")

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

        def finished(exit_code: int):
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
        self._tracking_start(
            tracking.build_configure_command(video, save_to=setup, load=load),
            video.parent, finished)

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
    # RUNNING A PROCESS WITHOUT FREEZING THE WINDOW
    # =========================================================================

    def _tracking_start(self, command: List[str], cwd: Path,
                        on_done: Callable[[int], None]):
        """Run `command` on a worker thread; call `on_done(exit_code)` on the
        main thread when it ends."""
        self._tracking_busy = True
        self._tracking_update_buttons()
        self._tracking_log(f"> {' '.join(command)}")

        def work():
            try:
                code = tracking.run_process(
                    command, cwd, on_line=self._tracking_queue.put)
            except Exception as exc:
                self._tracking_queue.put(f"Could not start idtracker.ai: {exc}")
                code = -1
            self._tracking_queue.put(("done", code, on_done))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(200, self._tracking_poll)

    def _tracking_poll(self):
        """Move queued output into the log box; finish when the process has."""
        try:
            while True:
                item = self._tracking_queue.get_nowait()
                if isinstance(item, tuple):
                    _, code, on_done = item
                    self._tracking_busy = False
                    self._tracking_update_buttons()
                    on_done(code)
                    return
                self._tracking_log(item)
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
