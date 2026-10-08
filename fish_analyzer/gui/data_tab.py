"""
fish_analyzer/gui/data_tab.py
=============================
Sessions & Units tab - which sessions are loaded, what unit their results are
in, and the button that runs the analysis.

One table lists every loaded session. One units choice applies to all of them:
with a fixed camera a pixel is the same physical size in every video, so one
scale keeps sessions comparable. idtracker.ai's body length is a per-video
estimate that moves with lighting and threshold, which is why it is offered as
a single value for the experiment and not taken separately from each video.
"""

from typing import List, Optional, Tuple
from pathlib import Path
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np

from ..data_structures import CalibrationSettings
from ..file_loading import TrajectoryFileLoader
from ..processing import ProcessingParameters, process_and_analyze_file
from .. import tracking
from ..shoaling import ShoalingCalculator, ShoalingParameters
from ..spatial import (SHAPELY_AVAILABLE, ThigmotaxisCalculator,
                       arena_in_units, idtrackerai_arena)
from .measure_dialog import MeasureScaleDialog, load_frame_image


class DataTabMixin:
    """
    Mixin providing the Sessions & Units tab.

    Expects the following attributes from base class:
    - self.root: tk.Tk
    - self.notebook: ttk.Notebook
    - self.loaded_files: Dict[str, LoadedTrajectoryFile]
    - self.processing_params: ProcessingParameters
    """

    SESSION_COLUMNS = (
        # id, heading, width
        ("name", "Session", 250),
        ("group", "Group", 130),
        ("reviewed", "Reviewed", 110),
        ("fish", "Fish", 50),
        ("minutes", "Minutes", 70),
        ("tracked", "Tracked (lowest fish)", 150),
        ("accuracy", "idtracker.ai accuracy", 140),
        ("body", "Body length (px)", 110),
        ("fps", "Frame rate", 80),
    )

    def _create_data_tab(self):
        """Create the tab: sessions table, units, and the run button."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Sessions & Units")
        self.data_tab_frame = tab

        #: (unit name, pixels per unit) every loaded session is calibrated in,
        #: or None before any session has been loaded.
        self._applied_units: Optional[Tuple[str, float]] = None
        #: True while the shared body length follows the loaded sessions.
        self._bl_is_auto = True

        self._create_sessions_section(tab)
        self._create_units_section(tab)
        self._create_run_section(tab)
        self._update_units_status()

    # =========================================================================
    # LAYOUT
    # =========================================================================

    def _create_sessions_section(self, parent):
        frame = tk.LabelFrame(parent, text="1. Sessions", font=("Arial", 12, "bold"))
        frame.pack(fill="both", expand=True, padx=20, pady=(10, 5))

        tk.Label(
            frame, justify=tk.LEFT, font=("Arial", 9), fg="gray", wraplength=1000,
            text="A session is one tracked video. Sessions tracked on the "
                 "Tracking tab arrive here with \"Load tracked sessions for "
                 "analysis\"; \"Add sessions...\" takes a session folder or a "
                 "folder of them. Names and groups label every plot and export. "
                 "\"Open in validator...\" is idtracker.ai's tool for checking "
                 "that fish were not swapped; saving there ticks Reviewed."
        ).pack(anchor="w", padx=10, pady=(6, 0))

        table = tk.Frame(frame)
        table.pack(fill="both", expand=True, padx=10, pady=6)
        scroll = tk.Scrollbar(table)
        scroll.pack(side="right", fill="y")
        self.sessions_tree = ttk.Treeview(
            table, columns=[c[0] for c in self.SESSION_COLUMNS],
            show="headings", height=6, selectmode="extended",
            yscrollcommand=scroll.set)
        for column, heading, width in self.SESSION_COLUMNS:
            self.sessions_tree.heading(column, text=heading)
            self.sessions_tree.column(
                column, width=width, anchor="w" if column == "name" else "center")
        self.sessions_tree.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.sessions_tree.yview)
        self.sessions_tree.bind("<Double-1>", self._on_session_double_click)

        buttons = tk.Frame(frame)
        buttons.pack(fill="x", padx=10, pady=(0, 8))
        tk.Button(buttons, text="Add sessions...",
                  command=self._browse_for_session_folder,
                  bg="lightblue", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(buttons, text="Rename...",
                  command=self._rename_session).pack(side="left", padx=5)
        tk.Button(buttons, text="Set group...",
                  command=self._set_group).pack(side="left", padx=5)
        tk.Button(buttons, text="Open in validator...",
                  command=self._review_selected_session).pack(side="left", padx=5)
        tk.Button(buttons, text="Details",
                  command=self._show_file_details).pack(side="left", padx=5)
        tk.Button(buttons, text="Remove",
                  command=self._remove_file).pack(side="left", padx=5)

    def _create_units_section(self, parent):
        frame = tk.LabelFrame(parent, text="2. Units", font=("Arial", 12, "bold"))
        frame.pack(fill="x", padx=20, pady=5)

        tk.Label(
            frame, justify=tk.LEFT, font=("Arial", 9), fg="gray", wraplength=1000,
            text="One scale is used for every session, so distances and speeds "
                 "can be compared between them. The scale is written into "
                 "every export."
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self.units_choice = tk.StringVar(value="bl")

        # --- centimetres ---
        cm_row = tk.Frame(frame)
        cm_row.pack(anchor="w", padx=10, pady=2)
        tk.Radiobutton(cm_row, text="Centimetres (best, if you can measure a "
                                    "known length in the video):",
                       variable=self.units_choice, value="cm",
                       command=self._update_units_status).pack(side="left")
        self.cm_pixels_var = tk.StringVar()
        self.cm_pixels_entry = tk.Entry(cm_row, textvariable=self.cm_pixels_var, width=9)
        self.cm_pixels_entry.pack(side="left", padx=(8, 3))
        tk.Label(cm_row, text="pixels =").pack(side="left")
        self.cm_length_var = tk.StringVar()
        self.cm_length_entry = tk.Entry(cm_row, textvariable=self.cm_length_var, width=7)
        self.cm_length_entry.pack(side="left", padx=3)
        tk.Label(cm_row, text="cm").pack(side="left")
        self.measure_button = tk.Button(
            cm_row, text="Measure on a video frame...",
            command=self._measure_scale)
        self.measure_button.pack(side="left", padx=12)

        # --- one body length for the experiment ---
        bl_row = tk.Frame(frame)
        bl_row.pack(anchor="w", padx=10, pady=2)
        tk.Radiobutton(bl_row, text="Body lengths, one value for the whole "
                                    "experiment:  1 BL =",
                       variable=self.units_choice, value="bl",
                       command=self._update_units_status).pack(side="left")
        self.bl_pixels_var = tk.StringVar()
        self.bl_pixels_entry = tk.Entry(bl_row, textvariable=self.bl_pixels_var, width=9)
        self.bl_pixels_entry.pack(side="left", padx=(8, 3))
        self.bl_pixels_entry.bind("<Key>", lambda e: self._bl_edited_by_user())
        tk.Label(bl_row, text="pixels").pack(side="left")
        self.bl_hint_label = tk.Label(bl_row, text="", font=("Arial", 9), fg="gray")
        self.bl_hint_label.pack(side="left", padx=10)

        status_row = tk.Frame(frame)
        status_row.pack(fill="x", padx=10, pady=(4, 8))
        self.apply_units_button = tk.Button(
            status_row, text="Apply units", command=self._apply_units)
        self.apply_units_button.pack(side="left", padx=5)
        self.units_status_var = tk.StringVar()
        self.units_status_label = tk.Label(
            status_row, textvariable=self.units_status_var,
            font=("Arial", 10, "bold"), fg="darkgreen", justify=tk.LEFT)
        self.units_status_label.pack(side="left", padx=10)

        for var in (self.cm_pixels_var, self.cm_length_var, self.bl_pixels_var):
            var.trace_add("write", lambda *_: self._update_units_status())

    def _create_run_section(self, parent):
        frame = tk.LabelFrame(parent, text="3. Analyse", font=("Arial", 12, "bold"))
        frame.pack(fill="x", padx=20, pady=(5, 10))

        row = tk.Frame(frame)
        row.pack(fill="x", padx=10, pady=8)

        # Kept as an attribute so _with_progress can disable it during a run.
        self.run_analysis_button = tk.Button(
            row, text="Run All Analysis",
            command=self._run_analysis_and_switch_tab,
            bg="lightgreen", font=("Arial", 14, "bold"))
        self.run_analysis_button.pack(side="left", padx=5)

        freeze = tk.Frame(row)
        freeze.pack(side="left", padx=25)
        tk.Label(freeze, text="A fish counts as frozen while slower than").pack(side="left")
        self.rest_threshold_var = tk.StringVar(value="0.5")
        tk.Entry(freeze, textvariable=self.rest_threshold_var, width=7).pack(side="left", padx=5)
        self.rest_unit_label = tk.Label(freeze, text="BL/s")
        self.rest_unit_label.pack(side="left")

        # Progress bar (hidden until analysis runs)
        self.analysis_progress = ttk.Progressbar(
            row, orient="horizontal", mode="determinate", length=300)
        self.analysis_progress.pack(side="left", padx=10)
        self.analysis_progress.pack_forget()

        tk.Label(
            frame, justify=tk.LEFT, font=("Arial", 9), fg="gray",
            text="Analyses every session: activity, time near the wall and "
                 "shoaling. The Results and Shoaling tabs fill in when it finishes."
        ).pack(anchor="w", padx=10, pady=(0, 8))

    # =========================================================================
    # DISCARDING RESULTS THAT NO LONGER MATCH THEIR SESSION
    # =========================================================================

    #: Result slots cached on LoadedTrajectoryFile that a calibration change
    #: invalidates. See _invalidate_results_for().
    _RESULT_SLOTS = ('processed_data', 'shoaling_results', 'thigmotaxis_results')

    def _invalidate_results_for(self, nicknames) -> list:
        """Discard cached analysis results for the given files.

        Metrics are scaled by calibration.scale_factor at *processing* time
        (processing.py), but the unit label is read at *export* time
        (export.py). Keeping results across a calibration change would
        therefore export the old numbers under the new unit name — silently.

        Returns the nicknames that actually had results to discard, so the
        caller can tell the user what was thrown away.
        """
        cleared = []
        for nickname in nicknames:
            loaded_file = self.loaded_files.get(nickname)
            if loaded_file is None:
                continue

            had_results = any(
                getattr(loaded_file, slot, None) is not None
                for slot in self._RESULT_SLOTS
            )
            for slot in self._RESULT_SLOTS:
                setattr(loaded_file, slot, None)

            if had_results:
                cleared.append(nickname)

        if cleared:
            # Clear the [#]=Analyzed markers in the spatial file list.
            self._update_spatial_files_list()
            self._refresh_result_tabs()

        return cleared

    def _purge_file_state(self, nickname: str):
        """Drop every piece of per-file state keyed by `nickname`.

        These side dictionaries are keyed by nickname rather than held on the
        LoadedTrajectoryFile, so removing or replacing a file used to leave
        them behind: the stale video reader was still displayed, and its
        cv2.VideoCapture was never released. Call this whenever a nickname stops referring to the session
        it originally referred to.
        """
        reader = self.video_readers.pop(nickname, None)
        if reader is not None:
            try:
                reader.close()
            except Exception as e:
                # Releasing a capture must never block removing a file.
                print(f"Note: could not close video for '{nickname}': {e}")

        self.file_arena_definitions.pop(nickname, None)
        self.file_roi_definitions.pop(nickname, None)
        self.file_groups.pop(nickname, None)

        # The spatial tab holds the arena currently being edited by value, not
        # by lookup, so it has to be dropped too.
        if self.current_arena_file == nickname:
            self.current_arena_file = None
            self.arena_definition = None
            self.arena_vertices = []

    @staticmethod
    def _invalidation_notice(cleared: list) -> str:
        """Message fragment naming the results a calibration change discarded."""
        if not cleared:
            return ""
        if len(cleared) == 1:
            what = f"'{cleared[0]}'"
        else:
            what = f"{len(cleared)} file(s)"
        return (f"\n\nExisting analysis results for {what} were discarded, "
                f"because metrics computed under the old calibration cannot "
                f"be relabelled with the new unit.\nRe-run the analysis.")

    # =========================================================================
    # SESSIONS
    # =========================================================================

    def _browse_for_session_folder(self):
        """Pick a session folder, or a folder of them, and add what is there."""
        folder = filedialog.askdirectory(
            title="Select a session folder, or a folder containing sessions",
            mustexist=True)
        if folder:
            self._add_path(Path(folder))

    @staticmethod
    def _is_session_folder(folder: Path) -> bool:
        return ((folder / "trajectories" / "trajectories.npy").is_file()
                or (folder / "trajectories.npy").is_file())

    @staticmethod
    def _session_name(path: Path) -> str:
        if path.is_dir():
            return path.name[8:] if path.name.startswith("session_") else path.name
        return path.stem

    def _add_path(self, path: Path) -> int:
        """Add what `path` holds and return how many sessions were added.

        `path` may be one session folder, a folder containing several (an
        experiment folder, as the Tracking tab leaves it), or a bare
        trajectories .npy file. Sessions are named after their folders.
        Failures are shown, not raised.
        """
        path = Path(path)
        if path.is_file() and path.suffix == ".npy":
            sessions = [path]
        elif path.is_dir() and self._is_session_folder(path):
            sessions = [path]
        elif path.is_dir():
            sessions = sorted((d for d in path.iterdir()
                               if d.is_dir() and self._is_session_folder(d)),
                              key=lambda d: d.name.lower())
        else:
            sessions = []
        if not sessions:
            messagebox.showerror(
                "No sessions found",
                "Choose an idtracker.ai session folder (for example "
                "session_MyVideo), or a folder that contains session folders."
                f"\n\nGot: {path}")
            return 0

        added, skipped, failed = [], [], []
        for session in sessions:
            nickname = self._session_name(session)
            if nickname in self.loaded_files:
                if len(sessions) > 1 or not messagebox.askyesno(
                        "Already loaded",
                        f"A session called '{nickname}' is already loaded. "
                        "Replace it?"):
                    skipped.append(nickname)
                    continue
            try:
                self._add_session(session, nickname)
                added.append(nickname)
            except Exception as e:
                failed.append(f"{nickname}: {e}")

        if failed:
            messagebox.showerror(
                "Some sessions could not be loaded",
                f"Loaded {len(added)} of {len(sessions)}."
                + self._format_outcome_list("Could not be loaded:", failed))
        elif len(sessions) > 1:
            messagebox.showinfo(
                "Sessions loaded",
                f"Loaded {len(added)} session(s) from {path.name}."
                + (self._format_outcome_list("Already loaded, left as they were:",
                                             skipped) if skipped else ""))
        return len(added)

    def _add_session(self, path: Path, nickname: str):
        """Load a session under `nickname`, give it the units in use, and show
        it in every list.

        No dialogs, and it raises on failure, so both the Add button and the
        Tracking tab's load-everything button can use it.
        """
        path = Path(path)
        if path.is_dir():
            loaded_file = TrajectoryFileLoader.load_from_session_folder(path, nickname)
        else:
            loaded_file = TrajectoryFileLoader.load_file(path, nickname)
        if nickname in self.loaded_files:
            # Purge only after the new session loads, so a failed load
            # leaves the existing one intact.
            self._purge_file_state(nickname)
        self.loaded_files[nickname] = loaded_file

        self._sessions_changed()
        self._update_inspector_file_dropdown()
        return loaded_file

    def _sessions_changed(self):
        """A session was added or removed: keep every session on one scale."""
        if self._bl_is_auto:
            self._fill_shared_body_length()
        if self._applied_units is not None and not (
                self._bl_is_auto and self._applied_units[0] == "BL"):
            # A scale has been chosen; a newcomer simply joins it.
            unit, pixels_per_unit = self._applied_units
            self._set_calibration(unit, pixels_per_unit)
        elif self.loaded_files:
            # Nothing chosen yet, or the automatic body length just moved
            # because the set of sessions did.
            self._apply_units(announce=False)
        self._update_sessions_table()
        self._update_units_status()

    def _selected_sessions(self) -> List[str]:
        return [n for n in self.sessions_tree.selection() if n in self.loaded_files]

    def _selected_session(self) -> Optional[str]:
        selected = self._selected_sessions()
        return selected[0] if selected else None

    def _group_of(self, nickname: str) -> str:
        """The group a session is exported under: the one set here, or
        failing that its name without a trailing number."""
        return self.file_groups.get(nickname) or self._auto_detect_group(nickname)

    def _update_sessions_table(self):
        tree = self.sessions_tree
        selected = self._selected_sessions()
        tree.delete(*tree.get_children())
        for nickname, loaded_file in self.loaded_files.items():
            metadata = loaded_file.metadata
            tracked = 100.0 * np.mean(
                ~np.isnan(loaded_file.trajectories[..., 0]), axis=0)
            tree.insert("", "end", iid=nickname, values=(
                nickname,
                self._group_of(nickname),
                self._reviewed_mark(loaded_file),
                loaded_file.n_fish,
                f"{loaded_file.duration_minutes:.1f}",
                f"{tracked.mean():.1f}%  ({tracked.min():.1f}%)",
                f"{metadata.estimated_accuracy * 100:.1f}%",
                f"{metadata.body_length:.1f}",
                f"{metadata.frames_per_second:.2f} fps",
            ))
        tree.selection_set([n for n in selected if n in self.loaded_files])

    # --- names and groups ---

    def _on_session_double_click(self, event):
        """Double-click a name to rename it, a group to change it."""
        row = self.sessions_tree.identify_row(event.y)
        if not row:
            return
        self.sessions_tree.selection_set(row)
        if self.sessions_tree.identify_column(event.x) == "#2":
            self._set_group()
        else:
            self._rename_session()

    def _rename_session(self):
        """Give the selected session a shorter name."""
        selected = self._selected_sessions()
        if len(selected) != 1:
            messagebox.showinfo("Rename", "Select one session in the table first.")
            return
        old = selected[0]
        new = simpledialog.askstring(
            "Rename session",
            "A short name for this session. It labels the plots and exports:",
            initialvalue=old, parent=self.root)
        if new is None:
            return
        new = new.strip()
        if not new or new == old:
            return
        if new in self.loaded_files:
            messagebox.showerror(
                "Rename", f"Another session is already called '{new}'.")
            return
        self._rename(old, new)

    def _rename(self, old: str, new: str):
        """Change a session's name everywhere it is used as a key.

        Results, arenas, groups and the open video all follow the session;
        nothing is recomputed or discarded.
        """
        # Rebuilt rather than popped and re-added, to keep the table order.
        self.loaded_files = {
            (new if name == old else name): loaded
            for name, loaded in self.loaded_files.items()}
        self.loaded_files[new].nickname = new

        for keyed in (self.file_arena_definitions,
                      self.file_roi_definitions, self.video_readers):
            if old in keyed:
                keyed[new] = keyed.pop(old)
        # A group that was chosen stays. One that was only ever derived from
        # the name is derived again, so "control1" lands in "control" and not
        # in a group named after the camera's original file name.
        if old in self.file_groups:
            self.file_groups[new] = self.file_groups.pop(old)

        if self.current_arena_file == old:
            self.current_arena_file = new
        if self.inspector_file_var.get() == old:
            self.inspector_file_var.set(new)

        self._update_sessions_table()
        self.sessions_tree.selection_set(new)
        self._refresh_session_lists()

    def _set_group(self):
        """Put the selected session(s) in an experimental group."""
        selected = self._selected_sessions()
        if not selected:
            messagebox.showinfo(
                "Set group", "Select one or more sessions in the table first.")
            return
        label = simpledialog.askstring(
            "Set group",
            f"Group for the {len(selected)} selected session(s), for example "
            "control or treated.\nSessions with the same group are pooled "
            "in group plots and share a Group value in exports:",
            initialvalue=self._group_of(selected[0]), parent=self.root)
        if label is None:
            return
        for nickname in selected:
            if label.strip():
                self.file_groups[nickname] = label.strip()
            else:
                self.file_groups.pop(nickname, None)
        self._update_sessions_table()
        self._refresh_result_tabs()

    def _refresh_session_lists(self):
        """Redraw every other tab's list of sessions."""
        self._update_inspector_file_dropdown()
        self._update_analysis_files_listbox()
        self._update_spatial_files_list()
        self._refresh_result_tabs()

    @staticmethod
    def _session_folder_of(loaded_file) -> Path:
        # .../session_X/trajectories/trajectories.npy -> .../session_X
        return Path(loaded_file.file_path).parent.parent

    def _reviewed_mark(self, loaded_file) -> str:
        """A tick and the date, once the session has been saved from
        idtracker.ai's validator."""
        reviewed = tracking.session_reviewed_on(self._session_folder_of(loaded_file))
        return f"\u2713 {reviewed}" if reviewed else ""

    def _review_selected_session(self):
        """Open idtracker.ai's validator on the selected session."""
        selected = self._selected_sessions()
        if len(selected) != 1:
            messagebox.showinfo(
                "Review tracking", "Select one session in the table first.")
            return
        self._review_session(
            self._session_folder_of(self.loaded_files[selected[0]]), selected[0])

    def _show_file_details(self):
        """Show detailed information about the selected session."""
        nickname = self._selected_session()
        if nickname is None:
            messagebox.showinfo("Details", "Select a session in the table first.")
            return
        messagebox.showinfo(f"Session: {nickname}",
                            self.loaded_files[nickname].summary())

    def _remove_file(self):
        """Remove the selected session."""
        nickname = self._selected_session()
        if nickname is None:
            messagebox.showinfo("Remove", "Select a session in the table first.")
            return
        if not messagebox.askyesno("Remove session", f"Remove '{nickname}'?"):
            return
        del self.loaded_files[nickname]
        self._purge_file_state(nickname)
        self._sessions_changed()
        self._update_inspector_file_dropdown()

    # =========================================================================
    # UNITS
    # =========================================================================

    def _session_body_lengths(self) -> List[float]:
        return [f.metadata.body_length for f in self.loaded_files.values()]

    def _fill_shared_body_length(self):
        """Suggest one body length: the median of the loaded sessions'."""
        lengths = self._session_body_lengths()
        self.bl_pixels_var.set(f"{float(np.median(lengths)):.1f}" if lengths else "")
        self._bl_is_auto = True

    def _bl_edited_by_user(self):
        # A typed value is the user's and must not be replaced when another
        # session is loaded.
        self._bl_is_auto = False

    def _units_from_controls(self) -> Tuple[str, float]:
        """(unit name, pixels per unit) as the controls stand.

        Raises ValueError with a message fit to show the user.
        """
        def number(text: str, what: str) -> float:
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"{what} is not a number.") from None
            if value <= 0:
                raise ValueError(f"{what} must be more than zero.")
            return value

        if self.units_choice.get() == "cm":
            if not self.cm_pixels_var.get().strip() or not self.cm_length_var.get().strip():
                raise ValueError(
                    "Centimetres needs a scale. Type how many pixels a known "
                    "length covers, or press \"Measure on a video frame...\".")
            pixels = number(self.cm_pixels_var.get(), "The pixel length")
            length = number(self.cm_length_var.get(), "The length in cm")
            return "cm", pixels / length

        if not self.bl_pixels_var.get().strip():
            raise ValueError("Load a session, or type the body length in pixels.")
        return "BL", number(self.bl_pixels_var.get(), "The body length")

    def _set_calibration(self, unit: str, pixels_per_unit: float) -> List[str]:
        """Calibrate every session in `unit`. Returns the sessions whose
        scale changed, because their results are no longer valid."""
        changed = []
        for nickname, loaded_file in self.loaded_files.items():
            old = loaded_file.calibration
            if (old.unit_name, round(old.pixels_per_unit, 6)) != (unit, round(pixels_per_unit, 6)):
                changed.append(nickname)
            loaded_file.calibration = CalibrationSettings(
                pixels_per_unit=pixels_per_unit, unit_name=unit,
                frame_rate=loaded_file.metadata.frames_per_second)
            # An arena is kept in the session's unit as well as in pixels, so
            # it has to follow a change of unit or it would sit at the wrong
            # size against the trajectories.
            arena = self.file_arena_definitions.get(nickname)
            if arena is not None:
                self.file_arena_definitions[nickname] = arena_in_units(
                    arena.vertices_pixels, loaded_file)
        return changed

    def _apply_units(self, announce: bool = True) -> bool:
        """Put every session on the scale the controls describe.

        Results computed under another scale are discarded: they are scaled
        when computed but labelled when exported, so keeping them would export
        old numbers under the new unit.
        """
        try:
            unit, pixels_per_unit = self._units_from_controls()
        except ValueError as e:
            if announce:
                messagebox.showerror("Units", str(e))
            return False

        previous = self._applied_units
        # Only across a change of unit. A body length that merely moved (as
        # the suggested one does while sessions are being loaded) leaves
        # "0.5 BL/s" meaning 0.5 BL/s.
        if previous is not None and previous[0] != unit:
            self._convert_freeze_threshold(previous[1], pixels_per_unit)

        changed = self._set_calibration(unit, pixels_per_unit)
        cleared = self._invalidate_results_for(changed)
        self._applied_units = (unit, pixels_per_unit)
        self.rest_unit_label.config(text=f"{unit}/s")
        self._update_sessions_table()
        self._update_units_status()
        if cleared:
            # Always said, even when triggered by loading a session: results
            # disappearing from the other tabs unexplained is worse.
            messagebox.showinfo(
                "Units applied",
                f"All sessions are now in {unit} "
                f"({pixels_per_unit:.2f} pixels per {unit})."
                + self._invalidation_notice(cleared))
        return True

    def _convert_freeze_threshold(self, old_pixels_per_unit: float,
                                  new_pixels_per_unit: float):
        """Keep the freeze threshold at the same physical speed.

        0.5 BL/s and 0.5 cm/s are very different speeds; leaving the number
        alone when the unit changes would silently redefine "frozen".
        """
        try:
            threshold = float(self.rest_threshold_var.get())
        except ValueError:
            return
        converted = threshold * old_pixels_per_unit / new_pixels_per_unit
        self.rest_threshold_var.set(f"{converted:.3g}")

    def _update_units_status(self):
        """Say what scale is in use, and whether the controls differ from it."""
        is_cm = self.units_choice.get() == "cm"
        for entry in (self.cm_pixels_entry, self.cm_length_entry):
            entry.config(state="normal" if is_cm else "disabled")
        self.bl_pixels_entry.config(state="disabled" if is_cm else "normal")
        self.measure_button.config(state="normal" if is_cm else "disabled")

        lengths = self._session_body_lengths()
        if len(lengths) > 1:
            self.bl_hint_label.config(
                text=f"(sessions measure {min(lengths):.1f} to {max(lengths):.1f} px; "
                     "approximate, it depends on lighting)")
        elif lengths:
            self.bl_hint_label.config(
                text="(from idtracker.ai's outline of the fish; approximate)")
        else:
            self.bl_hint_label.config(text="")

        if not self.loaded_files:
            self.units_status_var.set("No sessions loaded yet.")
            self.units_status_label.config(fg="gray40")
            self.apply_units_button.config(state="disabled")
            return

        try:
            wanted = self._units_from_controls()
        except ValueError as e:
            self.units_status_var.set(str(e))
            self.units_status_label.config(fg="#8a4b00")
            self.apply_units_button.config(state="disabled")
            return

        unit, pixels_per_unit = self._applied_units or wanted
        in_use = f"In use: {unit}, {pixels_per_unit:.2f} pixels per {unit}"
        if self._applied_units == wanted:
            self.units_status_var.set(
                f"{in_use}, for all {len(self.loaded_files)} session(s).")
            self.units_status_label.config(fg="darkgreen")
            self.apply_units_button.config(state="disabled")
        else:
            self.units_status_var.set(
                f"{in_use}.  Changed to {wanted[0]}, {wanted[1]:.2f} pixels per "
                f"{wanted[0]}: not applied yet.")
            self.units_status_label.config(fg="#8a4b00")
            self.apply_units_button.config(state="normal")

    def _measure_scale(self):
        """Open a video frame to click two points a known distance apart."""
        nickname = self._selected_session() or next(iter(self.loaded_files), None)
        if nickname is None:
            messagebox.showinfo(
                "Measure", "Load a session first, so there is a frame to measure on.")
            return
        loaded_file = self.loaded_files[nickname]
        image = load_frame_image(loaded_file)
        if image is None:
            messagebox.showinfo(
                "No frame to measure on",
                f"No video or background image was found for '{nickname}'.\n\n"
                "Type the pixel length and the length in cm instead.")
            return

        def use(pixels: float, centimetres: float):
            self.cm_pixels_var.set(f"{pixels:.1f}")
            self.cm_length_var.set(f"{centimetres:g}")
            self.units_choice.set("cm")
            self._update_units_status()

        MeasureScaleDialog(self.root, image, nickname, use)

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def _run_analysis_and_switch_tab(self):
        """Analyse every session, then show the Results tab."""
        if not self._run_analysis():
            return
        self.notebook.select(self.results_tab_frame)

    def _run_analysis(self) -> bool:
        """Run individual trajectory analysis on all loaded files.

        Returns False if it could not start (no sessions, or the units or the
        threshold are not usable).
        """
        if not self.loaded_files:
            messagebox.showerror("No sessions", "Load at least one session first.")
            return False

        # Whatever the units controls say is what gets analysed; an edit the
        # user forgot to apply must not be silently ignored.
        if not self._apply_units():
            return False

        try:
            params = self._get_processing_parameters_from_gui()
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return False

        # Record the parameters actually used — the Methods tab reads these to
        # describe the run, so they must not stay at the constructor defaults.
        self.processing_params = params

        total = len(self.loaded_files)
        # Per-file outcomes, so a partial failure is reported honestly instead
        # of the run ending with no dialog and stale plots still on screen.
        succeeded, failed, degraded = [], [], []

        for nickname in self._with_progress(
                list(self.loaded_files.keys()), label="Processing",
                button=self.run_analysis_button,
                progressbar=self.analysis_progress):
            loaded_file = self.loaded_files[nickname]
            try:
                fish_list = process_and_analyze_file(loaded_file, params)
                loaded_file.processed_data = fish_list
                succeeded.append(nickname)

                if len(fish_list) < loaded_file.n_fish:
                    degraded.append(
                        f"{nickname}: only {len(fish_list)} of "
                        f"{loaded_file.n_fish} fish analyzed"
                    )
                for problem in (self._analyse_wall_time(nickname),
                                self._analyse_shoaling(nickname)):
                    if problem:
                        degraded.append(f"{nickname}: {problem}")
            except Exception as e:
                # One bad file must not abandon the rest, and must not leave a
                # previous run's results attached pretending to be current.
                loaded_file.processed_data = None
                failed.append(f"{nickname}: {e}")
                print(f"[FAILED] {nickname}: {e}")
                traceback.print_exc()

        # Update file lists and auto-select all
        self._update_analysis_files_listbox()
        for i in range(self.analysis_files_listbox.size()):
            self.analysis_files_listbox.selection_set(i)

        if succeeded:
            self._update_analysis_visualizations()
        # The Video Inspector draws shoaling overlays from these results.
        if hasattr(self, '_inspector_rebuild_needed'):
            self._inspector_rebuild_needed()
        self._refresh_result_tabs()

        self.set_status(
            f"Analysis complete: {len(succeeded)} of {total} file(s) processed"
        )
        self._report_batch_outcome("Individual Analysis", total, succeeded,
                                   failed, degraded)
        return True

    def _refresh_result_tabs(self):
        """Redraw every tab that shows results."""
        self._update_results()
        self._update_shoaling()

    def _analyse_shoaling(self, nickname: str) -> Optional[str]:
        """Shoaling distances for one session. Returns a problem to report,
        or None. A lone fish has no neighbours, so it is simply skipped."""
        loaded_file = self.loaded_files[nickname]
        loaded_file.shoaling_results = None
        if loaded_file.n_fish < 2:
            return None
        # One sample a second, whatever the frame rate.
        params = ShoalingParameters(sample_interval_frames=max(
            1, int(round(loaded_file.metadata.frames_per_second))))
        self.shoaling_params = params
        try:
            loaded_file.shoaling_results = ShoalingCalculator(
                loaded_file, params).calculate()
        except Exception as e:
            print(f"[FAILED] shoaling for {nickname}: {e}")
            return f"shoaling could not be computed ({e})"
        return None

    def _analyse_wall_time(self, nickname: str) -> Optional[str]:
        """Time near the wall for one session. Returns a problem to report,
        or None.

        The arena is the one drawn on the Spatial Analysis tab if there is
        one, and otherwise the outline drawn in idtracker.ai's setup window,
        so a session tracked through this app needs no redrawing. With neither
        the measure is simply left empty.
        """
        loaded_file = self.loaded_files[nickname]
        loaded_file.thigmotaxis_results = None
        if not SHAPELY_AVAILABLE:
            return None
        arena = self.file_arena_definitions.get(nickname)
        if arena is None:
            arena = idtrackerai_arena(loaded_file)
            if arena is None:
                return None
            self.file_arena_definitions[nickname] = arena
        try:
            loaded_file.thigmotaxis_results = ThigmotaxisCalculator(
                loaded_file, arena).calculate()
        except Exception as e:
            print(f"[FAILED] wall time for {nickname}: {e}")
            return f"time near the wall could not be computed ({e})"
        return None

    def _get_processing_parameters_from_gui(self) -> ProcessingParameters:
        """Read processing parameters from the GUI inputs."""
        try:
            rest_threshold = float(self.rest_threshold_var.get())
        except ValueError:
            raise ValueError("The freeze speed threshold is not a number.") from None

        params = ProcessingParameters(
            min_valid_points=10,
            min_valid_percentage=0.01,
            rest_speed_threshold=rest_threshold,
        )
        params.validate()
        return params
