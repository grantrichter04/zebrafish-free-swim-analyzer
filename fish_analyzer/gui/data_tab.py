"""
fish_analyzer/gui/data_tab.py
=============================
Data Tab Mixin - File loading, calibration, and processing parameters.

This mixin provides all methods related to:
- Loading trajectory files
- Calibration settings
- Processing parameters
- Running the initial analysis
"""

from typing import Optional
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from ..data_structures import CalibrationSettings
from ..file_loading import TrajectoryFileLoader
from ..processing import ProcessingParameters, process_and_analyze_file


class DataTabMixin:
    """
    Mixin providing Data Tab functionality.
    
    Expects the following attributes from base class:
    - self.root: tk.Tk
    - self.notebook: ttk.Notebook
    - self.loaded_files: Dict[str, LoadedTrajectoryFile]
    - self.active_file: Optional[str]
    - self.processing_params: ProcessingParameters
    """

    def _create_data_tab(self):
        """Create the data loading and setup tab with scrolling support."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Data Setup & Calibration")

        # Create scrollable container
        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        canvas_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.scrollable_frame.bind("<Configure>", _on_frame_configure)

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        # Bind mousewheel only when cursor is over this canvas (not globally)
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # Add sections to the scrollable frame
        self._create_file_section(self.scrollable_frame)
        self._create_calibration_section(self.scrollable_frame)
        self._create_processing_parameters_section(self.scrollable_frame)

        # Main analyze button
        analyze_frame = tk.Frame(self.scrollable_frame)
        analyze_frame.pack(fill="x", padx=20, pady=20)

        tk.Button(
            analyze_frame, text="Run Individual Trajectory Analysis",
            command=self._run_analysis_and_switch_tab,
            bg="lightgreen", font=("Arial", 14, "bold"), height=2
        ).pack()

        # Progress bar (hidden until analysis runs)
        self.analysis_progress = ttk.Progressbar(
            analyze_frame, orient="horizontal", mode="determinate", length=400
        )
        self.analysis_progress.pack(pady=(5, 0))
        self.analysis_progress.pack_forget()  # Hide initially

        tk.Label(
            analyze_frame, text="Calculates speed, distance, freezing, bursting, and path straightness for each fish.\n"
                               "Shoaling and spatial analyses are run separately from their own tabs.",
            font=("Arial", 9), fg="gray", justify=tk.CENTER
        ).pack(pady=5)

    def _create_file_section(self, parent):
        """Create the file loading section."""
        file_frame = tk.LabelFrame(parent, text="Load Trajectory Files", font=("Arial", 12, "bold"))
        file_frame.pack(fill="x", padx=20, pady=10)

        # Session folder input
        controls = tk.Frame(file_frame)
        controls.pack(fill="x", pady=10, padx=10)

        tk.Label(controls, text="Session Folder:").pack(side="left", padx=5)
        self.file_path_var = tk.StringVar()
        tk.Entry(controls, textvariable=self.file_path_var, width=50).pack(side="left", padx=5)
        tk.Button(controls, text="Browse...", command=self._browse_for_session_folder,
                 bg="lightblue", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        
        # Help text
        tk.Label(file_frame, 
                text="Select an idtracker.ai session folder (e.g., session_MyExperiment). "
                     "The trajectories.npy and background.png will be found automatically.",
                font=("Arial", 9), fg="gray", wraplength=500, justify=tk.LEFT
        ).pack(anchor="w", padx=10, pady=(0, 10))

        # Loaded files list
        list_frame = tk.Frame(file_frame)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Label(list_frame, text="Loaded Files:", font=("Arial", 10, "bold")).pack(anchor="w")

        list_container = tk.Frame(list_frame)
        list_container.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")

        self.files_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set, height=6)
        self.files_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.files_listbox.yview)

        # File management buttons
        button_frame = tk.Frame(list_frame)
        button_frame.pack(fill="x", pady=5)

        tk.Button(button_frame, text="Set Active", command=self._set_active_file).pack(side="left", padx=5)
        tk.Button(button_frame, text="View Details", command=self._show_file_details).pack(side="left", padx=5)
        tk.Button(button_frame, text="Remove", command=self._remove_file).pack(side="left", padx=5)

    def _create_calibration_section(self, parent):
        """Create the calibration controls section."""
        cal_frame = tk.LabelFrame(parent, text="Calibration Settings", font=("Arial", 12, "bold"))
        cal_frame.pack(fill="x", padx=20, pady=10)

        self.calibration_info_label = tk.Label(
            cal_frame, text="Select an active file to configure calibration",
            font=("Arial", 10), fg="gray"
        )
        self.calibration_info_label.pack(pady=10)

        # Calibration method selection
        self.calibration_method = tk.StringVar(value="body_length")

        methods_frame = tk.Frame(cal_frame)
        methods_frame.pack(fill="x", padx=20, pady=5)

        tk.Label(methods_frame, text="Calibration method:", font=("Arial", 10, "bold")).pack(anchor="w", pady=5)

        # Body length option
        self.body_length_radio = tk.Radiobutton(
            methods_frame, text="Use body length from tracking data",
            variable=self.calibration_method, value="body_length",
            command=self._on_calibration_method_changed
        )
        self.body_length_radio.pack(anchor="w", padx=20)

        self.body_length_display = tk.Label(methods_frame, text="", font=("Arial", 9), fg="blue")
        self.body_length_display.pack(anchor="w", padx=40)

        # Custom calibration option
        self.custom_radio = tk.Radiobutton(
            methods_frame, text="Use custom calibration measurement",
            variable=self.calibration_method, value="custom",
            command=self._on_calibration_method_changed
        )
        self.custom_radio.pack(anchor="w", padx=20, pady=(10, 5))

        custom_inputs_frame = tk.Frame(methods_frame)
        custom_inputs_frame.pack(anchor="w", padx=40, pady=5)

        tk.Label(custom_inputs_frame, text="I measured:").grid(row=0, column=0, sticky="w", padx=5)
        self.pixels_var = tk.StringVar()
        self.pixels_entry = tk.Entry(custom_inputs_frame, textvariable=self.pixels_var, width=10)
        self.pixels_entry.grid(row=0, column=1, padx=5)
        tk.Label(custom_inputs_frame, text="pixels =").grid(row=0, column=2, padx=5)
        self.units_var = tk.StringVar(value="1")
        self.units_entry = tk.Entry(custom_inputs_frame, textvariable=self.units_var, width=10)
        self.units_entry.grid(row=0, column=3, padx=5)
        self.unit_name_var = tk.StringVar(value="cm")
        self.unit_name_entry = tk.Entry(custom_inputs_frame, textvariable=self.unit_name_var, width=8)
        self.unit_name_entry.grid(row=0, column=4, padx=5)

        tk.Label(
            custom_inputs_frame,
            text="(Example: '240 pixels = 10 cm' means 24 pixels per cm)",
            font=("Arial", 8), fg="gray"
        ).grid(row=1, column=0, columnspan=5, sticky="w", pady=2)

        self.custom_input_widgets = [self.pixels_entry, self.units_entry, self.unit_name_entry]

        # No calibration option
        self.no_calibration_radio = tk.Radiobutton(
            methods_frame, text="No calibration (keep pixel units)",
            variable=self.calibration_method, value="none",
            command=self._on_calibration_method_changed
        )
        self.no_calibration_radio.pack(anchor="w", padx=20, pady=(10, 5))

        # Frame rate input
        fps_frame = tk.Frame(cal_frame)
        fps_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(fps_frame, text="Frame rate:", font=("Arial", 10)).pack(side="left", padx=5)
        self.fps_var = tk.StringVar()
        self.fps_entry = tk.Entry(fps_frame, textvariable=self.fps_var, width=12)
        self.fps_entry.pack(side="left", padx=5)
        tk.Label(fps_frame, text="fps").pack(side="left", padx=2)
        self.fps_display = tk.Label(fps_frame, text="", font=("Arial", 9), fg="blue")
        self.fps_display.pack(side="left", padx=10)

        # Apply buttons
        tk.Button(
            cal_frame, text="Apply Calibration to Active File",
            command=self._apply_calibration, bg="lightgreen", font=("Arial", 10, "bold")
        ).pack(pady=10)

        tk.Button(
            cal_frame, text="Apply Calibration to ALL Files",
            command=self._apply_calibration_to_all, bg="orange", font=("Arial", 10, "bold")
        ).pack(pady=5)

        tk.Label(cal_frame, text="[!] This will update calibration for every loaded file",
                font=("Arial", 8), fg="darkorange").pack()

        self.calibration_summary = tk.Label(cal_frame, text="", font=("Arial", 9),
                                           fg="darkgreen", justify=tk.LEFT)
        self.calibration_summary.pack(pady=5)

        self._on_calibration_method_changed()

    def _create_processing_parameters_section(self, parent):
        """Create UI for configuring processing parameters."""
        params_frame = tk.LabelFrame(parent, text="Processing Parameters", font=("Arial", 12, "bold"))
        params_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(
            params_frame, text="Configure how trajectories are processed and analyzed",
            font=("Arial", 9), fg="gray"
        ).pack(anchor="w", padx=10, pady=5)

        settings_container = tk.Frame(params_frame)
        settings_container.pack(fill="x", padx=10, pady=10)

        params_col = tk.Frame(settings_container)
        params_col.pack(fill="x", expand=True, padx=5)

        tk.Label(params_col, text="Processing Settings:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))

        # Smoothing checkbox
        self.apply_smoothing_var = tk.BooleanVar(value=False)
        tk.Checkbutton(params_col, text="Apply Savitzky-Golay smoothing",
                      variable=self.apply_smoothing_var).pack(anchor="w")

        # Smoothing window
        smooth_frame = tk.Frame(params_col)
        smooth_frame.pack(anchor="w", padx=20, pady=2)
        tk.Label(smooth_frame, text="Window size:").pack(side="left")
        self.smoothing_window_var = tk.StringVar(value="5")
        tk.Entry(smooth_frame, textvariable=self.smoothing_window_var, width=5).pack(side="left", padx=5)
        tk.Label(smooth_frame, text="frames (must be odd)").pack(side="left")

        # Rest speed threshold
        rest_frame = tk.Frame(params_col)
        rest_frame.pack(anchor="w", pady=(10, 2))
        tk.Label(rest_frame, text="Rest speed threshold:").pack(side="left")
        self.rest_threshold_var = tk.StringVar(value="0.5")
        tk.Entry(rest_frame, textvariable=self.rest_threshold_var, width=8).pack(side="left", padx=5)
        self.rest_unit_label = tk.Label(rest_frame, text="BL/s")
        self.rest_unit_label.pack(side="left")

        # Help text
        help_text = ("Smoothing: OFF by default. Reduces tracking jitter but also dampens "
                    "real movement — can reduce measured speed by 20-40% for larval bout-based "
                    "locomotion at 30fps. Only enable for continuous adult swimming or high-fps data.\n"
                    "Rest threshold: speed below which fish are considered inactive.")
        tk.Label(params_frame, text=help_text, font=("Arial", 8), fg="gray",
                wraplength=600, justify=tk.LEFT).pack(anchor="w", padx=10, pady=5)

    # =========================================================================
    # CALIBRATION HELPER METHODS
    # =========================================================================

    def _on_calibration_method_changed(self):
        """Enable/disable custom calibration inputs based on selection."""
        method = self.calibration_method.get()
        state = tk.NORMAL if method == "custom" else tk.DISABLED
        for widget in self.custom_input_widgets:
            widget.config(state=state)

    def _browse_for_session_folder(self):
        """Open folder browser dialog and automatically load the session."""
        folder_path = filedialog.askdirectory(
            title="Select idtracker.ai Session Folder",
            mustexist=True
        )
        if folder_path:
            self.file_path_var.set(folder_path)
            self._load_session_folder(folder_path)
    
    def _load_session_folder(self, folder_path_str: str):
        """Load a session folder and prompt for nickname."""
        folder_path = Path(folder_path_str)
        
        # Generate suggested nickname from folder name
        folder_name = folder_path.name
        if folder_name.startswith('session_'):
            suggested_nickname = folder_name[8:]
        else:
            suggested_nickname = folder_name
        
        # Ask for nickname
        nickname = simpledialog.askstring(
            "File Nickname",
            f"Enter a nickname for this session:\n({folder_path.name})",
            initialvalue=suggested_nickname
        )
        if not nickname:
            return

        if nickname in self.loaded_files:
            if not messagebox.askyesno("Nickname Exists", f"'{nickname}' already exists. Replace it?"):
                return

        try:
            loaded_file = TrajectoryFileLoader.load_from_session_folder(folder_path, nickname)
            self.loaded_files[nickname] = loaded_file

            if self.active_file is None:
                self.active_file = nickname

            self._update_files_list()
            self._update_calibration_display()
            self._update_shoaling_file_dropdown()
            
            messagebox.showinfo(
                "Session Loaded", 
                f"Successfully loaded '{nickname}'!\n\n"
                f"Fish: {loaded_file.n_fish}\n"
                f"Frames: {loaded_file.n_frames}\n"
                f"Duration: {loaded_file.duration_minutes:.1f} minutes\n"
                f"Background: {'Found' if loaded_file.background_image_path else 'Not found'}"
            )
        except Exception as e:
            messagebox.showerror("Error Loading Session", f"Failed to load {folder_path.name}:\n\n{str(e)}")

    def _load_selected_file(self):
        """Load the file/folder specified in the path entry."""
        path_str = self.file_path_var.get()
        if not path_str:
            messagebox.showerror("Error", "Please select a session folder first")
            return

        path = Path(path_str)
        
        if path.is_dir():
            self._load_session_folder(path_str)
        elif path.is_file() and path.suffix == '.npy':
            self._load_legacy_npy_file(path)
        else:
            messagebox.showerror(
                "Invalid Path", 
                f"Please select either:\n"
                f"- An idtracker.ai session folder, or\n"
                f"- A trajectories.npy file\n\n"
                f"Got: {path}"
            )
    
    def _load_legacy_npy_file(self, file_path: Path):
        """Load a .npy file directly (legacy support)."""
        suggested_nickname = file_path.stem
        nickname = simpledialog.askstring(
            "File Nickname",
            "Enter a nickname for this file:",
            initialvalue=suggested_nickname
        )
        if not nickname:
            return

        if nickname in self.loaded_files:
            if not messagebox.askyesno("Nickname Exists", f"'{nickname}' already exists. Replace it?"):
                return

        try:
            loaded_file = TrajectoryFileLoader.load_file(file_path, nickname)
            self.loaded_files[nickname] = loaded_file

            if self.active_file is None:
                self.active_file = nickname

            self._update_files_list()
            self._update_calibration_display()
            self._update_shoaling_file_dropdown()
        except Exception as e:
            messagebox.showerror("Error Loading File", f"Failed to load {file_path.name}:\n\n{str(e)}")

    def _update_files_list(self):
        """Update the files listbox in the data tab."""
        self.files_listbox.delete(0, tk.END)
        for nickname in self.loaded_files.keys():
            display_text = f"> {nickname}" if nickname == self.active_file else f"  {nickname}"
            self.files_listbox.insert(tk.END, display_text)

    def _set_active_file(self):
        """Set the selected file as the active file for calibration."""
        selection = self.files_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a file from the list")
            return
        nickname = list(self.loaded_files.keys())[selection[0]]
        self.active_file = nickname
        self._update_files_list()
        self._update_calibration_display()

    def _show_file_details(self):
        """Show detailed information about the selected file."""
        selection = self.files_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a file from the list")
            return
        nickname = list(self.loaded_files.keys())[selection[0]]
        loaded_file = self.loaded_files[nickname]
        messagebox.showinfo(f"File Details: {nickname}", loaded_file.summary())

    def _remove_file(self):
        """Remove the selected file from the loaded files."""
        selection = self.files_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a file from the list")
            return
        nickname = list(self.loaded_files.keys())[selection[0]]
        if not messagebox.askyesno("Confirm Removal", f"Remove '{nickname}'?"):
            return
        del self.loaded_files[nickname]
        if self.active_file == nickname:
            self.active_file = next(iter(self.loaded_files), None)
        self._update_files_list()
        self._update_calibration_display()
        self._update_shoaling_file_dropdown()

    def _update_calibration_display(self):
        """Update the calibration section to reflect the active file."""
        if self.active_file is None or self.active_file not in self.loaded_files:
            self.calibration_info_label.config(text="Select an active file to configure calibration", fg="gray")
            self.body_length_display.config(text="")
            self.fps_display.config(text="")
            self.calibration_summary.config(text="")
            self.body_length_radio.config(state=tk.DISABLED)
            self.custom_radio.config(state=tk.DISABLED)
            self.no_calibration_radio.config(state=tk.DISABLED)
            self.fps_entry.config(state=tk.DISABLED)
            return

        loaded_file = self.loaded_files[self.active_file]
        metadata = loaded_file.metadata

        self.calibration_info_label.config(text=f"Configuring calibration for: {self.active_file}", fg="black")
        self.body_length_radio.config(state=tk.NORMAL)
        self.custom_radio.config(state=tk.NORMAL)
        self.no_calibration_radio.config(state=tk.NORMAL)
        self.fps_entry.config(state=tk.NORMAL)

        self.body_length_display.config(text=f"  (Body length from file: {metadata.body_length:.2f} pixels)")
        self.fps_var.set(f"{metadata.frames_per_second:.2f}")
        self.fps_display.config(text=f"(from file: {metadata.frames_per_second:.2f} fps)")

        current_cal = loaded_file.calibration
        summary_text = (f"Current calibration: {current_cal.unit_name}\n"
                       f"  {current_cal.pixels_per_unit:.2f} pixels per {current_cal.unit_name}")
        self.calibration_summary.config(text=summary_text)
        self.rest_unit_label.config(text=f"{current_cal.unit_name}/s")

    def _apply_calibration(self):
        """Apply calibration settings to the active file."""
        if self.active_file is None or self.active_file not in self.loaded_files:
            messagebox.showerror("Error", "No active file selected")
            return

        loaded_file = self.loaded_files[self.active_file]
        method = self.calibration_method.get()

        try:
            frame_rate = float(self.fps_var.get())
            if frame_rate <= 0:
                raise ValueError("Frame rate must be positive")
        except ValueError as e:
            messagebox.showerror("Invalid Frame Rate", f"Please enter a valid frame rate: {e}")
            return

        try:
            if method == "body_length":
                calibration = CalibrationSettings.from_body_lengths(
                    body_length_pixels=loaded_file.metadata.body_length, frame_rate=frame_rate)
            elif method == "custom":
                pixels = float(self.pixels_var.get())
                units = float(self.units_var.get())
                unit_name = self.unit_name_var.get().strip()
                if pixels <= 0 or units <= 0:
                    raise ValueError("Pixel and unit values must be positive")
                if not unit_name:
                    raise ValueError("Please specify a unit name")
                pixels_per_unit = pixels / units
                calibration = CalibrationSettings.from_physical_measurement(
                    pixels_per_unit=pixels_per_unit, unit_name=unit_name, frame_rate=frame_rate)
            elif method == "none":
                calibration = CalibrationSettings.no_calibration(frame_rate)
            else:
                return

            loaded_file.calibration = calibration
            self._update_calibration_display()
            messagebox.showinfo("Calibration Applied", f"Applied:\n\n{calibration}")
        except Exception as e:
            messagebox.showerror("Calibration Error", f"Failed:\n\n{e}")

    def _apply_calibration_to_all(self):
        """Apply calibration settings to all loaded files."""
        if not self.loaded_files:
            messagebox.showerror("No Files", "Please load at least one file first.")
            return

        file_count = len(self.loaded_files)
        if not messagebox.askyesno("Apply to All?", f"Apply calibration to all {file_count} file(s)?"):
            return

        method = self.calibration_method.get()

        try:
            frame_rate = float(self.fps_var.get())
            if frame_rate <= 0:
                raise ValueError("Frame rate must be positive")
        except ValueError as e:
            messagebox.showerror("Invalid Frame Rate", str(e))
            return

        try:
            for loaded_file in self.loaded_files.values():
                if method == "body_length":
                    calibration = CalibrationSettings.from_body_lengths(
                        body_length_pixels=loaded_file.metadata.body_length, frame_rate=frame_rate)
                elif method == "custom":
                    pixels = float(self.pixels_var.get())
                    units = float(self.units_var.get())
                    unit_name = self.unit_name_var.get().strip()
                    pixels_per_unit = pixels / units
                    calibration = CalibrationSettings.from_physical_measurement(
                        pixels_per_unit=pixels_per_unit, unit_name=unit_name, frame_rate=frame_rate)
                elif method == "none":
                    calibration = CalibrationSettings.no_calibration(frame_rate)
                loaded_file.calibration = calibration

            self._update_calibration_display()
            messagebox.showinfo("Done", f"Applied calibration to {file_count} file(s).")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def _run_analysis_and_switch_tab(self):
        """Run analysis and switch to the results tab."""
        self._run_analysis()
        self.notebook.select(1)

    def _run_analysis(self):
        """Run individual trajectory analysis on all loaded files."""
        if not self.loaded_files:
            messagebox.showerror("No Files", "Please load at least one trajectory file.")
            return

        try:
            params = self._get_processing_parameters_from_gui()
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return

        self.analysis_summary_text.delete("1.0", tk.END)
        self.analysis_summary_text.insert("1.0", f"Processing {len(self.loaded_files)} file(s)...\n\n")

        # Show and configure progress bar
        total = len(self.loaded_files)
        self.analysis_progress.pack(pady=(5, 0))
        self.analysis_progress['maximum'] = total
        self.analysis_progress['value'] = 0
        self.root.update()

        try:
            for idx, (nickname, loaded_file) in enumerate(self.loaded_files.items(), 1):
                self.set_status(f"Processing {idx}/{total}: {nickname}...")
                self.analysis_summary_text.insert(tk.END, f"Processing: {nickname}...\n")
                self.analysis_progress['value'] = idx - 1
                self.root.update()

                fish_list = process_and_analyze_file(loaded_file, params)
                loaded_file.processed_data = fish_list

                self.analysis_summary_text.insert(tk.END,
                    f"  [OK] Analyzed {len(fish_list)}/{loaded_file.n_fish} fish\n")
                self.analysis_progress['value'] = idx
                self.root.update()

            # Update file lists and auto-select all
            self._update_analysis_files_listbox()
            for i in range(self.analysis_files_listbox.size()):
                self.analysis_files_listbox.selection_set(i)

            self._update_analysis_visualizations()

            self.set_status(f"Analysis complete \u2014 {total} file(s) processed")
            messagebox.showinfo("Complete", f"Processed {total} file(s).\n"
                              "View results in the Individual Analysis tab.")
        except Exception as e:
            self.analysis_summary_text.delete("1.0", tk.END)
            self.analysis_summary_text.insert("1.0", f"Error: {str(e)}")
            self.set_status(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Hide progress bar when done
            self.analysis_progress.pack_forget()

    def _get_processing_parameters_from_gui(self) -> ProcessingParameters:
        """Read processing parameters from the GUI inputs."""
        apply_smoothing = self.apply_smoothing_var.get()
        smoothing_window = int(self.smoothing_window_var.get())
        rest_threshold = float(self.rest_threshold_var.get())

        params = ProcessingParameters(
            apply_smoothing=apply_smoothing,
            smoothing_window=smoothing_window,
            smoothing_polynomial_order=3,
            min_valid_points=10,
            min_valid_percentage=0.01,
            rest_speed_threshold=rest_threshold,
        )
        params.validate()
        return params
