"""
fish_analyzer/gui/spatial_tab.py
================================
Spatial Tab Mixin - Thigmotaxis and heatmap analysis.

This mixin provides all methods related to:
- Arena definition (polygon drawing)
- Thigmotaxis (wall-hugging) analysis
- Position density heatmaps
- Multi-file comparison
"""

from typing import Dict, List, Any
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt

from ..spatial import (
    ArenaDefinition,
    ThigmotaxisResults,
    ThigmotaxisCalculator,
    HeatmapGenerator,
    SHAPELY_AVAILABLE,
    compute_shared_heatmap_scale
)
from ..export import export_thigmotaxis_csv


class SpatialTabMixin:
    """
    Mixin providing Spatial Analysis Tab functionality.
    
    Expects the following attributes from base class:
    - self.root: tk.Tk
    - self.notebook: ttk.Notebook
    - self.loaded_files: Dict[str, LoadedTrajectoryFile]
    - self.arena_vertices: List
    - self.arena_definition: Optional[ArenaDefinition]
    - self.file_arena_definitions: Dict[str, ArenaDefinition]
    - self.current_arena_file: Optional[str]
    - self.arena_fig, self.arena_ax, self.arena_canvas
    - self._arena_width_bl, self._arena_height_bl
    """

    def _create_spatial_tab(self):
        """Create the spatial analysis tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Spatial Analysis")

        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)

        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=3)

        # ----- LEFT PANEL: Controls -----
        self._create_spatial_controls(left_panel)

        # ----- RIGHT PANEL: Results -----
        self._create_spatial_results(right_panel)

        # Initialize per-file arena storage
        if not hasattr(self, 'file_arena_definitions'):
            self.file_arena_definitions = {}
        if not hasattr(self, 'current_arena_file'):
            self.current_arena_file = None
        
        # For backwards compatibility
        self.spatial_file_var = tk.StringVar()
        self.spatial_compare_listbox = tk.Listbox(tk.Frame())

    def _create_spatial_controls(self, parent):
        """Create the control panel for spatial tab."""
        # File Status Section
        status_frame = tk.LabelFrame(parent, text="Files & Status", font=("Arial", 11, "bold"))
        status_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Label(status_frame, text="Select files for analysis:", 
                 font=("Arial", 9)).pack(anchor="w", padx=10, pady=2)
        
        # Legend
        legend_frame = tk.Frame(status_frame)
        legend_frame.pack(fill="x", padx=10, pady=2)
        tk.Label(legend_frame, text="[✓]=Arena  [■]=Analyzed", 
                 font=("Arial", 8), fg="gray").pack(anchor="w")
        
        # File status listbox
        list_container = tk.Frame(status_frame)
        list_container.pack(fill="x", padx=10, pady=5)
        
        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        
        self.spatial_files_listbox = tk.Listbox(
            list_container, selectmode=tk.EXTENDED, height=5,
            yscrollcommand=scrollbar.set, font=("Courier", 9)
        )
        self.spatial_files_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.spatial_files_listbox.yview)
        self.spatial_files_listbox.bind('<<ListboxSelect>>', self._on_spatial_file_selection_changed)

        # Arena Definition Section
        arena_frame = tk.LabelFrame(parent, text="Arena Definition", font=("Arial", 11, "bold"))
        arena_frame.pack(fill="x", padx=10, pady=10)
        
        self.arena_edit_label = tk.Label(
            arena_frame, text="Select a file to edit arena", 
            font=("Arial", 9), fg="gray"
        )
        self.arena_edit_label.pack(anchor="w", padx=10, pady=3)
        
        # Arena drawing buttons
        arena_btn_frame = tk.Frame(arena_frame)
        arena_btn_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(arena_btn_frame, text="Clear", 
                  command=self._clear_current_arena, width=8).pack(side="left", padx=2)
        tk.Button(arena_btn_frame, text="Complete", 
                  command=self._complete_arena, bg="lightgreen", width=8).pack(side="left", padx=2)
        tk.Button(arena_btn_frame, text="Rectangle", 
                  command=self._draw_rectangle_arena, width=8).pack(side="left", padx=2)
        
        # Apply arena buttons
        apply_frame = tk.Frame(arena_frame)
        apply_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(apply_frame, text="Apply Arena to Selected Files", 
                  command=self._apply_arena_to_selected,
                  bg="lightyellow", font=("Arial", 9)).pack(fill="x")
        
        self.arena_status_label = tk.Label(arena_frame, text="", font=("Arial", 9))
        self.arena_status_label.pack(anchor="w", padx=10, pady=2)

        # Analysis Settings Section
        settings_frame = tk.LabelFrame(parent, text="Analysis Settings", font=("Arial", 11, "bold"))
        settings_frame.pack(fill="x", padx=10, pady=10)
        
        # Border zone
        border_frame = tk.Frame(settings_frame)
        border_frame.pack(fill="x", padx=10, pady=3)
        tk.Label(border_frame, text="Border zone:").pack(side="left")
        self.border_pct_var = tk.StringVar(value="15")
        tk.Entry(border_frame, textvariable=self.border_pct_var, width=5).pack(side="left", padx=5)
        tk.Label(border_frame, text="% from wall").pack(side="left")
        
        # Sample interval
        sample_frame = tk.Frame(settings_frame)
        sample_frame.pack(fill="x", padx=10, pady=3)
        tk.Label(sample_frame, text="Sample every:").pack(side="left")
        self.spatial_sample_var = tk.StringVar(value="30")
        tk.Entry(sample_frame, textvariable=self.spatial_sample_var, width=5).pack(side="left", padx=5)
        tk.Label(sample_frame, text="frames").pack(side="left")
        
        # Plot smoothing
        smooth_frame = tk.Frame(settings_frame)
        smooth_frame.pack(fill="x", padx=10, pady=3)
        tk.Label(smooth_frame, text="Plot smoothing:").pack(side="left")
        self.spatial_smooth_var = tk.StringVar(value="30")
        tk.Entry(smooth_frame, textvariable=self.spatial_smooth_var, width=5).pack(side="left", padx=5)
        tk.Label(smooth_frame, text="seconds").pack(side="left")
        
        # Show individual fish option
        self.show_individual_fish_thig_var = tk.BooleanVar(value=True)
        tk.Checkbutton(settings_frame, text="Show individual fish in time series",
                       variable=self.show_individual_fish_thig_var).pack(anchor="w", padx=10)

        # Main Action Button
        tk.Button(
            parent, text="Run Analysis on Selected Files",
            command=self._run_spatial_analysis_batch,
            bg="lightgreen", font=("Arial", 11, "bold"), height=2
        ).pack(fill="x", padx=10, pady=10)
        
        # Update plots button
        tk.Button(
            parent, text="Update Visualizations",
            command=self._update_spatial_visualizations,
            bg="lightblue", font=("Arial", 10)
        ).pack(fill="x", padx=10, pady=5)

        tk.Button(
            parent, text="Export Thigmotaxis to CSV",
            command=self._export_thigmotaxis_csv,
            font=("Arial", 10)
        ).pack(fill="x", padx=10, pady=(0, 5))

        # Heatmap Settings
        heat_frame = tk.LabelFrame(parent, text="Heatmap Settings", font=("Arial", 11, "bold"))
        heat_frame.pack(fill="x", padx=10, pady=10)
        
        grid_frame = tk.Frame(heat_frame)
        grid_frame.pack(fill="x", padx=10, pady=3)
        tk.Label(grid_frame, text="Grid size:").pack(side="left")
        self.heatmap_grid_var = tk.StringVar(value="50")
        tk.Entry(grid_frame, textvariable=self.heatmap_grid_var, width=5).pack(side="left", padx=5)
        tk.Label(grid_frame, text="bins").pack(side="left")
        
        self.heatmap_mode_var = tk.StringVar(value="combined")
        tk.Radiobutton(heat_frame, text="Combined (all fish per file)",
                       variable=self.heatmap_mode_var, value="combined").pack(anchor="w", padx=10)
        tk.Radiobutton(heat_frame, text="Individual fish grid",
                       variable=self.heatmap_mode_var, value="individual").pack(anchor="w", padx=10)
        
        self.shared_colorscale_var = tk.BooleanVar(value=True)
        tk.Checkbutton(heat_frame, text="Use shared color scale",
                       variable=self.shared_colorscale_var).pack(anchor="w", padx=10)

    def _create_spatial_results(self, parent):
        """Create the results panel for spatial tab."""
        results_notebook = ttk.Notebook(parent)
        results_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Arena Definition Tab
        arena_tab = ttk.Frame(results_notebook)
        results_notebook.add(arena_tab, text="Arena Definition")
        self.arena_canvas_frame = tk.Frame(arena_tab, bg="white")
        self.arena_canvas_frame.pack(fill="both", expand=True)
        tk.Label(self.arena_canvas_frame,
                 text="Select a file from the list to define its arena",
                 font=("Arial", 11), fg="gray").pack(expand=True)

        # Results Summary Tab
        summary_tab = ttk.Frame(results_notebook)
        results_notebook.add(summary_tab, text="Results Summary")
        
        summary_scroll = tk.Scrollbar(summary_tab)
        summary_scroll.pack(side="right", fill="y")
        
        self.thigmotaxis_text = tk.Text(
            summary_tab, wrap=tk.WORD, font=("Courier", 10),
            yscrollcommand=summary_scroll.set
        )
        self.thigmotaxis_text.pack(side="left", fill="both", expand=True)
        summary_scroll.config(command=self.thigmotaxis_text.yview)
        
        self.thigmotaxis_text.insert("1.0",
            "SPATIAL ANALYSIS - THIGMOTAXIS & HEATMAPS\n"
            "=" * 60 + "\n\n"
            "WHAT THIS DOES:\n"
            "Thigmotaxis measures how much time fish spend near the walls\n"
            "(border zone) vs the center of the arena. High thigmotaxis\n"
            "(wall-hugging) is a common anxiety-like behavior in zebrafish.\n\n"
            "Heatmaps show where fish spend the most time in the arena.\n\n"
            "HOW TO USE (step by step):\n"
            "-" * 40 + "\n"
            "Step 1: Select a file from the list on the left.\n"
            "Step 2: Go to the 'Arena Definition' tab (right panel).\n"
            "        Click on the image to place arena boundary points,\n"
            "        OR click 'Rectangle' for a quick rectangular arena.\n"
            "Step 3: Click 'Complete' to finalize the arena shape.\n"
            "Step 4: (Optional) If multiple files share the same arena,\n"
            "        select them and click 'Apply Arena to Selected Files'.\n"
            "Step 5: Select all files you want to analyze, then click\n"
            "        'Run Analysis on Selected Files'.\n\n"
            "STATUS INDICATORS (in the file list):\n"
            "  [\u2713] = Arena has been defined for this file\n"
            "  [\u25a0] = Analysis has been completed for this file\n"
        )

        # Time Series Tab
        ts_tab = ttk.Frame(results_notebook)
        results_notebook.add(ts_tab, text="Time Series")
        self.thigmotaxis_plot_frame = tk.Frame(ts_tab, bg="white")
        self.thigmotaxis_plot_frame.pack(fill="both", expand=True)

        # Comparison Tab
        compare_tab = ttk.Frame(results_notebook)
        results_notebook.add(compare_tab, text="Comparison")
        self.thigmotaxis_compare_frame = tk.Frame(compare_tab, bg="white")
        self.thigmotaxis_compare_frame.pack(fill="both", expand=True)

        # Heatmaps Tab
        heatmap_tab = ttk.Frame(results_notebook)
        results_notebook.add(heatmap_tab, text="Heatmaps")
        self.heatmap_frame = tk.Frame(heatmap_tab, bg="white")
        self.heatmap_frame.pack(fill="both", expand=True)

    # =========================================================================
    # FILE LIST UPDATE
    # =========================================================================

    def _update_spatial_file_dropdown(self):
        """Update the spatial file list with status indicators."""
        self._update_spatial_files_list()
    
    def _update_spatial_files_list(self):
        """Update the spatial file list with status indicators."""
        if hasattr(self, 'spatial_files_listbox'):
            self.spatial_files_listbox.delete(0, tk.END)
            
            for filename in self.loaded_files.keys():
                has_arena = filename in self.file_arena_definitions
                loaded_file = self.loaded_files[filename]
                has_results = (hasattr(loaded_file, 'thigmotaxis_results') and 
                               loaded_file.thigmotaxis_results is not None)
                
                arena_mark = "✓" if has_arena else " "
                results_mark = "■" if has_results else " "
                
                display_text = f"[{arena_mark}][{results_mark}] {filename}"
                self.spatial_files_listbox.insert(tk.END, display_text)

    # =========================================================================
    # FILE SELECTION HANDLERS
    # =========================================================================

    def _on_spatial_file_selection_changed(self, event=None):
        """Handle file selection change - load first selected file for arena editing."""
        selection = self.spatial_files_listbox.curselection()
        if not selection:
            self.arena_edit_label.config(text="Select a file to edit arena", fg="gray")
            self.current_arena_file = None
            return
        
        file_keys = list(self.loaded_files.keys())
        first_idx = selection[0]
        first_file = file_keys[first_idx]
        
        self.current_arena_file = first_file
        self.arena_edit_label.config(text=f"Editing: {first_file}", fg="blue")
        
        # Load this file's arena if it exists
        if first_file in self.file_arena_definitions:
            self.arena_definition = self.file_arena_definitions[first_file]
            n_verts = len(self.arena_definition.vertices_bl)
            self.arena_status_label.config(
                text=f"Arena loaded ({n_verts} vertices) [✓]", fg="green"
            )
            self.arena_vertices = [tuple(v) for v in self.arena_definition.vertices_bl]
        else:
            self.arena_definition = None
            self.arena_vertices = []
            self.arena_status_label.config(text="No arena defined for this file", fg="gray")
        
        self._setup_arena_canvas_for_file(first_file)

    # =========================================================================
    # ARENA DEFINITION METHODS
    # =========================================================================

    def _setup_arena_canvas_for_file(self, filename: str):
        """Set up the arena drawing canvas for a specific file."""
        if filename not in self.loaded_files:
            return
        
        loaded_file = self.loaded_files[filename]
        
        for widget in self.arena_canvas_frame.winfo_children():
            widget.destroy()

        self.arena_fig = Figure(figsize=(8, 6), dpi=100)
        self.arena_ax = self.arena_fig.add_subplot(111)

        pixels_to_bl = 1.0 / loaded_file.metadata.body_length
        width_bl = loaded_file.metadata.video_width * pixels_to_bl
        height_bl = loaded_file.metadata.video_height * pixels_to_bl

        # Try to load and display background image
        background_loaded = False
        if loaded_file.background_image_path and loaded_file.background_image_path.exists():
            try:
                background_img = plt.imread(str(loaded_file.background_image_path))
                if len(background_img.shape) == 2:
                    self.arena_ax.imshow(background_img, extent=[0, width_bl, 0, height_bl],
                                        aspect='equal', origin='upper', alpha=0.8, cmap='gray')
                else:
                    self.arena_ax.imshow(background_img, extent=[0, width_bl, 0, height_bl],
                                        aspect='equal', origin='upper', alpha=0.8)
                background_loaded = True
            except Exception as e:
                print(f"Failed to load background image: {e}")

        self.arena_ax.set_xlim(0, width_bl)
        self.arena_ax.set_ylim(0, height_bl)
        self.arena_ax.set_xlabel('X (BL)', fontsize=10)
        self.arena_ax.set_ylabel('Y (BL)', fontsize=10)
        
        title = f'Define arena for: {filename}'
        if background_loaded:
            title += ' (background loaded)'
        self.arena_ax.set_title(title, fontsize=11)
        self.arena_ax.set_aspect('equal')
        
        if not background_loaded:
            self.arena_ax.grid(True, alpha=0.3)

        if self.arena_vertices:
            self._redraw_arena_on_canvas()

        self.arena_fig.tight_layout()

        self.arena_canvas = FigureCanvasTkAgg(self.arena_fig, master=self.arena_canvas_frame)
        self.arena_canvas.draw()
        self.arena_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.arena_canvas.mpl_connect('button_press_event', self._on_arena_click)

        self._arena_width_bl = width_bl
        self._arena_height_bl = height_bl

    def _on_arena_click(self, event):
        """Handle click on arena canvas - add vertex."""
        if event.inaxes != self.arena_ax or event.button != 1:
            return
        
        if self.current_arena_file is None:
            messagebox.showwarning("No File", "Please select a file first.")
            return

        self.arena_vertices.append((event.xdata, event.ydata))
        self._redraw_arena_on_canvas()

    def _redraw_arena_on_canvas(self):
        """Redraw the arena polygon on canvas."""
        if self.arena_ax is None:
            return

        # Remove previous arena elements
        for patch in self.arena_ax.patches[:]:
            patch.remove()
        for coll in self.arena_ax.collections[:]:
            if hasattr(coll, '_arena_marker'):
                coll.remove()
        for line in self.arena_ax.lines[:]:
            if hasattr(line, '_arena_line'):
                line.remove()

        if len(self.arena_vertices) > 0:
            xs, ys = zip(*self.arena_vertices)
            scatter = self.arena_ax.scatter(xs, ys, c='red', s=100, zorder=5, marker='o')
            scatter._arena_marker = True

            if len(self.arena_vertices) > 1:
                xs_line = list(xs) + [xs[0]]
                ys_line = list(ys) + [ys[0]]
                line, = self.arena_ax.plot(xs_line, ys_line, 'r-', linewidth=2, alpha=0.7)
                line._arena_line = True

        n_verts = len(self.arena_vertices)
        self.arena_status_label.config(text=f"{n_verts} vertices defined", fg="blue")
        self.arena_canvas.draw()

    def _clear_current_arena(self):
        """Clear the arena for the current file (with confirmation)."""
        if self.current_arena_file is None:
            return

        # Confirm before clearing a completed arena
        if self.current_arena_file in self.file_arena_definitions:
            if not messagebox.askyesno(
                "Clear Arena?",
                f"This will remove the arena definition for '{self.current_arena_file}'.\n"
                "You will need to redraw it. Continue?"
            ):
                return

        self.arena_vertices = []
        self.arena_definition = None
        
        if self.current_arena_file in self.file_arena_definitions:
            del self.file_arena_definitions[self.current_arena_file]
        
        self.arena_status_label.config(text="Arena cleared", fg="gray")
        
        if self.arena_ax is not None:
            for patch in self.arena_ax.patches[:]:
                patch.remove()
            for line in self.arena_ax.lines[:]:
                if hasattr(line, '_arena_line'):
                    line.remove()
            for coll in self.arena_ax.collections[:]:
                if hasattr(coll, '_arena_marker'):
                    coll.remove()
            self.arena_canvas.draw()
        
        self._update_spatial_files_list()

    def _complete_arena(self):
        """Complete the arena polygon for current file and show zones."""
        if len(self.arena_vertices) < 3:
            messagebox.showwarning("Insufficient Vertices", "Please define at least 3 vertices.")
            return
        
        if self.current_arena_file is None or self.current_arena_file not in self.loaded_files:
            messagebox.showwarning("No File", "Please select a file first.")
            return

        loaded_file = self.loaded_files[self.current_arena_file]
        pixels_to_bl = 1.0 / loaded_file.metadata.body_length

        vertices_bl = np.array(self.arena_vertices)
        vertices_pixels = vertices_bl / pixels_to_bl
        vertices_pixels[:, 1] = loaded_file.metadata.video_height - vertices_pixels[:, 1]

        self.arena_definition = ArenaDefinition(
            vertices_pixels=vertices_pixels,
            vertices_bl=vertices_bl
        )
        
        self.file_arena_definitions[self.current_arena_file] = self.arena_definition

        # Draw the arena with zones
        if self.arena_ax is not None:
            for patch in self.arena_ax.patches[:]:
                patch.remove()
            for line in self.arena_ax.lines[:]:
                line.remove()

            # Draw border zone (outer)
            polygon_outer = MplPolygon(vertices_bl, closed=True,
                                       fill=True, facecolor='orange', alpha=0.15,
                                       edgecolor='red', linewidth=2, label='Border Zone')
            self.arena_ax.add_patch(polygon_outer)

            # Draw center zone (inner)
            try:
                border_pct = float(self.border_pct_var.get()) / 100
                if SHAPELY_AVAILABLE:
                    from shapely.geometry import Polygon as ShapelyPolygon
                    arena_poly = ShapelyPolygon(vertices_bl)
                    arena_bounds = arena_poly.bounds
                    arena_width = arena_bounds[2] - arena_bounds[0]
                    arena_height = arena_bounds[3] - arena_bounds[1]
                    buffer_distance = min(arena_width, arena_height) * border_pct
                    center_poly = arena_poly.buffer(-buffer_distance)
                    
                    if not center_poly.is_empty and center_poly.geom_type == 'Polygon':
                        inner_coords = np.array(center_poly.exterior.coords)
                        polygon_inner = MplPolygon(inner_coords, closed=True,
                                                   fill=True, facecolor='green', alpha=0.2,
                                                   edgecolor='green', linewidth=2, 
                                                   linestyle='--', label='Center Zone')
                        self.arena_ax.add_patch(polygon_inner)
            except Exception as e:
                print(f"Could not draw center zone: {e}")

            # Overlay sample trajectories
            self._overlay_sample_trajectories(loaded_file, pixels_to_bl)
            
            self.arena_ax.legend(loc='upper right', fontsize=8)
            self.arena_canvas.draw()

        self.arena_status_label.config(
            text=f"Arena defined ({len(self.arena_vertices)} vertices) [✓]", fg="green"
        )
        
        self._update_spatial_files_list()
        
        messagebox.showinfo("Arena Complete", 
                            f"Arena defined for '{self.current_arena_file}'.\n\n"
                            f"Use 'Apply Arena to Selected Files' to copy this arena to other files.")

    def _overlay_sample_trajectories(self, loaded_file, pixels_to_bl):
        """Overlay downsampled trajectories on the arena canvas."""
        if self.arena_ax is None:
            return
        
        sample_rate = max(1, loaded_file.n_frames // 500)
        
        for fish_idx in range(loaded_file.n_fish):
            positions = loaded_file.trajectories[::sample_rate, fish_idx, :]
            valid_mask = ~np.isnan(positions[:, 0])
            
            if np.sum(valid_mask) < 2:
                continue
            
            x_bl = positions[valid_mask, 0] * pixels_to_bl
            y_bl = (loaded_file.metadata.video_height - positions[valid_mask, 1]) * pixels_to_bl
            
            self.arena_ax.plot(x_bl, y_bl, color='gray', linewidth=0.5, alpha=0.4)

    def _apply_arena_to_selected(self):
        """Apply the current arena definition to all selected files."""
        if self.arena_definition is None:
            messagebox.showwarning("No Arena", "Please define and complete an arena first.")
            return
        
        selection = self.spatial_files_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select files to apply arena to.")
            return
        
        file_keys = list(self.loaded_files.keys())
        selected_files = [file_keys[i] for i in selection]
        
        count = 0
        for filename in selected_files:
            if filename != self.current_arena_file:
                self.file_arena_definitions[filename] = self.arena_definition.copy()
                count += 1
        
        self._update_spatial_files_list()
        messagebox.showinfo("Arena Applied", 
                            f"Arena copied to {count} additional file(s).\n"
                            f"Total files with arena: {len(self.file_arena_definitions)}")

    def _draw_rectangle_arena(self):
        """Auto-draw a rectangle arena for current file."""
        if self.current_arena_file is None:
            messagebox.showwarning("No File", "Please select a file first.")
            return
        
        if self._arena_width_bl is None:
            messagebox.showwarning("Error", "Arena dimensions not set.")
            return

        margin = 0.02
        x_min = self._arena_width_bl * margin
        x_max = self._arena_width_bl * (1 - margin)
        y_min = self._arena_height_bl * margin
        y_max = self._arena_height_bl * (1 - margin)

        self.arena_vertices = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self._redraw_arena_on_canvas()
        self._complete_arena()

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def _run_spatial_analysis_batch(self):
        """Run thigmotaxis analysis on all selected files."""
        if not SHAPELY_AVAILABLE:
            messagebox.showerror("Missing Dependency", 
                                "Shapely is required. Install with: pip install shapely")
            return

        selection = self.spatial_files_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select files to analyze.")
            return
        
        file_keys = list(self.loaded_files.keys())
        selected_files = [file_keys[i] for i in selection]
        
        # Check which files have arenas
        files_without_arena = [f for f in selected_files if f not in self.file_arena_definitions]
        if files_without_arena:
            missing_str = "\n".join(files_without_arena[:5])
            if len(files_without_arena) > 5:
                missing_str += f"\n... and {len(files_without_arena) - 5} more"
            messagebox.showwarning("Missing Arenas", 
                                  f"These files need arena definitions:\n\n{missing_str}\n\n"
                                  "Define arena and use 'Apply Arena to Selected Files'.")
            return
        
        try:
            border_pct = float(self.border_pct_var.get()) / 100
            sample_interval = int(self.spatial_sample_var.get())
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", f"Check settings: {e}")
            return
        
        success_count = 0
        results_dict = {}
        
        for filename in selected_files:
            loaded_file = self.loaded_files[filename]
            arena = self.file_arena_definitions[filename]
            
            try:
                calculator = ThigmotaxisCalculator(
                    loaded_file, arena,
                    border_pct=border_pct, 
                    sample_interval=sample_interval
                )
                results = calculator.calculate()
                loaded_file.thigmotaxis_results = results
                results_dict[filename] = results
                success_count += 1
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
                import traceback
                traceback.print_exc()
        
        self._update_spatial_files_list()
        
        if results_dict:
            self._display_combined_thigmotaxis_summary(results_dict)
            self._plot_combined_thigmotaxis_timeseries(results_dict)
            self._plot_thigmotaxis_comparison_charts(results_dict)
            self._generate_comparison_heatmaps(selected_files)
        
        messagebox.showinfo("Analysis Complete", 
                            f"Successfully analyzed {success_count}/{len(selected_files)} files.")

    def _update_spatial_visualizations(self):
        """Update visualizations for files that have been analyzed."""
        results_dict = {}
        analyzed_files = []
        
        for filename, loaded_file in self.loaded_files.items():
            if hasattr(loaded_file, 'thigmotaxis_results') and loaded_file.thigmotaxis_results:
                results_dict[filename] = loaded_file.thigmotaxis_results
                analyzed_files.append(filename)
        
        if not results_dict:
            messagebox.showinfo("No Results", "No files have been analyzed yet.")
            return
        
        self._display_combined_thigmotaxis_summary(results_dict)
        self._plot_combined_thigmotaxis_timeseries(results_dict)
        self._plot_thigmotaxis_comparison_charts(results_dict)
        self._generate_comparison_heatmaps(analyzed_files)

    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================

    def _display_combined_thigmotaxis_summary(self, results_dict: Dict[str, ThigmotaxisResults]):
        """Display combined summary of all analyzed files."""
        self.thigmotaxis_text.delete("1.0", tk.END)
        
        lines = [
            "=" * 70,
            "THIGMOTAXIS ANALYSIS - ALL FILES",
            "=" * 70,
            "",
        ]
        
        lines.append("SUMMARY COMPARISON:")
        lines.append("-" * 70)
        lines.append(f"{'File':<20} {'Fish':<6} {'Mean %':<10} {'Std %':<10} {'Samples':<10}")
        lines.append("-" * 70)
        
        for filename, results in results_dict.items():
            short_name = filename[:19] if len(filename) > 19 else filename
            lines.append(
                f"{short_name:<20} {results.n_fish:<6} "
                f"{results.mean_pct_in_border:<10.1f} {results.std_pct_in_border:<10.1f} "
                f"{results.n_samples:<10}"
            )
        
        lines.append("-" * 70)
        lines.append("")
        
        for filename, results in results_dict.items():
            lines.append("")
            lines.append(f"FILE: {filename}")
            lines.append(results.summary())
        
        self.thigmotaxis_text.insert("1.0", "\n".join(lines))

    def _plot_combined_thigmotaxis_timeseries(self, results_dict: Dict[str, ThigmotaxisResults]):
        """Plot thigmotaxis time series for all files."""
        for widget in self.thigmotaxis_plot_frame.winfo_children():
            widget.destroy()
        
        if not results_dict:
            return
        
        try:
            smooth_seconds = float(self.spatial_smooth_var.get())
        except ValueError:
            smooth_seconds = 30.0
            self.spatial_smooth_var.set("30")
        
        show_individuals = self.show_individual_fish_thig_var.get() if hasattr(self, 'show_individual_fish_thig_var') else True
        
        n_files = len(results_dict)
        file_colors = plt.cm.tab10(np.linspace(0, 1, min(n_files, 10)))
        
        if n_files == 1:
            fig = Figure(figsize=(10, 6), dpi=100)
            axes = [fig.add_subplot(111)]
        else:
            n_cols = min(2, n_files)
            n_rows = (n_files + n_cols - 1) // n_cols
            fig = Figure(figsize=(6 * n_cols, 4 * n_rows), dpi=100)
            axes = [fig.add_subplot(n_rows, n_cols, i + 1) for i in range(n_files)]
        
        for idx, (filename, results) in enumerate(results_dict.items()):
            ax = axes[idx]
            color = file_colors[idx % len(file_colors)]
            time_minutes = results.timestamps / 60.0
            
            if show_individuals and hasattr(results, 'per_fish_in_border_samples'):
                for fish_idx in range(results.n_fish):
                    fish_data = results.get_smoothed_fish_timeseries(fish_idx, smooth_seconds)
                    ax.plot(time_minutes, fish_data, 
                           color=color, linewidth=0.5, alpha=0.3)
            
            group_data = results.get_smoothed_group_timeseries(smooth_seconds)
            ax.plot(time_minutes, group_data, 
                   color=color, linewidth=2.5, alpha=0.9,
                   label=f'Group avg: {results.mean_pct_in_border:.1f}%')
            
            ax.axhline(y=results.mean_pct_in_border, color='gray', 
                      linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('Time (minutes)', fontsize=10)
            ax.set_ylabel('% in Border Zone', fontsize=10)
            ax.set_title(f'{filename}', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.thigmotaxis_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_thigmotaxis_comparison_charts(self, results_dict: Dict[str, ThigmotaxisResults]):
        """Plot comparison bar charts and summary statistics."""
        for widget in self.thigmotaxis_compare_frame.winfo_children():
            widget.destroy()
        
        if not results_dict:
            return
        
        fig = Figure(figsize=(12, 5), dpi=100)
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        file_colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
        
        try:
            smooth_seconds = float(self.spatial_smooth_var.get())
        except ValueError:
            smooth_seconds = 30.0
            self.spatial_smooth_var.set("30")
        
        # Time series overlay
        for (filename, results), color in zip(results_dict.items(), file_colors):
            time_minutes = results.timestamps / 60.0
            smoothed_data = results.get_smoothed_group_timeseries(smooth_seconds)
            ax1.plot(time_minutes, smoothed_data, color=color, linewidth=2,
                    label=f'{filename[:20]} ({results.mean_pct_in_border:.1f}%)')
        
        ax1.set_xlabel('Time (minutes)', fontsize=11)
        ax1.set_ylabel('% Fish in Border Zone', fontsize=11)
        ax1.set_title('Thigmotaxis Over Time (Smoothed)', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Bar chart
        file_names = [f[:20] if len(f) > 20 else f for f in results_dict.keys()]
        means = [r.mean_pct_in_border for r in results_dict.values()]
        stds = [r.std_pct_in_border for r in results_dict.values()]
        x = np.arange(len(file_names))
        
        bars = ax2.bar(x, means, color=file_colors[:len(means)], 
                       edgecolor='black', yerr=stds, capsize=5)
        ax2.set_xlabel('File', fontsize=11)
        ax2.set_ylabel('Mean % Time in Border Zone', fontsize=11)
        ax2.set_title('Thigmotaxis Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(file_names, rotation=45, ha='right')
        ax2.set_ylim(0, 100)
        
        for bar, mean in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.thigmotaxis_compare_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # =========================================================================
    # HEATMAP METHODS
    # =========================================================================

    def _generate_comparison_heatmaps(self, selected_files: List[str]):
        """Generate heatmaps for selected files."""
        for widget in self.heatmap_frame.winfo_children():
            widget.destroy()
        
        valid_files = [f for f in selected_files if f in self.loaded_files]
        if not valid_files:
            return
        
        try:
            grid_size = int(self.heatmap_grid_var.get())
            if grid_size < 5:
                raise ValueError("Grid size must be at least 5")
        except ValueError:
            grid_size = 50
            self.heatmap_grid_var.set("50")
        
        mode = self.heatmap_mode_var.get()
        use_shared_scale = self.shared_colorscale_var.get() if hasattr(self, 'shared_colorscale_var') else True
        
        if mode == "combined":
            self._plot_combined_heatmaps(valid_files, grid_size, use_shared_scale)
        else:
            if len(valid_files) > 1:
                messagebox.showinfo(
                    "Individual Heatmap Mode",
                    f"Showing individual fish heatmaps for '{valid_files[0]}' only.\n\n"
                    "Individual mode displays one file at a time.\n"
                    "Use 'Combined' mode to compare across files.")
            self._plot_individual_fish_heatmaps(valid_files[0], grid_size, use_shared_scale)

    def _plot_combined_heatmaps(self, filenames: List[str], grid_size: int, use_shared_scale: bool):
        """Plot combined (all fish) heatmaps for multiple files."""
        n_files = len(filenames)
        
        heatmaps = []
        edges_list = []
        dimensions = []
        
        for filename in filenames:
            loaded_file = self.loaded_files[filename]
            generator = HeatmapGenerator(loaded_file, grid_size=grid_size)
            hm, xe, ye = generator.generate_combined_heatmap()
            heatmaps.append(hm)
            edges_list.append((xe, ye))
            
            pixels_to_bl = 1.0 / loaded_file.metadata.body_length
            width_bl = loaded_file.metadata.video_width * pixels_to_bl
            height_bl = loaded_file.metadata.video_height * pixels_to_bl
            dimensions.append((width_bl, height_bl))
        
        if use_shared_scale and len(heatmaps) > 1:
            vmin, vmax = compute_shared_heatmap_scale(heatmaps)
        else:
            vmin, vmax = None, None
        
        n_cols = min(3, n_files)
        n_rows = (n_files + n_cols - 1) // n_cols
        fig = Figure(figsize=(5 * n_cols, 4 * n_rows), dpi=100)
        
        im = None
        for idx, (filename, hm, (xe, ye), (w, h)) in enumerate(
                zip(filenames, heatmaps, edges_list, dimensions)):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            
            if vmin is not None:
                im = ax.imshow(hm, extent=[xe[0], xe[-1], ye[0], ye[-1]],
                              origin='lower', aspect='equal', cmap='hot', 
                              alpha=0.8, vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(hm, extent=[xe[0], xe[-1], ye[0], ye[-1]],
                              origin='lower', aspect='equal', cmap='hot', alpha=0.8)
            
            ax.set_xlim(0, w)
            ax.set_ylim(0, h)
            ax.set_xlabel('X (BL)', fontsize=9)
            ax.set_ylabel('Y (BL)', fontsize=9)
            ax.set_title(filename[:20], fontsize=10)
        
        if use_shared_scale and len(heatmaps) > 1 and im is not None:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('% Time', fontsize=9)
        
        fig.tight_layout()
        if use_shared_scale and len(heatmaps) > 1:
            fig.subplots_adjust(right=0.9)
        
        canvas = FigureCanvasTkAgg(fig, master=self.heatmap_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_individual_fish_heatmaps(self, filename: str, grid_size: int, use_shared_scale: bool):
        """Plot individual fish heatmaps for a single file."""
        if filename not in self.loaded_files:
            return
        
        loaded_file = self.loaded_files[filename]
        generator = HeatmapGenerator(loaded_file, grid_size=grid_size)
        
        heatmaps, x_edges, y_edges = generator.generate_all_individual_heatmaps()
        
        pixels_to_bl = 1.0 / loaded_file.metadata.body_length
        width_bl = loaded_file.metadata.video_width * pixels_to_bl
        height_bl = loaded_file.metadata.video_height * pixels_to_bl
        
        if use_shared_scale:
            vmin, vmax = compute_shared_heatmap_scale(heatmaps)
        else:
            vmin, vmax = None, None
        
        n_fish = len(heatmaps)
        n_cols = min(3, n_fish)
        n_rows = (n_fish + n_cols - 1) // n_cols
        
        fig = Figure(figsize=(4 * n_cols, 3.5 * n_rows), dpi=100)
        
        im = None
        for fish_idx, hm in enumerate(heatmaps):
            ax = fig.add_subplot(n_rows, n_cols, fish_idx + 1)
            
            if vmin is not None:
                im = ax.imshow(hm, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                              origin='lower', aspect='equal', cmap='hot', 
                              alpha=0.8, vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(hm, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                              origin='lower', aspect='equal', cmap='hot', alpha=0.8)
            
            ax.set_xlim(0, width_bl)
            ax.set_ylim(0, height_bl)
            ax.set_title(f'Fish {fish_idx}', fontsize=10)
            ax.tick_params(labelsize=7)
        
        if use_shared_scale and im is not None:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('% Time', fontsize=9)
        
        fig.suptitle(f'Individual Fish Heatmaps - {filename}', fontsize=12, fontweight='bold')
        fig.tight_layout()
        if use_shared_scale:
            fig.subplots_adjust(right=0.9, top=0.92)

        canvas = FigureCanvasTkAgg(fig, master=self.heatmap_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def _export_thigmotaxis_csv(self):
        """Export thigmotaxis results to a CSV file."""
        analyzed = {k: v for k, v in self.loaded_files.items()
                    if getattr(v, 'thigmotaxis_results', None) is not None}
        if not analyzed:
            messagebox.showwarning("No Data",
                                   "No files have thigmotaxis results yet.\n"
                                   "Run 'Run Analysis on Selected Files' first.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Export Thigmotaxis Results CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="thigmotaxis_results.csv"
        )
        if not output_path:
            return

        try:
            n_rows = export_thigmotaxis_csv(analyzed, Path(output_path))
            self.set_status(f"Exported thigmotaxis data to {Path(output_path).name}")
            messagebox.showinfo("Export Complete",
                                f"Exported {n_rows} file(s) to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{e}")
