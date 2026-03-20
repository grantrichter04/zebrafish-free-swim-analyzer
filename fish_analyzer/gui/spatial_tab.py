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
from .utils import create_sortable_treeview, embed_figure_with_toolbar


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

        # Analysis mode
        tk.Label(settings_frame, text="Mode:", font=("Arial", 9, "bold")).pack(anchor="w", padx=10, pady=(5, 0))
        self.spatial_mode_var = tk.StringVar(value="border")
        tk.Radiobutton(settings_frame, text="Border zone (thigmotaxis)",
                       variable=self.spatial_mode_var, value="border",
                       command=self._on_spatial_mode_change).pack(anchor="w", padx=20)
        tk.Radiobutton(settings_frame, text="Custom ROI (% time in drawn region)",
                       variable=self.spatial_mode_var, value="roi",
                       command=self._on_spatial_mode_change).pack(anchor="w", padx=20)

        # Border zone settings
        self.border_settings_frame = tk.Frame(settings_frame)
        self.border_settings_frame.pack(fill="x", padx=10, pady=3)
        tk.Label(self.border_settings_frame, text="Border zone:").pack(side="left")
        self.border_pct_var = tk.StringVar(value="15")
        tk.Entry(self.border_settings_frame, textvariable=self.border_pct_var, width=5).pack(side="left", padx=5)
        tk.Label(self.border_settings_frame, text="% from wall").pack(side="left")

        # ROI settings
        self.roi_settings_frame = tk.Frame(settings_frame)
        # Hidden by default
        self.roi_vertices = []
        self.roi_definition = None
        self.file_roi_definitions = {}

        roi_btn_frame = tk.Frame(self.roi_settings_frame)
        roi_btn_frame.pack(fill="x", padx=0, pady=3)
        tk.Button(roi_btn_frame, text="Draw ROI", command=self._start_roi_drawing,
                  bg="lightyellow", width=10).pack(side="left", padx=2)
        tk.Button(roi_btn_frame, text="Clear ROI", command=self._clear_roi,
                  width=8).pack(side="left", padx=2)
        tk.Button(roi_btn_frame, text="Complete ROI", command=self._complete_roi,
                  bg="lightgreen", width=10).pack(side="left", padx=2)
        self.roi_status_label = tk.Label(self.roi_settings_frame, text="No ROI defined",
                                         font=("Arial", 8), fg="gray")
        self.roi_status_label.pack(anchor="w")
        self._roi_drawing_mode = False
        
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

        # Results Summary Tab (Treeview tables)
        summary_tab = ttk.Frame(results_notebook)
        results_notebook.add(summary_tab, text="Results Summary")

        self.thigmotaxis_summary_frame = tk.Frame(summary_tab)
        self.thigmotaxis_summary_frame.pack(fill="both", expand=True)

        tk.Label(self.thigmotaxis_summary_frame,
            text="Spatial analysis results will appear here after running analysis.\n\n"
            "Steps: 1) Select file  2) Draw arena  3) Complete  4) Run Analysis",
            font=("Arial", 10), fg="gray", justify=tk.LEFT
        ).pack(padx=20, pady=20)

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

        # Methods tab
        methods_tab = ttk.Frame(results_notebook)
        results_notebook.add(methods_tab, text="Methods")
        methods_scroll = tk.Scrollbar(methods_tab)
        methods_scroll.pack(side="right", fill="y")
        self.spatial_methods_text = tk.Text(
            methods_tab, wrap=tk.WORD, font=("Courier", 10),
            yscrollcommand=methods_scroll.set
        )
        self.spatial_methods_text.pack(side="left", fill="both", expand=True)
        methods_scroll.config(command=self.spatial_methods_text.yview)
        self.spatial_methods_text.insert("1.0",
            "Methods text will appear here after running spatial analysis.\n\n"
            "This provides a draft paragraph describing the thigmotaxis\n"
            "methods and parameters used, suitable for a manuscript methods section."
        )

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

        # Try to load and display background image (always B&W for arena drawing)
        background_loaded = False
        if loaded_file.background_image_path and loaded_file.background_image_path.exists():
            try:
                background_img = plt.imread(str(loaded_file.background_image_path))
                # Convert to grayscale if color image
                if len(background_img.shape) == 3:
                    # Standard luminance conversion
                    background_img = np.dot(background_img[..., :3], [0.299, 0.587, 0.114])
                self.arena_ax.imshow(background_img, extent=[0, width_bl, 0, height_bl],
                                    aspect='equal', origin='upper', alpha=0.8, cmap='gray')
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
        """Handle click on arena canvas - add vertex (arena or ROI)."""
        if event.inaxes != self.arena_ax or event.button != 1:
            return

        if self.current_arena_file is None:
            messagebox.showwarning("No File", "Please select a file first.")
            return

        # If in ROI drawing mode, add to ROI vertices instead
        if getattr(self, '_roi_drawing_mode', False):
            self.roi_vertices.append((event.xdata, event.ydata))
            # Draw ROI vertex on canvas
            self.arena_ax.scatter([event.xdata], [event.ydata], c='cyan', s=80, zorder=10, marker='s')
            if len(self.roi_vertices) > 1:
                xs = [v[0] for v in self.roi_vertices]
                ys = [v[1] for v in self.roi_vertices]
                # Draw lines connecting ROI vertices
                self.arena_ax.plot(xs[-2:], ys[-2:], 'c--', linewidth=2, alpha=0.7)
            self.roi_status_label.config(
                text=f"{len(self.roi_vertices)} ROI vertices defined", fg="blue")
            self.arena_canvas.draw()
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
        """Run thigmotaxis or ROI analysis on all selected files."""
        if not SHAPELY_AVAILABLE:
            messagebox.showerror("Missing Dependency",
                                "Shapely is required. Install with: pip install shapely")
            return

        selection = self.spatial_files_listbox.curselection()
        if not selection:
            # Auto-select all files
            if self.spatial_files_listbox.size() > 0:
                self.spatial_files_listbox.select_set(0, tk.END)
                selection = self.spatial_files_listbox.curselection()
            else:
                messagebox.showwarning("No Files",
                                       "Load files in the Data tab first.")
                return

        file_keys = list(self.loaded_files.keys())
        selected_files = [file_keys[i] for i in selection]

        analysis_mode = self.spatial_mode_var.get() if hasattr(self, 'spatial_mode_var') else "border"

        if analysis_mode == "roi":
            self._run_roi_analysis(selected_files)
            return

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
            self._update_spatial_methods_text(results_dict, border_pct, sample_interval)

        messagebox.showinfo("Analysis Complete",
                            f"Successfully analyzed {success_count}/{len(selected_files)} files.")

    def _run_roi_analysis(self, selected_files: List[str]):
        """Run custom ROI analysis — compute % time each fish spends in the ROI polygon."""
        from shapely.geometry import Polygon as ShapelyPolygon, Point

        try:
            sample_interval = int(self.spatial_sample_var.get())
        except ValueError:
            sample_interval = 30

        # Check for ROI definitions
        files_without_roi = [f for f in selected_files if f not in self.file_roi_definitions]
        if files_without_roi:
            # Use the current ROI for all
            if self.roi_definition is not None:
                for f in files_without_roi:
                    self.file_roi_definitions[f] = self.roi_definition
            else:
                messagebox.showwarning("No ROI",
                                       "Please draw and complete an ROI before running analysis.")
                return

        for widget in self.thigmotaxis_summary_frame.winfo_children():
            widget.destroy()

        columns = [
            ("file", "File", 160), ("fish", "Fish", 60),
            ("pct_in_roi", "% Time in ROI", 100), ("frames_valid", "Valid Frames", 90),
        ]
        all_rows = []

        for filename in selected_files:
            loaded_file = self.loaded_files[filename]
            roi_bl = self.file_roi_definitions.get(filename)
            if roi_bl is None:
                continue

            roi_poly = ShapelyPolygon(roi_bl)
            pixels_to_bl = 1.0 / loaded_file.metadata.body_length
            n_fish = loaded_file.n_fish
            n_frames = loaded_file.n_frames

            for fish_idx in range(n_fish):
                in_roi = 0
                valid = 0
                for frame_idx in range(0, n_frames, sample_interval):
                    pos = loaded_file.trajectories[frame_idx, fish_idx, :]
                    if np.isnan(pos[0]):
                        continue
                    x_bl = pos[0] * pixels_to_bl
                    y_bl = (loaded_file.metadata.video_height - pos[1]) * pixels_to_bl
                    valid += 1
                    if roi_poly.contains(Point(x_bl, y_bl)):
                        in_roi += 1

                pct = (in_roi / valid * 100) if valid > 0 else 0
                all_rows.append((filename, f"Fish {fish_idx}", f"{pct:.1f}", valid))

        create_sortable_treeview(self.thigmotaxis_summary_frame, columns, all_rows,
                                 title="ROI Analysis — % Time in Custom Region")

        # Store ROI results so "Update Visualizations" can find them
        for filename in selected_files:
            loaded_file = self.loaded_files[filename]
            loaded_file.roi_analysis_done = True

        if all_rows:
            self.set_status(f"ROI analysis complete for {len(selected_files)} file(s).")

        messagebox.showinfo("ROI Analysis Complete",
                            f"Analyzed {len(selected_files)} file(s) with custom ROI.")

    def _update_spatial_visualizations(self):
        """Update visualizations for files that have been analyzed."""
        # Check analysis mode
        analysis_mode = self.spatial_mode_var.get() if hasattr(self, 'spatial_mode_var') else "border"

        # If in ROI mode, re-run ROI analysis on all files that have ROI definitions
        if analysis_mode == "roi":
            roi_files = [f for f in self.loaded_files
                         if f in self.file_roi_definitions
                         or self.roi_definition is not None]
            if roi_files:
                self._run_roi_analysis(roi_files)
                return
            else:
                messagebox.showinfo("No ROI",
                                     "Draw and complete an ROI first, then run analysis.")
                return

        # Border mode: look for thigmotaxis results
        results_dict = {}
        analyzed_files = []

        for filename, loaded_file in self.loaded_files.items():
            if hasattr(loaded_file, 'thigmotaxis_results') and loaded_file.thigmotaxis_results:
                results_dict[filename] = loaded_file.thigmotaxis_results
                analyzed_files.append(filename)

        if not results_dict:
            messagebox.showinfo("No Results",
                                "No files have been analyzed yet.\n\n"
                                "Click 'Run Analysis on Selected Files' first.")
            return

        self._display_combined_thigmotaxis_summary(results_dict)
        self._plot_combined_thigmotaxis_timeseries(results_dict)
        self._plot_thigmotaxis_comparison_charts(results_dict)
        self._generate_comparison_heatmaps(analyzed_files)

    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================

    def _display_combined_thigmotaxis_summary(self, results_dict: Dict[str, ThigmotaxisResults]):
        """Display combined summary using Treeview tables."""
        for widget in self.thigmotaxis_summary_frame.winfo_children():
            widget.destroy()

        columns = [
            ("file", "File", 160), ("fish", "Fish", 50),
            ("mean_pct", "Mean % Border", 100), ("std_pct", "Std % Border", 90),
            ("mean_center", "Mean % Center", 100), ("samples", "Samples", 70),
        ]

        rows = []
        for filename, results in results_dict.items():
            rows.append((
                filename, results.n_fish,
                f"{results.mean_pct_in_border:.1f}",
                f"{results.std_pct_in_border:.1f}",
                f"{100 - results.mean_pct_in_border:.1f}",
                results.n_samples,
            ))

        create_sortable_treeview(self.thigmotaxis_summary_frame, columns, rows,
                                 title="Thigmotaxis Summary")

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
        
        embed_figure_with_toolbar(fig, self.thigmotaxis_plot_frame)

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
        
        embed_figure_with_toolbar(fig, self.thigmotaxis_compare_frame)

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
        
        embed_figure_with_toolbar(fig, self.heatmap_frame)

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

        embed_figure_with_toolbar(fig, self.heatmap_frame)

    # =========================================================================
    # ROI MODE METHODS
    # =========================================================================

    def _on_spatial_mode_change(self):
        """Toggle visibility of border zone vs ROI settings."""
        mode = self.spatial_mode_var.get()
        if mode == "border":
            self.border_settings_frame.pack(fill="x", padx=10, pady=3)
            self.roi_settings_frame.pack_forget()
        else:
            self.border_settings_frame.pack_forget()
            self.roi_settings_frame.pack(fill="x", padx=10, pady=3)

    def _start_roi_drawing(self):
        """Enable ROI drawing mode on the arena canvas."""
        if self.current_arena_file is None:
            messagebox.showwarning("No File", "Select a file first.")
            return
        self._roi_drawing_mode = True
        self.roi_vertices = []
        self.roi_status_label.config(text="Click on arena to draw ROI vertices...", fg="blue")

    def _clear_roi(self):
        """Clear the current ROI."""
        self.roi_vertices = []
        self.roi_definition = None
        self._roi_drawing_mode = False
        if self.current_arena_file and self.current_arena_file in self.file_roi_definitions:
            del self.file_roi_definitions[self.current_arena_file]
        self.roi_status_label.config(text="ROI cleared", fg="gray")
        # Redraw arena without ROI
        if self.current_arena_file:
            self._setup_arena_canvas_for_file(self.current_arena_file)

    def _complete_roi(self):
        """Complete the ROI polygon."""
        if len(self.roi_vertices) < 3:
            messagebox.showwarning("Insufficient Vertices",
                                   "Please define at least 3 ROI vertices.")
            return
        if self.current_arena_file is None:
            return

        roi_bl = np.array(self.roi_vertices)
        self.roi_definition = roi_bl
        self.file_roi_definitions[self.current_arena_file] = roi_bl
        self._roi_drawing_mode = False

        # Draw ROI on canvas
        if self.arena_ax is not None:
            roi_polygon = MplPolygon(roi_bl, closed=True, fill=True,
                                     facecolor='cyan', alpha=0.2,
                                     edgecolor='cyan', linewidth=2,
                                     linestyle='--', label='ROI')
            self.arena_ax.add_patch(roi_polygon)
            self.arena_ax.legend(loc='upper right', fontsize=8)
            self.arena_canvas.draw()

        self.roi_status_label.config(
            text=f"ROI defined ({len(self.roi_vertices)} vertices)", fg="green")

    # =========================================================================
    # METHODS TEXT
    # =========================================================================

    def _update_spatial_methods_text(self, results_dict, border_pct, sample_interval):
        """Generate a methods paragraph for the spatial/thigmotaxis analysis."""
        self.spatial_methods_text.delete("1.0", tk.END)

        first_file = None
        for filename in results_dict:
            if filename in self.loaded_files:
                first_file = self.loaded_files[filename]
                break
        if first_file is None:
            return

        cal = first_file.calibration
        fps = cal.frame_rate
        n_files = len(results_dict)
        interval_s = sample_interval / fps
        border_pct_display = border_pct * 100

        text = (
            "METHODS \u2014 Spatial Analysis (Thigmotaxis)\n"
            "==========================================\n\n"
            "The following is a draft methods paragraph. Edit as needed.\n\n"
            "---\n\n"
            f"Thigmotaxis (wall-hugging behavior) was assessed using the "
            f"Zebrafish Free Swim Analyzer (v2.1). The arena boundary was "
            f"manually defined as a polygon on the background image for each "
            f"recording. A border zone was defined by insetting the arena "
            f"polygon by {border_pct_display:.0f}% of its width (using "
            f"Shapely polygon buffering), creating an outer annular region "
            f"representing the wall-proximal area and an inner region "
            f"representing the center.\n\n"
            f"Fish positions were sampled every {sample_interval} frames "
            f"({interval_s:.2f} s at {fps:.1f} fps). At each sample, each "
            f"fish's position was classified as either in the border zone "
            f"or center zone using Shapely point-in-polygon testing. "
            f"Fish with missing (NaN) position data at a given sample were "
            f"excluded from that sample's calculation. "
            f"The percentage of samples in the border zone was computed per "
            f"fish and averaged across fish for each file.\n\n"
            f"Position density heatmaps were generated using 2D histogram "
            f"binning of all valid fish positions, with NaN gaps linearly "
            f"interpolated prior to binning.\n\n"
            f"Analysis included {n_files} file(s).\n\n"
            "---\n\n"
            "KEY PARAMETERS:\n"
            f"  Border zone:           {border_pct_display:.0f}% of arena width (inset)\n"
            f"  Sample interval:       {sample_interval} frames ({interval_s:.2f} s)\n"
            f"  Distance unit:         {cal.unit_name}\n"
            f"  Frame rate:            {fps:.1f} fps\n"
            f"  Calibration:           {cal.scale_factor:.6f} {cal.unit_name}/px\n"
            f"  Arena type:            manually defined polygon\n"
        )

        self.spatial_methods_text.insert("1.0", text)

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
