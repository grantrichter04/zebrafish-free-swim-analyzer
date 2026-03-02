"""
fish_analyzer/spatial.py
========================
Layer 4: Spatial Analysis - Thigmotaxis and Heatmaps

THIGMOTAXIS ("wall-touching response"):
Many animals, when stressed or in unfamiliar environments, stay close to
walls rather than venturing into open spaces. By measuring how much time
fish spend near walls vs. in the center, we can quantify anxiety-like behavior.

ALGORITHM:
1. User defines arena boundary (polygon around the tank)
2. Create "center zone" by shrinking arena inward by X% (e.g., 15%)
3. The ring between arena edge and center zone = "border zone"
4. For each frame, check if each fish is in border or center
5. Calculate percentage of time in each zone

INTERPRETATION:
High thigmotaxis (much time near walls) often indicates:
- Stress or anxiety-like behavior
- Unfamiliar environment
- Cautious personality type

HEATMAPS:
Divide the arena into a grid and count how often each cell is occupied.
Shows spatial preferences beyond just edge vs. center.

ENHANCEMENTS (v2.0):
- Per-fish time series tracking (for individual fish line plotting)
- ArenaDefinition with copy() method for applying to multiple files
- Helper function for shared heatmap color scales
- Better smoothing support in results class
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from scipy.ndimage import uniform_filter1d

# Import from our package
from .data_structures import LoadedTrajectoryFile

# Check if shapely is available (optional dependency)
try:
    from shapely.geometry import Polygon as ShapelyPolygon, Point
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Note: shapely not installed. Install with 'pip install shapely' for thigmotaxis analysis.")


@dataclass
class ArenaDefinition:
    """
    Stores the arena polygon definition.
    
    The arena is defined by the user clicking vertices on the background image.
    We store both pixel coordinates (for raw data access) and body length
    coordinates (for analysis and display).
    
    ENHANCED: Added copy() method and normalized coordinate support
    for applying same arena to multiple files.
    """
    vertices_pixels: np.ndarray  # Shape: (n_vertices, 2) in pixel coordinates
    vertices_bl: np.ndarray      # Shape: (n_vertices, 2) in body length units

    def to_shapely(self, use_bl: bool = True) -> 'ShapelyPolygon':
        """
        Convert to a shapely Polygon for geometric operations.
        
        Parameters
        ----------
        use_bl : bool
            If True, use body length coordinates; if False, use pixels
            
        Returns
        -------
        ShapelyPolygon
            The arena as a shapely polygon
            
        Raises
        ------
        RuntimeError
            If shapely is not installed
        """
        if not SHAPELY_AVAILABLE:
            raise RuntimeError("Shapely not installed. Install with: pip install shapely")
        verts = self.vertices_bl if use_bl else self.vertices_pixels
        return ShapelyPolygon(verts)
    
    def copy(self) -> 'ArenaDefinition':
        """Create a deep copy of this arena definition."""
        return ArenaDefinition(
            vertices_pixels=self.vertices_pixels.copy(),
            vertices_bl=self.vertices_bl.copy()
        )
    
    def get_normalized_vertices(self, video_width: int, video_height: int, 
                                 body_length: float) -> np.ndarray:
        """
        Get vertices as normalized coordinates (0-1) relative to video dimensions.
        Useful for applying same arena shape to different files.
        
        Parameters
        ----------
        video_width : int
            Video width in pixels
        video_height : int
            Video height in pixels
        body_length : float
            Body length in pixels for BL conversion
            
        Returns
        -------
        np.ndarray
            Normalized vertices (0-1 range)
        """
        pixels_to_bl = 1.0 / body_length
        width_bl = video_width * pixels_to_bl
        height_bl = video_height * pixels_to_bl
        
        normalized = np.zeros_like(self.vertices_bl)
        normalized[:, 0] = self.vertices_bl[:, 0] / width_bl
        normalized[:, 1] = self.vertices_bl[:, 1] / height_bl
        return normalized
    
    @classmethod
    def from_normalized(cls, normalized_vertices: np.ndarray, 
                        video_width: int, video_height: int,
                        body_length: float) -> 'ArenaDefinition':
        """
        Create arena from normalized (0-1) coordinates.
        
        Parameters
        ----------
        normalized_vertices : np.ndarray
            Vertices in normalized (0-1) coordinates
        video_width : int
            Video width in pixels
        video_height : int  
            Video height in pixels
        body_length : float
            Body length in pixels
            
        Returns
        -------
        ArenaDefinition
            New arena definition
        """
        pixels_to_bl = 1.0 / body_length
        width_bl = video_width * pixels_to_bl
        height_bl = video_height * pixels_to_bl
        
        vertices_bl = np.zeros_like(normalized_vertices)
        vertices_bl[:, 0] = normalized_vertices[:, 0] * width_bl
        vertices_bl[:, 1] = normalized_vertices[:, 1] * height_bl
        
        # Convert to pixels
        vertices_pixels = vertices_bl / pixels_to_bl
        vertices_pixels[:, 1] = video_height - vertices_pixels[:, 1]
        
        return cls(vertices_pixels=vertices_pixels, vertices_bl=vertices_bl)


@dataclass
class ThigmotaxisResults:
    """
    Results from thigmotaxis analysis.
    
    Contains both per-fish statistics and time series data showing
    how wall-hugging behavior changes over time.
    
    ENHANCED: Now includes per-fish time series data for individual fish plotting.
    """
    # Per-fish overall results
    time_in_border_pct: np.ndarray      # Shape: (n_fish,) - % time in border zone
    time_in_center_pct: np.ndarray      # Shape: (n_fish,) - % time in center zone

    # Time series (sampled) - GROUP level
    timestamps: np.ndarray              # Time in seconds
    frame_indices: np.ndarray           # Frame numbers analyzed
    fish_in_border_per_sample: np.ndarray  # Shape: (n_samples,) - count of fish in border
    pct_in_border_per_sample: np.ndarray   # Shape: (n_samples,) - % of fish in border
    
    # Per-fish time series for individual lines
    # Shape: (n_samples, n_fish) - binary: 1 if in border, 0 if in center, NaN if invalid
    per_fish_in_border_samples: np.ndarray

    # Summary stats
    mean_pct_in_border: float           # Overall mean across fish
    std_pct_in_border: float

    # Parameters used
    border_zone_pct: float              # Border zone as % of arena
    n_fish: int
    n_samples: int
    frame_rate: float = 30.0
    sample_interval: int = 30

    def get_smoothed_group_timeseries(self, window_seconds: float = 5.0) -> np.ndarray:
        """
        Get smoothed group percentage time series.
        
        Parameters
        ----------
        window_seconds : float
            Smoothing window in seconds
            
        Returns
        -------
        np.ndarray
            Smoothed percentage data
        """
        if window_seconds <= 0 or self.n_samples < 3:
            return self.pct_in_border_per_sample.copy()
        
        # Calculate samples per second
        if self.n_samples > 1:
            time_span = self.timestamps[-1] - self.timestamps[0]
            samples_per_sec = (self.n_samples - 1) / time_span if time_span > 0 else 1
        else:
            samples_per_sec = 1
        
        window_samples = max(1, int(window_seconds * samples_per_sec))
        if window_samples >= self.n_samples:
            window_samples = max(1, self.n_samples // 3)
        
        return uniform_filter1d(self.pct_in_border_per_sample, size=window_samples, mode='nearest')
    
    def get_smoothed_fish_timeseries(self, fish_idx: int, window_seconds: float = 5.0) -> np.ndarray:
        """
        Get smoothed time series for a single fish.
        Returns percentage (0-100) of time in border for each sample window.
        
        Parameters
        ----------
        fish_idx : int
            Which fish to get data for
        window_seconds : float
            Smoothing window in seconds
            
        Returns
        -------
        np.ndarray
            Smoothed percentage data for this fish
        """
        if fish_idx >= self.n_fish:
            return np.full(self.n_samples, np.nan)
        
        # Get per-fish data (convert binary to percentage for smoothing)
        fish_data = self.per_fish_in_border_samples[:, fish_idx] * 100
        
        if window_seconds <= 0 or self.n_samples < 3:
            return fish_data
        
        # Handle NaN values
        nan_mask = np.isnan(fish_data)
        if np.all(nan_mask):
            return fish_data
        
        # Calculate window size
        if self.n_samples > 1:
            time_span = self.timestamps[-1] - self.timestamps[0]
            samples_per_sec = (self.n_samples - 1) / time_span if time_span > 0 else 1
        else:
            samples_per_sec = 1
        
        window_samples = max(1, int(window_seconds * samples_per_sec))
        if window_samples >= self.n_samples:
            window_samples = max(1, self.n_samples // 3)
        
        # Fill NaN with nearest valid for smoothing
        fish_clean = fish_data.copy()
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) > 0:
            for i in range(len(fish_clean)):
                if nan_mask[i]:
                    # Find nearest valid value
                    distances = np.abs(valid_indices - i)
                    nearest_idx = valid_indices[np.argmin(distances)]
                    fish_clean[i] = fish_data[nearest_idx]
        
        smoothed = uniform_filter1d(fish_clean, size=window_samples, mode='nearest')
        
        return smoothed

    def summary(self) -> str:
        """Generate human-readable summary of thigmotaxis results."""
        lines = [
            "=" * 60,
            "THIGMOTAXIS ANALYSIS",
            "=" * 60,
            "",
            f"Border zone: {self.border_zone_pct:.0%} from wall",
            f"Samples analyzed: {self.n_samples}",
            "",
            "Per-Fish Time in Border Zone:",
        ]

        for i, (border, center) in enumerate(zip(self.time_in_border_pct, self.time_in_center_pct)):
            lines.append(f"  Fish {i}: {border:.1f}% border, {center:.1f}% center")

        lines.append("")
        lines.append(f"Group mean: {self.mean_pct_in_border:.1f}% ± {self.std_pct_in_border:.1f}%")
        lines.append("")

        return "\n".join(lines)


class ThigmotaxisCalculator:
    """
    Calculate thigmotaxis (wall-hugging) behavior.
    
    This class determines what percentage of time each fish spends
    near the walls vs. in the center of the arena.
    
    ENHANCED: Now tracks per-fish time series data.
    """

    def __init__(self, loaded_file: LoadedTrajectoryFile,
                 arena: ArenaDefinition,
                 border_pct: float = 0.15,
                 sample_interval: int = 30):
        """
        Initialize the calculator.
        
        Parameters
        ----------
        loaded_file : LoadedTrajectoryFile
            The trajectory data to analyze
        arena : ArenaDefinition
            The arena boundary polygon
        border_pct : float
            Border zone as fraction of arena size (0.15 = 15% from wall)
        sample_interval : int
            Analyze every Nth frame for time series data
        """
        self.file = loaded_file
        self.arena = arena
        self.border_pct = border_pct
        self.sample_interval = sample_interval

        self.body_length_pixels = loaded_file.metadata.body_length
        self.pixels_to_bl = 1.0 / self.body_length_pixels

        if not SHAPELY_AVAILABLE:
            raise RuntimeError(
                "Shapely required for thigmotaxis analysis. "
                "Install with: pip install shapely"
            )

    def calculate(self) -> ThigmotaxisResults:
        """
        Run thigmotaxis analysis with per-fish time series tracking.
        
        Returns
        -------
        ThigmotaxisResults
            Complete results including per-fish and time series data
        """
        print(f"\nRunning thigmotaxis analysis...")
        print(f"  Border zone: {self.border_pct:.0%} from wall")

        # Create arena and inner zone polygons
        arena_poly = self.arena.to_shapely(use_bl=True)

        # Buffer inward to create center zone (negative buffer shrinks)
        arena_bounds = arena_poly.bounds  # (minx, miny, maxx, maxy)
        arena_width = arena_bounds[2] - arena_bounds[0]
        arena_height = arena_bounds[3] - arena_bounds[1]
        buffer_distance = min(arena_width, arena_height) * self.border_pct

        center_poly = arena_poly.buffer(-buffer_distance)

        if center_poly.is_empty:
            raise ValueError("Border zone too large - no center area remains")

        n_fish = self.file.n_fish
        n_frames = self.file.n_frames

        # Track per-fish overall statistics
        frames_in_border = np.zeros(n_fish)
        frames_in_center = np.zeros(n_fish)
        frames_valid = np.zeros(n_fish)

        # Sample frames for time series
        sample_frames = np.arange(0, n_frames, self.sample_interval)
        n_samples = len(sample_frames)

        # Initialize arrays
        fish_in_border_per_sample = np.zeros(n_samples)
        per_fish_in_border_samples = np.full((n_samples, n_fish), np.nan)  # Per-fish tracking
        timestamps = np.zeros(n_samples)

        print(f"  Analyzing {n_frames} frames ({n_samples} samples)...")

        # Create sample frame set for fast lookup
        sample_frame_set = set(sample_frames)
        sample_frame_to_idx = {frame: idx for idx, frame in enumerate(sample_frames)}

        # Main analysis loop - check every frame for per-fish stats
        for frame_idx in range(n_frames):
            positions_pixels = self.file.trajectories[frame_idx, :, :]
            is_sample_frame = frame_idx in sample_frame_set
            
            if is_sample_frame:
                sample_idx = sample_frame_to_idx[frame_idx]
                timestamps[sample_idx] = frame_idx / self.file.calibration.frame_rate

            for fish_idx in range(n_fish):
                if np.isnan(positions_pixels[fish_idx, 0]):
                    continue

                # Convert to BL and flip y
                x_bl = positions_pixels[fish_idx, 0] * self.pixels_to_bl
                y_bl = (self.file.metadata.video_height - positions_pixels[fish_idx, 1]) * self.pixels_to_bl

                point = Point(x_bl, y_bl)
                frames_valid[fish_idx] += 1

                in_border = False
                if arena_poly.contains(point):
                    if center_poly.contains(point):
                        frames_in_center[fish_idx] += 1
                    else:
                        frames_in_border[fish_idx] += 1
                        in_border = True
                
                # Record per-fish state at sample frames
                if is_sample_frame:
                    per_fish_in_border_samples[sample_idx, fish_idx] = 1.0 if in_border else 0.0

        # Calculate group time series (count and percentage)
        for sample_idx in range(n_samples):
            fish_states = per_fish_in_border_samples[sample_idx, :]
            valid_mask = ~np.isnan(fish_states)
            if np.any(valid_mask):
                fish_in_border_per_sample[sample_idx] = np.sum(fish_states[valid_mask])

        # Calculate overall percentages
        time_in_border_pct = np.zeros(n_fish)
        time_in_center_pct = np.zeros(n_fish)

        for i in range(n_fish):
            if frames_valid[i] > 0:
                time_in_border_pct[i] = (frames_in_border[i] / frames_valid[i]) * 100
                time_in_center_pct[i] = (frames_in_center[i] / frames_valid[i]) * 100

        pct_in_border_per_sample = (fish_in_border_per_sample / n_fish) * 100

        mean_pct = np.mean(time_in_border_pct)
        std_pct = np.std(time_in_border_pct)

        print(f"  Mean time in border: {mean_pct:.1f}% ± {std_pct:.1f}%")

        return ThigmotaxisResults(
            time_in_border_pct=time_in_border_pct,
            time_in_center_pct=time_in_center_pct,
            timestamps=timestamps,
            frame_indices=sample_frames,
            fish_in_border_per_sample=fish_in_border_per_sample,
            pct_in_border_per_sample=pct_in_border_per_sample,
            per_fish_in_border_samples=per_fish_in_border_samples,
            mean_pct_in_border=mean_pct,
            std_pct_in_border=std_pct,
            border_zone_pct=self.border_pct,
            n_fish=n_fish,
            n_samples=n_samples,
            frame_rate=self.file.calibration.frame_rate,
            sample_interval=self.sample_interval
        )


class HeatmapGenerator:
    """
    Generate position density heatmaps.
    
    Creates 2D histograms showing where fish spend their time,
    either for all fish combined or individually.
    
    ENHANCED: Added methods for comparing heatmaps across files.
    """

    def __init__(self, loaded_file: LoadedTrajectoryFile,
                 grid_size: int = 50,
                 arena: Optional[ArenaDefinition] = None):
        """
        Initialize the generator.
        
        Parameters
        ----------
        loaded_file : LoadedTrajectoryFile
            The trajectory data
        grid_size : int
            Number of bins in each dimension (grid_size x grid_size)
        arena : ArenaDefinition, optional
            Arena boundary for masking/clipping
        """
        self.file = loaded_file
        self.grid_size = grid_size
        self.arena = arena

        self.body_length_pixels = loaded_file.metadata.body_length
        self.pixels_to_bl = 1.0 / self.body_length_pixels

        # Calculate bounds in body lengths
        self.width_bl = loaded_file.metadata.video_width * self.pixels_to_bl
        self.height_bl = loaded_file.metadata.video_height * self.pixels_to_bl

    def generate_combined_heatmap(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate heatmap combining all fish positions.
        
        Returns
        -------
        heatmap : np.ndarray
            2D array with percentage of time spent in each cell
        x_edges : np.ndarray
            Bin edges in x direction
        y_edges : np.ndarray
            Bin edges in y direction
        """
        all_x = []
        all_y = []

        for fish_idx in range(self.file.n_fish):
            positions = self.file.trajectories[:, fish_idx, :]
            valid_mask = ~np.isnan(positions[:, 0])

            x_bl = positions[valid_mask, 0] * self.pixels_to_bl
            y_bl = (self.file.metadata.video_height - positions[valid_mask, 1]) * self.pixels_to_bl

            all_x.extend(x_bl)
            all_y.extend(y_bl)

        # Create 2D histogram
        heatmap, x_edges, y_edges = np.histogram2d(
            all_x, all_y,
            bins=self.grid_size,
            range=[[0, self.width_bl], [0, self.height_bl]]
        )

        # Normalize to percentage of time
        total_points = len(all_x)
        if total_points > 0:
            heatmap = (heatmap / total_points) * 100

        return heatmap.T, x_edges, y_edges  # Transpose for correct orientation

    def generate_individual_heatmap(self, fish_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate heatmap for a single fish.
        
        Parameters
        ----------
        fish_idx : int
            Which fish to generate heatmap for
            
        Returns
        -------
        heatmap : np.ndarray
            2D array with percentage of time spent in each cell
        x_edges : np.ndarray
            Bin edges in x direction
        y_edges : np.ndarray
            Bin edges in y direction
        """
        positions = self.file.trajectories[:, fish_idx, :]
        valid_mask = ~np.isnan(positions[:, 0])

        x_bl = positions[valid_mask, 0] * self.pixels_to_bl
        y_bl = (self.file.metadata.video_height - positions[valid_mask, 1]) * self.pixels_to_bl

        heatmap, x_edges, y_edges = np.histogram2d(
            x_bl, y_bl,
            bins=self.grid_size,
            range=[[0, self.width_bl], [0, self.height_bl]]
        )

        total_points = len(x_bl)
        if total_points > 0:
            heatmap = (heatmap / total_points) * 100

        return heatmap.T, x_edges, y_edges
    
    def generate_all_individual_heatmaps(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Generate heatmaps for all fish with consistent binning.
        
        Returns
        -------
        heatmaps : List[np.ndarray]
            List of heatmap arrays, one per fish
        x_edges : np.ndarray
            Bin edges in x direction
        y_edges : np.ndarray
            Bin edges in y direction
        """
        heatmaps = []
        x_edges = None
        y_edges = None
        
        for fish_idx in range(self.file.n_fish):
            hm, xe, ye = self.generate_individual_heatmap(fish_idx)
            heatmaps.append(hm)
            if x_edges is None:
                x_edges = xe
                y_edges = ye
        
        return heatmaps, x_edges, y_edges


def compute_shared_heatmap_scale(heatmaps: List[np.ndarray], percentile: float = 99) -> Tuple[float, float]:
    """
    Compute shared vmin/vmax for a list of heatmaps for fair comparison.
    
    Parameters
    ----------
    heatmaps : List[np.ndarray]
        List of heatmap arrays
    percentile : float
        Upper percentile to use for vmax (avoids outliers)
        
    Returns
    -------
    vmin, vmax : Tuple[float, float]
        Color scale limits
    """
    all_values = np.concatenate([h.flatten() for h in heatmaps])
    all_values = all_values[~np.isnan(all_values)]
    
    if len(all_values) == 0:
        return 0, 1
    
    vmin = 0  # Density always starts at 0
    vmax = np.percentile(all_values, percentile)
    
    if vmax <= vmin:
        vmax = np.max(all_values) if len(all_values) > 0 else 1
    
    return vmin, vmax
