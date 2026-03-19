"""
fish_analyzer/shoaling.py
=========================
Layer 3: Shoaling (Group Behavior) Analysis

Shoaling = fish swimming together as a group. This module calculates metrics
that quantify group cohesion and spatial organization.

METRICS CALCULATED:

1. NND (Nearest Neighbor Distance):
   For each fish, find the distance to its closest neighbor.
   Low NND = tight schooling, fish staying close together.
   High NND = spread out, fish swimming independently.

2. IID (Inter-Individual Distance):
   Average distance between ALL pairs of fish.
   For n fish, there are n*(n-1)/2 unique pairs.
   Less sensitive to outliers than NND.

3. Convex Hull Area:
   The smallest polygon that contains all fish positions.
   Like wrapping a rubber band around pins at each fish location.
   Larger area = more spread out group.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import ConvexHull

# Import from our package
from .data_structures import LoadedTrajectoryFile


@dataclass
class ShoalingParameters:
    """
    Parameters controlling shoaling analysis.
    
    SAMPLING STRATEGY:
    Instead of analyzing every frame (which would be slow and generate too
    much data), we sample at regular intervals (e.g., every 30 frames = 1/sec).
    This captures the same biological information much more efficiently.
    """
    sample_interval_frames: int = 30
    smoothing_window_seconds: float = 5.0

    def validate(self):
        """Validate parameters. Raises ValueError if invalid."""
        if self.sample_interval_frames < 1:
            raise ValueError(
                f"Sample interval must be at least 1 frame, got {self.sample_interval_frames}"
            )
        if self.smoothing_window_seconds < 0:
            raise ValueError(
                f"Smoothing window must be non-negative, got {self.smoothing_window_seconds}"
            )


@dataclass
class ShoalingResults:
    """
    Results from shoaling analysis for one file.
    
    All distances are in body lengths (BL), areas in BL².
    """
    # Time series data (at sampled intervals)
    timestamps: np.ndarray              # Time in seconds for each sample
    frame_indices: np.ndarray           # Original frame numbers for each sample
    mean_nnd_per_sample: np.ndarray     # Mean NND across all fish at each sample
    individual_nnd_per_sample: np.ndarray  # Shape: (n_samples, n_fish)
    mean_iid_per_sample: np.ndarray     # Mean distance between ALL pairs at each sample
    convex_hull_area_per_sample: np.ndarray  # Area of convex hull at each sample (BL²)

    # NND Summary statistics
    mean_nnd: float
    std_nnd: float
    median_nnd: float
    min_nnd: float
    max_nnd: float

    # IID Summary Statistics
    mean_iid: float
    std_iid: float
    median_iid: float
    min_iid: float
    max_iid: float

    # Convex hull summary statistics
    mean_hull_area: float
    std_hull_area: float
    median_hull_area: float
    min_hull_area: float
    max_hull_area: float

    # Per-fish summary
    per_fish_mean_nnd: np.ndarray

    # Quality information
    n_complete_frames: int
    n_total_frames: int
    n_samples_used: int
    n_fish: int
    completeness_percentage: float

    # Parameters used
    sample_interval_frames: int
    body_length_pixels: float
    frame_rate: float = 30.0

    def get_sample_index_for_frame(self, frame_idx: int) -> int:
        """Find the closest analyzed sample index for a given frame number."""
        differences = np.abs(self.frame_indices - frame_idx)
        return int(np.argmin(differences))

    def get_frame_for_sample_index(self, sample_idx: int) -> int:
        """Get the frame number for a given sample index."""
        if 0 <= sample_idx < len(self.frame_indices):
            return int(self.frame_indices[sample_idx])
        return int(self.frame_indices[0])

    def summary(self) -> str:
        """Human-readable summary of shoaling results."""
        lines = [
            "=" * 60,
            "SHOALING ANALYSIS RESULTS",
            "=" * 60,
            "",
            "Data Quality:",
            f"  Complete frames (all fish tracked): {self.n_complete_frames:,} / {self.n_total_frames:,} "
            f"({self.completeness_percentage:.1f}%)",
            f"  Samples analyzed: {self.n_samples_used:,} (every {self.sample_interval_frames} frames)",
            f"  Number of fish: {self.n_fish}",
            "",
            "-" * 60,
            "NEAREST NEIGHBOR DISTANCE (NND)",
            "-" * 60,
            "",
            "Summary Statistics (in body lengths):",
            f"  Mean NND:   {self.mean_nnd:.3f} BL",
            f"  Std Dev:    {self.std_nnd:.3f} BL",
            f"  Median:     {self.median_nnd:.3f} BL",
            f"  Range:      {self.min_nnd:.3f} - {self.max_nnd:.3f} BL",
            "",
            "Per-Fish Mean NND:",
        ]

        for i, fish_nnd in enumerate(self.per_fish_mean_nnd):
            lines.append(f"  Fish {i}: {fish_nnd:.3f} BL")

        lines.append("")
        lines.append("-" * 60)
        lines.append("GLOBAL INTER-INDIVIDUAL DISTANCE (IID)")
        lines.append("-" * 60)
        lines.append("  (Average distance between ALL fish pairs)")
        lines.append("")
        lines.append(f"  Mean IID:   {self.mean_iid:.3f} BL")
        lines.append(f"  Std Dev:    {self.std_iid:.3f} BL")
        lines.append(f"  Median:     {self.median_iid:.3f} BL")
        lines.append(f"  Range:      {self.min_iid:.3f} - {self.max_iid:.3f} BL")

        lines.append("")
        lines.append("-" * 60)
        lines.append("CONVEX HULL AREA")
        lines.append("-" * 60)
        lines.append("")
        lines.append("Summary Statistics (in body lengths²):")
        lines.append(f"  Mean Area:  {self.mean_hull_area:.2f} BL²")
        lines.append(f"  Std Dev:    {self.std_hull_area:.2f} BL²")
        lines.append(f"  Median:     {self.median_hull_area:.2f} BL²")
        lines.append(f"  Range:      {self.min_hull_area:.2f} - {self.max_hull_area:.2f} BL²")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


class ShoalingCalculator:
    """
    Calculates shoaling metrics: NND, IID, and Convex Hull Area.
    
    This is the main workhorse class for group behavior analysis.
    """

    def __init__(self, loaded_file: LoadedTrajectoryFile, params: ShoalingParameters):
        """
        Initialize the calculator.
        
        Parameters
        ----------
        loaded_file : LoadedTrajectoryFile
            The trajectory data to analyze
        params : ShoalingParameters
            Analysis settings (sample interval, etc.)
        """
        self.file = loaded_file
        self.params = params
        self.params.validate()

        # Store conversion factors for pixels → body lengths
        self.body_length_pixels = loaded_file.metadata.body_length
        self.pixels_to_bl = 1.0 / self.body_length_pixels
        self.pixels_sq_to_bl_sq = self.pixels_to_bl ** 2

    def calculate(self) -> ShoalingResults:
        """
        Run the complete shoaling analysis (NND + IID + Convex Hull).
        
        Returns
        -------
        ShoalingResults
            Complete results including time series and summary statistics
        """
        print(f"\nRunning shoaling analysis on {self.file.nickname}...")
        print(f"  Body length: {self.body_length_pixels:.1f} pixels")
        print(f"  Sample interval: every {self.params.sample_interval_frames} frames")

        # Find frames where ALL fish have valid positions
        complete_frame_mask = self._find_complete_frames()
        n_complete = np.sum(complete_frame_mask)
        n_total = self.file.n_frames

        print(f"  Complete frames: {n_complete:,} / {n_total:,} ({100*n_complete/n_total:.1f}%)")

        if n_complete == 0:
            raise ValueError("No frames found where all fish have valid positions.")

        # Get indices of complete frames and sample at regular intervals
        complete_frame_indices = np.where(complete_frame_mask)[0]
        sampled_indices = complete_frame_indices[::self.params.sample_interval_frames]
        n_samples = len(sampled_indices)

        print(f"  Samples to analyze: {n_samples:,}")

        if n_samples < 2:
            raise ValueError(f"Only {n_samples} samples available. Try reducing sample interval.")

        # Initialize arrays to store results
        n_fish = self.file.n_fish
        mean_nnd_per_sample = np.zeros(n_samples)
        individual_nnd_per_sample = np.zeros((n_samples, n_fish))
        mean_iid_per_sample = np.zeros(n_samples)
        convex_hull_area_per_sample = np.zeros(n_samples)
        timestamps = np.zeros(n_samples)

        # Calculate metrics for each sampled frame
        for i, frame_idx in enumerate(sampled_indices):
            positions = self.file.trajectories[frame_idx, :, :]

            # NND: nearest neighbor distance for each fish
            nnd_values = self._calculate_nnd_at_frame(positions)
            nnd_values_bl = nnd_values * self.pixels_to_bl
            mean_nnd_per_sample[i] = np.mean(nnd_values_bl)
            individual_nnd_per_sample[i, :] = nnd_values_bl

            # IID: mean of all pairwise distances
            iid_value = self._calculate_iid_at_frame(positions)
            mean_iid_per_sample[i] = iid_value * self.pixels_to_bl

            # Convex hull area
            hull_area_pixels = self._calculate_convex_hull_area(positions)
            convex_hull_area_per_sample[i] = hull_area_pixels * self.pixels_sq_to_bl_sq

            timestamps[i] = frame_idx / self.file.calibration.frame_rate

        # Calculate summary statistics
        mean_nnd = np.mean(mean_nnd_per_sample)
        std_nnd = np.std(mean_nnd_per_sample)
        median_nnd = np.median(mean_nnd_per_sample)
        min_nnd = np.min(mean_nnd_per_sample)
        max_nnd = np.max(mean_nnd_per_sample)

        mean_iid = np.mean(mean_iid_per_sample)
        std_iid = np.std(mean_iid_per_sample)
        median_iid = np.median(mean_iid_per_sample)
        min_iid = np.min(mean_iid_per_sample)
        max_iid = np.max(mean_iid_per_sample)

        per_fish_mean_nnd = np.mean(individual_nnd_per_sample, axis=0)

        mean_hull_area = np.mean(convex_hull_area_per_sample)
        std_hull_area = np.std(convex_hull_area_per_sample)
        median_hull_area = np.median(convex_hull_area_per_sample)
        min_hull_area = np.min(convex_hull_area_per_sample)
        max_hull_area = np.max(convex_hull_area_per_sample)

        print(f"  Mean NND: {mean_nnd:.3f} BL (std: {std_nnd:.3f})")
        print(f"  Mean IID: {mean_iid:.3f} BL (std: {std_iid:.3f})")
        print(f"  Mean Hull Area: {mean_hull_area:.2f} BL² (std: {std_hull_area:.2f})")
        print(f"  Analysis complete.\n")

        return ShoalingResults(
            timestamps=timestamps,
            frame_indices=sampled_indices,
            mean_nnd_per_sample=mean_nnd_per_sample,
            individual_nnd_per_sample=individual_nnd_per_sample,
            mean_iid_per_sample=mean_iid_per_sample,
            convex_hull_area_per_sample=convex_hull_area_per_sample,
            mean_nnd=mean_nnd, std_nnd=std_nnd, median_nnd=median_nnd,
            min_nnd=min_nnd, max_nnd=max_nnd,
            mean_iid=mean_iid, std_iid=std_iid, median_iid=median_iid,
            min_iid=min_iid, max_iid=max_iid,
            mean_hull_area=mean_hull_area, std_hull_area=std_hull_area,
            median_hull_area=median_hull_area, min_hull_area=min_hull_area,
            max_hull_area=max_hull_area,
            per_fish_mean_nnd=per_fish_mean_nnd,
            n_complete_frames=n_complete, n_total_frames=n_total,
            n_samples_used=n_samples, n_fish=n_fish,
            completeness_percentage=(n_complete / n_total) * 100,
            sample_interval_frames=self.params.sample_interval_frames,
            body_length_pixels=self.body_length_pixels,
            frame_rate=self.file.calibration.frame_rate
        )

    def _find_complete_frames(self) -> np.ndarray:
        """
        Find frames where ALL fish have valid (non-NaN) positions.
        
        Returns
        -------
        np.ndarray
            Boolean mask, True for frames where all fish are tracked
        """
        valid_x = ~np.isnan(self.file.trajectories[:, :, 0])
        valid_y = ~np.isnan(self.file.trajectories[:, :, 1])
        valid_positions = valid_x & valid_y
        return np.all(valid_positions, axis=1)

    def _calculate_nnd_at_frame(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate nearest neighbor distance for each fish at one frame.

        Uses scipy.spatial.distance.cdist for efficient vectorized computation.

        Parameters
        ----------
        positions : np.ndarray
            Shape (n_fish, 2) with [x, y] for each fish

        Returns
        -------
        np.ndarray
            Shape (n_fish,) with NND for each fish in PIXELS
        """
        dist_matrix = cdist(positions, positions, metric='euclidean')
        # Set self-distances to inf so they're not selected as minimum
        np.fill_diagonal(dist_matrix, np.inf)
        return np.min(dist_matrix, axis=1)

    def _calculate_iid_at_frame(self, positions: np.ndarray) -> float:
        """
        Calculate Inter-Individual Distance (mean of ALL pairwise distances).

        ALGORITHM:
        For n fish, calculate distance between every unique pair:
        (0,1), (0,2), (0,3)..., (1,2), (1,3)..., etc.
        That's n*(n-1)/2 pairs total.
        IID = mean of all these distances.

        We use scipy's pdist() which efficiently computes all pairwise
        distances in a vectorized way.

        Returns
        -------
        float
            Mean pairwise distance in PIXELS
        """
        if len(positions) < 2:
            return 0.0
        pairwise_distances = pdist(positions, metric='euclidean')
        return np.mean(pairwise_distances)

    def _calculate_convex_hull_area(self, positions: np.ndarray) -> float:
        """
        Calculate the area of the convex hull containing all fish.

        WHAT IS A CONVEX HULL?
        Imagine placing a pin at each fish's position, then stretching a
        rubber band around all the pins. The shape formed is the convex hull.
        It's the smallest convex polygon that contains all points.

        NOTE: scipy.spatial.ConvexHull.volume is actually AREA for 2D data

        Returns
        -------
        float
            Area in PIXELS² (caller converts to BL²)
        """
        if positions.shape[0] < 3:
            return 0.0
        try:
            hull = ConvexHull(positions)
            return hull.volume  # For 2D, "volume" is actually area
        except Exception:
            return 0.0

    def get_positions_at_frame(self, frame_idx: int) -> Tuple[np.ndarray, bool]:
        """
        Get fish positions at a specific frame, converted to body lengths.
        
        Returns
        -------
        positions_bl : np.ndarray
            Positions in body length units with Y flipped for plotting
        is_complete : bool
            True if all fish have valid positions at this frame
        """
        positions_pixels = self.file.trajectories[frame_idx, :, :]
        is_complete = not np.any(np.isnan(positions_pixels))
        
        # Convert to BL and flip Y for plotting
        positions_bl = positions_pixels.copy()
        positions_bl[:, 1] = self.file.metadata.video_height - positions_bl[:, 1]
        positions_bl = positions_bl * self.pixels_to_bl
        
        return positions_bl, is_complete

    def get_convex_hull_vertices(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get the vertices of the convex hull at a specific frame.
        
        Returns
        -------
        np.ndarray or None
            Hull vertices in BL coordinates, or None if hull can't be computed
        """
        positions_bl, is_complete = self.get_positions_at_frame(frame_idx)
        
        if not is_complete:
            # Filter to valid positions only
            valid_mask = ~np.isnan(positions_bl[:, 0])
            positions_bl = positions_bl[valid_mask]
        
        if len(positions_bl) < 3:
            return None
            
        try:
            hull = ConvexHull(positions_bl)
            return positions_bl[hull.vertices]
        except Exception:
            return None

    def get_all_pairwise_distances_at_frame(self, frame_idx: int) -> Tuple[np.ndarray, list]:
        """
        Get all pairwise distances and the pairs for visualization.
        
        Returns
        -------
        distances : np.ndarray
            Distances in body lengths
        pairs : list
            List of (i, j) tuples indicating which fish form each pair
        """
        positions_pixels = self.file.trajectories[frame_idx, :, :]
        n_fish = positions_pixels.shape[0]

        pairs = []
        distances = []

        for i in range(n_fish):
            for j in range(i + 1, n_fish):
                if not np.isnan(positions_pixels[i, 0]) and not np.isnan(positions_pixels[j, 0]):
                    dx = positions_pixels[i, 0] - positions_pixels[j, 0]
                    dy = positions_pixels[i, 1] - positions_pixels[j, 1]
                    dist = np.sqrt(dx**2 + dy**2) * self.pixels_to_bl
                    pairs.append((i, j))
                    distances.append(dist)

        return np.array(distances), pairs
