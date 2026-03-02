"""
fish_analyzer/processing.py
===========================
Layer 2: Individual Trajectory Processing and Metrics Calculation

This module transforms raw pixel data into clean, calibrated trajectories
and calculates individual behavioral metrics like speed, distance, and turn angles.

KEY TRANSFORMATIONS:
1. Flip Y axis (video Y=0 is top, we want Y=0 at bottom)
2. Apply calibration (pixels → body lengths)
3. Smooth trajectories to reduce tracking noise
4. Calculate derivatives (speed, acceleration)
5. Compute behavioral metrics (distance, sinuosity, turns)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import traja
from traja import TrajaDataFrame

# Import from our own package
from .data_structures import LoadedTrajectoryFile


@dataclass
class ProcessingParameters:
    """
    Parameters controlling how trajectories are processed.

    SMOOTHING EXPLANATION:
    Raw tracking has small random errors ("jitter"). Savitzky-Golay smoothing
    fits a polynomial to a sliding window of points, reducing noise while
    preserving real movement patterns better than simple averaging.

    - smoothing_window: How many frames to consider (must be odd!)
    - Higher value = smoother but may blur fast movements

    TURN ANGLE LAG EXPLANATION:
    To measure turning, we compare heading at frame N to heading at frame N+lag.
    Using consecutive frames (lag=1) gives noisy results because fish might
    barely move in one frame. Larger lag (e.g., 10 frames) gives more stable,
    meaningful turn measurements.
    """
    apply_smoothing: bool = True
    smoothing_window: int = 5
    smoothing_polynomial_order: int = 3
    min_valid_points: int = 1
    min_valid_percentage: float = 0.01
    turn_angle_lag: int = 10
    rest_speed_threshold: float = 0.5

    def validate(self):
        """Check that all parameters are valid. Raises ValueError if not."""
        if self.smoothing_window % 2 == 0:
            raise ValueError(
                f"Smoothing window must be odd, got {self.smoothing_window}. "
                f"Try {self.smoothing_window + 1}."
            )
        if self.smoothing_window < 3:
            raise ValueError(f"Smoothing window must be at least 3, got {self.smoothing_window}")
        if self.turn_angle_lag < 1:
            raise ValueError(f"Turn angle lag must be at least 1, got {self.turn_angle_lag}")
        if not 0 < self.min_valid_percentage <= 1.0:
            raise ValueError(f"Valid percentage must be between 0 and 1, got {self.min_valid_percentage}")
        if self.rest_speed_threshold < 0:
            raise ValueError(f"Rest threshold must be non-negative, got {self.rest_speed_threshold}")

    @classmethod
    def default_for_fish(cls) -> 'ProcessingParameters':
        """Get default parameters that work well for fish tracking."""
        return cls(
            apply_smoothing=True,
            smoothing_window=5,
            smoothing_polynomial_order=3,
            min_valid_points=1,
            min_valid_percentage=0.01,
            turn_angle_lag=10,
            rest_speed_threshold=0.5
        )


@dataclass
class FishTrajectory:
    """
    Complete information about one fish's trajectory and calculated metrics.
    
    This combines the processed spatial data (in the TrajaDataFrame) with
    all the derived metrics like speed, distance, sinuosity, etc.
    """
    fish_id: int
    identity_label: str
    trajectory: TrajaDataFrame
    metrics: Dict[str, Any] = field(default_factory=dict)
    n_valid_frames: int = 0
    n_total_frames: int = 0

    @property
    def valid_percentage(self) -> float:
        """What fraction of frames have valid (non-NaN) positions?"""
        if self.n_total_frames == 0:
            return 0.0
        return self.n_valid_frames / self.n_total_frames

    def summary(self) -> str:
        """Generate a human-readable summary of this fish's data."""
        lines = [
            f"Fish {self.fish_id} (ID: {self.identity_label})",
            f"Valid frames: {self.n_valid_frames}/{self.n_total_frames} ({self.valid_percentage:.1%})",
        ]
        if self.metrics:
            lines.append("\nMetrics:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                elif not isinstance(value, dict):  # Skip nested dicts like speed_time_series
                    lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class TrajectoryProcessor:
    """
    Converts raw idtracker.ai trajectory data into processed TrajaDataFrames.
    
    This class handles the coordinate transformations and smoothing. It creates
    one FishTrajectory object for each fish in the file.
    """

    def __init__(self, loaded_file: LoadedTrajectoryFile, params: ProcessingParameters):
        """
        Initialize the processor.
        
        Parameters
        ----------
        loaded_file : LoadedTrajectoryFile
            The loaded trajectory data to process
        params : ProcessingParameters
            Settings controlling smoothing, etc.
        """
        self.file = loaded_file
        self.params = params
        self.params.validate()

    def process_all_fish(self) -> List[FishTrajectory]:
        """
        Process trajectories for all fish in the file.
        
        Returns
        -------
        List[FishTrajectory]
            Processed trajectory for each fish that had sufficient valid data
        """
        processed_fish = []
        print(f"\nProcessing {self.file.n_fish} fish from {self.file.nickname}...")
        print(f"Using smoothing window: {self.params.smoothing_window} frames")
        print(f"Turn angle lag: {self.params.turn_angle_lag} frames")

        for fish_idx in range(self.file.n_fish):
            try:
                fish_traj = self._process_single_fish(fish_idx)
                if fish_traj is not None:
                    processed_fish.append(fish_traj)
                    print(f"  Fish {fish_idx}: ✓ ({fish_traj.valid_percentage:.1%} valid data)")
                else:
                    print(f"  Fish {fish_idx}: ✗ (insufficient valid data)")
            except Exception as e:
                print(f"  Fish {fish_idx}: ✗ (error: {e})")
                continue

        print(f"Successfully processed {len(processed_fish)}/{self.file.n_fish} fish\n")
        return processed_fish

    def _process_single_fish(self, fish_idx: int) -> Optional[FishTrajectory]:
        """Process trajectory for a single fish."""
        # Extract raw coordinates for this fish
        raw_coords = self.file.trajectories[:, fish_idx, :]
        
        # Transform coordinates (flip Y, apply calibration)
        transformed_coords = self._transform_coordinates(raw_coords)
        
        # Create DataFrame with time column
        df = self._create_dataframe_with_time(transformed_coords)

        # Check data quality
        valid_mask = ~(df['x'].isna() | df['y'].isna())
        n_valid = valid_mask.sum()
        n_total = len(df)
        valid_pct = n_valid / n_total if n_total > 0 else 0

        if n_valid < self.params.min_valid_points:
            return None
        if valid_pct < self.params.min_valid_percentage:
            return None

        # Create TrajaDataFrame (special pandas DataFrame for trajectories)
        trj = TrajaDataFrame(df)
        trj.fps = self.file.calibration.frame_rate
        trj.spatial_units = self.file.calibration.unit_name
        trj.time_units = "s"

        # Apply smoothing if requested
        if self.params.apply_smoothing:
            try:
                trj = traja.smooth_sg(
                    trj, 
                    w=self.params.smoothing_window, 
                    p=self.params.smoothing_polynomial_order
                )
            except Exception as e:
                # SVD convergence or other errors - continue with unsmoothed data
                print(f"    Note: Smoothing failed for fish {fish_idx}, using raw trajectory")

        return FishTrajectory(
            fish_id=fish_idx,
            identity_label=self.file.metadata.identity_labels[fish_idx],
            trajectory=trj,
            n_valid_frames=n_valid,
            n_total_frames=n_total
        )

    def _transform_coordinates(self, raw_coords: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from pixel space to calibrated space.
        
        This does two things:
        1. Flips Y axis (video convention is Y=0 at top, science is Y=0 at bottom)
        2. Scales by calibration factor (pixels → body lengths or other units)
        """
        transformed = raw_coords.copy()
        
        # Flip Y: new_y = video_height - old_y
        transformed[:, 1] = self.file.metadata.video_height - transformed[:, 1]
        
        # Scale by calibration
        transformed = transformed * self.file.calibration.scale_factor
        
        return transformed

    def _create_dataframe_with_time(self, coords: np.ndarray) -> pd.DataFrame:
        """Create a DataFrame with x, y, and time columns."""
        n_frames = len(coords)
        frame_numbers = np.arange(n_frames)
        time_seconds = frame_numbers / self.file.calibration.frame_rate
        
        return pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'time': time_seconds
        })


class MetricsCalculator:
    """
    Calculate behavioral metrics from processed trajectories.
    
    This class computes various metrics that describe how the fish moved:
    - Total distance traveled
    - Net displacement (straight-line from start to end)
    - Sinuosity (how much the path meanders)
    - Speed statistics (mean, max, median)
    - Turn angle statistics
    """

    def __init__(self, params: ProcessingParameters):
        self.params = params

    def calculate_all_metrics(self, fish: FishTrajectory) -> FishTrajectory:
        """
        Calculate all metrics for a fish and store them in fish.metrics.
        
        Parameters
        ----------
        fish : FishTrajectory
            The fish to calculate metrics for. Modified in place.
            
        Returns
        -------
        FishTrajectory
            The same fish object, with metrics populated.
        """
        trj = fish.trajectory
        
        # Distance metrics
        fish.metrics['total_distance'] = self._calc_total_distance(trj)
        fish.metrics['net_displacement'] = self._calc_net_displacement(trj)
        fish.metrics['sinuosity'] = self._calc_sinuosity(
            fish.metrics['total_distance'],
            fish.metrics['net_displacement']
        )
        
        # Speed metrics
        fish.metrics.update(self._calc_speed_metrics(trj))
        fish.metrics['speed_time_series'] = self._get_speed_time_series(trj)
        
        # Turn metrics
        fish.metrics.update(self._calc_turn_metrics(trj))
        
        return fish

    def _calc_total_distance(self, trj: TrajaDataFrame) -> float:
        """Calculate total path length (sum of all step lengths)."""
        return traja.length(trj)

    def _calc_net_displacement(self, trj: TrajaDataFrame) -> float:
        """Calculate straight-line distance from start to end."""
        return traja.distance(trj)

    def _calc_sinuosity(self, total_distance: float, net_displacement: float) -> float:
        """
        Calculate path sinuosity (tortuosity).
        
        Sinuosity = total_distance / net_displacement
        
        - Value = 1.0 means perfectly straight path
        - Value = 2.0 means path is twice as long as straight line
        - Higher values indicate more meandering/exploration
        """
        if net_displacement > 0 and not np.isnan(net_displacement):
            return total_distance / net_displacement
        return np.nan

    def _calc_speed_metrics(self, trj: TrajaDataFrame) -> Dict[str, float]:
        """Calculate summary statistics for speed."""
        try:
            derivs = trj.traja.get_derivatives()
            speed = derivs['speed'].dropna()
            
            if len(speed) == 0:
                return {
                    'mean_speed': np.nan,
                    'max_speed': np.nan,
                    'std_speed': np.nan,
                    'median_speed': np.nan
                }
            
            return {
                'mean_speed': float(speed.mean()),
                'max_speed': float(speed.max()),
                'std_speed': float(speed.std()),
                'median_speed': float(speed.median())
            }
        except Exception as e:
            print(f"Warning: Speed calculation failed: {e}")
            return {
                'mean_speed': np.nan,
                'max_speed': np.nan,
                'std_speed': np.nan,
                'median_speed': np.nan
            }

    def _get_speed_time_series(self, trj: TrajaDataFrame) -> Optional[Dict[str, np.ndarray]]:
        """Get speed as a time series for plotting."""
        try:
            derivs = trj.traja.get_derivatives()
            speed = derivs['speed'].values
            time = trj['time'].values
            
            # Handle length mismatch (derivatives are shorter by 1)
            min_len = min(len(speed), len(time))
            
            return {
                'time': time[:min_len],
                'speed': speed[:min_len]
            }
        except Exception:
            return None

    def _calc_turn_metrics(self, trj: TrajaDataFrame) -> Dict[str, float]:
        """
        Calculate turn angle statistics.
        
        Turn angles are measured as change in heading direction over 
        self.params.turn_angle_lag frames.
        """
        try:
            lag = self.params.turn_angle_lag
            
            # Calculate heading from position changes
            dx = trj.x.diff(periods=lag)
            dy = trj.y.diff(periods=lag)
            headings = np.degrees(np.arctan2(dy, dx))
            
            # Calculate turn angles (change in heading)
            turn_angles = headings.diff()
            
            # Normalize to [-180, 180] range
            turn_angles = ((turn_angles + 180) % 360) - 180
            turn_angles = turn_angles.dropna()
            
            if len(turn_angles) == 0:
                return {
                    'mean_turn_angle': np.nan,
                    'mean_abs_turn_angle': np.nan,
                    'std_turn_angle': np.nan
                }
            
            return {
                'mean_turn_angle': float(turn_angles.mean()),
                'mean_abs_turn_angle': float(np.abs(turn_angles).mean()),
                'std_turn_angle': float(turn_angles.std())
            }
        except Exception as e:
            print(f"Warning: Turn angle calculation failed: {e}")
            return {
                'mean_turn_angle': np.nan,
                'mean_abs_turn_angle': np.nan,
                'std_turn_angle': np.nan
            }


def process_and_analyze_file(
    loaded_file: LoadedTrajectoryFile,
    params: Optional[ProcessingParameters] = None
) -> List[FishTrajectory]:
    """
    Complete pipeline: process trajectories and calculate all metrics.
    
    This is a convenience function that combines TrajectoryProcessor and
    MetricsCalculator into a single call.
    
    Parameters
    ----------
    loaded_file : LoadedTrajectoryFile
        The loaded trajectory data
    params : ProcessingParameters, optional
        Processing settings. Uses defaults if not provided.
        
    Returns
    -------
    List[FishTrajectory]
        Processed trajectories with metrics for all fish
    """
    if params is None:
        params = ProcessingParameters.default_for_fish()
    
    # Process trajectories
    processor = TrajectoryProcessor(loaded_file, params)
    fish_list = processor.process_all_fish()
    
    # Calculate metrics
    calculator = MetricsCalculator(params)
    for fish in fish_list:
        calculator.calculate_all_metrics(fish)
    
    return fish_list
