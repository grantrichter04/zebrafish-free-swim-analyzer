"""
fish_analyzer/data_structures.py
================================
Foundation layer containing all data structures (dataclasses) that hold
trajectory data, metadata, and calibration settings.

These classes don't perform analysis - they organize and validate data.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np


@dataclass
class IdTrackerMetadata:
    """
    Metadata from an idtracker.ai trajectory file.

    idtracker.ai outputs a .npy file containing both the position data AND
    information about the video and tracking quality. This class stores
    that metadata in an organized way.
    """
    video_width: int
    video_height: int
    frames_per_second: float
    body_length: float
    estimated_accuracy: float
    fraction_identified: float
    n_individuals: int
    identity_labels: List[str]
    idtracker_version: str
    raw_metadata: Dict[str, Any]

    @classmethod
    def from_idtracker_file(cls, metadata_dict: dict) -> 'IdTrackerMetadata':
        """
        Factory method: Create IdTrackerMetadata from a raw dictionary.
        
        This is a @classmethod, meaning you call it on the class itself:
            metadata = IdTrackerMetadata.from_idtracker_file(data_dict)
        """
        try:
            return cls(
                video_width=int(metadata_dict['width']),
                video_height=int(metadata_dict['height']),
                frames_per_second=float(metadata_dict['frames_per_second']),
                body_length=float(metadata_dict['body_length']),
                estimated_accuracy=float(metadata_dict['estimated_accuracy']),
                fraction_identified=float(metadata_dict['fraction_identified']),
                n_individuals=len(metadata_dict['identities_labels']),
                identity_labels=list(metadata_dict['identities_labels']),
                idtracker_version=str(metadata_dict['version']),
                raw_metadata=metadata_dict
            )
        except KeyError as e:
            raise ValueError(
                f"Missing required field in idtracker.ai metadata: {e.args[0]}\n"
                f"Available fields: {list(metadata_dict.keys())}"
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data type in metadata: {e}")

    def quality_summary(self) -> str:
        """Generate human-readable tracking quality report."""
        lines = [
            "Tracking Quality Report:",
            f"  Estimated accuracy: {self.estimated_accuracy:.1%}",
            f"  Frames identified: {self.fraction_identified:.1%}",
            f"  Body length: {self.body_length:.1f} pixels"
        ]
        if self.estimated_accuracy > 0.95 and self.fraction_identified > 0.95:
            lines.append("  Assessment: Excellent tracking quality ✓")
        elif self.estimated_accuracy > 0.85 and self.fraction_identified > 0.85:
            lines.append("  Assessment: Good tracking quality")
        else:
            lines.append("  Assessment: ⚠ Low tracking quality - review results carefully")
        return "\n".join(lines)


@dataclass
class CalibrationSettings:
    """
    Settings for converting pixel coordinates to meaningful units.

    THE CALIBRATION PROBLEM:
    Raw tracking data is in pixels. To make results biologically meaningful
    and comparable across experiments, we convert to "body lengths" (BL)
    or physical units (cm, mm).

    EXAMPLE:
    If body_length = 50 pixels and a fish moves 100 pixels:
    - pixels_per_unit = 50
    - scale_factor = 1/50 = 0.02
    - distance in BL = 100 * 0.02 = 2.0 BL
    """
    pixels_per_unit: float
    unit_name: str
    frame_rate: float

    def __post_init__(self):
        """Validation that runs automatically after __init__."""
        if self.pixels_per_unit <= 0:
            raise ValueError(f"Pixels per unit must be positive, got {self.pixels_per_unit}")
        if self.frame_rate <= 0:
            raise ValueError(f"Frame rate must be positive, got {self.frame_rate}")
        if not self.unit_name or not self.unit_name.strip():
            raise ValueError("Unit name cannot be empty")

    @property
    def scale_factor(self) -> float:
        """Multiplier to convert pixels → units."""
        return 1.0 / self.pixels_per_unit

    @classmethod
    def from_body_lengths(cls, body_length_pixels: float, frame_rate: float) -> 'CalibrationSettings':
        """Create calibration using body length as the unit (most common)."""
        return cls(pixels_per_unit=body_length_pixels, unit_name="BL", frame_rate=frame_rate)

    @classmethod
    def from_physical_measurement(cls, pixels_per_unit: float, unit_name: str, 
                                   frame_rate: float) -> 'CalibrationSettings':
        """Create calibration using a physical measurement (e.g., cm)."""
        return cls(pixels_per_unit=pixels_per_unit, unit_name=unit_name, frame_rate=frame_rate)

    @classmethod
    def no_calibration(cls, frame_rate: float) -> 'CalibrationSettings':
        """Keep data in raw pixels (1 pixel = 1 unit)."""
        return cls(pixels_per_unit=1.0, unit_name="pixels", frame_rate=frame_rate)

    def convert_distance(self, pixels: float) -> float:
        """Convert a distance from pixels to calibrated units."""
        return pixels * self.scale_factor

    def convert_speed(self, pixels_per_frame: float) -> float:
        """Convert speed from pixels/frame to units/second."""
        return pixels_per_frame * self.scale_factor * self.frame_rate

    def __str__(self):
        return (f"{self.unit_name} ({self.pixels_per_unit:.2f} pixels per "
                f"{self.unit_name}, {self.frame_rate:.1f} fps)")


@dataclass
class LoadedTrajectoryFile:
    """
    A complete package containing everything about one trajectory file.

    This is the main data container that holds:
    - Raw trajectory data (x,y positions for each fish in each frame)
    - Metadata about the recording
    - Calibration settings
    - Results from analyses (filled in later)

    TRAJECTORY DATA SHAPE:
    trajectories is a 3D numpy array with shape (n_frames, n_individuals, 2)

    Example for 10,000 frames and 5 fish:
    - trajectories.shape = (10000, 5, 2)
    - trajectories[0, 0, :] = [x, y] of fish 0 in frame 0
    - trajectories[:, 0, :] = all positions of fish 0 (shape: 10000×2)
    """
    nickname: str
    file_path: Path
    metadata: IdTrackerMetadata
    trajectories: np.ndarray
    calibration: CalibrationSettings
    background_image_path: Optional[Path] = None
    video_file_path: Optional[Path] = None
    
    # These get filled in after analysis
    processed_data: Optional[List] = None
    shoaling_results: Optional[Any] = None  # Will be ShoalingResults
    thigmotaxis_results: Optional[Any] = None  # Will be ThigmotaxisResults

    def __post_init__(self):
        """Validate trajectory array shape after initialization."""
        actual_shape = self.trajectories.shape
        if len(actual_shape) != 3:
            raise ValueError(
                f"Trajectory array should be 3D (frames, individuals, coords), "
                f"got {len(actual_shape)}D array with shape {actual_shape}"
            )
        if actual_shape[1] != self.metadata.n_individuals:
            raise ValueError(
                f"Trajectory data has {actual_shape[1]} individuals but "
                f"metadata specifies {self.metadata.n_individuals}"
            )
        if actual_shape[2] != 2:
            raise ValueError(f"Expected 2 coordinates (x, y), got {actual_shape[2]}")
        if self.background_image_path is None:
            self.background_image_path = self._find_background_image()

    def _find_background_image(self) -> Optional[Path]:
        """Look for background.png in idtracker.ai's expected location."""
        try:
            experiment_folder = self.file_path.parent.parent
            background_path = experiment_folder / "preprocessing" / "background.png"
            if background_path.exists():
                return background_path
            return None
        except Exception:
            return None

    @property
    def n_frames(self) -> int:
        return self.trajectories.shape[0]

    @property
    def n_fish(self) -> int:
        return self.trajectories.shape[1]

    @property
    def duration_seconds(self) -> float:
        return self.n_frames / self.calibration.frame_rate

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    def summary(self) -> str:
        """Create detailed human-readable summary for display."""
        lines = [
            f"File: {self.nickname}",
            f"Path: {self.file_path.name}",
            "",
            f"Recording: {self.duration_minutes:.1f} minutes ({self.n_frames} frames)",
            f"Individuals tracked: {self.n_fish}",
            f"Frame rate: {self.calibration.frame_rate:.1f} fps",
            f"Resolution: {self.metadata.video_width} x {self.metadata.video_height} pixels",
            "",
            "Calibration:",
            f"  Units: {self.calibration.unit_name}",
            f"  Scale: {self.calibration.pixels_per_unit:.2f} pixels per {self.calibration.unit_name}",
            "",
        ]
        lines.append(self.metadata.quality_summary())
        lines.append("")
        lines.append("Data Quality per Individual:")
        quality_data = self.calculate_quality_per_individual()
        for info in quality_data:
            lines.append(
                f"  Fish {info['fish_id']} ({info['identity_label']}): "
                f"{info['valid_percentage']:.1f}% valid "
                f"({info['valid_frames']}/{info['total_frames']} frames)"
            )
            if info['valid_percentage'] < 50:
                lines.append("    ⚠ Warning: Less than 50% valid data")
            elif info['valid_percentage'] < 80:
                lines.append("    ⚠ Caution: Some data gaps present")
        return "\n".join(lines)

    def calculate_quality_per_individual(self) -> List[Dict[str, Any]]:
        """Calculate tracking quality statistics for each fish."""
        quality_info = []
        for fish_id in range(self.n_fish):
            fish_traj = self.trajectories[:, fish_id, :]
            valid_mask = ~(np.isnan(fish_traj[:, 0]) | np.isnan(fish_traj[:, 1]))
            valid_count = np.sum(valid_mask)
            nan_count = self.n_frames - valid_count
            valid_pct = (valid_count / self.n_frames) * 100 if self.n_frames > 0 else 0
            quality_info.append({
                'fish_id': fish_id,
                'identity_label': self.metadata.identity_labels[fish_id],
                'total_frames': self.n_frames,
                'valid_frames': valid_count,
                'nan_frames': nan_count,
                'valid_percentage': valid_pct
            })
        return quality_info
