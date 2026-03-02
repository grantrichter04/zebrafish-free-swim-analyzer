"""
fish_analyzer/file_loading.py
=============================
Handles loading idtracker.ai trajectory files from disk.

This module is intentionally separate from data_structures because loading
files is an I/O operation that might fail in various ways. Keeping it separate
makes testing and error handling cleaner.
"""

from pathlib import Path
from typing import Optional
import numpy as np

# Import from our own package - this is how modules talk to each other!
from .data_structures import (
    IdTrackerMetadata,
    CalibrationSettings,
    LoadedTrajectoryFile
)


class TrajectoryFileLoader:
    """
    Handles loading idtracker.ai trajectory files from disk.

    idtracker.ai saves data as .npy files containing a dictionary with:
    - 'trajectories': numpy array of positions
    - Metadata fields: 'width', 'height', 'frames_per_second', etc.
    
    Usage:
        # Load from session folder (recommended)
        loaded_file = TrajectoryFileLoader.load_from_session_folder(Path("session_MyExperiment"))
        
        # Or load directly from .npy file
        loaded_file = TrajectoryFileLoader.load_file(Path("trajectories.npy"))
    """

    @staticmethod
    def load_from_session_folder(session_folder: Path,
                                  nickname: Optional[str] = None,
                                  calibration: Optional[CalibrationSettings] = None) -> LoadedTrajectoryFile:
        """
        Load trajectory data from an idtracker.ai session folder.
        
        Automatically finds the trajectories.npy file and background image
        based on the standard idtracker.ai folder structure:
        
            session_folder/
                trajectories/
                    trajectories.npy
                preprocessing/
                    background.png
        
        Parameters
        ----------
        session_folder : Path
            Path to the session folder (e.g., session_MyExperiment)
        nickname : str, optional
            A friendly name for this file. Defaults to the session folder name.
        calibration : CalibrationSettings, optional
            Custom calibration. If not provided, uses body length from file.
            
        Returns
        -------
        LoadedTrajectoryFile
            Complete package containing trajectory data and metadata.
            
        Raises
        ------
        FileNotFoundError
            If the session folder or trajectories.npy doesn't exist
        ValueError
            If the file format is invalid
        """
        if not session_folder.exists():
            raise FileNotFoundError(f"Session folder not found: {session_folder}")
        
        if not session_folder.is_dir():
            raise ValueError(f"Expected a folder, got a file: {session_folder}")
        
        # Find trajectories.npy
        trajectories_path = session_folder / "trajectories" / "trajectories.npy"
        if not trajectories_path.exists():
            # Try alternate location (direct in session folder)
            trajectories_path = session_folder / "trajectories.npy"
            if not trajectories_path.exists():
                raise FileNotFoundError(
                    f"Could not find trajectories.npy in session folder.\n"
                    f"Looked in:\n"
                    f"  - {session_folder / 'trajectories' / 'trajectories.npy'}\n"
                    f"  - {session_folder / 'trajectories.npy'}"
                )
        
        # Use session folder name as default nickname
        if nickname is None:
            folder_name = session_folder.name
            # Remove 'session_' prefix if present
            if folder_name.startswith('session_'):
                nickname = folder_name[8:]  # Remove 'session_' prefix
            else:
                nickname = folder_name
        
        # Find background image
        background_path = session_folder / "preprocessing" / "background.png"
        if not background_path.exists():
            background_path = None
        
        # Load the actual file
        return TrajectoryFileLoader._load_trajectory_file(
            trajectories_path, 
            nickname, 
            calibration,
            background_path
        )

    @staticmethod
    def load_file(file_path: Path,
                  nickname: Optional[str] = None,
                  calibration: Optional[CalibrationSettings] = None) -> LoadedTrajectoryFile:
        """
        Load a trajectory file and return a LoadedTrajectoryFile object.

        Parameters
        ----------
        file_path : Path
            Path to the .npy file from idtracker.ai
        nickname : str, optional
            A friendly name for this file. Defaults to the filename without extension.
        calibration : CalibrationSettings, optional
            Custom calibration. If not provided, uses body length from file.

        Returns
        -------
        LoadedTrajectoryFile
            Complete package containing trajectory data and metadata.

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file format is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if nickname is None:
            nickname = file_path.stem  # filename without extension

        # Try to find associated background image
        background_path = TrajectoryFileLoader._find_background_image(file_path)
        
        return TrajectoryFileLoader._load_trajectory_file(
            file_path, nickname, calibration, background_path
        )

    @staticmethod
    def _load_trajectory_file(file_path: Path,
                               nickname: str,
                               calibration: Optional[CalibrationSettings],
                               background_path: Optional[Path]) -> LoadedTrajectoryFile:
        """
        Internal method to load trajectory file with all parameters specified.
        """
        # Load the .npy file (allow_pickle needed for dict storage)
        try:
            raw_data = np.load(file_path, allow_pickle=True)
        except Exception as e:
            raise ValueError(f"Failed to load numpy file: {e}")

        # idtracker.ai files are "0-d arrays containing a dict" (numpy quirk)
        if raw_data.shape != ():
            raise ValueError(
                f"Expected scalar array containing dictionary, got shape {raw_data.shape}. "
                "This doesn't look like an idtracker.ai trajectory file."
            )
        
        metadata_dict = raw_data.item()  # Extract the dictionary
        
        if not isinstance(metadata_dict, dict):
            raise ValueError(
                f"Expected dictionary inside file, got {type(metadata_dict).__name__}."
            )

        # Parse metadata using our dataclass's factory method
        metadata = IdTrackerMetadata.from_idtracker_file(metadata_dict)

        # Extract trajectories
        if 'trajectories' not in metadata_dict:
            raise ValueError(
                f"File is missing 'trajectories' field. "
                f"Available fields: {list(metadata_dict.keys())}"
            )
        trajectories = metadata_dict['trajectories']

        # Use default calibration if not specified
        if calibration is None:
            calibration = CalibrationSettings.from_body_lengths(
                body_length_pixels=metadata.body_length,
                frame_rate=metadata.frames_per_second
            )

        return LoadedTrajectoryFile(
            nickname=nickname,
            file_path=file_path,
            metadata=metadata,
            trajectories=trajectories,
            calibration=calibration,
            background_image_path=background_path
        )

    @staticmethod
    def _find_background_image(trajectory_path: Path) -> Optional[Path]:
        """
        Look for background image in idtracker.ai output structure.
        
        idtracker.ai typically creates this folder structure:
            experiment_folder/
                session_folder/
                    trajectories/
                        trajectories.npy  <-- your file is here
                    preprocessing/
                        background.png    <-- we're looking for this
        """
        try:
            session_folder = trajectory_path.parent.parent
            background_path = session_folder / "preprocessing" / "background.png"
            
            if background_path.exists():
                return background_path
            
            # Try alternate names
            preprocessing_folder = session_folder / "preprocessing"
            if preprocessing_folder.exists():
                for possible_name in ["background.png", "background.jpg", "bg.png"]:
                    bg = preprocessing_folder / possible_name
                    if bg.exists():
                        return bg
            return None
        except Exception:
            return None
