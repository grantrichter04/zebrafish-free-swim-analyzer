"""
fish_analyzer/fish_outlines.py
==============================
The outlines idtracker.ai drew around the animals in a frame, found again
from what it saved in the session folder: the thresholds in session.json, and
the background and region-of-interest images in preprocessing/.

This repeats idtracker.ai's own masking (process_frame in its
base/animals_detection/segmentation.py) rather than importing it, so the
Video Inspector does not need idtracker.ai loaded to show what was segmented.
It reads nothing but those three files and the frame it is given.
"""
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


class OutlineFinder:
    """Finds the blobs idtracker.ai would find in a frame of one session."""

    def __init__(self, intensity_ths, area_ths, background: Optional[np.ndarray] = None,
                 roi_mask: Optional[np.ndarray] = None):
        self.intensity_ths = (float(intensity_ths[0]), float(intensity_ths[1]))
        self.area_ths = (float(area_ths[0]), float(area_ths[1]))
        self.background = background
        self.roi_mask = roi_mask

    @classmethod
    def for_session(cls, session_folder: Path) -> Optional["OutlineFinder"]:
        """From a session folder, or None if its settings cannot be read."""
        session_folder = Path(session_folder)
        try:
            with open(session_folder / "session.json", encoding="utf-8") as file:
                settings = json.load(file)
            intensity_ths = settings["intensity_ths"]
            area_ths = settings["area_ths"]
        except (OSError, ValueError, KeyError):
            return None

        background = None
        if settings.get("use_bkg"):
            background = _read_gray(session_folder / "preprocessing" / "background.png")
            if background is None:
                return None     # the thresholds mean nothing without it
            # Before idtracker.ai 5.2.5 a background threshold was stored as
            # (0, value); it is (value, 255) now.
            if intensity_ths[0] == 0:
                intensity_ths = (intensity_ths[1], 255)
        roi_mask = _read_gray(session_folder / "preprocessing" / "ROI_mask.png")
        return cls(intensity_ths, area_ths, background, roi_mask)

    def find(self, frame: np.ndarray) -> List[np.ndarray]:
        """The outlines in `frame` (RGB or grey), in the frame's own pixels."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        # A session tracked at reduced resolution saved its background that size.
        reference = self.background if self.background is not None else self.roi_mask
        if reference is not None and reference.shape != gray.shape:
            gray = cv2.resize(gray, (reference.shape[1], reference.shape[0]),
                              interpolation=cv2.INTER_AREA)

        if self.background is None:
            mask = cv2.inRange(gray, self.intensity_ths[0], self.intensity_ths[1])
        else:
            mask = (cv2.absdiff(self.background, gray) > self.intensity_ths[0]
                    ).astype(np.uint8)
        if self.roi_mask is not None and self.roi_mask.shape == mask.shape:
            mask[self.roi_mask == 0] = 0

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        kept = [c for c in contours
                if self.area_ths[0] <= cv2.contourArea(c) <= self.area_ths[1]]
        if gray.shape != (height, width):
            scale = np.array([width / gray.shape[1], height / gray.shape[0]])
            kept = [np.round(c * scale).astype(np.int32) for c in kept]
        return kept


def _read_gray(path: Path) -> Optional[np.ndarray]:
    try:
        # np.fromfile, not cv2.imread: the data lives under paths with
        # apostrophes and other characters OpenCV's reader trips on.
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except (OSError, ValueError):
        return None
    return image
