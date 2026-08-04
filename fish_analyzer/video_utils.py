"""
fish_analyzer/video_utils.py
============================
Utilities for reading video frames for visualization.

Reads are cached and, when the caller walks forward, avoid seeking - on a
1288x964 MJPG recording that is 9.9 ms/frame against 64.7 ms when seeking every
frame.

A background preload thread used to live here. It drove cap.set() and cap.read()
on the same VideoCapture as the foreground with only the cache locked, and
playing 250 frames through it aborted the process with ffmpeg decode errors. It
was also unreachable: its only entry point, read_frame_fast, had no callers.
Removed rather than locked - see tests/test_video_utils.py.
"""

from pathlib import Path
from typing import Optional, Dict
import numpy as np
from threading import Lock

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Note: OpenCV not installed. Install with 'pip install opencv-python' for video frame support.")


class VideoFrameCache:
    """
    LRU cache for video frames with thread-safe access.
    
    PERFORMANCE: Increased default size to 100 frames for smoother scrubbing.
    """
    
    def __init__(self, max_cache_size: int = 100):
        """
        Parameters
        ----------
        max_cache_size : int
            Maximum number of frames to keep in cache (default 100)
        """
        self.cache: Dict[int, np.ndarray] = {}
        self.max_cache_size = max_cache_size
        self.access_order = []
        self.lock = Lock()  # Thread-safe
    
    def get(self, frame_num: int) -> Optional[np.ndarray]:
        """Get frame from cache if available (thread-safe)."""
        with self.lock:
            if frame_num in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(frame_num)
                self.access_order.append(frame_num)
                return self.cache[frame_num].copy()
        return None
    
    def put(self, frame_num: int, frame: np.ndarray):
        """Add frame to cache, evicting old frames if needed (thread-safe)."""
        with self.lock:
            # Remove if already exists
            if frame_num in self.cache:
                self.access_order.remove(frame_num)
            
            # Evict oldest if cache is full
            while len(self.cache) >= self.max_cache_size:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            
            # Add new frame
            self.cache[frame_num] = frame.copy()
            self.access_order.append(frame_num)
    
    def clear(self):
        """Clear all cached frames."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    

class VideoFrameReader:
    """
    Frame reader with an LRU cache and a seek-free path for forward walks.

    Single-threaded by design: the VideoCapture is not synchronised, so it must
    only ever be touched by its owner. An export that wants its own read
    position should open its own capture - see media_export.ExportFrameSource.
    """
    
    def __init__(self, video_path: Path, cache_size: int = 100, downsample_factor: float = 1.0):
        """
        Parameters
        ----------
        video_path : Path
            Path to the video file
        cache_size : int
            Number of frames to cache (default 100 for smoother playback)
        downsample_factor : float
            Factor to downsample frames (0.5 = half size, faster). 1.0 = no downsampling.
            
        Raises
        ------
        RuntimeError
            If OpenCV is not installed
        FileNotFoundError
            If video file doesn't exist
        ValueError
            If video file can't be opened
        """
        if not CV2_AVAILABLE:
            raise RuntimeError(
                "OpenCV is required for video frame reading.\n"
                "Install with: pip install opencv-python"
            )
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.cache = VideoFrameCache(cache_size)
        self.downsample_factor = downsample_factor
        
        # Open video to get properties
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Lets read_frame skip the seek when the caller walks forward.
        self.last_frame_read = -1
        self.sequential_reads = 0

        print(f"Video loaded: {self.width}x{self.height}, {self.total_frames} frames @ {self.fps:.1f} fps")
        if downsample_factor != 1.0:
            print(f"Downsampling: {downsample_factor}x (display: {int(self.width*downsample_factor)}x{int(self.height*downsample_factor)})")
    
    def read_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Read a specific frame from the video.

        Parameters
        ----------
        frame_number : int
            Frame number to read (0-indexed)

        Returns
        -------
        np.ndarray or None
            Frame as RGB numpy array, or None if failed
        """
        # Check cache first
        cached = self.cache.get(frame_number)
        if cached is not None:
            return cached

        # Validate frame number
        if frame_number < 0 or frame_number >= self.total_frames:
            return None
        
        # Detect sequential reading pattern
        is_sequential = (frame_number == self.last_frame_read + 1)
        
        if is_sequential:
            self.sequential_reads += 1
            # Just read next frame (fast!)
            ret, frame = self.cap.read()
        else:
            self.sequential_reads = 0
            # Need to seek (slow)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Downsample if requested
        if self.downsample_factor != 1.0:
            new_width = int(self.width * self.downsample_factor)
            new_height = int(self.height * self.downsample_factor)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Cache it
        self.cache.put(frame_number, frame_rgb)
        self.last_frame_read = frame_number

        return frame_rgb

    def close(self):
        """Release video file handle and clear cache."""
        if self.cap is not None:
            self.cap.release()
        self.cache.clear()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
