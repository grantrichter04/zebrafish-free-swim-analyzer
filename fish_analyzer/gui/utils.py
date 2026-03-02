"""
fish_analyzer/gui/utils.py
==========================
Shared utility functions for the GUI components.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def smooth_time_series(data: np.ndarray, window_seconds: float, frame_rate: float) -> np.ndarray:
    """
    Apply smoothing to a time series using a rolling average.
    
    Parameters
    ----------
    data : np.ndarray
        The time series data to smooth
    window_seconds : float
        The smoothing window size in seconds
    frame_rate : float
        The frame rate of the data (fps)
        
    Returns
    -------
    np.ndarray
        Smoothed data (same length as input)
    """
    if window_seconds <= 0:
        return data
    
    # Calculate window size in samples
    window_samples = int(window_seconds * frame_rate)
    if window_samples < 1:
        window_samples = 1
    
    # Handle NaN values by interpolating, smoothing, then restoring NaN positions
    nan_mask = np.isnan(data)
    if np.all(nan_mask):
        return data
    
    # Create a copy for smoothing
    smoothed = data.copy()
    
    # For data with NaNs, we'll work only on valid sections
    if np.any(nan_mask):
        # Simple approach: interpolate over NaNs, smooth, then restore NaNs
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) < 2:
            return data
        
        # Linear interpolation over NaN gaps
        smoothed = np.interp(
            np.arange(len(data)),
            valid_indices,
            data[valid_indices]
        )
        
        # Apply uniform filter (moving average)
        smoothed = uniform_filter1d(smoothed, size=window_samples, mode='nearest')
    else:
        smoothed = uniform_filter1d(data, size=window_samples, mode='nearest')
    
    return smoothed
