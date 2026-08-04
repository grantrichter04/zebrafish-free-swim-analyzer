"""Tests for the video frame reader.

Synthetic clips only - a few hundred KB written to pytest's tmp dirs.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fish_analyzer.video_utils import VideoFrameReader  # noqa: E402


def make_clip(path, n_frames=60, size=(160, 120)):
    """Frame i carries red == i, so a returned frame identifies itself."""
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
                         (w, h))
    assert vw.isOpened()
    for i in range(n_frames):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:, :, 2] = i                      # BGR: channel 2 is red
        vw.write(f)
    vw.release()
    return path


def index_of(frame):
    """Recover the encoded index from an RGB frame."""
    return int(np.median(frame[:, :, 0]))


def test_sequential_playback_returns_every_frame_in_order(tmp_path):
    """The failure this guards against was not subtle: a background preload
    thread drove cap.set() and cap.read() on the same VideoCapture as the
    foreground, and playing 250 frames of 1288x964 MJPG aborted the process
    with ffmpeg huffman decode errors."""
    reader = VideoFrameReader(make_clip(tmp_path / "clip.avi"))
    try:
        for i in range(60):
            frame = reader.read_frame(i)
            assert frame is not None, f"frame {i} was not returned"
            assert abs(index_of(frame) - i) <= 2, \
                f"asked for frame {i}, got {index_of(frame)}"
    finally:
        reader.close()


def test_reader_starts_no_background_threads(tmp_path):
    """The preload machinery is gone. Anything reintroducing a thread here
    shares an unsynchronised VideoCapture with the caller."""
    import threading

    before = threading.active_count()
    reader = VideoFrameReader(make_clip(tmp_path / "clip.avi", n_frames=30))
    try:
        for i in range(30):
            reader.read_frame(i)
        assert threading.active_count() == before
    finally:
        reader.close()


def test_random_access_returns_the_requested_frame(tmp_path):
    """Scrubbing jumps around; each seek must land where it was asked."""
    reader = VideoFrameReader(make_clip(tmp_path / "clip.avi"))
    try:
        for i in (40, 5, 23, 59, 0, 31):
            frame = reader.read_frame(i)
            assert frame is not None
            assert abs(index_of(frame) - i) <= 2
    finally:
        reader.close()


def test_out_of_range_returns_none(tmp_path):
    reader = VideoFrameReader(make_clip(tmp_path / "clip.avi", n_frames=10))
    try:
        assert reader.read_frame(-1) is None
        assert reader.read_frame(10) is None
    finally:
        reader.close()


def test_cache_returns_equal_frames_not_shared_buffers(tmp_path):
    """A caller mutating a returned frame must not corrupt the cache: the
    inspector composites overlays directly onto what it gets back."""
    reader = VideoFrameReader(make_clip(tmp_path / "clip.avi", n_frames=10))
    try:
        first = reader.read_frame(3)
        first[:] = 0
        second = reader.read_frame(3)
        assert second is not None
        assert second.any(), "cache handed out the buffer the caller zeroed"
    finally:
        reader.close()
