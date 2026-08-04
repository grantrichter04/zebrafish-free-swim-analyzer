"""Tests for clip export. No display and no real video needed."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

cv2 = pytest.importorskip("cv2")

from fish_analyzer import media_export  # noqa: E402
from fish_analyzer.media_export import (  # noqa: E402
    CodecUnavailable,
    Mp4Sink,
    PngSequenceSink,
)


def frame(h=32, w=48):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_png_sequence_writes_one_numbered_file_per_frame(tmp_path):
    sink = PngSequenceSink(tmp_path / "clip", fps=30.0, size=(48, 32))
    for _ in range(3):
        sink.write(frame())
    sink.close()

    written = sorted(p.name for p in (tmp_path / "clip").glob("*.png"))
    assert written == ["frame_000000.png", "frame_000001.png",
                       "frame_000002.png"]


def test_mp4_sink_writes_a_playable_file(tmp_path):
    out = tmp_path / "clip.mp4"
    sink = Mp4Sink(out, fps=30.0, size=(48, 32))
    for _ in range(10):
        sink.write(frame())
    sink.close()

    assert sink.path.exists() and sink.path.stat().st_size > 0


def test_mp4_sink_falls_back_when_the_primary_codec_will_not_open(tmp_path,
                                                                 monkeypatch):
    """OpenCV returns an unopened writer rather than raising, which is how a
    zero-byte file gets left behind. The fallback must engage."""
    real = cv2.VideoWriter
    attempts = []

    class Writer:
        def __init__(self, path, fourcc, fps, size):
            attempts.append(fourcc)
            self._real = real(path, fourcc, fps, size)
            # Reject only the first codec tried.
            self._ok = len(attempts) > 1 and self._real.isOpened()

        def isOpened(self):
            return self._ok

        def write(self, f):
            self._real.write(f)

        def release(self):
            self._real.release()

    monkeypatch.setattr(media_export.cv2, "VideoWriter", Writer)

    sink = Mp4Sink(tmp_path / "clip.mp4", fps=30.0, size=(48, 32))
    sink.write(frame())
    sink.close()

    assert len(attempts) == 2, "fallback codec was not attempted"
    assert sink.path.suffix == ".avi", "fallback should write .avi"


def test_mp4_sink_raises_when_no_codec_opens(tmp_path, monkeypatch):
    class DeadWriter:
        def __init__(self, *a):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(media_export.cv2, "VideoWriter", DeadWriter)

    with pytest.raises(CodecUnavailable):
        Mp4Sink(tmp_path / "clip.mp4", fps=30.0, size=(48, 32))

    leftover = tmp_path / "clip.mp4"
    assert not leftover.exists() or leftover.stat().st_size == 0
