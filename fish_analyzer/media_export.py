"""
fish_analyzer/media_export.py
=============================
Writing Video Inspector overlays out as images and clips.

No tkinter: the GUI supplies settings and a progress callback, this module
does the work. Frames are RGB uint8 throughout; only the sinks convert to BGR,
because that is what cv2.VideoWriter and cv2.imwrite expect.
"""
from pathlib import Path

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class CodecUnavailable(RuntimeError):
    """No usable video encoder. Raised instead of leaving a 0-byte file."""


def save_frame_png(rgb, path):
    """Write an RGB frame as a PNG. Raises IOError if cv2 refuses."""
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"cv2.imwrite refused to write {path}")


class PngSequenceSink:
    """Numbered lossless frames. Cannot fail on a missing codec."""

    def __init__(self, directory, fps, size):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.size = size
        self._n = 0

    def write(self, rgb):
        out = self.path / f"frame_{self._n:06d}.png"
        save_frame_png(rgb, out)
        self._n += 1

    def close(self):
        pass

    def discard_partial(self):
        """Completed PNGs are individually valid, so cancelling keeps them."""


class Mp4Sink:
    """MP4 via mp4v, falling back to MJPG/.avi when the encoder will not open.

    OpenCV signals a missing codec by returning a writer whose isOpened() is
    False and then silently accepting writes, which leaves an empty file. Every
    codec is checked before use.
    """

    CODECS = [("mp4v", ".mp4"), ("MJPG", ".avi")]

    def __init__(self, path, fps, size):
        path = Path(path)
        self.fps = fps
        self.size = size
        self._writer = None
        self.path = None
        self.fourcc = None

        for fourcc, ext in self.CODECS:
            candidate = path.with_suffix(ext)
            writer = cv2.VideoWriter(
                str(candidate), cv2.VideoWriter_fourcc(*fourcc), fps, size
            )
            if writer.isOpened():
                self._writer = writer
                self.path = candidate
                self.fourcc = fourcc
                return
            writer.release()
            if candidate.exists() and candidate.stat().st_size == 0:
                candidate.unlink()

        raise CodecUnavailable(
            "No usable video encoder was found (tried "
            + ", ".join(c for c, _ in self.CODECS)
            + "). Export as a PNG sequence instead."
        )

    def write(self, rgb):
        self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def discard_partial(self):
        """A truncated MP4 is unplayable, so a cancelled export removes it."""
        self.close()
        if self.path is not None and self.path.exists():
            self.path.unlink()
