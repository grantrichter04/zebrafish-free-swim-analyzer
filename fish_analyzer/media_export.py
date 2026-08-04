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


class ExportFrameSource:
    """Sequential frame reader owned by one export.

    Deliberately not VideoFrameReader: that class runs a background preload
    thread which calls cap.set() and cap.read() on the same capture the
    foreground uses, with only the frame cache locked. It also copies every
    frame twice through an LRU that a single forward pass cannot benefit from.

    Measured on a 1288x964 MJPG session: 9.9 ms/frame reading sequentially
    against 64.7 ms/frame when seeking per frame.
    """

    def __init__(self, video_path, start_frame=0):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        if start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def read(self):
        """Next frame as RGB uint8, or None at end of stream."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def cursor_x(t, x0, px_per_unit):
    """Pixel column for time `t` on a linear axis.

    Kept as a function of the transform rather than of a matplotlib axes so it
    can be tested without rendering anything.
    """
    return int(round(x0 + t * px_per_unit))


class TimeStrip:
    """The time panel rasterised once, with the cursor stamped per frame.

    Valid only while the axes are fixed for the whole session, which is true of
    the NND, IID and Hull panels (xlim is set once at rebuild) and false of the
    Bout panel, which scrolls its xlim every frame - use ScrollingStrip there.

    Rendering the figure per frame costs 50-100 ms; this costs a memcpy and a
    line, which is the same trick the live view uses when it blits.
    """

    def __init__(self, figure, axes, color=(255, 60, 60), width=2):
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        self._pristine = rgba[:, :, :3].copy()
        self._buffer = self._pristine.copy()
        self.color = color
        self.width = width

        x_at_0 = axes.transData.transform((0.0, 0.0))[0]
        x_at_1 = axes.transData.transform((1.0, 0.0))[0]
        self._x0 = x_at_0
        self._px_per_unit = x_at_1 - x_at_0

    @property
    def height(self):
        return self._pristine.shape[0]

    @property
    def width_px(self):
        return self._pristine.shape[1]

    def at(self, t):
        """The strip with the cursor at time `t`. Reuses one buffer."""
        np.copyto(self._buffer, self._pristine)
        x = cursor_x(t, self._x0, self._px_per_unit)
        x = max(0, min(self._buffer.shape[1] - 1, x))
        cv2.line(self._buffer, (x, 0), (x, self._buffer.shape[0]),
                 self.color, self.width)
        return self._buffer


class ScrollingStrip:
    """Time panel re-rendered per frame, for panels whose axes move.

    The Bout panel resets xlim to a window around the current frame on every
    update, so a strip rasterised once cannot represent it - cropping a
    pre-rendered image would drag the y-axis out of frame. This costs a full
    matplotlib draw per frame, which is why the export dialog warns first.
    """

    def __init__(self, figure, axes, window_s, total_s, color=(255, 60, 60),
                 width=2):
        self._figure = figure
        self._axes = axes
        self._window_s = window_s
        self._total_s = total_s
        self.color = color
        self.width = width

        figure.canvas.draw()
        self._height = np.asarray(figure.canvas.buffer_rgba()).shape[0]

    @property
    def height(self):
        return self._height

    def at(self, t):
        """Redraw the panel with its window centred on `t`."""
        start = max(0.0, t - self._window_s / 2)
        end = start + self._window_s
        if end > self._total_s:
            end = self._total_s
            start = max(0.0, end - self._window_s)

        self._axes.set_xlim(start, end)
        self._figure.canvas.draw()
        strip = np.asarray(self._figure.canvas.buffer_rgba())[:, :, :3].copy()

        x_at_start = self._axes.transData.transform((start, 0.0))[0]
        x_at_end = self._axes.transData.transform((end, 0.0))[0]
        px_per_s = (x_at_end - x_at_start) / max(1e-9, end - start)
        x = cursor_x(t - start, x_at_start, px_per_s)
        x = max(0, min(strip.shape[1] - 1, x))
        cv2.line(strip, (x, 0), (x, strip.shape[0]), self.color, self.width)
        return strip
