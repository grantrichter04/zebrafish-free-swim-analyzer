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


def make_video(path, n_frames=20, size=(48, 32)):
    """A tiny MJPG clip whose frame i has red channel == i * 10."""
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
                         (w, h))
    assert vw.isOpened()
    for i in range(n_frames):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:, :, 2] = i * 10          # BGR: channel 2 is red
        vw.write(f)
    vw.release()
    return path


def test_frame_source_reads_the_requested_range_in_order(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource

    video = make_video(tmp_path / "in.avi")
    with ExportFrameSource(video, start_frame=5) as src:
        frames = [src.read() for _ in range(3)]

    reds = [int(f[0, 0, 0]) for f in frames]      # RGB out: channel 0 is red
    assert reds == pytest.approx([50, 60, 70], abs=6)


def test_frame_source_returns_rgb(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource

    video = make_video(tmp_path / "in.avi")
    with ExportFrameSource(video, start_frame=10) as src:
        f = src.read()

    assert f[0, 0, 0] > f[0, 0, 2], "channels look like BGR, not RGB"


def test_frame_source_returns_none_past_the_end(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource

    video = make_video(tmp_path / "in.avi", n_frames=5)
    with ExportFrameSource(video, start_frame=0) as src:
        for _ in range(5):
            assert src.read() is not None
        assert src.read() is None


def test_cursor_x_maps_time_to_a_pixel_column():
    from fish_analyzer.media_export import cursor_x

    # An axes whose data origin sits at pixel column 80, 1.5 px per second.
    assert cursor_x(0.0, x0=80.0, px_per_unit=1.5) == 80
    assert cursor_x(100.0, x0=80.0, px_per_unit=1.5) == 230


def test_time_strip_moves_the_cursor_without_redrawing():
    """The static panels are rasterised once; only the cursor changes."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from fish_analyzer.media_export import TimeStrip

    fig = Figure(figsize=(4, 1), dpi=50)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot([0, 10], [0, 1])
    ax.set_xlim(0, 10)

    strip = TimeStrip(fig, ax)
    early = strip.at(1.0).copy()
    late = strip.at(9.0).copy()

    assert early.shape == late.shape
    assert not np.array_equal(early, late), "cursor did not move"
    assert strip.height == early.shape[0]


def test_scrolling_strip_moves_its_window_with_time():
    """The Bout panel scrolls, so its strip must differ between timepoints."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from fish_analyzer.media_export import ScrollingStrip

    fig = Figure(figsize=(4, 1), dpi=50)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, 100, 500), np.sin(np.linspace(0, 100, 500)))

    strip = ScrollingStrip(fig, ax, window_s=10.0, total_s=100.0)
    early = strip.at(5.0).copy()
    late = strip.at(80.0).copy()

    assert not np.array_equal(early, late), "window did not scroll"
    assert early.shape == late.shape


class RecordingSink:
    """Captures what the loop writes, so tests need no encoder."""

    def __init__(self):
        self.frames = []
        self.closed = False
        self.discarded = False

    def write(self, f):
        self.frames.append(f.copy())

    def close(self):
        self.closed = True

    def discard_partial(self):
        self.discarded = True


def test_export_clip_writes_one_frame_per_source_frame(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource, export_clip
    from fish_analyzer.overlay_render import OverlaySettings

    video = make_video(tmp_path / "in.avi", n_frames=20)
    traj = np.full((20, 1, 2), 10.0)
    sink = RecordingSink()

    with ExportFrameSource(video, start_frame=2) as src:
        export_clip(src, sink, None, traj,
                    OverlaySettings(show_positions=True),
                    scale=1.0, start_frame=2, end_frame=7, step=1)

    assert len(sink.frames) == 6, "inclusive range 2..7 is six frames"
    assert sink.closed is True


def test_export_clip_stacks_the_strip_under_the_frame(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource, export_clip
    from fish_analyzer.overlay_render import OverlaySettings

    video = make_video(tmp_path / "in.avi", n_frames=10)
    traj = np.full((10, 1, 2), 10.0)
    sink = RecordingSink()

    class FakeStrip:
        height = 40

        def at(self, t):
            return np.zeros((40, 48, 3), dtype=np.uint8)

    with ExportFrameSource(video, start_frame=0) as src:
        export_clip(src, sink, FakeStrip(), traj, OverlaySettings(),
                    scale=1.0, start_frame=0, end_frame=2, step=1, fps=30.0)

    assert sink.frames[0].shape == (32 + 40, 48, 3), \
        "strip not stacked below the frame"


def test_export_clip_stops_when_cancelled(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource, export_clip
    from fish_analyzer.overlay_render import OverlaySettings

    video = make_video(tmp_path / "in.avi", n_frames=20)
    traj = np.full((20, 1, 2), 10.0)
    sink = RecordingSink()

    with ExportFrameSource(video, start_frame=0) as src:
        result = export_clip(src, sink, None, traj, OverlaySettings(),
                             scale=1.0, start_frame=0, end_frame=19, step=1,
                             should_cancel=lambda: len(sink.frames) >= 3)

    assert result.cancelled is True
    assert sink.discarded is True, "partial output not discarded on cancel"
    assert len(sink.frames) == 3


def test_export_clip_reports_progress(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource, export_clip
    from fish_analyzer.overlay_render import OverlaySettings

    video = make_video(tmp_path / "in.avi", n_frames=10)
    traj = np.full((10, 1, 2), 10.0)
    seen = []

    with ExportFrameSource(video, start_frame=0) as src:
        export_clip(src, RecordingSink(), None, traj, OverlaySettings(),
                    scale=1.0, start_frame=0, end_frame=4, step=1,
                    on_progress=lambda done, total: seen.append((done, total)))

    assert seen[0] == (1, 5)
    assert seen[-1] == (5, 5)


def test_export_clip_stops_cleanly_when_the_video_ends_early(tmp_path):
    """The trajectory file can be longer than the video it came from."""
    from fish_analyzer.media_export import ExportFrameSource, export_clip
    from fish_analyzer.overlay_render import OverlaySettings

    video = make_video(tmp_path / "in.avi", n_frames=5)
    traj = np.full((50, 1, 2), 10.0)
    sink = RecordingSink()

    with ExportFrameSource(video, start_frame=0) as src:
        result = export_clip(src, sink, None, traj, OverlaySettings(),
                             scale=1.0, start_frame=0, end_frame=49, step=1)

    assert result.frames_written == 5
    assert result.cancelled is False
    assert sink.closed is True


def test_export_clip_honours_a_frame_step(tmp_path):
    from fish_analyzer.media_export import ExportFrameSource, export_clip
    from fish_analyzer.overlay_render import OverlaySettings

    video = make_video(tmp_path / "in.avi", n_frames=20)
    traj = np.full((20, 1, 2), 10.0)
    sink = RecordingSink()

    with ExportFrameSource(video, start_frame=0) as src:
        export_clip(src, sink, None, traj, OverlaySettings(), scale=1.0,
                    start_frame=0, end_frame=9, step=2)

    # 0, 2, 4, 6, 8
    assert len(sink.frames) == 5
    reds = [int(f[0, 0, 0]) for f in sink.frames]
    assert reds == pytest.approx([0, 20, 40, 60, 80], abs=6)
