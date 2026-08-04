"""Synthetic trajectories with analytically known ground truth.

Every builder returns a ``(n_frames, n_fish, 2)`` array in **pixel** coordinates
using the video convention idtracker.ai writes: origin top-left, ``y``
increasing *downward*. That matters for the turn-direction tests — the package
flips Y before computing headings, so "which way is the fish turning" is only
well defined once you fix the input convention.

``make_file`` wraps an array in the ``LoadedTrajectoryFile`` the analysis layer
expects, without touching the disk. Nothing here needs real session data.
"""
import numpy as np

from fish_analyzer.data_structures import (
    CalibrationSettings, IdTrackerMetadata, LoadedTrajectoryFile)

#: Defaults shared by the tests so expected values can be written as literals.
BODY_LENGTH_PX = 50.0
FPS = 30.0
VIDEO_SIZE = 1000


def make_file(traj, body_length=BODY_LENGTH_PX, fps=FPS, width=VIDEO_SIZE,
              height=VIDEO_SIZE, nickname="synthetic", calibration=None):
    """Wrap a pixel-space trajectory array in a LoadedTrajectoryFile."""
    traj = np.asarray(traj, dtype=float)
    n_fish = traj.shape[1]
    metadata = IdTrackerMetadata(
        video_width=width, video_height=height, frames_per_second=fps,
        body_length=body_length, estimated_accuracy=0.99,
        fraction_identified=0.99, n_individuals=n_fish,
        identity_labels=[str(i + 1) for i in range(n_fish)],
        idtracker_version="6.0.8", raw_metadata={})
    if calibration is None:
        calibration = CalibrationSettings.from_body_lengths(body_length, fps)
    return LoadedTrajectoryFile(
        nickname=nickname, file_path=None, metadata=metadata,
        trajectories=traj, calibration=calibration)


def straight_line(n=600, step_px=2.0, x0=100.0, y0=500.0):
    """Constant-velocity swimmer along +x. Zero turning, straightness 1.0."""
    t = np.arange(n)
    return np.stack([x0 + step_px * t, np.full(n, y0)], axis=1)[:, None, :]


def circler(n=600, radius_px=100.0, omega=0.05, cx=500.0, cy=500.0,
            clockwise_on_screen=True):
    """Constant-rate circler at a known angular rate.

    ``clockwise_on_screen=True`` traces the circle clockwise *as a viewer sees
    the video*. With the camera above the tank looking down, that is a fish
    turning to its own **right**.
    """
    t = np.arange(n)
    s = 1.0 if clockwise_on_screen else -1.0
    x = cx + radius_px * np.cos(s * omega * t)
    y = cy + radius_px * np.sin(s * omega * t)
    return np.stack([x, y], axis=1)[:, None, :]


def stationary(n=600, jitter_px=0.0, x0=500.0, y0=500.0, seed=0):
    """A fish that does not move, optionally with sub-pixel tracking jitter."""
    xy = np.full((n, 2), [x0, y0], dtype=float)
    if jitter_px:
        xy = xy + np.random.default_rng(seed).normal(0, jitter_px, size=(n, 2))
    return xy[:, None, :]


def jittered_straight_line(n=600, step_px=2.0, jitter_px=0.1, seed=7):
    """A straight swimmer plus Gaussian tracking noise. Truth is still 0 turn."""
    traj = straight_line(n=n, step_px=step_px).copy()
    traj[:, 0, :] += np.random.default_rng(seed).normal(0, jitter_px, size=(n, 2))
    return traj


def discrete_bouts(n_bouts=10, bout_frames=5, pause_frames=25, step_px=6.0,
                   x0=100.0, y0=500.0):
    """Exactly ``n_bouts`` straight darts separated by exact pauses."""
    xs, ys = [x0], [y0]
    for _ in range(n_bouts):
        for _ in range(bout_frames):
            xs.append(xs[-1] + step_px)
            ys.append(y0)
        for _ in range(pause_frames):
            xs.append(xs[-1])
            ys.append(y0)
    return np.stack([np.array(xs), np.array(ys)], axis=1)[:, None, :]


def single_swim_event(n=200, accel_frames=10, plateau_frames=60,
                      peak_step_px=5.0, x0=100.0, y0=500.0):
    """One swim event: ramp up, hold, ramp down. A single burst by any reading.

    Returns ``(traj, high_speed_frames)`` where ``high_speed_frames`` is the
    number of frames the fish is at or accelerating toward top speed — the span
    a "burst duration" should describe.
    """
    speed = np.zeros(n)
    a, p = accel_frames, plateau_frames
    speed[20:20 + a] = np.linspace(0, peak_step_px, a)
    speed[20 + a:20 + a + p] = peak_step_px
    speed[20 + a + p:20 + 2 * a + p] = np.linspace(peak_step_px, 0, a)
    x = x0 + np.cumsum(speed)
    traj = np.stack([x, np.full(n, y0)], axis=1)[:, None, :]
    return traj, a + p + a


def three_fish_fixed_geometry(n=300, sep_px=150.0, x=300.0, y=500.0):
    """Three motionless fish in a right triangle with known side lengths.

    A at (x, y), B at (x + sep, y), C at (x, y - sep) in pixel space, so
    ``AB = AC = sep`` and ``BC = sep * sqrt(2)``.
    """
    a = np.stack([np.full(n, x), np.full(n, y)], axis=1)
    b = np.stack([np.full(n, x + sep_px), np.full(n, y)], axis=1)
    c = np.stack([np.full(n, x), np.full(n, y - sep_px)], axis=1)
    return np.stack([a, b, c], axis=1)


def with_dropout(traj, start, stop, fish=0):
    """Blank a contiguous run of frames, the way a tracking gap looks."""
    out = traj.copy()
    out[start:stop, fish, :] = np.nan
    return out


def with_scattered_dropout(traj, fraction=0.05, fish=0, seed=1):
    """Blank isolated single frames, the other shape a tracking gap takes."""
    out = traj.copy()
    n = out.shape[0]
    idx = np.random.default_rng(seed).choice(
        np.arange(1, n - 1), size=int(fraction * n), replace=False)
    out[idx, fish, :] = np.nan
    return out
