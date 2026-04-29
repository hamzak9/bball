"""
shots.py — Shot event detection

A shot is defined as:
  1. Ball moves upward continuously for >= MIN_RISE_FRAMES frames
  2. Ball reaches a peak (cy stops decreasing, remember y=0 is top of frame)
  3. Ball then moves downward toward the basket

We detect shots by finding local minima in ball cy (peaks in real space)
and working backward/forward to find the release frame and landing frame.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .ball import BallDetection
from .pose import PoseFrame


MIN_RISE_FRAMES  = 4   # ball must rise at least this many frames
MIN_SHOT_FRAMES  = 8   # minimum total shot arc length in frames
PEAK_WINDOW      = 5   # frames around peak to check


@dataclass
class ShotEvent:
    release_frame: int    # frame ball leaves the hand
    peak_frame: int       # frame ball is at highest point
    land_frame: int       # frame ball reaches rim / comes down
    ball_frames: list[int]            # all frames with ball detection in this shot
    ball_cx: list[float]              # x positions
    ball_cy: list[float]              # y positions (normalized, 0=top)
    release_pose: Optional[PoseFrame] # pose at release
    land_pose: Optional[PoseFrame]    # pose at landing


def detect_shots(
    ball_detections: list[BallDetection],
    poses: list[PoseFrame],
    fps: float = 30.0,
    min_shots: int = 1,
) -> list[ShotEvent]:
    """
    Find all shot events in a video from ball trajectory + pose data.
    """
    if len(ball_detections) < MIN_SHOT_FRAMES:
        return []

    # Index poses by frame number for fast lookup
    pose_by_frame: dict[int, PoseFrame] = {p.frame: p for p in poses}

    # Build arrays of frame, cx, cy from detections
    frames = np.array([d.frame for d in ball_detections])
    cx_arr = np.array([d.cx     for d in ball_detections])
    cy_arr = np.array([d.cy     for d in ball_detections])  # 0 = top

    # Find peaks: local minima in cy (ball at its highest = smallest y)
    peak_indices = _find_peaks(cy_arr, min_prominence=0.05, min_distance=int(fps * 0.4))

    if not peak_indices:
        # Try with looser threshold
        peak_indices = _find_peaks(cy_arr, min_prominence=0.02, min_distance=int(fps * 0.3))

    shots = []
    for peak_i in peak_indices:
        # Find release: scan backward from peak, ball moving up (cy decreasing)
        release_i = peak_i
        for i in range(peak_i - 1, max(0, peak_i - int(fps * 1.5)), -1):
            if cy_arr[i] > cy_arr[i + 1] + 0.005:  # ball was lower (higher y)
                release_i = i
            else:
                break

        # Find landing: scan forward from peak, ball moving down (cy increasing)
        land_i = peak_i
        for i in range(peak_i + 1, min(len(cy_arr), peak_i + int(fps * 1.5))):
            if cy_arr[i] > cy_arr[i - 1] + 0.003:
                land_i = i
            else:
                break

        # Require minimum rise
        rise_frames = peak_i - release_i
        if rise_frames < MIN_RISE_FRAMES:
            continue

        # Get pose frames near release and landing
        release_frame = int(frames[release_i])
        peak_frame    = int(frames[peak_i])
        land_frame    = int(frames[land_i])

        release_pose = _nearest_pose(pose_by_frame, release_frame, window=4)
        land_pose    = _nearest_pose(pose_by_frame, land_frame, window=6)

        shot_range = slice(release_i, land_i + 1)
        shots.append(ShotEvent(
            release_frame = release_frame,
            peak_frame    = peak_frame,
            land_frame    = land_frame,
            ball_frames   = frames[shot_range].tolist(),
            ball_cx       = cx_arr[shot_range].tolist(),
            ball_cy       = cy_arr[shot_range].tolist(),
            release_pose  = release_pose,
            land_pose     = land_pose,
        ))

    return shots


def _find_peaks(arr: np.ndarray, min_prominence: float, min_distance: int) -> list[int]:
    """Find local minima (peaks in real space) in cy array."""
    peaks = []
    n = len(arr)
    for i in range(1, n - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            # Check prominence: how much lower than surroundings
            left_max  = arr[max(0, i - min_distance):i].max() if i > 0 else arr[i]
            right_max = arr[i + 1:min(n, i + min_distance + 1)].max() if i < n - 1 else arr[i]
            prominence = min(left_max, right_max) - arr[i]
            if prominence >= min_prominence:
                # Enforce min distance from last peak
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
    return peaks


def _nearest_pose(
    pose_by_frame: dict[int, PoseFrame],
    target_frame: int,
    window: int = 4,
) -> Optional[PoseFrame]:
    """Return the closest PoseFrame within ±window frames of target."""
    for offset in range(window + 1):
        for delta in ([0, -offset, offset] if offset > 0 else [0]):
            f = target_frame + delta
            if f in pose_by_frame:
                return pose_by_frame[f]
    return None
