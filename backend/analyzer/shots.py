"""
shots.py — Shot event detection

Handles three real-world scenarios:
  1. Full arc visible (rise + peak + fall) — ideal
  2. Only rise visible (video cuts off at peak or ball leaves frame)
  3. Only partial rise visible (short clip, single shot attempt)

Detection strategy:
  - Primary: find peaks (local minima in cy) → full arcs
  - Fallback 1: find sustained upward movement segments → partial arcs
  - Fallback 2: if only one continuous upward segment exists, treat it as one shot
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

from .ball import BallDetection
from .pose import PoseFrame

logger = logging.getLogger(__name__)

MIN_RISE_FRAMES = 3    # lowered: ball must rise at least this many frames
MIN_SHOT_FRAMES = 5    # lowered: minimum frames to count as a shot


@dataclass
class ShotEvent:
    release_frame: int
    peak_frame: int
    land_frame: int
    ball_frames: list[int]
    ball_cx: list[float]
    ball_cy: list[float]
    release_pose: Optional[PoseFrame]
    land_pose: Optional[PoseFrame]
    arc_complete: bool   # True if we saw both rise AND fall


def detect_shots(
    ball_detections: list[BallDetection],
    poses: list[PoseFrame],
    fps: float = 30.0,
) -> list[ShotEvent]:
    """
    Find all shot events. Tries full-arc detection first, then
    falls back to partial-arc detection for incomplete videos.
    """
    if len(ball_detections) < MIN_SHOT_FRAMES:
        logger.warning(f"Only {len(ball_detections)} ball detections — too few to detect shots")
        return []

    pose_by_frame = {p.frame: p for p in poses}
    frames = np.array([d.frame for d in ball_detections])
    cx_arr = np.array([d.cx     for d in ball_detections])
    cy_arr = np.array([d.cy     for d in ball_detections])

    logger.info(f"Shot detection: {len(ball_detections)} ball detections, cy range [{cy_arr.min():.3f}, {cy_arr.max():.3f}]")

    # ── Strategy 1: Full arc (peak detection) ────────────────────────────────
    shots = _detect_full_arcs(frames, cx_arr, cy_arr, pose_by_frame, fps)

    if shots:
        logger.info(f"Full arc detection: found {len(shots)} shot(s)")
        return shots

    # ── Strategy 2: Partial arc (upward movement segments) ───────────────────
    logger.info("No full arcs found — trying partial arc detection")
    shots = _detect_partial_arcs(frames, cx_arr, cy_arr, pose_by_frame, fps)

    if shots:
        logger.info(f"Partial arc detection: found {len(shots)} shot(s)")
        return shots

    # ── Strategy 3: Treat entire detection run as one shot ────────────────────
    # Useful for very short clips or extreme partial footage
    logger.info("Trying whole-trajectory fallback")
    shots = _detect_whole_trajectory(frames, cx_arr, cy_arr, pose_by_frame)

    if shots:
        logger.info(f"Whole-trajectory fallback: using {len(shots)} segment(s)")

    return shots


def _detect_full_arcs(frames, cx_arr, cy_arr, pose_by_frame, fps) -> list[ShotEvent]:
    """Detect complete arcs with visible rise + peak + fall."""
    shots = []

    # Try progressively looser thresholds
    for prominence, min_dist_mult in [(0.05, 0.4), (0.03, 0.3), (0.015, 0.2)]:
        min_dist = max(3, int(fps * min_dist_mult))
        peaks = _find_peaks(cy_arr, min_prominence=prominence, min_distance=min_dist)
        if peaks:
            logger.info(f"Found {len(peaks)} peak(s) with prominence={prominence}")
            break

    for peak_i in peaks:
        release_i = _find_release(cy_arr, peak_i, fps)
        land_i    = _find_landing(cy_arr, peak_i, fps)

        rise_frames = peak_i - release_i
        if rise_frames < MIN_RISE_FRAMES:
            continue

        arc_complete = (land_i > peak_i + 2)

        shots.append(_make_shot(
            frames, cx_arr, cy_arr, pose_by_frame,
            release_i, peak_i, land_i, arc_complete
        ))

    return shots


def _detect_partial_arcs(frames, cx_arr, cy_arr, pose_by_frame, fps) -> list[ShotEvent]:
    """
    Detect shots from upward-only movement.
    Looks for segments where the ball moves consistently upward (cy decreasing)
    for at least MIN_RISE_FRAMES frames.
    """
    shots = []
    n = len(cy_arr)

    # Compute frame-to-frame cy delta (negative = ball moving up)
    dy = np.diff(cy_arr)

    # Find contiguous upward-movement segments
    in_rise = False
    rise_start = 0

    for i in range(len(dy)):
        moving_up = dy[i] < 0.005  # allow tiny noise

        if moving_up and not in_rise:
            in_rise = True
            rise_start = i
        elif not moving_up and in_rise:
            rise_len = i - rise_start
            if rise_len >= MIN_RISE_FRAMES:
                peak_i    = i  # last upward frame
                release_i = rise_start
                land_i    = min(n-1, i + max(2, int(fps * 0.5)))

                shots.append(_make_shot(
                    frames, cx_arr, cy_arr, pose_by_frame,
                    release_i, peak_i, land_i, arc_complete=False
                ))
            in_rise = False

    # Handle case where video ends while ball is still rising
    if in_rise:
        rise_len = len(dy) - rise_start
        if rise_len >= MIN_RISE_FRAMES:
            peak_i    = n - 1
            release_i = rise_start
            land_i    = n - 1
            shots.append(_make_shot(
                frames, cx_arr, cy_arr, pose_by_frame,
                release_i, peak_i, land_i, arc_complete=False
            ))

    return shots


def _detect_whole_trajectory(frames, cx_arr, cy_arr, pose_by_frame) -> list[ShotEvent]:
    """
    Last resort: if the ball moved at all (any vertical range > 0.05),
    treat the whole detection window as one shot attempt.
    """
    cy_range = cy_arr.max() - cy_arr.min()
    if cy_range < 0.04:
        logger.warning(f"Ball vertical range only {cy_range:.3f} — may not be a shot")

    n = len(cy_arr)
    # Release = first frame, peak = lowest cy, land = last frame
    peak_i    = int(np.argmin(cy_arr))
    release_i = 0
    land_i    = n - 1

    return [_make_shot(
        frames, cx_arr, cy_arr, pose_by_frame,
        release_i, peak_i, land_i, arc_complete=False
    )]


def _make_shot(frames, cx_arr, cy_arr, pose_by_frame,
               release_i, peak_i, land_i, arc_complete) -> ShotEvent:
    release_frame = int(frames[release_i])
    peak_frame    = int(frames[peak_i])
    land_frame    = int(frames[land_i])
    shot_range    = slice(release_i, land_i + 1)

    return ShotEvent(
        release_frame = release_frame,
        peak_frame    = peak_frame,
        land_frame    = land_frame,
        ball_frames   = frames[shot_range].tolist(),
        ball_cx       = cx_arr[shot_range].tolist(),
        ball_cy       = cy_arr[shot_range].tolist(),
        release_pose  = _nearest_pose(pose_by_frame, release_frame, window=5),
        land_pose     = _nearest_pose(pose_by_frame, land_frame, window=8),
        arc_complete  = arc_complete,
    )


def _find_release(cy_arr, peak_i, fps) -> int:
    release_i = peak_i
    for i in range(peak_i - 1, max(0, peak_i - int(fps * 2.0)), -1):
        if cy_arr[i] > cy_arr[i + 1] + 0.004:
            release_i = i
        else:
            break
    return release_i


def _find_landing(cy_arr, peak_i, fps) -> int:
    land_i = peak_i
    n = len(cy_arr)
    for i in range(peak_i + 1, min(n, peak_i + int(fps * 2.0))):
        if cy_arr[i] > cy_arr[i - 1] + 0.003:
            land_i = i
        else:
            break
    return land_i


def _find_peaks(arr: np.ndarray, min_prominence: float, min_distance: int) -> list[int]:
    peaks = []
    n = len(arr)
    for i in range(1, n - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            left_max  = arr[max(0, i - min_distance):i].max() if i > 0 else arr[i]
            right_max = arr[i+1:min(n, i + min_distance + 1)].max() if i < n-1 else arr[i]
            prominence = min(left_max, right_max) - arr[i]
            if prominence >= min_prominence:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
    return peaks


def _nearest_pose(pose_by_frame, target_frame, window=5) -> Optional[PoseFrame]:
    for offset in range(window + 1):
        for delta in ([0] if offset == 0 else [-offset, offset]):
            f = target_frame + delta
            if f in pose_by_frame:
                return pose_by_frame[f]
    return None
