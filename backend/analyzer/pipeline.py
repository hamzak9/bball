"""
pipeline.py — Full analysis pipeline orchestrator

Takes one or more video file paths (keyed by angle: front/side/rear),
runs the full CV stack, and returns structured metrics.

Side view is preferred for arc angle (parabola fitting works best).
Front view is preferred for drift.
Either view works for balance and release timing.
"""

import time
import logging
from pathlib import Path
from typing import Optional

from .ball    import track_ball, smooth_trajectory
from .pose    import extract_poses, get_fps, get_video_info
from .shots   import detect_shots
from .metrics import aggregate_metrics

logger = logging.getLogger(__name__)


def run_analysis(
    video_paths: dict[str, str],  # {"front": "/tmp/...", "side": "/tmp/...", "rear": "/tmp/..."}
    max_frames: int = 1500,
) -> dict:
    """
    Full analysis pipeline.

    Priority logic:
      - Arc angle:  use side view if available, else front, else rear
      - Drift:      use front view if available, else skip
      - Balance:    use any view
      - Release:    use any view
      - Pocket:     use any view

    Returns a metrics dict ready to JSON-serialize and return to the frontend.
    """
    t0 = time.time()
    available = list(video_paths.keys())
    logger.info(f"Starting analysis. Available angles: {available}")

    # Pick the best video for arc + shot detection
    arc_angle = _pick_angle(available, preferred=["side", "front", "rear"])
    drift_angle = _pick_angle(available, preferred=["front", "rear", "side"])
    pose_angle = _pick_angle(available, preferred=["side", "front", "rear"])

    primary_path = video_paths[arc_angle]
    info = get_video_info(primary_path)
    fps  = info["fps"]

    logger.info(f"Video info: {info}")
    logger.info(f"Using {arc_angle} for arc, {drift_angle} for drift, {pose_angle} for pose")

    # ── 1. Ball tracking ───────────────────────────────────────────────────
    logger.info("Auto-detecting ball color and running tracker...")
    t1 = time.time()
    from .ball import detect_ball_color
    hsv_ranges = detect_ball_color(primary_path)
    raw_detections = track_ball(primary_path, max_frames=max_frames, hsv_ranges=hsv_ranges)
    detections = smooth_trajectory(raw_detections)
    logger.info(f"Ball tracker: {len(detections)} detections in {time.time()-t1:.1f}s")

    # ── 2. Pose estimation ─────────────────────────────────────────────────
    logger.info("Running pose estimator...")
    t2 = time.time()
    poses = extract_poses(video_paths[pose_angle], max_frames=max_frames)
    logger.info(f"Pose: {len(poses)} frames with detections in {time.time()-t2:.1f}s")

    # ── 3. Shot detection ──────────────────────────────────────────────────
    logger.info("Detecting shot events...")
    shots = detect_shots(detections, poses, fps=fps)
    logger.info(f"Found {len(shots)} shot(s)")

    if not shots:
        logger.warning("No shots detected — returning partial result")
        return _no_shots_result(available, info, fps, len(detections), len(poses))

    # ── 4. Metrics ─────────────────────────────────────────────────────────
    logger.info("Computing metrics...")
    metrics = aggregate_metrics(shots, poses, fps, info=info)

    # Add drift from front view if we have it separately
    if drift_angle != arc_angle and drift_angle in video_paths:
        logger.info(f"Running drift pass on {drift_angle} view...")
        front_dets = smooth_trajectory(track_ball(video_paths[drift_angle], max_frames=max_frames))
        front_poses = extract_poses(video_paths[drift_angle], max_frames=max_frames)
        front_shots = detect_shots(front_dets, front_poses, fps=fps)
        if front_shots:
            from .metrics import compute_drift, drift_score, drift_inches
            import numpy as np
            drift_vals = [compute_drift(s) for s in front_shots if compute_drift(s) is not None]
            if drift_vals:
                d_avg = float(np.mean(drift_vals))
                metrics["drift"]["score"]      = int(np.mean([drift_score(d) for d in drift_vals]))
                metrics["drift"]["avg_norm"]   = round(d_avg, 4)
                metrics["drift"]["avg_inches"] = round(drift_inches(d_avg), 1)
                metrics["drift"]["source"]     = drift_angle

    elapsed = round(time.time() - t0, 1)
    logger.info(f"Analysis complete in {elapsed}s. Overall score: {metrics['overall']}")

    return {
        "status":  "ok",
        "elapsed_sec": elapsed,
        "video_info": info,
        "angles_used": available,
        "ball_detections_total": len(detections),
        "pose_frames_total": len(poses),
        **metrics,
    }


def _pick_angle(available: list[str], preferred: list[str]) -> str:
    for a in preferred:
        if a in available:
            return a
    return available[0]


def _no_shots_result(available, info, fps, n_ball, n_pose) -> dict:
    if n_ball == 0:
        reason = "Ball not detected in any frame. Check that the ball is clearly visible and well-lit."
        tip = "Try filming in better light. The ball must be visible (not blurry or hidden) for at least a few frames."
    elif n_ball < 10:
        reason = f"Ball only detected in {n_ball} frame(s) — not enough to reconstruct a trajectory."
        tip = "Ensure the ball is in frame for the full shot arc, not just the release moment."
    else:
        reason = f"Ball detected in {n_ball} frames but no upward movement pattern found."
        tip = "Make sure the clip includes the ball actually being shot upward, not just held or dribbled."

    return {
        "status":  "no_shots_detected",
        "message": reason,
        "tip":     tip,
        "debug": {
            "angles_available": available,
            "ball_detections":  n_ball,
            "pose_frames":      n_pose,
            "fps":              fps,
            "video_info":       info,
        },
        "n_shots": 0,
        "overall": None,
    }
