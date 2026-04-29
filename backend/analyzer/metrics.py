"""
metrics.py — Compute the 5 shooting metrics from real shot data

All scores are 0–100. Higher is better.

Metrics:
  arc_angle    — launch angle of ball trajectory (target: 42–50°)
  drift        — lateral deviation from shot line (target: < 1 inch equiv.)
  release      — consistency of release timing (wrist velocity peak alignment)
  balance      — ankle symmetry at landing
  pocket       — consistency of wrist height at shot pocket setup
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import variation
from typing import Optional
from .shots import ShotEvent
from .pose import PoseFrame, angle_between


# ─── Arc angle ──────────────────────────────────────────────────────────────

def compute_arc_angle(shot: ShotEvent) -> Optional[float]:
    """
    Fit a parabola to the ball trajectory and return the launch angle in degrees.
    Uses the normalized (cx, cy) coordinates — works for side-view footage.
    """
    cx = np.array(shot.ball_cx)
    cy = np.array(shot.ball_cy)

    if len(cx) < 5:
        return None

    # In image coords: cy decreases as ball rises (0=top)
    # Flip cy so that "up" is positive
    cy_up = -cy

    try:
        # Fit quadratic: y = a*x^2 + b*x + c
        coeffs = np.polyfit(cx, cy_up, 2)
        a, b, _ = coeffs

        # Derivative at the first point gives launch angle
        x0 = cx[0]
        dy_dx = 2 * a * x0 + b

        # Convert slope to angle
        # Scale factor: aspect ratio of frame (cy is normalized by height, cx by width)
        # For a 16:9 frame: 1 unit cx = 16/9 * 1 unit cy in real space
        # We estimate this from the trajectory span
        cx_span = cx[-1] - cx[0]
        cy_span = cy_up[-1] - cy_up[0]

        angle_deg = float(np.degrees(np.arctan(abs(dy_dx))))

        # Sanity-check range
        if not (15 < angle_deg < 75):
            return None
        return angle_deg

    except (np.linalg.LinAlgError, ValueError):
        return None


def arc_score(angle_deg: float) -> int:
    """Score 0–100 for arc angle. Optimal window 42–50°."""
    if angle_deg < 30:
        return max(10, int(30 + angle_deg))
    if angle_deg <= 42:
        return int(50 + (angle_deg - 30) * 4)  # 50→98 over 30–42°
    if angle_deg <= 50:
        return 99
    if angle_deg <= 58:
        return int(99 - (angle_deg - 50) * 5)
    return max(20, int(60 - angle_deg))


# ─── Drift ──────────────────────────────────────────────────────────────────

def compute_drift(shot: ShotEvent) -> Optional[float]:
    """
    Measure lateral drift: horizontal deviation of ball from the straight line
    between release and landing position.
    Returns drift in normalized units (0..1 of frame width).
    Only meaningful on front-view footage.
    """
    if len(shot.ball_cx) < 4:
        return None

    cx = np.array(shot.ball_cx)
    cy = np.array(shot.ball_cy)

    # Ideal straight line from release to landing (in cx)
    x_release = cx[0]
    x_land    = cx[-1]
    x_ideal   = np.linspace(x_release, x_land, len(cx))

    # RMS deviation from straight line
    deviations = cx - x_ideal
    rms_drift = float(np.sqrt(np.mean(deviations**2)))

    return rms_drift


def drift_score(drift_normalized: float) -> int:
    """Score 0–100 for drift. < 0.01 = perfect, > 0.08 = poor."""
    if drift_normalized < 0.01:
        return 98
    if drift_normalized < 0.03:
        return int(98 - (drift_normalized - 0.01) / 0.02 * 30)
    if drift_normalized < 0.06:
        return int(68 - (drift_normalized - 0.03) / 0.03 * 30)
    return max(15, int(38 - drift_normalized * 200))


def drift_inches(drift_normalized: float, frame_width_px: int = 1920) -> float:
    """Approximate drift in inches. Assumes shooter is ~10 ft from camera."""
    # 1 frame width at 10ft distance ≈ 8 feet = 96 inches
    return drift_normalized * 96.0


# ─── Release timing ─────────────────────────────────────────────────────────

def compute_release_timing(
    shots: list[ShotEvent],
    poses_by_frame: dict[int, PoseFrame],
    fps: float,
) -> dict:
    """
    Measure when the ball leaves the hand relative to peak jump height.

    Ideal: ball releases at or just before peak jump (wrist highest point).
    Late release: wrist is already descending when ball leaves hand.
    Early release: wrist still rising significantly at release.

    Returns dict with per-shot timing offsets and consistency score.
    """
    offsets = []  # frames from peak jump to ball release (negative = early, positive = late)

    for shot in shots:
        if shot.release_pose is None:
            continue

        # Find wrist height around release — look at a window of poses
        release_f = shot.release_frame
        wrist_heights = []
        wrist_frames  = []

        for df in range(-12, 6):
            f = release_f + df
            if f in poses_by_frame:
                pose = poses_by_frame[f]
                wrist = pose.get_xy("right_wrist") or pose.get_xy("left_wrist")
                if wrist:
                    wrist_heights.append(wrist[1])  # y normalized, 0=top
                    wrist_frames.append(df)

        if len(wrist_heights) < 4:
            continue

        # Peak jump = minimum wrist_y (highest in image)
        peak_i = int(np.argmin(wrist_heights))
        peak_offset = wrist_frames[peak_i]  # 0 = at release frame

        # Positive peak_offset means peak was BEFORE release (late release)
        # Negative = peak AFTER release (early release)
        offsets.append(-peak_offset)  # flip: positive = early, negative = late

    if not offsets:
        return {"offsets": [], "score": 50, "mean_offset_frames": 0.0, "consistency": 0.0}

    mean_offset = float(np.mean(offsets))
    std_offset  = float(np.std(offsets))

    # Score: want offset close to 0 (release at peak), low std
    timing_penalty = abs(mean_offset) * 3          # 0–30 pts
    consistency_penalty = std_offset * 8           # 0–30 pts
    score = max(10, min(99, int(99 - timing_penalty - consistency_penalty)))

    return {
        "offsets": offsets,
        "score": score,
        "mean_offset_frames": round(mean_offset, 2),
        "consistency": round(std_offset, 2),
    }


# ─── Landing balance ────────────────────────────────────────────────────────

def compute_balance(shot: ShotEvent) -> Optional[float]:
    """
    Measure landing balance from ankle landmark symmetry.
    Returns asymmetry value 0..1 (0 = perfect symmetry).
    Only works if pose is available at landing.
    """
    if shot.land_pose is None:
        return None

    left_ankle  = shot.land_pose.get_xy("left_ankle")
    right_ankle = shot.land_pose.get_xy("right_ankle")

    if left_ankle is None or right_ankle is None:
        return None

    # Horizontal distance between ankles (normalized by frame width)
    ankle_width = abs(right_ankle[0] - left_ankle[0])

    # Height asymmetry: both feet should land at same y
    ankle_height_diff = abs(left_ankle[1] - right_ankle[1])

    # Hip position: should be centered between ankles
    left_hip  = shot.land_pose.get_xy("left_hip")
    right_hip = shot.land_pose.get_xy("right_hip")
    hip_asymmetry = 0.0
    if left_hip and right_hip:
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        ankle_center_x = (left_ankle[0] + right_ankle[0]) / 2
        hip_asymmetry = abs(hip_center_x - ankle_center_x)

    # Combined asymmetry score
    asymmetry = ankle_height_diff * 0.5 + hip_asymmetry * 0.5
    return float(asymmetry)


def balance_score(asymmetry: float) -> int:
    """Score 0–100 for balance. Lower asymmetry = better."""
    if asymmetry < 0.01:
        return 97
    if asymmetry < 0.03:
        return int(97 - (asymmetry - 0.01) / 0.02 * 25)
    if asymmetry < 0.07:
        return int(72 - (asymmetry - 0.03) / 0.04 * 35)
    return max(15, int(37 - asymmetry * 200))


# ─── Shot pocket ────────────────────────────────────────────────────────────

def compute_pocket_consistency(shots: list[ShotEvent]) -> dict:
    """
    Measure shot pocket consistency: how much does the wrist height
    at the start of each shot vary?

    Returns consistency score and per-shot pocket heights.
    """
    pocket_heights = []

    for shot in shots:
        if shot.release_pose is None:
            continue
        wrist = shot.release_pose.get_xy("right_wrist") or shot.release_pose.get_xy("left_wrist")
        shoulder = shot.release_pose.get_xy("right_shoulder") or shot.release_pose.get_xy("left_shoulder")

        if wrist is None or shoulder is None:
            continue

        # Pocket height: wrist y relative to shoulder y (normalized)
        # Negative means wrist is above shoulder
        pocket_h = wrist[1] - shoulder[1]
        pocket_heights.append(pocket_h)

    if len(pocket_heights) < 2:
        return {"heights": pocket_heights, "score": 60, "std": 0.0}

    std = float(np.std(pocket_heights))
    score = max(10, min(99, int(99 - std * 400)))

    return {
        "heights": pocket_heights,
        "score": score,
        "std": round(std, 4),
    }


# ─── Aggregate across shots ──────────────────────────────────────────────────

def aggregate_metrics(shots: list[ShotEvent], poses: list[PoseFrame], fps: float) -> dict:
    """
    Compute all 5 metrics across all detected shots.
    Returns the final metrics dict that the API returns to the frontend.
    """
    poses_by_frame = {p.frame: p for p in poses}

    arc_angles  = []
    drift_vals  = []
    balance_vals = []

    per_shot = []

    for i, shot in enumerate(shots):
        arc   = compute_arc_angle(shot)
        drift = compute_drift(shot)
        bal   = compute_balance(shot)

        if arc   is not None: arc_angles.append(arc)
        if drift is not None: drift_vals.append(drift)
        if bal   is not None: balance_vals.append(bal)

        per_shot.append({
            "shot_num":      i + 1,
            "release_frame": shot.release_frame,
            "peak_frame":    shot.peak_frame,
            "land_frame":    shot.land_frame,
            "arc_deg":       round(arc, 1)  if arc   is not None else None,
            "drift_norm":    round(drift, 4) if drift is not None else None,
            "balance_asym":  round(bal, 4)   if bal   is not None else None,
        })

    rel_timing = compute_release_timing(shots, poses_by_frame, fps)
    pocket     = compute_pocket_consistency(shots)

    # --- Scores ---
    n_shots = len(shots)

    if arc_angles:
        arc_avg = float(np.mean(arc_angles))
        arc_std = float(np.std(arc_angles))
        arc_s   = int(np.mean([arc_score(a) for a in arc_angles]))
        # penalize inconsistency
        arc_s   = max(10, arc_s - int(arc_std * 3))
    else:
        arc_avg = 45.0
        arc_std = 5.0
        arc_s   = 50

    if drift_vals:
        drift_avg = float(np.mean(drift_vals))
        drift_s   = int(np.mean([drift_score(d) for d in drift_vals]))
        drift_in  = round(drift_inches(drift_avg), 1)
    else:
        drift_avg = 0.03
        drift_s   = 65
        drift_in  = 0.0

    if balance_vals:
        bal_avg = float(np.mean(balance_vals))
        bal_s   = int(np.mean([balance_score(b) for b in balance_vals]))
    else:
        bal_avg = 0.03
        bal_s   = 65

    rel_s    = rel_timing["score"]
    pocket_s = pocket["score"]

    overall = int(
        arc_s    * 0.30 +
        drift_s  * 0.20 +
        rel_s    * 0.25 +
        bal_s    * 0.15 +
        pocket_s * 0.10
    )

    return {
        "n_shots":    n_shots,
        "overall":    overall,
        "arc": {
            "score":  arc_s,
            "avg_deg": round(arc_avg, 1),
            "std_deg": round(arc_std, 1),
            "per_shot": [round(a, 1) for a in arc_angles],
        },
        "drift": {
            "score":   drift_s,
            "avg_norm": round(drift_avg, 4),
            "avg_inches": drift_in,
        },
        "release": {
            "score":              rel_s,
            "mean_offset_frames": rel_timing["mean_offset_frames"],
            "consistency":        rel_timing["consistency"],
        },
        "balance": {
            "score":   bal_s,
            "avg_asym": round(bal_avg, 4),
        },
        "pocket": {
            "score": pocket_s,
            "std":   pocket["std"],
        },
        "per_shot": per_shot,
    }
