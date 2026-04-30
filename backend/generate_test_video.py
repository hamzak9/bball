"""
generate_test_video.py — Synthetic basketball shot video for CV pipeline testing

Generates a side-view video with:
  - Exact known launch angles (ground truth)
  - Realistic parabolic physics
  - Blue ball (to test color auto-detection)
  - Stick-figure shooter + basket
  - 5 shots at slightly varied angles around a target

Usage:
    python3 generate_test_video.py
    
Output:
    test_shot_45deg.mp4   — 5 shots, target 45°, blue ball
    test_shot_48deg.mp4   — 5 shots, target 48°, blue ball  
    test_shot_30deg.mp4   — 5 shots, target 30° (flat, bad form), blue ball
    ground_truth.json     — exact per-shot angles for each video
"""

import cv2
import numpy as np
import json
import math
from pathlib import Path

# ── Video settings ────────────────────────────────────────────────────────────
W, H   = 1280, 720
FPS    = 60
CODEC  = cv2.VideoWriter_fourcc(*'mp4v')

# ── Court colors (dark gym aesthetic) ────────────────────────────────────────
BG_COLOR      = (18, 20, 25)       # near-black background
FLOOR_COLOR   = (45, 80, 120)      # dark wood floor
LINE_COLOR    = (80, 120, 160)     # court line
BASKET_COLOR  = (30, 80, 220)      # red rim (BGR)
BOARD_COLOR   = (220, 220, 230)    # backboard
SHOOTER_COLOR = (200, 200, 210)    # stick figure
BALL_COLOR    = (200, 80, 20)      # blue ball (BGR: B=200, G=80, R=20)
BALL_SEAM     = (160, 50, 10)


# ── Scene layout (normalized → pixel coords) ─────────────────────────────────
FLOOR_Y   = int(H * 0.82)   # floor line y
SHOOTER_X = int(W * 0.15)   # shooter x position
BASKET_X  = int(W * 0.82)   # basket x position
BASKET_Y  = int(H * 0.28)   # basket height (rim)
BALL_R    = 18               # ball radius pixels


def world_to_px(x_m, y_m, origin_x, origin_y, scale=420):
    """Convert real-world meters to pixel coords. y inverted (up = negative screen y)."""
    px = int(origin_x + x_m * scale)
    py = int(origin_y - y_m * scale)
    return px, py


def draw_background(frame):
    frame[:] = BG_COLOR

    # Floor
    cv2.rectangle(frame, (0, FLOOR_Y), (W, H), (30, 55, 85), -1)
    cv2.line(frame, (0, FLOOR_Y), (W, FLOOR_Y), LINE_COLOR, 2)

    # Subtle court texture lines
    for x in range(0, W, 80):
        cv2.line(frame, (x, FLOOR_Y), (x, H), (35, 60, 90), 1)

    # Distance markers on floor
    for i, dist in enumerate([3, 4, 5, 6]):
        mx = SHOOTER_X + int(dist / 7.0 * (BASKET_X - SHOOTER_X))
        cv2.line(frame, (mx, FLOOR_Y - 4), (mx, FLOOR_Y + 4), LINE_COLOR, 1)
        cv2.putText(frame, f"{dist}m", (mx - 8, FLOOR_Y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, LINE_COLOR, 1)


def draw_basket(frame):
    # Backboard
    bx = BASKET_X + 30
    cv2.rectangle(frame, (bx, BASKET_Y - 40), (bx + 8, BASKET_Y + 25),
                  BOARD_COLOR, -1)
    # Rim
    cv2.line(frame, (BASKET_X - 12, BASKET_Y), (BASKET_X + 12, BASKET_Y),
             BASKET_COLOR, 4)
    # Net (simple lines)
    for nx in range(BASKET_X - 10, BASKET_X + 12, 5):
        cv2.line(frame, (nx, BASKET_Y), (nx + 2, BASKET_Y + 28),
                 (80, 80, 80), 1)
    cv2.line(frame, (BASKET_X - 10, BASKET_Y + 28), (BASKET_X + 12, BASKET_Y + 28),
             (70, 70, 70), 1)


def draw_shooter(frame, phase):
    """
    Draw a stick-figure shooter.
    phase: 0.0 = ready/crouch, 0.5 = release, 1.0 = follow-through
    """
    bx, by = SHOOTER_X, FLOOR_Y

    # Crouch on wind-up, extend on release
    crouch = max(0, 1.0 - phase * 2) * 12
    hip_y  = by - int(H * 0.14) + int(crouch)
    head_y = by - int(H * 0.36) + int(crouch * 0.5)

    # Legs
    cv2.line(frame, (bx, hip_y), (bx - 12, by), SHOOTER_COLOR, 3)
    cv2.line(frame, (bx, hip_y), (bx + 8,  by), SHOOTER_COLOR, 3)
    # Torso
    cv2.line(frame, (bx, hip_y), (bx, head_y + 18), SHOOTER_COLOR, 3)
    # Head
    cv2.circle(frame, (bx, head_y), 12, SHOOTER_COLOR, 2)

    # Shooting arm — raises during shot
    arm_raise = min(1.0, phase * 1.8)
    elbow_x = bx + int(22 * arm_raise)
    elbow_y = head_y + int(20 - 15 * arm_raise)
    wrist_x = bx + int(32 * arm_raise)
    wrist_y = head_y + int(10 - 28 * arm_raise)

    cv2.line(frame, (bx + 8, head_y + 20), (elbow_x, elbow_y), SHOOTER_COLOR, 3)
    cv2.line(frame, (elbow_x, elbow_y),     (wrist_x, wrist_y), SHOOTER_COLOR, 2)

    # Guide arm
    cv2.line(frame, (bx - 5, head_y + 22), (bx - 18, head_y + 8), SHOOTER_COLOR, 2)

    return wrist_x, wrist_y


def draw_ball(frame, cx, cy, spin_angle):
    cv2.circle(frame, (cx, cy), BALL_R, BALL_COLOR, -1)
    # Seam lines (rotate with spin)
    for angle_offset in [0, 90]:
        a = math.radians(spin_angle + angle_offset)
        x1 = int(cx + BALL_R * math.cos(a))
        y1 = int(cy + BALL_R * math.sin(a))
        x2 = int(cx - BALL_R * math.cos(a))
        y2 = int(cy - BALL_R * math.sin(a))
        cv2.line(frame, (x1, y1), (x2, y2), BALL_SEAM, 1)
    # Highlight
    hx = int(cx - BALL_R * 0.3)
    hy = int(cy - BALL_R * 0.3)
    cv2.circle(frame, (hx, hy), int(BALL_R * 0.25), (230, 140, 80), -1)
    cv2.circle(frame, (cx, cy), BALL_R, BALL_SEAM, 1)


def draw_angle_annotation(frame, launch_angle_deg, shot_num, total_shots):
    """Draw ground-truth angle in corner — this is what the CV should find."""
    cv2.rectangle(frame, (W - 260, 10), (W - 10, 80), (20, 25, 35), -1)
    cv2.rectangle(frame, (W - 260, 10), (W - 10, 80), (80, 90, 100), 1)
    cv2.putText(frame, "GROUND TRUTH", (W - 248, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 130, 140), 1)
    cv2.putText(frame, f"Arc: {launch_angle_deg:.1f} deg", (W - 248, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 210, 220), 2)
    cv2.putText(frame, f"Shot {shot_num}/{total_shots}", (W - 248, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 110, 120), 1)


def draw_hud(frame, state):
    cv2.putText(frame, "SHOTLAB — SYNTHETIC TEST", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 100, 120), 1)
    cv2.putText(frame, f"Ball: BLUE  |  View: SIDE  |  {state}", (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 80, 100), 1)


def simulate_shot(
    launch_angle_deg: float,
    v0: float = 8.5,
    origin_x: int = SHOOTER_X,
    origin_y: int = FLOOR_Y - 110,  # wrist release height
    scale: float = 420,
) -> list[tuple[int, int, float]]:
    """
    Simulate a basketball trajectory under gravity.
    Returns list of (px, py, t) at each frame.
    
    Physics:
        x(t) = v0 * cos(θ) * t
        y(t) = v0 * sin(θ) * t - 0.5 * g * t²
    """
    g = 9.81  # m/s²
    theta = math.radians(launch_angle_deg)
    vx = v0 * math.cos(theta)
    vy = v0 * math.sin(theta)

    points = []
    dt = 1.0 / FPS
    t = 0.0

    while True:
        x_m = vx * t
        y_m = vy * t - 0.5 * g * t * t

        px = int(origin_x + x_m * scale)
        py = int(origin_y - y_m * scale)

        points.append((px, py, t))

        # Stop when ball hits floor level or goes off screen
        if py >= FLOOR_Y or px > W + 50:
            break
        if t > 3.0:
            break

        t += dt

    return points


def generate_video(
    output_path: str,
    target_angle: float,
    n_shots: int = 5,
    angle_variance: float = 1.5,
    pause_between_shots: float = 1.2,
) -> list[float]:
    """
    Generate a synthetic test video with n_shots at ~target_angle.
    Returns list of actual per-shot angles used.
    """
    out = cv2.VideoWriter(output_path, CODEC, FPS, (W, H))
    actual_angles = []

    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    for shot_num in range(1, n_shots + 1):
        # Vary angle slightly around target (realistic shooter variance)
        angle = target_angle + rng.normal(0, angle_variance)
        angle = float(np.clip(angle, target_angle - 4, target_angle + 4))
        actual_angles.append(round(angle, 2))

        trajectory = simulate_shot(angle)

        # ── Wind-up frames (shooter crouching, no ball) ──────────────────────
        windup_frames = int(FPS * 0.4)
        for f in range(windup_frames):
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            draw_background(frame)
            draw_basket(frame)
            phase = f / windup_frames * 0.4
            draw_shooter(frame, phase)
            draw_hud(frame, f"SHOT {shot_num} — WIND-UP")
            draw_angle_annotation(frame, angle, shot_num, n_shots)
            out.write(frame)

        # ── Shot frames ───────────────────────────────────────────────────────
        spin = 0.0
        for i, (bx, by, t) in enumerate(trajectory):
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            draw_background(frame)
            draw_basket(frame)

            phase = min(1.0, t * 3.5)
            draw_shooter(frame, phase)

            # Ball
            spin += 8  # degrees per frame
            draw_ball(frame, bx, by, spin)

            # Arc trail
            if i >= 3:
                for j in range(max(0, i - 12), i):
                    tx, ty, _ = trajectory[j]
                    alpha = (j - max(0, i-12)) / 12
                    c = int(60 + alpha * 80)
                    cv2.circle(frame, (tx, ty), int(BALL_R * 0.3 * alpha), (c, c//2, c//4), -1)

            draw_hud(frame, f"SHOT {shot_num} — IN FLIGHT  t={t:.2f}s")
            draw_angle_annotation(frame, angle, shot_num, n_shots)
            out.write(frame)

        # ── Landing flash ─────────────────────────────────────────────────────
        for _ in range(int(FPS * 0.15)):
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            draw_background(frame)
            draw_basket(frame)
            draw_shooter(frame, 1.0)
            if trajectory:
                bx, by, _ = trajectory[-1]
                draw_ball(frame, bx, by, spin)
            draw_hud(frame, f"SHOT {shot_num} — LANDED")
            draw_angle_annotation(frame, angle, shot_num, n_shots)
            out.write(frame)

        # ── Pause between shots ───────────────────────────────────────────────
        pause_frames = int(FPS * pause_between_shots)
        for f in range(pause_frames):
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            draw_background(frame)
            draw_basket(frame)
            draw_shooter(frame, 0.0)
            draw_hud(frame, f"SHOT {shot_num} COMPLETE — {angle:.1f} deg")
            draw_angle_annotation(frame, angle, shot_num, n_shots)
            out.write(frame)

    out.release()
    print(f"  Written: {output_path}  ({n_shots} shots)")
    return actual_angles


def main():
    out_dir = Path(__file__).parent
    ground_truth = {}

    configs = [
        ("test_45deg.mp4",  45.0, "Good form — optimal arc"),
        ("test_48deg.mp4",  48.0, "Slightly high — still good"),
        ("test_30deg.mp4",  30.0, "Flat arc — bad form"),
        ("test_65deg.mp4",  65.0, "Very high — like the bug we saw"),
    ]

    print("Generating synthetic test videos...")
    print(f"Resolution: {W}x{H}  FPS: {FPS}  Ball: BLUE")
    print()

    for filename, target_angle, description in configs:
        path = str(out_dir / filename)
        print(f"[{target_angle}°] {description}")
        angles = generate_video(path, target_angle, n_shots=5)
        ground_truth[filename] = {
            "target_angle":  target_angle,
            "description":   description,
            "per_shot_angles": angles,
            "mean_angle":    round(sum(angles) / len(angles), 2),
            "n_shots":       len(angles),
        }
        print(f"    Actual angles: {angles}")
        print(f"    Mean: {ground_truth[filename]['mean_angle']}°")
        print()

    # Save ground truth JSON
    gt_path = str(out_dir / "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Ground truth saved: {gt_path}")
    print()
    print("Now run the pipeline against each video:")
    print("  python3 run_test.py")


if __name__ == "__main__":
    main()
