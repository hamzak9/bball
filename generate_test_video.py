"""
generate_test_video.py — Synthetic basketball shot video with known ground truth

Generates side-view videos where every parameter is exact:
  - Ball follows real projectile physics (exact arc angle)
  - Ball is bright blue (easily detectable)
  - Each shot is clearly separated
  - Ground truth JSON tells you exactly what to expect

Run: python3 generate_test_video.py
"""

import cv2
import numpy as np
import json
import math
from pathlib import Path

W, H  = 1280, 720
FPS   = 60
OUTPUT = Path(__file__).parent / "test_videos"
OUTPUT.mkdir(exist_ok=True)

# Colors (BGR)
BG_COLOR       = (28,  32,  38)
COURT_COLOR    = (42,  47,  56)
LINE_COLOR     = (75,  82,  95)
BALL_FILL      = (220, 130,  40)   # bright blue (B=220 dominant)
BALL_OUTLINE   = (160,  80,  20)
HOOP_COLOR     = ( 50, 130, 240)   # orange rim
BOARD_COLOR    = (200, 205, 215)
FIGURE_COLOR   = (210, 215, 225)
SHADOW_COLOR   = ( 18,  22,  28)


def draw_court(frame):
    floor_y = int(H * 0.82)
    cv2.rectangle(frame, (0, floor_y), (W, H), COURT_COLOR, -1)
    cv2.line(frame, (0, floor_y), (W, floor_y), LINE_COLOR, 2)
    cv2.line(frame, (int(W*.1), floor_y), (int(W*.1), H), LINE_COLOR, 1)
    cv2.line(frame, (int(W*.5), floor_y), (int(W*.5), H), LINE_COLOR, 1)
    return floor_y


def draw_hoop(frame, hx, hy):
    # Support pole
    cv2.line(frame, (hx+22, hy+10), (hx+22, int(H*0.82)), (60,65,75), 4)
    # Backboard
    cv2.rectangle(frame, (hx+14, hy-40), (hx+28, hy+14), BOARD_COLOR, -1)
    cv2.rectangle(frame, (hx+14, hy-40), (hx+28, hy+14), (130,135,145), 2)
    # Inner square
    cv2.rectangle(frame, (hx+15, hy-18), (hx+27, hy+2), (130,135,145), 1)
    # Rim
    cv2.line(frame, (hx-22, hy), (hx+14, hy), HOOP_COLOR, 4)
    # Net (5 lines)
    for i in range(5):
        x = hx - 20 + i*10
        cv2.line(frame, (x, hy), (x+3, hy+30), (160,165,175), 1)
    cv2.line(frame, (hx-20, hy+30), (hx+14, hy+30), (160,165,175), 1)


def draw_figure(frame, fx, floor_y, phase):
    """phase 0=ready, 0.5=release, 1.0=follow-through"""
    lift = int(math.sin(min(phase,1)*math.pi) * 14)
    def y(raw): return raw - lift

    ay = floor_y;        ky = floor_y-60;  hy2= floor_y-110
    sy = floor_y-168;   ny = floor_y-188; hy3= floor_y-205
    c, lw = FIGURE_COLOR, 3

    # Legs
    cv2.line(frame,(fx, y(ay)),(fx,   y(ky)), c, lw)
    cv2.line(frame,(fx, y(ky)),(fx-4, y(hy2)),c, lw)
    # Torso
    cv2.line(frame,(fx-2,y(hy2)),(fx,y(sy)), c, lw)
    cv2.line(frame,(fx,  y(sy)), (fx,y(ny)), c, lw)
    # Head
    cv2.circle(frame,(fx, y(hy3)), 15, c, lw)

    # Shooting arm — rises with phase
    arm_raise = min(phase*2, 1.0)
    ex = fx + 22
    ey = y(sy) - int(arm_raise * 30)
    wx = fx + 28
    wy = ey - int(arm_raise * 55)
    cv2.line(frame,(fx, y(sy)), (ex, ey), c, lw)
    cv2.line(frame,(ex, ey),    (wx, wy), c, lw+1)

    # Guide arm
    gx = fx - 14
    gy = y(sy) - int(arm_raise * 20)
    cv2.line(frame,(fx, y(sy)),(gx, gy), c, 2)

    return (wx, wy)   # wrist position


def compute_trajectory(rx, ry, tx, ty, angle_deg, fps):
    """
    Real projectile physics. Returns list of (px, py) pixel positions.
    rx,ry = release point pixels. tx,ty = target pixels.
    angle_deg = launch angle (degrees above horizontal, toward target).
    """
    PPM = 70.0   # pixels per meter

    # Convert to metres, flip y (physics: up=positive, pixels: up=smaller y)
    rx_m = rx / PPM
    ry_m = (H - ry) / PPM   # flip: ry_m increases upward
    tx_m = tx / PPM
    ty_m = (H - ty) / PPM

    dx = tx_m - rx_m
    dy = ty_m - ry_m   # positive = target is above release
    g  = 9.81

    angle_rad = math.radians(angle_deg)
    tan_a  = math.tan(angle_rad)
    cos2_a = math.cos(angle_rad) ** 2

    denom = dx * tan_a - dy
    if denom <= 0:
        raise ValueError(f"Angle {angle_deg}° cannot reach target (denom={denom:.3f})")

    v0_sq = (g * dx**2) / (2 * cos2_a * denom)
    v0    = math.sqrt(v0_sq)
    vx    = v0 * math.cos(angle_rad)
    vy    = v0 * math.sin(angle_rad)

    positions = []
    dt = 1.0 / fps
    t  = 0.0
    peaked = False

    while t < 6.0:
        px_m = rx_m + vx * t
        py_m = ry_m + vy * t - 0.5 * g * t**2

        # Back to pixels
        px_px = px_m * PPM
        py_px = H - py_m * PPM   # flip back

        positions.append((px_px, py_px))

        # Track peak
        if vy - g*t < 0:
            peaked = True

        # Stop: ball passed target x AND has peaked (on descent)
        if peaked and px_px >= tx - 8:
            break

        # Stop: ball went very low (missed badly)
        if py_px > H * 0.95:
            break

        t += dt

    return positions


def generate_video(filename, angle_deg, n_shots=5, noise_std=0.0):
    out_path = str(OUTPUT / filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    floor_y  = int(H * 0.82)   # 590
    shooter_x = int(W * 0.16)  # 204
    hoop_x    = int(W * 0.80)  # 1024
    hoop_y    = int(H * 0.36)  # 259

    # Release point = wrist position at release
    release_x = shooter_x + 30
    release_y = floor_y - 205   # ~385px from top

    np.random.seed(42)
    actual_angles = []
    inter_shot    = int(FPS * 1.0)   # 1s rest between shots

    # Static "ready" pose for a few frames at start
    for _ in range(int(FPS * 0.3)):
        f = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
        draw_court(f); draw_hoop(f, hoop_x, hoop_y)
        wrist = draw_figure(f, shooter_x, floor_y, 0.0)
        cv2.circle(f, wrist, 14, BALL_OUTLINE, -1)
        cv2.circle(f, wrist, 12, BALL_FILL, -1)
        writer.write(f)

    for shot_i in range(n_shots):
        a = angle_deg + float(np.random.normal(0, noise_std))
        a = float(np.clip(a, 25, 68))
        actual_angles.append(round(a, 2))

        try:
            traj = compute_trajectory(release_x, release_y, hoop_x, hoop_y, a, FPS)
        except ValueError as e:
            print(f"  Shot {shot_i+1} skip: {e}")
            continue

        n = len(traj)

        # Wind-up (0.3s)
        for f_i in range(int(FPS*0.3)):
            phase = f_i / (FPS*0.3) * 0.45
            frame = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
            draw_court(frame); draw_hoop(frame, hoop_x, hoop_y)
            wrist = draw_figure(frame, shooter_x, floor_y, phase)
            cv2.circle(frame, wrist, 14, BALL_OUTLINE, -1)
            cv2.circle(frame, wrist, 12, BALL_FILL, -1)
            writer.write(frame)

        # Ball in flight
        for fi, (bx, by) in enumerate(traj):
            bxi, byi = int(bx), int(by)
            phase = 0.5 + (fi/max(1,n-1)) * 0.5
            frame = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
            draw_court(frame); draw_hoop(frame, hoop_x, hoop_y)
            draw_figure(frame, shooter_x, floor_y, phase)

            # Shadow
            shadow_r = max(4, 14 - int((floor_y - byi)/60))
            cv2.ellipse(frame,(bxi,floor_y),(shadow_r,3),0,0,360,SHADOW_COLOR,-1)

            # Ball — large + bright for reliable detection
            cv2.circle(frame,(bxi,byi), 16, BALL_OUTLINE, -1)
            cv2.circle(frame,(bxi,byi), 14, BALL_FILL, -1)
            # Seam
            cv2.ellipse(frame,(bxi,byi),(14,5),15,0,360,BALL_OUTLINE,1)
            cv2.line(frame,(bxi-14,byi),(bxi+14,byi),BALL_OUTLINE,1)

            # Label on first shot only
            if shot_i == 0 and fi == 0:
                cv2.putText(frame, f"GT: {a:.1f} deg", (30,38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80,220,80), 2, cv2.LINE_AA)

            writer.write(frame)

        # Follow-through + rest
        for f_i in range(inter_shot):
            phase = max(0.5, 1.0 - f_i/inter_shot*0.4)
            frame = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
            draw_court(frame); draw_hoop(frame, hoop_x, hoop_y)
            draw_figure(frame, shooter_x, floor_y, phase)
            writer.write(frame)

    writer.release()

    gt = {
        "file":           out_path,
        "n_shots":        len(actual_angles),
        "target_angle":   angle_deg,
        "actual_angles":  actual_angles,
        "avg_angle":      round(float(np.mean(actual_angles)), 2),
        "std_angle":      round(float(np.std(actual_angles)), 2),
        "drift_inches":   0.0,
        "ball_color":     "blue (BGR 220,130,40)",
        "view":           "side",
        "fps":            FPS,
        "resolution":     f"{W}x{H}",
        "release_px":     [release_x, release_y],
        "hoop_px":        [hoop_x, hoop_y],
    }
    return gt


def main():
    print("Generating synthetic test videos with known ground truth...\n")

    tests = [
        ("test_45deg_5shots.mp4",        45.0, 5, 0.0),   # perfect 45° x5
        ("test_52deg_5shots.mp4",        52.0, 5, 0.0),   # high arc x5
        ("test_38deg_5shots.mp4",        38.0, 5, 0.0),   # flat/low arc x5
        ("test_45deg_noisy_8shots.mp4",  45.0, 8, 4.0),   # consistent target, noisy
    ]

    all_gt = {}
    for fname, angle, n, noise in tests:
        print(f"  {fname}")
        print(f"    angle={angle}°  shots={n}  noise=±{noise}°")
        gt = generate_video(fname, angle, n, noise)
        all_gt[fname] = gt
        print(f"    actual: {gt['actual_angles']}  avg={gt['avg_angle']}°\n")

    gt_path = str(OUTPUT / "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(all_gt, f, indent=2)

    print(f"Ground truth → {gt_path}")
    print()
    print("── Run pipeline test ───────────────────────────────────")
    print("cd backend")
    print("python3 -c \"")
    print("from analyzer.pipeline import run_analysis; import json")
    print("r = run_analysis({'side': '../test_videos/test_45deg_5shots.mp4'})")
    print("print('arc avg:', r['arc']['avg_deg'], '(expected 45.0)')")
    print("print('n_shots:', r['n_shots'], '(expected 5)')")
    print("\"")


if __name__ == "__main__":
    main()
