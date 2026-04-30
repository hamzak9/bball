import sys, logging
logging.disable(logging.CRITICAL)

sys.path.insert(0, '.')
from analyzer.ball import track_ball, smooth_trajectory, detect_ball_color
from analyzer.shots import detect_shots
from analyzer.metrics import compute_arc_angle
import numpy as np

cases = [
    ('test_45deg.mp4', 44.7),
    ('test_30deg.mp4', 29.7),
    ('test_48deg.mp4', 47.7),
    ('test_65deg.mp4', 64.7),
]

print(f"\n{'Video':<22} {'Expected':>10} {'Got':>8} {'Error':>8} {'Shots':>6}  Status")
print("-" * 65)

for name, expected in cases:
    try:
        hsv = detect_ball_color(name)
        dets = smooth_trajectory(track_ball(name, max_frames=1200, hsv_ranges=hsv))
        shots = detect_shots(dets, [], fps=60.0)
        arcs = [a for a in [compute_arc_angle(s) for s in shots] if a is not None]
        mean_arc = round(float(np.mean(arcs)), 1) if arcs else None
        error = round(abs(mean_arc - expected), 1) if mean_arc is not None else None
        status = '✅ PASS' if (error is not None and error <= 5) else ('⚠  CLOSE' if (error is not None and error <= 10) else '❌ FAIL')
        got_str = f"{mean_arc}°" if mean_arc is not None else "—"
        err_str = f"±{error}°" if error is not None else "—"
        print(f"{name:<22} {expected:>9}° {got_str:>8} {err_str:>9} {len(shots):>6}  {status}")
        if arcs:
            print(f"  {'':22} per-shot: {[f'{a:.1f}°' for a in arcs]}")
    except Exception as e:
        print(f"{name:<22} ERROR: {e}")

print()
