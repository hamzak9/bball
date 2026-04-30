"""
run_test.py — Run the CV pipeline against synthetic videos and compare to ground truth

Usage:
    cd backend
    python3 run_test.py

Prints a table showing ground truth vs pipeline output for each video.
"""

import json
import sys
from pathlib import Path

# Suppress TF/mediapipe log noise
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

from analyzer.pipeline import run_analysis


def run_test(video_file: str, ground_truth: dict) -> dict:
    path = str(Path(__file__).parent / video_file)
    if not Path(path).exists():
        return {"error": f"File not found: {path}. Run generate_test_video.py first."}

    print(f"\nRunning: {video_file}")
    print(f"  Expected arc: {ground_truth['mean_angle']}° ({ground_truth['description']})")

    try:
        result = run_analysis({"side": path})
    except Exception as e:
        return {"error": str(e)}

    if result.get("status") == "no_shots_detected":
        return {
            "status":   "no_shots_detected",
            "message":  result.get("message"),
            "tip":      result.get("tip"),
            "debug":    result.get("debug"),
        }

    got_arc    = result.get("arc", {}).get("avg_deg")
    got_shots  = result.get("n_shots", 0)
    expected   = ground_truth["mean_angle"]
    arc_error  = abs(got_arc - expected) if got_arc is not None else None

    status = "✅ PASS" if (arc_error is not None and arc_error <= 5.0) else "❌ FAIL"

    return {
        "status":          status,
        "expected_arc":    expected,
        "expected_shots":  ground_truth["n_shots"],
        "got_arc":         got_arc,
        "got_shots":       got_shots,
        "arc_error_deg":   round(arc_error, 1) if arc_error else None,
        "overall_score":   result.get("overall"),
        "scores": {
            "arc":     result.get("arc",     {}).get("score"),
            "drift":   result.get("drift",   {}).get("score"),
            "release": result.get("release", {}).get("score"),
            "balance": result.get("balance", {}).get("score"),
            "pocket":  result.get("pocket",  {}).get("score"),
        },
        "per_shot_arcs": result.get("arc", {}).get("per_shot", []),
    }


def main():
    gt_path = Path(__file__).parent / "ground_truth.json"
    if not gt_path.exists():
        print("ERROR: ground_truth.json not found.")
        print("Run this first:  python3 generate_test_video.py")
        sys.exit(1)

    with open(gt_path) as f:
        ground_truth = json.load(f)

    results = {}
    for video_file, gt in ground_truth.items():
        results[video_file] = run_test(video_file, gt)

    # ── Print results table ───────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  PIPELINE ACCURACY RESULTS")
    print("═" * 70)
    print(f"  {'Video':<22} {'Expected':>10} {'Got':>8} {'Error':>8} {'Shots':>7}  {'Status'}")
    print("─" * 70)

    passes, total = 0, 0
    for video_file, r in results.items():
        if "error" in r:
            print(f"  {video_file:<22}  ERROR: {r['error']}")
            continue

        if r["status"] == "no_shots_detected":
            print(f"  {video_file:<22}  NO SHOTS — {r.get('message', '')[:35]}")
            total += 1
            continue

        exp   = f"{r['expected_arc']}°"
        got   = f"{r['got_arc']}°" if r['got_arc'] else "—"
        err   = f"{r['arc_error_deg']}°" if r['arc_error_deg'] else "—"
        shots = f"{r['got_shots']}/{r['expected_shots']}"
        status = r["status"]

        print(f"  {video_file:<22} {exp:>10} {got:>8} {err:>8} {shots:>7}  {status}")

        if r.get("per_shot_arcs"):
            arcs = [f"{a}°" for a in r["per_shot_arcs"]]
            print(f"  {'':22}  Per-shot: {', '.join(arcs)}")

        if "✅" in status:
            passes += 1
        total += 1

    print("─" * 70)
    print(f"  Result: {passes}/{total} passed  (threshold: ±5°)")
    print("═" * 70)
    print()
    print("Interpretation:")
    print("  ±2°  = excellent (measurement noise is normal)")
    print("  ±5°  = acceptable (partial arc or lighting)")
    print("  >5°  = needs fixing in the CV pipeline")
    print()

    # Save full results
    out_path = Path(__file__).parent / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved: {out_path}")


if __name__ == "__main__":
    main()
