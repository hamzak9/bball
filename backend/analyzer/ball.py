"""
ball.py — Basketball detection and tracking

Strategy:
  1. Auto-detect ball color from the first few frames (samples circular blobs,
     finds the dominant hue — works for orange, blue, red, yellow, white balls)
  2. HSV color segmentation using the detected color range
  3. Hough circle transform on the masked region
  4. Temporal smoothing to reject false detections
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BallDetection:
    frame: int
    cx: float       # normalized 0..1
    cy: float       # normalized 0..1
    radius: float   # normalized
    confidence: float


# Fallback HSV ranges if auto-detect fails (covers orange + common colors)
_FALLBACK_RANGES = [
    ((5,  100, 100), (25, 255, 255)),   # orange
    ((0,  120,  80), ( 8, 255, 255)),   # red-orange
    ((95, 100,  80), (130, 255, 255)),  # blue
    ((35,  80,  80), ( 85, 255, 255)),  # green/teal
]


def detect_ball_color(video_path: str, sample_frames: int = 30) -> list[tuple]:
    """
    Sample frames from the video and detect the dominant ball color.
    Returns a list of HSV (lo, hi) range tuples to use for masking.

    Works by:
    1. Finding circular blobs across sampled frames
    2. Sampling their dominant HSV hue
    3. Building a tight HSV range around that hue
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _FALLBACK_RANGES

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    step     = max(1, n_frames // sample_frames)

    hue_samples = []

    for i in range(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break

        if height > 720:
            frame = cv2.resize(frame, (int(width * 720 / height), 720))

        h, w = frame.shape[:2]
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        min_r = max(8, int(min(h, w) * 0.015))
        max_r = min(100, int(min(h, w) * 0.12))

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=min_r * 3,
            param1=60, param2=22,
            minRadius=min_r, maxRadius=max_r,
        )
        if circles is None:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for (cx, cy, r) in np.round(circles[0]).astype(int):
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), max(2, r - 2), 255, -1)
            pixels = hsv[mask > 0]
            if len(pixels) == 0:
                continue
            # Filter out near-white and near-black (court markings, lines)
            sat = pixels[:, 1]
            val = pixels[:, 2]
            colored = pixels[(sat > 60) & (val > 60)]
            if len(colored) < 10:
                continue
            hue_samples.extend(colored[:, 0].tolist())

    cap.release()

    if len(hue_samples) < 20:
        logger.warning("Ball color auto-detect failed — using fallback (orange+blue)")
        return _FALLBACK_RANGES

    # Find dominant hue cluster via histogram
    hue_arr = np.array(hue_samples, dtype=np.float32)
    hist = np.zeros(180, dtype=np.float32)
    for h in hue_arr:
        hist[int(h) % 180] += 1

    # Smooth histogram
    hist = np.convolve(hist, np.ones(7) / 7, mode='same')

    dominant_hue = int(np.argmax(hist))
    logger.info(f"Auto-detected ball hue: {dominant_hue} (0=red, 30=yellow, 60=green, 120=blue, 150=purple)")

    hsv_ranges = _hue_to_ranges(dominant_hue)
    logger.info(f"Using HSV ranges: {hsv_ranges}")
    return hsv_ranges


def _hue_to_ranges(hue: int) -> list[tuple]:
    """Build HSV (lo, hi) ranges for a given dominant hue."""
    margin = 18   # ± hue tolerance
    sat_lo, val_lo = 70, 60

    lo_hue = (hue - margin) % 180
    hi_hue = (hue + margin) % 180

    if lo_hue < hi_hue:
        return [((lo_hue, sat_lo, val_lo), (hi_hue, 255, 255))]
    else:
        # Wraps around 0/180
        return [
            ((lo_hue, sat_lo, val_lo), (179, 255, 255)),
            ((0,      sat_lo, val_lo), (hi_hue, 255, 255)),
        ]


def detect_ball_in_frame(
    frame: np.ndarray,
    hsv_ranges: list[tuple],
    prev_pos: Optional[tuple] = None,
    search_radius_px: int = 150,
) -> Optional[BallDetection]:
    """Detect basketball in a single BGR frame using given HSV ranges."""
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Build color mask
    mask = np.zeros((H, W), dtype=np.uint8)
    for (lo, hi) in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Restrict search window if we have a prior position
    search_mask = mask.copy()
    if prev_pos is not None:
        px, py = int(prev_pos[0]), int(prev_pos[1])
        window = np.zeros_like(mask)
        x1, y1 = max(0, px - search_radius_px), max(0, py - search_radius_px)
        x2, y2 = min(W, px + search_radius_px), min(H, py + search_radius_px)
        window[y1:y2, x1:x2] = 255
        candidate = cv2.bitwise_and(mask, window)
        search_mask = candidate if candidate.sum() > 300 else mask

    blurred = cv2.GaussianBlur(search_mask, (9, 9), 2)

    min_r = max(8,  int(min(H, W) * 0.015))
    max_r = min(120, int(min(H, W) * 0.14))

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min_r * 2,
        param1=50, param2=16,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    best, best_score = None, -1

    for (cx, cy, r) in circles:
        circle_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), r, 255, -1)
        overlap = cv2.bitwise_and(mask, circle_mask)
        fill = overlap.sum() / 255 / (np.pi * r * r)

        if fill < 0.20:
            continue

        prox = 0.0
        if prev_pos is not None:
            dist = np.hypot(cx - prev_pos[0], cy - prev_pos[1])
            prox = max(0, 1.0 - dist / (search_radius_px * 1.5))

        score = fill * 0.6 + prox * 0.4
        if score > best_score:
            best_score, best = score, (cx, cy, r)

    if best is None:
        return None

    cx, cy, r = best
    return BallDetection(frame=0, cx=cx/W, cy=cy/H, radius=r/min(H,W), confidence=best_score)


def track_ball(video_path: str, max_frames: int = 2000, hsv_ranges: list[tuple] | None = None) -> list[BallDetection]:
    """
    Run ball detection across all frames of a video.
    Auto-detects ball color if hsv_ranges not provided.
    """
    if hsv_ranges is None:
        logger.info("Auto-detecting ball color...")
        hsv_ranges = detect_ball_color(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)

    detections: list[BallDetection] = []
    prev_px = None
    frame_idx = 0

    while frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        scale = 1.0
        if height > 720:
            scale = 720 / height
            frame = cv2.resize(frame, (int(width * scale), 720))

        det = detect_ball_in_frame(
            frame, hsv_ranges,
            prev_pos=prev_px,
            search_radius_px=int(150 * scale) if prev_px else 220,
        )

        if det is not None:
            det.frame = frame_idx
            detections.append(det)
            prev_px = (det.cx * frame.shape[1], det.cy * frame.shape[0])
        else:
            prev_px = None

        frame_idx += 1

    cap.release()
    logger.info(f"Ball tracking: {len(detections)}/{n_frames} frames detected")
    return detections


def smooth_trajectory(detections: list[BallDetection], window: int = 3) -> list[BallDetection]:
    if len(detections) < 3:
        return detections

    filtered = [detections[0]]
    for i in range(1, len(detections)):
        prev, curr = filtered[-1], detections[i]
        if abs(curr.cx - prev.cx) > 0.18 or abs(curr.cy - prev.cy) > 0.18:
            continue
        filtered.append(curr)

    smoothed = []
    for i, d in enumerate(filtered):
        lo = max(0, i - window)
        hi = min(len(filtered), i + window + 1)
        cx = np.mean([x.cx for x in filtered[lo:hi]])
        cy = np.mean([x.cy for x in filtered[lo:hi]])
        smoothed.append(BallDetection(d.frame, cx, cy, d.radius, d.confidence))

    return smoothed
