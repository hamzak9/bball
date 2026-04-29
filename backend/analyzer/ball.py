"""
ball.py — Basketball detection and tracking

Strategy:
  1. HSV color segmentation for orange basketball
  2. Hough circle transform on the masked region
  3. Kalman-style temporal smoothing to reject false detections
  4. Returns list of (frame_idx, cx, cy, radius) for confident detections
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BallDetection:
    frame: int
    cx: float   # normalized 0..1
    cy: float   # normalized 0..1
    radius: float  # normalized
    confidence: float


# HSV ranges for a basketball under typical gym/outdoor lighting
# Two ranges to handle the hue wrap-around at red-orange
BALL_HSV_RANGES = [
    ((5,  100, 100), (25, 255, 255)),   # orange core
    ((0,  120,  80), ( 8, 255, 255)),   # red-orange edge
]


def detect_ball_in_frame(
    frame: np.ndarray,
    prev_pos: Optional[tuple] = None,
    search_radius_px: int = 120,
) -> Optional[BallDetection]:
    """
    Detect basketball in a single BGR frame.
    If prev_pos (cx_px, cy_px) is given, restricts search to a region around it.
    """
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Build orange mask
    mask = np.zeros((H, W), dtype=np.uint8)
    for (lo, hi) in BALL_HSV_RANGES:
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
        x1 = max(0, px - search_radius_px)
        y1 = max(0, py - search_radius_px)
        x2 = min(W, px + search_radius_px)
        y2 = min(H, py + search_radius_px)
        window[y1:y2, x1:x2] = 255
        search_mask = cv2.bitwise_and(mask, window)
        if search_mask.sum() < 500:
            search_mask = mask  # fall back to full frame

    # Blur for circle detection
    blurred = cv2.GaussianBlur(search_mask, (9, 9), 2)

    # Hough circles — min/max radius tuned for a basketball at typical shooting distance
    min_r = max(8,  int(min(H, W) * 0.018))
    max_r = min(120, int(min(H, W) * 0.14))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_r * 2,
        param1=50,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # Score each circle: prefer ones with high mask coverage + close to prior
    best = None
    best_score = -1

    for (cx, cy, r) in circles:
        # Check mask fill inside circle
        circle_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), r, 255, -1)
        overlap = cv2.bitwise_and(mask, circle_mask)
        area = np.pi * r * r
        fill = overlap.sum() / 255 / area

        if fill < 0.25:  # less than 25% orange fill → not a ball
            continue

        # Proximity bonus
        prox = 0.0
        if prev_pos is not None:
            dist = np.hypot(cx - prev_pos[0], cy - prev_pos[1])
            prox = max(0, 1.0 - dist / (search_radius_px * 1.5))

        score = fill * 0.6 + prox * 0.4
        if score > best_score:
            best_score = score
            best = (cx, cy, r)

    if best is None:
        return None

    cx, cy, r = best
    return BallDetection(
        frame=0,
        cx=cx / W,
        cy=cy / H,
        radius=r / min(H, W),
        confidence=best_score,
    )


def track_ball(video_path: str, max_frames: int = 2000) -> list[BallDetection]:
    """
    Run ball detection across all frames of a video.
    Returns list of BallDetection (one per frame where ball is found).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)

    detections: list[BallDetection] = []
    prev_px = None
    frame_idx = 0

    while frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Downsample large frames for speed (process at max 720p)
        scale = 1.0
        if height > 720:
            scale = 720 / height
            frame = cv2.resize(frame, (int(width * scale), 720))

        det = detect_ball_in_frame(
            frame,
            prev_pos=prev_px,
            search_radius_px=int(120 * scale) if prev_px else 200,
        )

        if det is not None:
            det.frame = frame_idx
            detections.append(det)
            prev_px = (det.cx * frame.shape[1], det.cy * frame.shape[0])
        else:
            prev_px = None  # lost the ball — reset search

        frame_idx += 1

    cap.release()
    return detections


def smooth_trajectory(detections: list[BallDetection], window: int = 3) -> list[BallDetection]:
    """
    Simple moving-average smoothing of ball positions.
    Removes detections that are outliers (jump > 15% of frame in one step).
    """
    if len(detections) < 3:
        return detections

    # Remove spatial outliers
    filtered = [detections[0]]
    for i in range(1, len(detections)):
        prev = filtered[-1]
        curr = detections[i]
        if abs(curr.cx - prev.cx) > 0.15 or abs(curr.cy - prev.cy) > 0.15:
            continue  # too big a jump — likely false detection
        filtered.append(curr)

    # Moving average
    smoothed = []
    for i, d in enumerate(filtered):
        lo = max(0, i - window)
        hi = min(len(filtered), i + window + 1)
        cx = np.mean([x.cx for x in filtered[lo:hi]])
        cy = np.mean([x.cy for x in filtered[lo:hi]])
        smoothed.append(BallDetection(d.frame, cx, cy, d.radius, d.confidence))

    return smoothed
