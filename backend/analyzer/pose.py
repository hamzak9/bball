"""
pose.py — MediaPipe BlazePose landmark extraction (Tasks API, mediapipe 0.10+)

Auto-downloads the pose landmarker model on first run and caches it locally.
"""

import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

logger = logging.getLogger(__name__)

# Model file — cached next to this module
MODEL_PATH = Path(__file__).parent.parent / "pose_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

# Landmark indices we use
LM = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
}


@dataclass
class PoseFrame:
    frame: int
    landmarks: dict
    visibility: dict

    def get(self, name: str) -> Optional[tuple]:
        if self.visibility.get(name, 0.0) < 0.35:
            return None
        return self.landmarks.get(name)

    def get_xy(self, name: str) -> Optional[tuple]:
        pt = self.get(name)
        return (pt[0], pt[1]) if pt else None


def angle_between(a, b, c) -> float:
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def ensure_model() -> str:
    """Download model if not cached. Returns path to model file."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return str(MODEL_PATH)

    logger.info(f"Downloading pose landmarker model (~29MB)...")
    logger.info(f"  from: {MODEL_URL}")
    logger.info(f"  to:   {MODEL_PATH}")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        size_mb = MODEL_PATH.stat().st_size / 1_000_000
        logger.info(f"Model downloaded: {size_mb:.1f}MB")
        return str(MODEL_PATH)
    except Exception as e:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise RuntimeError(
            f"Could not download pose model: {e}\n"
            f"Manual fix: download the file from\n  {MODEL_URL}\n"
            f"and save it to\n  {MODEL_PATH}"
        )


def extract_poses(video_path: str, max_frames: int = 2000) -> list[PoseFrame]:
    """
    Run MediaPipe Pose on every frame of the video.
    Returns list of PoseFrame for frames where a person is detected.
    """
    model_path = ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)

    results_list: list[PoseFrame] = []

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        while frame_idx < n_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Downsample for speed
            if height > 720:
                scale = 720 / height
                frame = cv2.resize(frame, (int(width * scale), 720))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # VIDEO mode requires monotonically increasing timestamps in ms
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lms = result.pose_landmarks[0]
                world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

                frame_lms = {}
                frame_vis = {}
                for name, idx in LM.items():
                    if idx < len(lms):
                        lm = lms[idx]
                        frame_lms[name] = (lm.x, lm.y, lm.z)
                        frame_vis[name] = getattr(lm, 'visibility', 1.0)

                results_list.append(PoseFrame(frame_idx, frame_lms, frame_vis))

            frame_idx += 1

    cap.release()
    return results_list


def get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps":      cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width":    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info
