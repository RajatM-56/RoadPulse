#!/usr/bin/env python3
"""
RoadPulse — Pipeline Utilities

Shared helpers for video I/O, JSON serialisation, and COCO class mapping.
Used by detect.py (Phase 1), and will be extended for tracking (Phase 2+).
"""

import os
import json
import glob
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# COCO class mapping — vehicle classes we care about for traffic detection
# These are the COCO class IDs (0-indexed) that correspond to vehicles.
# ---------------------------------------------------------------------------

COCO_VEHICLE_CLASS_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Full COCO class names (for reference / display)
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

# Colour palette for drawing — distinct colours per class
BOX_COLOURS = {
    1: (0, 255, 127),    # bicycle — spring green
    2: (255, 200, 0),    # car — gold
    3: (0, 165, 255),    # motorcycle — orange
    5: (255, 0, 100),    # bus — deep pink
    7: (100, 100, 255),  # truck — coral blue
}
DEFAULT_BOX_COLOUR = (200, 200, 200)  # fallback grey


# ---------------------------------------------------------------------------
# Video file discovery
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}


def get_video_clips(directory: str) -> List[str]:
    """
    Find all video files in a directory (non-recursive).

    Args:
        directory: Path to folder containing video clips.

    Returns:
        Sorted list of absolute paths to video files.
    """
    clips = []
    if not os.path.isdir(directory):
        print(f"  WARNING: Directory not found: {directory}")
        return clips

    for entry in sorted(os.listdir(directory)):
        ext = os.path.splitext(entry)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            clips.append(os.path.join(directory, entry))

    return clips


# ---------------------------------------------------------------------------
# Video writer wrapper
# ---------------------------------------------------------------------------

class VideoWriter:
    """
    Thin wrapper around cv2.VideoWriter that auto-configures from a source video.
    Handles codec, fps, and resolution matching.
    """

    def __init__(self, output_path: str, fps: float, width: int, height: int,
                 codec: str = "mp4v"):
        """
        Args:
            output_path: Path for the output video file.
            fps: Frames per second.
            width: Frame width in pixels.
            height: Frame height in pixels.
            codec: FourCC codec string (default: mp4v for .mp4 files).
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.output_path = output_path
        self.frame_count = 0

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at {output_path}")

    @classmethod
    def from_capture(cls, cap: cv2.VideoCapture, output_path: str,
                     codec: str = "mp4v") -> "VideoWriter":
        """Create a VideoWriter matching a source cv2.VideoCapture."""
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cls(output_path, fps, width, height, codec)

    def write(self, frame: np.ndarray):
        """Write a single frame."""
        self.writer.write(frame)
        self.frame_count += 1

    def release(self):
        """Finalise and close the video file."""
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


# ---------------------------------------------------------------------------
# Detection JSON I/O
# ---------------------------------------------------------------------------

def save_detections_json(detections: List[Dict[str, Any]], path: str):
    """
    Save a list of detection dicts to a JSON file.

    Args:
        detections: List of dicts, each with keys like frame_idx, bbox_xyxy, etc.
        path: Output JSON file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(detections, f, indent=2)
    print(f"  Saved {len(detections)} detections → {path}")


def load_detections_json(path: str) -> List[Dict[str, Any]]:
    """Load detections from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_detection(frame: np.ndarray, bbox_xyxy: List[float],
                   class_id: int, confidence: float,
                   class_name: Optional[str] = None,
                   track_id: Optional[int] = None) -> np.ndarray:
    """
    Draw a single detection box + label on a frame.

    Args:
        frame: BGR image (modified in-place).
        bbox_xyxy: [x1, y1, x2, y2] bounding box coordinates.
        class_id: COCO class ID.
        confidence: Detection confidence [0, 1].
        class_name: Optional override for class label text.
        track_id: Optional track ID (for Phase 2+).

    Returns:
        The frame with the detection drawn on it.
    """
    x1, y1, x2, y2 = [int(c) for c in bbox_xyxy]
    colour = BOX_COLOURS.get(class_id, DEFAULT_BOX_COLOUR)
    name = class_name or COCO_NAMES.get(class_id, f"cls_{class_id}")

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Label text
    if track_id is not None:
        label = f"ID:{track_id} {name} {confidence:.2f}"
    else:
        label = f"{name} {confidence:.2f}"

    # Label background
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                  colour, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)

    return frame


def is_vehicle_class(class_id: int) -> bool:
    """Check if a COCO class ID is a vehicle."""
    return class_id in COCO_VEHICLE_CLASS_IDS
