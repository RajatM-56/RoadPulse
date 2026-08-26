import os
import json
import cv2
import numpy as np
import pytest

from pipeline.utils import draw_detection, VideoWriter, get_video_clips
from pipeline.detect import process_clip, load_model


def test_corrupted_video_handling(tmp_path):
    """Test process_clip gracefully handles unopenable/corrupted video files."""
    corrupt_video = str(tmp_path / "corrupted.mp4")
    with open(corrupt_video, "wb") as f:
        f.write(b"NOT_A_VALID_MP4_HEADER_DATA")

    model = load_model("yolo26n.pt")
    out_dir = str(tmp_path / "corrupt_out")

    # Should return None and not crash
    result = process_clip(model, corrupt_video, out_dir)
    assert result is None


def test_zero_detections_clip(tmp_path):
    """Test process_clip handles video with no vehicles (e.g. blank frames) without error."""
    blank_video = str(tmp_path / "blank.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(blank_video, fourcc, 10.0, (320, 240))
    for _ in range(5):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()

    model = load_model("yolo26n.pt")
    out_dir = str(tmp_path / "blank_out")

    summary = process_clip(model, blank_video, out_dir, conf_thresh=0.5)
    assert summary is not None
    assert summary["total_detections_vehicle"] == 0
    assert summary["json_entries"] == 0
    assert os.path.exists(summary["output_video"])
    assert os.path.exists(summary["output_json"])

    with open(summary["output_json"]) as f:
        detections = json.load(f)
        assert detections == []


def test_extreme_bounding_boxes():
    """Test draw_detection handles out-of-bounds coordinates without crashing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Bounding box exceeding image bounds or negative coordinates
    bbox_out_of_bounds = [-50, -20, 800, 600]
    res_frame = draw_detection(frame, bbox_out_of_bounds, class_id=2, confidence=0.9)
    assert res_frame is not None
    assert res_frame.shape == (480, 640, 3)


def test_odd_aspect_ratio_video(tmp_path):
    """Test process_clip handles odd aspect ratios (e.g. 1920x300 panoramic or vertical 400x800)."""
    vert_video = str(tmp_path / "vertical.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(vert_video, fourcc, 15.0, (360, 640))
    for _ in range(5):
        frame = np.full((640, 360, 3), 120, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    model = load_model("yolo26n.pt")
    out_dir = str(tmp_path / "vert_out")

    summary = process_clip(model, vert_video, out_dir)
    assert summary is not None
    assert summary["total_frames"] == 5
    assert os.path.exists(summary["output_video"])
