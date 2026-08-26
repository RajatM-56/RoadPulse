import os
import json
import subprocess
import sys
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from pipeline.detect import load_model, process_clip, main


def test_load_model():
    """Verify load_model successfully loads yolo26n or falls back to yolo11n."""
    model = load_model("yolo26n.pt")
    assert model is not None
    assert hasattr(model, "predict")


def test_load_model_fallback():
    """Verify fallback mechanism when invalid model specified."""
    with patch("ultralytics.YOLO") as mock_yolo:
        mock_yolo.side_effect = [Exception("NotFound"), MagicMock()]
        model = load_model("yolo26n_nonexistent.pt")
        assert model is not None
        assert mock_yolo.call_count == 2
        mock_yolo.assert_called_with("yolo11n.pt")


def test_load_model_custom_failure():
    """Verify non-yolo26n model failure raises exception."""
    with patch("ultralytics.YOLO") as mock_yolo:
        mock_yolo.side_effect = Exception("ModelNotFound")
        with pytest.raises(Exception):
            load_model("custom_invalid_model.pt")


def test_process_clip_with_vehicles(sample_video_clip, tmp_path):
    """Test process_clip function detecting vehicles in a clip."""
    model = load_model("yolo26n.pt")
    out_dir = str(tmp_path / "out")

    # Run vehicle_only mode (default)
    summary = process_clip(
        model=model,
        clip_path=sample_video_clip,
        out_dir=out_dir,
        conf_thresh=0.25,
        imgsz=640,
        vehicle_only=True,
        show_all_classes=False,
    )

    assert summary is not None
    assert summary["clip_name"] == "test_clip"
    assert summary["total_frames"] == 10
    assert summary["total_detections_vehicle"] > 0
    assert summary["total_detections_vehicle"] == summary["json_entries"]
    assert os.path.exists(summary["output_video"])
    assert os.path.exists(summary["output_json"])

    # Verify JSON content structure
    with open(summary["output_json"]) as f:
        detections = json.load(f)
        assert len(detections) > 0
        for det in detections:
            assert "frame_idx" in det
            assert "bbox_xyxy" in det
            assert len(det["bbox_xyxy"]) == 4
            assert "class_id" in det
            assert "class_name" in det
            assert "confidence" in det
            assert det["confidence"] >= 0.25
            assert det["class_id"] in [1, 2, 3, 5, 7]  # vehicle classes


def test_process_clip_all_classes(sample_video_clip, tmp_path):
    """Test process_clip with vehicle_only=False to ensure all classes are captured in JSON."""
    model = load_model("yolo26n.pt")
    out_dir = str(tmp_path / "out_all")

    summary = process_clip(
        model=model,
        clip_path=sample_video_clip,
        out_dir=out_dir,
        conf_thresh=0.25,
        imgsz=640,
        vehicle_only=False,
        show_all_classes=True,
    )

    assert summary is not None
    assert summary["json_entries"] == summary["total_detections_all"]
    assert summary["total_detections_all"] >= summary["total_detections_vehicle"]

    with open(summary["output_json"]) as f:
        detections = json.load(f)
        classes = {d["class_name"] for d in detections}
        assert "bus" in classes or "car" in classes


def test_process_clip_progress_logging(tmp_path):
    """Test process_clip with 105 frames to trigger frame_idx % 100 logging branch."""
    video_path = str(tmp_path / "long_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (160, 120))
    dummy_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for _ in range(105):
        writer.write(dummy_frame)
    writer.release()

    model = load_model("yolo26n.pt")
    out_dir = str(tmp_path / "long_out")
    summary = process_clip(model, video_path, out_dir, conf_thresh=0.5, imgsz=160)
    assert summary["total_frames"] == 105


def test_process_clip_confidence_filtering(sample_video_clip, tmp_path):
    """Test that higher confidence threshold filters out lower confidence detections."""
    model = load_model("yolo26n.pt")
    
    # Low confidence threshold
    out_dir_low = str(tmp_path / "out_low")
    summary_low = process_clip(
        model, sample_video_clip, out_dir_low, conf_thresh=0.1
    )

    # High confidence threshold
    out_dir_high = str(tmp_path / "out_high")
    summary_high = process_clip(
        model, sample_video_clip, out_dir_high, conf_thresh=0.9
    )

    assert summary_low["total_detections_all"] >= summary_high["total_detections_all"]


def test_detect_main_direct(sample_video_clip, tmp_path):
    """Test main() directly in Python process with multiple arguments."""
    data_dir = os.path.dirname(sample_video_clip)
    out_dir = str(tmp_path / "direct_main_out")

    test_args = [
        "detect.py",
        "--data-dir", data_dir,
        "--out-dir", out_dir,
        "--conf-thresh", "0.25",
        "--imgsz", "320",
        "--all-classes",
        "--show-all",
    ]

    with patch.object(sys, "argv", test_args):
        main()

    summary_file = os.path.join(out_dir, "batch_summary.json")
    assert os.path.exists(summary_file)
    with open(summary_file) as f:
        data = json.load(f)
        assert len(data) >= 1


def test_detect_main_empty_dir_exit(tmp_path):
    """Test main() calls sys.exit(1) when no clips found."""
    empty_dir = str(tmp_path / "empty_dir")
    os.makedirs(empty_dir, exist_ok=True)
    out_dir = str(tmp_path / "empty_out")

    test_args = [
        "detect.py",
        "--data-dir", empty_dir,
        "--out-dir", out_dir,
    ]

    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
