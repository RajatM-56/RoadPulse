import os
import json
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from pipeline.utils import (
    COCO_VEHICLE_CLASS_IDS,
    COCO_NAMES,
    BOX_COLOURS,
    DEFAULT_BOX_COLOUR,
    is_vehicle_class,
    get_video_clips,
    VideoWriter,
    save_detections_json,
    load_detections_json,
    draw_detection,
)


def test_coco_vehicle_classes():
    """Verify that COCO vehicle classes are properly defined and recognized."""
    expected_ids = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    assert COCO_VEHICLE_CLASS_IDS == expected_ids

    for cid in expected_ids:
        assert is_vehicle_class(cid) is True

    # Non-vehicle classes
    assert is_vehicle_class(0) is False  # person
    assert is_vehicle_class(4) is False  # airplane
    assert is_vehicle_class(8) is False  # boat
    assert is_vehicle_class(999) is False  # invalid class


def test_get_video_clips(tmp_path):
    """Test video file discovery with supported extensions and directory validation."""
    assert get_video_clips(str(tmp_path / "non_existent")) == []

    # Create dummy files
    (tmp_path / "clip1.mp4").write_bytes(b"dummy")
    (tmp_path / "clip2.avi").write_bytes(b"dummy")
    (tmp_path / "clip3.mkv").write_bytes(b"dummy")
    (tmp_path / "clip4.mov").write_bytes(b"dummy")
    (tmp_path / "clip5.webm").write_bytes(b"dummy")
    (tmp_path / "not_a_video.txt").write_text("hello")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.mp4").write_bytes(b"dummy")

    clips = get_video_clips(str(tmp_path))
    filenames = [os.path.basename(c) for c in clips]

    assert len(clips) == 5
    assert "clip1.mp4" in filenames
    assert "clip2.avi" in filenames
    assert "clip3.mkv" in filenames
    assert "clip4.mov" in filenames
    assert "clip5.webm" in filenames
    assert "not_a_video.txt" not in filenames
    assert "nested.mp4" not in filenames


def test_video_writer(tmp_path):
    """Test VideoWriter class initialization, writing, and context manager."""
    out_path = str(tmp_path / "nested_dir" / "output.mp4")
    w, h, fps = 640, 480, 25.0

    with VideoWriter(out_path, fps, w, h) as writer:
        assert writer.frame_count == 0
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(frame)
        writer.write(frame)
        assert writer.frame_count == 2

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0

    cap = cv2.VideoCapture(out_path)
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == h
    cap.release()


def test_video_writer_failure(tmp_path):
    """Test VideoWriter raises RuntimeError if cv2 fails to open writer."""
    with patch("cv2.VideoWriter") as mock_writer_cls:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_writer_cls.return_value = mock_instance

        with pytest.raises(RuntimeError) as exc_info:
            VideoWriter(str(tmp_path / "test.mp4"), 25.0, 640, 480)
        assert "Failed to open VideoWriter" in str(exc_info.value)


def test_video_writer_from_capture(tmp_path, sample_video_clip):
    """Test VideoWriter.from_capture helper."""
    cap = cv2.VideoCapture(sample_video_clip)
    assert cap.isOpened()

    out_path = str(tmp_path / "copy.mp4")
    writer = VideoWriter.from_capture(cap, out_path)
    
    ret, frame = cap.read()
    assert ret is True
    writer.write(frame)
    writer.release()
    cap.release()

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_json_serialization(tmp_path):
    """Test saving and loading detections JSON."""
    detections = [
        {
            "frame_idx": 0,
            "bbox_xyxy": [10.5, 20.0, 100.2, 200.8],
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.8951,
        },
        {
            "frame_idx": 1,
            "bbox_xyxy": [15.0, 25.0, 105.0, 205.0],
            "class_id": 5,
            "class_name": "bus",
            "confidence": 0.9421,
        }
    ]

    json_path = str(tmp_path / "nested" / "detections.json")
    save_detections_json(detections, json_path)
    assert os.path.exists(json_path)

    loaded = load_detections_json(json_path)
    assert loaded == detections


def test_draw_detection():
    """Test bounding box and label rendering on frames."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [50, 50, 200, 150]
    
    # 1. Vehicle detection without track_id
    res_frame = draw_detection(frame.copy(), bbox, class_id=2, confidence=0.85)
    assert res_frame is not None
    assert np.any(res_frame > 0)

    # 2. Detection with track_id (Phase 2 format)
    res_frame2 = draw_detection(frame.copy(), bbox, class_id=5, confidence=0.91, track_id=42)
    assert res_frame2 is not None
    assert np.any(res_frame2 > 0)

    # 3. Non-vehicle class using default colour
    res_frame3 = draw_detection(frame.copy(), bbox, class_id=0, confidence=0.75, class_name="person")
    assert res_frame3 is not None
    assert np.any(res_frame3 > 0)
