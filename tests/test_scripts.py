import os
import sys
import subprocess
import cv2
import pytest
from unittest.mock import patch, MagicMock

import scripts.setup_env as setup_env
import scripts.download_data as download_data


def test_setup_env_functions():
    """Verify setup_env diagnostic functions execute without throwing unhandled exceptions."""
    setup_env.check_python()
    setup_env.check_gpu()
    setup_env.check_opencv()

    model_name = setup_env.check_yolo()
    assert model_name in ["yolo26n", "yolo11n"]


def test_setup_env_main_direct():
    """Test setup_env.main directly."""
    with patch.object(setup_env, "install_requirements"):
        setup_env.main()


def test_generate_synthetic_clip(tmp_path):
    """Verify generate_synthetic_clip generates valid, readable video."""
    clip_path = str(tmp_path / "synthetic_test.mp4")
    success = download_data.generate_synthetic_clip(clip_path, clip_type="fixed")
    assert success is True
    assert os.path.exists(clip_path)

    cap = cv2.VideoCapture(clip_path)
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1280
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 720
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 125
    cap.release()


def test_download_sample_clips(tmp_path):
    """Verify download_sample_clips downloads or synthesizes clips."""
    data_dir = str(tmp_path / "data")
    success = download_data.download_sample_clips(data_dir)
    assert success is True
    clips_dir = os.path.join(data_dir, "sample_clips")
    assert os.path.exists(os.path.join(clips_dir, "fixed_cam_sample.mp4"))
    assert os.path.exists(os.path.join(clips_dir, "drone_sample.mp4"))


def test_download_data_main_cli_sample_clips(tmp_path):
    """Test download_data.main() with --sample-clips."""
    data_dir = str(tmp_path / "cli_data")
    test_args = [
        "download_data.py",
        "--data-dir", data_dir,
        "--sample-clips",
    ]
    with patch.object(sys, "argv", test_args):
        download_data.main()

    clips_dir = os.path.join(data_dir, "sample_clips")
    assert os.path.exists(os.path.join(clips_dir, "fixed_cam_sample.mp4"))


def test_download_data_cli_no_args():
    """Verify download_data.main errors when no targets are specified."""
    test_args = [
        "download_data.py",
        "--data-dir", "./data",
    ]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            download_data.main()
        assert exc_info.value.code == 1
