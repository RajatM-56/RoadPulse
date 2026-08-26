import os
import sys
import pytest
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture providing a temporary directory Path object."""
    return tmp_path


@pytest.fixture
def sample_image_with_vehicles():
    """Load ultralytics bus.jpg or create a high-contrast mock image with vehicle shapes."""
    try:
        import ultralytics
        assets_path = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
        if os.path.exists(assets_path):
            img = cv2.imread(assets_path)
            if img is not None:
                return img
    except Exception:
        pass

    # Fallback to generated image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple vehicle shape
    cv2.rectangle(img, (100, 200), (300, 350), (200, 200, 200), -1)
    return img


@pytest.fixture
def sample_video_clip(tmp_path, sample_image_with_vehicles):
    """Generate a short 10-frame video clip containing real/detectable vehicles."""
    video_path = str(tmp_path / "test_clip.mp4")
    h, w, _ = sample_image_with_vehicles.shape
    fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for _ in range(10):
        writer.write(sample_image_with_vehicles)
    writer.release()

    return video_path
