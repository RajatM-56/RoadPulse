#!/usr/bin/env python3
"""
RoadPulse Phase 1 — Environment Setup & Validation

Checks Python version, GPU availability, VRAM, installs dependencies,
and verifies YOLO model loading. Run this first on any new environment.

Usage:
    python scripts/setup_env.py
"""

import sys
import subprocess
import importlib
import os


def check_python():
    """Verify Python >= 3.9."""
    v = sys.version_info
    status = "OK" if (v.major, v.minor) >= (3, 9) else "FAIL"
    print(f"[{status}] Python version: {v.major}.{v.minor}.{v.micro}")
    if status == "FAIL":
        print("  ERROR: Python >= 3.9 required. Colab should have this by default.")
        sys.exit(1)


def install_requirements():
    """Install requirements.txt if not already satisfied."""
    req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    req_path = os.path.abspath(req_path)

    if not os.path.exists(req_path):
        print("[WARN] requirements.txt not found, skipping pip install.")
        return

    print("[...] Installing / verifying dependencies from requirements.txt ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", req_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[WARN] pip install had issues:\n{result.stderr}")
    else:
        print("[OK]  Dependencies installed / verified.")


def check_gpu():
    """Check GPU availability and VRAM."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            status = "OK" if vram_gb >= 6.0 else "WARN"
            print(f"[{status}] GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
            if vram_gb < 6.0:
                print("  WARNING: < 6 GB VRAM. Training may OOM; inference should be fine.")
        else:
            print("[WARN] No CUDA GPU detected. Will run on CPU (slow but functional).")
            print("  If on Colab: Runtime → Change runtime type → GPU (T4 or better).")
    except ImportError:
        print("[WARN] PyTorch not installed. GPU check skipped.")
        print("  On Colab, torch is pre-installed. Locally, install via pytorch.org.")


def check_yolo():
    """Verify YOLO model can be loaded. Try YOLO26n first, fallback to YOLO11n."""
    try:
        from ultralytics import YOLO

        # Try YOLO26n first
        print("[...] Loading YOLO26n (yolo26n.pt) ...")
        try:
            model = YOLO("yolo26n.pt")
            print(f"[OK]  YOLO26n loaded successfully.")
            print(f"      Model type: {model.type}")
            print(f"      Task: {model.task}")
            return "yolo26n"
        except Exception as e:
            print(f"[WARN] YOLO26n failed to load: {e}")
            print("[...] Falling back to YOLO11n (yolo11n.pt) ...")

        # Fallback to YOLO11n
        try:
            model = YOLO("yolo11n.pt")
            print(f"[OK]  YOLO11n loaded successfully (fallback).")
            print(f"      Model type: {model.type}")
            print(f"      Task: {model.task}")
            return "yolo11n"
        except Exception as e:
            print(f"[FAIL] YOLO11n also failed: {e}")
            print("  Try: pip install -U ultralytics")
            return None

    except ImportError:
        print("[FAIL] ultralytics not installed.")
        print("  Run: pip install ultralytics")
        return None


def check_opencv():
    """Verify OpenCV is available."""
    try:
        import cv2
        print(f"[OK]  OpenCV version: {cv2.__version__}")
    except ImportError:
        print("[WARN] OpenCV not found. Installing ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "opencv-python-headless"], check=True)
        import cv2
        print(f"[OK]  OpenCV installed: {cv2.__version__}")


def main():
    print("=" * 60)
    print("  RoadPulse — Environment Setup & Validation")
    print("=" * 60)
    print()

    check_python()
    print()

    install_requirements()
    print()

    check_gpu()
    print()

    check_opencv()
    print()

    model_name = check_yolo()
    print()

    # Summary
    print("=" * 60)
    print("  Setup Summary")
    print("=" * 60)
    if model_name:
        print(f"  ✓ Ready to run inference with {model_name}.pt")
        print(f"  Next step: python scripts/download_data.py --sample-clips --data-dir ./data")
    else:
        print("  ✗ YOLO model could not be loaded. Fix the issues above first.")
    print()


if __name__ == "__main__":
    main()
