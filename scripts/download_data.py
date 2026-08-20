#!/usr/bin/env python3
"""
RoadPulse Phase 1 — Dataset Download Script

Downloads sample clips and dataset subsets needed for each phase.
Phase 1 only needs --sample-clips. Other sub-commands are scaffolded
for future phases.

Usage:
    # Phase 1 — download sample video clips for smoke testing
    python scripts/download_data.py --sample-clips --data-dir ./data

    # Phase 2+ — download BMD-45 val split
    python scripts/download_data.py --bmd45-val --data-dir ./data

    # Phase 2+ — download small BMD-45 train subset
    python scripts/download_data.py --bmd45-train-mini --data-dir ./data
"""

import argparse
import os
import sys
import urllib.request
import ssl
import json


# ---------------------------------------------------------------------------
# Sample clip sources — public domain / CC-licensed traffic videos
# These are fallback URLs; the Colab walkthrough provides primary instructions.
# ---------------------------------------------------------------------------

SAMPLE_CLIPS = {
    "fixed_cam_sample.mp4": {
        "description": "Fixed traffic camera clip (public domain)",
        # Public traffic camera recording — direct download
        "urls": [
            # Primary: PETS2009 / MOT benchmark sample
            "https://motchallenge.net/data/MOT17-04-raw.webm",
        ],
    },
    "drone_sample.mp4": {
        "description": "Drone / UAV traffic clip (VisDrone-like)",
        "urls": [
            # Primary: VisDrone sample
            "https://github.com/VisDrone/VisDrone-Dataset/raw/master/demo.mp4",
        ],
    },
}


def download_file(url, dest_path, desc=""):
    """Download a file from URL to dest_path with progress."""
    print(f"  Downloading {desc or url}")
    print(f"  → {dest_path}")

    # Create SSL context that doesn't verify (for Colab compatibility)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (RoadPulse downloader)"
        })
        with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None

            with open(dest_path, "wb") as f:
                downloaded = 0
                block_size = 8192
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

        file_size = os.path.getsize(dest_path)
        print(f"\n  ✓ Downloaded {file_size / (1024*1024):.1f} MB")
        return True

    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def generate_synthetic_clip(dest_path, clip_type="fixed"):
    """
    Generate a short synthetic test clip using OpenCV.
    Not ideal (no real vehicles), but guarantees the pipeline can run end-to-end.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  ✗ OpenCV not available for synthetic clip generation.")
        return False

    print(f"  Generating synthetic {clip_type} clip ...")

    width, height, fps, duration = 1280, 720, 25, 5  # 5 second clip
    total_frames = fps * duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dest_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("  ✗ Could not create video writer.")
        return False

    # Simulate vehicles as colored rectangles moving across a road
    np.random.seed(42)
    n_vehicles = 8
    vehicles = []
    for i in range(n_vehicles):
        x = np.random.randint(0, width - 80)
        y = np.random.randint(height // 4, height - 60)
        w = np.random.randint(40, 100)
        h = np.random.randint(30, 60)
        dx = np.random.choice([-3, -2, 2, 3])
        dy = np.random.choice([-1, 0, 1])
        color = tuple(int(c) for c in np.random.randint(100, 255, 3))
        vehicles.append({"x": x, "y": y, "w": w, "h": h,
                         "dx": dx, "dy": dy, "color": color})

    for frame_idx in range(total_frames):
        # Gray road background
        frame = np.full((height, width, 3), (80, 80, 80), dtype=np.uint8)

        # Road markings
        for lane_y in range(height // 4, height, height // 6):
            cv2.line(frame, (0, lane_y), (width, lane_y),
                     (120, 120, 120), 1)

        # Dashed center line
        for x_start in range(0, width, 60):
            offset = (frame_idx * 3) % 60
            x = (x_start + offset) % width
            cv2.line(frame, (x, height // 2), (x + 25, height // 2),
                     (200, 200, 200), 2)

        # Draw and move vehicles
        for v in vehicles:
            cv2.rectangle(frame,
                          (int(v["x"]), int(v["y"])),
                          (int(v["x"] + v["w"]), int(v["y"] + v["h"])),
                          v["color"], -1)
            # Windshield
            cv2.rectangle(frame,
                          (int(v["x"] + 5), int(v["y"] + 3)),
                          (int(v["x"] + v["w"] - 5), int(v["y"] + v["h"] // 3)),
                          (180, 220, 240), -1)

            v["x"] = (v["x"] + v["dx"]) % (width - v["w"])
            v["y"] = max(height // 4,
                         min(height - v["h"], v["y"] + v["dy"]))

        # Label
        label = f"SYNTHETIC TEST - {clip_type.upper()} CAM"
        cv2.putText(frame, label, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames}",
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(frame)

    writer.release()
    file_size = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"  ✓ Synthetic clip generated: {file_size:.1f} MB ({total_frames} frames)")
    print(f"  NOTE: Synthetic clip has no real vehicles — YOLO will detect few/no objects.")
    print(f"        Use real clips for the actual Phase 1 demo. See walkthrough.")
    return True


def download_sample_clips(data_dir):
    """Download or generate sample video clips for smoke testing."""
    clips_dir = os.path.join(data_dir, "sample_clips")
    os.makedirs(clips_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Downloading sample video clips")
    print("=" * 60)

    results = {}
    for filename, info in SAMPLE_CLIPS.items():
        dest_path = os.path.join(clips_dir, filename)

        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
            print(f"\n[SKIP] {filename} already exists ({os.path.getsize(dest_path) / (1024*1024):.1f} MB)")
            results[filename] = True
            continue

        print(f"\n[...] {info['description']}")
        success = False
        for url in info["urls"]:
            if download_file(url, dest_path, info["description"]):
                success = True
                break

        if not success:
            print(f"\n  Auto-download failed for {filename}.")
            print(f"  Generating synthetic fallback clip ...")
            clip_type = "fixed" if "fixed" in filename else "drone"
            success = generate_synthetic_clip(dest_path, clip_type)

        results[filename] = success

    # Summary
    print("\n" + "-" * 60)
    all_ok = all(results.values())
    for fname, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {fname}")

    if not all_ok:
        print("\n  Some clips could not be downloaded or generated.")
        print("  You can manually place .mp4 files in:")
        print(f"    {clips_dir}/")

    print(f"\n  Clips directory: {clips_dir}")

    if any("SYNTHETIC" in open(os.path.join(clips_dir, f), "rb").read(200).decode("utf-8", errors="ignore")
           for f in results if results[f] and os.path.exists(os.path.join(clips_dir, f))):
        pass  # synthetic clips present, already warned above

    return all_ok


def download_bmd45_val(data_dir):
    """Download BMD-45 val split from HuggingFace."""
    print("\n" + "=" * 60)
    print("  Downloading BMD-45 validation split")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ✗ 'datasets' package not installed.")
        print("    Run: pip install datasets")
        return False

    bmd45_dir = os.path.join(data_dir, "bmd45", "val")
    os.makedirs(bmd45_dir, exist_ok=True)

    print(f"  Saving to: {bmd45_dir}")
    print("  This downloads ~5 GB. Be patient ...")

    try:
        ds = load_dataset("iisc-aim/BMD-45", split="val")
        print(f"  ✓ Loaded {len(ds)} val images from HuggingFace")

        # Save images and annotations
        images_dir = os.path.join(bmd45_dir, "images")
        labels_dir = os.path.join(bmd45_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for idx, sample in enumerate(ds):
            # Save image
            img = sample["image"]
            img_path = os.path.join(images_dir, f"{idx:06d}.jpg")
            img.save(img_path)

            # Save YOLO-format label
            objects = sample["objects"]
            label_path = os.path.join(labels_dir, f"{idx:06d}.txt")
            img_w, img_h = img.size

            with open(label_path, "w") as f:
                bboxes = objects["bbox"]
                categories = objects["categories"]
                for bbox, cat in zip(bboxes, categories):
                    # BMD-45 bbox format: [x_min, y_min, width, height] (COCO-style)
                    x_min, y_min, w, h = bbox
                    # Convert to YOLO format: [class cx cy w h] (normalized)
                    cx = (x_min + w / 2) / img_w
                    cy = (y_min + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    f.write(f"{cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            if (idx + 1) % 500 == 0:
                print(f"  Saved {idx + 1}/{len(ds)} ...")

        print(f"  ✓ BMD-45 val split saved: {len(ds)} images + labels")
        return True

    except Exception as e:
        print(f"  ✗ Failed to download BMD-45: {e}")
        return False


def download_bmd45_train_mini(data_dir, n_samples=500):
    """Download a small subset of BMD-45 train split for smoke testing."""
    print("\n" + "=" * 60)
    print(f"  Downloading BMD-45 train mini ({n_samples} images)")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ✗ 'datasets' package not installed.")
        return False

    bmd45_dir = os.path.join(data_dir, "bmd45", "train_mini")
    os.makedirs(bmd45_dir, exist_ok=True)

    try:
        ds = load_dataset("iisc-aim/BMD-45", split=f"train[:{n_samples}]")
        print(f"  ✓ Loaded {len(ds)} train images")

        images_dir = os.path.join(bmd45_dir, "images")
        labels_dir = os.path.join(bmd45_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for idx, sample in enumerate(ds):
            img = sample["image"]
            img_path = os.path.join(images_dir, f"{idx:06d}.jpg")
            img.save(img_path)

            objects = sample["objects"]
            label_path = os.path.join(labels_dir, f"{idx:06d}.txt")
            img_w, img_h = img.size

            with open(label_path, "w") as f:
                bboxes = objects["bbox"]
                categories = objects["categories"]
                for bbox, cat in zip(bboxes, categories):
                    x_min, y_min, w, h = bbox
                    cx = (x_min + w / 2) / img_w
                    cy = (y_min + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    f.write(f"{cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            if (idx + 1) % 100 == 0:
                print(f"  Saved {idx + 1}/{len(ds)} ...")

        print(f"  ✓ BMD-45 train mini saved: {len(ds)} images + labels")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="RoadPulse dataset download script"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Root data directory (e.g., ./data or "
                             "/content/drive/MyDrive/roadpulse/data)")
    parser.add_argument("--sample-clips", action="store_true",
                        help="Download sample video clips for smoke testing")
    parser.add_argument("--bmd45-val", action="store_true",
                        help="Download BMD-45 val split (10,194 images, ~5 GB)")
    parser.add_argument("--bmd45-train-mini", action="store_true",
                        help="Download BMD-45 train mini (500 images)")
    parser.add_argument("--n-train-samples", type=int, default=500,
                        help="Number of train samples for --bmd45-train-mini")

    args = parser.parse_args()

    # Must specify at least one download target
    if not any([args.sample_clips, args.bmd45_val, args.bmd45_train_mini]):
        parser.print_help()
        print("\nERROR: Specify at least one download target.")
        sys.exit(1)

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Data directory: {data_dir}")

    if args.sample_clips:
        download_sample_clips(data_dir)

    if args.bmd45_val:
        download_bmd45_val(data_dir)

    if args.bmd45_train_mini:
        download_bmd45_train_mini(data_dir, args.n_train_samples)

    print("\nDone.")


if __name__ == "__main__":
    main()
