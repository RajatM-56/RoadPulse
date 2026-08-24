#!/usr/bin/env python3
"""
RoadPulse Phase 3 — VisDrone Dataset Preparation

Downloads the VisDrone-DET dataset, converts annotations to YOLO format,
and curates a ~2,500-image train subset for time-constrained fine-tuning.

VisDrone annotations use a custom CSV-like format:
  <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

object_category mapping (original → our 0-indexed):
  0: ignored regions  (skip)
  1: pedestrian  → 0
  2: people      → 1
  3: bicycle     → 2
  4: car         → 3
  5: van         → 4
  6: truck       → 5
  7: tricycle    → 6
  8: awning-tricycle → 7
  9: bus         → 8
  10: motor      → 9
  11: others     (skip)

Designed for Google Colab.

Usage:
    !pip install tqdm requests
    !python scripts/prepare_visdrone.py \
        --data-dir ./data \
        --n-train 2500
"""

import argparse
import os
import sys
import glob
import random
import zipfile
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


# VisDrone category mapping: original_id → (yolo_id, name)
# Categories 0 (ignored) and 11 (others) are skipped.
VISDRONE_CAT_MAP = {
    1:  (0, "pedestrian"),
    2:  (1, "people"),
    3:  (2, "bicycle"),
    4:  (3, "car"),
    5:  (4, "van"),
    6:  (5, "truck"),
    7:  (6, "tricycle"),
    8:  (7, "awning-tricycle"),
    9:  (8, "bus"),
    10: (9, "motor"),
}

VISDRONE_CLASS_NAMES = {v[0]: v[1] for v in VISDRONE_CAT_MAP.values()}
NUM_CLASSES = 10

# VisDrone-DET download URLs
VISDRONE_URLS = {
    "trainval_images": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "trainval_annotations": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val_images": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
}


def download_file(url: str, dest_path: str, desc: str = "") -> bool:
    """Download a file with progress bar."""
    import urllib.request
    import ssl

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"  [SKIP] Already exists: {os.path.basename(dest_path)}")
        return True

    print(f"  Downloading {desc or os.path.basename(dest_path)} ...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (RoadPulse downloader)"
        })
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            with open(dest_path, "wb") as f:
                downloaded = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"\r    {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
        print(f"\n  ✓ Downloaded {os.path.getsize(dest_path)/(1024*1024):.1f} MB")
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def convert_visdrone_annotation(ann_path: str, img_w: int, img_h: int) -> list:
    """
    Convert a single VisDrone annotation file to YOLO format lines.

    Returns list of strings like "class_id cx cy w h\n".
    """
    lines_out = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue

            bbox_left = float(parts[0])
            bbox_top = float(parts[1])
            bbox_w = float(parts[2])
            bbox_h = float(parts[3])
            # parts[4] = score (unused)
            cat_id = int(parts[5])
            # parts[6] = truncation, parts[7] = occlusion

            # Skip ignored regions and "others"
            if cat_id not in VISDRONE_CAT_MAP:
                continue
            # Skip zero-area boxes
            if bbox_w <= 0 or bbox_h <= 0:
                continue

            yolo_id = VISDRONE_CAT_MAP[cat_id][0]

            # Convert to YOLO normalised format
            cx = (bbox_left + bbox_w / 2) / img_w
            cy = (bbox_top + bbox_h / 2) / img_h
            nw = bbox_w / img_w
            nh = bbox_h / img_h

            # Clamp
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))

            lines_out.append(f"{yolo_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    return lines_out


def convert_split(src_images_dir: str, src_anns_dir: str,
                  dst_images_dir: str, dst_labels_dir: str,
                  indices: list | None = None) -> int:
    """
    Convert a VisDrone split to YOLO format.

    Args:
        src_images_dir: Directory with original VisDrone images.
        src_anns_dir: Directory with original VisDrone annotation .txt files.
        dst_images_dir: Output directory for images.
        dst_labels_dir: Output directory for YOLO label files.
        indices: If provided, only convert these indices (for subsetting).

    Returns:
        Number of images processed.
    """
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(src_images_dir, "*.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(src_images_dir, "*.png")))

    if indices is not None:
        image_files = [image_files[i] for i in indices if i < len(image_files)]

    count = 0
    for img_path in tqdm(image_files, desc="  Converting", unit="img"):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(src_anns_dir, f"{basename}.txt")

        if not os.path.isfile(ann_path):
            continue

        # Get image dimensions
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Convert annotation
        yolo_lines = convert_visdrone_annotation(ann_path, img_w, img_h)

        # Copy image
        dst_img = os.path.join(dst_images_dir, f"{basename}.jpg")
        if not os.path.exists(dst_img):
            shutil.copy2(img_path, dst_img)

        # Write YOLO label
        dst_lbl = os.path.join(dst_labels_dir, f"{basename}.txt")
        with open(dst_lbl, "w") as f:
            f.writelines(yolo_lines)

        count += 1

    return count


def curated_train_indices(src_anns_dir: str, n_samples: int) -> list:
    """
    Stratified sampling across VisDrone classes.
    Scans annotation files and picks indices ensuring class coverage.
    """
    print(f"  Stratified sampling {n_samples} images ...")

    ann_files = sorted(glob.glob(os.path.join(src_anns_dir, "*.txt")))
    cat_to_indices = defaultdict(list)

    for idx, ann_path in enumerate(tqdm(ann_files, desc="  Scanning", unit="ann")):
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    cat_id = int(parts[5])
                    if cat_id in VISDRONE_CAT_MAP:
                        cat_to_indices[cat_id].append(idx)
                        break  # one hit per image is enough for stratification

    selected = set()
    per_class = max(1, n_samples // max(len(cat_to_indices), 1))

    for cid in sorted(cat_to_indices.keys()):
        pool = list(set(cat_to_indices[cid]))
        random.shuffle(pool)
        selected.update(pool[:per_class])

    # Fill remainder
    all_indices = list(range(len(ann_files)))
    if len(selected) < n_samples:
        remaining = list(set(all_indices) - selected)
        random.shuffle(remaining)
        selected.update(remaining[:n_samples - len(selected)])

    result = sorted(selected)[:n_samples]
    print(f"  Selected {len(result)} images across {len(cat_to_indices)} classes")
    return result


def try_ultralytics_download(data_dir: str) -> bool:
    """
    Attempt to use Ultralytics' built-in VisDrone download.
    This downloads the full dataset and auto-converts labels.
    """
    try:
        from ultralytics import YOLO
        from ultralytics.data.utils import check_det_dataset

        print("  Attempting Ultralytics auto-download of VisDrone ...")
        # This triggers the download if not present
        data_dict = check_det_dataset("VisDrone.yaml")
        print(f"  ✓ Ultralytics download succeeded")
        print(f"    Dataset root: {data_dict.get('path', 'unknown')}")
        return True
    except Exception as e:
        print(f"  Ultralytics auto-download failed: {e}")
        print(f"  Falling back to manual download ...")
        return False


def write_data_yaml(data_dir: str, train_dir: str, val_dir: str) -> str:
    """Write the YOLO data.yaml for VisDrone."""
    yaml_path = os.path.join(data_dir, "visdrone", "data.yaml")
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    root = os.path.abspath(os.path.join(data_dir, "visdrone"))

    lines = [
        f"path: {root}",
        f"train: {os.path.relpath(train_dir, root)}",
        f"val: {os.path.relpath(val_dir, root)}",
        f"nc: {NUM_CLASSES}",
        "names:",
    ]
    for cid in range(NUM_CLASSES):
        lines.append(f"  {cid}: {VISDRONE_CLASS_NAMES[cid]}")

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  ✓ data.yaml written: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="RoadPulse Phase 3 — Prepare VisDrone dataset"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root data directory"
    )
    parser.add_argument(
        "--n-train", type=int, default=2500,
        help="Number of train images to use (default: 2500)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--use-ultralytics", action="store_true",
        help="Use Ultralytics built-in VisDrone download (downloads full dataset)"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("  RoadPulse Phase 3 — VisDrone Data Preparation")
    print("=" * 60)

    visdrone_base = os.path.join(args.data_dir, "visdrone")
    os.makedirs(visdrone_base, exist_ok=True)

    # ── Strategy 1: Use Ultralytics auto-download ─────────────────────────
    if args.use_ultralytics:
        if try_ultralytics_download(args.data_dir):
            print("\n  Using Ultralytics-managed VisDrone dataset.")
            print("  data.yaml = VisDrone.yaml (built-in)")
            print("  You can pass 'VisDrone.yaml' directly to finetune_visdrone.py")
            return

    # ── Strategy 2: Manual download + convert ─────────────────────────────
    downloads_dir = os.path.join(visdrone_base, "_downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    # Download train zip
    print("\n" + "=" * 60)
    print("  Step 1/4: Downloading VisDrone-DET train split")
    print("=" * 60)

    train_zip = os.path.join(downloads_dir, "VisDrone2019-DET-train.zip")
    train_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip"
    download_file(train_url, train_zip, "VisDrone train split (~1.8 GB)")

    # Download val zip
    print("\n" + "=" * 60)
    print("  Step 2/4: Downloading VisDrone-DET val split")
    print("=" * 60)

    val_zip = os.path.join(downloads_dir, "VisDrone2019-DET-val.zip")
    val_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"
    download_file(val_url, val_zip, "VisDrone val split (~0.1 GB)")

    # Extract
    for zp, name in [(train_zip, "train"), (val_zip, "val")]:
        extract_dir = os.path.join(downloads_dir, name)
        if os.path.isdir(extract_dir) and len(os.listdir(extract_dir)) > 0:
            print(f"  [SKIP] {name} already extracted")
        elif os.path.isfile(zp):
            print(f"  Extracting {name} ...")
            with zipfile.ZipFile(zp, "r") as z:
                z.extractall(downloads_dir)
            print(f"  ✓ Extracted {name}")

    # Locate extracted directories
    # VisDrone zips typically extract to VisDrone2019-DET-{train,val}/
    train_imgs_src = None
    train_anns_src = None
    val_imgs_src = None
    val_anns_src = None

    for candidate in ["VisDrone2019-DET-train", "train"]:
        p = os.path.join(downloads_dir, candidate)
        if os.path.isdir(p):
            imgs = os.path.join(p, "images")
            anns = os.path.join(p, "annotations")
            if os.path.isdir(imgs) and os.path.isdir(anns):
                train_imgs_src = imgs
                train_anns_src = anns
                break

    for candidate in ["VisDrone2019-DET-val", "val"]:
        p = os.path.join(downloads_dir, candidate)
        if os.path.isdir(p):
            imgs = os.path.join(p, "images")
            anns = os.path.join(p, "annotations")
            if os.path.isdir(imgs) and os.path.isdir(anns):
                val_imgs_src = imgs
                val_anns_src = anns
                break

    if not train_imgs_src or not val_imgs_src:
        print("\n  ERROR: Could not locate extracted VisDrone directories.")
        print("  Expected structure: VisDrone2019-DET-{train,val}/images/ + annotations/")
        print(f"  Downloads dir contents: {os.listdir(downloads_dir)}")
        # List subdirs for debugging
        for d in os.listdir(downloads_dir):
            full = os.path.join(downloads_dir, d)
            if os.path.isdir(full):
                print(f"    {d}/: {os.listdir(full)[:5]}")
        sys.exit(1)

    # ── Convert val split (full) ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 3/4: Converting val split to YOLO format")
    print("=" * 60)

    val_images_dir = os.path.join(visdrone_base, "val", "images")
    val_labels_dir = os.path.join(visdrone_base, "val", "labels")

    existing_val = len(glob.glob(os.path.join(val_images_dir, "*.jpg")))
    if existing_val > 100:
        print(f"  [SKIP] Val already has {existing_val} images")
    else:
        n_val = convert_split(val_imgs_src, val_anns_src, val_images_dir, val_labels_dir)
        print(f"  ✓ Val: {n_val} images converted")

    # ── Convert curated train subset ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Step 4/4: Converting curated train subset ({args.n_train} images)")
    print("=" * 60)

    train_images_dir = os.path.join(visdrone_base, "train_mini", "images")
    train_labels_dir = os.path.join(visdrone_base, "train_mini", "labels")

    existing_train = len(glob.glob(os.path.join(train_images_dir, "*.jpg")))
    if existing_train >= args.n_train * 0.9:
        print(f"  [SKIP] Train subset already has {existing_train} images")
    else:
        indices = curated_train_indices(train_anns_src, args.n_train)
        n_train = convert_split(
            train_imgs_src, train_anns_src,
            train_images_dir, train_labels_dir,
            indices=indices,
        )
        print(f"  ✓ Train subset: {n_train} images converted")

    # ── Write data.yaml ───────────────────────────────────────────────────
    yaml_path = write_data_yaml(
        args.data_dir,
        train_dir=os.path.join(visdrone_base, "train_mini", "images"),
        val_dir=os.path.join(visdrone_base, "val", "images"),
    )

    # ── Summary ───────────────────────────────────────────────────────────
    n_train_final = len(glob.glob(os.path.join(train_images_dir, "*.jpg")))
    n_val_final = len(glob.glob(os.path.join(val_images_dir, "*.jpg")))

    print("\n" + "=" * 60)
    print("  VisDrone Preparation Complete")
    print("=" * 60)
    print(f"  Train images: {n_train_final}")
    print(f"  Val images:   {n_val_final}")
    print(f"  Classes:      {NUM_CLASSES}")
    print(f"  data.yaml:    {yaml_path}")
    print(f"\n  Next: python scripts/finetune_visdrone.py --data-yaml {yaml_path}")
    print()


if __name__ == "__main__":
    main()
