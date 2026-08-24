#!/usr/bin/env python3
"""
RoadPulse Phase 2 — BMD-45 Dataset Preparation

Downloads a curated train subset and full val split of BMD-45 from HuggingFace,
converts annotations to YOLO format, and writes the data.yaml config.

Designed for Google Colab.  Run once before fine-tuning.

Usage (Colab):
    !pip install datasets huggingface_hub Pillow tqdm
    !python scripts/prepare_bmd45.py \
        --data-dir ./data \
        --n-train 2000

Usage (local):
    python scripts/prepare_bmd45.py --data-dir ./data --n-train 2000
"""

import argparse
import os
import json
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# BMD-45 class mapping (0-indexed)
# The dataset stores integer category IDs in objects["categories"].
# We discover the full list at download time and fall back to this default
# if the dataset doesn't expose a formal taxonomy.
# ---------------------------------------------------------------------------

BMD45_DEFAULT_NAMES = {
    0: "hatchback",
    1: "sedan",
    2: "SUV",
    3: "bus",
    4: "truck",
    5: "three-wheeler",
    6: "two-wheeler",
    7: "LCV",
    8: "mini-bus",
    9: "tempo-traveller",
    10: "multi-axle-truck",
    11: "tractor",
    12: "motorcycle-rickshaw",
    13: "bicycle",
}


def discover_classes(dataset_split, max_scan: int = 5000) -> dict:
    """
    Scan a portion of the dataset to discover all unique category IDs.
    Returns a dict mapping int → class_name.  Uses the default mapping
    where possible; unknown IDs get placeholder names.
    """
    all_cats = set()
    for i, sample in enumerate(dataset_split):
        cats = sample["objects"]["categories"]
        all_cats.update(int(c) for c in cats)
        if i >= max_scan:
            break

    mapping = {}
    for cid in sorted(all_cats):
        mapping[cid] = BMD45_DEFAULT_NAMES.get(cid, f"class_{cid}")

    return mapping


def convert_sample(sample, images_dir: str, labels_dir: str, idx: int):
    """
    Save one HuggingFace sample as a YOLO-format image + label file.

    BMD-45 bbox format (from HF): [x_min, y_min, width, height] in pixels
    YOLO label format: class_id  cx  cy  w  h  (all normalised 0-1)
    """
    img = sample["image"]
    img_w, img_h = img.size

    # Save image as JPEG
    img_path = os.path.join(images_dir, f"{idx:06d}.jpg")
    img.save(img_path, "JPEG", quality=95)

    # Build YOLO label
    objects = sample["objects"]
    bboxes = objects["bbox"]
    categories = objects["categories"]

    label_path = os.path.join(labels_dir, f"{idx:06d}.txt")
    with open(label_path, "w") as f:
        for bbox, cat in zip(bboxes, categories):
            x_min, y_min, w, h = bbox
            # Normalise to [0, 1]
            cx = (x_min + w / 2) / img_w
            cy = (y_min + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            # Clamp to valid range
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            f.write(f"{int(cat)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")


def curated_train_indices(dataset, n_samples: int) -> list:
    """
    Stratified sampling: scan the train split and pick indices so that
    every class gets roughly equal representation.  Falls back to random
    sampling if scanning is too slow.
    """
    print(f"  Stratified sampling {n_samples} images from train split ...")
    print(f"  (Scanning up to 8000 images to build class index) ...")

    cat_to_indices = defaultdict(list)
    scan_limit = min(len(dataset), max(n_samples * 4, 8000))

    for i in tqdm(range(scan_limit), desc="  Scanning", unit="img"):
        cats = dataset[i]["objects"]["categories"]
        for c in set(int(x) for x in cats):
            cat_to_indices[c].append(i)

    # Round-robin pick from each class
    selected = set()
    per_class = max(1, n_samples // max(len(cat_to_indices), 1))

    for cid in sorted(cat_to_indices.keys()):
        pool = cat_to_indices[cid]
        random.shuffle(pool)
        selected.update(pool[:per_class])

    # Fill remainder randomly from scanned range
    if len(selected) < n_samples:
        remaining = list(set(range(scan_limit)) - selected)
        random.shuffle(remaining)
        selected.update(remaining[: n_samples - len(selected)])

    result = sorted(selected)[:n_samples]
    print(f"  Selected {len(result)} images across {len(cat_to_indices)} classes")
    return result


def write_data_yaml(data_dir: str, class_mapping: dict):
    """Write the YOLO data.yaml config."""
    yaml_path = os.path.join(data_dir, "bmd45", "data.yaml")

    nc = max(class_mapping.keys()) + 1  # number of classes

    lines = [
        f"path: {os.path.abspath(os.path.join(data_dir, 'bmd45'))}",
        "train: train_mini/images",
        "val: val/images",
        f"nc: {nc}",
        "names:",
    ]
    for cid in range(nc):
        name = class_mapping.get(cid, f"class_{cid}")
        lines.append(f"  {cid}: {name}")

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  ✓ data.yaml written: {yaml_path}")
    print(f"    Classes: {nc}")
    for cid in range(nc):
        print(f"      {cid}: {class_mapping.get(cid, f'class_{cid}')}")

    return yaml_path


def download_and_prepare(data_dir: str, n_train: int = 2000):
    """Main pipeline: download, convert, write config."""
    from datasets import load_dataset

    bmd45_base = os.path.join(data_dir, "bmd45")

    # ── 1. Download val split ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 1/3: Downloading BMD-45 val split")
    print("=" * 60)

    val_images_dir = os.path.join(bmd45_base, "val", "images")
    val_labels_dir = os.path.join(bmd45_base, "val", "labels")
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Check if val already prepared
    existing_val = len([f for f in os.listdir(val_images_dir) if f.endswith(".jpg")])
    if existing_val > 1000:
        print(f"  [SKIP] Val split already has {existing_val} images.")
        ds_val = load_dataset("iisc-aim/BMD-45", split="val")
    else:
        ds_val = load_dataset("iisc-aim/BMD-45", split="val")
        print(f"  Loaded {len(ds_val)} val images from HuggingFace")
        print(f"  Saving to: {val_images_dir}")

        for idx in tqdm(range(len(ds_val)), desc="  Val", unit="img"):
            convert_sample(ds_val[idx], val_images_dir, val_labels_dir, idx)

        print(f"  ✓ Val split: {len(ds_val)} images saved")

    # ── 2. Download curated train subset ──────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Step 2/3: Downloading BMD-45 train subset ({n_train} images)")
    print("=" * 60)

    train_images_dir = os.path.join(bmd45_base, "train_mini", "images")
    train_labels_dir = os.path.join(bmd45_base, "train_mini", "labels")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)

    existing_train = len([f for f in os.listdir(train_images_dir) if f.endswith(".jpg")])
    if existing_train >= n_train * 0.9:
        print(f"  [SKIP] Train subset already has {existing_train} images.")
    else:
        # Load the full train split (streaming would be better for huge data,
        # but for 2K samples the direct approach is simpler)
        ds_train = load_dataset("iisc-aim/BMD-45", split="train")
        print(f"  Full train split: {len(ds_train)} images")

        indices = curated_train_indices(ds_train, n_train)

        print(f"  Converting {len(indices)} images to YOLO format ...")
        for out_idx, src_idx in enumerate(tqdm(indices, desc="  Train", unit="img")):
            convert_sample(ds_train[src_idx], train_images_dir, train_labels_dir, out_idx)

        print(f"  ✓ Train subset: {len(indices)} images saved")

    # ── 3. Discover classes and write data.yaml ───────────────────────────
    print("\n" + "=" * 60)
    print("  Step 3/3: Discovering classes & writing data.yaml")
    print("=" * 60)

    class_mapping = discover_classes(ds_val)
    yaml_path = write_data_yaml(data_dir, class_mapping)

    # Save class mapping as JSON for reference
    mapping_path = os.path.join(bmd45_base, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"  Class mapping saved: {mapping_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BMD-45 Preparation Complete")
    print("=" * 60)
    n_train_actual = len([f for f in os.listdir(train_images_dir) if f.endswith(".jpg")])
    n_val_actual = len([f for f in os.listdir(val_images_dir) if f.endswith(".jpg")])
    print(f"  Train images: {n_train_actual}")
    print(f"  Val images:   {n_val_actual}")
    print(f"  Classes:      {len(class_mapping)}")
    print(f"  data.yaml:    {yaml_path}")
    print(f"\n  Next: python scripts/finetune_bmd45.py --data-yaml {yaml_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="RoadPulse Phase 2 — Prepare BMD-45 dataset in YOLO format"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root data directory (e.g., ./data or /content/RoadPulse/data)"
    )
    parser.add_argument(
        "--n-train", type=int, default=2000,
        help="Number of train images to sample (default: 2000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("  RoadPulse Phase 2 — BMD-45 Data Preparation")
    print("=" * 60)

    download_and_prepare(args.data_dir, args.n_train)


if __name__ == "__main__":
    main()
