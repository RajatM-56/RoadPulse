#!/usr/bin/env python3
"""
RoadPulse Phase 3 — Sample Frame Extractor for Visual Verification

Selects 5-10 diverse images from the VisDrone val split, runs the fine-tuned
drone model, and saves high-res annotated PNGs for manual inspection.

Small objects (box area < 32x32 px) are highlighted with thicker borders
and a diamond marker so you can visually confirm the model catches
small/distant vehicles at altitude.

Designed for Google Colab.

Usage:
    python scripts/sample_frames_visdrone.py \
        --model models/drone_best.pt \
        --data-yaml ./data/visdrone/data.yaml \
        --out-dir outputs/phase3/sample_frames \
        --n-frames 10
"""

import argparse
import os
import sys
import glob
import random

import cv2
import numpy as np


# Colour palette — 10 distinct colours for VisDrone's 10 classes
CLASS_COLOURS = {
    0: (0, 200, 255),    # pedestrian — amber
    1: (200, 200, 0),    # people — olive
    2: (0, 255, 127),    # bicycle — spring green
    3: (255, 200, 0),    # car — gold
    4: (255, 128, 0),    # van — tangerine
    5: (100, 100, 255),  # truck — coral blue
    6: (0, 255, 255),    # tricycle — cyan
    7: (128, 0, 255),    # awning-tricycle — purple
    8: (255, 0, 100),    # bus — deep pink
    9: (0, 165, 255),    # motor — orange
}
DEFAULT_COLOUR = (200, 200, 200)

SMALL_OBJ_THRESHOLD = 32  # pixels — boxes smaller than this get highlighted


def draw_detection_with_small_highlight(
    frame, bbox_xyxy, class_id, confidence, class_name,
    highlight_small=True
):
    """
    Draw a detection box. If the box is smaller than SMALL_OBJ_THRESHOLD,
    draw a thicker border and a diamond marker for visibility.
    """
    x1, y1, x2, y2 = [int(c) for c in bbox_xyxy]
    w = x2 - x1
    h = y2 - y1
    colour = CLASS_COLOURS.get(class_id, DEFAULT_COLOUR)

    is_small = (w < SMALL_OBJ_THRESHOLD or h < SMALL_OBJ_THRESHOLD)
    thickness = 3 if is_small else 2

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

    # Small-object diamond marker
    if is_small and highlight_small:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        size = max(8, min(w, h) // 2)
        pts = np.array([
            [cx, cy - size],
            [cx + size, cy],
            [cx, cy + size],
            [cx - size, cy],
        ], dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

    # Label
    small_tag = " ◆" if is_small else ""
    label = f"{class_name} {confidence:.2f}{small_tag}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    (tw, th_text), baseline = cv2.getTextSize(label, font, scale, 1)

    # Label background
    label_y = max(y1, th_text + 8)
    cv2.rectangle(frame, (x1, label_y - th_text - 6),
                  (x1 + tw + 4, label_y), colour, -1)
    cv2.putText(frame, label, (x1 + 2, label_y - 4),
                font, scale, (0, 0, 0), 1, cv2.LINE_AA)

    return is_small


def select_diverse_images(images_dir: str, n: int, seed: int = 42) -> list:
    """
    Select n diverse images from the val set.
    Picks from different parts of the sorted list to get variety
    in scene content, altitude, and lighting.
    """
    all_imgs = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not all_imgs:
        all_imgs = sorted(glob.glob(os.path.join(images_dir, "*.png")))

    if len(all_imgs) <= n:
        return all_imgs

    # Evenly-spaced selection for diversity + a few random picks
    step = len(all_imgs) // (n - 2) if n > 2 else len(all_imgs) // n
    selected = [all_imgs[i * step] for i in range(min(n - 2, len(all_imgs) // step))]

    # Add 2 random picks for variety
    remaining = [img for img in all_imgs if img not in selected]
    random.seed(seed)
    random.shuffle(remaining)
    selected.extend(remaining[:max(0, n - len(selected))])

    return selected[:n]


def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="RoadPulse Phase 3 — Sample frame extractor for visual verification"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to fine-tuned drone model (models/drone_best.pt)"
    )
    parser.add_argument(
        "--data-yaml", required=True,
        help="Path to VisDrone data.yaml"
    )
    parser.add_argument(
        "--out-dir", default="outputs/phase3/sample_frames",
        help="Output directory for annotated frames"
    )
    parser.add_argument(
        "--n-frames", type=int, default=10,
        help="Number of sample frames to extract (default: 10)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.15,
        help="Confidence threshold (default: 0.15 — low to show sensitivity)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="Inference image size (default: 1280)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  RoadPulse Phase 3 — Sample Frame Extractor")
    print("=" * 60)

    if not os.path.isfile(args.model):
        print(f"\n  ERROR: Model not found: {args.model}")
        sys.exit(1)

    device = detect_device()

    from ultralytics import YOLO
    model = YOLO(args.model)
    class_names = model.names  # {int: str}
    print(f"  Model loaded: {args.model}")
    print(f"  Classes: {len(class_names)}")
    print(f"  Conf threshold: {args.conf}")
    print(f"  Device: {device}")

    # Find val images directory from data.yaml
    val_images_dir = None
    with open(args.data_yaml) as f:
        data_root = None
        val_rel = None
        for line in f:
            line = line.strip()
            if line.startswith("path:"):
                data_root = line.split(":", 1)[1].strip()
            elif line.startswith("val:"):
                val_rel = line.split(":", 1)[1].strip()
        if data_root and val_rel:
            val_images_dir = os.path.join(data_root, val_rel)

    if not val_images_dir or not os.path.isdir(val_images_dir):
        print(f"\n  ERROR: Val images dir not found. Parsed: {val_images_dir}")
        sys.exit(1)

    print(f"  Val images dir: {val_images_dir}")

    # Select diverse images
    selected = select_diverse_images(val_images_dir, args.n_frames)
    print(f"  Selected {len(selected)} sample images")

    os.makedirs(args.out_dir, exist_ok=True)

    # Process each image
    index_lines = [
        "# Phase 3 — Sample Detection Frames",
        "",
        "> Drone fine-tuned model on VisDrone val images.",
        f"> Confidence threshold: {args.conf}",
        "> ◆ marks indicate small objects (box < 32px).",
        "",
    ]

    for i, img_path in enumerate(selected):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n  [{i+1}/{len(selected)}] {basename}")

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"    ERROR: Could not read image")
            continue

        h_orig, w_orig = frame.shape[:2]

        # Run inference
        results = model.predict(
            frame, conf=args.conf, imgsz=args.imgsz,
            verbose=False, device=device,
        )[0]

        # Draw detections
        n_total = 0
        n_small = 0
        class_counts = {}

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            for j in range(len(boxes)):
                cls_id = int(boxes.cls[j].item())
                conf = float(boxes.conf[j].item())
                bbox = boxes.xyxy[j].tolist()
                cls_name = class_names.get(cls_id, f"cls_{cls_id}")

                is_small = draw_detection_with_small_highlight(
                    frame, bbox, cls_id, conf, cls_name
                )
                n_total += 1
                if is_small:
                    n_small += 1
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        # Add info overlay
        info_lines_cv = [
            f"Detections: {n_total} ({n_small} small)",
            f"Image: {basename} ({w_orig}x{h_orig})",
            f"Model: drone_best.pt | conf>={args.conf}",
        ]
        y_offset = h_orig - 15 * len(info_lines_cv) - 10
        for info_line in info_lines_cv:
            cv2.putText(frame, info_line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            y_offset += 18

        # Save annotated frame
        out_path = os.path.join(args.out_dir, f"sample_{i+1:02d}_{basename}.png")
        cv2.imwrite(out_path, frame)
        print(f"    ✓ {n_total} detections ({n_small} small) → {os.path.basename(out_path)}")

        # Add to index
        class_summary = ", ".join(f"{k}: {v}" for k, v in sorted(class_counts.items()))
        index_lines.extend([
            f"### Frame {i+1}: `{basename}`",
            "",
            f"- **Detections:** {n_total} total, {n_small} small (< 32px)",
            f"- **Classes:** {class_summary or 'none'}",
            f"- **Resolution:** {w_orig}×{h_orig}",
            f"- **File:** `{os.path.basename(out_path)}`",
            "",
        ])

    # Write index markdown
    index_path = os.path.join(args.out_dir, "sample_index.md")
    with open(index_path, "w") as f:
        f.write("\n".join(index_lines))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Sample Frames Complete")
    print("=" * 60)
    print(f"  Frames saved: {len(selected)}")
    print(f"  Output dir:   {args.out_dir}")
    print(f"  Index file:   {index_path}")
    print(f"\n  Review the annotated PNGs to verify small/distant")
    print(f"  vehicles at altitude are being detected.")
    print()


if __name__ == "__main__":
    main()
