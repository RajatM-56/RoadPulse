#!/usr/bin/env python3
"""
RoadPulse Phase 2 — Side-by-Side Comparison Video

Runs both the COCO-pretrained and fine-tuned models on the same video clip
and renders a side-by-side annotated comparison.

Left half:  COCO pretrained detections
Right half: BMD-45 fine-tuned detections

Designed for Google Colab.

Usage:
    python scripts/compare_video.py \
        --clip data/sample_clips/fixed_cam_sample.mp4 \
        --finetuned-model models/fixed_cam_best.pt \
        --out-dir outputs/phase2

    # Use a specific COCO model:
    python scripts/compare_video.py \
        --clip data/sample_clips/fixed_cam_sample.mp4 \
        --finetuned-model models/fixed_cam_best.pt \
        --coco-model yolo26n.pt
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


# Colour palette for bounding boxes — 20 distinct colours
BOX_COLOURS = [
    (255, 200, 0),    # gold
    (0, 255, 127),    # spring green
    (0, 165, 255),    # orange
    (255, 0, 100),    # deep pink
    (100, 100, 255),  # coral blue
    (0, 255, 255),    # cyan
    (255, 255, 0),    # yellow
    (128, 0, 255),    # purple
    (255, 128, 0),    # tangerine
    (0, 200, 0),      # green
    (200, 200, 0),    # olive
    (255, 0, 255),    # magenta
    (100, 255, 100),  # light green
    (255, 100, 100),  # salmon
    (100, 200, 255),  # sky blue
    (200, 100, 255),  # lavender
    (255, 200, 200),  # peach
    (200, 255, 200),  # mint
    (150, 150, 150),  # grey
    (255, 150, 50),   # amber
]


def draw_detections(frame, results, label_prefix: str = ""):
    """
    Draw bounding boxes and labels on a frame from YOLO results.

    Args:
        frame: BGR image (modified in-place)
        results: Single ultralytics Results object
        label_prefix: Optional prefix for the label text
    Returns:
        count of detections drawn
    """
    n_det = 0
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes
        names = results.names  # {int: str} class mapping

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = [int(c) for c in boxes.xyxy[i].tolist()]

            colour = BOX_COLOURS[cls_id % len(BOX_COLOURS)]
            cls_name = names.get(cls_id, f"cls_{cls_id}")

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Draw label
            label = f"{label_prefix}{cls_name} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1
            )
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
            )
            n_det += 1

    return n_det


def add_panel_label(frame, text: str, position: str = "top-left",
                    bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """Add a panel title label to a frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    if position == "top-left":
        org = (10, 30)
    elif position == "top-center":
        org = ((w - tw) // 2, 30)
    else:
        org = (10, 30)

    # Background rect
    cv2.rectangle(
        frame,
        (org[0] - 5, org[1] - th - 5),
        (org[0] + tw + 5, org[1] + 5),
        bg_color, -1
    )
    cv2.putText(frame, text, org, font, scale, text_color, thickness, cv2.LINE_AA)


def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="RoadPulse Phase 2 — Side-by-side COCO vs fine-tuned video"
    )
    parser.add_argument(
        "--clip", required=True,
        help="Path to input video clip"
    )
    parser.add_argument(
        "--finetuned-model", required=True,
        help="Path to fine-tuned model (e.g., models/fixed_cam_best.pt)"
    )
    parser.add_argument(
        "--coco-model", default="yolo26n.pt",
        help="COCO pretrained model (default: yolo26n.pt)"
    )
    parser.add_argument(
        "--out-dir", default="outputs/phase2",
        help="Output directory (default: outputs/phase2)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (default: 640)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Max frames to process (default: all)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  RoadPulse Phase 2 — Comparison Video")
    print("=" * 60)

    if not os.path.isfile(args.clip):
        print(f"\n  ERROR: Video clip not found: {args.clip}")
        sys.exit(1)
    if not os.path.isfile(args.finetuned_model):
        print(f"\n  ERROR: Fine-tuned model not found: {args.finetuned_model}")
        sys.exit(1)

    device = detect_device()
    print(f"  Device: {device}")

    from ultralytics import YOLO

    # Load models
    print(f"\n  Loading COCO model: {args.coco_model}")
    try:
        coco_model = YOLO(args.coco_model)
    except Exception:
        print(f"  [WARN] {args.coco_model} not available, trying yolo11n.pt")
        coco_model = YOLO("yolo11n.pt")
    print(f"  ✓ COCO model loaded ({len(coco_model.names)} classes)")

    print(f"  Loading fine-tuned model: {args.finetuned_model}")
    ft_model = YOLO(args.finetuned_model)
    print(f"  ✓ Fine-tuned model loaded ({len(ft_model.names)} classes)")

    # Open video
    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        print(f"\n  ERROR: Could not open video: {args.clip}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)

    print(f"\n  Input:  {args.clip}")
    print(f"  Resolution: {width}x{height} @ {fps:.1f} fps")
    print(f"  Frames: {total_frames}")

    # Output: side-by-side → double width
    out_w = width * 2
    out_h = height
    clip_name = os.path.splitext(os.path.basename(args.clip))[0]
    out_path = os.path.join(args.out_dir, f"comparison_coco_vs_finetuned_{clip_name}.mp4")
    os.makedirs(args.out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        print(f"  ERROR: Could not create output video: {out_path}")
        sys.exit(1)

    print(f"  Output: {out_path} ({out_w}x{out_h})")
    print()

    frame_idx = 0
    total_coco_dets = 0
    total_ft_dets = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        # Make two copies
        frame_coco = frame.copy()
        frame_ft = frame.copy()

        # Run COCO model
        coco_results = coco_model.predict(
            frame, conf=args.conf, imgsz=args.imgsz,
            verbose=False, device=device,
        )[0]
        n_coco = draw_detections(frame_coco, coco_results)
        total_coco_dets += n_coco

        # Run fine-tuned model
        ft_results = ft_model.predict(
            frame, conf=args.conf, imgsz=args.imgsz,
            verbose=False, device=device,
        )[0]
        n_ft = draw_detections(frame_ft, ft_results)
        total_ft_dets += n_ft

        # Add panel labels
        add_panel_label(frame_coco, f"COCO Pretrained ({n_coco} det)",
                        bg_color=(0, 0, 180))
        add_panel_label(frame_ft, f"BMD-45 Fine-Tuned ({n_ft} det)",
                        bg_color=(0, 140, 0))

        # Add frame counter
        counter = f"Frame {frame_idx}/{total_frames}"
        cv2.putText(frame_coco, counter, (10, height - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame_ft, counter, (10, height - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Concatenate side-by-side
        # Draw a thin divider line
        cv2.line(frame_coco, (width - 1, 0), (width - 1, height), (255, 255, 255), 2)

        combined = np.hstack([frame_coco, frame_ft])
        writer.write(combined)

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            print(
                f"  Frame {frame_idx}/{total_frames} "
                f"({frame_idx / total_frames * 100:.0f}%) | "
                f"{fps_actual:.1f} fps | "
                f"COCO: {total_coco_dets} det, FT: {total_ft_dets} det"
            )

    cap.release()
    writer.release()
    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Comparison Video Complete")
    print("=" * 60)
    print(f"  Frames processed: {frame_idx}")
    print(f"  Time: {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps)" if elapsed > 0 else "")
    print(f"  COCO total detections:      {total_coco_dets}")
    print(f"  Fine-tuned total detections: {total_ft_dets}")
    print(f"  Output: {out_path}")
    file_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  File size: {file_mb:.1f} MB")
    print()


if __name__ == "__main__":
    main()
