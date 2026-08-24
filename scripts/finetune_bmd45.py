#!/usr/bin/env python3
"""
RoadPulse Phase 2 — Fine-Tune YOLO26n on BMD-45

Loads pretrained COCO YOLO26n weights and fine-tunes on the curated BMD-45
train subset prepared by prepare_bmd45.py.

Designed for Google Colab (T4/A100 GPU).  Also works on Apple Silicon (MPS).

Usage (Colab):
    !pip install ultralytics>=8.4
    !python scripts/finetune_bmd45.py \
        --data-yaml ./data/bmd45/data.yaml \
        --epochs 20 \
        --batch 16

Usage (local / MPS):
    python scripts/finetune_bmd45.py \
        --data-yaml ./data/bmd45/data.yaml \
        --epochs 20 \
        --batch 8 \
        --device mps
"""

import argparse
import os
import sys
import shutil


def detect_device(requested: str = "auto") -> str:
    """Pick the best available device."""
    import torch

    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  GPU detected: {name}")
        return "0"  # first CUDA device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Apple MPS detected")
        return "mps"
    else:
        print("  No GPU detected — using CPU (will be slow!)")
        return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="RoadPulse Phase 2 — Fine-tune YOLO26n on BMD-45"
    )
    parser.add_argument(
        "--data-yaml", required=True,
        help="Path to data.yaml (from prepare_bmd45.py)"
    )
    parser.add_argument(
        "--model", default="yolo26n.pt",
        help="Base model (default: yolo26n.pt, fallback: yolo11n.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16; use 8 for MPS / low VRAM)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: auto, 0 (cuda), mps, cpu (default: auto)"
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience (default: 5)"
    )
    parser.add_argument(
        "--project", default="runs/bmd45",
        help="Project directory for training outputs"
    )
    parser.add_argument(
        "--name", default="finetune",
        help="Run name within project directory"
    )
    parser.add_argument(
        "--out-model", default="models/fixed_cam_best.pt",
        help="Path to save the best checkpoint (default: models/fixed_cam_best.pt)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  RoadPulse Phase 2 — Fine-Tune YOLO26n on BMD-45")
    print("=" * 60)

    # Verify data.yaml exists
    if not os.path.isfile(args.data_yaml):
        print(f"\n  ERROR: data.yaml not found: {args.data_yaml}")
        print(f"  Run prepare_bmd45.py first.")
        sys.exit(1)

    # Detect device
    device = detect_device(args.device)

    # Load base model
    from ultralytics import YOLO

    print(f"\n  Loading base model: {args.model}")
    try:
        model = YOLO(args.model)
        print(f"  ✓ Loaded: {args.model}")
    except Exception as e:
        if "yolo26n" in args.model:
            print(f"  [WARN] {args.model} failed ({e}), falling back to yolo11n.pt")
            model = YOLO("yolo11n.pt")
            print(f"  ✓ Fallback loaded: yolo11n.pt")
        else:
            raise

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n  Training configuration:")
    print(f"    data.yaml:  {args.data_yaml}")
    print(f"    Epochs:     {args.epochs}")
    print(f"    Batch size: {args.batch}")
    print(f"    Image size: {args.imgsz}")
    print(f"    Device:     {device}")
    print(f"    Patience:   {args.patience}")
    print(f"    Project:    {args.project}")
    print(f"    Run name:   {args.name}")
    print()

    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Augmentation settings — good defaults for fine-tuning
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        # Optimiser
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        # Misc
        verbose=True,
        plots=True,
    )

    # ── Copy best checkpoint ──────────────────────────────────────────────
    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.isfile(best_path):
        os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
        shutil.copy2(best_path, args.out_model)
        size_mb = os.path.getsize(args.out_model) / (1024 * 1024)
        print(f"\n  ✓ Best checkpoint saved: {args.out_model} ({size_mb:.1f} MB)")
    else:
        # Fallback: try last.pt
        last_path = os.path.join(args.project, args.name, "weights", "last.pt")
        if os.path.isfile(last_path):
            os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
            shutil.copy2(last_path, args.out_model)
            print(f"\n  ⚠ best.pt not found; saved last.pt as: {args.out_model}")
        else:
            print(f"\n  ✗ ERROR: No checkpoint found in {args.project}/{args.name}/weights/")
            sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Fine-Tuning Complete")
    print("=" * 60)
    print(f"  Best model:  {args.out_model}")
    print(f"  Train logs:  {args.project}/{args.name}/")
    print(f"  Plots:       {args.project}/{args.name}/")
    print(f"\n  Next steps:")
    print(f"    python scripts/eval_bmd45.py \\")
    print(f"        --model {args.out_model} \\")
    print(f"        --data-yaml {args.data_yaml}")
    print()


if __name__ == "__main__":
    main()
