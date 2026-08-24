#!/usr/bin/env python3
"""
RoadPulse Phase 3 — Fine-Tune YOLO26n on VisDrone for Drone/Aerial Detection

Loads pretrained COCO YOLO26n and fine-tunes on a curated VisDrone subset.
Uses imgsz=1280 to exploit YOLO26's ProgLoss + STAL for small-object
sensitivity at drone altitudes.

Designed for Google Colab (T4/A100 GPU).

Usage (Colab):
    !python scripts/finetune_visdrone.py \
        --data-yaml ./data/visdrone/data.yaml \
        --epochs 25 \
        --batch 4 \
        --imgsz 1280

    # A100 (more VRAM):
    !python scripts/finetune_visdrone.py \
        --data-yaml ./data/visdrone/data.yaml \
        --epochs 25 \
        --batch 16 \
        --imgsz 1280
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
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"  GPU detected: {name} ({vram:.1f} GB VRAM)")
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Apple MPS detected")
        return "mps"
    else:
        print("  WARNING: No GPU detected — training on CPU will be very slow!")
        return "cpu"


def recommend_batch_size(device: str, imgsz: int) -> int:
    """Suggest a safe batch size based on device and image size."""
    import torch

    if device == "cpu" or device == "mps":
        return 2 if imgsz >= 1280 else 4

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        if imgsz >= 1280:
            if vram >= 40:    # A100
                return 16
            elif vram >= 20:  # A5000 / L4
                return 8
            else:             # T4 (16 GB)
                return 4
        else:  # imgsz=640
            if vram >= 40:
                return 32
            elif vram >= 16:
                return 16
            else:
                return 8

    return 4


def main():
    parser = argparse.ArgumentParser(
        description="RoadPulse Phase 3 — Fine-tune YOLO26n on VisDrone (drone altitude)"
    )
    parser.add_argument(
        "--data-yaml", required=True,
        help="Path to VisDrone data.yaml (from prepare_visdrone.py)"
    )
    parser.add_argument(
        "--model", default="yolo26n.pt",
        help="Base model (default: yolo26n.pt, fallback: yolo11n.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=25,
        help="Training epochs (default: 25)"
    )
    parser.add_argument(
        "--batch", type=int, default=0,
        help="Batch size (default: 0 = auto-detect safe value for your GPU)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="Input image size (default: 1280 — critical for small-object STAL)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: auto, 0 (cuda), mps, cpu (default: auto)"
    )
    parser.add_argument(
        "--patience", type=int, default=7,
        help="Early stopping patience (default: 7)"
    )
    parser.add_argument(
        "--project", default="runs/visdrone",
        help="Project directory for training outputs"
    )
    parser.add_argument(
        "--name", default="finetune",
        help="Run name within project directory"
    )
    parser.add_argument(
        "--out-model", default="models/drone_best.pt",
        help="Path to save the best checkpoint (default: models/drone_best.pt)"
    )
    parser.add_argument(
        "--multi-scale", action="store_true", default=True,
        help="Enable multi-scale training (default: True, helps scale variance)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  RoadPulse Phase 3 — Fine-Tune YOLO26n on VisDrone")
    print("  (Drone / Aerial altitude — ProgLoss + STAL active)")
    print("=" * 60)

    # Verify data.yaml
    if not os.path.isfile(args.data_yaml):
        print(f"\n  ERROR: data.yaml not found: {args.data_yaml}")
        print(f"  Run prepare_visdrone.py first.")
        sys.exit(1)

    # Detect device
    device = detect_device(args.device)

    # Auto batch size
    batch = args.batch
    if batch <= 0:
        batch = recommend_batch_size(device, args.imgsz)
        print(f"  Auto batch size: {batch} (based on device + imgsz={args.imgsz})")

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
    print(f"    data.yaml:    {args.data_yaml}")
    print(f"    Epochs:       {args.epochs}")
    print(f"    Batch size:   {batch}")
    print(f"    Image size:   {args.imgsz}  ← high res for small-object STAL")
    print(f"    Device:       {device}")
    print(f"    Multi-scale:  {args.multi_scale}")
    print(f"    Patience:     {args.patience}")
    print(f"    Project:      {args.project}")
    print(f"    Run name:     {args.name}")
    print()
    print("  NOTE: YOLO26 ProgLoss + STAL are automatically active.")
    print("        These dynamically reweight loss for small objects and")
    print("        ensure label assignment coverage for tiny targets.")
    print()

    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Multi-scale: varies imgsz ±50% each batch → scale robustness
        multi_scale=args.multi_scale,
        # Augmentation — tuned for aerial small-object fine-tuning
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,       # slight rotation (drone tilt)
        translate=0.15,
        scale=0.7,           # aggressive scale augmentation for altitude variation
        fliplr=0.5,
        flipud=0.1,          # slight vertical flip (drone can be inverted)
        mosaic=1.0,          # mosaic helps with scale diversity
        mixup=0.1,           # light mixup for regularization
        copy_paste=0.1,      # copy-paste augmentation for rare small objects
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
    print("  Fine-Tuning Complete (Drone Model)")
    print("=" * 60)
    print(f"  Best model:    {args.out_model}")
    print(f"  Train logs:    {args.project}/{args.name}/")
    print(f"  Plots:         {args.project}/{args.name}/")
    print(f"\n  Next steps:")
    print(f"    python scripts/eval_visdrone.py \\")
    print(f"        --model {args.out_model} \\")
    print(f"        --data-yaml {args.data_yaml}")
    print()


if __name__ == "__main__":
    main()
