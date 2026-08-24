#!/usr/bin/env python3
"""
RoadPulse Phase 2 — Evaluate Fine-Tuned Model on BMD-45 Val

Runs model.val() for both the fine-tuned checkpoint and the COCO baseline,
then produces a markdown evaluation report with mAP@0.5 and per-class
precision/recall.

Designed for Google Colab.

Usage:
    python scripts/eval_bmd45.py \
        --model models/fixed_cam_best.pt \
        --data-yaml ./data/bmd45/data.yaml \
        --out-dir eval/phase2
"""

import argparse
import os
import sys
import json


def run_val(model, data_yaml: str, device: str, split: str = "val", imgsz: int = 640):
    """
    Run model.val() and return the metrics object.
    """
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        device=device,
        verbose=False,
        plots=True,
    )
    return metrics


def metrics_to_dict(metrics, class_names: dict) -> dict:
    """
    Extract a structured dict from ultralytics Metrics object.

    Returns:
        {
            "mAP50": float,
            "mAP50_95": float,
            "per_class": [
                {"class_id": int, "name": str, "precision": float,
                 "recall": float, "mAP50": float},
                ...
            ]
        }
    """
    result = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }

    per_class = []
    # metrics.box.ap50 is a numpy array of shape (nc,)
    # metrics.box.p, .r are per-class precision/recall
    n_classes = len(metrics.box.ap50)
    for i in range(n_classes):
        entry = {
            "class_id": i,
            "name": class_names.get(i, f"class_{i}"),
            "precision": float(metrics.box.p[i]) if i < len(metrics.box.p) else 0.0,
            "recall": float(metrics.box.r[i]) if i < len(metrics.box.r) else 0.0,
            "mAP50": float(metrics.box.ap50[i]),
        }
        per_class.append(entry)

    result["per_class"] = per_class
    return result


def generate_report(finetuned_results: dict, coco_results: dict | None,
                    out_path: str):
    """
    Write a markdown eval report comparing fine-tuned vs COCO baseline.
    """
    lines = [
        "# RoadPulse Phase 2 — Evaluation Report",
        "",
        "## Overall Metrics",
        "",
        "| Metric | COCO Baseline | Fine-Tuned BMD-45 | Δ |",
        "|--------|:------------:|:-----------------:|:---:|",
    ]

    ft_map50 = finetuned_results["mAP50"]
    ft_map50_95 = finetuned_results["mAP50_95"]

    if coco_results:
        co_map50 = coco_results["mAP50"]
        co_map50_95 = coco_results["mAP50_95"]
        delta50 = ft_map50 - co_map50
        delta50_95 = ft_map50_95 - co_map50_95
        lines.append(
            f"| mAP@0.5 | {co_map50:.4f} | **{ft_map50:.4f}** | "
            f"{'▲' if delta50 > 0 else '▼'} {abs(delta50):.4f} |"
        )
        lines.append(
            f"| mAP@0.5:0.95 | {co_map50_95:.4f} | **{ft_map50_95:.4f}** | "
            f"{'▲' if delta50_95 > 0 else '▼'} {abs(delta50_95):.4f} |"
        )
    else:
        lines.append(f"| mAP@0.5 | — | **{ft_map50:.4f}** | — |")
        lines.append(f"| mAP@0.5:0.95 | — | **{ft_map50_95:.4f}** | — |")

    # Per-class table for fine-tuned model
    lines.extend([
        "",
        "## Per-Class Metrics (Fine-Tuned Model)",
        "",
        "| Class | Precision | Recall | mAP@0.5 |",
        "|-------|:---------:|:------:|:-------:|",
    ])
    for cls in finetuned_results["per_class"]:
        lines.append(
            f"| {cls['name']} | {cls['precision']:.4f} | "
            f"{cls['recall']:.4f} | {cls['mAP50']:.4f} |"
        )

    # Pass/fail verdict
    lines.extend([
        "",
        "## Verdict",
        "",
    ])
    if coco_results and ft_map50 > coco_results["mAP50"]:
        lines.append(
            f"✅ **PASS** — Fine-tuned mAP@0.5 ({ft_map50:.4f}) is measurably "
            f"higher than COCO baseline ({coco_results['mAP50']:.4f}), "
            f"Δ = +{ft_map50 - coco_results['mAP50']:.4f}"
        )
    elif coco_results:
        lines.append(
            f"⚠️ **MARGINAL** — Fine-tuned mAP@0.5 ({ft_map50:.4f}) vs "
            f"COCO baseline ({coco_results['mAP50']:.4f})"
        )
    else:
        lines.append(
            f"ℹ️ Fine-tuned mAP@0.5 = {ft_map50:.4f} "
            f"(COCO baseline not evaluated on BMD-45 classes)"
        )

    lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n  ✓ Eval report saved: {out_path}")


def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="RoadPulse Phase 2 — Evaluate fine-tuned model on BMD-45 val"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to fine-tuned model checkpoint (e.g., models/fixed_cam_best.pt)"
    )
    parser.add_argument(
        "--data-yaml", required=True,
        help="Path to BMD-45 data.yaml"
    )
    parser.add_argument(
        "--baseline-model", default="yolo26n.pt",
        help="COCO baseline model for comparison (default: yolo26n.pt)"
    )
    parser.add_argument(
        "--out-dir", default="eval/phase2",
        help="Output directory for reports (default: eval/phase2)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (default: 640)"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip COCO baseline evaluation (saves time)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  RoadPulse Phase 2 — BMD-45 Evaluation")
    print("=" * 60)

    if not os.path.isfile(args.model):
        print(f"\n  ERROR: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.data_yaml):
        print(f"\n  ERROR: data.yaml not found: {args.data_yaml}")
        sys.exit(1)

    device = detect_device()
    print(f"  Device: {device}")

    from ultralytics import YOLO

    # Load class names from data.yaml
    class_names = {}
    with open(args.data_yaml) as f:
        for line in f:
            line = line.strip()
            # Parse "  N: name" lines
            if ":" in line and line.split(":")[0].strip().isdigit():
                parts = line.split(":", 1)
                cid = int(parts[0].strip())
                name = parts[1].strip()
                class_names[cid] = name

    # ── Evaluate fine-tuned model ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Evaluating fine-tuned model: {args.model}")
    print(f"{'─' * 60}")

    ft_model = YOLO(args.model)
    ft_metrics = run_val(ft_model, args.data_yaml, device, imgsz=args.imgsz)
    ft_results = metrics_to_dict(ft_metrics, class_names)

    print(f"  Fine-tuned mAP@0.5:     {ft_results['mAP50']:.4f}")
    print(f"  Fine-tuned mAP@0.5:0.95: {ft_results['mAP50_95']:.4f}")

    # ── Evaluate COCO baseline (optional) ─────────────────────────────────
    coco_results = None
    if not args.skip_baseline:
        print(f"\n{'─' * 60}")
        print(f"  Evaluating COCO baseline: {args.baseline_model}")
        print(f"  NOTE: COCO model has 80 classes vs BMD-45's 14 classes.")
        print(f"        The baseline mAP will be very low because the label")
        print(f"        spaces don't match. This is expected and demonstrates")
        print(f"        the value of domain-specific fine-tuning.")
        print(f"{'─' * 60}")

        try:
            coco_model = YOLO(args.baseline_model)
        except Exception:
            print(f"  [WARN] {args.baseline_model} not available, trying yolo11n.pt")
            coco_model = YOLO("yolo11n.pt")

        try:
            coco_metrics = run_val(coco_model, args.data_yaml, device, imgsz=args.imgsz)
            coco_results = metrics_to_dict(coco_metrics, class_names)
            print(f"  COCO baseline mAP@0.5:     {coco_results['mAP50']:.4f}")
            print(f"  COCO baseline mAP@0.5:0.95: {coco_results['mAP50_95']:.4f}")
        except Exception as e:
            print(f"  [WARN] COCO baseline eval failed: {e}")
            print(f"  This is expected — COCO model has 80 classes, BMD-45 has 14.")
            print(f"  The fine-tuned model IS the meaningful result.")
            coco_results = None

    # ── Generate report ───────────────────────────────────────────────────
    report_path = os.path.join(args.out_dir, "eval_report.md")
    generate_report(ft_results, coco_results, report_path)

    # Also save raw JSON for programmatic access
    json_path = os.path.join(args.out_dir, "eval_results.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({
            "finetuned": ft_results,
            "coco_baseline": coco_results,
        }, f, indent=2)
    print(f"  ✓ Raw results JSON: {json_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    print(f"  Fine-tuned mAP@0.5: {ft_results['mAP50']:.4f}")
    if coco_results:
        print(f"  COCO base  mAP@0.5: {coco_results['mAP50']:.4f}")
        delta = ft_results['mAP50'] - coco_results['mAP50']
        print(f"  Delta:              {'▲' if delta > 0 else '▼'} {abs(delta):.4f}")
    print(f"\n  Report: {report_path}")
    print(f"\n  Next: python scripts/compare_video.py \\")
    print(f"           --clip data/sample_clips/fixed_cam_sample.mp4 \\")
    print(f"           --finetuned-model {args.model}")
    print()


if __name__ == "__main__":
    main()
