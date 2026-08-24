#!/usr/bin/env python3
"""
RoadPulse Phase 3 — Evaluate Drone Model on VisDrone Val Split

Runs model.val() for both the fine-tuned drone checkpoint and the COCO
baseline, then produces a markdown eval report with mAP@0.5, per-class
precision/recall, and small-object focus metrics.

Designed for Google Colab.

Usage:
    python scripts/eval_visdrone.py \
        --model models/drone_best.pt \
        --data-yaml ./data/visdrone/data.yaml \
        --out-dir eval/phase3
"""

import argparse
import os
import sys
import json


SMALL_OBJECT_CLASSES = {"bicycle", "tricycle", "awning-tricycle", "motor"}


def run_val(model, data_yaml: str, device: str, imgsz: int = 1280):
    """Run model.val() and return metrics."""
    metrics = model.val(
        data=data_yaml,
        split="val",
        imgsz=imgsz,
        device=device,
        verbose=False,
        plots=True,
    )
    return metrics


def metrics_to_dict(metrics, class_names: dict) -> dict:
    """Extract structured metrics from ultralytics Metrics object."""
    result = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }

    per_class = []
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


def generate_report(ft_results: dict, coco_results: dict | None,
                    out_path: str):
    """Write markdown evaluation report."""
    lines = [
        "# RoadPulse Phase 3 — Drone Model Evaluation Report",
        "",
        "> Model fine-tuned on VisDrone-DET with imgsz=1280 (ProgLoss + STAL)",
        "",
        "## Overall Metrics",
        "",
        "| Metric | COCO Baseline | Drone Fine-Tuned | Δ |",
        "|--------|:------------:|:----------------:|:---:|",
    ]

    ft_map50 = ft_results["mAP50"]
    ft_map50_95 = ft_results["mAP50_95"]

    if coco_results:
        co_map50 = coco_results["mAP50"]
        co_map50_95 = coco_results["mAP50_95"]
        d50 = ft_map50 - co_map50
        d95 = ft_map50_95 - co_map50_95
        lines.append(
            f"| mAP@0.5 | {co_map50:.4f} | **{ft_map50:.4f}** | "
            f"{'▲' if d50 > 0 else '▼'} {abs(d50):.4f} |"
        )
        lines.append(
            f"| mAP@0.5:0.95 | {co_map50_95:.4f} | **{ft_map50_95:.4f}** | "
            f"{'▲' if d95 > 0 else '▼'} {abs(d95):.4f} |"
        )
    else:
        lines.append(f"| mAP@0.5 | — | **{ft_map50:.4f}** | — |")
        lines.append(f"| mAP@0.5:0.95 | — | **{ft_map50_95:.4f}** | — |")

    # Per-class table
    lines.extend([
        "",
        "## Per-Class Metrics (Drone Fine-Tuned)",
        "",
        "| Class | Precision | Recall | mAP@0.5 | Small-Object? |",
        "|-------|:---------:|:------:|:-------:|:-------------:|",
    ])
    for cls in ft_results["per_class"]:
        is_small = "✦" if cls["name"] in SMALL_OBJECT_CLASSES else ""
        lines.append(
            f"| {cls['name']} | {cls['precision']:.4f} | "
            f"{cls['recall']:.4f} | {cls['mAP50']:.4f} | {is_small} |"
        )

    # Small-object focus section
    small_classes = [c for c in ft_results["per_class"]
                     if c["name"] in SMALL_OBJECT_CLASSES]
    if small_classes:
        avg_recall = sum(c["recall"] for c in small_classes) / len(small_classes)
        avg_map = sum(c["mAP50"] for c in small_classes) / len(small_classes)
        lines.extend([
            "",
            "## Small-Object Focus (✦ marked above)",
            "",
            f"| Metric | Value |",
            f"|--------|:-----:|",
            f"| Avg Recall (small classes) | {avg_recall:.4f} |",
            f"| Avg mAP@0.5 (small classes) | {avg_map:.4f} |",
            f"| Classes | {', '.join(c['name'] for c in small_classes)} |",
        ])

    # Verdict
    lines.extend(["", "## Verdict", ""])
    if coco_results and ft_map50 > coco_results["mAP50"]:
        lines.append(
            f"✅ **PASS** — Drone fine-tuned mAP@0.5 ({ft_map50:.4f}) is "
            f"measurably higher than COCO baseline ({coco_results['mAP50']:.4f}), "
            f"Δ = +{ft_map50 - coco_results['mAP50']:.4f}"
        )
    elif coco_results:
        lines.append(
            f"⚠️ **MARGINAL** — Drone mAP@0.5 ({ft_map50:.4f}) vs "
            f"COCO baseline ({coco_results['mAP50']:.4f})"
        )
    else:
        lines.append(
            f"ℹ️ Drone fine-tuned mAP@0.5 = {ft_map50:.4f} "
            f"(COCO baseline not evaluated)"
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
        description="RoadPulse Phase 3 — Evaluate drone model on VisDrone val"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to fine-tuned drone model (e.g., models/drone_best.pt)"
    )
    parser.add_argument(
        "--data-yaml", required=True,
        help="Path to VisDrone data.yaml"
    )
    parser.add_argument(
        "--baseline-model", default="yolo26n.pt",
        help="COCO baseline model for comparison (default: yolo26n.pt)"
    )
    parser.add_argument(
        "--out-dir", default="eval/phase3",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="Inference image size (default: 1280)"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip COCO baseline evaluation"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  RoadPulse Phase 3 — VisDrone Evaluation")
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
            if ":" in line and line.split(":")[0].strip().isdigit():
                parts = line.split(":", 1)
                cid = int(parts[0].strip())
                name = parts[1].strip()
                class_names[cid] = name

    # ── Evaluate fine-tuned model ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Evaluating drone model: {args.model}")
    print(f"{'─' * 60}")

    ft_model = YOLO(args.model)
    ft_metrics = run_val(ft_model, args.data_yaml, device, imgsz=args.imgsz)
    ft_results = metrics_to_dict(ft_metrics, class_names)

    print(f"  Drone mAP@0.5:     {ft_results['mAP50']:.4f}")
    print(f"  Drone mAP@0.5:0.95: {ft_results['mAP50_95']:.4f}")

    # ── Evaluate COCO baseline ────────────────────────────────────────────
    coco_results = None
    if not args.skip_baseline:
        print(f"\n{'─' * 60}")
        print(f"  Evaluating COCO baseline: {args.baseline_model}")
        print(f"  NOTE: COCO (80 classes) vs VisDrone (10 classes) = label mismatch.")
        print(f"        Low baseline mAP is expected and demonstrates the need")
        print(f"        for domain-specific fine-tuning on aerial imagery.")
        print(f"{'─' * 60}")

        try:
            coco_model = YOLO(args.baseline_model)
        except Exception:
            print(f"  [WARN] {args.baseline_model} not available, trying yolo11n.pt")
            coco_model = YOLO("yolo11n.pt")

        try:
            coco_metrics = run_val(coco_model, args.data_yaml, device, imgsz=args.imgsz)
            coco_results = metrics_to_dict(coco_metrics, class_names)
            print(f"  COCO baseline mAP@0.5: {coco_results['mAP50']:.4f}")
        except Exception as e:
            print(f"  [WARN] COCO baseline eval failed: {e}")
            coco_results = None

    # ── Generate report ───────────────────────────────────────────────────
    report_path = os.path.join(args.out_dir, "eval_report.md")
    generate_report(ft_results, coco_results, report_path)

    # Raw JSON
    json_path = os.path.join(args.out_dir, "eval_results.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({
            "finetuned_drone": ft_results,
            "coco_baseline": coco_results,
        }, f, indent=2)
    print(f"  ✓ Raw results JSON: {json_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    print(f"  Drone mAP@0.5: {ft_results['mAP50']:.4f}")
    if coco_results:
        delta = ft_results["mAP50"] - coco_results["mAP50"]
        print(f"  COCO mAP@0.5:  {coco_results['mAP50']:.4f}")
        print(f"  Delta:         {'▲' if delta > 0 else '▼'} {abs(delta):.4f}")
    print(f"\n  Report: {report_path}")
    print(f"\n  Next: python scripts/sample_frames_visdrone.py \\")
    print(f"           --model {args.model} \\")
    print(f"           --data-yaml {args.data_yaml}")
    print()


if __name__ == "__main__":
    main()
