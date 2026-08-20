#!/usr/bin/env python3
"""
RoadPulse Phase 1 — Batch Detection Inference

Runs YOLO object detection on a folder of video clips.
Produces annotated output videos + raw detections JSON for each clip.

No tracking, no incident logic — detection only (Phase 1 scope).

Usage:
    python pipeline/detect.py \\
        --data-dir ./data/sample_clips \\
        --out-dir ./outputs/phase1 \\
        --conf-thresh 0.25

    # On Colab with GPU:
    python pipeline/detect.py \\
        --data-dir /content/drive/MyDrive/roadpulse/data/sample_clips \\
        --out-dir /content/drive/MyDrive/roadpulse/outputs/phase1 \\
        --conf-thresh 0.25
"""

import argparse
import os
import sys
import time
import json

import cv2

# Add project root to path so we can import pipeline.utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.utils import (
    get_video_clips,
    VideoWriter,
    save_detections_json,
    draw_detection,
    is_vehicle_class,
    COCO_NAMES,
    COCO_VEHICLE_CLASS_IDS,
)


def load_model(model_name: str = "yolo26n.pt"):
    """
    Load a YOLO model. Tries yolo26n first, falls back to yolo11n.

    Args:
        model_name: Model filename (e.g., "yolo26n.pt").

    Returns:
        Loaded YOLO model instance.
    """
    from ultralytics import YOLO

    try:
        print(f"[...] Loading model: {model_name}")
        model = YOLO(model_name)
        print(f"[OK]  Model loaded: {model_name}")
        return model
    except Exception as e:
        if "yolo26n" in model_name:
            print(f"[WARN] {model_name} failed ({e}), falling back to yolo11n.pt")
            model = YOLO("yolo11n.pt")
            print(f"[OK]  Fallback model loaded: yolo11n.pt")
            return model
        else:
            raise


def process_clip(model, clip_path: str, out_dir: str,
                 conf_thresh: float = 0.25,
                 imgsz: int = 640,
                 vehicle_only: bool = True,
                 show_all_classes: bool = False):
    """
    Run detection on a single video clip.

    Args:
        model: Loaded YOLO model.
        clip_path: Path to input video file.
        out_dir: Directory for output files.
        conf_thresh: Confidence threshold for detections.
        imgsz: Inference image size.
        vehicle_only: If True, only keep vehicle classes in JSON output.
        show_all_classes: If True, draw ALL detected classes (not just vehicles).

    Returns:
        Dict with clip processing summary.
    """
    clip_name = os.path.splitext(os.path.basename(clip_path))[0]
    out_video_path = os.path.join(out_dir, f"{clip_name}_annotated.mp4")
    out_json_path = os.path.join(out_dir, f"{clip_name}_detections.json")

    print(f"\n{'─' * 60}")
    print(f"  Processing: {clip_name}")
    print(f"  Input:  {clip_path}")
    print(f"  Output: {out_video_path}")
    print(f"{'─' * 60}")

    # Open source video
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open video: {clip_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Resolution: {width}x{height} @ {fps:.1f} fps")
    print(f"  Total frames: {total_frames}")
    print(f"  Confidence threshold: {conf_thresh}")
    print(f"  Inference size: {imgsz}")

    # Prepare output
    os.makedirs(out_dir, exist_ok=True)
    writer = VideoWriter(out_video_path, fps, width, height)

    all_detections = []
    frame_idx = 0
    total_vehicle_dets = 0
    total_all_dets = 0
    start_time = time.time()

    # Run inference frame-by-frame using stream mode (memory efficient)
    # We read frames manually for full control over annotation rendering
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference on this frame
        results = model.predict(
            frame,
            conf=conf_thresh,
            imgsz=imgsz,
            verbose=False,
            device=None,  # auto-detect GPU/CPU
        )

        result = results[0]  # single image → single result

        # Extract detections
        frame_dets = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]

                total_all_dets += 1

                # Determine if this is a vehicle class
                is_vehicle = is_vehicle_class(class_id)
                if is_vehicle:
                    total_vehicle_dets += 1

                # Store in JSON (vehicle-only or all, based on flag)
                if not vehicle_only or is_vehicle:
                    det = {
                        "frame_idx": frame_idx,
                        "bbox_xyxy": [round(c, 1) for c in bbox],
                        "class_id": class_id,
                        "class_name": COCO_NAMES.get(class_id, f"cls_{class_id}"),
                        "confidence": round(confidence, 4),
                    }
                    frame_dets.append(det)

                # Draw on frame — draw vehicles always, others only if show_all
                if is_vehicle or show_all_classes:
                    draw_detection(
                        frame, bbox, class_id, confidence,
                        class_name=COCO_NAMES.get(class_id),
                    )

        all_detections.extend(frame_dets)

        # Add frame counter overlay
        cv2.putText(
            frame,
            f"Frame {frame_idx}/{total_frames} | Vehicles: {len([d for d in frame_dets if is_vehicle_class(d['class_id'])])}",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

        # Progress logging every 100 frames
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_idx}/{total_frames} "
                  f"({frame_idx/total_frames*100:.0f}%) "
                  f"| {fps_actual:.1f} fps | "
                  f"{total_vehicle_dets} vehicle detections so far")

    cap.release()
    writer.release()
    elapsed = time.time() - start_time

    # Save detections JSON
    save_detections_json(all_detections, out_json_path)

    # Summary
    summary = {
        "clip_name": clip_name,
        "input_path": clip_path,
        "output_video": out_video_path,
        "output_json": out_json_path,
        "total_frames": frame_idx,
        "total_detections_all": total_all_dets,
        "total_detections_vehicle": total_vehicle_dets,
        "json_entries": len(all_detections),
        "processing_time_s": round(elapsed, 1),
        "avg_fps": round(frame_idx / elapsed, 1) if elapsed > 0 else 0,
    }

    print(f"\n  ✓ Done: {clip_name}")
    print(f"    Frames processed: {frame_idx}")
    print(f"    All detections:   {total_all_dets}")
    print(f"    Vehicle dets:     {total_vehicle_dets}")
    print(f"    JSON entries:     {len(all_detections)}")
    print(f"    Time: {elapsed:.1f}s ({summary['avg_fps']} fps)")
    print(f"    Video: {out_video_path}")
    print(f"    JSON:  {out_json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="RoadPulse Phase 1 — Batch detection inference"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing input video clips (.mp4, .avi, etc.)"
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Directory for annotated videos and detection JSON files"
    )
    parser.add_argument(
        "--model", default="yolo26n.pt",
        help="YOLO model to use (default: yolo26n.pt, fallback: yolo11n.pt)"
    )
    parser.add_argument(
        "--conf-thresh", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (default: 640)"
    )
    parser.add_argument(
        "--all-classes", action="store_true",
        help="Detect and save ALL COCO classes, not just vehicles"
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Draw boxes for ALL detected classes (not just vehicles)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  RoadPulse — Batch Detection Inference (Phase 1)")
    print("=" * 60)

    # Discover video clips
    clips = get_video_clips(args.data_dir)
    if not clips:
        print(f"\nERROR: No video files found in {args.data_dir}")
        print(f"  Supported formats: .mp4, .avi, .mov, .mkv, .webm")
        print(f"  Run download_data.py --sample-clips first, or place clips manually.")
        sys.exit(1)

    print(f"\nFound {len(clips)} video clip(s):")
    for c in clips:
        print(f"  • {os.path.basename(c)}")

    # Load model
    model = load_model(args.model)

    # Process each clip
    summaries = []
    for clip_path in clips:
        summary = process_clip(
            model, clip_path, args.out_dir,
            conf_thresh=args.conf_thresh,
            imgsz=args.imgsz,
            vehicle_only=not args.all_classes,
            show_all_classes=args.show_all,
        )
        if summary:
            summaries.append(summary)

    # Final report
    print("\n" + "=" * 60)
    print("  Batch Inference Summary")
    print("=" * 60)
    print(f"  Clips processed: {len(summaries)}/{len(clips)}")

    total_frames = sum(s["total_frames"] for s in summaries)
    total_veh = sum(s["total_detections_vehicle"] for s in summaries)
    total_time = sum(s["processing_time_s"] for s in summaries)

    print(f"  Total frames:    {total_frames}")
    print(f"  Total vehicle detections: {total_veh}")
    print(f"  Total time:      {total_time:.1f}s")
    if total_time > 0:
        print(f"  Overall FPS:     {total_frames / total_time:.1f}")

    print(f"\n  Output directory: {args.out_dir}")
    print(f"\n  Output files:")
    for s in summaries:
        print(f"    • {os.path.basename(s['output_video'])}")
        print(f"      {os.path.basename(s['output_json'])}")

    # Save batch summary JSON
    summary_path = os.path.join(args.out_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n  Batch summary: {summary_path}")

    print("\n  Phase 1 complete ✓")
    print()


if __name__ == "__main__":
    main()
