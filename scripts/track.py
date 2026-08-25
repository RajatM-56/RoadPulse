#!/usr/bin/env python3
"""
RoadPulse Phase 4 — Tracking and Kinematics Extraction

Runs YOLO object tracking (ByteTrack or BoT-SORT) on video clips, extracting
real-time kinematics (position history, speed, direction, dwell time).
Produces annotated tracking videos with ID trails and structured JSON logs.

Usage:
    python scripts/track.py \
        --clip data/sample_clips/fixed_cam_sample.mp4 \
        --model models/fixed_cam_best.pt \
        --tracker configs/bytetrack.yaml \
        --out-dir outputs/tracks
"""

import argparse
import os
import sys
import time
import json
import math
from collections import defaultdict

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.utils import VideoWriter, BOX_COLOURS, DEFAULT_BOX_COLOUR


def detect_device(requested: str = "auto") -> str:
    """Pick the best available device."""
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def calculate_kinematics(history: list, fps: float) -> dict:
    """
    Calculate speed, direction, and dwell time based on track history.
    Args:
        history: List of (cx, cy) centroids.
        fps: Frames per second.
    Returns:
        Dict with speed (pixels/sec), direction (dx, dy), dwell_time (frames).
    """
    dwell_time = len(history)
    
    if len(history) < 2:
        return {
            "speed_px_s": 0.0,
            "direction": [0.0, 0.0],
            "dwell_frames": dwell_time
        }
    
    # Use points over a short window to smooth velocity calculation
    # e.g., last point vs a point up to 5 frames ago
    window = min(5, len(history) - 1)
    pt1 = history[-1 - window]
    pt2 = history[-1]
    
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    
    dist_px = math.sqrt(dx**2 + dy**2)
    frames_passed = window
    
    speed_px_frame = dist_px / frames_passed
    speed_px_s = speed_px_frame * fps
    
    # Normalize direction vector
    if dist_px > 0:
        dir_x = dx / dist_px
        dir_y = dy / dist_px
    else:
        dir_x, dir_y = 0.0, 0.0
        
    return {
        "speed_px_s": round(speed_px_s, 2),
        "direction": [round(dir_x, 3), round(dir_y, 3)],
        "dwell_frames": dwell_time
    }


def main():
    parser = argparse.ArgumentParser(description="RoadPulse Phase 4 — Tracking")
    parser.add_argument("--clip", required=True, help="Path to input video clip")
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--tracker", required=True, help="Path to tracker config (.yaml)")
    parser.add_argument("--out-dir", default="outputs/tracks", help="Output directory")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    parser.add_argument("--device", default="auto", help="Device (auto, 0, mps, cpu)")
    parser.add_argument("--trail-length", type=int, default=60, help="Max length of visual trail")

    args = parser.parse_args()

    print("=" * 60)
    print("  RoadPulse Phase 4 — Tracking & Kinematics")
    print("=" * 60)

    if not os.path.isfile(args.clip):
        print(f"ERROR: Clip not found: {args.clip}")
        sys.exit(1)
    if not os.path.isfile(args.model) and not args.model.startswith("yolo"):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.tracker):
        print(f"ERROR: Tracker config not found: {args.tracker}")
        sys.exit(1)

    device = detect_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    from ultralytics import YOLO
    print(f"  Loading model: {args.model}")
    model = YOLO(args.model)
    class_names = model.names

    clip_name = os.path.splitext(os.path.basename(args.clip))[0]
    out_vid_path = os.path.join(args.out_dir, f"{clip_name}_tracked.mp4")
    out_json_path = os.path.join(args.out_dir, f"{clip_name}_kinematics.json")

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = VideoWriter(out_vid_path, fps, width, height)

    # Data structures for tracking
    track_history = defaultdict(list)
    final_kinematics = {}
    
    print(f"\n  Processing: {clip_name}")
    print(f"  Tracker:    {os.path.basename(args.tracker)}")
    print(f"  Video:      {width}x{height} @ {fps:.1f} fps")

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking using persist=True
        results = model.track(
            frame,
            conf=args.conf_thresh,
            imgsz=args.imgsz,
            tracker=args.tracker,
            persist=True,
            verbose=False,
            device=device,
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            clss = results.boxes.cls.int().cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
                x1, y1, x2, y2 = box
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                cls_name = class_names.get(cls_id, f"cls_{cls_id}")

                # Update history
                track_history[track_id].append((cx, cy))
                
                # Calculate kinematics
                kinematics = calculate_kinematics(track_history[track_id], fps)
                
                # Store latest state for JSON dump
                final_kinematics[int(track_id)] = {
                    "class_name": cls_name,
                    "final_speed_px_s": kinematics["speed_px_s"],
                    "final_direction": kinematics["direction"],
                    "total_dwell_frames": kinematics["dwell_frames"],
                    "history": track_history[track_id]
                }

                # Draw box
                colour = BOX_COLOURS.get(cls_id, DEFAULT_BOX_COLOUR)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)

                # Draw label with ID and Speed
                label = f"ID:{track_id} {cls_name} {kinematics['speed_px_s']:.0f}px/s"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

                # Draw trail
                points = np.array(track_history[track_id][-args.trail_length:], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colour, thickness=2)
                
        # Overlay frame info
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} | Active Tracks: {len(results.boxes) if results.boxes else 0}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
        writer.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            print(f"    Frame {frame_idx}/{total_frames} ({fps_actual:.1f} fps)")

    cap.release()
    writer.release()
    elapsed = time.time() - start_time

    # Save JSON log
    with open(out_json_path, "w") as f:
        json.dump(final_kinematics, f, indent=2)

    print(f"\n  ✓ Tracking Complete")
    print(f"    Time:          {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"    Unique Tracks: {len(final_kinematics)}")
    print(f"    Video:         {out_vid_path}")
    print(f"    Kinematics:    {out_json_path}")
    print()


if __name__ == "__main__":
    main()
