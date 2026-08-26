#!/usr/bin/env python3
"""
RoadPulse Phase 7 — Unified End-to-End Pipeline Execution Script

Executes the complete RoadPulse ML/CV pipeline in a single unified run:
Input Video Clip -> YOLO Object Detection -> Multi-Object Tracking & Kinematics -> Rule-Based Incident Engine

Outputs:
- Tracked & Annotated Video (.mp4)
- Telemetry & Kinematics Log (.json)
- Incident Classification Alerts Log (.json)

Usage:
    python scripts/run_pipeline.py \
        --clip data/sample_clips/fixed_cam_sample.mp4 \
        --model yolo11n.pt \
        --tracker configs/bytetrack.yaml \
        --out-dir outputs/e2e
"""

import argparse
import os
import sys
import json
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.incidents import IncidentDetector


def main():
    parser = argparse.ArgumentParser(description="RoadPulse Phase 7 — End-to-End Pipeline")
    parser.add_argument("--clip", required=True, help="Path to input video clip")
    parser.add_argument("--model", default="yolo11n.pt", help="Path to YOLO model checkpoint")
    parser.add_argument("--tracker", default="configs/bytetrack.yaml", help="Path to tracker config")
    parser.add_argument("--out-dir", default="outputs/e2e", help="Output directory")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()

    print("=" * 65)
    print("  RoadPulse Phase 7 — Unified End-to-End ML Pipeline Execution")
    print("=" * 65)

    if not os.path.isfile(args.clip):
        print(f"ERROR: Video clip not found: {args.clip}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(args.clip))[0]

    # Step 1: Run Detection & Multi-Object Tracking
    print("\n[STEP 1/3] Running Object Detection & Multi-Object Tracking...")
    track_cmd = [
        sys.executable, "scripts/track.py",
        "--clip", args.clip,
        "--model", args.model,
        "--tracker", args.tracker,
        "--out-dir", args.out_dir,
        "--conf-thresh", str(args.conf_thresh)
    ]
    import subprocess
    proc = subprocess.run(track_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"ERROR in tracking step:\n{proc.stderr}")
        sys.exit(1)

    kinematics_json_path = os.path.join(args.out_dir, f"{clip_name}_kinematics.json")
    print(f"  ✓ Tracking complete -> {kinematics_json_path}")

    # Step 2: Run Incident Classification Engine
    print("\n[STEP 2/3] Running Rule-Based Incident Detection & Severity Scoring...")
    inc_cmd = [
        sys.executable, "scripts/detect_incidents.py",
        "--clip", args.clip,
        "--kinematics", kinematics_json_path,
        "--out-dir", args.out_dir
    ]
    proc2 = subprocess.run(inc_cmd, capture_output=True, text=True)
    if proc2.returncode != 0:
        print(f"ERROR in incident detection step:\n{proc2.stderr}")
        sys.exit(1)

    incidents_json_path = os.path.join(args.out_dir, f"{clip_name}_incidents.json")
    incidents_mp4_path = os.path.join(args.out_dir, f"{clip_name}_incidents.mp4")

    print(f"  ✓ Incident engine complete -> {incidents_json_path}")

    # Step 3: Pipeline Execution Summary
    print("\n[STEP 3/3] End-to-End Pipeline Summary")
    print("-" * 65)

    with open(incidents_json_path, "r") as f:
        inc_data = json.load(f)

    print(f"  Input Clip:           {args.clip}")
    print(f"  Total Incidents:      {inc_data.get('total_incidents', 0)}")
    print(f"  Annotated MP4:        {incidents_mp4_path}")
    print(f"  Kinematics Log JSON:  {kinematics_json_path}")
    print(f"  Incidents Log JSON:   {incidents_json_path}")
    print("=" * 65)
    print("  ✓ Pipeline Execution Completed Successfully")
    print("=" * 65)


if __name__ == "__main__":
    main()
