#!/usr/bin/env python3
"""
RoadPulse Phase 5 — Incident Detection & Severity Scoring Script

Processes kinematics telemetry and input video clips to identify traffic incidents:
- Congestion & Flow Reduction
- Blockage & Obstruction
- Traffic Violations (Wrong-Way Driving)
- Collision-Linked Congestion

Outputs:
- Structured JSON log: outputs/incidents/<clip_name>_incidents.json
- Annotated alert video: outputs/incidents/<clip_name>_incidents.mp4

Usage:
    python scripts/detect_incidents.py \
        --clip data/sample_clips/fixed_cam_sample.mp4 \
        --kinematics outputs/tracks/fixed_cam_sample_kinematics.json \
        --out-dir outputs/incidents
"""

import argparse
import os
import sys
import json
import time

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.incidents import IncidentDetector
from pipeline.utils import VideoWriter


def main():
    parser = argparse.ArgumentParser(description="RoadPulse Phase 5 — Incident Classification")
    parser.add_argument("--clip", required=True, help="Path to input video clip")
    parser.add_argument("--kinematics", required=True, help="Path to kinematics JSON log")
    parser.add_argument("--out-dir", default="outputs/incidents", help="Output directory")
    parser.add_argument("--lane-dir-x", type=float, default=1.0, help="Expected lane traffic direction dx")
    parser.add_argument("--lane-dir-y", type=float, default=0.0, help="Expected lane traffic direction dy")

    args = parser.parse_args()

    print("=" * 60)
    print("  RoadPulse Phase 5 — Incident Detection & Severity Scoring")
    print("=" * 60)

    if not os.path.isfile(args.clip):
        print(f"ERROR: Video clip not found: {args.clip}")
        sys.exit(1)
    if not os.path.isfile(args.kinematics):
        print(f"ERROR: Kinematics JSON log not found: {args.kinematics}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.kinematics, "r") as f:
        kinematics_data = json.load(f)

    detector = IncidentDetector()
    detector.set_reference_lane_vector(args.lane_dir_x, args.lane_dir_y)

    print(f"  Clip:       {args.clip}")
    print(f"  Kinematics: {args.kinematics}")
    print(f"  Tracks:     {len(kinematics_data)} active tracks")

    # Run Incident Classifier
    incidents = detector.analyze_kinematics(kinematics_data)

    clip_basename = os.path.splitext(os.path.basename(args.clip))[0]
    out_json_path = os.path.join(args.out_dir, f"{clip_basename}_incidents.json")
    out_vid_path = os.path.join(args.out_dir, f"{clip_basename}_incidents.mp4")

    # Save structured incident records JSON
    with open(out_json_path, "w") as f:
        json.dump({
            "clip_id": clip_basename,
            "total_incidents": len(incidents),
            "incidents": incidents
        }, f, indent=2)

    # Render Incident Alert Overlay Video
    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = VideoWriter(out_vid_path, fps, width, height)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay Incident Alerts Banner if incidents detected
        if incidents:
            # Top Banner Background
            cv2.rectangle(frame, (0, 0), (width, 60), (15, 23, 42), -1)
            cv2.line(frame, (0, 60), (width, 60), (0, 242, 254), 2)

            # Title
            cv2.putText(frame, f"ROADPULSE INCIDENT ENGINE | ALERTS DETECTED: {len(incidents)}",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 242, 254), 2)

            # Draw incident location markers
            for inc in incidents:
                loc = inc.get("location", [width // 2, height // 2])
                inc_type = inc.get("type", "Incident")
                severity = inc.get("severity", 0.5)

                color = (0, 0, 255) if severity > 0.7 else (0, 165, 255)

                # Pulsing circle marker
                cv2.circle(frame, (loc[0], loc[1]), 25, color, 2)
                cv2.circle(frame, (loc[0], loc[1]), 5, color, -1)

                # Alert label
                label = f"{inc_type} (Sev:{severity:.2f})"
                cv2.putText(frame, label, (max(10, loc[0] - 80), max(80, loc[1] - 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\n  ✓ Incident Detection Complete")
    print(f"    Total Incidents Flagged: {len(incidents)}")
    print(f"    Incident Log JSON:        {out_json_path}")
    print(f"    Annotated Alert Video:    {out_vid_path}")
    print()


if __name__ == "__main__":
    main()
