#!/usr/bin/env python3
"""
RoadPulse — Aerial Drone Traffic Tracker
Uses MOG2 background subtraction + contour detection for top-down drone footage
where standard YOLO models fail to detect small vehicles.

Outputs:
  - outputs/tracks/drone_sample_tracked.mp4
  - outputs/tracks/drone_sample_kinematics.json
  - outputs/incidents/drone_sample_incidents.mp4
  - outputs/incidents/drone_sample_incidents.json
"""

import cv2
import numpy as np
import json
import math
import os
import sys
import time
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_VIDEO   = "data/sample_clips/drone_sample.mp4"
OUT_TRACK_VID = "outputs/tracks/drone_sample_tracked.mp4"
OUT_KIN_JSON  = "outputs/tracks/drone_sample_kinematics.json"
OUT_INC_VID   = "outputs/incidents/drone_sample_incidents.mp4"
OUT_INC_JSON  = "outputs/incidents/drone_sample_incidents.json"

MIN_AREA      = 150     # min contour area to count as a vehicle
MAX_AREA      = 8000    # ignore very large blobs (noise / shadows)
MAX_DIST      = 60      # max px distance to match track between frames
MIN_FRAMES    = 8       # track must persist N frames to be kept
TRAIL_LEN     = 50      # trail length for visualisation

# Incident thresholds
SLOW_SPEED_PX_S    = 18.0   # below this → congestion candidate
STOP_SPEED_PX_S    = 5.0    # near-zero → stopped / blockage
WRONG_WAY_ANGLE    = 150    # degrees, opposite to expected flow

COLORS = [
    (0,255,0),(0,200,255),(255,160,0),(255,0,180),
    (0,128,255),(128,255,0),(255,64,64),(0,255,200),
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def angle_between(dx, dy, ref_dx=1.0, ref_dy=0.0):
    """Angle in degrees between vector and reference direction."""
    dot  = dx*ref_dx + dy*ref_dy
    mag1 = math.hypot(dx, dy)
    mag2 = math.hypot(ref_dx, ref_dy)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    cos_a = max(-1, min(1, dot/(mag1*mag2)))
    return math.degrees(math.acos(cos_a))

def make_writer(path, fps, w, h):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

# ── Tracker ──────────────────────────────────────────────────────────────────
class CentroidTracker:
    def __init__(self):
        self.next_id  = 0
        self.tracks   = {}   # id -> {centroid, history, missed, color}

    def update(self, centroids):
        if not self.tracks:
            for c in centroids:
                self._new(c)
            return dict(self.tracks)

        # Greedy matching
        used_track = set()
        used_det   = set()
        pairs = []
        for i, c in enumerate(centroids):
            for tid, t in self.tracks.items():
                if tid in used_track:
                    continue
                d = dist(c, t["centroid"])
                if d < MAX_DIST:
                    pairs.append((d, i, tid))
        pairs.sort()

        for d, i, tid in pairs:
            if i in used_det or tid in used_track:
                continue
            self.tracks[tid]["centroid"] = centroids[i]
            self.tracks[tid]["history"].append(centroids[i])
            self.tracks[tid]["missed"]  = 0
            used_det.add(i)
            used_track.add(tid)

        # New tracks for unmatched detections
        for i, c in enumerate(centroids):
            if i not in used_det:
                self._new(c)

        # Age out missing tracks
        dead = [tid for tid, t in self.tracks.items()
                if tid not in used_track and t["missed"] > 5]
        for tid in dead:
            del self.tracks[tid]
        for tid in self.tracks:
            if tid not in used_track:
                self.tracks[tid]["missed"] += 1

        return dict(self.tracks)

    def _new(self, c):
        self.tracks[self.next_id] = {
            "centroid": c,
            "history":  [c],
            "missed":   0,
            "color":    COLORS[self.next_id % len(COLORS)],
            "frame_born": 0,
        }
        self.next_id += 1


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RoadPulse — Aerial Drone Traffic Tracker")
    print("=" * 60)

    if not os.path.isfile(INPUT_VIDEO):
        print(f"ERROR: {INPUT_VIDEO} not found"); sys.exit(1)

    cap  = cv2.VideoCapture(INPUT_VIDEO)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Input: {INPUT_VIDEO}  {W}x{H} @ {fps:.1f}fps  {TOTAL} frames")

    track_writer = make_writer(OUT_TRACK_VID, fps, W, H)
    inc_writer   = make_writer(OUT_INC_VID,   fps, W, H)

    fgbg    = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
    tracker = CentroidTracker()

    # Store final kinematics per track
    finished_kin  = {}   # tid -> list of history
    frame_idx     = 0
    t0            = time.time()

    # Expected traffic direction: horizontal / rightward
    REF_DX, REF_DY = 1.0, 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Background subtraction ──────────────────────────────────────────
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, (5, 5), 0)
        fgmask  = fgbg.apply(blur)

        # Remove shadows (value 127), keep foreground (255)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        boxes     = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > MAX_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
            centroids.append((cx, cy))
            boxes.append((x, y, w, h))

        # ── Track update ────────────────────────────────────────────────────
        tracks = tracker.update(centroids)

        # ── Annotate tracking video ─────────────────────────────────────────
        track_frame = frame.copy()

        # Draw a subtle dark header bar
        cv2.rectangle(track_frame, (0, 0), (W, 44), (10, 18, 40), -1)
        cv2.line(track_frame, (0, 44), (W, 44), (0, 220, 230), 1)
        cv2.putText(track_frame,
            f"RoadPulse | Drone Aerial Tracker | Frame {frame_idx}/{TOTAL} | Vehicles: {len(tracks)}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 230), 1)

        for tid, t in tracks.items():
            col = t["color"]
            cx, cy = int(t["centroid"][0]), int(t["centroid"][1])

            # Draw trail
            hist = t["history"][-TRAIL_LEN:]
            for j in range(1, len(hist)):
                alpha = j / len(hist)
                tc = tuple(int(c * alpha) for c in col)
                cv2.line(track_frame,
                         (int(hist[j-1][0]), int(hist[j-1][1])),
                         (int(hist[j][0]),   int(hist[j][1])),
                         tc, 2)

            # Draw bounding dot & ID
            cv2.circle(track_frame, (cx, cy), 6, col, -1)
            cv2.putText(track_frame, f"V{tid}",
                        (cx+8, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        track_writer.write(track_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            fps_real = frame_idx / elapsed
            print(f"  Frame {frame_idx}/{TOTAL}  |  {fps_real:.1f} fps  |  active tracks: {len(tracks)}")

    cap.release()
    track_writer.release()

    # ── Build kinematics JSON from all tracks ───────────────────────────────
    kin_out = {}
    for tid, t in tracker.tracks.items():
        h = t["history"]
        if len(h) < MIN_FRAMES:
            continue
        speeds = []
        for j in range(1, len(h)):
            d = dist(h[j], h[j-1])
            speeds.append(d * fps)

        avg_speed = np.mean(speeds) if speeds else 0.0
        min_speed = np.min(speeds)  if speeds else 0.0

        if len(h) >= 2:
            dx = h[-1][0] - h[0][0]
            dy = h[-1][1] - h[0][1]
        else:
            dx, dy = 0.0, 0.0

        kin_out[str(tid)] = {
            "track_id":       tid,
            "history":        [[int(x), int(y)] for x, y in h],
            "avg_speed_px_s": round(float(avg_speed), 2),
            "min_speed_px_s": round(float(min_speed), 2),
            "dwell_time_s":   round(len(h) / fps, 2),
            "direction":      [round(dx, 1), round(dy, 1)],
            "class_label":    "car",
        }

    os.makedirs(os.path.dirname(OUT_KIN_JSON), exist_ok=True)
    with open(OUT_KIN_JSON, "w") as f:
        json.dump(kin_out, f, indent=2)
    print(f"\n  ✓ Kinematics saved → {OUT_KIN_JSON}  ({len(kin_out)} tracks)")

    # ── Incident Detection Pass ─────────────────────────────────────────────
    incidents = []
    inc_id    = 1

    for tid, k in kin_out.items():
        avg_spd = k["avg_speed_px_s"]
        min_spd = k["min_speed_px_s"]
        hist    = k["history"]
        loc     = hist[len(hist)//2] if hist else [W//2, H//2]

        # Blockage / stopped vehicle
        if min_spd < STOP_SPEED_PX_S and k["dwell_time_s"] > 2.0:
            incidents.append({
                "incident_id":    f"INC-BLK-DRONE-{inc_id:02d}",
                "type":           "Blockage & Obstruction",
                "severity":       0.82,
                "confidence":     0.78,
                "zone":           "Wipro Avenue Drone Sector",
                "location":       loc,
                "timestamp_s":    round(len(hist) / fps, 1),
                "track_id":       int(tid),
                "lifecycle_status": "New",
            })
            inc_id += 1

        # Congestion / slow vehicle
        elif avg_spd < SLOW_SPEED_PX_S:
            incidents.append({
                "incident_id":    f"INC-CONG-DRONE-{inc_id:02d}",
                "type":           "Congestion & Flow Reduction",
                "severity":       0.65,
                "confidence":     0.71,
                "zone":           "Wipro Avenue Drone Sector",
                "location":       loc,
                "timestamp_s":    round(len(hist) / fps, 1),
                "track_id":       int(tid),
                "lifecycle_status": "New",
            })
            inc_id += 1

        # Wrong-way detection (moving strongly opposite to expected flow)
        dx, dy = k["direction"]
        ang = angle_between(dx, dy, REF_DX, REF_DY)
        if ang > WRONG_WAY_ANGLE and k["dwell_time_s"] > 1.5:
            incidents.append({
                "incident_id":    f"INC-VIO-DRONE-{inc_id:02d}",
                "type":           "Traffic Violation (Wrong-Way)",
                "severity":       0.90,
                "confidence":     0.82,
                "zone":           "Wipro Avenue Drone Sector",
                "location":       loc,
                "timestamp_s":    round(len(hist) / fps, 1),
                "track_id":       int(tid),
                "lifecycle_status": "New",
            })
            inc_id += 1

    # ── Render Incident Alert Overlay Video ─────────────────────────────────
    cap2 = cv2.VideoCapture(INPUT_VIDEO)
    frame_idx2 = 0
    while True:
        ret, frame = cap2.read()
        if not ret:
            break

        # Dark header bar
        cv2.rectangle(frame, (0, 0), (W, 54), (15, 23, 42), -1)
        cv2.line(frame, (0, 54), (W, 54), (0, 242, 254), 2)

        cv2.putText(frame,
            f"ROADPULSE INCIDENT ENGINE  |  DRONE AERIAL  |  ALERTS: {len(incidents)}",
            (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 242, 254), 2)
        cv2.putText(frame,
            f"Frame {frame_idx2}/{TOTAL}",
            (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 200, 255), 1)

        for inc in incidents:
            loc = inc.get("location", [W//2, H//2])
            sev = inc.get("severity", 0.5)
            color = (0, 0, 220) if sev > 0.75 else (0, 140, 255)

            # Pulsing rings
            r1 = 20 + int(5 * math.sin(frame_idx2 * 0.15))
            cv2.circle(frame, (loc[0], loc[1]), r1, color, 2)
            cv2.circle(frame, (loc[0], loc[1]), 5, color, -1)

            label = f"{inc['type'].split('(')[0].strip()} (Sev:{sev:.2f})"
            cv2.putText(frame, label,
                        (max(8, loc[0]-80), max(70, loc[1]-26)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        inc_writer.write(frame)
        frame_idx2 += 1

    cap2.release()
    inc_writer.release()

    os.makedirs(os.path.dirname(OUT_INC_JSON), exist_ok=True)
    with open(OUT_INC_JSON, "w") as f:
        json.dump({
            "clip_id":        "drone_sample",
            "total_incidents": len(incidents),
            "incidents":       incidents,
        }, f, indent=2)

    print(f"  ✓ Incident alerts saved → {OUT_INC_JSON}  ({len(incidents)} incidents)")
    print(f"  ✓ Annotated video saved → {OUT_INC_VID}")
    print("\n" + "=" * 60)
    print(f"  ✓ Pipeline complete  |  Tracks: {len(kin_out)}  |  Incidents: {len(incidents)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
