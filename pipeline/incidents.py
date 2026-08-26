#!/usr/bin/env python3
"""
RoadPulse Phase 5 — Rule-Based Incident Detection Engine

Implements four traffic incident classes:
1. Congestion & Flow (sustained low-speed vehicle clusters)
2. Blockage & Obstruction (stationary vehicles in drivable lane past duration threshold)
3. Traffic Violations (wrong-way driving direction vs reference lane vector)
4. Collision-Linked Congestion (abrupt cascading deceleration across adjacent tracks)

Outputs structured incident records with severity and confidence scores.
"""

import math
from typing import List, Dict, Any, Tuple, Optional


class IncidentDetector:
    """Rule-based incident classifier operating on vehicle kinematics telemetry."""

    def __init__(
        self,
        congestion_speed_thresh: float = 30.0,
        congestion_min_vehicles: int = 3,
        blockage_dwell_thresh: int = 40,
        wrong_way_cos_thresh: float = -0.6,
        decel_thresh_px_s: float = 80.0
    ):
        """
        Args:
            congestion_speed_thresh: Speed below which vehicle is considered congested (px/s).
            congestion_min_vehicles: Minimum vehicles in zone to trigger congestion.
            blockage_dwell_thresh: Minimum stationary frames to flag blockage.
            wrong_way_cos_thresh: Cosine similarity threshold for wrong-way direction vs lane vector.
            decel_thresh_px_s: Speed drop threshold indicating sudden deceleration.
        """
        self.congestion_speed_thresh = congestion_speed_thresh
        self.congestion_min_vehicles = congestion_min_vehicles
        self.blockage_dwell_thresh = blockage_dwell_thresh
        self.wrong_way_cos_thresh = wrong_way_cos_thresh
        self.decel_thresh_px_s = decel_thresh_px_s

        # Reference lane flow direction vector (default: rightwards [1.0, 0.0])
        self.reference_lane_vector = [1.0, 0.0]

    def set_reference_lane_vector(self, dx: float, dy: float):
        """Set expected traffic flow direction vector for violation checks."""
        mag = math.sqrt(dx**2 + dy**2)
        if mag > 0:
            self.reference_lane_vector = [dx / mag, dy / mag]

    def analyze_kinematics(self, kinematics_data: Dict[str, Any], fps: float = 25.0) -> List[Dict[str, Any]]:
        """
        Analyze kinematics history for all track IDs and detect incidents.

        Args:
            kinematics_data: Dict of track_id -> telemetry dict.
            fps: Video frames per second.

        Returns:
            List of incident records dicts.
        """
        incidents = []

        # Track lists for congestion checks
        slow_vehicles = []
        rapid_decel_tracks = []

        for track_id_str, item in kinematics_data.items():
            track_id = int(track_id_str)
            cls_name = item.get("class_name", "vehicle")
            speed = item.get("final_speed_px_s", 0.0)
            direction = item.get("final_direction", [0.0, 0.0])
            dwell_frames = item.get("total_dwell_frames", 0)
            history = item.get("history", [])

            last_pos = history[-1] if history else [0, 0]
            timestamp_sec = round(dwell_frames / fps, 2)

            # -------------------------------------------------------------
            # Rule 1: Blockage & Obstruction
            # -------------------------------------------------------------
            if dwell_frames >= self.blockage_dwell_thresh and speed < (self.congestion_speed_thresh * 0.5):
                severity = min(1.0, round(0.5 + (dwell_frames - self.blockage_dwell_thresh) / 100.0, 2))
                confidence = round(min(0.95, 0.7 + (dwell_frames / 150.0)), 2)

                incidents.append({
                    "incident_id": f"INC-BLK-{track_id}",
                    "type": "Blockage & Obstruction",
                    "track_id": track_id,
                    "class_name": cls_name,
                    "zone": f"Lane_Sector_({int(last_pos[0]//200)}, {int(last_pos[1]//200)})",
                    "timestamp_s": timestamp_sec,
                    "duration_frames": dwell_frames,
                    "location": [int(last_pos[0]), int(last_pos[1])],
                    "severity": severity,
                    "confidence": confidence,
                    "status": "active"
                })

            # -------------------------------------------------------------
            # Rule 2: Traffic Violations (Wrong-Way Driving)
            # -------------------------------------------------------------
            dir_x, dir_y = direction[0], direction[1]
            dir_mag = math.sqrt(dir_x**2 + dir_y**2)
            if dir_mag > 0.3 and speed > 15.0:
                dot_prod = (dir_x * self.reference_lane_vector[0]) + (dir_y * self.reference_lane_vector[1])
                if dot_prod < self.wrong_way_cos_thresh:
                    severity = min(1.0, round(0.6 + abs(dot_prod) * 0.35, 2))
                    confidence = round(min(0.92, 0.75 + (speed / 200.0)), 2)

                    incidents.append({
                        "incident_id": f"INC-VIO-{track_id}",
                        "type": "Traffic Violation (Wrong-Way)",
                        "track_id": track_id,
                        "class_name": cls_name,
                        "zone": f"Lane_Sector_({int(last_pos[0]//200)}, {int(last_pos[1]//200)})",
                        "timestamp_s": timestamp_sec,
                        "location": [int(last_pos[0]), int(last_pos[1])],
                        "direction": direction,
                        "severity": severity,
                        "confidence": confidence,
                        "status": "alert"
                    })

            # Collect slow vehicles for congestion analysis
            if speed < self.congestion_speed_thresh and dwell_frames > 15:
                slow_vehicles.append((track_id, last_pos, speed))

            # Analyze trajectory for abrupt deceleration
            if len(history) >= 10:
                # Calculate speed 10 frames ago vs current
                p1, p2, p3 = history[-10], history[-5], history[-1]
                v_initial = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) * (fps / 5)
                v_final = math.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2) * (fps / 4)
                if (v_initial - v_final) > self.decel_thresh_px_s:
                    rapid_decel_tracks.append((track_id, last_pos, v_initial - v_final))

        # -------------------------------------------------------------
        # Rule 3: Congestion & Flow (Cluster Check)
        # -------------------------------------------------------------
        if len(slow_vehicles) >= self.congestion_min_vehicles:
            avg_x = int(sum(v[1][0] for v in slow_vehicles) / len(slow_vehicles))
            avg_y = int(sum(v[1][1] for v in slow_vehicles) / len(slow_vehicles))
            avg_speed = round(sum(v[2] for v in slow_vehicles) / len(slow_vehicles), 1)

            severity = min(1.0, round(0.4 + (len(slow_vehicles) * 0.12), 2))
            confidence = round(min(0.98, 0.70 + (len(slow_vehicles) * 0.05)), 2)

            incidents.append({
                "incident_id": f"INC-CONG-ZONE-1",
                "type": "Congestion & Flow Reduction",
                "affected_tracks": [v[0] for v in slow_vehicles],
                "zone": f"Grid_Zone_({avg_x//250}, {avg_y//250})",
                "timestamp_s": round(max(v[2] for v in slow_vehicles if isinstance(v[2], (int, float))), 2),
                "location": [avg_x, avg_y],
                "avg_zone_speed_px_s": avg_speed,
                "severity": severity,
                "confidence": confidence,
                "status": "warning"
            })

        # -------------------------------------------------------------
        # Rule 4: Collision-Linked Congestion ("Possible Incident, Verify")
        # -------------------------------------------------------------
        if len(rapid_decel_tracks) >= 2:
            avg_x = int(sum(t[1][0] for t in rapid_decel_tracks) / len(rapid_decel_tracks))
            avg_y = int(sum(t[1][1] for t in rapid_decel_tracks) / len(rapid_decel_tracks))

            incidents.append({
                "incident_id": f"INC-COL-LINKED-1",
                "type": "Collision-Linked Congestion (Possible Incident, Verify)",
                "affected_tracks": [t[0] for t in rapid_decel_tracks],
                "zone": f"Sector_({avg_x//250}, {avg_y//250})",
                "location": [avg_x, avg_y],
                "severity": 0.85,
                "confidence": 0.65,  # Fuzziest, lowest confidence per spec
                "status": "verification_required"
            })

        return incidents
