#!/usr/bin/env python3
"""
Unit tests for Phase 5 Incident Engine and Phase 7 End-to-End Pipeline
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.incidents import IncidentDetector


def test_incident_detector_blockage():
    detector = IncidentDetector(blockage_dwell_thresh=10)
    mock_kinematics = {
        "1": {
            "class_name": "car",
            "final_speed_px_s": 5.0,
            "final_direction": [0.0, 0.0],
            "total_dwell_frames": 25,
            "history": [[100, 100], [101, 100]]
        }
    }
    incidents = detector.analyze_kinematics(mock_kinematics)
    assert len(incidents) >= 1
    assert any(inc["type"] == "Blockage & Obstruction" for inc in incidents)


def test_incident_detector_wrong_way():
    detector = IncidentDetector(wrong_way_cos_thresh=-0.5)
    detector.set_reference_lane_vector(1.0, 0.0)

    mock_kinematics = {
        "2": {
            "class_name": "car",
            "final_speed_px_s": 50.0,
            "final_direction": [-1.0, 0.0],  # Moving directly opposite to lane
            "total_dwell_frames": 30,
            "history": [[500, 200], [450, 200]]
        }
    }
    incidents = detector.analyze_kinematics(mock_kinematics)
    assert len(incidents) >= 1
    assert any("Wrong-Way" in inc["type"] for inc in incidents)
