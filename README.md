# RoadPulse — Traffic Incident Detection ML/CV Pipeline

> ELCIA Smart City Drone-AI Challenge 2026

Converts traffic camera and drone footage into automated incident alerts:
detects vehicles, tracks them, and flags incidents (congestion, blockage,
wrong-way driving, collision-linked congestion) with severity/confidence scores.

## Scope

This repo contains the **ML/CV pipeline only** — no dashboard, no UI, no
streaming. Everything runs in **batch mode**: a folder of pre-recorded video
clips goes in, annotated videos + structured incident logs come out.

## Project Structure

```
RoadPulse/
├── data/                  # Raw + processed dataset subsets
│   └── sample_clips/      # Small test videos
├── models/                # Saved checkpoints (one subfolder per phase)
├── pipeline/              # Detection, tracking, incident-logic code
│   ├── detect.py          # Batch detection inference (Phase 1)
│   └── utils.py           # Shared helpers
├── outputs/
│   ├── tracks/            # Tracked videos + kinematics logs
│   └── incidents/         # Incident JSON/CSV logs
├── eval/                  # Metrics reports, per phase
├── scripts/
│   ├── setup_env.py       # Environment check + install
│   └── download_data.py   # Dataset download
├── configs/
│   ├── bytetrack.yaml     # ByteTrack config (fixed-cam)
│   └── botsort_cmc.yaml   # BoT-SORT + CMC config (drone)
└── requirements.txt
```

## Quick Start

```bash
# 1. Setup environment
python scripts/setup_env.py

# 2. Get sample clips
python scripts/download_data.py --sample-clips --data-dir ./data

# 3. Run batch detection
python pipeline/detect.py \
    --data-dir ./data/sample_clips \
    --out-dir ./outputs/phase1
```

## Tech Stack

- **Detection:** YOLO26n via `ultralytics` (fallback: YOLO11n)
- **Tracking:** ByteTrack (fixed-cam), BoT-SORT + CMC (drone)
- **Incident logic:** Rule-based on tracking output

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✓ | Environment setup, batch detection inference |
| 2 | — | Fine-tuning on BMD-45, multi-object tracking |
| 3 | — | Incident classification + severity scoring |
| 4 | — | Evaluation + metrics reporting |
