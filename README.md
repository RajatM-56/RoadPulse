# RoadPulse — Traffic Incident Detection ML/CV Pipeline

> ELCIA Smart City Drone-AI Challenge 2026

Converts traffic camera and drone footage into automated incident alerts:
detects vehicles, tracks them, and flags incidents (congestion, blockage,
wrong-way driving, collision-linked congestion) with severity/confidence scores.

## Scope

This repo contains the **RoadPulse ML/CV pipeline** — featuring both **batch processing** (video folders → structured logs + annotated MP4s) and an interactive **Web Dashboard UI** ([`dashboard/server.py`](file:///Users/ratnamsmac/Documents/Projects/Elcia/dashboard/server.py)) for real-time telemetry, track inspection, and pipeline controls.

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

## Deployment (Render)

This repository includes a `render.yaml` blueprint configuration to deploy the interactive dashboard.

To deploy:
1. Sign in to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New** > **Blueprint**.
3. Connect your repository.
4. Render will automatically pick up the `render.yaml` configuration, pre-install the CPU-optimized PyTorch build to stay within memory limits, install dependencies, and launch the web server.

