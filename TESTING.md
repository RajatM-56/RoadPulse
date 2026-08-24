# RoadPulse ML Pipeline — Comprehensive Testing Guide

This document outlines the step-by-step procedure to test the entire RoadPulse ML pipeline, covering Phase 1 (Batch Detection), Phase 2 (Fine-Tuning on BMD-45), and Phase 3 (Drone-Specific Fine-Tuning on VisDrone).

All testing is designed to be executed in a Python 3 environment. For Phase 2 and 3 training steps, a GPU (Google Colab T4/A100 or Apple Silicon MPS) is highly recommended.

---

## Prerequisites

1.  **Environment Setup**: Ensure you have a working Python environment (Python 3.12+ recommended).
2.  **Dependencies**: Install required packages.
    ```bash
    pip install -r requirements.txt
    ```
    *(If running in Colab, see the specific `!pip install` commands in the respective sections).*

---

## Phase 1: Batch Detection Inference (Smoke Test)

This phase verifies the core detection pipeline using a pretrained YOLO model on sample video clips.

### Step 1.1: Download Sample Data
Download the short sample video clips (fixed-camera and drone footage).

```bash
python scripts/download_data.py --sample-clips --data-dir ./data
```
*Expected Output*: Two video files (`fixed_cam_sample.mp4`, `drone_sample.mp4`) should be present in `data/sample_clips/`.

### Step 1.2: Run Batch Detection
Execute the detection script on the sample clips. This uses the baseline `yolo26n.pt` model.

```bash
python pipeline/detect.py \
    --data-dir ./data/sample_clips \
    --out-dir ./outputs/phase1 \
    --conf-thresh 0.25
```
*Expected Output*:
*   The script should process both clips successfully.
*   Check `outputs/phase1/` for:
    *   `*_annotated.mp4` (Videos with bounding boxes drawn).
    *   `*_detections.json` (Raw detection data).
    *   `batch_summary.json` (Overall summary).
*   *Visual Check*: Open the annotated MP4 files and verify that vehicles are being detected and highlighted with bounding boxes.

---

## Phase 2: Fixed-Camera Fine-Tuning (BMD-45 Dataset)

This phase tests the fine-tuning of the model on the India-specific BMD-45 dataset.

### Step 2.1: Prepare BMD-45 Data
Download a curated training subset (~2000 images) and the full validation set, and convert them to YOLO format.

```bash
python scripts/prepare_bmd45.py --data-dir ./data --n-train 2000
```
*Expected Output*:
*   `data/bmd45/train_mini/` and `data/bmd45/val/` populated with `.jpg` and `.txt` files.
*   `data/bmd45/data.yaml` generated containing 14 class names.

### Step 2.2: Fine-Tune Model
*(Note: Recommended to run on a machine with a GPU or Apple MPS).*

```bash
# On Mac with Apple Silicon (MPS):
python scripts/finetune_bmd45.py --data-yaml ./data/bmd45/data.yaml --epochs 5 --batch 8 --device mps

# On Google Colab (CUDA):
!python scripts/finetune_bmd45.py --data-yaml ./data/bmd45/data.yaml --epochs 20 --batch 16
```
*(For a quick local smoke test, use `--epochs 1` or `--epochs 2` just to verify the training loop runs without crashing).*
*Expected Output*: A new model weights file saved at `models/fixed_cam_best.pt`.

### Step 2.3: Evaluate Phase 2 Model
Evaluate the fine-tuned model against the BMD-45 validation split and compare it with the baseline COCO model.

```bash
python scripts/eval_bmd45.py \
    --model models/fixed_cam_best.pt \
    --data-yaml ./data/bmd45/data.yaml \
    --out-dir eval/phase2
```
*Expected Output*:
*   `eval/phase2/eval_report.md` generated.
*   *Verification*: Open the report. The fine-tuned mAP@0.5 should be reported. (If you ran very few epochs for a smoke test, the score might be low, but the script must complete).

### Step 2.4: Comparison Video
Generate a side-by-side comparison video.

```bash
python scripts/compare_video.py \
    --clip data/sample_clips/fixed_cam_sample.mp4 \
    --finetuned-model models/fixed_cam_best.pt \
    --out-dir outputs/phase2
```
*Expected Output*: `outputs/phase2/comparison_coco_vs_finetuned_fixed_cam_sample.mp4` generated. Left side shows COCO detections, right side shows BMD-45 detections.

---

## Phase 3: Drone/Aerial Fine-Tuning (VisDrone Dataset)

This phase tests adapting the model for drone-altitude footage using VisDrone, leveraging ProgLoss and STAL (via large image size).

### Step 3.1: Prepare VisDrone Data
Download a curated training subset (~2500 images) and the validation set.

```bash
python scripts/prepare_visdrone.py --data-dir ./data --n-train 2500
```
*Expected Output*:
*   `data/visdrone/train_mini/` and `data/visdrone/val/` populated.
*   `data/visdrone/data.yaml` generated containing 10 class names.

### Step 3.2: Fine-Tune Drone Model
*(Note: Requires a GPU with substantial VRAM due to `imgsz=1280`. For testing on weaker hardware, reduce `--imgsz` to 640 or 960, though 1280 is needed for best small-object performance).*

```bash
# On Google Colab (T4 GPU):
!python scripts/finetune_visdrone.py --data-yaml ./data/visdrone/data.yaml --epochs 25 --imgsz 1280 --batch 4
```
*(For a quick local smoke test, use `--epochs 1 --imgsz 640`).*
*Expected Output*: A new model weights file saved at `models/drone_best.pt`.

### Step 3.3: Evaluate Phase 3 Model
Evaluate the drone model on the VisDrone validation split.

```bash
python scripts/eval_visdrone.py \
    --model models/drone_best.pt \
    --data-yaml ./data/visdrone/data.yaml \
    --out-dir eval/phase3 \
    --imgsz 1280
```
*Expected Output*: `eval/phase3/eval_report.md` generated, highlighting performance on small-object classes (bicycle, tricycle, motor).

### Step 3.4: Extract Sample Frames
Generate annotated frames to visually verify small object detection.

```bash
python scripts/sample_frames_visdrone.py \
    --model models/drone_best.pt \
    --data-yaml ./data/visdrone/data.yaml \
    --out-dir outputs/phase3/sample_frames \
    --n-frames 5 \
    --imgsz 1280
```
*Expected Output*:
*   5 `.png` images saved in `outputs/phase3/sample_frames/`.
*   A `sample_index.md` file summarizing the detections.
*   *Visual Check*: Open the PNGs and look for the red diamond markers (◆) indicating that small objects (<32px) were successfully detected.

---
**Testing Complete.** If all scripts execute without errors and produce the expected output files, the pipeline is verified.
