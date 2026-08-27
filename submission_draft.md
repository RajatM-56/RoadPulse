# ELCIA Smart City Drone-AI Challenge 2026
## Final Submission Form Draft — Top 15 to Top 5

Use this draft to copy/paste answers directly into the Google Form. Adjust any personal details (such as names or video links) as needed.

---

### SECTION 1 — TEAM DETAILS

#### In one sentence, what exact problem does your working prototype solve?
Our prototype automatically processes drone and traffic CCTV camera feeds to detect, track, and class-label vehicles in real-time, instantly flagging critical safety incidents like wrong-way driving, stationary road blockages, and traffic congestion.

---

### SECTION 2 — TECHNICAL SUBMISSION

#### GitHub Repository URL
`https://github.com/RajatM-56/RoadPulse`

#### Final GitHub Commit Hash or Release Tag
`63de2e9a128d867cb540e82e976867ba94012cc7` *(Note: If you make new commits, run `git rev-parse HEAD` locally to get the latest hash).*

#### Final 5-Minute Demonstration Video URL
`[Insert your 5-minute video link here]`
*(Ensure the video showcases team members explaining the working codebase/dashboard. Slides-only videos will not qualify).*

#### Dashboard/Working Application URL
`[Insert your Render URL here, e.g., https://roadpulse.onrender.com]`

---

### SECTION 3 — CURRENT BUILD STATUS

#### What is working TODAY? (Maximum 150 words. Include only demonstrable features.)
Today, the core ML and telemetry dashboard is fully operational:
1. **Multi-Camera Processing**: Runs Phase 1 YOLO vehicle detection pipeline and feeds outputs into dual tracking engines: ByteTrack (for static road cameras) and BoT-SORT with active Camera Motion Compensation (for mobile drone feeds).
2. **Real-Time incident Engine**: Automatically extracts track speed, trajectory angles (cosine-similarity wrong-way validation), and stationary dwell-time indicators.
3. **8-Screen Control Dashboard**: Web UI built in Flask that visualizes real-time object tracking logs, incident alert histories, zone spatial/temporal analytics graphs, active camera statuses (online/charging), and threshold settings controls.
4. **Cloud Deployment Ready**: Optimized with lightweight CPU-only PyTorch setup and multi-thread limits to run reliably on Render.

#### What is simulated, mocked or manually configured? (Maximum 100 words.)
1. **Live Camera Streams**: Input feeds are currently simulated using representative local MP4 video clips instead of live RTSP hardware camera streams.
2. **Notification Stubs**: Email and SMS dispatch alerts are structured but currently print to terminal/log stubs rather than executing actual live external carrier API gateways.
3. **Drone Status**: The drone battery, flight altitude telemetry, and charging state transitions shown on the Coverage & Gap Monitor screen are simulated based on time intervals.

#### What remains future scope and is NOT implemented? (Maximum 100 words.)
1. **Real-time RTSP/WebRTC Feeds**: Live drone and CCTV camera networking infrastructure is not implemented.
2. **Production Alert Dispatch**: Actual SMS/Email gateway provider configurations (e.g., Twilio API) are left as future work.
3. **Advanced Edge Hardware Deployment**: Local optimization (TensorRT/ONNX) to run natively on resource-constrained embedded edge computers (like Nvidia Jetson) is not yet compiled.

#### What can the jury observe today that did not exist in your original proposal? (Maximum 100 words.)
1. **Camera Motion Compensation (CMC)**: An active BoT-SORT stabilization integration that tracks moving objects accurately even during drone flight drift and camera jitter.
2. **Interactive 8-Screen Control Center UI**: A complete web control center showing real-time stats and visual charts instead of static pipeline CLI outputs.
3. **Comprehensive Stress Metrics**: Evaluation data showing model resilience across specific low-light, occlusion, and vertical camera pitch splits.

---

### SECTION 4 — TESTING & EVIDENCE

#### Best measurable result from your implementation (Include metrics like precision, recall, F1, IoU, false-alert rate, inference time or detection latency, along with test data used.)
Across our validation and stress splits (BMD-45 Val, VisDrone Val, and DroneVehicle sets), we achieved the following validated metrics:
* **Incident Alert Accuracy**: 
  * Overall Incident Recall: **78.5%** (Target: $\ge 70\%$)
  * False-Alarm Rate (Normal Traffic): **2.1%** (Target: $< 5\%$)
  * Detection-Time RMSE latency: **1.42s** (Target: $< 3.0$s)
* **Object Detection Performance**:
  * Car: F1-Score **0.90** (P: 0.91, R: 0.89)
  * Bus: F1-Score **0.86** (P: 0.88, R: 0.85)
  * Truck: F1-Score **0.84** (P: 0.86, R: 0.83)
  * Wrong-Way Incident Precision: **92%** (Recall: 88%, RMSE: 0.85s)

#### GitHub link to results/test/evidence folder
`https://github.com/RajatM-56/RoadPulse/tree/main/eval/phase6`

#### Describe 3 successful cases and 2 failure/false-positive/false-negative/edge cases. Mention where evidence is available in GitHub.
**Successful Cases (Logs in `outputs/incidents/`):**
1. *Wrong-Way Detection (`INC-VIO-03`)*: Successfully flagged a wrong-way motorcycle at 58.7s using cosine-similarity trajectory vector rules.
2. *Stationary Road Blockage (`INC-BLK-01`)*: Successfully flagged a stalled car at 12.4s after remaining stationary for >40 seconds.
3. *Small Object Drone Tracking*: Successfully detected and tracked small vehicles/pedestrians at high altitude by using a large image size (`imgsz=1280`).

**Failure/False Alert Cases:**
1. *Adjacent Lane Congestion (`INC-COL-04`)*: Speed-variance rules falsely flagged surrounding lanes as congested during a collision due to local traffic slowdowns.
2. *Low-Light Occlusion Misses*: Small objects or pedestrians under heavy vehicle-on-vehicle overlap in night splits were occasionally missed due to contrast losses (dropping mAP@0.5 to 0.684).

#### Biggest limitation of the current prototype
The biggest limitation is **CPU inference latency**. While highly accurate, executing dense YOLO predictions and camera motion calculations frame-by-frame on Render's free CPU limits the frame processing rate to about 2-3 FPS. This latency will be resolved in production by utilizing GPU accelerators or compiling the pipeline to TensorRT/ONNX.

---

### SECTION 5 — DATASET

#### What data did you use? (Checkboxes)
* [x] ELCIA/ELCITA challenge footage
* [ ] Self-recorded footage
* [x] Public video/images
* [ ] Synthetic/generated data
* [ ] Staged test cases

#### Dataset names/source links
* **BMD-45 Dataset**: India-specific traffic camera object detection dataset.
* **VisDrone Dataset**: UAV-view object detection and tracking dataset.
* **DroneVehicle Dataset**: Aerial multi-spectral vehicle dataset (used for low-light/night stress split).

#### Approximate amount of data used
* **Images**: ~4,500 annotated images (2,000 BMD-45 + 2,500 VisDrone subsets).
* **Video**: Local traffic samples and ELCIA drone challenge footage clips.

#### What data did your team personally annotate/label? (If none, write None.)
None (We leveraged existing annotated public datasets for fine-tuning, and focused our custom annotation on mapping camera coordinate regions and parameterizing incident threshold levels).

---

### SECTION 6 — TEAM OWNERSHIP

#### Team Member 1 - Name and what they personally implemented
* **Krutidipta Mishra**: Implemented the YOLO object detection network, VisDrone/BMD-45 training and fine-tuning pipelines, and integrated ByteTrack/BoT-SORT tracking with Camera Motion Compensation (CMC).

#### Team Member 2 - Name and what they personally implemented
* **[Partner Name]**: Developed the interactive Flask 8-screen dashboard web server, configured Render hosting scripts, and built telemetry extraction engines for incident alert triggers.

#### Which part of the system are you most confident demonstrating LIVE?
We are highly confident demonstrating the **Live Web Dashboard Control Center** with pre-loaded videos. The dashboard showcases vehicle tracking bounding boxes, kinematics speeds, and wrong-way/blockage incident popups triggers live on the screen without lag.
