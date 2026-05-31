# Retail Shoplifting Anomaly Detection System

This project is a deep learning based shoplifting anomaly detection system built as a course project by **Soban Khan**, **Tahreem Fatima**, and **Basima Maham**.

The main goal of this project is to take a normal CCTV-style retail video, detect whether suspicious shoplifting activity is present, and then localize the most suspicious time window in the video. The system also extracts a short suspicious clip, a key frame, and shows the result through a simple web dashboard.

---

## Project Overview

Our system follows a two-stage anomaly detection pipeline:

```text
MP4 CCTV Video
→ I3D Feature Extraction
→ Stage 1: AnomalyNet Video-Level Detection
→ Stage 2: BiLSTM Temporal Localization
→ Suspicious Timestamp
→ Suspicious Clip + Key Frame
→ Web Dashboard Result
```

The project is designed to work as a demo product where a user can upload a retail video and receive a clear detection result.

---

## What the System Does

The system performs the following steps:

1. Takes an uploaded MP4 video.
2. Extracts real I3D video features from the input video.
3. Uses **Stage 1 AnomalyNet** to decide whether the full video is suspicious or normal.
4. Uses a corrected **Stage 2 BiLSTM model** to find the suspicious time window.
5. Generates:

   * Suspicious timestamp
   * Suspicious video clip
   * Suspicious key frame
   * Top-K suspicious time windows
   * JSON result
6. Shows everything in a frontend dashboard.

---

## Final Pipeline

The final product pipeline uses:

* **I3D feature extraction** for raw video processing
* **AnomalyNet** for video-level anomaly detection
* **Corrected Direct BiLSTM** for temporal localization
* **FastAPI backend** for API handling
* **React frontend** for the dashboard

The earlier S3D-based feature extraction was removed because it did not match the feature meaning expected by the trained model. The final version uses real I3D features only.

---

## Backend

The backend is built with **FastAPI**.

Main backend features:

* Upload MP4 videos
* Run the existing deep learning pipeline
* Store results by job ID
* Return detection results as JSON
* Serve generated media files
* Show previous detection history
* Delete previous detection records

Main endpoints:

```text
GET    /api/health
POST   /api/detect
GET    /api/results/{job_id}
GET    /api/history
GET    /api/media/{job_id}/{filename}
DELETE /api/results/{job_id}
```

---

## Frontend

The frontend is built using **React with Vite**.

Main frontend features:

* Video upload page
* Detection loading state
* Separate result page for each video
* Previous detection history
* Delete previous records
* Light mode and dark mode
* Dark mode with neon pink/cyber style
* Result dashboard with:

  * Alert badge
  * Video filename
  * Stage 1 score
  * Suspicious timestamp
  * Suspicious clip player
  * Key frame preview
  * Top-K suspicious windows
  * Simulated match profile card

---

## Simulated Person Matching

The project includes a simulated person matching feature for demo purposes only.

This is **not real face recognition**.

The system uses synthetic Pakistani demo profiles to show how a future dashboard could display matching information. This feature is only used to make the frontend/product demo more complete.

Example warning shown in the dashboard:

```text
Simulated match for demo only, not real face recognition.
```

---

## Model Performance Summary

The final corrected Stage 2 model improved temporal localization and reduced the earlier end-of-video bias.

Important final metrics:

```text
Stage 1 Video AUC: 0.979592
Stage 1 Accuracy: 0.928571
Stage 1 Precision: 0.875000
Stage 1 Recall: 1.000000
Stage 1 F1 Score: 0.933333

Corrected Stage 2 Temporal AUC: 0.924834
Corrected Stage 2 Video AUC: 0.988290
Peak overlap count: 10/21
Late-bin bias count: 0
```

For the known demo video `Shoplifting001_x264.mp4`, the corrected Stage 2 model predicts the suspicious window around:

```text
57.6s – 64.0s
```

which is close to the annotated suspicious region.

---

## How to Run the Backend

From the main project folder:

```bash
conda activate shoplifting
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload
```

Backend will run at:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

---

## How to Run the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend will usually run at:

```text
http://localhost:5173
```

---

## Example Backend Detection Command

The backend internally runs the existing detection pipeline like this:

```bash
python run_video_detection.py \
  --video path/to/video.mp4 \
  --device cpu \
  --stage2-checkpoint temporal_model_direct_bilstm_corrected.pth \
  --top-k 5 \
  --simulate-person-match
```

---

## Important Files

Core pipeline files:

```text
run_video_detection.py
final_two_stage_inference.py
extract_i3d_features_from_video_real.py
simulate_person_match.py
```

Backend files:

```text
backend/main.py
backend/jobs.py
backend/requirements.txt
```

Frontend folder:

```text
frontend/
```

Important model/checkpoint files:

```text
anomaly_net_weights.pth
temporal_model_direct_bilstm_corrected.pth
weights/i3d/rgb_imagenet.pt
```

---

## Notes

This project is built as an academic deep learning project and a product-style demo. The system can detect suspicious activity and localize a suspicious time window, but it is not a final real-world surveillance product.

For real-world deployment, more work would be needed, such as:

* Testing on more unseen CCTV videos
* Better multi-person tracking
* Real person detection and tracking
* Stronger privacy and ethics handling
* More robust deployment setup

---

## Group Members

* Soban Khan
* Tahreem Fatima
* Basima Maham

---

## Final Status

The core ML pipeline, backend API, and frontend dashboard are working together. The system can upload a video, run shoplifting anomaly detection, localize the suspicious part, generate useful outputs, and display everything in a clean dashboard.
