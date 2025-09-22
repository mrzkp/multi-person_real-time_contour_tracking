# Multi-Person Real-Time Contour Tracking

Real-time pipeline to detect, track, and visualize multiple people with colored contours and persistent IDs. Logs events and per-frame tracking data for downstream analysis.

## Features

- Colored contour overlays (no boxes), consistent by ID
- Multi-object tracking via ByteTrack (default) or DeepSORT
- Live summary overlay with active ID count, ID list, average confidence, and FPS
- Trajectory tracing (optional)
- Entry/Exit event logging (JSONL)
- Per-frame tracking logs (JSONL)
- Snapshot export of first appearance per unique ID
- Output video standardization with configurable resize strategy: `fit` | `fill` | `stretch`
- File, webcam, or RTSP/HTTP stream input
- Optional live display window

## Quickstart

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

1. Download sample videos (Market Square and People Walking)

   ```bash
   python videos/input/download.py
   ```

1. Configure paths, thresholds, and tracker in `config.yaml`

   - File input (default): set `paths.source_path` to a video path.
   - Webcam: set `paths.source_path: "0"` (or another index). Display is hard-coded off; to enable the live window, set `DISPLAY = True` in `main.py`.
   - Stream: set `paths.source_path` to an `rtsp://` or `http(s)://` URL.

   Note: Render and YOLO inference settings are hard-coded in `main.py` (constants near the top).

1. Run

   ```bash
   python main.py
   ```

View the result file, e.g. on macOS:

```bash
open videos/output/result-people.mp4
```

Logs are written to the `logs/` directory. Snapshots are saved under `snapshots/<video_stem>/`.

## Tracker selection (ByteTrack by default)

This project supports both ByteTrack (default) and DeepSORT.

- To switch trackers, open `config.yaml` and set `tracker.type`.
- Supported values: `bytetrack` (default) or `deepsort`.

Example `config.yaml` excerpt:

```yaml
paths:
  source_path: "videos/input/people-walking.mp4"
  target_path: "videos/output/result-people.mp4"
  weights_path: "weights/outputs/yolov8n-seg.pt"
  logs_dir: "logs"
  snapshots_dir: "snapshots"

thresholds:
  detection_conf: 0.20
  match_iou: 0.40

tracker:
  type: bytetrack
  track_thresh: 0.40
  match_thresh: 0.55
  track_buffer: 150
```

Notes:

- DeepSORT is provided by `deep-sort-realtime` and uses an appearance embedder (PyTorch MobilenetV2 by default).
- ByteTrack is provided by Supervision and tends to be fast and robust for person tracking in many scenes.

## Example results: People Walking (ByteTrack vs DeepSORT)

The following sample outputs were generated from `videos/input/people-walking.mp4` using the two supported trackers.

- ByteTrack (default)

<!-- markdownlint-disable MD033 -->
<video src="videos/output/result-people.mp4" controls width="640"></video>

- DeepSORT

<video src="videos/output/result-people-deepsort.mp4" controls width="640"></video>
<!-- markdownlint-enable MD033 -->

If the embedded players do not render on GitHub, use the direct links below:

- [ByteTrack result (MP4)](videos/output/result-people.mp4)
- [DeepSORT result (MP4)](videos/output/result-people-deepsort.mp4)

## How it works

`main.py` orchestrates video capture (file/webcam/stream), inference, tracking, annotation, optional display, and writing the output video. It also ensures directories exist and names logs/snapshots robustly for non-file sources.

`utils.py` provides helpers for detection, tracking adapters, IoU/matching, annotation (polygons/labels/traces/summary overlay), event/snapshot logging, and output resizing.

## Assumptions

- Python 3.12+ with listed libraries; YOLOv8n-seg.pt auto-downloads if missing; no internet for package installation during runtime. YOLOv8n-seg.pt model is auto-downloaded if missing.
- Valid video sources; people as class ID 0; sufficient hardware for real-time.
- A valid config.yaml file exists with paths, thresholds, and tracker settings; defaults provided if missing (e.g., detection confidence 0.10, IoU 0.10).
- Assumes sufficient CPU/GPU for real-time processing; no explicit GPU acceleration beyond YOLO defaults.
- Frames are BGR format; detections and tracks are non-empty and properly formatted; no extreme occlusions or crowd densities that break tracking.
- Finally, assuming good faith in users for person tracking, aka, no advesarial inputs (e.g., malformed inputs).


## Limitations

- Tracking Robustness: Potential ID switching or loss during occlusions, fast motion, or similar appearances, as trackers (ByteTrack/DeepSort) rely on IoU and embeddings without advanced re-identification. No handling for long-term occlusions or re-entry.
- Performance: FPS may drop on low-end hardware due to segmentation (YOLOv8-seg) and annotations; no dynamic optimization (e.g., skipping frames). Fixed output resolution (1280x720) may distort or pad inputs.
- Customization Gaps: No integration with Roboflow for custom models (despite problem mention); hardcoded YOLO params (e.g., imgsz=640, max_det=300). Limited to people only; no multi-class support.
- Output and Logging: Logs are JSONL (not CSV); snapshots only on first detection (no updates). No error recovery (e.g., video read failures crash the script). Overlay summary clips long ID lists with a scrollbar but doesn't scroll interactively.
- Scaling: Struggles with high entity counts (>300 detections) due to YOLO limits; no distributed processing.

## Design Decisions

- Split into main.py (core loop, config loading, video I/O) and utils.py (helpers like detection, tracking, annotation) for readability and reusability; context managers for resource safety (e.g., video capture/release).
- Uses Supervision annotators for polygons (contours), labels (ID overlays), and traces (trajectories); color by track ID for consistency. Summary overlay is dynamic (wraps IDs, EMA-smoothed FPS, optional scrollbar) but fixed-position ("top_left") to avoid occlusion.
- Hardcoded YOLO params for speed (e.g., no half-precision by default); IoU-based matching for aligning tracks to detections.
- JSONL for frame-level tracks and events (enter/exit); timestamps derived from frame index/FPS for simplicity. Snapshots cropped to bbox on first sighting for minimal storage.
- "Fit" (letterbox) with black padding to preserve aspect ratio in outputs, prioritizing visual fidelity over filling the frame.

## Additional notes

The pipeline logs per-frame tracking data and entry/exit events in JSONL format under the `logs/` directory. It also saves a snapshot of each unique ID on first detection under `snapshots/<video_stem>/`.