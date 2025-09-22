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
- Optional live summary window

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

This project supports both ByteTrack and DeepSORT.

- To switch trackers, open `config.yaml` and set `tracker.type`.
- Supported values: `bytetrack` or `deepsort`.

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

`main.py` is, essentially, the orchestrator for the entire repo. Here, extract recorded videos frame by frame and run inference via your designated model. This file calls `utils.py` file for helper methods for tracking, annotation, summary display, and logging. Furthermore, `utils.py` also enables proper video display options.

## Assumptions
- Assuming we are using Python 3.12+. 
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
- Scaling: Struggles with high entity counts (>300 detections) due to YOLO limits. Also no distributed processing.

## Design Decisions

- Split into main.py (core loop, config loading, video I/O) and utils.py (helpers like detection, tracking, annotation) for readability and reusability; context managers for resource safety (e.g., video capture/release).
- Uses Supervision annotators for polygons (contours), labels (ID overlays), and traces (trajectories); color by track ID for consistency. Summary overlay is dynamic (wraps IDs, EMA-smoothed FPS, optional scrollbar) but fixed-position ("top_left") to avoid occlusion.
- Hardcoded YOLO params for speed (e.g., no half-precision by default); IoU-based matching for aligning tracks to detections.
- JSONL for frame-level tracks and events (enter/exit); timestamps derived from frame index/FPS for simplicity. Snapshots cropped to bbox on first sighting for minimal storage.
- "Fit" (letterbox) with black padding to preserve aspect ratio in outputs, prioritizing visual fidelity over filling the frame.

## Additional notes

- You are, by default, provided YOLOv8n-seg.pt, however, I should note, you can easily replace said model with whatever model of your choosing in `main.py`.
- The pipeline logs per-frame tracking data and entry/exit events in JSONL format under the `logs/` directory. We also save a snapshot for each unique ID on first detection in the `snapshots/<video_stem>/` path.

## How would I would improve upon this project

Data acquisition and pre-processing is probably where most feature updates will occur in. Take this example, for creating curated videos for training examples for future models.

1. At a high level, say you have many drones that are constantly streaming video footage and are offloading it to some datastore(s). We will assume this datastore is some blob storage, like AWS S3.

2. We will assume there are n  such datastores. Typically, we want to perform some amount of pre-processing to these videos before we ran actually run the training loop. To do so on `n` datastores, we can utilize some distributed processing system, like Apache Spark, to process said videos. After processing the videos, we can delete them from the blob storage and  designate the processed videos to some other `m` datastores.

3. Given these `m` datastores of cleaned data, we can then run training loops on another server. This server will be optimized for compute, with multiple GPUs.

4. To efficiently feed data from the m datastores into the training server without downloading the entire dataset locally (minimizing storage costs and transfer times), we leverage on-demand streaming. We can prefetch data using PyTorch's DataLoader with prefetch_factor and num_workers > 1 for parallel batch loading. We can improve efficiency further by using tools like Apache Arrow, to read from optimized formats with some additional formatting on each item in the S3. This can be done from memory-mapped arrays to tensors, as an example.

5. In the actual training scripts, accommodate for remote paths in the Dataset class. Furthermore, change Dataloader to support pipelined, multi-threaded data ingestion (parallelism) across GPUs via PyTorch's DistributedDataParallel.

Later upgrades would revolve around automation of such processes and monitoring. That and real-time ingestion of streamed content.
