import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
import time
import os
import json
from pathlib import Path
from utils import (
    detect_persons,
    track as track_people,
    build_labels,
    log_tracked,
    update_events_and_snapshots,
    annotate_frame,
    overlay_summary,
    make_bytetrack,
    make_deepsort,
    ensure_size,
)
import yaml
from contextlib import contextmanager, suppress

# Load configuration
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)
paths = CFG.get("paths", {})
thresholds = CFG.get("thresholds", {})
tracker_cfg = CFG.get("tracker", {})
SOURCE_PATH = paths.get("source_path", "videos/input/people-walking.mp4")
TARGET_PATH = paths.get("target_path", "videos/output/result.mp4")
WEIGHTS_PATH = paths.get("weights_path", "weights/yolov8n-seg.pt")
LOGS_DIR = paths.get("logs_dir", "logs")
SNAPSHOTS_DIR = paths.get("snapshots_dir", "snapshots")
DET_CONF = float(thresholds.get("detection_conf", 0.10))
MATCH_IOU = float(thresholds.get("match_iou", 0.10))
"""Use YOLOv8 segmentation model; Ultralytics will auto-download if not present"""
model = YOLO(WEIGHTS_PATH)
PERSON_CLASS_ID = 0
polygon_annotator = sv.PolygonAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(
    text_position=sv.Position.TOP_CENTER,
    color_lookup=sv.ColorLookup.TRACK,
)
trace_annotator = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK)


# Runtime state
SEEN_IDS: set[int] = set()         # IDs we have ever seen
ACTIVE_IDS_PREV: set[int] = set()  # IDs active in previous frame
PROC_FPS_EMA: float | None = None
FPS: float | None = None

def _infer_capture_source(src_val: str):
    """Return the file path and a stem name for logging/snapshots.
    Assumes src_val is always a local file path.
    """
    s = str(src_val).strip()
    try:
        stem = Path(s).stem
    except Exception:
        stem = "session"
    return s, stem


CAP_SRC, VIDEO_STEM = _infer_capture_source(SOURCE_PATH)
SNAP_DIR = os.path.join(SNAPSHOTS_DIR, VIDEO_STEM)
# Ensure directories
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)
tp_dir = os.path.dirname(TARGET_PATH)
if tp_dir:
    os.makedirs(tp_dir, exist_ok=True)

_tracker_type = str(tracker_cfg.get("type", "bytetrack")).lower()
tracker = None


def callback(frame, idx):
    """
    Per each frame in the source_path, returns annotations over frames to the target_path.
    """
    global SEEN_IDS, ACTIVE_IDS_PREV, PROC_FPS_EMA
    t0 = time.time()

    _, person_detections = detect_persons(
        model,
        frame,
        PERSON_CLASS_ID,
        conf=DET_CONF,
        predict_params={
            "imgsz": 640,
            "iou": 0.35,
            "max_det": 300,
            "classes": [0],
            "half": False,
            "agnostic_nms": False,
        },
    )
    tracked_detections = track_people(tracker, person_detections, frame)
    labels = build_labels(tracked_detections)
    ids_now, SEEN_IDS = update_events_and_snapshots(
        frame,
        tracked_detections,
        idx,
        FPS,
        SNAP_DIR,
        EVENTS_FP,
        SEEN_IDS,
        ACTIVE_IDS_PREV,
    )
    ACTIVE_IDS_PREV = ids_now
    log_tracked(LOG_FP, tracked_detections, idx, FPS)
    annotated_frame = annotate_frame(
        frame,
        tracked_detections,
        person_detections,
        labels,
        polygon_annotator,
        label_annotator,
        trace_annotator,
        iou_thresh=MATCH_IOU,
    )
    dt = time.time() - t0
    annotated_frame, PROC_FPS_EMA = overlay_summary(
        annotated_frame,
        tracked_detections,
        ids_now,
        dt,
        PROC_FPS_EMA,
        position="top_left",
        margin=10,
        max_width=420,
        max_height_frac=0.45,
        max_height_px_cap=220,
    )
    return annotated_frame


@contextmanager
def video_capture(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {src}")
    try:
        yield cap
    finally:
        cap.release()


@contextmanager
def video_writer(path, fourcc, fps, size):
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {path}")
    try:
        yield writer
    finally:
        writer.release()


log_path = os.path.join(LOGS_DIR, f"{VIDEO_STEM}.jsonl")
events_path = os.path.join(LOGS_DIR, f"{VIDEO_STEM}_events.jsonl")

with open(log_path, "w") as LOG_FP, open(events_path, "w") as EVENTS_FP:
    with video_capture(CAP_SRC) as cap:
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        FPS = cap_fps if cap_fps > 0 else 30.0

        # init tracker
        if tracker is None:
            if _tracker_type == "deepsort":
                tracker = make_deepsort(tracker_cfg)
            else:
                tracker = make_bytetrack(tracker_cfg, FPS)

        target_size = (1280, 720)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        with video_writer(TARGET_PATH, fourcc, FPS, target_size) as writer:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame = callback(frame, idx)
                out_frame = ensure_size(
                    annotated_frame,
                    target_w=1280,
                    target_h=720,
                    pad_color=(0, 0, 0),
                    strategy="fit",
                )

                writer.write(out_frame)
                idx += 1

