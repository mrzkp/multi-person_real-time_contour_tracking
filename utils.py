import os
import cv2
import json
import time
import numpy as np
import supervision as sv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from ultralytics import YOLO
import inspect
from deep_sort_realtime.deepsort_tracker import DeepSort


# ------------- Low-level helpers -------------


def draw_summary_overlay(
    scene: np.ndarray,
    active_ids: List[int],
    avg_conf: Optional[float],
    fps_cur: Optional[float],
    position: str = "top_left",
    margin: int = 10,
    max_width: int = 420,
    max_height_frac: float = 0.45,
    max_height_px_cap: Optional[int] = 220,
) -> np.ndarray:
    assert (
        isinstance(scene, np.ndarray) and scene.ndim == 3
    ), "scene must be an HxWxC image"
    assert isinstance(active_ids, list), "active_ids must be a list"
    assert isinstance(position, str), "position must be a string"
    assert isinstance(margin, int) and margin >= 0, "margin must be a non-negative int"
    assert isinstance(max_width, int) and max_width > 50, "max_width must be > 50"
    assert (
        0.1 <= max_height_frac <= 0.9
    ), "max_height_frac should be between 0.1 and 0.9"
    if max_height_px_cap is not None:
        assert (
            isinstance(max_height_px_cap, int) and max_height_px_cap > 50
        ), "max_height_px_cap must be > 50"
    overlay = scene.copy()
    h, w = scene.shape[:2]
    pad = int(margin)
    line_h = 22
    # Build dynamic lines with ID list wrapping to fit within the box width
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    def text_width(s: str) -> int:
        return cv2.getTextSize(s, font, font_scale, thickness)[0][0]

    # Prepare base lines (without IDs)
    pre_lines = [
        f"Tracked: {len(active_ids)}",
    ]
    # Prepare wrapped ID lines
    id_text = "IDs: " + ", ".join(map(str, active_ids))
    # We'll wrap based on available text width inside the box (reserve space for optional scrollbar)
    box_w = min(int(max_width), w - 2 * pad)
    sb_width = 6
    sb_margin = 4
    reserved_scroll_w = sb_width + sb_margin
    content_w = box_w - 2 * pad - reserved_scroll_w
    words = id_text.split(" ")
    id_lines: List[str] = []
    cur = ""
    for word in words:
        candidate = word if cur == "" else cur + " " + word
        if text_width(candidate) <= content_w:
            cur = candidate
        else:
            if cur:
                id_lines.append(cur)
            # If a single word is longer than content width, force split (rare for IDs)
            if text_width(word) > content_w:
                # naive hard split
                tmp = word
                while text_width(tmp) > content_w and len(tmp) > 1:
                    # find a cut that fits
                    for cut in range(len(tmp) - 1, 0, -1):
                        if text_width(tmp[:cut]) <= content_w:
                            id_lines.append(tmp[:cut])
                            tmp = tmp[cut:]
                            break
                    else:
                        break
                cur = tmp
            else:
                cur = word
    if cur:
        id_lines.append(cur)

    post_lines = [
        f"Avg conf: {avg_conf:.2f}" if avg_conf is not None else "Avg conf: n/a",
        f"FPS: {fps_cur:.1f}" if fps_cur is not None else "FPS: n/a",
    ]
    lines = pre_lines + id_lines + post_lines
    # Determine max height and clamp if too many lines; show scroll indicator and bar
    # Use a fixed pixel cap to keep the overlay size relatively consistent across resolutions.
    max_box_h_frac = min(int(max_height_frac * h), h - 2 * pad)
    if max_height_px_cap is not None:
        max_box_h = min(max_box_h_frac, max_height_px_cap)
    else:
        max_box_h = max_box_h_frac
    max_lines = max(3, int((max_box_h - 2 * pad) / line_h))
    show_lines = lines
    clipped = False
    if len(lines) > max_lines:
        clipped = True
        remaining = len(lines) - (max_lines - 1)
        show_lines = lines[: max_lines - 1] + [f"... (+{remaining} more)"]
    box_h = pad + line_h * len(show_lines) + pad

    # coordinattes
    pos = position.lower().replace("-", "_")
    x1 = pad if ("left" in pos) else (w - pad - box_w)
    y1 = pad if ("top" in pos) else (h - pad - box_h)
    x2, y2 = x1 + box_w, y1 + box_h

    # make the background translucent
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    alpha = 0.4
    scene = cv2.addWeighted(overlay, alpha, scene, 1 - alpha, 0)
    y = y1 + pad + 16
    for text in show_lines:
        cv2.putText(
            scene,
            text,
            (x1 + pad, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_h

    # Draw a simple scrollbar if clipped
    if clipped:
        groove_x1 = x2 - sb_margin - sb_width
        groove_x2 = x2 - sb_margin
        groove_y1 = y1 + pad
        groove_y2 = y2 - pad
        cv2.rectangle(
            scene, (groove_x1, groove_y1), (groove_x2, groove_y2), (200, 200, 200), 1
        )
        groove_h = groove_y2 - groove_y1
        knob_h = max(8, int(groove_h * (len(show_lines) / len(lines))))
        knob_y1 = groove_y1
        knob_y2 = min(groove_y1 + knob_h, groove_y2)
        cv2.rectangle(
            scene,
            (groove_x1 + 1, knob_y1 + 1),
            (groove_x2 - 1, knob_y2 - 1),
            (255, 255, 255),
            -1,
        )
    return scene


def iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    assert (
        isinstance(a_xyxy, np.ndarray) and a_xyxy.shape[-1] == 4
    ), "a_xyxy must be Nx4"
    assert (
        isinstance(b_xyxy, np.ndarray) and b_xyxy.shape[-1] == 4
    ), "b_xyxy must be Mx4"
    if a_xyxy.size == 0 or b_xyxy.size == 0:
        return np.zeros((len(a_xyxy), len(b_xyxy)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = np.split(a_xyxy, 4, axis=1)
    bx1, by1, bx2, by2 = np.split(b_xyxy, 4, axis=1)
    inter_x1 = np.maximum(ax1, bx1.T)
    inter_y1 = np.maximum(ay1, by1.T)
    inter_x2 = np.minimum(ax2, bx2.T)
    inter_y2 = np.minimum(ay2, by2.T)
    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter = inter_w * inter_h
    area_a = np.clip((ax2 - ax1), 0, None) * np.clip((ay2 - ay1), 0, None)
    area_b = np.clip((bx2 - bx1), 0, None) * np.clip((by2 - by1), 0, None)
    union = area_a + area_b.T - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def match_tracked_to_person(
    tracked: sv.Detections, persons: sv.Detections, iou_thresh: float = 0.1
) -> Tuple[List[int], List[int]]:
    assert isinstance(tracked, sv.Detections), "tracked must be sv.Detections"
    assert isinstance(persons, sv.Detections), "persons must be sv.Detections"
    assert 0.0 <= iou_thresh <= 1.0, "iou_thresh must be between 0 and 1"
    if len(tracked) == 0 or len(persons) == 0:
        return [], []
    ious = iou_matrix(tracked.xyxy, persons.xyxy)
    candidates = []
    for ti in range(ious.shape[0]):
        pj = int(np.argmax(ious[ti]))
        candidates.append((ti, pj, float(ious[ti, pj])))
    candidates.sort(key=lambda x: x[2], reverse=True)
    used_t = set()
    used_p = set()
    keep_indices: List[int] = []
    keep_ids: List[int] = []
    for ti, pj, iou in candidates:
        if iou < iou_thresh:
            continue
        if ti in used_t or pj in used_p:
            continue
        used_t.add(ti)
        used_p.add(pj)
        keep_indices.append(pj)
        tid = tracked.tracker_id[ti]
        keep_ids.append(int(tid) if tid is not None else -1)
    return keep_indices, keep_ids


# ------------- Pipeline helpers -------------


def timestamp(idx: int, fps: Optional[float]) -> Optional[float]:
    assert isinstance(idx, int) and idx >= 0, "idx must be non-negative int"
    return (float(idx) / float(fps)) if fps not in (None, 0) else None


def detect_persons(
    model: YOLO,
    frame: np.ndarray,
    person_class_id: int,
    conf: float = 0.10,
    predict_params: Optional[Dict[str, Any]] = None,
) -> Tuple[sv.Detections, sv.Detections]:
    assert isinstance(model, YOLO), "model must be an ultralytics.YOLO instance"
    assert (
        isinstance(frame, np.ndarray) and frame.ndim == 3
    ), "frame must be an HxWxC image"
    assert (
        isinstance(person_class_id, int) and person_class_id >= 0
    ), "person_class_id must be non-negative int"
    assert 0.0 <= conf <= 1.0, "conf must be in [0,1]"
    assert predict_params is None or isinstance(
        predict_params, dict
    ), "predict_params must be a dict or None"
    kwargs = dict(retina_masks=True, conf=conf)
    if predict_params:
        kwargs.update(predict_params)
    results = model.predict(frame, **kwargs)[0]
    detections = sv.Detections.from_ultralytics(results)
    person_mask = detections.class_id == person_class_id
    person_detections = detections[person_mask]
    return detections, person_detections


def make_bytetrack(tracker_cfg: Optional[Dict[str, Any]], fps: int) -> sv.ByteTrack:
    """Create a ByteTrack instance, filtering kwargs based on the installed supervision version.

    This avoids TypeError for unsupported __init__ kwargs across versions.
    """
    assert isinstance(fps, (int, float)) and fps >= 0, "fps must be >= 0"
    tracker_cfg = tracker_cfg or {}
    try:
        sig = inspect.signature(sv.ByteTrack)
        allowed = set(sig.parameters.keys())
    except Exception:
        allowed = set()
    kwargs: Dict[str, Any] = {}
    if "frame_rate" in allowed:
        kwargs["frame_rate"] = int(fps)
    if "track_thresh" in allowed and "track_thresh" in tracker_cfg:
        kwargs["track_thresh"] = float(tracker_cfg.get("track_thresh"))
    if "match_thresh" in allowed and "match_thresh" in tracker_cfg:
        kwargs["match_thresh"] = float(tracker_cfg.get("match_thresh"))
    if "track_buffer" in allowed and "track_buffer" in tracker_cfg:
        kwargs["track_buffer"] = int(tracker_cfg.get("track_buffer"))
    try:
        return sv.ByteTrack(**kwargs) if kwargs else sv.ByteTrack()
    except TypeError:
        # Fall back to default constructor if kwargs incompatible
        return sv.ByteTrack()


def make_deepsort(tracker_cfg: Optional[Dict[str, Any]]) -> DeepSort:
    """Create a DeepSort instance, filtering kwargs based on the installed deep-sort-realtime version.

    This avoids TypeError for unsupported __init__ kwargs across versions.
    """
    tracker_cfg = tracker_cfg or {}
    try:
        sig = inspect.signature(DeepSort)
        allowed = set(sig.parameters.keys())
    except Exception:
        allowed = set()
    # Known params and their casters
    casters: Dict[str, Any] = {
        "max_age": int,
        "n_init": int,
        "nn_budget": int,
        "max_iou_distance": float,
        "embedder": str,
        "embedder_gpu": bool,
        "half": bool,
        "bgr": bool,
        "polygon": bool,
        "max_cosine_distance": float,
        "nms_max_overlap": float,
        "ema_alpha": float,
    }
    kwargs: Dict[str, Any] = {}
    for k, v in (tracker_cfg or {}).items():
        if k in allowed:
            caster = casters.get(k, None)
            try:
                kwargs[k] = caster(v) if caster is not None and v is not None else v
            except Exception:
                kwargs[k] = v
    try:
        return DeepSort(**kwargs) if kwargs else DeepSort()
    except TypeError:
        # Fallback to default constructor if kwargs incompatible
        return DeepSort()


def track(
    tracker: Any, person_detections: sv.Detections, frame: Optional[np.ndarray] = None
) -> sv.Detections:
    """Update tracker state with current detections and return tracks as sv.Detections.

    Supports both Supervision's ByteTrack (update_with_detections) and deep-sort-realtime's
    DeepSort (update_tracks). For DeepSort, we pass the current frame (if provided)
    to enable on-the-fly embedding extraction.
    """
    assert isinstance(
        person_detections, sv.Detections
    ), "person_detections must be sv.Detections"
    if hasattr(tracker, "update_with_detections"):
        return tracker.update_with_detections(person_detections)

    if hasattr(tracker, "update_tracks"):
        xyxy = person_detections.xyxy
        bbs: List[Tuple[List[float], float, int]] = []
        confs = (
            person_detections.confidence
            if getattr(person_detections, "confidence", None) is not None
            else [None] * len(person_detections)
        )
        class_ids = (
            person_detections.class_id
            if getattr(person_detections, "class_id", None) is not None
            else [0] * len(person_detections)
        )
        for i in range(len(person_detections)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            conf_val = (
                float(confs[i]) if (confs is not None and confs[i] is not None) else 1.0
            )
            cls_val = (
                int(class_ids[i])
                if (class_ids is not None and class_ids[i] is not None)
                else 0
            )
            bbs.append(([x1, y1, w, h], conf_val, cls_val))

        tracks = tracker.update_tracks(bbs, frame=frame)

        xyxy_list: List[List[float]] = []
        conf_list: List[Optional[float]] = []
        id_list: List[int] = []
        for t in tracks:
            # Only keep confirmed tracks
            if hasattr(t, "is_confirmed") and not t.is_confirmed():
                continue
            # Get bbox as ltrb (xyxy)
            try:
                l, top, r, b = t.to_ltrb()
            except TypeError:
                # Older versions may have different signatures
                l, top, r, b = t.to_ltrb
            xyxy_list.append([float(l), float(top), float(r), float(b)])
            # Extract confidence if available
            conf_val: Optional[float] = None
            if hasattr(t, "det_conf") and t.det_conf is not None:
                try:
                    conf_val = float(t.det_conf)
                except Exception:
                    conf_val = None
            elif hasattr(t, "get_det_supplementary"):
                try:
                    supp = t.get_det_supplementary()
                    if isinstance(supp, dict):
                        for key in ("det_conf", "conf", "confidence"):
                            if key in supp and supp[key] is not None:
                                conf_val = float(supp[key])
                                break
                except Exception:
                    pass
            conf_list.append(conf_val)
            # Track ID
            tid = getattr(t, "track_id", None)
            id_list.append(int(tid) if tid is not None else -1)

        # Build Supervision Detections
        if len(xyxy_list) == 0:
            dets_empty = sv.Detections(xyxy=np.empty((0, 4), dtype=np.float32))
            dets_empty.tracker_id = np.empty((0,), dtype=int)
            return dets_empty

        dets = sv.Detections(xyxy=np.array(xyxy_list, dtype=np.float32))
        # Assign tracker IDs
        dets.tracker_id = np.array(id_list, dtype=int)
        # Assign confidences if any available
        if any(c is not None for c in conf_list):
            dets.confidence = np.array(
                [c if c is not None else np.nan for c in conf_list], dtype=np.float32
            )
        return dets

    raise AssertionError(
        "Unsupported tracker type: expected DeepSort.update_tracks or ByteTrack.update_with_detections"
    )


def build_labels(tracked_detections: sv.Detections) -> List[str]:
    n = len(tracked_detections)
    tids = getattr(tracked_detections, "tracker_id", None)

    # return placeholder labels of correct length if no detections, also print to terminal.
    if tids is None or (hasattr(tids, "__len__") and len(tids) != n):
        print("NO DETECTIONS.")
        return ["ID ?"] * n

    labels: List[str] = []
    for tid in tids:
        try:
            if tid is None:
                labels.append("ID ?")
            else:
                # handle nan values if present
                if isinstance(tid, float) and np.isnan(tid):
                    labels.append("ID ?")
                else:
                    labels.append(f"ID {int(tid)}")
        except Exception:
            labels.append("ID ?")
    return labels


def log_tracked(
    log_fp, tracked_detections: sv.Detections, idx: int, fps: Optional[float]
) -> None:
    if log_fp is None or len(tracked_detections) == 0:
        return
    confidences = (
        tracked_detections.confidence
        if getattr(tracked_detections, "confidence", None) is not None
        else [None] * len(tracked_detections)
    )
    for xyxy, conf, tracker_id in zip(
        tracked_detections.xyxy, confidences, tracked_detections.tracker_id
    ):
        record = {
            "frame": int(idx),
            "timestamp": timestamp(idx, fps),
            "id": int(tracker_id) if tracker_id is not None else None,
            "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
            "confidence": float(conf) if conf is not None else None,
        }
        log_fp.write(json.dumps(record) + "\n")


def update_events_and_snapshots(
    frame: np.ndarray,
    tracked_detections: sv.Detections,
    idx: int,
    fps: Optional[float],
    snap_dir: str,
    events_fp,
    seen_ids: Set[int],
    active_ids_prev: Set[int],
) -> Tuple[Set[int], Set[int]]:
    assert (
        isinstance(frame, np.ndarray) and frame.ndim == 3
    ), "frame must be an HxWxC image"
    assert (
        isinstance(snap_dir, str) and len(snap_dir) > 0
    ), "snap_dir must be a non-empty string"
    assert (
        hasattr(events_fp, "write") or events_fp is None
    ), "events_fp must be file-like or None"
    assert isinstance(seen_ids, set), "seen_ids must be a set"
    assert isinstance(active_ids_prev, set), "active_ids_prev must be a set"
    tids = getattr(tracked_detections, "tracker_id", None)
    ids_now: Set[int] = (
        set(int(tid) for tid in tids if tid is not None)
        if (tids is not None and hasattr(tids, "__iter__"))
        else set()
    )
    new_ids = ids_now - active_ids_prev
    exited_ids = active_ids_prev - ids_now

    # ID->bbox mapping
    id_to_bbox: Dict[int, Any] = {}
    if tids is not None and hasattr(tids, "__iter__"):
        for xyxy, tid in zip(tracked_detections.xyxy, tids):
            if tid is not None:
                id_to_bbox[int(tid)] = xyxy
    ts = timestamp(idx, fps)

    # events + snapshots into json + creating .jpg
    if new_ids:
        for tid in sorted(new_ids):
            if events_fp is not None:
                events_fp.write(
                    json.dumps(
                        {
                            "event": "enter",
                            "frame": int(idx),
                            "timestamp": ts,
                            "id": tid,
                        }
                    )
                    + "\n"
                )
            if tid not in seen_ids and tid in id_to_bbox:
                x1, y1, x2, y2 = id_to_bbox[tid]
                H, W = frame.shape[:2]
                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                xi2, yi2 = min(W, int(x2)), min(H, int(y2))
                if xi2 > xi1 and yi2 > yi1:
                    crop = frame[yi1:yi2, xi1:xi2]
                    os.makedirs(snap_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(snap_dir, f"id_{tid}.jpg"), crop)
        seen_ids.update(new_ids)

    # Exit events
    if exited_ids:
        for tid in sorted(exited_ids):
            if events_fp is not None:
                events_fp.write(
                    json.dumps(
                        {"event": "exit", "frame": int(idx), "timestamp": ts, "id": tid}
                    )
                    + "\n"
                )

    return ids_now, seen_ids


def annotate_frame(
    frame: np.ndarray,
    tracked_detections: sv.Detections,
    person_detections: sv.Detections,
    labels: List[str],
    polygon_annotator: sv.PolygonAnnotator,
    label_annotator: Optional[sv.LabelAnnotator],
    trace_annotator: Optional[sv.TraceAnnotator],
    iou_thresh: float = 0.1,
) -> np.ndarray:
    assert (
        isinstance(frame, np.ndarray) and frame.ndim == 3
    ), "frame must be an HxWxC image"
    assert isinstance(
        tracked_detections, sv.Detections
    ), "tracked_detections must be sv.Detections"
    assert isinstance(
        person_detections, sv.Detections
    ), "person_detections must be sv.Detections"
    assert isinstance(labels, list), "labels must be a list"
    assert 0.0 <= iou_thresh <= 1.0, "iou_thresh must be between 0 and 1"
    annotated = frame.copy()
    keep_idx, keep_ids = match_tracked_to_person(
        tracked_detections, person_detections, iou_thresh=iou_thresh
    )
    if len(keep_idx) > 0:
        persons_matched = person_detections[keep_idx]
        persons_matched.tracker_id = np.array(keep_ids, dtype=int)
        annotated = polygon_annotator.annotate(annotated, detections=persons_matched)
    if len(tracked_detections) > 0:
        if label_annotator is not None:
            annotated = label_annotator.annotate(
                annotated, detections=tracked_detections, labels=labels
            )
        if trace_annotator is not None:
            annotated = trace_annotator.annotate(
                annotated, detections=tracked_detections
            )
    return annotated


def overlay_summary(
    annotated: np.ndarray,
    tracked_detections: sv.Detections,
    ids_now: Set[int],
    dt: float,
    proc_fps_ema: Optional[float],
    position: str = "top_left",
    margin: int = 10,
    max_width: int = 420,
    max_height_frac: float = 0.45,
    max_height_px_cap: Optional[int] = 220,
) -> Tuple[np.ndarray, Optional[float]]:
    assert (
        isinstance(annotated, np.ndarray) and annotated.ndim == 3
    ), "annotated must be an HxWxC image"
    assert isinstance(
        tracked_detections, sv.Detections
    ), "tracked_detections must be sv.Detections"
    assert isinstance(ids_now, set), "ids_now must be a set"
    assert isinstance(dt, float) and dt >= 0.0, "dt must be non-negative float"
    fps_inst = (1.0 / dt) if dt > 0 else None
    if fps_inst is not None:
        proc_fps_ema = (
            fps_inst if proc_fps_ema is None else (0.9 * proc_fps_ema + 0.1 * fps_inst)
        )
    conf_overlay = None
    if (
        hasattr(tracked_detections, "confidence")
        and tracked_detections.confidence is not None
    ):
        conf_list = [float(c) for c in tracked_detections.confidence if c is not None]
        if conf_list:
            conf_overlay = float(np.mean(conf_list))
    assert isinstance(position, str), "position must be a string"
    assert isinstance(margin, int) and margin >= 0, "margin must be a non-negative int"
    assert isinstance(max_width, int) and max_width > 50, "max_width must be > 50"
    assert (
        0.1 <= max_height_frac <= 0.9
    ), "max_height_frac should be between 0.1 and 0.9"
    if max_height_px_cap is not None:
        assert (
            isinstance(max_height_px_cap, int) and max_height_px_cap > 50
        ), "max_height_px_cap must be > 50"
    annotated = draw_summary_overlay(
        annotated,
        active_ids=sorted(list(ids_now)),
        avg_conf=conf_overlay,
        fps_cur=proc_fps_ema,
        position=position,
        margin=margin,
        max_width=max_width,
        max_height_frac=max_height_frac,
        max_height_px_cap=max_height_px_cap,
    )
    return annotated, proc_fps_ema


def ensure_size(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
    pad_color: tuple[int, int, int] = (0, 0, 0),
    strategy: str = "fit",
) -> np.ndarray:
    """Resize frame to exact target resolution using letterbox padding (fit without distortion).

    - Scales the image to fit entirely within the target size, preserving aspect ratio.
    - Pads the remaining area with pad_color so the final frame is exactly (target_h, target_w).
    """
    assert isinstance(frame, np.ndarray) and frame.ndim == 3, "frame must be HxWxC"
    assert isinstance(target_w, int) and target_w > 0, "target_w must be > 0"
    assert isinstance(target_h, int) and target_h > 0, "target_h must be > 0"
    assert isinstance(strategy, str), "strategy must be a string"
    strategy_lc = strategy.lower()

    H, W = frame.shape[:2]
    if strategy_lc == "stretch":
        # Direct resize to target size (distorts aspect ratio)
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if strategy_lc == "fill":
        # Scale to cover the target, then center-crop
        scale = max(target_w / W, target_h / H)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        # Center crop to target
        x_off = max(0, (new_w - target_w) // 2)
        y_off = max(0, (new_h - target_h) // 2)
        crop = resized[y_off : y_off + target_h, x_off : x_off + target_w]
        # In rare rounding cases, pad if crop is slightly smaller
        if crop.shape[0] != target_h or crop.shape[1] != target_w:
            canvas = np.full((target_h, target_w, 3), pad_color, dtype=frame.dtype)
            ch, cw = crop.shape[:2]
            yo = (target_h - ch) // 2
            xo = (target_w - cw) // 2
            canvas[yo : yo + ch, xo : xo + cw] = crop
            return canvas
        return crop

    # "fit" (letterbox)
    scale = min(target_w / W, target_h / H)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    # create canvas and paste centered
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=frame.dtype)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas
