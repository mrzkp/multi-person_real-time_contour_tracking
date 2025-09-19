import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("weights/yolov8n-seg.pt")
tracker = sv.ByteTrack()
PERSON_CLASS_ID = 0

polygon_annotator = sv.PolygonAnnotator()
label_annotator = sv.LabelAnnotator()  # unique ids
trace_annotator = sv.TraceAnnotator()  # trace following detections


def callback(frame, idx):
    """
    Per each frame in the source_path, returns annotations over frames to the target_path.
    """

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Keep only persons (COCO class id 0)
    person_mask = detections.class_id == PERSON_CLASS_ID
    person_detections = detections[person_mask]

    # Track only person detections so traces follow people specifically
    tracked_detections = tracker.update_with_detections(person_detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id in zip(
            tracked_detections.class_id, tracked_detections.tracker_id
        )
    ]

    annotated_frame = polygon_annotator.annotate(
        frame.copy(), detections=person_detections
    )
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=tracked_detections, labels=labels
    )
    return trace_annotator.annotate(annotated_frame, detections=tracked_detections)


sv.process_video(
    source_path="videos/input/people-walking.mp4", target_path="videos/output/result.mp4", callback=callback
)
