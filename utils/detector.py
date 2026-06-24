"""YOLO11 object detection helper functions."""

from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO


# Classes that describe what an animal is doing rather than a separate object.
POSE_CLASSES = {"eating", "laying", "run", "sit", "stand"}

# Minimum overlap before a pose box is treated as belonging to an animal box.
POSE_IOU_THRESHOLD = 0.3


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    """Load the pretrained YOLO11 nano model only once."""
    return YOLO("animals.pt")


def _iou(box_a, box_b) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def pair_detections(results) -> list[dict]:
    """Group YOLO detections into one entry per animal with its pose.

    Animal boxes and pose boxes (``stand``, ``sit``, ...) are returned by the
    model as separate detections. This pairs each pose with the animal it best
    overlaps. The pose may be missing, so ``position`` is ``None`` in that case.
    Any pose that doesn't match an animal is kept as its own entry so nothing
    is silently dropped.

    Each entry holds ``class_name``, ``position``, ``confidence`` and the raw
    ``coords`` tuple ``(x1, y1, x2, y2)``.
    """
    animals = []
    poses = []

    for box in results.boxes:
        class_name = results.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        coords = tuple(box.xyxy[0].cpu().numpy().astype(int))
        detection = {"class_name": class_name, "confidence": confidence, "coords": coords}
        (poses if class_name in POSE_CLASSES else animals).append(detection)

    used_poses = set()
    entries = []

    for animal in animals:
        best_index = None
        best_iou = POSE_IOU_THRESHOLD
        for index, pose in enumerate(poses):
            if index in used_poses:
                continue
            overlap = _iou(animal["coords"], pose["coords"])
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index

        position = None
        if best_index is not None:
            used_poses.add(best_index)
            position = poses[best_index]["class_name"]

        entries.append({**animal, "position": position})

    # Poses that never matched an animal still get reported on their own.
    for index, pose in enumerate(poses):
        if index not in used_poses:
            entries.append({**pose, "position": None})

    return entries


def merge_detections(results) -> list[dict]:
    """Format paired detections as display rows (one dict per detected animal)."""
    rows = []
    for object_number, entry in enumerate(pair_detections(results), start=1):
        x1, y1, x2, y2 = entry["coords"]
        rows.append(
            {
                "Object": object_number,
                "Class name": entry["class_name"],
                "Position": entry["position"],
                "Confidence score": round(entry["confidence"], 3),
                "Box": f"({x1}, {y1}) to ({x2}, {y2})",
            }
        )
    return rows


def detect_objects(image: Image.Image, confidence: float):
    """Run object detection on a PIL image.

    The model returns a list of results. Because we process one image at a time,
    this function returns only the first result.
    """
    model = load_model()
    image_array = np.array(image.convert("RGB"))
    # YOLO returns every object above the confidence threshold in this image.
    # max_det keeps the app ready for crowded images with many objects.
    results = model(image_array, conf=confidence, max_det=300)
    return results[0]


def get_detection_dataframe(results) -> pd.DataFrame:
    """Convert YOLO detections into a readable pandas DataFrame.

    Each row is one animal with its pose folded into the ``Position`` column.
    ``Position`` is empty when no pose was detected for that animal.
    """
    rows = merge_detections(results)

    return pd.DataFrame(
        rows,
        columns=["Object", "Class name", "Position", "Confidence score", "Box"],
    )


def draw_detections(image: Image.Image, results, show_confidence: bool = True) -> Image.Image:
    """Draw bounding boxes, labels, and optional confidence scores on an image."""
    annotated_image = np.array(image.convert("RGB")).copy()

    for entry in pair_detections(results):
        class_name = entry["class_name"]
        confidence_score = entry["confidence"]
        x1, y1, x2, y2 = entry["coords"]

        label = class_name
        if entry["position"]:
            label = f"{class_name} ({entry['position']})"
        if show_confidence:
            label = f"{label} {confidence_score:.2f}"

        # Draw a green rectangle around the detected object.
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            color=(0, 180, 90),
            thickness=2,
        )

        # Draw a filled label background so text stays readable.
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_width, text_height = text_size
        label_y = max(y1, text_height + 10)

        cv2.rectangle(
            annotated_image,
            (x1, label_y - text_height - 10),
            (x1 + text_width + 8, label_y),
            color=(0, 180, 90),
            thickness=-1,
        )
        cv2.putText(
            annotated_image,
            label,
            (x1 + 4, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return Image.fromarray(annotated_image)
