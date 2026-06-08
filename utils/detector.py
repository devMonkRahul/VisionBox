"""YOLO11 object detection helper functions."""

from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    """Load the pretrained YOLO11 nano model only once."""
    return YOLO("yolo11n.pt")


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
    """Convert YOLO detections into a readable pandas DataFrame."""
    rows = []

    for object_number, box in enumerate(results.boxes, start=1):
        class_id = int(box.cls[0])
        class_name = results.names[class_id]
        confidence_score = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        rows.append(
            {
                "Object": object_number,
                "Class name": class_name,
                "Confidence score": round(confidence_score, 3),
                "Box": f"({x1}, {y1}) to ({x2}, {y2})",
            }
        )

    return pd.DataFrame(
        rows,
        columns=["Object", "Class name", "Confidence score", "Box"],
    )


def draw_detections(image: Image.Image, results, show_confidence: bool = True) -> Image.Image:
    """Draw bounding boxes, labels, and optional confidence scores on an image."""
    annotated_image = np.array(image.convert("RGB")).copy()

    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = results.names[class_id]
        confidence_score = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        label = class_name
        if show_confidence:
            label = f"{class_name} {confidence_score:.2f}"

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
