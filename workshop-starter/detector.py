"""
detector.py — the object detection logic we build together in the workshop.

We use a YOLO11 model (from the `ultralytics` library) fine-tuned to detect
5 animals (cat, cow, dog, horse, sheep) AND 5 poses (eating, laying, run, sit,
stand). The model reports animals and poses as SEPARATE boxes. In Part 2 we pair
each animal with the pose box it overlaps, so we can say things like "dog (sit)".

We build this in two parts:
  PART 1 — basic detection: load the model, run it, draw the boxes.
  PART 2 — pose pairing: match animals to poses, combined labels + a results table.
"""
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Pose classes describe what an animal is DOING, not a separate object.
POSE_CLASSES = {"eating", "laying", "run", "sit", "stand"}

# How much an animal box and a pose box must overlap to count as a pair.
POSE_IOU_THRESHOLD = 0.3

# TODO (step 1): import the YOLO class from ultralytics
# from ultralytics import YOLO


# ---------------------------------------------------------------------------
# PART 1 — BASIC DETECTION
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_model():
    """Load the YOLO model once and reuse it (lru_cache = load only the first time).
    TODO (step 2): return YOLO("animals.pt")
    """
    raise NotImplementedError("Load the YOLO model here")


def detect_objects(image: Image.Image, confidence: float):
    """Run detection on a PIL image and return the YOLO result object.
    TODO (step 3):
      1. model = load_model()
      2. array = np.array(image)
      3. results = model(array, conf=confidence)
      4. return results[0]
    """
    raise NotImplementedError("Run YOLO detection here")


def draw_boxes(image: Image.Image, results):
    """Return the image with boxes + labels drawn on it.

    TODO (step 4 — PART 1): start simple, let Ultralytics draw for us:
        annotated = results.plot()                    # numpy array, BGR
        return Image.fromarray(annotated[:, :, ::-1])  # BGR -> RGB

    TODO (step 8 — PART 2): come back and REPLACE the body so it draws combined
    labels like "dog (sit)". Loop over pair_detections(results); for each entry
    draw a cv2.rectangle on the box and a cv2.putText label built from
    entry["class_name"] + entry["position"]. (Full steps in the README.)
    """
    raise NotImplementedError("Draw the boxes here")


# ---------------------------------------------------------------------------
# PART 2 — POSE PAIRING
# ---------------------------------------------------------------------------

def _iou(box_a, box_b) -> float:
    """Intersection-over-Union of two (x1, y1, x2, y2) boxes — how much they overlap.
    TODO (step 5): find the overlapping rectangle, its area (0 if none), then
    return intersection / (area_a + area_b - intersection). (Formula in README.)
    """
    raise NotImplementedError("Compute IoU here")


def pair_detections(results) -> list[dict]:
    """Group detections into one entry per animal with its best-matching pose.

    Returns a list of dicts: class_name, position (pose name or None), confidence,
    coords (x1, y1, x2, y2).

    TODO (step 6):
      1. Split results.boxes into `animals` and `poses` using POSE_CLASSES.
      2. For each animal, pick the unused pose with the highest _iou that is
         >= POSE_IOU_THRESHOLD; that pose becomes the animal's "position".
      3. Keep leftover poses as their own entries (position = None).
    (Full walkthrough in the README.)
    """
    raise NotImplementedError("Pair animals with poses here")


def build_results_table(results) -> pd.DataFrame:
    """Build a table (one row per detection) for the UI.
    TODO (step 7): loop over pair_detections(results) and build rows with columns
    Object, Class, Position, Confidence. Return pd.DataFrame(rows).
    """
    raise NotImplementedError("Build the results table here")
