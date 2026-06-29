# Workshop Starter Folder for VisionBox (2-hour edition, with pose pairing)

## Context

The user is a workshop speaker who will build object detection **live** in front of
students. They need a clean **starter folder** to hand out so everyone begins from the
same place. The focus is the object-detection logic, not the UI; webcam capture is
dropped (image upload only).

**Update:** the workshop is now ~2 hours, so the starter is expanded to include
**pose pairing** — the model detects animals (cat/cow/dog/horse/sheep) and poses
(eating/laying/run/sit/stand) as *separate* boxes, and we pair each animal with the pose
it overlaps (via IoU) to produce combined labels like **"dog (sit)"**.

Confirmed choices:
- **Custom cv2 drawing** (built live) so combined "dog (sit)" labels appear on the image.
- **A results table** in the UI (Object, Class, Position, Confidence).
- **Progressive structure**: Part 1 = basic detection; Part 2 = pose pairing.

The starter files already exist at `d:\VisionBox\workshop-starter\` from the previous
session. This plan **revises** `app.py`, `detector.py`, `requirements.txt`, and
`README.md` to the 2-hour pose-pairing version. `animals.pt` is already copied and stays.

## Folder (unchanged layout)

```
workshop-starter/
├── app.py            # complete: upload + slider + run button + annotated image + table
├── detector.py       # STUBBED, two parts: PART 1 basic, PART 2 pose pairing
├── animals.pt        # already copied (5,481,754 bytes) — keep
├── requirements.txt  # add pandas
└── README.md         # 2-hour flow, Part 1 / Part 2 steps, IoU + pairing walkthrough
```

## File contents

### `app.py` (complete — given to students)

The table render is wrapped in `try/except NotImplementedError` so the app runs and
demos fully after **Part 1** (shows boxes + a "table coming in Part 2" note), then shows
the real table once `build_results_table` is implemented in Part 2.

```python
import streamlit as st
from PIL import Image

from detector import detect_objects, draw_boxes, build_results_table

st.set_page_config(page_title="VisionBox", page_icon="📦")
st.title("📦 VisionBox — Object Detection")
st.caption("Upload an image and detect animals (and what they're doing) with YOLO11.")

confidence = st.slider("Confidence threshold", 0.10, 1.00, 0.25, 0.05)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=480)

    if st.button("Run detection"):
        with st.spinner("Detecting..."):
            results = detect_objects(image, confidence)
            annotated = draw_boxes(image, results)
        st.image(annotated, caption="Detected objects", width=480)

        # PART 2: the results table. Until build_results_table() is written this
        # shows a friendly note instead of crashing.
        try:
            table = build_results_table(results)
            st.subheader("Detections")
            st.dataframe(table, use_container_width=True)
        except NotImplementedError:
            st.info("Results table comes in Part 2 — build build_results_table().")
```

### `detector.py` (STUBBED, progressive — built live)

```python
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
```

### `requirements.txt`

```
streamlit
ultralytics
opencv-python
pillow
numpy
pandas
```

### `README.md` (student instructions — 2-hour flow)

Update to include:
- Same setup section (venv, `pip install -r requirements.txt`, `streamlit run app.py`),
  with the "install before the session" note.
- **What we build, in two parts:**
  - **Part 1 — Basic detection:** `load_model` → `detect_objects` → `draw_boxes`
    (simple `results.plot()`). Checkpoint: run the app, see boxes; the table shows a
    "coming in Part 2" note.
  - **Part 2 — Pose pairing:** `_iou` → `pair_detections` → upgrade `draw_boxes` to
    custom cv2 drawing with "dog (sit)" labels → `build_results_table`.
- A short **concept box**: the model returns animals and poses as separate boxes; IoU
  measures box overlap; we attach each pose to the animal it overlaps most.
- The **IoU formula** and the **pairing algorithm** in plain words (so students can fill
  the stubs without the answer being handed to them verbatim).
- Classes: animals = cat, cow, dog, horse, sheep · poses = eating, laying, run, sit, stand.

## Speaker reference (completed `detector.py` bodies — NOT handed to students)

```python
from ultralytics import YOLO   # step 1

@lru_cache(maxsize=1)
def load_model():
    return YOLO("animals.pt")

def detect_objects(image, confidence):
    model = load_model()
    results = model(np.array(image), conf=confidence)
    return results[0]

def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0.0

def pair_detections(results):
    animals, poses = [], []
    for box in results.boxes:
        name = results.names[int(box.cls[0])]
        det = {
            "class_name": name,
            "confidence": float(box.conf[0]),
            "coords": tuple(box.xyxy[0].cpu().numpy().astype(int)),
        }
        (poses if name in POSE_CLASSES else animals).append(det)

    used, entries = set(), []
    for animal in animals:
        best_i, best_iou = None, POSE_IOU_THRESHOLD
        for i, pose in enumerate(poses):
            if i in used:
                continue
            overlap = _iou(animal["coords"], pose["coords"])
            if overlap >= best_iou:
                best_i, best_iou = i, overlap
        position = None
        if best_i is not None:
            used.add(best_i)
            position = poses[best_i]["class_name"]
        entries.append({**animal, "position": position})
    for i, pose in enumerate(poses):
        if i not in used:
            entries.append({**pose, "position": None})
    return entries

def draw_boxes(image, results):                       # PART 2 (upgraded) version
    annotated = np.array(image).copy()
    for e in pair_detections(results):
        x1, y1, x2, y2 = e["coords"]
        label = e["class_name"]
        if e["position"]:
            label += f" ({e['position']})"
        label += f" {e['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 90), 2)
        cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 90), 2, cv2.LINE_AA)
    return Image.fromarray(annotated)

def build_results_table(results):
    rows = [
        {"Object": i, "Class": e["class_name"],
         "Position": e["position"], "Confidence": round(e["confidence"], 3)}
        for i, e in enumerate(pair_detections(results), start=1)
    ]
    return pd.DataFrame(rows, columns=["Object", "Class", "Position", "Confidence"])
```

## Verification

1. `cd d:\VisionBox\workshop-starter`; `pip install -r requirements.txt`.
2. Stub check: `python -m py_compile app.py detector.py` (both compile).
3. Launch stubbed app: `streamlit run app.py` — upload an image; **Run detection** raises
   `NotImplementedError` (the part to build live).
4. Part 1 checkpoint: temporarily implement only `load_model`, `detect_objects`, and the
   simple `draw_boxes` (results.plot) → app shows boxes + the "table in Part 2" note.
5. Part 2 end-to-end: paste the full speaker-reference bodies → upload an animal photo →
   confirm combined labels ("dog (sit)") on the image and the 4-column table. Revert
   `detector.py` to the stub before handing out.
6. Confirm `animals.pt` is present (5,481,754 bytes).

## Notes / decisions
- Flat layout (no `utils/` package) — fewer moving parts for first-years.
- Part 1 uses `results.plot()`; Part 2 replaces it with custom cv2 drawing so combined
  animal+pose labels can be shown — this is the teaching payoff of pose pairing.
- `app.py` is given complete but degrades gracefully after Part 1 via try/except, so the
  speaker can run and demo at the Part-1 checkpoint without editing the UI.
- Original project's extra stats/metrics UI stays omitted to keep focus on detection.
- README explains IoU + the pairing algorithm in words (not code) so the stubs remain a
  genuine live-coding exercise.
