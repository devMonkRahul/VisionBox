# VisionBox - Animal Detector

VisionBox is a beginner-friendly Computer Vision project for first-year engineering students. It uses a YOLO11 model fine-tuned on an animals dataset to detect animals **and the pose they are in** from an image, through a simple Streamlit web application.

The project is designed for a 90-minute workshop: students can run the app, upload an image, see bounding boxes, read confidence scores, and understand how object detection works without training a model.

## Features

- Upload JPG, JPEG, or PNG images
- Capture a webcam snapshot using Streamlit
- Run detection with a YOLO11 model (`animals.pt`) fine-tuned on the animals dataset
- Detect the animal **and pair it with its pose** (e.g. `cow (stand)`)
- Display the original image and annotated image side by side
- Show bounding boxes, animal labels, pose, and optional confidence scores
- Count total detected animals
- Display a detection table with class name, position, confidence, and box coordinates
- Show the top 5 detections
- Show simple statistics:
  - Number of detections
  - Unique object classes
  - Highest confidence prediction
- Download the annotated result image

## Supported Classes

The model is trained on 10 classes, split into **animals** and **poses**:

| Animals | Poses |
| ------- | ----- |
| cat     | eating |
| cow     | laying |
| dog     | run    |
| horse   | sit    |
| sheep   | stand  |

The detector pairs each pose with the animal whose bounding box it overlaps most, and folds it into a `Position` column. When no pose is detected for an animal, `Position` is left empty.

## Dataset

The model is trained on the **Animals (animals-p1cov)** dataset from Roboflow Universe.

- Dataset page: <https://universe.roboflow.com/sesese/animals-p1cov/dataset/3>
- Download (YOLOv11 format): <https://universe.roboflow.com/sesese/animals-p1cov/dataset/3/download/yolov11>
- Workspace: `sesese` · Project: `animals-p1cov` · Version: `3` · License: CC BY 4.0

## Folder Structure

```text
visionbox/
|
├── app.py
├── animals.pt
├── requirements.txt
├── README.md
├── assets/
└── utils/
    ├── detector.py
    └── helpers.py
```

## Installation

### Option 1: Using uv

If you have `uv` installed, set up the project with:

```bash
uv sync
```

Run the app with:

```bash
uv run streamlit run app.py
```

### Option 2: Using pip

Create and activate a virtual environment if you want to keep dependencies separate.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run The App

If you used `uv`, run:

```bash
uv run streamlit run app.py
```

If you used `pip`, run:

```bash
streamlit run app.py
```

Make sure `animals.pt` is present in the project root so the app can load the fine-tuned model.

## How To Use

1. Open the Streamlit app in your browser.
2. Upload an image or capture a webcam snapshot.
3. Adjust the confidence threshold in the sidebar.
4. Click **Run object detection**.
5. View the annotated image, object count, detection table (with the paired pose), and statistics.
6. Download the annotated image if needed.

## Workshop Notes

This project is useful for explaining:

- What Computer Vision means
- What object detection does
- The difference between classification and detection
- Bounding boxes
- Class labels and how multiple classes (animal + pose) can describe one subject
- Confidence scores
- Why fine-tuned models are useful

## Future Improvements

- Add true live video detection using a WebRTC Streamlit component
- Add support for short video files
- Save detection history
- Add charts for class frequency
- Let students compare different YOLO11 model sizes
- Add a short quiz section for workshop participants
