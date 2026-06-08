# VisionBox - Real-Time Object Detector

VisionBox is a beginner-friendly Computer Vision project for first-year engineering students. It uses a pretrained YOLO11 model to detect everyday objects in images through a simple Streamlit web application.

The project is designed for a 90-minute workshop: students can run the app, upload an image, see bounding boxes, read confidence scores, and understand how object detection works without training a model.

## Features

- Upload JPG, JPEG, or PNG images
- Capture a webcam snapshot using Streamlit
- Run object detection with pretrained `yolo11n.pt`
- Display the original image and annotated image side by side
- Show bounding boxes, object labels, and optional confidence scores
- Count total detected objects
- Display a detection table with class names and confidence scores
- Show the top 5 detections
- Show simple statistics:
  - Number of detections
  - Unique object classes
  - Highest confidence prediction
- Download the annotated result image

## Supported Objects

VisionBox uses the default COCO classes supported by YOLO11. Common examples include:

- Person
- Car
- Dog
- Cat
- Bottle
- Cell phone
- Chair
- Laptop

## Folder Structure

```text
visionbox/
|
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   └── sample_image.jpg
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

The first run may take a little longer because Ultralytics downloads the pretrained `yolo11n.pt` model.

## How To Use

1. Open the Streamlit app in your browser.
2. Upload an image or capture a webcam snapshot.
3. Adjust the confidence threshold in the sidebar.
4. Click **Run object detection**.
5. View the annotated image, object count, detection table, and statistics.
6. Download the annotated image if needed.

## Screenshots

Add workshop screenshots here after running the app:

```text
assets/screenshot_home.png
assets/screenshot_results.png
```

## Workshop Notes

This project is useful for explaining:

- What Computer Vision means
- What object detection does
- The difference between classification and detection
- Bounding boxes
- Class labels
- Confidence scores
- Why pretrained models are useful

## Future Improvements

- Add true live video detection using a WebRTC Streamlit component
- Add support for short video files
- Save detection history
- Add charts for class frequency
- Let students compare different YOLO11 model sizes
- Add a short quiz section for workshop participants
