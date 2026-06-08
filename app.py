"""VisionBox Streamlit app.

This app helps beginners try object detection with a pretrained YOLO11 model.
Students can upload an image, capture a webcam snapshot, and inspect detections.
"""

from io import BytesIO

import streamlit as st
from PIL import Image

from utils.detector import (
    detect_objects,
    draw_detections,
    get_detection_dataframe,
)
from utils.helpers import (
    get_annotated_image_bytes,
    get_class_counts,
    get_detection_stats,
    get_top_detections,
)


st.set_page_config(
    page_title="VisionBox - Object Detector",
    layout="wide",
)


def read_image(uploaded_file) -> Image.Image:
    """Convert an uploaded file into a RGB PIL image."""
    return Image.open(uploaded_file).convert("RGB")


def render_detection_summary(total_objects: int, stats: dict, show_count: bool) -> None:
    """Display beginner-friendly detection statistics."""
    if show_count:
        st.metric("Objects detected", total_objects)

    stat_columns = st.columns(3)
    stat_columns[0].metric("Unique classes", stats["unique_classes"])
    stat_columns[1].metric("Highest confidence", stats["highest_confidence"])
    stat_columns[2].metric("Top prediction", stats["top_prediction"])


def run_detection_workflow(image: Image.Image, confidence_threshold: float, show_confidence: bool, show_count: bool) -> None:
    """Run YOLO11 detection and render all result sections."""
    with st.spinner("Running YOLO11 object detection..."):
        results = detect_objects(image, confidence_threshold)
        detection_table = get_detection_dataframe(results)
        annotated_image = draw_detections(image, results, show_confidence=show_confidence)

    total_objects = len(detection_table)
    class_counts = get_class_counts(detection_table)
    stats = get_detection_stats(detection_table)

    st.subheader("Detection Results")
    image_columns = st.columns(2)

    with image_columns[0]:
        st.image(image, caption="Original Image", width='stretch')

    with image_columns[1]:
        st.image(annotated_image, caption="Annotated Image", width='stretch')

        image_bytes = get_annotated_image_bytes(annotated_image)
        st.download_button(
            label="Download annotated image",
            data=image_bytes,
            file_name="visionbox_annotated.jpg",
            mime="image/jpeg",
        )

    st.subheader("Statistics")
    render_detection_summary(total_objects, stats, show_count)

    st.subheader("Detection Table")
    if detection_table.empty:
        st.info("No objects were detected. Try lowering the confidence threshold or using a clearer image.")
    else:
        st.dataframe(detection_table, width='stretch', hide_index=True)

    if not class_counts.empty:
        st.subheader("Objects Found In This Image")
        st.dataframe(class_counts, width='stretch', hide_index=True)

    top_detections = get_top_detections(detection_table, limit=5)
    if not top_detections.empty:
        st.subheader("Top Detected Objects")
        st.dataframe(top_detections, width='stretch', hide_index=True)


def main() -> None:
    """Build the Streamlit interface."""
    st.title("VisionBox – Real-Time Object Detector")
    st.write("Detect everyday objects using YOLO11 and Computer Vision")

    with st.sidebar:
        st.header("Controls")
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=1.00,
            value=0.25,
            step=0.05,
            help="Higher values keep only predictions the model is more confident about.",
        )
        show_confidence = st.toggle("Show confidence scores", value=True)
        show_count = st.toggle("Show object count", value=True)

        st.divider()
        st.caption("Model: YOLO11 nano pretrained on COCO classes")

    st.subheader("1. Upload Image")
    input_mode = st.tabs(["Upload image", "Webcam snapshot"])

    selected_image = None

    with input_mode[0]:
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            selected_image = read_image(uploaded_file)

    with input_mode[1]:
        camera_photo = st.camera_input("Open webcam and capture an image")
        if camera_photo is not None:
            selected_image = read_image(BytesIO(camera_photo.getvalue()))

    if selected_image is None:
        st.info("Upload an image or capture a webcam snapshot to begin detection.")
        return

    st.subheader("2. Run Detection")
    if st.button("Run object detection", type="primary"):
        run_detection_workflow(
            image=selected_image,
            confidence_threshold=confidence_threshold,
            show_confidence=show_confidence,
            show_count=show_count,
        )
    else:
        st.image(selected_image, caption="Ready for detection", width='stretch')


if __name__ == "__main__":
    main()
