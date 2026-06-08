"""Small helper functions for the VisionBox app."""

from io import BytesIO

import pandas as pd
from PIL import Image


def get_annotated_image_bytes(image: Image.Image) -> bytes:
    """Convert a PIL image into downloadable JPEG bytes."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def get_top_detections(detection_table: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Return the most confident detections."""
    if detection_table.empty:
        return detection_table

    return detection_table.sort_values(
        by="Confidence score",
        ascending=False,
    ).head(limit)


def get_class_counts(detection_table: pd.DataFrame) -> pd.DataFrame:
    """Count how many times each object class appears in one image."""
    if detection_table.empty:
        return pd.DataFrame(columns=["Class name", "Count"])

    return (
        detection_table["Class name"]
        .value_counts()
        .rename_axis("Class name")
        .reset_index(name="Count")
    )


def get_detection_stats(detection_table: pd.DataFrame) -> dict:
    """Create simple statistics for the results section."""
    if detection_table.empty:
        return {
            "unique_classes": 0,
            "highest_confidence": "N/A",
            "top_prediction": "N/A",
        }

    top_row = detection_table.sort_values(
        by="Confidence score",
        ascending=False,
    ).iloc[0]

    return {
        "unique_classes": detection_table["Class name"].nunique(),
        "highest_confidence": f"{top_row['Confidence score']:.3f}",
        "top_prediction": top_row["Class name"],
    }
