"""Utilities for exporting detector outputs in a stable JSON contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from .dataset import CLASS_NAMES


CLASS_ID_TO_NAME: Dict[int, str] = {index: name for index, name in enumerate(CLASS_NAMES)}


def xyxy_to_normalized_center(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> Dict[str, float]:
    """Convert pixel xyxy boxes to normalized center-format boxes."""
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0
    return {
        "x_center": x_center / image_width,
        "y_center": y_center / image_height,
        "width": width / image_width,
        "height": height / image_height,
    }


def build_detection_record(
    class_id: int,
    confidence: float,
    xyxy: Sequence[float],
    image_width: int,
    image_height: int,
) -> Dict[str, object]:
    """Build one detector record that the graph pipeline can consume directly."""
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return {
        "class_id": int(class_id),
        "class_name": CLASS_ID_TO_NAME[int(class_id)],
        "confidence": float(confidence),
        "bbox": xyxy_to_normalized_center(x1, y1, x2, y2, image_width, image_height),
        "bbox_pixels": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },
    }


def build_detection_payload(
    image_path: str,
    image_width: int,
    image_height: int,
    detections: Iterable[Dict[str, object]],
    model_type: str,
    checkpoint: Optional[str] = None,
) -> Dict[str, object]:
    """Wrap detector records in the agreed top-level payload shape."""
    return {
        "image_path": image_path,
        "image_size": {
            "width": int(image_width),
            "height": int(image_height),
        },
        "model": {
            "type": model_type,
            "checkpoint": checkpoint,
        },
        "detections": list(detections),
    }


def save_detection_payload(payload: Dict[str, object], output_path: Path) -> None:
    """Write one detector payload as pretty-printed JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")