# Detector Output Contract

This folder defines the exact detector payload expected by the graph/GNN integration code.

## Canonical JSON Shape

```json
{
  "image_path": "dataset_ds2_dense/images/example.png",
  "image_size": {
    "width": 2480,
    "height": 1748
  },
  "model": {
    "type": "custom-yolo",
    "checkpoint": "outputs/custom_best.pth"
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "notehead-full",
      "confidence": 0.8712,
      "bbox": {
        "x_center": 0.3511,
        "y_center": 0.5210,
        "width": 0.0184,
        "height": 0.0281
      },
      "bbox_pixels": {
        "x1": 847.0,
        "y1": 886.0,
        "x2": 892.0,
        "y2": 935.0
      }
    }
  ]
}
```

## Contract Rules

- Bounding boxes for integration are center-based and normalized to `[0, 1]`.
- Pixel boxes are included as `bbox_pixels` for debugging and alignment checks.
- The confidence field name is always `confidence`.
- Class IDs must match `class_mapping.json` in this folder.
- One JSON file should correspond to one page image.

## Current Status

- Schema is frozen.
- Real detector outputs still need to be generated from a trained checkpoint.