"""Generate detector JSON files that match the graph integration contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image

from .detection_contract import build_detection_payload, build_detection_record, save_detection_payload
from .inference import load_model, preprocess_image


def _rescale_custom_boxes(boxes: torch.Tensor, original_width: int, original_height: int, img_size: int) -> torch.Tensor:
    """Map detector boxes from resized image coordinates back to the original page size."""
    if boxes.numel() == 0:
        return boxes

    scaled = boxes.clone()
    scaled[:, [0, 2]] *= original_width / img_size
    scaled[:, [1, 3]] *= original_height / img_size
    return scaled


def export_custom_detections(
    checkpoint_path: Path,
    image_paths: List[Path],
    output_dir: Path,
    device: str,
    conf_thresh: float,
    img_size: int,
) -> None:
    """Export detections from the custom YOLO model."""
    model = load_model(str(checkpoint_path), device=device)

    for image_path in image_paths:
        image_tensor, original_image = preprocess_image(str(image_path), img_size=img_size)
        image_tensor = image_tensor.to(device)
        image_height, image_width = original_image.shape[:2]

        with torch.no_grad():
            raw_predictions = model(image_tensor)
            detections = model.get_detections(raw_predictions, conf_thresh=conf_thresh, img_size=img_size)[0]

        boxes = _rescale_custom_boxes(detections["boxes"].cpu(), image_width, image_height, img_size)
        scores = detections["scores"].cpu().tolist()
        labels = detections["labels"].cpu().tolist()

        records = [
            build_detection_record(label, score, box.tolist(), image_width, image_height)
            for box, score, label in zip(boxes, scores, labels)
        ]
        payload = build_detection_payload(
            image_path=str(image_path),
            image_width=image_width,
            image_height=image_height,
            detections=records,
            model_type="custom-yolo",
            checkpoint=str(checkpoint_path),
        )
        save_detection_payload(payload, output_dir / f"{image_path.stem}.json")


def export_yolov8_detections(
    checkpoint_path: Path,
    image_paths: List[Path],
    output_dir: Path,
    conf_thresh: float,
) -> None:
    """Export detections from a YOLOv8 checkpoint."""
    from ultralytics import YOLO

    model = YOLO(str(checkpoint_path))

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        result = model.predict(source=str(image_path), conf=conf_thresh, verbose=False)[0]

        boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        scores = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        labels = result.boxes.cls.cpu().tolist() if result.boxes is not None else []

        records = [
            build_detection_record(int(label), score, box, image_width, image_height)
            for box, score, label in zip(boxes, scores, labels)
        ]
        payload = build_detection_payload(
            image_path=str(image_path),
            image_width=image_width,
            image_height=image_height,
            detections=records,
            model_type="yolov8",
            checkpoint=str(checkpoint_path),
        )
        save_detection_payload(payload, output_dir / f"{image_path.stem}.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export detector outputs as integration JSON.")
    parser.add_argument("--model-type", choices=["custom", "yolov8"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("sample_detections"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(args.image_dir.glob("*.png"))[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {args.image_dir}")

    if args.model_type == "custom":
        export_custom_detections(
            checkpoint_path=args.checkpoint,
            image_paths=image_paths,
            output_dir=args.output_dir,
            device=args.device,
            conf_thresh=args.conf_thresh,
            img_size=args.img_size,
        )
    else:
        export_yolov8_detections(
            checkpoint_path=args.checkpoint,
            image_paths=image_paths,
            output_dir=args.output_dir,
            conf_thresh=args.conf_thresh,
        )

    print(f"Saved {len(image_paths)} detection payloads to {args.output_dir}")


if __name__ == "__main__":
    main()