"""Thin wrappers for training and exporting a YOLOv8 detector for Melodious."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def train_yolov8(
    data_yaml: Path,
    model_name: str = "yolov8m.pt",
    epochs: int = 50,
    img_size: int = 1024,
    batch: int = 8,
    project: str = "outputs/yolov8",
    device: Optional[str] = None,
) -> None:
    """Fine-tune a pretrained YOLOv8 detector on the converted Melodious dataset."""
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        project=project,
        device=device,
        exist_ok=True,
    )


def export_yolov8(checkpoint: Path, format_name: str = "onnx", int8: bool = False) -> None:
    """Export a trained YOLOv8 checkpoint for deployment."""
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    model.export(format=format_name, int8=int8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or export a YOLOv8 detector for Melodious.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--model", type=str, default="yolov8m.pt")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--img-size", type=int, default=1024)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--project", type=str, default="outputs/yolov8")
    train_parser.add_argument("--device", type=str, default=None)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--format", type=str, default="onnx")
    export_parser.add_argument("--int8", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_yolov8(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch=args.batch,
            project=args.project,
            device=args.device,
        )
        return

    export_yolov8(args.checkpoint, format_name=args.format, int8=args.int8)


if __name__ == "__main__":
    main()