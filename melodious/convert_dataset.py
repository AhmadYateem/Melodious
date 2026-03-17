"""Convert the current DeepScores subset into a YOLOv8-friendly dataset layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml

from .dataset import CLASS_NAMES, TARGET_CLASSES


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _category_lookup(coco_data: Dict) -> Dict[str, str]:
    return {category_id: info["name"] for category_id, info in coco_data["categories"].items()}


def _annotation_lines(coco_data: Dict, image_info: Dict, category_lookup: Dict[str, str]) -> List[str]:
    width = image_info["width"]
    height = image_info["height"]
    lines: List[str] = []

    for annotation_id in image_info.get("ann_ids", []):
        annotation = coco_data["annotations"].get(str(annotation_id))
        if not annotation:
            continue

        for category_id in annotation.get("cat_id", []):
            class_name = category_lookup.get(str(category_id))
            if class_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = annotation["a_bbox"]
            box_width = x2 - x1
            box_height = y2 - y1
            x_center = x1 + box_width / 2.0
            y_center = y1 + box_height / 2.0
            class_id = TARGET_CLASSES[class_name]
            lines.append(
                f"{class_id} {x_center / width:.6f} {y_center / height:.6f} {box_width / width:.6f} {box_height / height:.6f}"
            )
            break

    return lines


def convert_split(dataset_root: Path, output_root: Path, split: str) -> Path:
    """Convert one COCO split into YOLO label text files plus an image manifest."""
    split_json = dataset_root / f"deepscores_{split}.json"
    coco_data = _load_json(split_json)
    category_lookup = _category_lookup(coco_data)

    label_dir = output_root / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    image_manifest = output_root / f"{split}.txt"

    image_paths: List[str] = []
    for image_info in coco_data["images"]:
        image_path = dataset_root / "images" / image_info["filename"]
        if not image_path.exists():
            continue

        lines = _annotation_lines(coco_data, image_info, category_lookup)
        (label_dir / f"{Path(image_info['filename']).stem}.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
        image_paths.append(str(image_path.resolve()))

    image_manifest.write_text("\n".join(image_paths), encoding="utf-8")
    return image_manifest


def write_data_yaml(output_root: Path, train_manifest: Path, val_manifest: Path) -> Path:
    """Write the Ultralytics data configuration file."""
    data_yaml = output_root / "data.yaml"
    payload = {
        "path": str(output_root.resolve()),
        "train": str(train_manifest.resolve()),
        "val": str(val_manifest.resolve()),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DeepScores subset to YOLO format.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_ds2_dense"))
    parser.add_argument("--output-root", type=Path, default=Path("yolo_dataset"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    train_manifest = convert_split(args.dataset_root, args.output_root, "train")
    val_manifest = convert_split(args.dataset_root, args.output_root, "test")
    data_yaml = write_data_yaml(args.output_root, train_manifest, val_manifest)

    print(f"Wrote YOLO dataset to {args.output_root}")
    print(f"Data config: {data_yaml}")


if __name__ == "__main__":
    main()