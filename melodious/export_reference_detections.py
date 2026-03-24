"""Export contract-valid reference detections from dataset annotations.

This is a meeting-safe fallback when no trained detector checkpoint exists yet.
Each JSON file uses the exact detector contract but is sourced from labeled
DeepScores pages so the graph side can integrate against real page data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .dataset import DeepScoresDataset
from .detection_contract import build_detection_payload, build_detection_record, save_detection_payload


def export_reference_payloads(
    dataset_root: Path,
    split: str,
    output_dir: Path,
    limit: int,
    start_index: int = 0,
) -> List[Path]:
    """Export a small set of real-page payloads from ground-truth annotations."""
    dataset = DeepScoresDataset(str(dataset_root), split=split, augment=False)
    saved_paths: List[Path] = []

    end_index = min(len(dataset.annotations), start_index + limit)
    for index in range(start_index, end_index):
        annotation = dataset.annotations[index]
        image_width = int(annotation["width"])
        image_height = int(annotation["height"])
        records = [
            build_detection_record(label, 1.0, box, image_width, image_height)
            for box, label in zip(annotation["boxes"], annotation["labels"])
        ]
        payload = build_detection_payload(
            image_path=str(Path(annotation["image_path"]).resolve()),
            image_width=image_width,
            image_height=image_height,
            detections=records,
            model_type="reference-ground-truth",
            checkpoint=None,
        )

        output_path = output_dir / f"{Path(annotation['filename']).stem}.json"
        save_detection_payload(payload, output_path)
        saved_paths.append(output_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reference detection payloads from DeepScores labels.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_ds2_dense"))
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("sample_detections/reference"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = export_reference_payloads(
        dataset_root=args.dataset_root,
        split=args.split,
        output_dir=args.output_dir,
        limit=args.limit,
        start_index=args.start_index,
    )
    print(f"Saved {len(saved_paths)} reference payloads to {args.output_dir}")


if __name__ == "__main__":
    main()