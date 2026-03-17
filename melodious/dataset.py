"""
Dataset loader for DeepScores v2 Dense dataset.

Handles loading images and annotations from the dataset_ds2_dense folder,
including oriented bounding boxes for music notation symbols.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2


# Target symbol classes for the prototype - common music notation symbols
# Maps DeepScores category names to class indices (0-14)
TARGET_CLASSES = {
    "noteheadBlack": 0,
    "noteheadBlackOnLine": 0,
    "noteheadBlackInSpace": 0,
    "noteheadHalf": 1,
    "noteheadHalfOnLine": 1,
    "noteheadHalfInSpace": 1,
    "noteheadWhole": 2,
    "noteheadWholeOnLine": 2,
    "noteheadWholeInSpace": 2,
    "clefG": 3,
    "clefF": 4,
    "clefC": 5,
    "clefCAlto": 5,
    "clefCTenor": 5,
    "rest8th": 6,
    "restQuarter": 7,
    "restHalf": 8,
    "restWhole": 9,
    "accidentalSharp": 10,
    "accidentalFlat": 11,
    "accidentalNatural": 12,
    "beam": 13,
    "stem": 14,
}

# Human-readable class names (15 classes total)
CLASS_NAMES = ["notehead-full", "notehead-half", "notehead-whole", "clefG", "clefF", 
               "clefC", "rest-8th", "rest-quarter", "rest-half", "rest-whole",
               "accidentalSharp", "accidentalFlat", "accidentalNatural", "beam", "stem"]

# Number of unique classes (use this for model configuration)
NUM_CLASSES = len(CLASS_NAMES)  # 15


class DeepScoresDataset(Dataset):
    """
    Dataset loader for DeepScores v2 format.
    
    Loads images and bounding box annotations from the dataset_ds2_dense directory.
    The dataset uses COCO-style format with:
    - images: list of image metadata
    - annotations: dict of annotation_id -> annotation
    - categories: dict of category_id -> category info
    
    Each annotation contains:
    - a_bbox: axis-aligned bounding box [x_min, y_min, x_max, y_max]
    - o_bbox: oriented bounding box [x0, y0, x1, y1, x2, y2, x3, y3]
    - cat_id: list of category IDs
    - img_id: image ID
    """
    
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train',
        img_size: int = 640,
        max_samples: Optional[int] = None,
        augment: bool = False
    ):
        """
        Args:
            root_dir: Path to dataset_ds2_dense directory
            split: 'train' or 'test'
            img_size: Target image size for resizing
            max_samples: Limit dataset size for faster prototyping
            augment: Apply simple augmentations during training
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'
        
        # Load annotations
        ann_file = self.root_dir / f"deepscores_{split}.json"
        
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                f"Expected files: deepscores_train.json or deepscores_test.json"
            )
        
        print(f"Loading DeepScores v2 annotations from {ann_file}...")
        
        # Load the full JSON file (it's manageable for the dense subset)
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        print(f"  Found {len(self.coco_data['images'])} images")
        print(f"  Found {len(self.coco_data['annotations'])} annotations")
        print(f"  Found {len(self.coco_data['categories'])} categories")
        
        # Process annotations
        self.annotations = self._build_image_annotations(max_samples)
        
        print(f"Loaded {len(self.annotations)} images with {self._count_boxes()} target symbol boxes")
        
    def _build_image_annotations(self, max_samples: Optional[int]) -> List[Dict]:
        """
        Build per-image annotation lists from COCO format.
        Filters to target symbol classes and converts to our format.
        """
        # Build category name lookup
        cat_id_to_name = {cat_id: cat_info['name'] 
                         for cat_id, cat_info in self.coco_data['categories'].items()}
        
        # Process images
        images = self.coco_data['images']
        if max_samples:
            images = images[:max_samples]
        
        annotations = []
        num_with_boxes = 0
        total_boxes = 0
        
        for img_info in images:
            img_id = str(img_info['id'])
            filename = img_info['filename']
            img_path = self.root_dir / "images" / filename
            
            if not img_path.exists():
                print(f"  Warning: Image not found: {img_path}")
                continue
            
            # Get annotations for this image
            ann_ids = img_info.get('ann_ids', [])
            
            boxes = []
            labels = []
            
            for ann_id in ann_ids:
                ann_id = str(ann_id)
                if ann_id not in self.coco_data['annotations']:
                    continue
                
                ann = self.coco_data['annotations'][ann_id]
                
                # Get category names
                cat_ids = ann.get('cat_id', [])
                if not cat_ids:
                    continue
                
                # Check if any category matches our target classes
                for cat_id in cat_ids:
                    cat_id = str(cat_id)
                    if cat_id not in cat_id_to_name:
                        continue
                    
                    cat_name = cat_id_to_name[cat_id]
                    
                    if cat_name in TARGET_CLASSES:
                        # Get bounding box (use axis-aligned for simplicity)
                        bbox = ann.get('a_bbox')
                        if bbox and len(bbox) == 4:
                            # bbox is [x_min, y_min, x_max, y_max]
                            boxes.append(bbox)
                            labels.append(TARGET_CLASSES[cat_name])
                        break  # Only add once per annotation
            
            if boxes:
                num_with_boxes += 1
                total_boxes += len(boxes)
            
            annotations.append({
                'image_path': str(img_path),
                'image_id': img_id,
                'filename': filename,
                'width': img_info['width'],
                'height': img_info['height'],
                'boxes': boxes,
                'labels': labels
            })
        
        print(f"  Images with target symbols: {num_with_boxes}/{len(annotations)}")
        print(f"  Total target symbol boxes: {total_boxes}")
        
        return annotations
    
    def _count_boxes(self) -> int:
        """Count total bounding boxes across all images."""
        return sum(len(ann['boxes']) for ann in self.annotations)
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            image: Tensor of shape (3, H, W)
            target: Dict with keys 'boxes', 'labels', 'image_id'
        """
        ann = self.annotations[idx]
        
        # Load image
        img_path = ann['image_path']
        image = Image.open(img_path).convert('RGB')
        orig_w = ann['width']
        orig_h = ann['height']
        
        # Resize image
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Convert to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Process boxes
        boxes = []
        if ann['boxes']:
            for box in ann['boxes']:
                # box is [x_min, y_min, x_max, y_max] in original image coordinates
                x_min, y_min, x_max, y_max = box
                
                # Scale to resized image coordinates
                x_min = x_min / orig_w * self.img_size
                y_min = y_min / orig_h * self.img_size
                x_max = x_max / orig_w * self.img_size
                y_max = y_max / orig_h * self.img_size
                
                boxes.append([x_min, y_min, x_max, y_max])
        
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(ann['labels'], dtype=torch.int64) if ann['labels'] else torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': ann['image_id'],
            'orig_size': (orig_w, orig_h),
            'img_size': (self.img_size, self.img_size)
        }
        
        return image, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Custom collate function for DataLoader.

    - Stacks images into a tensor of shape (B, 3, H, W)
    - Leaves targets as a list of dicts to support variable numbers of boxes
    """
    images: List[torch.Tensor] = []
    targets: List[Dict] = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images_tensor = torch.stack(images, 0)

    return images_tensor, targets


def create_dataloaders(
    root_dir: str,
    batch_size: int = 4,
    img_size: int = 640,
    num_workers: int = 2,
    max_train_samples: Optional[int] = 100,
    max_val_samples: Optional[int] = 20
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        root_dir: Path to dataset_ds2_dense
        batch_size: Batch size for training
        img_size: Image size for resizing
        num_workers: Number of dataloader workers
        max_train_samples: Limit training set size for faster prototyping
        max_val_samples: Limit validation set size
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = DeepScoresDataset(
        root_dir=root_dir,
        split='train',
        img_size=img_size,
        max_samples=max_train_samples,
        augment=True
    )
    
    # For prototype, use test set as validation
    val_dataset = DeepScoresDataset(
        root_dir=root_dir,
        split='test',
        img_size=img_size,
        max_samples=max_val_samples,
        augment=False
    )
    
    train_loader: DataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader: DataLoader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
