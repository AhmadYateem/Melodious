"""
Inference and visualization utilities for music notation detection.

Provides functions to run detection on images and visualize results with
bounding boxes and class labels.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image

from .model import YOLODetector
from .dataset import CLASS_NAMES


# Colors for visualization (BGR format for OpenCV)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128)
]


def load_model(checkpoint_path: str, device: str = 'cuda') -> YOLODetector:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    from .model import create_yolo_model
    
    model = create_yolo_model(num_classes=len(CLASS_NAMES), device=device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model


def preprocess_image(image_path: str, img_size: int = 640) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess image for inference.
    
    Args:
        image_path: Path to image file
        img_size: Target size for resizing
    
    Returns:
        image_tensor: Preprocessed tensor (1, 3, H, W)
        original_image: Original image as numpy array
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # Resize
    image_resized = image.resize((img_size, img_size), Image.BILINEAR)
    
    # Convert to tensor
    image_tensor = np.array(image_resized).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, original_image


def draw_detections(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    img_size: int = 640,
    score_thresh: float = 0.3
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: Original image as numpy array
        boxes: Detected boxes (N, 4) in normalized coordinates
        labels: Class labels (N,)
        scores: Confidence scores (N,)
        img_size: Size used for detection
        score_thresh: Threshold for displaying detections
    
    Returns:
        Annotated image
    """
    # Copy image
    annotated = image.copy()
    h, w = annotated.shape[:2]
    
    # Convert boxes to image coordinates
    if len(boxes) > 0:
        boxes = boxes.clone()
        
        # Scale boxes to original image size
        scale_x = w / img_size
        scale_y = h / img_size
        
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
    
    # Draw each detection
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        
        # Get box coordinates
        x1, y1, x2, y2 = box.int().tolist()
        
        # Clip to image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Get class name and color
        label_idx = int(label.item())
        class_name = CLASS_NAMES[label_idx] if 0 <= label_idx < len(CLASS_NAMES) else f"class_{label_idx}"
        color = COLORS[label_idx % len(COLORS)]
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_text = f"{class_name}: {score.item():.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label_text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return annotated


def run_inference(
    model: YOLODetector,
    image_path: str,
    output_path: str,
    device: str = 'cuda',
    conf_thresh: float = 0.3,
    img_size: int = 640
) -> Dict:
    """
    Run inference on a single image and save annotated result.
    
    Args:
        model: Trained model
        image_path: Path to input image
        output_path: Path to save annotated image
        device: Device for inference
        conf_thresh: Confidence threshold
        img_size: Image size for detection
    
    Returns:
        Dict with detection results
    """
    # Preprocess
    image_tensor, original_image = preprocess_image(image_path, img_size)
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        predictions = model.predict(image_tensor, conf_thresh=conf_thresh)
    
    # Get detections
    pred = predictions[0]
    boxes = pred['boxes'].cpu()
    labels = pred['labels'].cpu()
    scores = pred['scores'].cpu()
    
    # Draw annotations
    annotated_image = draw_detections(
        original_image, boxes, labels, scores, img_size, conf_thresh
    )
    
    # Convert RGB to BGR for saving
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    # Save
    cv2.imwrite(output_path, annotated_image)
    
    print(f"Saved annotated image to {output_path}")
    print(f"Detected {len(boxes)} objects")
    
    # Return detection summary
    return {
        'num_detections': len(boxes),
        'boxes': boxes,
        'labels': labels,
        'scores': scores,
        'image_path': image_path,
        'output_path': output_path
    }


def batch_inference(
    model: YOLODetector,
    image_dir: str,
    output_dir: str,
    device: str = 'cuda',
    conf_thresh: float = 0.3,
    img_size: int = 640,
    max_images: int = 10
):
    """
    Run inference on multiple images.
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        output_dir: Directory to save results
        device: Device for inference
        conf_thresh: Confidence threshold
        img_size: Image size for detection
        max_images: Maximum number of images to process
    """
    from tqdm import tqdm
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get image files
    image_files = sorted(list(image_dir.glob('*.png')))[:max_images]
    
    print(f"Processing {len(image_files)} images...")
    
    all_results = []
    for img_path in tqdm(image_files):
        output_path = output_dir / f"annotated_{img_path.name}"
        
        try:
            result = run_inference(
                model, str(img_path), str(output_path),
                device, conf_thresh, img_size
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\nProcessed {len(all_results)} images successfully")
    print(f"Results saved to {output_dir}")
    
    return all_results
