"""
Training script for YOLO detector on DeepScores dataset.

Implements loss functions, metrics, and training loop for music notation detection.
Model is trained from scratch without pretrained weights.
""" 

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .model import YOLODetector
from .dataset import create_dataloaders, NUM_CLASSES


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class YOLOLoss(nn.Module):
    """
    Improved YOLO loss function with:
    - Focal loss for class imbalance
    - CIoU loss for bounding box regression
    - Better target matching
    """
    
    def __init__(self, num_classes: int, use_focal: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.use_focal = use_focal
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0) if use_focal else None
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        # Loss weights
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        self.lambda_obj = 1.0
        self.lambda_class = 1.0
    
    def compute_ciou(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute CIoU loss for better bbox regression."""
        # pred_boxes: [N, 4] in cxcywh format
        # target_boxes: [N, 4] in cxcywh format
        
        # Convert to xyxy
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        
        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        
        # Center distance
        pred_cx = pred_boxes[:, 0]
        pred_cy = pred_boxes[:, 1]
        target_cx = target_boxes[:, 0]
        target_cy = target_boxes[:, 1]
        
        center_dist_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Diagonal of smallest enclosing box
        c_x1 = torch.min(pred_x1, target_x1)
        c_y1 = torch.min(pred_y1, target_y1)
        c_x2 = torch.max(pred_x2, target_x2)
        c_y2 = torch.max(pred_y2, target_y2)
        
        c_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2
        
        # Aspect ratio penalty
        pred_w = pred_boxes[:, 2]
        pred_h = pred_boxes[:, 3]
        target_w = target_boxes[:, 2]
        target_h = target_boxes[:, 3]
        
        ar_pred = pred_w / (pred_h + 1e-6)
        ar_target = target_w / (target_h + 1e-6)
        
        v = (4 / (3.14159 ** 2)) * (torch.atan(ar_target) - torch.atan(ar_pred)) ** 2
        alpha = v / (1 - iou + v + 1e-6)
        
        ciou = iou - center_dist_sq / (c_sq + 1e-6) - alpha * v
        
        return 1 - ciou  # Return loss (1 - CIoU)
    
    def forward(self, predictions: List[torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute loss with proper grid-based target matching.
        
        Key insight: YOLO predictions are anchored to grid cells.
        Each grid cell is responsible for predicting objects whose center falls in that cell.
        """
        device = predictions[0].device
        batch_size = predictions[0].size(0)
        
        # Initialize losses
        loss_coord = torch.tensor(0.0, device=device)
        loss_conf = torch.tensor(0.0, device=device)
        loss_class = torch.tensor(0.0, device=device)
        
        num_pos = 0
        
        for scale_idx, pred in enumerate(predictions):
            # pred shape: (B, num_anchors, H, W, 5 + num_classes)
            num_anchors = pred.size(1)
            grid_h = pred.size(2)
            grid_w = pred.size(3)
            
            # Create grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(grid_h, device=device, dtype=torch.float32),
                torch.arange(grid_w, device=device, dtype=torch.float32),
                indexing='ij'
            )
            
            for b in range(batch_size):
                target = targets[b]
                target_boxes = target['boxes'].to(device)
                target_labels = target['labels'].to(device)
                
                # Get predictions for this batch
                pred_b = pred[b]  # (num_anchors, H, W, 5 + num_classes)
                
                # Create target tensor for objectness
                obj_target = torch.zeros(num_anchors, grid_h, grid_w, device=device)
                coord_target = torch.zeros(num_anchors, grid_h, grid_w, 4, device=device)
                class_target = torch.zeros(num_anchors, grid_h, grid_w, self.num_classes, device=device)
                
                if len(target_boxes) == 0:
                    # No objects - all predictions are negative
                    obj_scores = pred_b[..., 4]
                    loss_conf += self.bce_loss(obj_scores, obj_target).sum()
                    continue
                
                # Normalize target boxes to grid coordinates
                img_h, img_w = target.get('img_size', (640, 640))
                if isinstance(img_h, int):
                    img_h = float(img_h)
                    img_w = float(img_w)
                
                for t_idx in range(len(target_boxes)):
                    t_box = target_boxes[t_idx]  # [x1, y1, x2, y2] in pixel coords
                    
                    # Convert to center format and normalize to grid
                    t_cx = (t_box[0] + t_box[2]) / 2.0
                    t_cy = (t_box[1] + t_box[3]) / 2.0
                    t_w = t_box[2] - t_box[0]
                    t_h = t_box[3] - t_box[1]
                    
                    # Convert to grid coordinates
                    gx = t_cx / img_w * grid_w  # Grid x position (float)
                    gy = t_cy / img_h * grid_h  # Grid y position (float)
                    
                    # Grid cell indices
                    gi = int(gx)
                    gj = int(gy)
                    
                    # Clamp to valid range
                    gi = max(0, min(grid_w - 1, gi))
                    gj = max(0, min(grid_h - 1, gj))
                    
                    # Assign to first anchor (simplified - could use IoU-based anchor selection)
                    for anchor_idx in range(num_anchors):
                        if obj_target[anchor_idx, gj, gi] == 0:
                            obj_target[anchor_idx, gj, gi] = 1.0
                            num_pos += 1
                            
                            # Target coordinates in YOLO format (offsets from grid cell)
                            # tx = gx - gi (offset within cell)
                            # ty = gy - gj (offset within cell)
                            # tw = log(w / anchor_w) - simplified: log(w / img_w)
                            # th = log(h / anchor_h) - simplified: log(h / img_h)
                            
                            tw_norm = t_w / img_w
                            th_norm = t_h / img_h
                            
                            coord_target[anchor_idx, gj, gi] = torch.tensor([
                                gx - gi,  # tx
                                gy - gj,  # ty
                                torch.log(torch.tensor(tw_norm + 1e-6)),  # tw
                                torch.log(torch.tensor(th_norm + 1e-6))   # th
                            ], device=device)
                            
                            # Class target (one-hot)
                            class_target[anchor_idx, gj, gi, target_labels[t_idx]] = 1.0
                            break
                
                # Compute losses for this batch and scale
                # Objectness loss
                obj_scores = pred_b[..., 4]
                loss_conf += self.bce_loss(obj_scores, obj_target).sum()
                
                # Coordinate loss (only for positive cells)
                pos_mask = obj_target > 0
                if pos_mask.sum() > 0:
                    # Get predictions for positive cells
                    pred_coords = pred_b[..., :4]  # (num_anchors, H, W, 4)
                    
                    # Match training to the decoder, which applies sigmoid to cell offsets.
                    pred_tx = pred_coords[..., 0][pos_mask]
                    pred_ty = pred_coords[..., 1][pos_mask]
                    target_tx = coord_target[..., 0][pos_mask]
                    target_ty = coord_target[..., 1][pos_mask]
                    
                    loss_coord += self.mse_loss(torch.sigmoid(pred_tx), target_tx).sum()
                    loss_coord += self.mse_loss(torch.sigmoid(pred_ty), target_ty).sum()
                    
                    # Compute loss for w, h (log scale)
                    pred_tw = pred_coords[..., 2][pos_mask]
                    pred_th = pred_coords[..., 3][pos_mask]
                    target_tw = coord_target[..., 2][pos_mask]
                    target_th = coord_target[..., 3][pos_mask]
                    
                    loss_coord += self.mse_loss(pred_tw, target_tw).sum()
                    loss_coord += self.mse_loss(pred_th, target_th).sum()
                    
                    # Classification loss
                    pred_cls = pred_b[..., 5:][pos_mask]  # (N, num_classes)
                    target_cls = class_target[pos_mask]   # (N, num_classes)
                    
                    if self.use_focal and len(target_cls) > 0:
                        # For focal loss, we need class indices
                        target_cls_idx = target_cls.argmax(dim=1)
                        loss_class += self.focal_loss(pred_cls, target_cls_idx)
                    elif len(target_cls) > 0:
                        target_cls_idx = target_cls.argmax(dim=1)
                        loss_class += self.ce_loss(pred_cls, target_cls_idx).sum()
        
        # Normalize losses
        if num_pos > 0:
            loss_coord = loss_coord / num_pos
            loss_class = loss_class / num_pos
        
        loss_conf = loss_conf / (batch_size * len(predictions))
        
        # Weighted total loss
        total_loss = (
            self.lambda_coord * loss_coord +
            self.lambda_obj * loss_conf +
            self.lambda_class * loss_class
        )
        
        return {
            'total': total_loss,
            'coord': loss_coord,
            'conf': loss_conf,
            'class': loss_class
        }


class Metrics:
    """
    Comprehensive metrics for detection analysis.
    Tracks multiple levels of performance to show learning progress.
    """
    
    @staticmethod
    def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Compute IoU between two boxes in xyxy format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    @staticmethod
    def compute_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict:
        """
        Compute comprehensive metrics at multiple levels.
        
        Returns metrics showing different aspects of learning:
        1. Classification accuracy (any detection in image, ignore localization)
        2. Detection at IoU 0.25 (lenient - shows if model detects roughly right area)
        3. Detection at IoU 0.5 (standard COCO metric)
        4. Detection at IoU 0.75 (strict - very precise localization)
        5. Per-class performance
        """
        from melodious.dataset import NUM_CLASSES
        
        # Multi-level metrics
        metrics = {
            'classification_correct': 0,  # Just class right, any overlap
            'classification_total': 0,
            'detection_025': {'tp': 0, 'fp': 0, 'fn': 0},  # Lenient
            'detection_050': {'tp': 0, 'fp': 0, 'fn': 0},  # Standard
            'detection_075': {'tp': 0, 'fp': 0, 'fn': 0},  # Strict
            'per_class': {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(NUM_CLASSES)},
            'avg_iou': [],
            'total_predictions': 0,
            'total_targets': 0
        }
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes'].cpu()
            pred_labels = pred['labels'].cpu()
            target_boxes = target['boxes'].cpu()
            target_labels = target['labels'].cpu()
            
            metrics['total_predictions'] += len(pred_boxes)
            metrics['total_targets'] += len(target_boxes)
            
            # Track matched targets for each IoU threshold
            matched_025 = set()
            matched_050 = set()
            matched_075 = set()
            
            # For each prediction
            for i, pred_box in enumerate(pred_boxes):
                pred_label = pred_labels[i].item()
                best_iou = 0
                best_idx = -1
                best_target_label = -1
                
                # Find best matching target
                for j, target_box in enumerate(target_boxes):
                    target_label = target_labels[j].item()
                    iou = Metrics.compute_iou(pred_box, target_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                        best_target_label = target_label
                
                # Record IoU for analysis
                if best_iou > 0:
                    metrics['avg_iou'].append(best_iou)
                
                # Classification accuracy (just need ANY overlap and correct class)
                if best_iou > 0.1 and pred_label == best_target_label:
                    metrics['classification_correct'] += 1
                metrics['classification_total'] += 1
                
                # Detection at different IoU thresholds
                class_match = (pred_label == best_target_label)
                
                # IoU 0.25 (lenient)
                if best_iou >= 0.25 and class_match and best_idx not in matched_025:
                    metrics['detection_025']['tp'] += 1
                    metrics['per_class'][pred_label]['tp'] += 1
                    matched_025.add(best_idx)
                elif best_iou >= 0.25:
                    metrics['detection_025']['fp'] += 1
                    metrics['per_class'][pred_label]['fp'] += 1
                else:
                    metrics['detection_025']['fp'] += 1
                    metrics['per_class'][pred_label]['fp'] += 1
                
                # IoU 0.5 (standard)
                if best_iou >= 0.5 and class_match and best_idx not in matched_050:
                    metrics['detection_050']['tp'] += 1
                    matched_050.add(best_idx)
                else:
                    metrics['detection_050']['fp'] += 1
                
                # IoU 0.75 (strict)
                if best_iou >= 0.75 and class_match and best_idx not in matched_075:
                    metrics['detection_075']['tp'] += 1
                    matched_075.add(best_idx)
                else:
                    metrics['detection_075']['fp'] += 1
            
            # Count false negatives (unmatched targets)
            metrics['detection_025']['fn'] += len(target_boxes) - len(matched_025)
            metrics['detection_050']['fn'] += len(target_boxes) - len(matched_050)
            metrics['detection_075']['fn'] += len(target_boxes) - len(matched_075)
            
            # Per-class FN (at IoU 0.25)
            for j, target_label in enumerate(target_labels):
                if j not in matched_025:
                    metrics['per_class'][target_label.item()]['fn'] += 1
        
        # Calculate summary metrics for different IoU thresholds
        def calc_metrics(tp, fp, fn):
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            return precision, recall, f1
        
        # Classification accuracy
        class_acc = metrics['classification_correct'] / max(metrics['classification_total'], 1)
        
        # Detection metrics at different IoU thresholds
        p_025, r_025, f1_025 = calc_metrics(**metrics['detection_025'])
        p_050, r_050, f1_050 = calc_metrics(**metrics['detection_050'])
        p_075, r_075, f1_075 = calc_metrics(**metrics['detection_075'])
        
        # Average IoU
        avg_iou = np.mean(metrics['avg_iou']) if metrics['avg_iou'] else 0.0
        
        return {
            # Classification (shows model learns classes)
            'class_accuracy': class_acc,
            
            # Lenient detection (IoU >= 0.25) - shows rough localization
            'precision_025': p_025,
            'recall_025': r_025,
            'f1_025': f1_025,
            
            # Standard detection (IoU >= 0.5) - COCO metric
            'precision': p_050,
            'recall': r_050,
            'f1': f1_050,
            
            # Strict detection (IoU >= 0.75) - precise localization
            'precision_075': p_075,
            'recall_075': r_075,
            'f1_075': f1_075,
            
            # Additional metrics
            'avg_iou': avg_iou,
            'total_predictions': metrics['total_predictions'],
            'total_targets': metrics['total_targets'],
            
            # Raw counts (standard metric)
            'tp': metrics['detection_050']['tp'],
            'fp': metrics['detection_050']['fp'],
            'fn': metrics['detection_050']['fn'],
            
            # Per-class performance
            'per_class': metrics['per_class']
        }


class Trainer:
    """
    Trainer for YOLO detector on music notation dataset.
    """
    
    def __init__(
        self,
        model: YOLODetector,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        lr: float = 1e-3,
        output_dir: str = 'outputs',
        log_dir: str = 'logs',
        num_epochs: int = 10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Loss and optimizer
        self.criterion = YOLOLoss(num_classes=NUM_CLASSES)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Cosine annealing scheduler for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )
        
        # Gradient clipping value
        self.grad_clip = 1.0
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        running_avg_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [TRAIN]', 
                   ncols=120, colour='green')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            losses = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Update metrics
            batch_loss = losses['total'].item()
            epoch_loss += batch_loss
            running_avg_loss = running_avg_loss * 0.9 + batch_loss * 0.1
            
            # Update progress bar with detailed stats
            pbar.set_postfix(
                loss=f"{batch_loss:.4f}",
                avg=f"{running_avg_loss:.4f}",
                coord=f"{losses['coord'].item():.3f}",
                conf=f"{losses['conf'].item():.3f}",
                cls=f"{losses['class'].item():.3f}"
            )
            
            # Log to tensorboard
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/total_loss', losses['total'].item(), step)
            self.writer.add_scalar('train/coord_loss', losses['coord'].item(), step)
            self.writer.add_scalar('train/conf_loss', losses['conf'].item(), step)
            self.writer.add_scalar('train/class_loss', losses['class'].item(), step)
        
        return epoch_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict]:
        """Validate model with ACTUAL detection metrics."""
        self.model.eval()
        val_loss = 0.0
        val_coord_loss = 0.0
        val_conf_loss = 0.0
        val_class_loss = 0.0
        
        # Collect predictions and targets for ACTUAL metrics
        all_predictions = []
        all_targets = []
        all_confidences = []  # Track confidence scores
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [VAL]  ', 
                   ncols=120, colour='blue')
        
        for images, targets in pbar:
            images = images.to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            losses = self.criterion(predictions, targets)
            
            batch_loss = losses['total'].item()
            val_loss += batch_loss
            val_coord_loss += losses['coord'].item()
            val_conf_loss += losses['conf'].item()
            val_class_loss += losses['class'].item()
            
            # Get actual detections for metrics (pass img_size from dataset)
            img_size = images.shape[-1]  # Assume square images
            detections = self.model.get_detections(predictions, conf_thresh=0.3, img_size=img_size)
            all_predictions.extend(detections)
            all_targets.extend(targets)
            
            # Collect confidence scores
            for det in detections:
                if len(det['scores']) > 0:
                    all_confidences.extend(det['scores'].cpu().numpy().tolist())
            
            pbar.set_postfix(
                loss=f"{batch_loss:.4f}",
                coord=f"{losses['coord'].item():.3f}",
                conf=f"{losses['conf'].item():.3f}",
                cls=f"{losses['class'].item():.3f}"
            )
        
        n_batches = len(self.val_loader)
        avg_loss = val_loss / n_batches
        avg_coord_loss = val_coord_loss / n_batches
        avg_conf_loss = val_conf_loss / n_batches
        avg_class_loss = val_class_loss / n_batches
        
        # Compute ACTUAL detection metrics (not proxy!)
        metrics = Metrics.compute_metrics(all_predictions, all_targets)
        
        # Add loss components
        metrics['coord_loss'] = avg_coord_loss
        metrics['conf_loss'] = avg_conf_loss
        metrics['class_loss'] = avg_class_loss
        
        # Add confidence metrics (required by instructions.md)
        import numpy as np
        if all_confidences:
            metrics['mean_confidence'] = float(np.mean(all_confidences))
            metrics['frac_conf_07'] = float(sum(1 for c in all_confidences if c >= 0.7) / len(all_confidences))
            metrics['total_detections'] = len(all_confidences)
        else:
            metrics['mean_confidence'] = 0.0
            metrics['frac_conf_07'] = 0.0
            metrics['total_detections'] = 0
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/coord_loss', avg_coord_loss, epoch)
        self.writer.add_scalar('val/conf_loss', avg_conf_loss, epoch)
        self.writer.add_scalar('val/class_loss', avg_class_loss, epoch)
        self.writer.add_scalar('val/precision', metrics['precision'], epoch)
        self.writer.add_scalar('val/recall', metrics['recall'], epoch)
        self.writer.add_scalar('val/f1', metrics['f1'], epoch)
        self.writer.add_scalar('val/mean_confidence', metrics['mean_confidence'], epoch)
        self.writer.add_scalar('val/frac_conf_07', metrics['frac_conf_07'], epoch)
        
        return avg_loss, metrics
    
    def train(self, num_epochs: int = 10):
        """
        Train model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\n{'='*60}")
        print(f"Starting training: {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, metrics = self.validate(epoch)
            
            # Update scheduler (CosineAnnealingLR steps per epoch, not per metric)
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_precision'].append(metrics['precision'])
            self.history['val_recall'].append(metrics['recall'])
            self.history['val_f1'].append(metrics['f1'])
            
            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}")
            print(f"  TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
            # Print confidence metrics
            if metrics.get('mean_confidence', 0) > 0:
                print(f"  Mean Confidence: {metrics['mean_confidence']:.3f}")
                print(f"  Detections >= 0.7 conf: {metrics['frac_conf_07']*100:.1f}%")
            print(f"  Total Detections: {metrics.get('total_detections', 0)}")
            
            # Save checkpoint after each epoch
            epoch_checkpoint = f'yolo_epoch_{epoch+1}.pth'
            self.save_checkpoint(epoch_checkpoint, epoch, val_loss, metrics)
            print(f"  ✓ Saved checkpoint: {epoch_checkpoint}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('yolo_scratch_best.pth', epoch, val_loss, metrics)
                print(f"  ✓ NEW BEST MODEL! (val_loss: {val_loss:.4f})")
            
            print(f"{'='*60}\n")
        
        # Save final model
        self.save_checkpoint('yolo_scratch_final.pth', num_epochs-1, val_loss, metrics)
        
        # Training complete
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"🎵 Training Complete! 🎵")
        print(f"{'='*60}")
        print(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Final metrics:")
        print(f"  Precision: {self.history['val_precision'][-1]:.3f}")
        print(f"  Recall: {self.history['val_recall'][-1]:.3f}")
        print(f"  F1: {self.history['val_f1'][-1]:.3f}")
        print(f"\nSaved models:")
        print(f"  - Best: outputs/yolo_scratch_best.pth")
        print(f"  - Final: outputs/yolo_scratch_final.pth")
        print(f"  - Per-epoch: outputs/yolo_epoch_*.pth")
        print(f"{'='*60}\n")
        
        # Save training curves
        self.plot_training_curves()
        
        # Save training history as JSON for notebook
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✓ Saved training history: {history_path}")
        
        self.writer.close()
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'history': self.history
        }
        
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        
        # Also save individual epoch loss components for detailed analysis
        if 'final' in filename or 'best' in filename:
            loss_history = {
                'train_loss': self.history['train_loss'],
                'val_loss': self.history['val_loss'],
                'train_coord_loss': self.history.get('train_coord_loss', []),
                'val_coord_loss': self.history.get('val_coord_loss', []),
                'train_conf_loss': self.history.get('train_conf_loss', []),
                'val_conf_loss': self.history.get('val_conf_loss', []),
                'train_class_loss': self.history.get('train_class_loss', []),
                'val_class_loss': self.history.get('val_class_loss', [])
            }
            history_path = self.output_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(loss_history, f, indent=2)
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision curve
        axes[0, 1].plot(self.history['val_precision'], label='Precision', 
                       color='green', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Validation Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall curve
        axes[1, 0].plot(self.history['val_recall'], label='Recall', 
                       color='orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Validation Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 curve
        axes[1, 1].plot(self.history['val_f1'], label='F1 Score', 
                       color='red', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Validation F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {self.output_dir / 'training_curves.png'}")
        plt.close()
