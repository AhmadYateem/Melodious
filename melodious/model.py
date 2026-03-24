"""
YOLO detector built from scratch in PyTorch.

This implementation creates a simplified YOLO architecture without using
pretrained weights. The model is trained from random initialization on
the DeepScores dataset subset.

Architecture: Backbone (ResNet-inspired) + Detection Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and LeakyReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for feature extraction."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(channels // 2, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class YOLOBackbone(nn.Module):
    """
    Custom backbone for YOLO detector.
    Inspired by Darknet but simplified for music notation.
    """
    
    def __init__(self):
        super().__init__()
        
        # Stem: 3 -> 32
        self.stem = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        
        # Stage 1: 32 -> 64
        self.stage1 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64)
        )
        
        # Stage 2: 64 -> 128
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        # Stage 3: 128 -> 256
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        
        # Stage 4: 256 -> 512
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )
    
    def forward(self, x):
        """
        Returns:
            Multi-scale features for detection at different resolutions
        """
        x = self.stem(x)
        
        s1 = self.stage1(x)     # 1/2 scale
        s2 = self.stage2(s1)    # 1/4 scale
        s3 = self.stage3(s2)    # 1/8 scale
        s4 = self.stage4(s3)    # 1/16 scale
        
        return s2, s3, s4  # Return multi-scale features


class DetectionHead(nn.Module):
    """
    YOLO detection head that predicts bounding boxes and class probabilities.
    """
    
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Each anchor predicts: x, y, w, h, objectness, class_probs
        self.num_outputs = num_anchors * (5 + num_classes)
        
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1),
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, self.num_outputs, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature map of shape (B, C, H, W)
        
        Returns:
            predictions: (B, num_anchors, H, W, 5 + num_classes)
        """
        batch_size = x.size(0)
        grid_h, grid_w = x.size(2), x.size(3)
        
        # Predict
        predictions = self.conv(x)
        
        # Reshape to (B, num_anchors, H, W, 5 + num_classes)
        predictions = predictions.view(
            batch_size, 
            self.num_anchors, 
            5 + self.num_classes,
            grid_h, 
            grid_w
        ).permute(0, 1, 3, 4, 2).contiguous()
        
        return predictions


class YOLODetector(nn.Module):
    """
    Complete YOLO detector for music notation symbols.
    Trained from scratch without pretrained weights.
    """
    
    def __init__(self, num_classes: int = 15):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = YOLOBackbone()
        
        # Detection heads at multiple scales
        self.head_small = DetectionHead(512, num_classes, num_anchors=3)  # 1/16 scale
        self.head_medium = DetectionHead(256, num_classes, num_anchors=3)  # 1/8 scale
        self.head_large = DetectionHead(128, num_classes, num_anchors=3)   # 1/4 scale
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            List of predictions at different scales
        """
        # Extract multi-scale features
        feat_large, feat_medium, feat_small = self.backbone(x)
        
        # Predict at each scale
        pred_small = self.head_small(feat_small)    # Small objects, large grid
        pred_medium = self.head_medium(feat_medium)  # Medium objects
        pred_large = self.head_large(feat_large)     # Large objects, fine grid
        
        return pred_large, pred_medium, pred_small
    
    def get_detections(self, preds, conf_thresh: float = 0.3, img_size: int = 640):
        """
        Get detections from raw model predictions (for training/validation).
        
        Args:
            preds: Raw predictions from forward pass
            conf_thresh: Confidence threshold
            img_size: Target image size for scaling coordinates
        
        Returns:
            List of detections per image with boxes in pixel coordinates [x1,y1,x2,y2]
        """
        batch_size = preds[0].size(0)
        all_detections = []
        
        for b in range(batch_size):
            boxes = []
            scores = []
            labels = []
            
            for pred in preds:
                # Extract predictions for this image
                # pred shape: (num_anchors, H, W, 5 + num_classes)
                p = pred[b]
                num_anchors, grid_h, grid_w, _ = p.shape
                
                # Create grid offsets for this scale
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(grid_h, device=p.device, dtype=p.dtype),
                    torch.arange(grid_w, device=p.device, dtype=p.dtype),
                    indexing='ij'
                )
                # Grid shape: (H, W) - need to broadcast with (num_anchors, H, W)
                grid_x = grid_x.unsqueeze(0)  # (1, H, W)
                grid_y = grid_y.unsqueeze(0)  # (1, H, W)
                
                # Get objectness and class scores
                obj_conf = torch.sigmoid(p[..., 4:5])  # (num_anchors, H, W, 1)
                class_probs = torch.softmax(p[..., 5:], dim=-1)  # (num_anchors, H, W, num_classes)
                class_conf, class_pred = torch.max(class_probs, dim=-1, keepdim=True)
                
                # Combine confidences
                conf = obj_conf * class_conf  # (num_anchors, H, W, 1)
                
                # Filter by confidence
                mask = (conf > conf_thresh).squeeze(-1)  # (num_anchors, H, W)
                
                if mask.sum() > 0:
                    # Get raw predictions for coordinates
                    # tx, ty are offsets from grid cell (sigmoid to get 0-1 offset within cell)
                    # tw, th are log-scaled width/height
                    tx = torch.sigmoid(p[..., 0])  # (num_anchors, H, W)
                    ty = torch.sigmoid(p[..., 1])
                    tw = p[..., 2].clamp(max=5)  # Clamp to prevent overflow
                    th = p[..., 3].clamp(max=5)
                    
                    # Apply grid offsets - grid broadcasts from (1, H, W) to (num_anchors, H, W)
                    bx = tx + grid_x.float()  # (num_anchors, H, W)
                    by = ty + grid_y.float()  # (num_anchors, H, W)
                    
                    # Width and height are encoded as log fractions of the resized image.
                    bw = torch.exp(tw).clamp(max=float(img_size))
                    bh = torch.exp(th).clamp(max=float(img_size))
                    
                    # Convert to pixel coordinates
                    stride_x = img_size / grid_w
                    stride_y = img_size / grid_h
                    
                    # Center in pixels
                    cx = bx * stride_x
                    cy = by * stride_y
                    pw = (bw * img_size).clamp(max=float(img_size))
                    ph = (bh * img_size).clamp(max=float(img_size))
                    
                    # Convert to xyxy format
                    x1 = (cx - pw / 2)[mask]
                    y1 = (cy - ph / 2)[mask]
                    x2 = (cx + pw / 2)[mask]
                    y2 = (cy + ph / 2)[mask]
                    
                    # Clamp to image bounds
                    x1 = x1.clamp(0, img_size)
                    y1 = y1.clamp(0, img_size)
                    x2 = x2.clamp(0, img_size)
                    y2 = y2.clamp(0, img_size)
                    
                    boxes_filtered = torch.stack([x1, y1, x2, y2], dim=-1)
                    scores_filtered = conf[mask].squeeze(-1)
                    labels_filtered = class_pred[mask].squeeze(-1)
                    
                    boxes.append(boxes_filtered)
                    scores.append(scores_filtered)
                    labels.append(labels_filtered)
            
            if len(boxes) > 0:
                all_detections.append({
                    'boxes': torch.cat(boxes, dim=0),
                    'scores': torch.cat(scores, dim=0),
                    'labels': torch.cat(labels, dim=0).long()
                })
            else:
                all_detections.append({
                    'boxes': torch.zeros((0, 4), device=preds[0].device),
                    'scores': torch.zeros((0,), device=preds[0].device),
                    'labels': torch.zeros((0,), device=preds[0].device, dtype=torch.long)
                })
        
        return all_detections
    
    def predict(self, x, conf_thresh: float = 0.5, iou_thresh: float = 0.45):
        """
        Run inference and post-process predictions.
        
        Args:
            x: Input images (B, 3, H, W)
            conf_thresh: Confidence threshold for filtering
            iou_thresh: IoU threshold for NMS
        
        Returns:
            List of detections per image, each containing boxes, scores, labels
        """
        self.eval()
        with torch.no_grad():
            pred_large, pred_medium, pred_small = self.forward(x)
            
            predictions = self.get_detections(
                [pred_large, pred_medium, pred_small],
                conf_thresh=conf_thresh,
                img_size=x.shape[-1]
            )
            
            # Apply NMS
            predictions = self._apply_nms(predictions, iou_thresh=iou_thresh)
        
        return predictions
    
    def _decode_predictions(self, preds: List[torch.Tensor], conf_thresh: float):
        """
        Decode raw predictions to bounding boxes.
        Simplified decoding for prototype.
        """
        batch_size = preds[0].size(0)
        all_boxes = []
        
        for b in range(batch_size):
            boxes = []
            scores = []
            labels = []
            
            for pred in preds:
                # Extract predictions for this image
                p = pred[b]  # (num_anchors, H, W, 5 + num_classes)
                
                # Get objectness and class scores
                obj_conf = torch.sigmoid(p[..., 4:5])
                class_conf, class_pred = torch.max(torch.softmax(p[..., 5:], dim=-1), dim=-1, keepdim=True)
                
                # Combine confidences
                conf = obj_conf * class_conf
                
                # Filter by confidence
                mask = (conf > conf_thresh).squeeze(-1)
                
                if mask.sum() > 0:
                    # Get box coordinates (simplified, assuming center format)
                    xy = torch.sigmoid(p[..., :2])
                    wh = torch.exp(p[..., 2:4])
                    
                    # Convert to xyxy format (simplified)
                    # In full implementation, this would account for anchors and grid offsets
                    boxes_filtered = torch.cat([xy, wh], dim=-1)[mask]
                    scores_filtered = conf[mask]
                    labels_filtered = class_pred[mask]
                    
                    boxes.append(boxes_filtered)
                    scores.append(scores_filtered)
                    labels.append(labels_filtered)
            
            if len(boxes) > 0:
                all_boxes.append({
                    'boxes': torch.cat(boxes, dim=0),
                    'scores': torch.cat(scores, dim=0),
                    'labels': torch.cat(labels, dim=0)
                })
            else:
                all_boxes.append({
                    'boxes': torch.zeros((0, 4)),
                    'scores': torch.zeros((0,)),
                    'labels': torch.zeros((0,), dtype=torch.long)
                })
        
        return all_boxes
    
    def _apply_nms(self, predictions: List[Dict], iou_thresh: float):
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        from torchvision.ops import nms
        
        for pred in predictions:
            if len(pred['boxes']) > 0:
                # Ensure scores is 1D
                scores = pred['scores'].squeeze()
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)
                
                # Apply NMS
                keep = nms(pred['boxes'], scores, iou_thresh)
                pred['boxes'] = pred['boxes'][keep]
                pred['scores'] = pred['scores'][keep]
                pred['labels'] = pred['labels'][keep]
        
        return predictions


def create_yolo_model(num_classes: int = 15, device: str = 'cuda'):
    """
    Create YOLO detector model from scratch.
    
    Args:
        num_classes: Number of target symbol classes
        device: Device to place model on
    
    Returns:
        model: YOLODetector instance
    """
    model = YOLODetector(num_classes=num_classes)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Created YOLO detector from scratch:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model
