"""Debug script to check model output shapes."""

import torch
from melodious.model import create_yolo_model

# Create model
model = create_yolo_model(num_classes=15, device='cpu')
print("Model created")

# Create dummy input
batch_size = 2
img_size = 640
x = torch.randn(batch_size, 3, img_size, img_size)

# Forward pass
print(f"\nInput shape: {x.shape}")
predictions = model(x)

print(f"\nNumber of prediction scales: {len(predictions)}")
for i, pred in enumerate(predictions):
    print(f"  Scale {i}: {pred.shape}")
    
# Check expected shape
num_classes = 15
num_anchors = 3
expected_channels = num_anchors * (5 + num_classes)  # 3 * 20 = 60
print(f"\nExpected output channels: {expected_channels}")
print(f"  = {num_anchors} anchors * ({5} bbox+conf + {num_classes} classes)")
