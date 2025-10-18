"""Quick test of the updated dataset loader."""

from melodious.dataset import DeepScoresDataset, CLASS_NAMES

# Load a small subset
print("Loading dataset...")
ds = DeepScoresDataset('dataset_ds2_dense', split='train', max_samples=10)

print(f"\nDataset statistics:")
print(f"  Number of images: {len(ds)}")
print(f"  Total boxes: {sum(len(ann['boxes']) for ann in ds.annotations)}")

print(f"\nSample annotations:")
for i in range(min(3, len(ds.annotations))):
    ann = ds.annotations[i]
    print(f"\n  Image {i}: {ann['filename']}")
    print(f"    Size: {ann['width']}x{ann['height']}")
    print(f"    Boxes: {len(ann['boxes'])}")
    if ann['labels']:
        label_names = [CLASS_NAMES[l] for l in ann['labels'][:5]]
        print(f"    First labels: {label_names}")

print(f"\nLoading first batch...")
img, target = ds[0]
print(f"  Image tensor shape: {img.shape}")
print(f"  Boxes tensor shape: {target['boxes'].shape}")
print(f"  Labels tensor shape: {target['labels'].shape}")

print("\n✓ Dataset loader working correctly!")
