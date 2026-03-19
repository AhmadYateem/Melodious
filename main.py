"""
Main training script for Melodious music notation detector.

Trains YOLO model from scratch on DeepScores dataset subset.
Run with: python main.py
"""

import argparse
import torch

from melodious.dataset import create_dataloaders
from melodious.model import create_yolo_model
from melodious.train import Trainer


def main():
    """
    CLI entrypoint that wires together dataloaders, model creation,
    and the training loop (see `melodious/train.py:Trainer`).

    Keep flags minimal and consistent with README quick commands.
    """
    parser = argparse.ArgumentParser(description='Train Melodious music notation detector')
    parser.add_argument('--dataset', type=str, default='dataset_ds2_dense',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--max-train-samples', type=int, default=None,
                       help='Max training samples (None = use all 1362 images)')
    parser.add_argument('--max-val-samples', type=int, default=None,
                       help='Max validation samples (None = use all 352 images)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of dataloader workers')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory for outputs')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for logs')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will be slow on CPU.")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    print(f"\n{'='*60}")
    print("Melodious: Music Notation Detection")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max train samples: {args.max_train_samples}")
    print(f"Max val samples: {args.max_val_samples}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        root_dir=args.dataset,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    
    # Create model
    print("\nCreating YOLO model from scratch...")
    model = create_yolo_model(num_classes=15, device=device)
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        num_epochs=args.epochs
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    print("\nTraining complete!")
    print(f"Model saved to: {args.output_dir}/yolo_scratch.pth")
    print(f"Training curves saved to: {args.output_dir}/training_curves.png")
    print(f"TensorBoard logs: {args.log_dir}/tensorboard")


if __name__ == '__main__':
    main()
