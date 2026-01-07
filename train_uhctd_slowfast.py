#!/usr/bin/env python3
"""
Training script for UHCTD Camera Tampering Detection using SlowFast
"""

import sys
import os
sys.path.append('SlowFast-main')

import torch
from tqdm import tqdm
from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.utils.checkpoint import load_checkpoint
from slowfast.datasets import build_dataset
from slowfast.utils.meters import TrainMeter, ValMeter
from slowfast.utils.misc import launch_job

# Explicitly import Uhctd dataset to ensure registry registration
from slowfast.datasets.uhctd_dataset import Uhctd

# IMPORTANT: Set the multiprocessing start method to 'spawn'
# This is crucial for fixing hangs when using multiple workers,
# especially on mounted filesystems (like /mnt/d/) or with CUDA.
torch.multiprocessing.set_start_method('spawn', force=True)

def setup_config():
    """Setup configuration for UHCTD training"""
    cfg = get_cfg()

    # Load UHCTD configuration
    config_file = "SlowFast-main/configs/UHCTD/SLOWFAST_UHCTD.yaml"

    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        return None

    cfg.merge_from_file(config_file)
    cfg.freeze()

    return cfg

def validate_dataset(cfg):
    """Validate that the UHCTD dataset loads correctly"""
    print("Validating UHCTD dataset...")

    try:
        # Build dataset
        train_dataset = build_dataset("UHCTD", cfg, "train")
        val_dataset = build_dataset("UHCTD", cfg, "test")

        print(f"Training dataset: {len(train_dataset)} clips")
        print(f"Validation dataset: {len(val_dataset)} clips")

        # Test getting a sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            frames, labels, idx, time_idx, extra_data = sample
            print(f"Sample loaded successfully")
            print(f"   Frames shape: {frames.shape if hasattr(frames, 'shape') else 'Variable'}")
            print(f"   Labels: {labels}")
            print(f"   Video: {os.path.basename(extra_data['video_path'])}")

        return True

    except Exception as e:
        print(f"Dataset validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_model(cfg):
    """Train the SlowFast model on UHCTD dataset"""
    print("Starting UHCTD training...")

    # Build model
    model = build_model(cfg)

    # Setup optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )

    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.SOLVER.MAX_EPOCH
    )

    # Build datasets - Force sequential processing for video data to prevent GPU starvation
    train_loader = torch.utils.data.DataLoader(
        build_dataset("UHCTD", cfg, "train"),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # Critical: sequential loading prevents memory explosion
        pin_memory=True  # Keep for fast GPU transfer
    )

    val_loader = torch.utils.data.DataLoader(
        build_dataset("UHCTD", cfg, "test"),
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=8,  # Critical: sequential loading prevents memory explosion
        pin_memory=True  # Keep for fast GPU transfer
    )

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Training on device: {device}")
    print(f"Total training batches: {len(train_loader)}")
    print(f"Total validation batches: {len(val_loader)}")

    best_acc = 0.0

    # Total epochs progress bar
    epoch_pbar = tqdm(range(cfg.SOLVER.MAX_EPOCH), desc="Training",
                      unit="epoch", ncols=100, position=0)

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Training batches progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.SOLVER.MAX_EPOCH} [Train]",
                         unit="batch", ncols=100, leave=False)

        for frames, labels, _, _, _ in train_pbar:
            # frames is a list of tensors for multi-pathway (SlowFast)
            frames = [f.to(device) for f in frames]
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = torch.nn.functional.cross_entropy(outputs, labels.argmax(dim=1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels.argmax(dim=1)).sum().item()

            # Update progress bar with current loss
            current_loss = train_loss / len(train_loader)
            current_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        train_acc = 100. * train_correct / train_total
        train_pbar.close()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Validation progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.SOLVER.MAX_EPOCH} [Val]",
                       unit="batch", ncols=100, leave=False)

        with torch.no_grad():
            for frames, labels, _, _, _ in val_pbar:
                # frames is a list of tensors for multi-pathway (SlowFast)
                frames = [f.to(device) for f in frames]
                labels = labels.to(device)

                outputs = model(frames)
                loss = torch.nn.functional.cross_entropy(outputs, labels.argmax(dim=1))

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels.argmax(dim=1)).sum().item()

                # Update progress bar with validation loss
                current_val_loss = val_loss / len(val_loader)
                current_val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'val_loss': f'{current_val_loss:.4f}',
                    'val_acc': f'{current_val_acc:.2f}%'
                })

        val_acc = 100. * val_correct / val_total
        val_pbar.close()

        scheduler.step()

        # Update overall progress bar with epoch results
        epoch_pbar.set_postfix({
            'train_acc': f'{train_acc:.2f}%',
            'val_acc': f'{val_acc:.2f}%',
            'best': f'{best_acc:.2f}%'
        })

        print(f"Epoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
            }, os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
            print(f"Saved best model with validation accuracy: {best_acc:.2f}%")

    epoch_pbar.close()
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")

def main():
    print("UHCTD + SlowFast Camera Tampering Detection Training")
    print("="*55)

    # Setup configuration
    cfg = setup_config()
    if cfg is None:
        return

    print(f"Configuration loaded from: {cfg}")

    # Validate dataset
    if not validate_dataset(cfg):
        print("Dataset validation failed. Please check your data paths.")
        return

    # Confirm training
    answer = input("\nStart training? This may take several hours. (y/n): ")
    if answer.lower() not in ['y', 'yes']:
        print("Training cancelled.")
        return

    # Start training
    train_model(cfg)

if __name__ == "__main__":
    main()
