#!/usr/bin/env python3
"""
Training script for UHCTD Camera Tampering Detection using SlowFast.

Key improvements over original:
  ● --config flag to select RGB-only (SLOWFAST_UHCTD_RGB.yaml) or flow (SLOWFAST_UHCTD.yaml)
  ● 1:1:1:1 balanced class weights (no manual up-weighting needed; dataset is now balanced)
  ● Early stopping based on macro F1 (correct metric under class imbalance)
  ● Works with the updated uhctd_dataset.py that produces much more training data
"""

import sys
import os
import argparse

sys.path.append('SlowFast-main')

import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from slowfast.config.defaults import get_cfg
from slowfast.models import build_model

# Explicitly import Uhctd dataset to ensure registry registration
from slowfast.datasets.uhctd_dataset import Uhctd
from slowfast.datasets import build_dataset

# Spawn is required for multi-worker DataLoader on mounted filesystems + CUDA
torch.multiprocessing.set_start_method('spawn', force=True)

# ════════════════════════════════════════════════════════════════════════════
# Experiment definitions
# ════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "E1": {
        "name": "Cam A Train → Cam A Test",
        "train_cameras": "Camera A",
        "description": "Train on Camera A (Days 1-2), Test on Camera A (Days 3-6)",
    },
    "E2": {
        "name": "Cam B Train → Cam A Test",
        "train_cameras": "Camera B",
        "description": "Train on Camera B (Days 1-2), Test on Camera A (Days 3-6)",
    },
    "E3": {
        "name": "Cam A+B Train → Cam A Test",
        "train_cameras": "Camera A,Camera B",
        "description": "Train on Camera A+B (Days 1-2), Test on Camera A (Days 3-6)",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# Config setup
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = "SlowFast-main/configs/UHCTD/SLOWFAST_UHCTD_RGB.yaml"

# Auto-detect the fast-pathway mode from the config filename.
# The env var is read by uhctd_dataset.py at dataset construction time.
def _set_fast_mode_from_config(config_path):
    """
    Inspect config filename and set UHCTD_FAST_MODE accordingly.
      *REFDIFF* → 'refdiff'
      *RGB*     → 'rgb'
      otherwise → leave unset (falls back to INPUT_CHANNEL_NUM detection)
    """
    upper = os.path.basename(config_path).upper()
    if 'REFDIFF' in upper:
        os.environ['UHCTD_FAST_MODE'] = 'refdiff'
    elif 'RGB' in upper:
        os.environ['UHCTD_FAST_MODE'] = 'rgb'
    # else: leave existing env var (or unset → dataset auto-detects from channel count)
    mode = os.environ.get('UHCTD_FAST_MODE', '<auto from INPUT_CHANNEL_NUM>')
    print(f"Fast-pathway mode: UHCTD_FAST_MODE={mode}")


def setup_config(config_path):
    _set_fast_mode_from_config(config_path)

    cfg = get_cfg()

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return None

    cfg.merge_from_file(config_path)
    cfg.freeze()
    return cfg


# ════════════════════════════════════════════════════════════════════════════
# Dataset validation
# ════════════════════════════════════════════════════════════════════════════

def validate_dataset(cfg):
    """Quickly validate dataset loads and returns correct shapes."""
    print("Validating UHCTD dataset…")
    try:
        train_ds = build_dataset("UHCTD", cfg, "train")
        val_ds   = build_dataset("UHCTD", cfg, "test")
        print(f"  Training clips : {len(train_ds)}")
        print(f"  Validation clips: {len(val_ds)}")

        if len(train_ds) > 0:
            sample = train_ds[0]
            frames, labels, idx, time_idx, extra = sample
            print(f"  Sample shapes  : slow={frames[0].shape}  fast={frames[1].shape}")
            print(f"  Label          : {labels}  (video: {os.path.basename(extra['video_path'])})")

        return True
    except Exception as e:
        import traceback
        print(f"Dataset validation failed: {e}")
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════════

def train_model(cfg, max_epochs=20, early_stop_patience=5,
                experiment="E1", model_tag="SlowFast_R50_RGB"):
    exp_info = EXPERIMENTS[experiment]
    checkpoint_name = f"{experiment}_{model_tag}_best.pth"
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, checkpoint_name)

    print("\nStarting UHCTD training…")
    print(f"  Experiment  : {experiment} — {exp_info['name']}")
    print(f"  Description : {exp_info['description']}")
    print(f"  Max epochs  : {max_epochs}  |  Patience: {early_stop_patience}")
    print(f"  Checkpoint  : {checkpoint_name}")
    print(f"  Output dir  : {cfg.OUTPUT_DIR}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(cfg)

    # ── Optimizer (SGD with Nesterov momentum) ───────────────────────────────
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        nesterov=True,
    )

    # ── LR scheduler (cosine with linear warmup) ─────────────────────────────
    warmup_epochs = int(getattr(cfg.SOLVER, 'WARMUP_EPOCHS', 3))
    warmup_lr     = float(getattr(cfg.SOLVER, 'WARMUP_START_LR', cfg.SOLVER.BASE_LR * 0.1))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from warmup_lr → BASE_LR
            return warmup_lr / cfg.SOLVER.BASE_LR + \
                   (1.0 - warmup_lr / cfg.SOLVER.BASE_LR) * epoch / warmup_epochs
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Camera filter ────────────────────────────────────────────────────────
    os.environ['UHCTD_TRAIN_CAMERAS'] = exp_info['train_cameras']
    print(f"  Train cameras: {exp_info['train_cameras']}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader = torch.utils.data.DataLoader(
        build_dataset("UHCTD", cfg, "train"),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        build_dataset("UHCTD", cfg, "test"),
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\n  Device          : {device}")
    print(f"  Train batches   : {len(train_loader)}  (batch_size={cfg.TRAIN.BATCH_SIZE})")
    print(f"  Val batches     : {len(val_loader)}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    # Dataset is now balanced 1:1:1:1, so uniform weights are correct.
    # If you still observe class imbalance after construction, increase
    # tampering-class weights slightly (e.g., [1, 1.5, 1.5, 2.0]).
    criterion = torch.nn.CrossEntropyLoss()
    print("  Loss: CrossEntropyLoss (uniform weights — dataset is 1:1:1:1 balanced)")

    # ── Training state ────────────────────────────────────────────────────────
    best_f1         = 0.0
    best_acc        = 0.0
    epochs_no_improve = 0

    epoch_pbar = tqdm(range(max_epochs), desc="Epochs", unit="ep",
                      ncols=100, position=0)

    for epoch in epoch_pbar:
        # ── Train phase ───────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        train_pbar = tqdm(
            train_loader,
            desc=f"E{epoch+1:02d}/{max_epochs} [Train]",
            unit="batch", ncols=100, leave=False,
        )

        for frames, labels, _, _, _ in train_pbar:
            frames = [f.to(device) for f in frames]
            labels = labels.to(device)
            true_labels = labels.argmax(dim=1)

            optimizer.zero_grad()
            outputs = model(frames)
            loss    = criterion(outputs, true_labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss      += loss.item()
            _, predicted    = outputs.max(1)
            train_total     += true_labels.shape[0]
            train_correct   += predicted.eq(true_labels).sum().item()

            train_pbar.set_postfix(
                loss=f"{train_loss/len(train_loader):.4f}",
                acc=f"{100.*train_correct/train_total:.1f}%",
            )

        train_acc = 100. * train_correct / train_total
        train_pbar.close()

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds, val_labels_all = [], []

        val_pbar = tqdm(
            val_loader,
            desc=f"E{epoch+1:02d}/{max_epochs} [Val]  ",
            unit="batch", ncols=100, leave=False,
        )

        with torch.no_grad():
            for frames, labels, _, _, _ in val_pbar:
                frames      = [f.to(device) for f in frames]
                labels      = labels.to(device)
                true_labels = labels.argmax(dim=1)

                outputs  = model(frames)
                loss     = criterion(outputs, true_labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total   += true_labels.shape[0]
                val_correct += predicted.eq(true_labels).sum().item()

                val_preds.extend(predicted.cpu().numpy())
                val_labels_all.extend(true_labels.cpu().numpy())

                val_pbar.set_postfix(
                    val_loss=f"{val_loss/len(val_loader):.4f}",
                    val_acc=f"{100.*val_correct/val_total:.1f}%",
                )

        val_acc      = 100. * val_correct / val_total
        val_macro_f1 = f1_score(val_labels_all, val_preds, average='macro', zero_division=0)
        val_pbar.close()

        scheduler.step()
        best_acc = max(best_acc, val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_pbar.set_postfix(
            tr_acc=f"{train_acc:.1f}%",
            val_f1=f"{val_macro_f1:.4f}",
            best_f1=f"{best_f1:.4f}",
            lr=f"{current_lr:.5f}",
        )

        print(
            f"\nEpoch {epoch+1:02d}/{max_epochs}  "
            f"lr={current_lr:.5f}  "
            f"train_loss={train_loss/len(train_loader):.4f}  train_acc={train_acc:.2f}%  "
            f"val_loss={val_loss/len(val_loader):.4f}  val_acc={val_acc:.2f}%  "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        # ── Save best model based on macro F1 ────────────────────────────────
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            epochs_no_improve = 0
            torch.save({
                'epoch':            epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy':     val_acc,
                'val_macro_f1':     val_macro_f1,
                'experiment':       experiment,
                'model_tag':        model_tag,
            }, checkpoint_path)
            print(f"  ✓ Saved best model → {checkpoint_name} "
                  f"(acc={val_acc:.2f}%, F1={best_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  ✗ No F1 improvement "
                  f"({epochs_no_improve}/{early_stop_patience}) "
                  f"[best F1={best_f1:.4f}]")
            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping after {epoch+1} epochs.")
                break

    epoch_pbar.close()
    print("\n" + "="*60)
    print("Training complete!")
    print(f"  Checkpoint      : {checkpoint_path}")
    print(f"  Best val acc    : {best_acc:.2f}%")
    print(f"  Best val F1     : {best_f1:.4f}")
    print("="*60)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="UHCTD SlowFast Training")
    parser.add_argument("--experiment", type=str, default="E1",
                        choices=["E1", "E2", "E3"])
    parser.add_argument("--model", type=str, default="SlowFast_R50_RGB",
                        help="Model tag for checkpoint naming "
                             "(default: SlowFast_R50_RGB)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help=f"YAML config path (default: {DEFAULT_CONFIG}). "
                             "Use SLOWFAST_UHCTD_RGB.yaml for RGB-only, "
                             "SLOWFAST_UHCTD.yaml for RGB+Flow.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Max training epochs (default: 20)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience on val macro F1 (default: 5)")
    args = parser.parse_args()

    exp_info = EXPERIMENTS[args.experiment]

    print("UHCTD + SlowFast Camera Tampering Detection — Training")
    print("=" * 60)
    print(f"Experiment  : {args.experiment} — {exp_info['name']}")
    print(f"Model tag   : {args.model}")
    print(f"Config      : {args.config}")
    print(f"Epochs      : {args.epochs}  |  Patience: {args.patience}")
    print(f"Checkpoint  : {args.experiment}_{args.model}_best.pth")
    print("=" * 60)

    cfg = setup_config(args.config)
    if cfg is None:
        return

    if not validate_dataset(cfg):
        print("Dataset validation failed. Aborting.")
        return

    answer = input("\nStart training? (y/n): ").strip().lower()
    if answer not in ('y', 'yes'):
        print("Training cancelled.")
        return

    train_model(
        cfg,
        max_epochs=args.epochs,
        early_stop_patience=args.patience,
        experiment=args.experiment,
        model_tag=args.model,
    )


if __name__ == "__main__":
    main()
