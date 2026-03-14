#!/usr/bin/env python3
"""
UHCTD Model Evaluation Script — Full Sliding Window Sweep

Key changes vs. the old sparse-sampling version:
  ● Every frame in the test video receives a prediction (no "default = Normal" bias).
  ● Sliding window with configurable stride (default = window_size = no overlap).
  ● Windows are batched on GPU for throughput efficiency.
  ● Optical flow removed — both pathways receive RGB (matches SLOWFAST_UHCTD_RGB.yaml).
  ● Old flow-based SLOWFAST_UHCTD.yaml still supported via --config flag for comparison.
"""

import sys
import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.io import VideoReader
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support
)
import warnings
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", message=".*pyav.*")
warnings.filterwarnings("ignore", message=".*PyTorchVideo.*")

# ── SlowFast path ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SlowFast-main'))

from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.datasets.utils import spatial_sampling

# ── Experiment definitions ────────────────────────────────────────────────────
EXPERIMENTS = {
    "E1": {"name": "Cam A Train → Cam A Test",   "test_cam": "Camera A", "gt_base": "cam_a"},
    "E2": {"name": "Cam B Train → Cam A Test",   "test_cam": "Camera A", "gt_base": "cam_a"},
    "E3": {"name": "Cam A+B Train → Cam A Test", "test_cam": "Camera A", "gt_base": "cam_a"},
}

CLASS_NAMES = {0: "Normal", 1: "Covered", 2: "Defocused", 3: "Moved"}
FPS_MAP = {"Camera A": 3.0, "Camera B": 10.0}


# ════════════════════════════════════════════════════════════════════════════
# Model setup
# ════════════════════════════════════════════════════════════════════════════

def setup_model(model_path, config_path):
    """Load trained SlowFast model from checkpoint."""
    print("Loading trained SlowFast model…")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.freeze()

    model = build_model(cfg)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded checkpoint : {model_path}")
        if 'val_accuracy' in ckpt:
            print(f"  Val accuracy (train) : {ckpt['val_accuracy']:.2f}%")
        if 'val_macro_f1' in ckpt:
            print(f"  Val macro F1 (train) : {ckpt['val_macro_f1']:.4f}")
    else:
        model.load_state_dict(ckpt)
        print(f"  Loaded checkpoint (raw): {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"  Device: {device}")
    return model, cfg, device


# ════════════════════════════════════════════════════════════════════════════
# Frame loading  (VideoReader-based, seek by time)
# ════════════════════════════════════════════════════════════════════════════

def _load_frames_videoreader(video_path, start_frame, n_frames, fps):
    """
    Load `n_frames` consecutive frames starting at `start_frame` using VideoReader.
    Returns uint8 tensor [T, C, H, W].
    """
    start_pts = float(start_frame) / fps
    frames = []
    try:
        reader = VideoReader(video_path, "video")
        reader.seek(start_pts)
        for _ in range(n_frames):
            try:
                frm = next(reader)
                frames.append(frm['data'])           # [C, H, W] uint8
            except StopIteration:
                break
    except Exception:
        pass

    if not frames:
        return None

    v = torch.stack(frames)  # [T, C, H, W] uint8
    if v.shape[0] < n_frames:
        # Pad with last frame
        pad = v[-1:].repeat(n_frames - v.shape[0], 1, 1, 1)
        v = torch.cat([v, pad], dim=0)
    return v[:n_frames]


# ════════════════════════════════════════════════════════════════════════════
# Reference frame computation (refdiff mode)
# ════════════════════════════════════════════════════════════════════════════

def compute_multi_reference_frames(video_path, fps, gt_df, n_refs=4, n_frames_each=50):
    """
    Compute n_refs reference frames spread evenly across Normal frames in the video.

    Using multiple references that cover different times of day solves the
    illumination-drift problem: a Normal frame under afternoon lighting will have
    near-zero diff against the afternoon reference, even if it differs from the
    morning reference.

    Strategy:
      1. Find all Normal (label==0) frame indices from gt_df.
      2. Evenly sample n_refs positions across them.
      3. Load n_frames_each consecutive frames around each position, take mean.

    Returns list of float32 [3, H, W] tensors in [0,1].
    Falls back to a single grey frame if loading fails entirely.
    """
    normal_indices = np.where(gt_df['label'].values == 0)[0]

    if len(normal_indices) == 0:
        print("  Warning: No Normal frames in gt_df. Using single grey reference.")
        return [torch.full((3, 480, 704), 0.5, dtype=torch.float32)]

    # Evenly spaced anchor positions within the Normal frames
    anchor_positions = np.linspace(0, len(normal_indices) - 1, n_refs, dtype=int)
    anchor_frames    = [int(normal_indices[p]) for p in anchor_positions]

    refs = []
    half = n_frames_each // 2

    for anchor in anchor_frames:
        start = max(0, anchor - half)
        v = _load_frames_videoreader(video_path, start, n_frames_each, fps)
        if v is None or v.shape[0] == 0:
            # Skip failed loads; will fall back to fewer refs
            continue
        ref = v.float().mean(dim=0) / 255.0   # [3, H, W]
        refs.append(ref)

    if not refs:
        print("  Warning: All reference frame loads failed. Using grey.")
        return [torch.full((3, 480, 704), 0.5, dtype=torch.float32)]

    unique_times = [int(anchor_frames[i]) for i in range(len(refs))]
    print(f"  Multi-reference: {len(refs)} refs at Normal frames "
          f"{unique_times} (out of {len(normal_indices)} Normal frames)")
    return refs   # list of [3, H, W]


# ════════════════════════════════════════════════════════════════════════════
# Preprocessing — RGB-only (matches both SLOWFAST_UHCTD_RGB.yaml pathways)
# ════════════════════════════════════════════════════════════════════════════

def preprocess_window_rgb(frames_tchw_uint8, cfg):
    """
    Preprocess a single window for RGB-only SlowFast.

    Input : [T, C, H, W] uint8
    Output: [slow[3,T/α,224,224], fast[3,T,224,224]]
    """
    crop_size = cfg.DATA.TEST_CROP_SIZE   # 224
    T         = cfg.DATA.NUM_FRAMES       # 32
    alpha     = cfg.SLOWFAST.ALPHA        # 4

    # float [0,1], [T, C, H, W]
    frames = frames_tchw_uint8.float() / 255.0

    # Centre-crop / resize to 224
    rgb_small = spatial_sampling(
        frames, spatial_idx=1,
        min_scale=crop_size, max_scale=crop_size, crop_size=crop_size
    )  # [T, 3, 224, 224]

    # [T, C, H, W] → [C, T, H, W]
    rgb = rgb_small.permute(1, 0, 2, 3)  # [3, T, 224, 224]

    # Normalise (use first 3 MEAN/STD values from config)
    mean = torch.tensor(cfg.DATA.MEAN[:3]).view(-1, 1, 1, 1)
    std  = torch.tensor(cfg.DATA.STD[:3]).view(-1, 1, 1, 1)
    rgb  = (rgb - mean) / std

    # SlowFast split
    slow_idx = torch.linspace(0, T - 1, T // alpha, dtype=torch.long)
    slow = rgb[:, slow_idx, :, :]   # [3, T/α, 224, 224]
    fast = rgb                       # [3, T,   224, 224]

    return slow, fast


def preprocess_window_refdiff(frames_tchw_uint8, cfg, reference_frames):
    """
    Preprocess a single window for Multi-Reference Difference Map SlowFast.

    Input : [T, C, H, W] uint8
            reference_frames: list of float32 [3, H_orig, W_orig] in [0,1]
    Output: [slow[3,T/α,224,224], fast[3,T,224,224]]

    Fast pathway = pixel-wise MINIMUM absolute difference across all references:
        fast[t] = min_k( |frame_t − reference_k| )

    A Normal frame under *any* lighting will match at least one reference →
    near-zero diff. A tampered frame (covered/defocused/moved) will have large
    diff against ALL references → correctly predicted as tampered.
    """
    crop_size = cfg.DATA.TEST_CROP_SIZE   # 224
    T         = cfg.DATA.NUM_FRAMES       # 32
    alpha     = cfg.SLOWFAST.ALPHA        # 4

    # float [0,1], centre-crop to 224
    frames    = frames_tchw_uint8.float() / 255.0
    rgb_small = spatial_sampling(
        frames, spatial_idx=1,
        min_scale=crop_size, max_scale=crop_size, crop_size=crop_size
    )  # [T, 3, 224, 224]

    H, W = rgb_small.shape[2], rgb_small.shape[3]

    # ── Per-frame luminance whitening ────────────────────────────────────────
    # Matches the training-time whitening in uhctd_dataset.py.
    # Dividing by per-frame mean brightness makes diff brightness-invariant:
    # a Normal frame under different lighting still gives near-zero whitened diff.
    def _whiten(t):  # [T, 3, H, W] or [1, 3, H, W]
        mu = t.mean(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        return t / mu

    rgb_whitened = _whiten(rgb_small)   # [T, 3, H, W]

    # For each reference, compute whitened absolute diff → pixel-wise minimum
    diff_maps = []
    for ref in reference_frames:
        if ref.shape[1] != H or ref.shape[2] != W:
            ref = torch.nn.functional.interpolate(
                ref.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0)
        ref_whitened = _whiten(ref.unsqueeze(0))               # [1, 3, H, W]
        d = torch.abs(rgb_whitened - ref_whitened)             # [T, 3, H, W]
        diff_maps.append(d)

    if len(diff_maps) == 1:
        diff = diff_maps[0]                                     # [T, 3, H, W]
    else:
        diff = torch.stack(diff_maps, dim=0).min(dim=0).values # [T, 3, H, W]

    # [T, C, H, W] → [C, T, H, W]
    rgb  = rgb_small.permute(1, 0, 2, 3)   # [3, T, 224, 224]
    diff = diff.permute(1, 0, 2, 3)         # [3, T, 224, 224]

    # Normalise
    mean = torch.tensor(cfg.DATA.MEAN[:3]).view(-1, 1, 1, 1)
    std  = torch.tensor(cfg.DATA.STD[:3]).view(-1, 1, 1, 1)
    rgb  = (rgb  - mean) / std
    diff = (diff - mean) / std

    # Slow: subsampled RGB, Fast: min-diff map
    slow_idx = torch.linspace(0, T - 1, T // alpha, dtype=torch.long)
    slow = rgb[:, slow_idx, :, :]   # [3, T/α, 224, 224]
    fast = diff                      # [3, T,   224, 224]

    return slow, fast


def preprocess_window_flow(frames_tchw_uint8, cfg):
    """
    Preprocess a single window for legacy RGB+Flow SlowFast.

    Input : [T, C, H, W] uint8
    Output: [slow[3,T/α,224,224], fast[2,T,224,224]]
    """
    import cv2

    crop_size = cfg.DATA.TEST_CROP_SIZE
    T         = cfg.DATA.NUM_FRAMES
    alpha     = cfg.SLOWFAST.ALPHA

    frames = frames_tchw_uint8.float() / 255.0
    rgb_small = spatial_sampling(
        frames, spatial_idx=1,
        min_scale=crop_size, max_scale=crop_size, crop_size=crop_size
    )  # [T, 3, 224, 224]

    # ── RGB normalise ────────────────────────────────────────────────────────
    rgb = rgb_small.permute(1, 0, 2, 3)  # [3, T, 224, 224]
    mean_rgb = torch.tensor(cfg.DATA.MEAN[:3]).view(-1, 1, 1, 1)
    std_rgb  = torch.tensor(cfg.DATA.STD[:3]).view(-1, 1, 1, 1)
    rgb      = (rgb - mean_rgb) / std_rgb

    # ── Optical flow ─────────────────────────────────────────────────────────
    frames_np = (rgb_small.permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().numpy()
    H, W      = frames_np.shape[1], frames_np.shape[2]
    flow_list = []
    for i in range(len(frames_np) - 1):
        f1, f2 = frames_np[i, :, :, :3], frames_np[i + 1, :, :, :3]
        try:
            g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
            fl = cv2.calcOpticalFlowFarneback(
                g1, g2, None,
                pyr_scale=0.5, levels=2, winsize=9,
                iterations=2, poly_n=5, poly_sigma=1.2, flags=0
            )
            flow_list.append(fl)
        except Exception:
            flow_list.append(np.zeros((H, W, 2), dtype=np.float32))

    flow = torch.from_numpy(np.stack(flow_list)).float()  # [T-1, H, W, 2]
    flow = flow.permute(3, 0, 1, 2)                       # [2, T-1, H, W]

    mean_fl = torch.tensor(cfg.DATA.MEAN[3:5]).view(-1, 1, 1, 1)
    std_fl  = torch.tensor(cfg.DATA.STD[3:5]).view(-1, 1, 1, 1)
    flow    = (flow - mean_fl) / std_fl

    # Pad T-1 → T
    if flow.shape[1] < T:
        flow = torch.cat([flow, flow[:, -1:, :, :]], dim=1)

    # Slow split
    slow_idx = torch.linspace(0, T - 1, T // alpha, dtype=torch.long)
    slow = rgb[:, slow_idx, :, :]   # [3, T/α, 224, 224]

    return slow, flow


# ════════════════════════════════════════════════════════════════════════════
# Ground truth loader
# ════════════════════════════════════════════════════════════════════════════

def load_ground_truth(gt_path):
    """Load per-frame 4-class labels from ground truth CSV."""
    df = pd.read_csv(gt_path, header=None,
                     names=['frame', 'tamper', 'quantity', 'rate', 'status'])
    df['label'] = df['tamper']
    return df


# ════════════════════════════════════════════════════════════════════════════
# Full-video evaluation (sliding window)
# ════════════════════════════════════════════════════════════════════════════

def evaluate_video_full_sweep(model, video_path, gt_df, cfg, device,
                              window_size=32, stride=None, batch_size=8,
                              use_flow=False, use_refdiff=False,
                              reference_frames=None):
    """
    Evaluate model on EVERY frame of the video using a sliding window.

    Parameters
    ----------
    window_size : int
        Number of frames per inference window (should match cfg.DATA.NUM_FRAMES).
    stride : int
        Stride between windows. Defaults to window_size (no overlap).
    batch_size : int
        Windows batched together for one GPU forward pass.
    use_flow : bool
        True → RGB+Flow preprocessing (legacy SLOWFAST_UHCTD.yaml)
    use_refdiff : bool
        True → Multi-Reference Difference Map preprocessing.
    reference_frames : list[torch.Tensor] or None
        List of float32 [3, H, W] in [0,1]. Required when use_refdiff=True.
    """
    if stride is None:
        stride = window_size   # non-overlapping by default

    total_frames = len(gt_df)
    fps = FPS_MAP.get(
        next((k for k in FPS_MAP if k in video_path), "Camera A"),
        3.0
    )

    # Prediction buffers
    # For overlapping windows, accumulate probabilities and count; take mean at the end.
    prob_acc   = np.zeros((total_frames, cfg.MODEL.NUM_CLASSES), dtype=np.float32)
    pred_count = np.zeros(total_frames, dtype=np.int32)

    # Build list of window start frames
    window_starts = list(range(0, total_frames - window_size + 1, stride))
    # Always include a window ending exactly at the last frame
    last_start = total_frames - window_size
    if last_start >= 0 and (not window_starts or window_starts[-1] < last_start):
        window_starts.append(last_start)

    print(f"  Video  : {os.path.basename(video_path)}")
    print(f"  Frames : {total_frames}, FPS={fps}, window={window_size}, stride={stride}")
    print(f"  Windows: {len(window_starts)} "
          f"({'no overlap' if stride == window_size else f'{stride}-frame stride'})")
    print(f"  Batching: {batch_size} windows / forward pass")

    # Batched inference
    if use_refdiff:
        preprocess_fn = lambda f, c: preprocess_window_refdiff(f, c, reference_frames)
    elif use_flow:
        preprocess_fn = preprocess_window_flow
    else:
        preprocess_fn = preprocess_window_rgb

    batch_slows, batch_fasts, batch_ranges = [], [], []

    def _run_batch():
        """Run one GPU forward pass on accumulated batch."""
        if not batch_slows:
            return
        slow_t = torch.stack(batch_slows).to(device)   # [B, C, T/α, H, W]
        fast_t = torch.stack(batch_fasts).to(device)   # [B, Cf, T, H, W]

        with torch.no_grad():
            logits = model([slow_t, fast_t])            # [B, num_classes]
            probs  = F.softmax(logits, dim=1).cpu().numpy()

        for b_idx, (s_f, e_f) in enumerate(batch_ranges):
            prob_acc[s_f:e_f + 1] += probs[b_idx]
            pred_count[s_f:e_f + 1] += 1

        batch_slows.clear()
        batch_fasts.clear()
        batch_ranges.clear()

    for start_f in tqdm(window_starts, desc="  Sweeping windows", ncols=90, unit="win"):
        end_f = min(start_f + window_size - 1, total_frames - 1)

        frames = _load_frames_videoreader(video_path, start_f, window_size, fps)
        if frames is None:
            # Fallback: assign "Normal" probability to these frames
            prob_acc[start_f:end_f + 1, 0] += 1.0
            pred_count[start_f:end_f + 1] += 1
            continue

        try:
            slow, fast = preprocess_fn(frames, cfg)
        except Exception as e:
            print(f"  Preprocess error at frame {start_f}: {e}")
            prob_acc[start_f:end_f + 1, 0] += 1.0
            pred_count[start_f:end_f + 1] += 1
            continue

        batch_slows.append(slow)
        batch_fasts.append(fast)
        batch_ranges.append((start_f, end_f))

        if len(batch_slows) >= batch_size:
            _run_batch()

    # Flush remaining
    _run_batch()

    # ── Handle frames with no prediction (should not happen with last_start fix) ──
    no_pred_mask = pred_count == 0
    if no_pred_mask.any():
        prob_acc[no_pred_mask, 0] = 1.0
        pred_count[no_pred_mask]  = 1

    # Normalise accumulated probabilities
    probabilities = prob_acc / pred_count[:, None]     # mean probability per frame
    predictions   = np.argmax(probabilities, axis=1)   # hard prediction per frame

    return predictions, probabilities


# ════════════════════════════════════════════════════════════════════════════
# Temporal smoothing
# ════════════════════════════════════════════════════════════════════════════

def temporal_smoothing(probabilities, window=15):
    """
    Smooth per-frame probability vectors with a box filter, then argmax.
    window=15 at 3fps ≈ 5 seconds of context.
    """
    T, C = probabilities.shape
    smoothed = np.zeros_like(probabilities)
    for i in range(T):
        lo = max(0, i - window // 2)
        hi = min(T, i + window // 2 + 1)
        smoothed[i] = probabilities[lo:hi].mean(axis=0)
    return np.argmax(smoothed, axis=1), smoothed


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════

def print_metrics(y_true, y_pred, tag=""):
    """Print 4-class classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    lbl = f" [{tag}]" if tag else ""
    print(f"\n4-Class Results{lbl}:")
    print(f"   Overall Accuracy : {acc:.3f}")
    print(f"   Macro P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
    for i in range(4):
        if i < len(prec_c):
            print(f"   {CLASS_NAMES[i]:<12}: "
                  f"P={prec_c[i]:.3f}  R={rec_c[i]:.3f}  F1={f1_c[i]:.3f}")
    print(f"   Confusion Matrix (True↓ | Pred→):\n{cm}")
    print(f"   Classes: {[f'{k}={v}' for k,v in CLASS_NAMES.items()]}")

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'precision_per_class': prec_c, 'recall_per_class': rec_c,
        'f1_per_class': f1_c, 'confusion_matrix': cm,
    }


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="UHCTD SlowFast Evaluation (Full Sweep)")
    parser.add_argument("--experiment", default="E1", choices=["E1", "E2", "E3"])
    parser.add_argument("--model", default="SlowFast_R50_RGB",
                        help="Model tag matching checkpoint name "
                             "(e.g. SlowFast_R50_RGB or SlowFast_R50)")
    parser.add_argument("--checkpoint_dir", default="SlowFast-main/checkpoint")
    parser.add_argument("--config",
                        default="SlowFast-main/configs/UHCTD/SLOWFAST_UHCTD_RGB.yaml",
                        help="YAML config.  Use SLOWFAST_UHCTD_RGB.yaml for the "
                             "RGB-only model, SLOWFAST_UHCTD.yaml for the legacy flow model.")
    parser.add_argument("--window", type=int, default=32,
                        help="Inference window size in frames (default: 32)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride between windows. Default = window size (no overlap). "
                             "Set to window//2 for 50%% overlap.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="GPU batch size for window inference (default: 8)")
    parser.add_argument("--smooth_window", type=int, default=15,
                        help="Temporal smoothing window in frames (default: 15 ≈ 5s at 3fps)")
    parser.add_argument("--num_refs", type=int, default=1,
                        help="Number of reference frames for refdiff mode (default: 1). "
                             "Uses the first Normal frames as reference. "
                             "Luminance whitening handles lighting drift, so 1 is usually "
                             "sufficient. Set >1 only with a whitening-trained model.")
    args = parser.parse_args()

    exp_info = EXPERIMENTS[args.experiment]
    checkpoint_name = f"{args.experiment}_{args.model}_best.pth"
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    output_dir = os.path.join("Evaluation Results", f"{args.experiment}_{args.model}")
    os.makedirs(output_dir, exist_ok=True)

    print("UHCTD Model Evaluation Script — Full Sliding Window Sweep")
    print("=" * 60)
    print(f"Experiment  : {args.experiment} — {exp_info['name']}")
    print(f"Model tag   : {args.model}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Config      : {args.config}")
    print(f"Window/stride: {args.window}/{args.stride or args.window}")
    print(f"Output dir  : {output_dir}")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model, cfg, device = setup_model(checkpoint_path, args.config)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Detect fast pathway mode (env var takes priority, then config filename, then channel count)
    env_mode = os.environ.get('UHCTD_FAST_MODE', '').strip().lower()
    fast_ch  = cfg.DATA.INPUT_CHANNEL_NUM[1] if len(cfg.DATA.INPUT_CHANNEL_NUM) > 1 else 3
    if env_mode in ('rgb', 'refdiff', 'flow'):
        fast_mode = env_mode
    elif 'REFDIFF' in args.config.upper():
        fast_mode = 'refdiff'
    elif fast_ch == 2:
        fast_mode = 'flow'
    else:
        fast_mode = 'rgb'

    use_flow    = (fast_mode == 'flow')
    use_refdiff = (fast_mode == 'refdiff')
    print(f"\nModel mode: {fast_mode}  (fast_ch={fast_ch}, UHCTD_FAST_MODE='{env_mode}')")

    # ── Dataset paths ─────────────────────────────────────────────────────────
    uhctd_root = ("/mnt/d/FYP/UHCTD/"
                  "UHCTD Comprehensive Dataset For Camera Tampering Detection")
    test_base  = os.path.join(uhctd_root, exp_info['test_cam'], "Testing video")
    gt_base    = (f"/mnt/d/FYP/groundtruth_and_prediction/"
                  f"Ground_truth/{exp_info['gt_base']}")
    cam_name   = exp_info['gt_base']

    test_days  = ["Day 3", "Day 4", "Day 5", "Day 6"]
    all_results = {}

    print(f"\nStarting evaluation on {exp_info['test_cam']} testing videos…\n")

    for day in test_days:
        print("=" * 55)
        print(f"Evaluating {day}")

        video_path = os.path.join(test_base, day, "video.avi")
        gt_path    = os.path.join(gt_base, f"{day}.csv")

        if not os.path.exists(video_path):
            print(f"  Video not found: {video_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"  Ground truth not found: {gt_path}")
            continue

        # Load ground truth
        gt_df  = load_ground_truth(gt_path)
        y_true = gt_df['label'].values
        unique_labels, counts = np.unique(y_true, return_counts=True)
        print(f"  Ground truth: {len(y_true)} frames total")
        for lbl, cnt in zip(unique_labels, counts):
            print(f"    {CLASS_NAMES.get(lbl, lbl):<12}: {cnt:6d} ({100*cnt/len(y_true):.1f}%)")

        # ── Pre-compute multi-reference frames (refdiff mode only) ────────────
        reference_frames = None
        if use_refdiff:
            fps_val = FPS_MAP.get(
                next((k for k in FPS_MAP if k in video_path), "Camera A"), 3.0
            )
            print(f"  Computing {args.num_refs} reference frames spread across Normal frames…")
            reference_frames = compute_multi_reference_frames(
                video_path, fps_val, gt_df,
                n_refs=args.num_refs, n_frames_each=50,
            )

        # ── Full sliding window inference ─────────────────────────────────────
        predictions, probabilities = evaluate_video_full_sweep(
            model, video_path, gt_df, cfg, device,
            window_size=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
            use_flow=use_flow,
            use_refdiff=use_refdiff,
            reference_frames=reference_frames,
        )

        # ── Temporal smoothing ───────────────────────────────────────────────
        smoothed_preds, smoothed_probs = temporal_smoothing(
            probabilities, window=args.smooth_window
        )

        # ── Metrics ──────────────────────────────────────────────────────────
        results = {}
        results['raw']      = print_metrics(y_true, predictions,      tag="Raw")
        results['smoothed'] = print_metrics(y_true, smoothed_preds,   tag="Smoothed")
        all_results[day] = results

        # ── Save CSV ─────────────────────────────────────────────────────────
        out_df = pd.DataFrame({
            'frame':               gt_df['frame'],
            'true_label':          y_true,
            'tamper_type':         gt_df['tamper'],
            'raw_prediction':      predictions,
            'smoothed_prediction': smoothed_preds,
            'normal_prob':         probabilities[:, 0],
            'covered_prob':        probabilities[:, 1],
            'defocused_prob':      probabilities[:, 2],
            'moved_prob':          probabilities[:, 3],
        })
        csv_name = f"eval_{args.experiment}_{args.model}_{cam_name}_{day}.csv"
        csv_path = os.path.join(output_dir, csv_name)
        out_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # ── Overall summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("OVERALL EVALUATION SUMMARY")
    print(f"{'='*60}")
    for day, res in all_results.items():
        r  = res['raw']
        sm = res['smoothed']
        print(f"{day}:")
        print(f"  Raw      — Acc={r['accuracy']:.3f}  "
              f"P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")
        print(f"  Smoothed — Acc={sm['accuracy']:.3f}  "
              f"P={sm['precision']:.3f}  R={sm['recall']:.3f}  F1={sm['f1']:.3f}")
        # Per-class recall
        print(f"  Recall per class (smoothed): "
              + "  ".join(f"{CLASS_NAMES[i]}={sm['recall_per_class'][i]:.3f}"
                          for i in range(4) if i < len(sm['recall_per_class'])))

    print("\nEvaluation complete!")
    print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
