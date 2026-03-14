#!/usr/bin/env python3
"""
Places365 ResNet50 → SlowFast 3D Weight Inflation Script
=========================================================
Converts a 2D Places365-pretrained ResNet50 checkpoint into a 3D SlowFast
checkpoint by inflating convolutional filters along the temporal dimension.

Usage (run from TamperingDetection/):
    python inflate_places365_to_slowfast.py \
        --places365 ../Backup/UHCTD_Pretrained/resnet50_places365.pth.tar \
        --config SlowFast-main/configs/UHCTD/SLOWFAST_UHCTD_Places365.yaml \
        --output SlowFast-main/checkpoint/places365_slowfast_slow_pathway.pth

What it does:
  1. Loads the 2D Places365 ResNet50 checkpoint.
  2. Builds the SlowFast model to discover the exact 3D weight shapes and names.
  3. Maps 2D weights exclusively onto the Slow pathway (pathway 0).
     The Fast pathway (pathway 1) keeps its random initialisation—correct
     because it processes either RGB or RefDiff, neither of which Places365
     has ever seen.
  4. Inflates conv filters using the I3D formula:
         W_3D[:, :, t, :, :] = W_2D / T   for all t in {0 … T-1}
  5. Zero-pads extra input channels that arise from the Fast→Slow lateral
     fusion connections (these are learned from scratch during fine-tuning).
  6. The classification head (head.projection.*) is left as random init
     because we have 4 classes, not 365.
  7. Saves a training-ready checkpoint compatible with train_uhctd_slowfast.py.

Theoretical rationale:
  The original UHCTD paper (Yohanandan et al.) used Places365-pretrained
  ResNet50 as a spatial feature extractor that understands "what a normal
  surveillance scene looks like". By inflating those weights into the Slow
  pathway of SlowFast, we give the model the same scene-geometry priors but
  within a 3D temporal architecture, following the weight inflation approach
  of Carreira & Zisserman (I3D, CVPR 2017).

Key mapping: standard torchvision ResNet50 → SlowFast Slow (pathway 0)
  conv1.weight              → s1.pathway0_stem.conv.weight      (T=1)
  bn1.*                     → s1.pathway0_stem.bn.*
  layer1.N.conv1            → s2.pathway0_resN.branch2.a        (T=1)
  layer1.N.bn1.*            → s2.pathway0_resN.branch2.a_bn.*
  layer1.N.conv2            → s2.pathway0_resN.branch2.b        (T=1 for s2/s3)
  layer1.N.bn2.*            → s2.pathway0_resN.branch2.b_bn.*
  layer1.N.conv3            → s2.pathway0_resN.branch2.c        (T=1)
  layer1.N.bn3.*            → s2.pathway0_resN.branch2.c_bn.*
  layer1.N.downsample.0     → s2.pathway0_resN.branch1          (T=1, block 0 only)
  layer1.N.downsample.1.*   → s2.pathway0_resN.branch1_bn.*
  layer2 → s3, layer3 → s4 (T=3 for b conv), layer4 → s5 (T=3 for b conv)
"""

import sys
import os
import argparse
from collections import OrderedDict

import torch

# ── SlowFast paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'SlowFast-main'))

from slowfast.config.defaults import get_cfg
from slowfast.models import build_model

# ════════════════════════════════════════════════════════════════════════════
# Architecture mapping tables
# ════════════════════════════════════════════════════════════════════════════

# ResNet50 stage name → SlowFast stage name
STAGE_MAP = {
    'layer1': 's2',   # residual blocks 0-2  (3 blocks)
    'layer2': 's3',   # residual blocks 0-3  (4 blocks)
    'layer3': 's4',   # residual blocks 0-5  (6 blocks) → branch2.b has T=3
    'layer4': 's5',   # residual blocks 0-2  (3 blocks) → branch2.b has T=3
}

# resnet50 block-internal sub-name → SlowFast branch name
# (used to reconstruct the full SlowFast key from the ResNet50 key)
BLOCK_SUBKEY_MAP = {
    'conv1':         'branch2.a',
    'bn1':           'branch2.a_bn',
    'conv2':         'branch2.b',
    'bn2':           'branch2.b_bn',
    'conv3':         'branch2.c',
    'bn3':           'branch2.c_bn',
    'downsample.0':  'branch1',
    'downsample.1':  'branch1_bn',
}

BN_PARAMS = ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked')

# ════════════════════════════════════════════════════════════════════════════
# Step 1 – load Places365 checkpoint
# ════════════════════════════════════════════════════════════════════════════

def load_places365(path: str) -> OrderedDict:
    """
    Load and normalise a Places365 ResNet50 checkpoint.

    Returns a clean state_dict with no 'module.' prefix.
    """
    print(f"\n{'='*60}")
    print(f"Loading Places365 checkpoint: {path}")
    raw = torch.load(path, map_location='cpu', weights_only=False)

    # Some formats use {'state_dict': …, 'arch': …, 'best_prec1': …}
    if isinstance(raw, dict) and 'state_dict' in raw:
        sd = raw['state_dict']
        arch = raw.get('arch', 'unknown')
        acc  = raw.get('best_prec1', raw.get('best_acc1', None))
        print(f"  Architecture  : {arch}")
        if acc is not None:
            print(f"  Best val acc  : {acc:.3f}")
    else:
        sd = raw

    # Strip DataParallel 'module.' prefix if present
    clean = OrderedDict()
    for k, v in sd.items():
        new_k = k[len('module.'):] if k.startswith('module.') else k
        clean[new_k] = v

    print(f"  Total keys    : {len(clean)}")
    shapes = {k: tuple(v.shape) for k, v in clean.items()}
    print(f"  Stem conv     : conv1.weight → {shapes.get('conv1.weight', 'MISSING')}")
    print(f"  FC layer      : fc.weight    → {shapes.get('fc.weight', 'MISSING')} (will be SKIPPED)")

    return clean


# ════════════════════════════════════════════════════════════════════════════
# Step 2 – build SlowFast model and get its state_dict
# ════════════════════════════════════════════════════════════════════════════

def build_slowfast(config_path: str):
    """Build slave SlowFast model and return (model, state_dict)."""
    print(f"\n{'='*60}")
    print(f"Building SlowFast model from config: {config_path}")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.freeze()

    model = build_model(cfg)
    sd = model.state_dict()

    # Print shape summary for diagnostic purposes
    slow_keys = [k for k in sd.keys() if 'pathway0' in k]
    fast_keys = [k for k in sd.keys() if 'pathway1' in k]
    head_keys = [k for k in sd.keys() if 'head.' in k]
    fuse_keys = [k for k in sd.keys() if '_fuse.' in k]
    print(f"  Slow pathway keys  : {len(slow_keys)}")
    print(f"  Fast pathway keys  : {len(fast_keys)}  [will NOT be loaded - stays random]")
    print(f"  Fusion layer keys  : {len(fuse_keys)}  [will NOT be loaded - stays random]")
    print(f"  Head keys          : {len(head_keys)}  [will NOT be loaded - stays random]")

    return model, sd, cfg


# ════════════════════════════════════════════════════════════════════════════
# Step 3 – build the key translation map
# ════════════════════════════════════════════════════════════════════════════

def build_key_map(places365_sd: OrderedDict) -> dict:
    """
    Build a mapping:
        places365_key → slowfast_key

    Only Slow-pathway keys are included; fc, fast-pathway, and head keys are
    not in the output (they stay as random init in the target model).

    Returns dict: {p365_key: sf_key}
    """
    mapping = {}

    # ── Stem ──────────────────────────────────────────────────────────────────
    if 'conv1.weight' in places365_sd:
        mapping['conv1.weight'] = 's1.pathway0_stem.conv.weight'

    for sfx in BN_PARAMS:
        p_key = f'bn1.{sfx}'
        if p_key in places365_sd:
            mapping[p_key] = f's1.pathway0_stem.bn.{sfx}'

    # ── Residual layers ────────────────────────────────────────────────────────
    for p_layer, sf_stage in STAGE_MAP.items():
        # We discover block indices dynamically from the existing keys
        block_set = set()
        for k in places365_sd:
            if k.startswith(f'{p_layer}.'):
                parts = k.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    block_set.add(int(parts[1]))

        for b in sorted(block_set):
            for p_sub, sf_branch in BLOCK_SUBKEY_MAP.items():
                if p_sub.startswith('downsample'):
                    if 'downsample.0' == p_sub:
                        # shortcut Conv
                        p_key = f'{p_layer}.{b}.downsample.0.weight'
                        if p_key in places365_sd:
                            mapping[p_key] = f'{sf_stage}.pathway0_res{b}.branch1.weight'
                    else:
                        # shortcut BN (downsample.1)
                        for sfx in BN_PARAMS:
                            p_key = f'{p_layer}.{b}.downsample.1.{sfx}'
                            if p_key in places365_sd:
                                mapping[p_key] = f'{sf_stage}.pathway0_res{b}.branch1_bn.{sfx}'
                elif p_sub.startswith('bn'):
                    # BN layer inside the bottleneck
                    for sfx in BN_PARAMS:
                        p_key = f'{p_layer}.{b}.{p_sub}.{sfx}'
                        if p_key in places365_sd:
                            mapping[p_key] = f'{sf_stage}.pathway0_res{b}.{sf_branch}.{sfx}'
                else:
                    # Conv layer
                    p_key = f'{p_layer}.{b}.{p_sub}.weight'
                    if p_key in places365_sd:
                        mapping[p_key] = f'{sf_stage}.pathway0_res{b}.{sf_branch}.weight'

    return mapping


# ════════════════════════════════════════════════════════════════════════════
# Step 4 – inflate and apply weights
# ════════════════════════════════════════════════════════════════════════════

def inflate_conv_weight(w2d: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Inflate a 2D conv weight [C_out, C_in_2d, H, W] → 3D [C_out, C_in_3d, T, H, W].

    I3D inflation formula (Carreira & Zisserman, CVPR 2017):
        W_3D[:, :, t, :, :] = W_2D / T    for all t

    Channel padding:
        If C_in_3d > C_in_2d (fusion-channel mismatch), the extra input
        channels are zero-padded.  This is correct because those channels carry
        information from the Fast pathway, which Places365 has never seen.
        The network will learn those channels from scratch during fine-tuning
        while the Places365 channels provide a strong warm start.

    Special cases:
        T = 1 → equivalent to inserting the temporal dim (division by 1 is no-op).
    """
    assert w2d.dim() == 4, f"Expected 4-D weight, got {w2d.shape}"

    C_out_2d, C_in_2d, H2, W2 = w2d.shape
    C_out_3d, C_in_3d, T, H3, W3 = target_shape

    assert C_out_2d == C_out_3d, (
        f"Output channel mismatch: 2D={C_out_2d}, 3D={C_out_3d}"
    )
    assert H2 == H3 and W2 == W3, (
        f"Spatial mismatch: 2D=({H2},{W2}), 3D=({H3},{W3})"
    )

    # Allocate output (zeros → extra input channels stay at zero)
    w3d = torch.zeros(target_shape, dtype=w2d.dtype)

    n_in = min(C_in_2d, C_in_3d)     # number of channels we can copy
    # expand along T, normalise
    w3d[:, :n_in, :, :, :] = (
        w2d[:, :n_in, :, :].unsqueeze(2).expand(-1, -1, T, -1, -1) / T
    )

    return w3d


def apply_inflated_weights(
    places365_sd: OrderedDict,
    sf_sd: OrderedDict,
    key_map: dict,
) -> tuple:
    """
    Apply Places365 weights to a copy of the SlowFast state_dict.

    Returns (new_sd, stats) where stats contains matched/skipped/padded counts.
    """
    new_sd = OrderedDict()
    for k, v in sf_sd.items():
        new_sd[k] = v.clone()      # start from SlowFast random init

    matched  = []   # (sf_key, p365_shape, sf_shape, T, padded_channels)
    skipped  = []   # (sf_key, reason)

    for p_key, sf_key in key_map.items():
        w_src = places365_sd[p_key]

        if sf_key not in new_sd:
            skipped.append((sf_key, 'key absent from SlowFast model'))
            continue

        w_dst = new_sd[sf_key]

        # ── Conv weight (4D src → 5D dst) ────────────────────────────────────
        if w_src.dim() == 4 and w_dst.dim() == 5:
            try:
                inflated = inflate_conv_weight(w_src, tuple(w_dst.shape))
                extra_ch = w_dst.shape[1] - w_src.shape[1]
                T_val    = w_dst.shape[2]
                new_sd[sf_key] = inflated
                matched.append({
                    'sf_key':       sf_key,
                    'p365_shape':   tuple(w_src.shape),
                    'sf_shape':     tuple(w_dst.shape),
                    'T':            T_val,
                    'padded_ch':    max(0, extra_ch),
                })
            except Exception as exc:
                skipped.append((sf_key, str(exc)))

        # ── BN / scalar (1D) ─────────────────────────────────────────────────
        elif w_src.dim() == 1 and w_dst.dim() == 1:
            if w_src.shape == w_dst.shape:
                new_sd[sf_key] = w_src.clone()
                matched.append({
                    'sf_key':       sf_key,
                    'p365_shape':   tuple(w_src.shape),
                    'sf_shape':     tuple(w_dst.shape),
                    'T':            None,
                    'padded_ch':    0,
                })
            else:
                skipped.append((sf_key, f'BN shape mismatch {w_src.shape} vs {w_dst.shape}'))

        # ── num_batches_tracked (scalar ≈ 0-dim) ─────────────────────────────
        elif w_src.dim() == 0 and w_dst.dim() == 0:
            new_sd[sf_key] = w_src.clone()
            matched.append({
                'sf_key':     sf_key,
                'p365_shape': tuple(w_src.shape),
                'sf_shape':   tuple(w_dst.shape),
                'T':          None,
                'padded_ch':  0,
            })

        # ── Unexpected ──────────────────────────────────────────────────────
        else:
            skipped.append((
                sf_key,
                f'dim mismatch src.dim()={w_src.dim()} dst.dim()={w_dst.dim()}'
            ))

    stats = {
        'matched':  matched,
        'skipped':  skipped,
        'n_matched': len(matched),
        'n_skipped': len(skipped),
        'n_padded':  sum(1 for m in matched if m['padded_ch'] > 0),
        'n_inflated': sum(1 for m in matched if m['T'] is not None and m['T'] > 1),
    }
    return new_sd, stats


# ════════════════════════════════════════════════════════════════════════════
# Step 5 – report
# ════════════════════════════════════════════════════════════════════════════

def print_report(stats: dict, slowfast_sd: OrderedDict):
    """Print a detailed summary of the inflation results."""
    print(f"\n{'='*60}")
    print("INFLATION REPORT")
    print(f"{'='*60}")
    print(f"  Keys matched (loaded) : {stats['n_matched']}")
    print(f"  Keys skipped          : {stats['n_skipped']}")
    print(f"  Conv weights inflated : {stats['n_inflated']}  (T > 1)")
    print(f"  Fusion-channel pads   : {stats['n_padded']}  (zeros for Fast→Slow channels)")

    # Count how many SlowFast keys were NOT touched
    slow_keys = [k for k in slowfast_sd.keys() if 'pathway0' in k]
    matched_set = {m['sf_key'] for m in stats['matched']}
    untouched_slow = [k for k in slow_keys if k not in matched_set]
    print(f"\n  SlowFast Slow pathway keys   : {len(slow_keys)}")
    print(f"    └─ From Places365          : {sum(1 for k in slow_keys if k in matched_set)}")
    print(f"    └─ Not matched (random)    : {len(untouched_slow)}")
    if untouched_slow:
        for k in untouched_slow[:5]:
            print(f"         {k}")
        if len(untouched_slow) > 5:
            print(f"         … and {len(untouched_slow)-5} more")

    print(f"\n  Fast pathway             : RANDOM INIT (by design)")
    print(f"  Fusion layers            : RANDOM INIT (by design)")
    print(f"  Classification head      : RANDOM INIT (4-class ≠ 365-class)")

    print(f"\n  Matched weights (sample):")
    print(f"  {'Places365 shape':<22}  {'T':>3}  {'Pad':>5}  SlowFast key")
    print(f"  {'-'*22}  {'-'*3}  {'-'*5}  {'-'*40}")
    for m in stats['matched'][:12]:
        T_str  = str(m['T']) if m['T'] else 'N/A'
        pad_s  = f"+{m['padded_ch']}" if m['padded_ch'] > 0 else '-'
        print(f"  {str(m['p365_shape']):<22}  {T_str:>3}  {pad_s:>5}  {m['sf_key']}")
    if len(stats['matched']) > 12:
        print(f"  … and {len(stats['matched'])-12} more matched keys")

    if stats['skipped']:
        print(f"\n  Skipped keys:")
        for sf_key, reason in stats['skipped'][:10]:
            print(f"    ✗ {sf_key[:55]:<55} : {reason}")
        if len(stats['skipped']) > 10:
            print(f"    … and {len(stats['skipped'])-10} more skipped")

    print(f"\n{'='*60}")
    print("PATHWAY INITIALISATION STRATEGY")
    print(f"{'='*60}")
    print("  ┌─────────────────────────────────────────────────────────")
    print("  │  Component            │ Init          │ Rationale")
    print("  ├─────────────────────────────────────────────────────────")
    print("  │  Slow pathway (RGB)   │ Places365 ✓   │ Scene geometry priors")
    print("  │  Fast pathway         │ Random        │ Novel temporal/RefDiff signal")
    print("  │  Fusion layers        │ Random        │ Learns cross-pathway correlation")
    print("  │  Classification head  │ Random        │ 4-class tamper, not 365-scene")
    print("  └─────────────────────────────────────────────────────────")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Inflate Places365 ResNet50 weights into SlowFast 3D Slow pathway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--places365',
        default='../Backup/UHCTD_Pretrained/resnet50_places365.pth.tar',
        help='Path to resnet50_places365.pth.tar  '
             '(default: ../Backup/UHCTD_Pretrained/resnet50_places365.pth.tar)',
    )
    parser.add_argument(
        '--config',
        default='SlowFast-main/configs/UHCTD/SLOWFAST_UHCTD_Places365.yaml',
        help='SlowFast YAML config to determine model architecture  '
             '(default: SlowFast-main/configs/UHCTD/SLOWFAST_UHCTD_Places365.yaml)',
    )
    parser.add_argument(
        '--output',
        default='SlowFast-main/checkpoint/places365_slowfast_slow_pathway.pth',
        help='Output path for the inflated checkpoint  '
             '(default: SlowFast-main/checkpoint/places365_slowfast_slow_pathway.pth)',
    )
    args = parser.parse_args()

    # ── Resolve paths relative to script location ─────────────────────────────
    def _abs(p):
        return os.path.join(SCRIPT_DIR, p) if not os.path.isabs(p) else p

    places365_path = _abs(args.places365)
    config_path    = _abs(args.config)
    output_path    = _abs(args.output)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(places365_path):
        print(f"ERROR: Places365 checkpoint not found: {places365_path}")
        return 1
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Step 1: Load Places365 ────────────────────────────────────────────────
    places365_sd = load_places365(places365_path)

    # ── Step 2: Build SlowFast and get its initial state_dict ─────────────────
    model, sf_sd, cfg = build_slowfast(config_path)

    # ── Step 3: Build the key translation map ─────────────────────────────────
    print(f"\n{'='*60}")
    print("Building key translation map …")
    key_map = build_key_map(places365_sd)
    print(f"  Translation pairs found: {len(key_map)}")

    # ── Step 4: Apply inflated weights ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Inflating and applying weights …")
    new_sd, stats = apply_inflated_weights(places365_sd, sf_sd, key_map)

    # ── Step 5: Report ─────────────────────────────────────────────────────────
    print_report(stats, sf_sd)

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Saving inflated checkpoint → {output_path}")

    # We save in the same format expected by train_uhctd_slowfast.py --pretrained
    payload = {
        'model_state_dict':       new_sd,
        'inflation_stats': {
            'n_matched':  stats['n_matched'],
            'n_skipped':  stats['n_skipped'],
            'n_inflated': stats['n_inflated'],
            'n_padded':   stats['n_padded'],
        },
        'source': 'places365_resnet50_inflated_to_slowfast',
        'config': config_path,
        'places365_ckpt': places365_path,
    }
    torch.save(payload, output_path)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"  Saved: {output_path}  ({size_mb:.1f} MB)")
    print(f"\n  Next step: train with")
    print(f"    python train_uhctd_slowfast.py \\")
    print(f"        --experiment E1 \\")
    print(f"        --model SlowFast_R50_Places365 \\")
    print(f"        --config SlowFast-main/configs/UHCTD/SLOWFAST_UHCTD_Places365.yaml \\")
    print(f"        --pretrained SlowFast-main/checkpoint/places365_slowfast_slow_pathway.pth")
    print(f"{'='*60}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
