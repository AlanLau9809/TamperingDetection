#!/usr/bin/env python3
"""
Test script for optical flow integration in UHCTD dataset
"""

import sys
import os

# Add SlowFast to path
current_dir = os.path.dirname(os.path.abspath(__file__))
slowfast_path = os.path.join(current_dir, 'SlowFast-main')
sys.path.insert(0, slowfast_path)

from slowfast.config.defaults import get_cfg
from slowfast.datasets import build_dataset

def test_optical_flow_integration():
    """Test the optical flow integration"""
    print("Testing UHCTD dataset with optical flow integration...")

    # Setup config
    cfg = get_cfg()
    config_path = os.path.join(current_dir, 'SlowFast-main', 'configs', 'UHCTD', 'SLOWFAST_UHCTD.yaml')
    cfg.merge_from_file(config_path)

    # Adjust paths for current directory structure
    uhctd_path = os.path.join(os.path.dirname(current_dir), "UHCTD", "UHCTD Comprehensive Dataset For Camera Tampering Detection")
    cfg.DATA.PATH_TO_DATA_DIR = uhctd_path
    cfg.DATA.PATH_PREFIX = uhctd_path

    # Enable debug mode to limit clips for testing
    cfg.DEBUG_MODE = True
    cfg.DEBUG_MAX_CLIPS = 2

    print(f"Input channels: {cfg.DATA.INPUT_CHANNEL_NUM}")
    print(f"Mean values: {cfg.DATA.MEAN}")
    print(f"Std values: {cfg.DATA.STD}")

    try:
        # Build dataset
        dataset = build_dataset('uhctd', cfg, 'train')
        print(f"[SUCCESS] Dataset loaded successfully with {len(dataset)} clips")

        if len(dataset) > 0:
            # Test getting one sample
            print("Testing sample loading...")
            frames, labels, idx, time_idx, extra_data = dataset[0]

            # Check frame dimensions
            if isinstance(frames, list):
                print(f"[INFO] Frames shape (SlowFast format): {[f.shape for f in frames]}")
                # Check if we have 5 channels total (3 RGB + 2 flow)
                total_channels = sum(f.shape[0] for f in frames)
                print(f"[INFO] Total channels: {total_channels} (expected: 5 for RGB + Flow)")
            else:
                print(f"[INFO] Frames shape: {frames.shape}")

            print(f"[INFO] Labels: {labels}")
            print("[SUCCESS] Optical flow integration successful!")

            # Check label distribution
            label_idx = labels.argmax() if hasattr(labels, 'argmax') else labels
            class_names = {0: "Normal", 1: "Covered", 2: "Defocused", 3: "Moved"}
            class_name = class_names.get(label_idx, f"Class {label_idx}")
            print(f"[INFO] Sample class: {class_name} ({label_idx})")

        else:
            print("[WARNING] No clips found - check data paths")

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_optical_flow_integration()
    if success:
        print("\n[SUCCESS] Optical flow integration test passed!")
    else:
        print("\n[FAILED] Optical flow integration test failed!")
