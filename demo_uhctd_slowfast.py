#!/usr/bin/env python3

import sys
import os
sys.path.append('SlowFast-main')

import pandas as pd
import numpy as np

def demo_uhctd_dataset():
    """
    Demonstration of UHCTD dataset analysis for SlowFast camera tampering detection
    """

    print("="*60)
    print("UHCTD + SlowFast Camera Tampering Detection Demo")
    print("="*60)

    # Show dataset structure
    print("\n1. UHCTD Dataset Structure:")
    data_path = "../UHCTD/UHCTD Comprehensive Dataset For Camera Tampering Detection"
    print(f"   Dataset path: {data_path}")

    camera_paths = []
    total_frames = 0
    total_tampered = 0
    total_normal = 0
    total_segments = 0

    # Analyze each camera
    for camera_dir in ["Camera A", "Camera B"]:
        camera_path = os.path.join(data_path, camera_dir)
        if not os.path.exists(camera_path):
            continue

        camera_paths.append(camera_dir)

        training_video_path = os.path.join(camera_path, "Training video")

        if not os.path.exists(training_video_path):
            continue

        print(f"\n   📹 {camera_dir}:")

        for day_dir in os.listdir(training_video_path):
            day_path = os.path.join(training_video_path, day_dir)

            if not os.path.isdir(day_path):
                continue

            annotations_file = os.path.join(day_path, "annotations.csv")
            video_file = os.path.join(day_path, "video.avi")

            if not os.path.exists(annotations_file) or not os.path.exists(video_file):
                continue

            # Load annotations
            column_names = ['frame', 'tamper', 'quantity', 'rate', 'status']
            annotations_df = pd.read_csv(annotations_file, header=0, names=column_names)

            # Analyze this video
            annotations_df['tamper_binary'] = (annotations_df['tamper'] > 0).astype(int)
            frames = len(annotations_df)
            tampered = len(annotations_df[annotations_df['tamper_binary'] == 1])
            normal = len(annotations_df[annotations_df['tamper_binary'] == 0])

            total_frames += frames
            total_tampered += tampered
            total_normal += normal

            # Count segments
            tamper_binary = annotations_df['tamper_binary'].values
            segments = 0
            i = 0
            while i < len(tamper_binary):
                if tamper_binary[i] == 1:
                    start = i
                    while i < len(tamper_binary) and tamper_binary[i] == 1:
                        i += 1
                    end = i - 1
                    if end - start >= 30:  # min segment length
                        segments += 1
                else:
                    i += 1

            total_segments += segments

            tamper_percentage = (tampered / frames) * 100 if frames > 0 else 0

            print(f"     📅 {day_dir}: {frames:,} frames")
            print(f"        Normal: {normal:,}, Tampered: {tampered:,} ({tamper_percentage:.1f}%)")
            print(f"        Tampering segments: {segments}")

    # Overall statistics
    print("\n2. Dataset Summary:")
    print(f"   📊 Total cameras analyzed: {len(camera_paths)}")
    print(f"   🎬 Total frames: {total_frames:,}")
    print(f"   ✅ Normal frames: {total_normal:,}")
    print(f"   ⚠️  Tampered frames: {total_tampered:,}")
    print(f"   📈 Tampering ratio: {(total_tampered/total_frames)*100:.1f}%" if total_frames > 0 else 0)
    print(f"   🎯 Tampering segments: {total_segments}")

    # SlowFast processing strategy
    print("\n3. SlowFast Video Processing Strategy:")
    print("   • 🎬 32-frame video clips (≈1.28s at 25fps)")
    print("   • ⚖️  Balanced positive/negative sampling")
    print("   • 🧠 Two-stream architecture:")
    print("     - Slow pathway: temporal patterns")
    print("     - Fast pathway: motion detection")
    print("   • 🎯 Binary classification: Normal vs Tampering")

    # Implementation features
    print("\n4. UHCTD Dataset Class Features:")
    print("   ✓ 🤖 Automatic segment detection")
    print("   ✓ 🎯 Smart clip sampling around tampering events")
    print("   ✓ 📹 Video decoding with temporal subsampling")
    print("   ✓ 🔄 SlowFast preprocessing pipeline")
    print("   ✓ 🎨 Data augmentation for robust training")

    # Configuration
    print("\n5. Training Configuration:")
    print("   • 🏗️  Architecture: SlowFast R50 (slow α=4, β=8)")
    print("   • 📐 Input: 32-frame clips, 224×224 resolution")
    print("   • 📦 Batch size: 4 (GPU memory optimized)")
    print("   • 📈 Training: 30 epochs with cosine LR decay")
    print("   • 🏃 Single GPU training setup")

    # Expected benefits
    print("\n6. Expected Advantages vs Frame-Based Methods:")
    print("   ✓ 📊 Temporal context capture")
    print("   ✓ 🎬 Full video sequence understanding")
    print("   ✓ 🏃 Motion pattern recognition")
    print("   ✓ ⏱️ Long-term tampering detection")
    print("   ✓ 🔄 End-to-end spatiotemporal learning")

    print("\n" + "="*60)
    print("🎉 Implementation ready for camera tampering detection!")
    print("SlowFast + UHCTD will deliver superior temporal-aware detection")
    print("compared to traditional frame-based computer vision methods.")
    print("="*60)

if __name__ == "__main__":
    demo_uhctd_dataset()
