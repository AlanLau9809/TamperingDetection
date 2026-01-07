#!/usr/bin/env python3
"""
UHCTD Model Evaluation Script
Evaluates trained SlowFast model on testing videos against ground truth annotations.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision.io as io
from torchvision.io import VideoReader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Add SlowFast to path
sys.path.append('SlowFast-main')

from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.utils.checkpoint import load_checkpoint
from slowfast.utils.misc import launch_job
from slowfast.datasets.utils import pack_pathway_output, spatial_sampling, tensor_normalize

def setup_model(model_path="SlowFast-main/checkpoint/best_model.pth", config_path="SlowFast-main/configs/UHCTD/SLOWFAST_UHCTD.yaml"):
    """Load trained SlowFast model"""
    print("Loading trained SlowFast model...")

    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.freeze()

    # Build and load model
    model = build_model(cfg)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from checkpoint: {model_path}")
            if 'val_accuracy' in checkpoint:
                print(f"   Best validation accuracy during training: {checkpoint['val_accuracy']:.2f}%")
        else:
            print("Warning: No model_state_dict found in checkpoint, loading directly")
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model ready for evaluation on device: {device}")
    return model, cfg, device

def preprocess_frames(frames, cfg):
    """Preprocess frames for SlowFast (consistent with training)"""
    # Input frames are [T, C, H, W] from VideoReader, uint8
    frames = frames.float() / 255.0

    # spatial_sampling expects [T, C, H, W] and returns [T, C, H, W]
    frames = spatial_sampling(
        frames,
        spatial_idx=1,  # Center crop
        min_scale=cfg.DATA.TEST_CROP_SIZE,
        max_scale=cfg.DATA.TEST_CROP_SIZE,
        crop_size=cfg.DATA.TEST_CROP_SIZE,
        random_horizontal_flip=False,
    )

    # Normalize manually like in the dataset class
    mean = torch.tensor(cfg.DATA.MEAN, dtype=frames.dtype).view(1, -1, 1, 1)
    std = torch.tensor(cfg.DATA.STD, dtype=frames.dtype).view(1, -1, 1, 1)
    frames = (frames - mean) / std

    # Permute to [C, T, H, W] for the model
    frames = frames.permute(1, 0, 2, 3)

    # Pack for SlowFast (returns a list of tensors)
    frames = pack_pathway_output(cfg, frames)

    return frames

def load_ground_truth(gt_path, is_4_class=False):
    """Load ground truth annotations"""
    df = pd.read_csv(gt_path, header=None, names=['frame', 'tamper', 'quantity', 'rate', 'status'])

    if is_4_class:
        # Keep all 4 classes: 0=normal, 1=covered, 2=defocused, 3=moved
        df['label'] = df['tamper']
    else:
        # Convert to binary: any tampering (1,2,3) = 1, normal (0) = 0
        df['label'] = (df['tamper'] > 0).astype(int)

    return df

def predict_video_segment(model, video_path, start_frame, end_frame, cfg, device):
    """Predict tampering class for a video segment using a memory-efficient method."""
    try:
        if "Camera A" in video_path:
            fps = 3.0
        elif "Camera B" in video_path:
            fps = 10.0
        else:
            fps = 30.0

        start_pts = float(start_frame) / fps
        num_frames_to_read = end_frame - start_frame + 1

        frames_list = []
        try:
            reader = VideoReader(video_path, "video")
        except Exception as e:
            print(f"Critical error opening video file {video_path} with VideoReader: {e}")
            if cfg.MODEL.NUM_CLASSES == 4:
                return 0, np.array([1.0, 0.0, 0.0, 0.0])
            else:
                return 0, 0.0

        try:
            reader.seek(start_pts)
            for _ in range(num_frames_to_read):
                frame = next(reader)
                frames_list.append(frame['data'])
            v_frames = torch.stack(frames_list)
        except StopIteration:
            if not frames_list:
                print(f"No frames loaded for segment {start_frame}-{end_frame} (EOF at seek)")
                if cfg.MODEL.NUM_CLASSES == 4: return 0, np.array([1.0, 0.0, 0.0, 0.0])
                else: return 0, 0.0
            v_frames = torch.stack(frames_list)
        except Exception as load_error:
            print(f"Failed to load segment {start_frame}-{end_frame} at {start_pts:.1f}s: {load_error}")
            if cfg.MODEL.NUM_CLASSES == 4:
                return 0, np.array([1.0, 0.0, 0.0, 0.0])
            else:
                return 0, 0.0
        
        # v_frames from VideoReader is [T, C, H, W]
        num_loaded_frames = v_frames.shape[0]
        target_frames = cfg.DATA.NUM_FRAMES

        if num_loaded_frames < target_frames:
            if num_loaded_frames > 0:
                last_frame = v_frames[-1:].repeat(target_frames - num_loaded_frames, 1, 1, 1)
                v_frames = torch.cat([v_frames, last_frame], dim=0)
            else:
                # This case is handled by the error checking above, but as a fallback:
                return 0, np.array([1.0, 0.0, 0.0, 0.0]) if cfg.MODEL.NUM_CLASSES == 4 else 0, 0.0
        elif num_loaded_frames > target_frames:
            indices = torch.linspace(0, num_loaded_frames - 1, target_frames, dtype=torch.long)
            v_frames = v_frames[indices]
        
        frames = preprocess_frames(v_frames, cfg)

        with torch.no_grad():
            frames = [f.unsqueeze(0).to(device) for f in frames]
            outputs = model(frames)

            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            num_classes = cfg.MODEL.NUM_CLASSES
            if num_classes == 4:
                return preds.item(), probs[0].cpu().numpy()
            else:
                return preds.item(), probs[0][1].item()

    except Exception as e:
        print(f"Error processing segment {start_frame}-{end_frame}: {str(e)[:100]}...")
        if cfg.MODEL.NUM_CLASSES == 4:
            return 0, np.array([1.0, 0.0, 0.0, 0.0])
        else:
            return 0, 0.0

def evaluate_video(model, video_path, gt_df, cfg, device, window_size=32, stride=16, max_samples=1000):
    """Evaluate model on video using sliding window with sampling for memory efficiency"""
    print(f"Evaluating video: {os.path.basename(video_path)}")
    print(f"   Using {max_samples} sample windows for evaluation (memory efficient)")

    total_frames = len(gt_df)
    predictions = np.zeros(total_frames)
    probabilities = np.zeros((total_frames, cfg.MODEL.NUM_CLASSES))  # Store probabilities for all classes

    is_4_class = cfg.MODEL.NUM_CLASSES == 4

    # Sample windows for evaluation instead of processing all frames
    sampled_windows = []

    y_labels = gt_df['label'].values

    if is_4_class:
        # For 4-class, sample from all class types
        for class_id in [0, 1, 2, 3]:  # Normal, Covered, Defocused, Moved
            class_frames = np.where(y_labels == class_id)[0]
            if len(class_frames) > max_samples // 4:
                class_sample = np.random.choice(class_frames, max_samples // 4, replace=False)
                for frame_idx in class_sample:
                    start_frame = max(0, frame_idx - window_size // 2)
                    end_frame = min(total_frames - 1, start_frame + window_size - 1)
                    sampled_windows.append((start_frame, end_frame))
    else:
        # Binary case - sample normal and tampering
        normal_frames = np.where(y_labels == 0)[0]
        tampering_frames = np.where(y_labels == 1)[0]

        # Sample normal frames
        if len(normal_frames) > max_samples // 2:
            normal_sample = np.random.choice(normal_frames, max_samples // 2, replace=False)
            for frame_idx in normal_sample:
                start_frame = max(0, frame_idx - window_size // 2)
                end_frame = min(total_frames - 1, start_frame + window_size - 1)
                sampled_windows.append((start_frame, end_frame))

        # Sample tampering frames
        if len(tampering_frames) > max_samples // 2:
            tampering_sample = np.random.choice(tampering_frames, max_samples // 2, replace=False)
            for frame_idx in tampering_sample:
                start_frame = max(0, frame_idx - window_size // 2)
                end_frame = min(total_frames - 1, start_frame + window_size - 1)
                sampled_windows.append((start_frame, end_frame))

    # Evaluate sampled windows
    print(f"   Processing {len(sampled_windows)} sampled windows...")
    for start_frame, end_frame in tqdm(sampled_windows, desc="Evaluating samples"):

        # Get model prediction for this window
        pred_class, prob_values = predict_video_segment(
            model, video_path, start_frame, end_frame, cfg, device
        )

        if is_4_class:
            # For 4-class, assign prediction and all probabilities
            predictions[start_frame:end_frame+1] = pred_class
            probabilities[start_frame:end_frame+1] = prob_values  # Store all class probabilities
        else:
            # Binary case
            predictions[start_frame:end_frame+1] = pred_class
            probabilities[start_frame:end_frame+1, 1] = prob_values  # Tampering probability

    return predictions, probabilities

def temporal_smoothing(predictions, probabilities, cfg, window_size=5):
    """Apply temporal smoothing to reduce false positives"""
    is_4_class = cfg.MODEL.NUM_CLASSES == 4

    if is_4_class:
        # For 4-class, smooth the probability arrays and take argmax
        smoothed_probs = np.zeros_like(probabilities)
        for i in range(len(probabilities)):
            start = max(0, i - window_size // 2)
            end = min(len(probabilities), i + window_size // 2 + 1)
            smoothed_probs[i] = np.mean(probabilities[start:end], axis=0)

        # Take the class with highest smoothed probability
        smoothed_predictions = np.argmax(smoothed_probs, axis=1)
        return smoothed_predictions, smoothed_probs
    else:
        # Binary smoothing - keep original logic
        smoothed = np.copy(predictions)
        for i in range(len(predictions)):
            start = max(0, i - window_size // 2)
            end = min(len(predictions), i + window_size // 2 + 1)
            smoothed[i] = np.mean(predictions[start:end])

        # Convert back to binary
        return (smoothed > 0.5).astype(int), None

def calculate_metrics(y_true, y_pred, cfg, video_name="Video"):
    """Calculate comprehensive evaluation metrics for 4-class or binary"""
    is_4_class = cfg.MODEL.NUM_CLASSES == 4

    accuracy = accuracy_score(y_true, y_pred)

    if is_4_class:
        # 4-class: macro-average metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

        # Per-class metrics for all 4 classes
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

        # Confusion matrix for 4 classes
        cm = confusion_matrix(y_true, y_pred)

        class_names = {0: "Normal", 1: "Covered", 2: "Defocused", 3: "Moved"}

        print(f"\n{video_name} Results (4-Class):")
        print(f"   Overall Accuracy: {accuracy:.3f}")
        print(f"   Macro Precision: {precision:.3f}, Macro Recall: {recall:.3f}, Macro F1: {f1:.3f}")

        for i in range(4):
            if i < len(precision_per_class):
                class_name = class_names.get(i, f"Class {i}")
                print(f"   {class_name} ({i}): P={precision_per_class[i]:.3f}, R={recall_per_class[i]:.3f}, F1={f1_per_class[i]:.3f}")

        print(f"   Confusion Matrix (True ↓ | Predicted → ):\n{cm}")
        print("   Classes: [0=Normal, 1=Covered, 2=Defocused, 3=Moved]")
    else:
        # Binary: traditional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n{video_name} Results (Binary):")
        print(f"   Overall Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}")
        print(f"   Normal Class (0): Precision={precision_per_class[0]:.3f}, Recall={recall_per_class[0]:.3f}, F1={f1_per_class[0]:.3f}")
        if len(precision_per_class) > 1:
            print(f"   Tampering Class (1): Precision={precision_per_class[1]:.3f}, Recall={recall_per_class[1]:.3f}, F1={f1_per_class[1]:.3f}")
        print(f"   Confusion Matrix (True ↓ | Predicted → ):\n{cm}")
        print("   Classes: [0=Normal, 1=Tampering]")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }

def main():
    print("UHCTD Model Evaluation Script")
    print("=" * 50)

    # Define the output directory for evaluation results
    output_results_folder = "Evaluation Results"
    os.makedirs(output_results_folder, exist_ok=True)

    # Setup model
    try:
        model, cfg, device = setup_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Test videos and ground truth paths
    test_base = "/mnt/d/FYP/UHCTD/UHCTD Comprehensive Dataset For Camera Tampering Detection/Camera A/Testing video"
    gt_base = "/mnt/d/FYP/groundtruth_and_prediction/Ground_truth/cam_a"

    # Extract camera identifier (e.g., "cam_a") from gt_base
    cam_name = os.path.basename(gt_base)

    test_days = ["Day 3", "Day 4", "Day 5", "Day 6"]
    all_results = {}

    print("\nStarting evaluation on test videos...")

    for day in test_days:
        print(f"\n{'='*50}")
        print(f"Evaluating {day}")

        # Paths
        video_dir = os.path.join(test_base, day)
        video_path = os.path.join(video_dir, "video.avi")
        gt_path = os.path.join(gt_base, f"{day}.csv")

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        if not os.path.exists(gt_path):
            print(f"Ground truth not found: {gt_path}")
            continue

        # Load ground truth
        is_4_class = cfg.MODEL.NUM_CLASSES == 4
        gt_df = load_ground_truth(gt_path, is_4_class)
        y_true = gt_df['label'].values

        if is_4_class:
            # For 4-class, show distribution of all classes
            unique_labels, counts = np.unique(y_true, return_counts=True)
            class_names = {0: "Normal", 1: "Covered", 2: "Defocused", 3: "Moved"}
            print(f"Ground truth loaded: {len(y_true)} frames")
            for label, count in zip(unique_labels, counts):
                class_name = class_names.get(label, f"Class {label}")
                percentage = (count / len(y_true)) * 100
                print(f"   {class_name}: {count} frames ({percentage:.1f}%)")
        else:
            # Binary case
            print(f"Ground truth loaded: {len(y_true)} frames")
            print(f"   Normal frames: {np.sum(y_true == 0)}, Tampering frames: {np.sum(y_true == 1)}")

        # Evaluate video
        predictions, probabilities = evaluate_video(model, video_path, gt_df, cfg, device)

        # Apply temporal smoothing
        if is_4_class:
            smoothed_predictions, smoothed_probabilities = temporal_smoothing(predictions, probabilities, cfg, window_size=5)
        else:
            smoothed_predictions, _ = temporal_smoothing(predictions, probabilities, cfg, window_size=5)

        # Calculate metrics for both raw and smoothed predictions
        results = {}
        results['raw'] = calculate_metrics(y_true, predictions, cfg, f"{day} (Raw)")
        results['smoothed'] = calculate_metrics(y_true, smoothed_predictions, cfg, f"{day} (Smoothed)")

        all_results[day] = results

        # Save predictions for analysis
        output_dict = {
            'frame': gt_df['frame'],
            'true_label': y_true,
            'tamper_type': gt_df['tamper'],
            'raw_prediction': predictions,
            'smoothed_prediction': smoothed_predictions
        }

        if is_4_class:
            # For 4-class, add individual probability columns
            output_dict['normal_prob'] = probabilities[:, 0]
            output_dict['covered_prob'] = probabilities[:, 1]
            output_dict['defocused_prob'] = probabilities[:, 2]
            output_dict['moved_prob'] = probabilities[:, 3]
        else:
            # For binary, add tampering probability
            output_dict['tampering_probability'] = probabilities[:, 1]

        output_df = pd.DataFrame(output_dict)
        output_path = os.path.join(output_results_folder, f"evaluation_results_{cam_name}_{day}.csv")
        output_df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL EVALUATION SUMMARY")
    print(f"{'='*60}")

    for day, results in all_results.items():
        raw_acc = results['raw']['accuracy']
        smooth_acc = results['smoothed']['accuracy']
        print(f"{day}: Raw={raw_acc:.3f}, Smoothed={smooth_acc:.3f}")

    print("\nEvaluation complete! Check the generated CSV files for detailed analysis.")

if __name__ == "__main__":
    main()
