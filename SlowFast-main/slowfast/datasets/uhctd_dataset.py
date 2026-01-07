#!/usr/bin/env python3

import os
import random
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torchvision.io as io
from slowfast.utils.env import pathmgr

from . import decoder as decoder, utils as utils, video_container as container
from .build import DATASET_REGISTRY

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Uhctd(torch.utils.data.Dataset):
    """
    UHCTD (University of Houston Camera Tampering Detection) Dataset loader.
    Constructs the UHCTD video loader, sampling clips focused on tampering events.

    UHCTD has long video files (24+ hours) with frame-level tampering annotations:
    frame, tamper, quantity, rate, status
    where tamper = 0 (normal), 1 (covered), 2 (defocussed), 3 (moved)
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the UHCTD video loader.

        Args:
            cfg (CfgNode): configs.
            mode (string): Options include `train`, `val`, or `test` mode.
            num_retries (int): number of retries for failed video decodes.
        """
        # Only support train, val, and test mode.
        assert mode in ["train", "val", "test"], f"Split '{mode}' not supported for UHCTD"
        self.mode = mode
        self.cfg = cfg
        self._num_retries = num_retries
        self.skip_rows = getattr(cfg.DATA, 'SKIP_ROWS', 0)

        # Video processing parameters
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES

        # Augmentation params for train/val
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP

        if self.mode == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE

        logger.info("Constructing UHCTD {}...".format(mode))
        self._construct_loader()
        self.print_summary()

    def _construct_loader(self):
        """
        Construct the video loader from UHCTD dataset structure.
        """
        data_path = self.cfg.DATA.PATH_TO_DATA_DIR
        video_prefix = self.cfg.DATA.PATH_PREFIX
        self._is_4_class = self.cfg.MODEL.NUM_CLASSES == 4  # Determine if using 4-class mode

        self._video_clips = []  # List of (video_path, start_frame, end_frame, label) tuples

        # Walk through all camera directories
        for camera_dir in os.listdir(data_path):
            camera_path = os.path.join(data_path, camera_dir)

            if not os.path.isdir(camera_path):
                continue

            logger.info(f"Processing camera: {camera_dir}")

            # Process training video folders
            training_video_path = os.path.join(camera_path, "Training video")
            if not os.path.exists(training_video_path):
                logger.warning(f"Training video path not found: {training_video_path}")
                continue

            for day_dir in os.listdir(training_video_path):
                day_path = os.path.join(training_video_path, day_dir)

                if not os.path.isdir(day_path):
                    continue

                # Check for video and annotations
                video_file = os.path.join(day_path, "video.avi")
                annotations_file = os.path.join(day_path, "annotations.csv")

                if not os.path.exists(video_file) or not os.path.exists(annotations_file):
                    logger.warning(f"Missing video or annotations in {day_path}")
                    continue

                # Load annotations with proper headers
                column_names = ['frame', 'tamper', 'quantity', 'rate', 'status']
                annotations_df = pd.read_csv(annotations_file, header=0, names=column_names)

                # Process annotations to create video clips
                self._process_annotations(video_file, annotations_df)

        # Limit clips for debug mode
        if hasattr(self.cfg, 'DEBUG_MODE') and self.cfg.DEBUG_MODE:
            max_clips = getattr(self.cfg, 'DEBUG_MAX_CLIPS', 10)
            if len(self._video_clips) > max_clips:
                self._video_clips = self._video_clips[:max_clips]
                logger.info(f"DEBUG: Limited to {max_clips} clips for debug mode")

        logger.info(f"Total clips found: {len(self._video_clips)}")

    def _process_annotations(self, video_path, annotations_df):
        """
        Process annotations to create balanced clips for 4-class or binary classification.

        Args:
            video_path (str): Path to the video file
            annotations_df (pd.DataFrame): Annotations dataframe
        """
        clip_length = self._seq_len

        # Get tampering segments by tampering type (for 4-class)
        if self._is_4_class:
            tampering_segments_by_type = self._find_tampering_segments_by_type(annotations_df)
            normal_segments = self._find_normal_segments(annotations_df, min_segment_length=150)
        else:
            # Legacy binary mode
            annotations_df['tamper_binary'] = (annotations_df['tamper'] > 0).astype(int)
            tampering_segments_by_type = {1: self._find_tampering_segments(annotations_df)}
            normal_segments = self._find_normal_segments(annotations_df, min_segment_length=150)

        # Sample clips from tampering segments
        max_clips_per_type = 50  # Limit to prevent too many clips
        for tamper_type, segments in tampering_segments_by_type.items():
            clips_added = 0
            for segment in segments:
                if clips_added >= max_clips_per_type:
                    break

                start_frame, end_frame = segment
                segment_duration = end_frame - start_frame + 1

                if segment_duration >= clip_length:
                    # Sample clips from this segment
                    num_clips = min(5, segment_duration // clip_length)
                    for i in range(num_clips):
                        if clips_added >= max_clips_per_type:
                            break
                        clip_start = start_frame + i * (segment_duration - clip_length) // max(1, num_clips - 1)
                        clip_end = clip_start + clip_length - 1
                        self._video_clips.append((video_path, clip_start, clip_end, tamper_type))
                        clips_added += 1

        # Sample normal clips (class 0)
        normal_clips_added = 0
        target_normals = len(self._video_clips) // 4  # Balance with ~25% normal clips
        for segment in normal_segments:
            if normal_clips_added >= target_normals:
                break

            start_frame, end_frame = segment
            segment_duration = end_frame - start_frame + 1

            if segment_duration >= clip_length:
                num_clips = min(3, segment_duration // clip_length)
                for i in range(num_clips):
                    if normal_clips_added >= target_normals:
                        break
                    clip_start = start_frame + i * (segment_duration - clip_length) // max(1, num_clips - 1)
                    clip_end = clip_start + clip_length - 1
                    self._video_clips.append((video_path, clip_start, clip_end, 0))  # Normal class
                    normal_clips_added += 1

    def _find_tampering_segments_by_type(self, annotations_df, min_segment_length=30):
        """
        Find consecutive tampering segments grouped by tampering type.

        Args:
            annotations_df (pd.DataFrame): Annotations dataframe
            min_segment_length (int): Minimum segment length in frames

        Returns:
            Dict: {tamper_type: [(start_frame, end_frame), ...], ...}
        """
        segments_by_type = {1: [], 2: [], 3: []}  # 1=covered, 2=defocused, 3=moved
        tamper_values = annotations_df['tamper'].values

        i = 0
        while i < len(tamper_values):
            tamper_type = tamper_values[i]
            if tamper_type > 0:  # Tampering event
                start = i
                # Continue while same tamper type or continue until normal (0)
                while i < len(tamper_values) and tamper_values[i] > 0:
                    i += 1
                end = i - 1

                if end - start >= min_segment_length:
                    segments_by_type[tamper_type].append((start, end))
            else:
                i += 1

        return segments_by_type

    def _find_tampering_segments(self, annotations_df, min_segment_length=30):
        """
        Find consecutive tampering segments.

        Args:
            annotations_df (pd.DataFrame): Annotations dataframe
            min_segment_length (int): Minimum segment length in frames

        Returns:
            List of (start_frame, end_frame) tuples for tampering segments
        """
        segments = []
        tamper_binary = annotations_df['tamper_binary'].values

        i = 0
        while i < len(tamper_binary):
            if tamper_binary[i] == 1:
                start = i
                while i < len(tamper_binary) and tamper_binary[i] == 1:
                    i += 1
                end = i - 1

                if end - start >= min_segment_length:
                    segments.append((start, end))
            else:
                i += 1

        return segments

    def _find_normal_segments(self, annotations_df, min_segment_length=150):
        """
        Find consecutive normal segments.

        Args:
            annotations_df (pd.DataFrame): Annotations dataframe
            min_segment_length (int): Minimum segment length in frames

        Returns:
            List of (start_frame, end_frame) tuples for normal segments
        """
        segments = []

        # Handle both binary and 4-class mode
        if self._is_4_class:
            tamper_values = annotations_df['tamper'].values
        else:
            tamper_values = annotations_df['tamper_binary'].values

        i = 0
        while i < len(tamper_values):
            if tamper_values[i] == 0:  # Normal (no tampering)
                start = i
                while i < len(tamper_values) and tamper_values[i] == 0:
                    i += 1
                end = i - 1

                if end - start >= min_segment_length:
                    segments.append((start, end))
            else:
                i += 1

        return segments

    def print_summary(self):
        """
        Print dataset summary.
        """
        logger.info("=== UHCTD dataset summary ===")
        logger.info("Split: {}".format(self.mode))
        logger.info(f"Mode: {'4-class' if self._is_4_class else 'Binary'} classification")
        logger.info("Number of clips: {}".format(len(self._video_clips)))

        # Count labels
        labels = [clip[3] for clip in self._video_clips]
        unique_labels, counts = np.unique(labels, return_counts=True)

        class_names = {
            0: "Normal",
            1: "Covered",
            2: "Defocused",
            3: "Moved"
        }

        for label, count in zip(unique_labels, counts):
            if self._is_4_class:
                class_name = class_names.get(label, f"Class {label}")
            else:
                class_name = "Normal" if label == 0 else "Tampering"
            logger.info(f"  {class_name} clips: {count}")

        # Calculate class distribution percentage
        total = len(self._video_clips)
        for label, count in zip(unique_labels, counts):
            percentage = (count / total) * 100
            logger.info(f"    ({percentage:.1f}% of total)")

    def __len__(self):
        """
        Returns the number of clips in the dataset.
        """
        return len(self._video_clips)

    def __getitem__(self, idx):
        """
        Get a video clip and its label.

        Args:
            idx (int): clip index

        Returns:
            frames: Processed video frames [C, T, H, W]
            label: Label tensor
            idx: Clip index
            time_idx: Time indices (empty for UHCTD)
            extra_data: Additional data dict
        """
        video_path, start_frame, end_frame, label = self._video_clips[idx]

        # Retry loading video up to num_retries times
        for i_try in range(self._num_retries):
            try:
                video_container = container.get_video_container(
                    video_path,
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )

                if video_container is None:
                    raise ValueError("Video container is None")

                # Sample frames from the specific clip
                frames = self._sample_frames_from_clip(
                    video_path, start_frame, end_frame
                )

                if frames is None:
                    raise ValueError("Failed to sample frames")

                # Process frames
                frames = self._preprocess_frames(frames)

                # Create label array for multi-class (4 classes: normal, covered, defocussed, moved)
                # For now, we'll use binary classification (normal vs tampering)
                label_arr = np.zeros(self._num_classes, dtype=np.int32)
                label_arr[label] = 1  # label is 0 (normal) or 1 (tampering)

                frames = utils.pack_pathway_output(self.cfg, frames)

                return frames, label_arr, idx, np.zeros(1), {"video_path": video_path}

            except Exception as e:
                logger.warning(
                    f"Failed to load clip {idx} from {video_path}: {e}, trial {i_try}"
                )
                if i_try < self._num_retries - 1:
                    continue
                else:
                    # Return a random valid clip as fallback
                    return self.__getitem__(random.randint(0, len(self._video_clips) - 1))

    def _sample_frames_from_clip(self, video_path, start_frame, end_frame):
        """
        Sample a fixed number of frames from a clip segment.

        Args:
            video_path (str): Path to the video file
            start_frame, end_frame: Frame range

        Returns:
            frames: Sampled frames [T, H, W, C]
        """
        try:
            # Determine FPS based on camera
            if "Camera A" in video_path:
                fps = 3.0  # Camera A: 3 fps
            elif "Camera B" in video_path:
                fps = 10.0  # Camera B: 10 fps
            else:
                fps = 30.0  # Default fallback

            # Calculate start/end time in seconds
            start_pts = start_frame / fps
            end_pts = (end_frame + 1) / fps

            # Decode frames from the video path
            v_frames, _, _ = io.read_video(
                video_path, 
                start_pts=start_pts, 
                end_pts=end_pts, 
                pts_unit="sec"
            )

            if v_frames is None or v_frames.shape[0] < self._video_length:
                logger.warning(f"Failed to read sufficient frames: got {v_frames.shape[0] if v_frames is not None else 0}, need {self._video_length}")
                return None

            # Subsample to exactly 32 frames uniformly from the decoded range
            indices = torch.linspace(0, v_frames.shape[0] - 1, self._video_length, dtype=torch.long)
            frames = v_frames[indices]

            return frames

        except Exception as e:
            logger.warning(f"Error decoding frames: {e}")
            return None

    def _preprocess_frames(self, frames):
        """
        Preprocess frames for SlowFast: resize, crop, normalize.

        Args:
            frames: Input frames tensor [T, H, W, C]
        """
        # Convert to float and normalize to [0, 1]
        frames = frames.float() / 255.0

        # Permute from [T, H, W, C] to [T, C, H, W]
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]

        if self.mode == "train":
            # Training augmentations
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=-1,
                min_scale=self._jitter_min_scale,
                max_scale=self._jitter_max_scale,
                crop_size=self._crop_size,
                random_horizontal_flip=self.random_horizontal_flip,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )
        else:
            # Validation/Test: center crop
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=1,  # Center crop
                min_scale=self._crop_size,
                max_scale=self._crop_size,
                crop_size=self._crop_size,
                random_horizontal_flip=False,
                inverse_uniform_sampling=False,
            )

        # Normalize with mean/std
        mean = torch.tensor(self._data_mean, dtype=frames.dtype, device=frames.device).view(1, -1, 1, 1)
        std = torch.tensor(self._data_std, dtype=frames.dtype, device=frames.device).view(1, -1, 1, 1)
        frames = (frames - mean) / std

        frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]

        return frames

    @property
    def num_videos(self):
        """
        Returns the number of video clips in the dataset.
        """
        return len(self._video_clips)
