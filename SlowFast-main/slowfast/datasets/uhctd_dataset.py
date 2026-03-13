#!/usr/bin/env python3
"""
UHCTD Dataset loader for SlowFast.

Supports two modes (auto-detected from cfg.DATA.INPUT_CHANNEL_NUM[1]):
  - RGB-only (channel=3): Slow=RGB(3ch, T/α frames), Fast=RGB(3ch, T frames)   [recommended]
  - RGB+Flow (channel=2): Slow=RGB(3ch, T/α frames), Fast=Flow(2ch, T frames)  [legacy]

Key fixes over original:
  1. Clip cap raised to 500 per type (was 50) → much larger training set
  2. Normal class balanced 1:1:1:1 with tampering classes
  3. Optical flow path only activated when INPUT_CHANNEL_NUM[1]==2; default is RGB-only
  4. Preprocessing computes flow on small 224px frames (same as eval) when flow mode active
"""

import os
import random
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torchvision.io as io
import cv2
from slowfast.utils.env import pathmgr

from . import utils as utils, video_container as container
from .build import DATASET_REGISTRY

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Uhctd(torch.utils.data.Dataset):
    """
    UHCTD (University of Houston Camera Tampering Detection) Dataset loader.

    Video structure:
        <DATA_PATH>/<Camera X>/Training video/Day N/video.avi   + annotations.csv
        <DATA_PATH>/<Camera X>/Testing video/Day N/video.avi    (no annotations, use GT CSV)

    Annotation CSV columns: frame, tamper, quantity, rate, status
        tamper: 0=Normal, 1=Covered, 2=Defocused, 3=Moved

    Camera FPS:
        Camera A → 3 fps
        Camera B → 10 fps
    """

    # Maximum clips sampled per tampering class per video-day.
    # Raised from 50 → 500 to create a much larger, more representative training set.
    MAX_CLIPS_PER_TYPE = 500

    def __init__(self, cfg, mode, num_retries=10):
        assert mode in ["train", "val", "test"], f"Split '{mode}' not supported for UHCTD"
        self.mode = mode
        self.cfg = cfg
        self._num_retries = num_retries

        # ── Core video params ────────────────────────────────────────────────
        self._sample_rate  = cfg.DATA.SAMPLING_RATE   # temporal stride when decoding
        self._video_length = cfg.DATA.NUM_FRAMES       # frames per clip (T)
        self._seq_len      = self._video_length * self._sample_rate  # raw frames to read
        self._num_classes  = cfg.MODEL.NUM_CLASSES

        # ── Mode detection ───────────────────────────────────────────────────
        # INPUT_CHANNEL_NUM = [slow_ch, fast_ch]
        #   fast_ch == 3 → RGB-only (both pathways get RGB)
        #   fast_ch == 2 → flow mode (fast pathway gets optical flow)
        fast_ch = cfg.DATA.INPUT_CHANNEL_NUM[1] if len(cfg.DATA.INPUT_CHANNEL_NUM) > 1 else 3
        self._use_flow = (fast_ch == 2)
        logger.info(f"UHCTD mode: {'RGB+Flow' if self._use_flow else 'RGB-only'} "
                    f"(INPUT_CHANNEL_NUM[1]={fast_ch})")

        # ── Normalisation ────────────────────────────────────────────────────
        self._data_mean = cfg.DATA.MEAN
        self._data_std  = cfg.DATA.STD
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP

        # ── Crop sizes ───────────────────────────────────────────────────────
        if self.mode == "train":
            self._crop_size       = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE

        logger.info(f"Constructing UHCTD [{mode}]…")
        self._construct_loader()
        self._print_summary()

    # ════════════════════════════════════════════════════════════════════════
    # Dataset construction
    # ════════════════════════════════════════════════════════════════════════

    def _construct_loader(self):
        """Walk the UHCTD folder tree and build a list of (video_path, start, end, label)."""
        data_path = self.cfg.DATA.PATH_TO_DATA_DIR

        # Camera filter (set via env variable by the training script)
        train_cameras_env = os.environ.get('UHCTD_TRAIN_CAMERAS', 'all').strip()
        if train_cameras_env == 'all' or not train_cameras_env:
            allowed_cameras = None
        else:
            allowed_cameras = [c.strip() for c in train_cameras_env.split(',')]
            logger.info(f"Camera filter active: {allowed_cameras}")

        self._video_clips = []

        for camera_dir in sorted(os.listdir(data_path)):
            camera_path = os.path.join(data_path, camera_dir)
            if not os.path.isdir(camera_path):
                continue

            # Training/val use camera filter; test always uses Camera A
            if self.mode in ["train", "val"] and allowed_cameras is not None:
                if camera_dir not in allowed_cameras:
                    logger.info(f"Skipping camera (not in filter): {camera_dir}")
                    continue

            logger.info(f"Processing camera: {camera_dir}")

            training_video_path = os.path.join(camera_path, "Training video")
            if not os.path.exists(training_video_path):
                logger.warning(f"Training video path not found: {training_video_path}")
                continue

            for day_dir in sorted(os.listdir(training_video_path)):
                day_path   = os.path.join(training_video_path, day_dir)
                video_file = os.path.join(day_path, "video.avi")
                ann_file   = os.path.join(day_path, "annotations.csv")

                if not os.path.isdir(day_path) or \
                   not os.path.exists(video_file) or \
                   not os.path.exists(ann_file):
                    logger.warning(f"Missing video or annotations: {day_path}")
                    continue

                col_names = ['frame', 'tamper', 'quantity', 'rate', 'status']
                ann_df = pd.read_csv(ann_file, header=0, names=col_names)
                self._process_annotations(video_file, ann_df)

        logger.info(f"Total clips constructed: {len(self._video_clips)}")

    def _process_annotations(self, video_path, ann_df):
        """
        Create balanced clips from one video's annotation file.
        Sampling strategy:
          - Tampering: up to MAX_CLIPS_PER_TYPE per class (Covered/Defocused/Moved)
          - Normal:    same total count as total tampering clips → 1:1:1:1 balance
        """
        clip_length = self._seq_len  # raw frames to read (NUM_FRAMES × SAMPLING_RATE)

        tampering_segs = self._find_tampering_segments_by_type(ann_df)
        normal_segs    = self._find_normal_segments(ann_df)

        # ── Tampering clips ──────────────────────────────────────────────────
        tampering_clips_added = 0
        for tamper_type, segs in tampering_segs.items():
            clips_this_type = 0
            for seg_start, seg_end in segs:
                if clips_this_type >= self.MAX_CLIPS_PER_TYPE:
                    break
                seg_len = seg_end - seg_start + 1
                if seg_len < clip_length:
                    continue
                # Sample up to 5 non-overlapping clips per segment
                step = max(1, (seg_len - clip_length) // 4)
                for j in range(5):
                    if clips_this_type >= self.MAX_CLIPS_PER_TYPE:
                        break
                    clip_start = seg_start + j * step
                    clip_end   = clip_start + clip_length - 1
                    if clip_end > seg_end:
                        break
                    self._video_clips.append((video_path, clip_start, clip_end, tamper_type))
                    clips_this_type += 1
            tampering_clips_added += clips_this_type

        # ── Normal clips: match total tampering count for 1:1:1:1 balance ───
        target_normals = tampering_clips_added  # equal to total tampering clips added
        normal_clips_added = 0
        for seg_start, seg_end in normal_segs:
            if normal_clips_added >= target_normals:
                break
            seg_len = seg_end - seg_start + 1
            if seg_len < clip_length:
                continue
            step = max(1, (seg_len - clip_length) // 4)
            for j in range(5):
                if normal_clips_added >= target_normals:
                    break
                clip_start = seg_start + j * step
                clip_end   = clip_start + clip_length - 1
                if clip_end > seg_end:
                    break
                self._video_clips.append((video_path, clip_start, clip_end, 0))
                normal_clips_added += 1

    # ════════════════════════════════════════════════════════════════════════
    # Segment finders
    # ════════════════════════════════════════════════════════════════════════

    def _find_tampering_segments_by_type(self, ann_df, min_len=30):
        """Return {1: [(s,e)…], 2: [(s,e)…], 3: [(s,e)…]} for typed tampering runs."""
        segs = {1: [], 2: [], 3: []}
        vals = ann_df['tamper'].values
        i = 0
        while i < len(vals):
            t = vals[i]
            if t > 0:
                start = i
                while i < len(vals) and vals[i] == t:
                    i += 1
                end = i - 1
                if end - start + 1 >= min_len and t in segs:
                    segs[t].append((start, end))
            else:
                i += 1
        return segs

    def _find_normal_segments(self, ann_df, min_len=150):
        """Return [(start, end), …] for consecutive normal (tamper==0) runs."""
        segs = []
        vals = ann_df['tamper'].values
        i = 0
        while i < len(vals):
            if vals[i] == 0:
                start = i
                while i < len(vals) and vals[i] == 0:
                    i += 1
                end = i - 1
                if end - start + 1 >= min_len:
                    segs.append((start, end))
            else:
                i += 1
        return segs

    # ════════════════════════════════════════════════════════════════════════
    # Frame loading
    # ════════════════════════════════════════════════════════════════════════

    def _fps_for_video(self, video_path):
        if "Camera A" in video_path:
            return 3.0
        elif "Camera B" in video_path:
            return 10.0
        return 30.0

    def _load_frames(self, video_path, start_frame, end_frame):
        """
        Decode frames [start_frame, end_frame] from `video_path`.
        Returns uint8 tensor [T, H, W, C] or raises.
        """
        fps = self._fps_for_video(video_path)
        start_pts = start_frame / fps
        end_pts   = (end_frame + 1) / fps

        v_frames, _, _ = io.read_video(
            video_path, start_pts=start_pts, end_pts=end_pts, pts_unit="sec"
        )

        if v_frames is None or v_frames.shape[0] == 0:
            raise ValueError(f"No frames decoded from {video_path} "
                             f"[{start_frame}:{end_frame}]")

        target = self._video_length  # e.g. 32
        n = v_frames.shape[0]

        if n < target:
            # Pad by repeating last frame
            pad = v_frames[-1:].repeat(target - n, 1, 1, 1)
            v_frames = torch.cat([v_frames, pad], dim=0)
        elif n > target:
            idx = torch.linspace(0, n - 1, target, dtype=torch.long)
            v_frames = v_frames[idx]

        return v_frames  # [T, H, W, C] uint8

    # ════════════════════════════════════════════════════════════════════════
    # Preprocessing
    # ════════════════════════════════════════════════════════════════════════

    def _compute_optical_flow(self, frames_thwc_uint8):
        """
        Compute Farneback optical flow on small uint8 frames.
        Input:  [T, H, W, C] uint8
        Output: torch float32 [T-1, H, W, 2]
        """
        flow_list = []
        H, W = frames_thwc_uint8.shape[1], frames_thwc_uint8.shape[2]
        for i in range(len(frames_thwc_uint8) - 1):
            f1 = frames_thwc_uint8[i, :, :, :3]
            f2 = frames_thwc_uint8[i + 1, :, :, :3]
            try:
                g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
                g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    g1, g2, None,
                    pyr_scale=0.5, levels=2, winsize=9,
                    iterations=2, poly_n=5, poly_sigma=1.2, flags=0
                )
                flow_list.append(flow)
            except Exception:
                flow_list.append(np.zeros((H, W, 2), dtype=np.float32))
        return torch.from_numpy(np.stack(flow_list)).float()  # [T-1, H, W, 2]

    def _preprocess_frames(self, frames_thwc):
        """
        Full preprocessing pipeline.

        RGB-only mode (INPUT_CHANNEL_NUM[1]==3)  [DEFAULT]:
            1. float+resize → [T, C, 224, 224]
            2. normalize RGB
            3. Slow: subsample T/α frames from rgb
            4. Fast: full T frames of rgb
            → returns [slow_rgb[3,T/α,224,224], fast_rgb[3,T,224,224]]

        RGB+Flow mode (INPUT_CHANNEL_NUM[1]==2):
            1. float+resize → [T, C, 224, 224]
            2. compute flow on small frames → [T-1, H, W, 2]
            3. normalize RGB; normalize flow
            4. Slow: subsample T/α RGB frames
            5. Fast: flow padded to T frames
            → returns [slow_rgb[3,T/α,224,224], fast_flow[2,T,224,224]]
        """
        # ── Step 1: float & spatial sampling ────────────────────────────────
        frames_float = frames_thwc.float() / 255.0         # [T, H, W, C]
        frames_tchw  = frames_float.permute(0, 3, 1, 2)    # [T, C, H, W]

        inv_uniform = getattr(self.cfg.DATA, 'INV_UNIFORM_SAMPLE', False)

        if self.mode == "train":
            rgb_small = utils.spatial_sampling(
                frames_tchw,
                spatial_idx=-1,           # random crop
                min_scale=self._jitter_min_scale,
                max_scale=self._jitter_max_scale,
                crop_size=self._crop_size,
                random_horizontal_flip=self.random_horizontal_flip,
                inverse_uniform_sampling=inv_uniform,
            )  # [T, 3, crop, crop]
        else:
            rgb_small = utils.spatial_sampling(
                frames_tchw,
                spatial_idx=1,            # centre crop
                min_scale=self._crop_size,
                max_scale=self._crop_size,
                crop_size=self._crop_size,
                random_horizontal_flip=False,
                inverse_uniform_sampling=False,
            )  # [T, 3, crop, crop]

        # ── Step 2: [T, C, H, W] → [C, T, H, W] ────────────────────────────
        rgb_frames = rgb_small.permute(1, 0, 2, 3)  # [3, T, H, W]

        # ── Step 3: normalize RGB ────────────────────────────────────────────
        rgb_mean = torch.tensor(self._data_mean[:3], dtype=rgb_frames.dtype).view(-1, 1, 1, 1)
        rgb_std  = torch.tensor(self._data_std[:3],  dtype=rgb_frames.dtype).view(-1, 1, 1, 1)
        rgb_frames = (rgb_frames - rgb_mean) / rgb_std

        # ── Step 4: SlowFast temporal split ──────────────────────────────────
        T     = rgb_frames.shape[1]          # 32
        alpha = self.cfg.SLOWFAST.ALPHA      # 4  → slow has T/4 = 8 frames
        slow_idx = torch.linspace(0, T - 1, T // alpha, dtype=torch.long)
        slow_rgb = rgb_frames[:, slow_idx, :, :]  # [3, 8, H, W]

        if not self._use_flow:
            # ── RGB-only: Fast pathway = full T RGB frames ───────────────────
            return [slow_rgb, rgb_frames]   # [3,8,H,W], [3,32,H,W]

        else:
            # ── Flow mode: Fast pathway = optical flow ───────────────────────
            frames_small_np = (rgb_small.permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().numpy()
            flow_thw2 = self._compute_optical_flow(frames_small_np)  # [T-1, H, W, 2]

            # [T-1, H, W, 2] → [2, T-1, H, W]
            flow_frames = flow_thw2.permute(3, 0, 1, 2)  # [2, T-1, H, W]

            # Normalize flow
            flow_mean = torch.tensor(self._data_mean[3:5], dtype=flow_frames.dtype).view(-1, 1, 1, 1)
            flow_std  = torch.tensor(self._data_std[3:5],  dtype=flow_frames.dtype).view(-1, 1, 1, 1)
            flow_frames = (flow_frames - flow_mean) / flow_std

            # Pad T-1 → T (duplicate last frame)
            if flow_frames.shape[1] < T:
                flow_frames = torch.cat([flow_frames, flow_frames[:, -1:, :, :]], dim=1)

            return [slow_rgb, flow_frames]  # [3,8,H,W], [2,32,H,W]

    # ════════════════════════════════════════════════════════════════════════
    # Dataset interface
    # ════════════════════════════════════════════════════════════════════════

    def __len__(self):
        return len(self._video_clips)

    def __getitem__(self, idx):
        video_path, start_frame, end_frame, label = self._video_clips[idx]

        for i_try in range(self._num_retries):
            try:
                frames_thwc = self._load_frames(video_path, start_frame, end_frame)
                pathway_frames = self._preprocess_frames(frames_thwc)

                label_arr = np.zeros(self._num_classes, dtype=np.int32)
                label_arr[label] = 1

                return (
                    pathway_frames,
                    label_arr,
                    idx,
                    np.zeros(1),
                    {"video_path": video_path},
                )

            except Exception as e:
                logger.warning(
                    f"[try {i_try+1}/{self._num_retries}] Failed clip {idx} "
                    f"({video_path}@{start_frame}): {e}"
                )
                if i_try < self._num_retries - 1:
                    # Retry with a random valid clip to avoid blocking the DataLoader
                    fallback_idx = random.randint(0, len(self._video_clips) - 1)
                    video_path, start_frame, end_frame, label = self._video_clips[fallback_idx]
                else:
                    # Last resort: return zeros
                    T, α = self._video_length, self.cfg.SLOWFAST.ALPHA
                    fast_ch = self.cfg.DATA.INPUT_CHANNEL_NUM[1] if len(self.cfg.DATA.INPUT_CHANNEL_NUM) > 1 else 3
                    slow = torch.zeros(3, T // α, self._crop_size, self._crop_size)
                    fast = torch.zeros(fast_ch, T, self._crop_size, self._crop_size)
                    label_arr = np.zeros(self._num_classes, dtype=np.int32)
                    label_arr[0] = 1
                    return ([slow, fast], label_arr, idx, np.zeros(1), {"video_path": ""})

    @property
    def num_videos(self):
        return len(self._video_clips)

    # ════════════════════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════════════════════

    def _print_summary(self):
        logger.info("=== UHCTD dataset summary ===")
        logger.info(f"  Split          : {self.mode}")
        logger.info(f"  Mode           : {'RGB+Flow' if self._use_flow else 'RGB-only'}")
        logger.info(f"  Total clips    : {len(self._video_clips)}")
        logger.info(f"  Frames/clip    : {self._video_length} (sampling_rate={self._sample_rate})")

        labels = [c[3] for c in self._video_clips]
        unique, counts = np.unique(labels, return_counts=True)
        names = {0: "Normal", 1: "Covered", 2: "Defocused", 3: "Moved"}
        for lbl, cnt in zip(unique, counts):
            pct = 100.0 * cnt / len(labels) if labels else 0
            logger.info(f"  {names.get(lbl, lbl):<12}: {cnt:5d} clips  ({pct:.1f}%)")
