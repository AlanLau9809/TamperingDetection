#!/usr/bin/env python3
"""
UHCTD Dataset loader for SlowFast.

Supports three modes (selected via UHCTD_FAST_MODE environment variable):
  - 'rgb'     : Slow=RGB(3ch, T/α frames), Fast=RGB(3ch, T frames)    [default]
  - 'refdiff' : Slow=RGB(3ch, T/α frames), Fast=|frame-ref|(3ch, T)   [recommended]
  - 'flow'    : Slow=RGB(3ch, T/α frames), Fast=Flow(2ch, T frames)   [legacy]

'refdiff' mode computes |Frame_t − Reference_Frame| for every fast-pathway frame,
where the reference frame is the mean of the first ~100 Normal frames of each video.
This gives the fast pathway a persistent spatial-change signal that easily distinguishes
Moved/Covered/Defocused even when the tampering is fully settled (zero optical flow).

Key design:
  1. Clip cap raised to 500 per type (was 50) → much larger training set
  2. Normal class balanced 1:1:1:1 with tampering classes
  3. Reference frames pre-computed once at dataset construction, cached by video_path
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
        # Priority: UHCTD_FAST_MODE env var overrides INPUT_CHANNEL_NUM detection.
        #   'rgb'     → both pathways get RGB (INPUT_CHANNEL_NUM=[3,3])
        #   'refdiff' → fast pathway gets |frame_t - reference| (INPUT_CHANNEL_NUM=[3,3])
        #   'flow'    → fast pathway gets optical flow (INPUT_CHANNEL_NUM=[3,2])
        env_mode = os.environ.get('UHCTD_FAST_MODE', '').strip().lower()
        fast_ch  = cfg.DATA.INPUT_CHANNEL_NUM[1] if len(cfg.DATA.INPUT_CHANNEL_NUM) > 1 else 3
        if env_mode in ('rgb', 'refdiff', 'flow'):
            self._fast_mode = env_mode
        elif fast_ch == 2:
            self._fast_mode = 'flow'
        else:
            self._fast_mode = 'rgb'

        self._use_flow    = (self._fast_mode == 'flow')
        self._use_refdiff = (self._fast_mode == 'refdiff')
        self._use_gray    = (self._fast_mode == 'gray')
        logger.info(f"UHCTD fast_mode={self._fast_mode}  "
                    f"(INPUT_CHANNEL_NUM[1]={fast_ch}, UHCTD_FAST_MODE='{env_mode}')")

        # Reference frames cache: {video_path: float_tensor[3,H,W] in [0,1]}
        # Populated during _construct_loader when refdiff mode is active.
        self._reference_frames: dict = {}

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

                # Pre-compute multi-reference frames for refdiff mode (list of [3,H,W] tensors)
                if self._use_refdiff and video_file not in self._reference_frames:
                    refs = self._compute_reference_frames(video_file, ann_df, n_refs=5, n_frames_each=50)
                    self._reference_frames[video_file] = refs  # list of [3, H, W] tensors

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
    # Reference frame computation (refdiff mode)
    # ════════════════════════════════════════════════════════════════════════

    def _compute_reference_frames(self, video_path, ann_df, n_refs=10, n_frames_each=30):
        """
        Compute n_refs reference frames spread evenly across Normal frames in the video.

        Each reference is the temporal mean of n_frames_each consecutive Normal frames
        at that anchor position. Returns a list of float32 [3, H, W] tensors in [0,1].

        Using multiple references that cover different times of day allows the model
        to learn to recognise that "min(diff) near zero" = Normal, while
        "min(diff) large" = tampered — even across full day/night lighting cycles.
        Falls back to a single grey frame if loading fails entirely.
        """
        fps = self._fps_for_video(video_path)
        vals = ann_df['tamper'].values
        normal_indices = np.where(vals == 0)[0]

        if len(normal_indices) == 0:
            logger.warning(f"No Normal frames in {video_path}. Using grey reference.")
            return [torch.full((3, 224, 224), 0.5, dtype=torch.float32)]

        # Evenly spaced anchor positions within the Normal frames
        anchor_positions = np.linspace(0, len(normal_indices) - 1, n_refs, dtype=int)
        anchor_frames    = [int(normal_indices[p]) for p in anchor_positions]

        refs = []
        half = n_frames_each // 2
        for anchor in anchor_frames:
            start = max(0, anchor - half)
            start_pts = float(start) / fps
            end_pts   = float(start + n_frames_each) / fps
            try:
                v_frames, _, _ = io.read_video(
                    video_path, start_pts=start_pts, end_pts=end_pts, pts_unit="sec"
                )
                if v_frames is None or v_frames.shape[0] == 0:
                    continue
                ref_hwc = v_frames.float().mean(dim=0) / 255.0   # [H, W, 3]
                ref_chw = ref_hwc.permute(2, 0, 1)               # [3, H, W]
                refs.append(ref_chw)
            except Exception as e:
                logger.warning(f"Reference load failed at frame {anchor} in {video_path}: {e}")
                continue

        if not refs:
            logger.warning(f"All reference loads failed for {video_path}. Using grey.")
            return [torch.full((3, 224, 224), 0.5, dtype=torch.float32)]

        logger.info(f"Computed {len(refs)} reference frames at Normal frames "
                    f"{anchor_frames[:len(refs)]} for {os.path.basename(os.path.dirname(video_path))}")
        return refs  # list of [3, H, W] float32 tensors

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

    def _preprocess_frames(self, frames_thwc, reference_frame=None):
        """
        Full preprocessing pipeline.

        'rgb' mode [default]:
            Slow=RGB(3ch, T/α), Fast=RGB(3ch, T)

        'refdiff' mode:
            Slow=RGB(3ch, T/α), Fast=|frame_t − reference|(3ch, T)
            reference_frame: float32 [3,H,W] in [0,1], un-normalized

        'flow' mode:
            Slow=RGB(3ch, T/α), Fast=Flow(2ch, T)
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

        if self._use_gray:
            # ── Grayscale: Convert both pathways to luminance (3-ch replicated) ─
            # Grayscale = 0.299*R + 0.587*G + 0.114*B, then replicate to 3 channels
            def _to_gray_3ch(rgb_chw):  # [3, T, H, W]
                gray = 0.299*rgb_chw[0] + 0.587*rgb_chw[1] + 0.114*rgb_chw[2]  # [T, H, W]
                return gray.unsqueeze(0).expand(3, -1, -1, -1)  # [3, T, H, W]

            slow_gray = _to_gray_3ch(slow_rgb)  # [3, T/alpha, H, W]
            fast_gray = _to_gray_3ch(rgb_frames)  # [3, T, H, W]
            return [slow_gray, fast_gray]  # both grayscale, 3-ch replicated

        elif self._use_refdiff:
            # ── RefDiff: Fast pathway = |whiten(frame_t) − whiten(reference)| ──
            # Per-frame luminance whitening removes global brightness drift
            # (day/night lighting changes), making the diff purely structural.
            # whiten(f) = f / mean_brightness(f)  →  brightness-invariant texture

            def _whiten(t_thwc):
                """Whiten [T, 3, H, W] float: divide each frame by its mean brightness."""
                mu = t_thwc.mean(dim=[2, 3], keepdim=True).clamp(min=1e-6)
                return t_thwc / mu   # [T, 3, H, W], values roughly in [0, ~3]

            rgb_whitened = _whiten(rgb_small)  # [T, 3, H, W]

            H, W = rgb_small.shape[2], rgb_small.shape[3]

            # Support both:
            #   single ref  → old single-tensor [3,H,W]  (backward compat)
            #   multi ref   → new list of [3,H,W] tensors (n_refs=5)
            if reference_frame is None:
                ref_list = [rgb_small[0]]      # first frame of clip as fallback
            elif isinstance(reference_frame, list):
                ref_list = reference_frame     # list of [3, H, W] tensors
            else:
                ref_list = [reference_frame]   # single tensor → wrap in list

            # Compute pixel-wise MINIMUM absolute diff across all references.
            # For a Normal frame: at least one reference matches the lighting
            # → min(diff) is near zero.
            # For a tampered frame: ALL references are structurally different
            # → min(diff) stays large.
            diff_maps = []
            for ref in ref_list:
                if ref.shape[1] != H or ref.shape[2] != W:
                    ref = torch.nn.functional.interpolate(
                        ref.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
                    ).squeeze(0)
                ref_exp = ref.unsqueeze(0)            # [1, 3, H, W]
                ref_wh  = _whiten(ref_exp)            # [1, 3, H, W]
                diff_maps.append(torch.abs(rgb_whitened - ref_wh))  # [T, 3, H, W]

            if len(diff_maps) == 1:
                diff_frames = diff_maps[0]                              # [T, 3, H, W]
            else:
                diff_frames = torch.stack(diff_maps, dim=0).min(dim=0).values  # [T, 3, H, W]

            # [T, 3, H, W] → [3, T, H, W]
            diff_frames = diff_frames.permute(1, 0, 2, 3)

            # Normalize: whitened diff range is ~[0,2]; use same mean/std as RGB
            diff_frames = (diff_frames - rgb_mean) / rgb_std

            return [slow_rgb, diff_frames]  # [3,T/α,H,W], [3,T,H,W]

        elif not self._use_flow:
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

        ref_frame = self._reference_frames.get(video_path, None) if self._use_refdiff else None

        for i_try in range(self._num_retries):
            try:
                frames_thwc = self._load_frames(video_path, start_frame, end_frame)
                pathway_frames = self._preprocess_frames(frames_thwc, reference_frame=ref_frame)

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
        logger.info(f"  Mode           : {self._fast_mode}")
        logger.info(f"  Total clips    : {len(self._video_clips)}")
        logger.info(f"  Frames/clip    : {self._video_length} (sampling_rate={self._sample_rate})")

        labels = [c[3] for c in self._video_clips]
        unique, counts = np.unique(labels, return_counts=True)
        names = {0: "Normal", 1: "Covered", 2: "Defocused", 3: "Moved"}
        for lbl, cnt in zip(unique, counts):
            pct = 100.0 * cnt / len(labels) if labels else 0
            logger.info(f"  {names.get(lbl, lbl):<12}: {cnt:5d} clips  ({pct:.1f}%)")
