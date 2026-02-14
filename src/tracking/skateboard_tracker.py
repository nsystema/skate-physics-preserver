"""
Skateboard Tracker - SAM 2.1 Video Predictor Wrapper
=====================================================
Optimized for RTX 3070 8GB VRAM via:
  - Hiera-Small backbone (not Large)
  - offload_video_to_cpu=True  (frame embeddings on CPU RAM)
  - offload_state_to_cpu=True  (tracking state on CPU RAM)
  - float16 autocast (Ampere-native)
  - Aggressive VRAM cleanup between runs
"""

import os
import gc
import tempfile
import shutil

import cv2
import numpy as np
import torch


class SkateboardTracker:
    """
    Wrapper around SAM 2.1 Video Predictor for skateboard tracking.
    Manages inference state, mask propagation, and VRAM lifecycle.
    """

    # Default model config shipped with the sam2 package
    DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

    def __init__(self, checkpoint_path, config_path=None, device="cuda"):
        """
        Initialize SAM 2.1 Hiera-Small model.

        Args:
            checkpoint_path: Path to sam2.1_hiera_small.pt
            config_path:     SAM2 config YAML (None = package default for small)
            device:          'cuda' or 'cpu'
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {checkpoint_path}\n"
                "Download from: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
            )

        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path or self.DEFAULT_CONFIG

        self.predictor = None
        self.inference_state = None
        self.video_info = {}
        self._frames_dir = None  # temp dir for extracted frames

        self._build_predictor()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_predictor(self):
        """Build the SAM2 video predictor with 8GB-safe settings."""
        from sam2.build_sam import build_sam2_video_predictor

        print(f"[SAM2] Initializing Hiera-Small on {self.device} ...")

        # float16 autocast for Ampere (RTX 3070 = compute 8.6)
        if self.device == "cuda" and torch.cuda.is_available():
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self.predictor = build_sam2_video_predictor(
            self.config_path,
            self.checkpoint_path,
            device=self.device,
        )

    def _extract_frames(self, video_path):
        """
        Extract video frames to a temp directory of JPEG files.
        SAM2 video predictor is most reliable with frame directories.
        Returns the path to the temp directory.
        """
        tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Stash video metadata
        self.video_info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            filename = os.path.join(tmp_dir, f"{idx:05d}.jpg")
            cv2.imwrite(filename, frame)
            idx += 1
        cap.release()

        # Update count in case metadata was wrong
        self.video_info["count"] = idx
        print(f"[SAM2] Extracted {idx} frames ({self.video_info['width']}x{self.video_info['height']} @ {self.video_info['fps']:.1f}fps)")
        return tmp_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_video(self, video_path):
        """
        Load a video into the SAM 2.1 memory bank.

        Extracts frames to a temp directory, then initializes the
        inference state with CPU offloading to fit in 8 GB VRAM.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Extract frames to temp dir
        self._frames_dir = self._extract_frames(video_path)

        print("[SAM2] Loading frames into memory bank (CPU-offloaded) ...")
        self.inference_state = self.predictor.init_state(
            video_path=self._frames_dir,
            offload_video_to_cpu=True,     # keep frame embeddings on CPU
            offload_state_to_cpu=True,     # keep tracking state on CPU
        )

    def add_initial_prompt(self, frame_idx, bbox, obj_id=1):
        """
        Seed the tracker with a bounding box on a specific frame.

        Args:
            frame_idx: 0-indexed frame number
            bbox:      [x1, y1, x2, y2] coordinates of the object
            obj_id:    Integer object ID (1=skateboard, 2=skater, etc.)
        """
        if self.inference_state is None:
            raise RuntimeError("Call init_video() before adding prompts.")

        print(f"[SAM2] Adding box prompt on frame {frame_idx} (obj {obj_id}): {bbox}")
        _, _out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=np.array(bbox, dtype=np.float32),
        )
        return out_mask_logits

    def propagate_and_save(self, output_dir):
        """
        Propagate the mask through the entire video and save as PNG sequence.

        Output: Grayscale PNGs (0 = background, 255 = object) named
                frame_00000.png, frame_00001.png, ...

        Args:
            output_dir: Directory to write mask PNGs.
        """
        if self.inference_state is None:
            raise RuntimeError("Call init_video() and add_initial_prompt() first.")

        os.makedirs(output_dir, exist_ok=True)
        h, w = self.video_info["height"], self.video_info["width"]
        total = self.video_info["count"]
        saved = 0

        print(f"[SAM2] Propagating masks across {total} frames ...")

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state
        ):
            # Blank canvas (black = no object)
            final_mask = np.zeros((h, w), dtype=np.uint8)

            if 1 in out_obj_ids:
                idx = list(out_obj_ids).index(1)
                mask_logit = out_mask_logits[idx]
                mask_bool = (mask_logit > 0.0).cpu().numpy().squeeze()

                if mask_bool.ndim == 2:
                    # Resize if SAM output size differs from original video
                    if mask_bool.shape != (h, w):
                        mask_uint8 = (mask_bool.astype(np.uint8)) * 255
                        mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
                        final_mask = mask_uint8
                    else:
                        final_mask[mask_bool] = 255

            filename = f"frame_{out_frame_idx:05d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), final_mask)
            saved += 1

            if saved % 50 == 0:
                print(f"  ... saved {saved}/{total} masks")

        print(f"[SAM2] Done. {saved} masks saved to {output_dir}")
        return saved

    def propagate_and_save_multi(self, output_dirs, progress_callback=None):
        """
        Propagate masks for multiple tracked objects to separate directories.

        This is the multi-object counterpart to ``propagate_and_save``.
        Each object ID maps to its own output directory.

        Args:
            output_dirs:       dict mapping obj_id (int) to output dir path
                               e.g. {1: "output/mask_skateboard", 2: "output/mask_skater"}
            progress_callback: optional callable(fraction: float) where fraction
                               is in [0, 1].  Called after each frame.

        Returns:
            dict mapping obj_id to number of frames saved.
        """
        if self.inference_state is None:
            raise RuntimeError("Call init_video() and add_initial_prompt() first.")

        h, w = self.video_info["height"], self.video_info["width"]
        total = self.video_info["count"]

        for d in output_dirs.values():
            os.makedirs(d, exist_ok=True)

        counts: dict[int, int] = {oid: 0 for oid in output_dirs}
        frame_num = 0

        print(f"[SAM2] Propagating {len(output_dirs)} object(s) across {total} frames ...")

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state
        ):
            obj_id_list = [int(o) for o in out_obj_ids]

            for i, oid in enumerate(obj_id_list):
                if oid not in output_dirs:
                    continue

                mask_logit = out_mask_logits[i]
                mask_bool = (mask_logit > 0.0).cpu().numpy().squeeze()

                final_mask = np.zeros((h, w), dtype=np.uint8)
                if mask_bool.ndim == 2:
                    if mask_bool.shape != (h, w):
                        mask_uint8 = mask_bool.astype(np.uint8) * 255
                        final_mask = cv2.resize(mask_uint8, (w, h),
                                                interpolation=cv2.INTER_NEAREST)
                    else:
                        final_mask[mask_bool] = 255

                filename = f"frame_{out_frame_idx:05d}.png"
                cv2.imwrite(os.path.join(output_dirs[oid], filename), final_mask)
                counts[oid] += 1

            frame_num += 1
            if progress_callback and total > 0:
                progress_callback(frame_num / total)
            if frame_num % 50 == 0:
                print(f"  ... processed {frame_num}/{total} frames")

        for oid, cnt in counts.items():
            print(f"[SAM2] Object {oid}: {cnt} masks -> {output_dirs[oid]}")

        return counts

    def propagate_yield(self):
        """
        Generator that yields (frame_idx, binary_mask_np) tuples.
        Used by evaluate_iou.py for frame-by-frame comparison without
        writing to disk.
        """
        if self.inference_state is None:
            raise RuntimeError("Call init_video() and add_initial_prompt() first.")

        h, w = self.video_info["height"], self.video_info["width"]

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state
        ):
            final_mask = np.zeros((h, w), dtype=np.uint8)

            if 1 in out_obj_ids:
                idx = list(out_obj_ids).index(1)
                mask_logit = out_mask_logits[idx]
                mask_bool = (mask_logit > 0.0).cpu().numpy().squeeze()

                if mask_bool.ndim == 2:
                    if mask_bool.shape != (h, w):
                        mask_uint8 = (mask_bool.astype(np.uint8)) * 255
                        final_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        final_mask[mask_bool] = 255

            yield out_frame_idx, final_mask

    def reset(self):
        """Clear tracking state (keeps model loaded)."""
        if self.predictor is not None and self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
        self.inference_state = None

    def cleanup(self):
        """
        Release all GPU memory. Call between pipeline stages
        (e.g., after tracking, before generation).
        """
        self.reset()
        if self._frames_dir and os.path.isdir(self._frames_dir):
            shutil.rmtree(self._frames_dir, ignore_errors=True)
            self._frames_dir = None
        self.predictor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[SAM2] GPU memory released.")
