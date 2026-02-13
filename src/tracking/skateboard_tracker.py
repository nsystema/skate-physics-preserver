"""
SkateboardTracker: SAM 2.1 Video Predictor wrapper for object tracking.
Optimized for Colab T4 with Hiera-Small and frame caps.
"""

import os
import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor


class SkateboardTracker:
    """
    A wrapper around SAM 2.1 Video Predictor tailored for skateboard tracking.
    Manages the inference state and mask propagation.
    Supports frame_load_cap for VRAM-constrained environments (e.g., Colab T4).
    """

    def __init__(self, checkpoint_path, config_path, device="cuda", frame_load_cap=None):
        """
        Initialize the SAM 2.1 model.

        Args:
            checkpoint_path (str): Path to sam2.1_hiera_s.pt (Small) or sam2.1_hiera_large.pt
            config_path (str): Path to sam2.1_hiera_s.yaml or sam2.1_hiera_l.yaml
            device (str): Compute device ('cuda' or 'cpu').
            frame_load_cap (int, optional): Max frames to process (for T4 VRAM). None = all frames.
        """
        print(f"Initializing SAM 2.1 on {device}...")
        self.device = device
        self.frame_load_cap = frame_load_cap
        if self.device == "cuda":
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True

        self.predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
        self.inference_state = None
        self.video_info = {}

    def init_video(self, video_path):
        """
        Load a video into the SAM 2.1 memory bank.
        This step preprocesses the video and extracts static embeddings.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"Loading video into memory: {video_path}")
        self.inference_state = self.predictor.init_state(video_path=video_path)

        cap = cv2.VideoCapture(video_path)
        total_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.frame_load_cap is not None:
            total_count = min(total_count, self.frame_load_cap)
        self.video_info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "count": total_count,
        }
        cap.release()

    def add_initial_prompt(self, frame_idx, bbox):
        """
        Seed the tracking with a bounding box on a specific frame.

        Args:
            frame_idx (int): The frame number (0-indexed).
            bbox (list): [x1, y1, x2, y2] coordinates of the skateboard.
        """
        print(f"Adding prompt to frame {frame_idx}: {bbox}")

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            box=np.array(bbox, dtype=np.float32),
        )
        return out_mask_logits

    def propagate_and_save(self, output_dir):
        """
        Propagate the mask through the video and save as PNG sequence.

        Args:
            output_dir (str): Directory to save the masks.
        """
        os.makedirs(output_dir, exist_ok=True)
        print("Propagating masks...")

        frame_count = 0
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state
        ):
            if self.frame_load_cap is not None and frame_count >= self.frame_load_cap:
                break

            final_mask = np.zeros(
                (self.video_info["height"], self.video_info["width"]), dtype=np.uint8
            )

            if 1 in out_obj_ids:
                idx = list(out_obj_ids).index(1)
                mask_logit = out_mask_logits[idx]
                mask_bool = (mask_logit > 0.0).cpu().numpy().squeeze()
                if mask_bool.ndim == 2:
                    final_mask[mask_bool] = 255

            filename = f"frame_{out_frame_idx:05d}.png"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, final_mask)

            if out_frame_idx % 50 == 0:
                print(f"  Processed frame {out_frame_idx}/{self.video_info['count']}")

            frame_count += 1

        print(f"Completed. Masks saved to {output_dir}")

    def propagate_yield(self):
        """
        Generator that yields mask logits frame-by-frame (for evaluate_iou).
        Yields numpy arrays (H, W) as binary masks (0/255 uint8).
        """
        frame_count = 0
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state
        ):
            if self.frame_load_cap is not None and frame_count >= self.frame_load_cap:
                break

            final_mask = np.zeros(
                (self.video_info["height"], self.video_info["width"]), dtype=np.uint8
            )

            if 1 in out_obj_ids:
                idx = list(out_obj_ids).index(1)
                mask_logit = out_mask_logits[idx]
                mask_bool = (mask_logit > 0.0).cpu().numpy().squeeze()
                if mask_bool.ndim == 2:
                    final_mask[mask_bool] = 255

            yield final_mask
            frame_count += 1
