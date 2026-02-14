"""
Skater Pose Extractor - DWPose via rtmlib
==========================================
Extracts 133-keypoint whole-body skeleton from video frames.
Outputs:
  - OpenPose-compatible RGB skeleton images (for ControlNet)
  - Per-frame JSON keypoint data (for advanced ComfyUI workflows)

VRAM footprint: ~1-1.5 GB via ONNX Runtime GPU
"""

import os
import gc
import json

import cv2
import numpy as np


class SkaterPoseExtractor:
    """
    Wrapper for DWPose (rtmlib + ONNX Runtime) that extracts
    whole-body 133-keypoint skeletons from video.
    """

    def __init__(self, device="cuda", backend="onnxruntime", mode="performance"):
        """
        Initialize DWPose model.

        Args:
            device:  'cuda' or 'cpu'
            backend: 'onnxruntime' (recommended) or 'opencv'
            mode:    'performance' (RTMW-x, best quality)
                     'balanced'    (RTMW-l)
                     'lightweight' (RTMW-m, fastest)
        """
        from rtmlib import Wholebody

        print(f"[DWPose] Initializing (backend={backend}, mode={mode}) ...")
        self.pose_model = Wholebody(
            to_openpose=True,
            mode=mode,
            backend=backend,
            device=device,
        )
        print("[DWPose] Model ready.")

    def process_video(self, video_path, output_dir_img, output_dir_json=None):
        """
        Process a video and save pose visualizations + optional JSON.

        Args:
            video_path:      Path to source MP4.
            output_dir_img:  Directory for skeleton PNG images.
            output_dir_json: Directory for per-frame JSON (optional).

        Returns:
            Number of frames processed.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        os.makedirs(output_dir_img, exist_ok=True)
        if output_dir_json:
            os.makedirs(output_dir_json, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        print(f"[DWPose] Processing {total_frames} frames ...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Inference: keypoints (N, 133, 2), scores (N, 133)
            keypoints, scores = self.pose_model(frame)

            # --- Render skeleton on black canvas ---
            height, width = frame.shape[:2]
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            pose_img = self._draw_pose(canvas, keypoints, scores)

            img_filename = f"frame_{frame_idx:05d}.png"
            cv2.imwrite(os.path.join(output_dir_img, img_filename), pose_img)

            # --- Save JSON (optional) ---
            if output_dir_json:
                json_data = self._format_json(keypoints, scores)
                json_filename = f"frame_{frame_idx:05d}.json"
                with open(os.path.join(output_dir_json, json_filename), "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  ... processed {frame_idx}/{total_frames}")

        cap.release()
        print(f"[DWPose] Done. {frame_idx} pose frames saved to {output_dir_img}")
        return frame_idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_pose(canvas, keypoints, scores, kpt_thr=0.3):
        """
        Draw OpenPose-compatible skeleton on canvas.
        Uses rtmlib's draw_skeleton with correct color mapping.
        """
        from rtmlib import draw_skeleton

        return draw_skeleton(
            canvas,
            keypoints,
            scores,
            kpt_thr=kpt_thr,
        )

    @staticmethod
    def _format_json(keypoints, scores):
        """
        Format keypoints into the OpenPose JSON structure.

        Output structure:
        {
          "version": 1.3,
          "people": [
            {
              "pose_keypoints_2d": [x1, y1, c1, x2, y2, c2, ...],
              "face_keypoints_2d": [...],
              "hand_left_keypoints_2d": [...],
              "hand_right_keypoints_2d": [...]
            }
          ]
        }
        """
        people_list = []

        for kp, score in zip(keypoints, scores):
            n_points = len(kp)

            # Build flat [x, y, confidence, ...] array
            flat_all = []
            for (x, y), s in zip(kp, score):
                flat_all.extend([float(x), float(y), float(s)])

            # Split into body / face / hands per COCO-WholeBody standard
            # Body: 0-16 (17 points), Feet: 17-22 (6 points)
            # Face: 23-90 (68 points), Left Hand: 91-111 (21 points)
            # Right Hand: 112-132 (21 points)
            person_dict = {
                "pose_keypoints_2d": flat_all[:23 * 3],       # body + feet
                "face_keypoints_2d": flat_all[23 * 3:91 * 3],  # face
                "hand_left_keypoints_2d": flat_all[91 * 3:112 * 3],
                "hand_right_keypoints_2d": flat_all[112 * 3:133 * 3],
            }

            # Fallback for models with fewer points
            if n_points < 133:
                person_dict = {"pose_keypoints_2d": flat_all}

            people_list.append(person_dict)

        return {"version": 1.3, "people": people_list}

    def cleanup(self):
        """Release model memory."""
        self.pose_model = None
        gc.collect()
        # ONNX Runtime GPU memory is released when the session is deleted
        print("[DWPose] Model released.")
