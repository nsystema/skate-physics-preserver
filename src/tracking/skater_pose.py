"""
SkaterPoseExtractor: DWPose (rtmlib) wrapper for human pose estimation.
Outputs OpenPose-compatible skeleton images and JSON for ComfyUI ControlNet.
"""

import os
import cv2
import json
import numpy as np
from rtmlib import Wholebody, draw_skeleton


class SkaterPoseExtractor:
    """
    Wrapper for DWPose (via rtmlib) to extract and render OpenPose-compatible skeletons.
    Uses 'lightweight' mode for Colab T4 VRAM; 'performance' for local RTX.
    """

    def __init__(self, device="cuda", backend="onnxruntime", mode="lightweight"):
        """
        Args:
            device: 'cuda' or 'cpu'
            backend: 'onnxruntime' for DWPose
            mode: 'lightweight' (RTMW-l, T4-friendly) or 'performance' (RTMW-x)
        """
        print(f"Initializing DWPose (Backend: {backend}, Mode: {mode})...")
        self.pose_model = Wholebody(
            to_openpose=True,
            mode=mode,
            backend=backend,
            device=device,
        )

    def process_video(self, video_path, output_dir_img, output_dir_json=None, frame_load_cap=None):
        """
        Process the video to generate pose visualizations and optional JSON.

        Args:
            video_path: Path to input video
            output_dir_img: Directory for pose PNG images
            output_dir_json: Optional directory for pose JSON per frame
            frame_load_cap: Max frames to process (for T4). None = all.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        os.makedirs(output_dir_img, exist_ok=True)
        if output_dir_json:
            os.makedirs(output_dir_json, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_load_cap is not None:
            total_frames = min(total_frames, frame_load_cap)

        print(f"Processing {total_frames} frames...")

        while cap.isOpened() and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, scores = self.pose_model(frame)

            height, width, _ = frame.shape
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            pose_img = draw_skeleton(canvas, keypoints, scores, kpt_thr=0.3, to_openpose=True)

            img_filename = f"frame_{frame_idx:05d}.png"
            cv2.imwrite(os.path.join(output_dir_img, img_filename), pose_img)

            if output_dir_json:
                json_data = self._format_json(keypoints, scores)
                json_filename = f"frame_{frame_idx:05d}.json"
                with open(os.path.join(output_dir_json, json_filename), "w") as f:
                    json.dump(json_data, f)

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  Processed frame {frame_idx}/{total_frames}")

        cap.release()
        print("Pose processing complete.")

    def _format_json(self, keypoints, scores):
        """Format keypoints into OpenPose JSON structure."""
        people_list = []
        for kp, score in zip(keypoints, scores):
            pose_data = []
            for (x, y), s in zip(kp, score):
                pose_data.extend([float(x), float(y), float(s)])
            person_dict = {"pose_keypoints_2d": pose_data}
            people_list.append(person_dict)
        return {"version": 1.3, "people": people_list}
