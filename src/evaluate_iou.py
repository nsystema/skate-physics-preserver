"""
Automated Physics Validation: IoU evaluation for Zero-Clipping Benchmark.
Compares original object mask vs. generated object mask (via reverse-tracking).
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.skateboard_tracker import SkateboardTracker

IOU_THRESHOLD = 0.90


def load_mask_sequence(path):
    """
    Load mask sequence from directory of PNGs or from video file.
    Returns list of (H, W) uint8 arrays (0 or 255).
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "frame_*.png")))
        masks = []
        for f in files:
            m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                masks.append((m > 127).astype(np.uint8) * 255)
        return masks

    if os.path.isfile(path) and path.lower().endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(path)
        masks = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masks.append((gray > 127).astype(np.uint8) * 255)
        cap.release()
        return masks

    raise ValueError(f"Path must be directory of PNGs or video file: {path}")


def calculate_iou(mask_a, mask_b):
    """
    Computes IoU between two binary masks (numpy arrays).
    Masks are expected to be uint8 (0 or 255) or boolean.
    """
    flat_a = mask_a.flatten().astype(bool)
    flat_b = mask_b.flatten().astype(bool)
    intersection = np.logical_and(flat_a, flat_b).sum()
    union = np.logical_or(flat_a, flat_b).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def main():
    parser = argparse.ArgumentParser(description="Evaluate IoU for Zero-Clipping Benchmark")
    parser.add_argument("-original", required=True, help="Path to original mask (dir or video)")
    parser.add_argument("-generated", required=True, help="Path to generated output video")
    parser.add_argument("-bbox", required=True, help="Original Frame 0 bbox [x1,y1,x2,y2]")
    parser.add_argument(
        "--sam_checkpoint",
        default=None,
        help="Path to SAM 2.1 checkpoint",
    )
    parser.add_argument(
        "--sam_config",
        default=None,
        help="Path to SAM 2.1 config",
    )
    parser.add_argument(
        "--frame_cap",
        type=int,
        default=50,
        help="Max frames to validate",
    )
    args = parser.parse_args()

    bbox_str = args.bbox.replace("[", "").replace("]", "")
    bbox = [int(float(x)) for x in bbox_str.split(",")]
    if len(bbox) != 4:
        print("Error: bbox must be [x1,y1,x2,y2]")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sam_checkpoint = args.sam_checkpoint or os.path.join(
        base_dir, "checkpoints", "sam2.1_hiera_s.pt"
    )
    sam_config = args.sam_config or os.path.join(
        base_dir, "configs", "sam2.1", "sam2.1_hiera_s.yaml"
    )

    gt_frames = load_mask_sequence(args.original)
    if args.frame_cap:
        gt_frames = gt_frames[: args.frame_cap]
    total_frames = len(gt_frames)
    print(f"Ground truth frames: {total_frames}")

    print("Isolating generated object via SAM 2.1...")
    tracker = SkateboardTracker(
        sam_checkpoint, sam_config, frame_load_cap=args.frame_cap
    )
    tracker.init_video(args.generated)
    tracker.add_initial_prompt(frame_idx=0, bbox=bbox)
    gen_masks = list(tracker.propagate_yield())

    failed_frames = []
    min_iou = 1.0

    print(f"Starting validation on {min(len(gt_frames), len(gen_masks))} frames...")

    for i in range(min(len(gt_frames), len(gen_masks))):
        gt_mask = gt_frames[i]
        pred_mask = gen_masks[i]

        # Resize pred to match gt if needed
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        iou = calculate_iou(gt_mask, pred_mask)
        min_iou = min(min_iou, iou)

        status = "PASS" if iou > IOU_THRESHOLD else "FAIL"
        if iou <= IOU_THRESHOLD:
            failed_frames.append((i, iou))

        print(f"Frame {i:04d}: IoU = {iou:.4f} [{status}]")

    print("-" * 30)
    print(f"Minimum IoU: {min_iou:.4f}")

    if failed_frames:
        print(f"FAILED: {len(failed_frames)} frames dropped below {IOU_THRESHOLD} IoU.")
        print("First failure at Frame", failed_frames[0])
        sys.exit(1)
    else:
        print("SUCCESS: Zero-Clipping Benchmark Passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
