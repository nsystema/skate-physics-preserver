#!/usr/bin/env python3
"""
evaluate_iou.py - Automated Physics Validation (Zero-Clipping Benchmark)
=========================================================================
Usage:
  python src/evaluate_iou.py \\
    --original output/mask_skateboard \\
    --generated output/generated/output.mp4 \\
    --bbox "100,200,350,400" \\
    --sam-checkpoint checkpoints/sam2.1_hiera_small.pt

Or with metadata from extract_physics.py:
  python src/evaluate_iou.py \\
    --metadata output/tracking_metadata.json \\
    --generated output/generated/output.mp4

Logic:
  1. Load ground-truth mask sequence (original tracking output)
  2. Re-run SAM 2.1 on the GENERATED video using the SAME Frame-0 bbox
  3. Compare frame-by-frame IoU between original and generated masks
  4. Pass/Fail: IoU > 0.90 on ALL frames

Why "reverse tracking"?
  The generated video is a flat RGB file with no alpha channel.
  We must re-isolate the generated object (e.g., hoverboard) to measure
  whether it occupies the same pixel space as the original skateboard.
"""

import argparse
import json
import os
import sys
import glob

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def calculate_iou(mask_a, mask_b):
    """
    Compute Intersection over Union between two binary masks.

    Args:
        mask_a: np.ndarray (H, W), uint8 (0 or 255) or bool
        mask_b: np.ndarray (H, W), uint8 (0 or 255) or bool

    Returns:
        float: IoU score in [0.0, 1.0]
    """
    flat_a = mask_a.flatten().astype(bool)
    flat_b = mask_b.flatten().astype(bool)

    intersection = np.logical_and(flat_a, flat_b).sum()
    union = np.logical_or(flat_a, flat_b).sum()

    if union == 0:
        # Both masks are empty -> perfect match of "nothingness"
        return 1.0

    return float(intersection) / float(union)


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

def load_mask_sequence(mask_dir):
    """
    Load a directory of grayscale mask PNGs into a list of numpy arrays.
    Expects naming: frame_00000.png, frame_00001.png, ...
    """
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    files = sorted(glob.glob(os.path.join(mask_dir, "frame_*.png")))
    if not files:
        # Try any PNG files
        files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    if not files:
        raise FileNotFoundError(f"No mask PNGs found in {mask_dir}")

    masks = []
    for fpath in files:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARN] Failed to read: {fpath}, using blank mask")
            if masks:
                img = np.zeros_like(masks[-1])
            else:
                img = np.zeros((480, 640), dtype=np.uint8)
        masks.append(img)

    print(f"[GT] Loaded {len(masks)} ground-truth masks from {mask_dir}")
    return masks


# ---------------------------------------------------------------------------
# Reverse tracking on generated video
# ---------------------------------------------------------------------------

def extract_generated_masks(video_path, bbox, sam_checkpoint, sam_config, device="cuda"):
    """
    Run SAM 2.1 on the generated video to isolate the generated object.
    Uses the same Frame-0 bbox as the original tracking pass.

    Returns:
        list of (frame_idx, mask_np) tuples
    """
    from tracking.skateboard_tracker import SkateboardTracker

    tracker = SkateboardTracker(
        checkpoint_path=sam_checkpoint,
        config_path=sam_config,
        device=device,
    )
    tracker.init_video(video_path)
    tracker.add_initial_prompt(frame_idx=0, bbox=bbox)

    results = []
    for frame_idx, mask in tracker.propagate_yield():
        results.append((frame_idx, mask))

    tracker.cleanup()
    return results


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def run_validation(gt_masks, gen_masks, threshold=0.90, verbose=True):
    """
    Compare ground-truth and generated masks frame-by-frame.

    Args:
        gt_masks:  list of np.ndarray (ground truth)
        gen_masks: list of (frame_idx, np.ndarray) (generated, from SAM2)
        threshold: IoU threshold for pass/fail (default 0.90)
        verbose:   Print per-frame results

    Returns:
        dict with summary statistics
    """
    total_frames = min(len(gt_masks), len(gen_masks))
    if total_frames == 0:
        print("[ERROR] No frames to compare.")
        return {"passed": False, "total_frames": 0}

    iou_scores = []
    failed_frames = []
    min_iou = 1.0
    max_iou = 0.0

    for i in range(total_frames):
        gt_mask = gt_masks[i]
        _, gen_mask = gen_masks[i]

        # Ensure same dimensions
        if gt_mask.shape != gen_mask.shape:
            gen_mask = cv2.resize(
                gen_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        iou = calculate_iou(gt_mask, gen_mask)
        iou_scores.append(iou)
        min_iou = min(min_iou, iou)
        max_iou = max(max_iou, iou)

        status = "PASS" if iou > threshold else "FAIL"
        if iou <= threshold:
            failed_frames.append((i, iou))

        if verbose:
            bar = "#" * int(iou * 40) + "-" * (40 - int(iou * 40))
            print(f"  Frame {i:04d}: IoU = {iou:.4f} [{bar}] {status}")

    # Summary
    avg_iou = np.mean(iou_scores) if iou_scores else 0.0
    median_iou = np.median(iou_scores) if iou_scores else 0.0
    passed = len(failed_frames) == 0

    return {
        "passed": passed,
        "total_frames": total_frames,
        "failed_count": len(failed_frames),
        "failed_frames": failed_frames[:10],  # first 10 failures
        "min_iou": float(min_iou),
        "max_iou": float(max_iou),
        "avg_iou": float(avg_iou),
        "median_iou": float(median_iou),
        "threshold": threshold,
        "all_scores": [float(s) for s in iou_scores],
    }


def print_report(results):
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("  ZERO-CLIPPING BENCHMARK REPORT")
    print("=" * 60)
    print(f"  Total Frames Evaluated : {results['total_frames']}")
    print(f"  IoU Threshold          : {results['threshold']:.2f}")
    print(f"  Minimum IoU            : {results['min_iou']:.4f}")
    print(f"  Maximum IoU            : {results['max_iou']:.4f}")
    print(f"  Average IoU            : {results['avg_iou']:.4f}")
    print(f"  Median IoU             : {results['median_iou']:.4f}")
    print(f"  Failed Frames          : {results['failed_count']}")
    print()

    if results["passed"]:
        print("  *** RESULT: PASSED ***")
        print(f"  All {results['total_frames']} frames maintained IoU > {results['threshold']:.2f}")
    else:
        print("  *** RESULT: FAILED ***")
        print(f"  {results['failed_count']} frame(s) dropped below {results['threshold']:.2f}")
        print()
        print("  First failures:")
        for frame_idx, iou in results["failed_frames"]:
            print(f"    Frame {frame_idx:04d}: IoU = {iou:.4f}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IoU between original tracking masks and generated video"
    )
    parser.add_argument(
        "--original", "-o",
        help="Path to original mask directory (frame_*.png)"
    )
    parser.add_argument(
        "--generated", "-g", required=True,
        help="Path to generated output video (MP4)"
    )
    parser.add_argument(
        "--bbox",
        help="Original Frame-0 bounding box as 'x1,y1,x2,y2'"
    )
    parser.add_argument(
        "--metadata", "-m",
        help="Path to tracking_metadata.json (auto-fills --original and --bbox)"
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="./checkpoints/sam2.1_hiera_small.pt",
        help="SAM 2.1 checkpoint path"
    )
    parser.add_argument(
        "--sam-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM 2.1 config YAML"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="IoU threshold for pass/fail (default: 0.90)"
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-frame output")
    parser.add_argument(
        "--save-report",
        help="Save JSON report to this path"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve metadata
    # ------------------------------------------------------------------
    original_dir = args.original
    bbox = None

    if args.metadata:
        if not os.path.isfile(args.metadata):
            print(f"[ERROR] Metadata file not found: {args.metadata}")
            sys.exit(1)
        with open(args.metadata, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not original_dir:
            original_dir = meta.get("masks_dir")
        if not args.bbox:
            bbox = meta.get("bbox")
        print(f"[META] Loaded metadata from {args.metadata}")

    if not original_dir:
        print("[ERROR] --original or --metadata is required")
        sys.exit(1)

    if args.bbox:
        bbox = [int(v.strip()) for v in args.bbox.split(",")]

    if not bbox:
        print("[ERROR] --bbox or --metadata (with bbox) is required")
        sys.exit(1)

    if not os.path.exists(args.generated):
        print(f"[ERROR] Generated video not found: {args.generated}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: Load ground-truth masks
    # ------------------------------------------------------------------
    print("\n--- Step 1: Loading Ground-Truth Masks ---")
    gt_masks = load_mask_sequence(original_dir)

    # ------------------------------------------------------------------
    # Step 2: Reverse-track generated video
    # ------------------------------------------------------------------
    print("\n--- Step 2: Reverse-Tracking Generated Video ---")
    print(f"  Video: {args.generated}")
    print(f"  BBox:  {bbox}")
    gen_masks = extract_generated_masks(
        video_path=args.generated,
        bbox=bbox,
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        device=args.device,
    )
    print(f"  Extracted {len(gen_masks)} masks from generated video")

    # ------------------------------------------------------------------
    # Step 3: Frame-by-frame IoU validation
    # ------------------------------------------------------------------
    print("\n--- Step 3: Frame-by-Frame IoU Validation ---")
    results = run_validation(
        gt_masks, gen_masks,
        threshold=args.threshold,
        verbose=not args.quiet,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print_report(results)

    # Save JSON report if requested
    if args.save_report:
        os.makedirs(os.path.dirname(args.save_report) or ".", exist_ok=True)
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Report saved to: {args.save_report}")

    # Exit code: 0 = passed, 1 = failed
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
