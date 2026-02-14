#!/usr/bin/env python3
"""
extract_physics.py - Master Tracking Orchestrator
===================================================
Usage:
  python src/extract_physics.py --video input.mp4 --output output/

Extracts:
  1. DWPose 133-keypoint skeleton sequence (Pass 1)
  2. SAM 2.1 skateboard alpha mask sequence (Pass 2)

Runs passes SEQUENTIALLY with full VRAM cleanup between them
so the entire pipeline fits in 8 GB VRAM (RTX 3070).

Interactive mode (default): Opens a window to draw bounding box.
Headless mode:              Pass --bbox "x1,y1,x2,y2" on CLI.
"""

import argparse
import json
import os
import sys
import gc

import cv2
import torch


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------

def get_user_box_interactive(video_path):
    """Open first frame and let the user draw a ROI around the skateboard."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Cannot read first frame for ROI selection.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Draw a box around the SKATEBOARD in the pop-up window.")
    print("  Press SPACE or ENTER to confirm. Press C to cancel.")
    print("=" * 60 + "\n")

    roi = cv2.selectROI("Select Skateboard", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("[ERROR] No selection made. Exiting.")
        sys.exit(1)

    x, y, w, h = roi
    bbox = [int(x), int(y), int(x + w), int(y + h)]
    print(f"[INFO] Selected bbox: {bbox}")
    return bbox


def parse_bbox_string(bbox_str):
    """Parse 'x1,y1,x2,y2' string into [x1, y1, x2, y2] list of ints."""
    parts = [int(v.strip()) for v in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 values for bbox, got {len(parts)}: {bbox_str}")
    return parts


def auto_detect_bbox(video_path):
    """
    Fallback: auto-detect a prominent object in the lower half of frame 0
    using simple contour detection. Returns a rough bbox or None.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    h, w = frame.shape[:2]
    # Focus on bottom 60% of the frame (where skateboards typically are)
    roi = frame[int(h * 0.4):, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Pick the largest contour by area
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)  # noqa: F841

    # Offset y back to full-frame coordinates
    y_offset = int(h * 0.4)
    bbox = [x, y + y_offset, x + bw, y + y_offset + bh]
    return bbox


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract tracking data (masks + poses) for ComfyUI V2V pipeline"
    )
    parser.add_argument("--video", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output folder base")
    parser.add_argument(
        "--bbox", type=str, default=None,
        help="Skateboard bounding box as 'x1,y1,x2,y2' (headless mode). "
             "If omitted, opens interactive selector."
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="./checkpoints/sam2.1_hiera_small.pt",
        help="Path to SAM 2.1 checkpoint (default: Hiera-Small for 8GB VRAM)"
    )
    parser.add_argument(
        "--sam-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM 2.1 config YAML"
    )
    parser.add_argument(
        "--pose-mode",
        default="balanced",
        choices=["performance", "balanced", "lightweight"],
        help="DWPose quality mode (default: balanced for 8GB VRAM)"
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--check", action="store_true",
        help="Run dependency check only (no processing)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Dependency check mode
    # ------------------------------------------------------------------
    if args.check:
        return run_dependency_check(args.device)

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Setup output directories
    # ------------------------------------------------------------------
    masks_dir = os.path.join(args.output, "mask_skateboard")
    poses_dir = os.path.join(args.output, "pose_skater")
    json_dir = os.path.join(args.output, "pose_json")
    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # PASS 1: DWPose (lighter on VRAM, run first)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PASS 1 / 2 : DWPose Skeleton Extraction")
    print("=" * 60)

    from tracking.skater_pose import SkaterPoseExtractor

    pose_extractor = SkaterPoseExtractor(
        device=args.device,
        backend="onnxruntime",
        mode=args.pose_mode,
    )
    n_pose_frames = pose_extractor.process_video(args.video, poses_dir, json_dir)
    pose_extractor.cleanup()

    # Force VRAM release before SAM2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[VRAM] Cleared after DWPose pass.")

    # ------------------------------------------------------------------
    # Determine bounding box
    # ------------------------------------------------------------------
    if args.bbox:
        user_bbox = parse_bbox_string(args.bbox)
    else:
        user_bbox = get_user_box_interactive(args.video)

    # ------------------------------------------------------------------
    # PASS 2: SAM 2.1 Mask Propagation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PASS 2 / 2 : SAM 2.1 Object Mask Propagation")
    print("=" * 60)

    from tracking.skateboard_tracker import SkateboardTracker

    tracker = SkateboardTracker(
        checkpoint_path=args.sam_checkpoint,
        config_path=args.sam_config,
        device=args.device,
    )
    tracker.init_video(args.video)
    tracker.add_initial_prompt(frame_idx=0, bbox=user_bbox)
    n_mask_frames = tracker.propagate_and_save(masks_dir)
    tracker.cleanup()

    # ------------------------------------------------------------------
    # Save metadata (bbox + counts) for downstream scripts
    # ------------------------------------------------------------------
    metadata = {
        "video": os.path.abspath(args.video),
        "bbox": user_bbox,
        "mask_frames": n_mask_frames,
        "pose_frames": n_pose_frames,
        "output_dir": os.path.abspath(args.output),
        "masks_dir": os.path.abspath(masks_dir),
        "poses_dir": os.path.abspath(poses_dir),
        "json_dir": os.path.abspath(json_dir),
    }
    meta_path = os.path.join(args.output, "tracking_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Mask frames : {n_mask_frames}  -> {masks_dir}")
    print(f"  Pose frames : {n_pose_frames}  -> {poses_dir}")
    print(f"  Pose JSON   : {n_pose_frames}  -> {json_dir}")
    print(f"  Metadata    : {meta_path}")
    print(f"  BBox used   : {user_bbox}")

    if n_mask_frames != n_pose_frames:
        print(f"\n  [WARNING] Frame count mismatch! Masks={n_mask_frames}, Poses={n_pose_frames}")
        print("  This may cause sync issues in ComfyUI. Verify video integrity.")

    print()
    return 0


# ---------------------------------------------------------------------------
# Dependency checker
# ---------------------------------------------------------------------------

def run_dependency_check(device):
    """Verify all dependencies are importable and GPU is accessible."""
    import importlib

    print("=" * 60)
    print("  DEPENDENCY CHECK")
    print("=" * 60)
    ok = True

    # PyTorch
    try:
        _torch = importlib.import_module("torch")
        print(f"  [OK] torch {_torch.__version__}")
        if device == "cuda":
            if _torch.cuda.is_available():
                gpu = _torch.cuda.get_device_name(0)
                vram = _torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
                print(f"  [OK] CUDA available: {gpu} ({vram:.1f} GB)")
            else:
                print("  [WARN] CUDA not available")
                ok = False
    except ImportError:
        print("  [FAIL] torch not installed")
        ok = False

    # SAM2
    try:
        importlib.import_module("sam2.build_sam")
        print("  [OK] sam2 (SAM 2.1)")
    except ImportError:
        print("  [FAIL] sam2 not installed (pip install git+https://github.com/facebookresearch/sam2.git@main)")
        ok = False

    # rtmlib
    try:
        importlib.import_module("rtmlib")
        print("  [OK] rtmlib (DWPose)")
    except ImportError:
        print("  [FAIL] rtmlib not installed (pip install rtmlib==0.2.0)")
        ok = False

    # ONNX Runtime
    try:
        _ort = importlib.import_module("onnxruntime")
        providers = _ort.get_available_providers()
        print(f"  [OK] onnxruntime {_ort.__version__} (providers: {providers})")
        if "CUDAExecutionProvider" not in providers:
            print("  [WARN] CUDAExecutionProvider not available in onnxruntime")
    except ImportError:
        print("  [FAIL] onnxruntime-gpu not installed")
        ok = False

    # OpenCV
    try:
        _cv2 = importlib.import_module("cv2")
        print(f"  [OK] opencv {_cv2.__version__}")
    except ImportError:
        print("  [FAIL] opencv-python not installed")
        ok = False

    # numpy
    try:
        _np = importlib.import_module("numpy")
        print(f"  [OK] numpy {_np.__version__}")
    except ImportError:
        print("  [FAIL] numpy not installed")
        ok = False

    print("=" * 60)
    if ok:
        print("  All dependencies OK. Ready to run.")
    else:
        print("  Some dependencies missing. Fix the [FAIL] items above.")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
