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

Provide --bbox "x1,y1,x2,y2" or let the script auto-detect.
For interactive selection, use the web UI (src/app.py) instead.
"""

import argparse
import json
import os
import re
import sys
import gc

import cv2
import torch


# ---------------------------------------------------------------------------
# YouTube / URL helpers
# ---------------------------------------------------------------------------

_YOUTUBE_RE = re.compile(
    r"(https?://)?(www\.)?"
    r"(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)"
    r"[A-Za-z0-9_\-]+"
)


def is_youtube_url(path: str) -> bool:
    """Return True if *path* looks like a YouTube URL."""
    return bool(_YOUTUBE_RE.match(path))


def download_youtube_video(url: str, output_dir: str) -> str:
    """
    Download a YouTube video using yt-dlp and return the local file path.

    Downloads at best quality up to 1080p (keeping file size reasonable for an
    8 GB VRAM card).  The file is saved into *output_dir* with yt-dlp's default
    title-based filename.
    """
    try:
        import yt_dlp
    except ImportError:
        print("[ERROR] yt-dlp is required for YouTube URL support.")
        print("        Install it with:  pip install yt-dlp")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    outtmpl = os.path.join(output_dir, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
    }

    print(f"\n[INFO] Downloading YouTube video: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # yt-dlp may change the extension after merging; ensure .mp4
        base, _ = os.path.splitext(filename)
        final_path = base + ".mp4"

    if not os.path.isfile(final_path):
        # Fallback: look for any mp4 file yt-dlp just wrote
        for f in os.listdir(output_dir):
            if f.endswith(".mp4"):
                final_path = os.path.join(output_dir, f)
                break

    if not os.path.isfile(final_path):
        print("[ERROR] yt-dlp finished but no .mp4 file was found.")
        sys.exit(1)

    print(f"[INFO] Downloaded video saved to: {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------

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
    parser.add_argument("--video", "-i", default=None,
                        help="Input video path or YouTube URL")
    parser.add_argument("--output", "-o", default=None, help="Output folder base")
    parser.add_argument(
        "--bbox", type=str, default=None,
        help="Skateboard bounding box as 'x1,y1,x2,y2'. "
             "If omitted, attempts contour-based auto-detection."
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
    # Dependency check mode (no --video/--output required)
    # ------------------------------------------------------------------
    if args.check:
        return run_dependency_check(args.device)

    # ------------------------------------------------------------------
    # Validate required args (after --check so it can run standalone)
    # ------------------------------------------------------------------
    if not args.video:
        parser.error("--video/-i is required (unless using --check)")
    if not args.output:
        parser.error("--output/-o is required (unless using --check)")

    # ------------------------------------------------------------------
    # Resolve YouTube URLs → local file
    # ------------------------------------------------------------------
    video_path = args.video
    if is_youtube_url(video_path):
        download_dir = os.path.join(args.output, "downloads")
        video_path = download_youtube_video(video_path, download_dir)

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video not found: {video_path}")
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
    n_pose_frames = pose_extractor.process_video(video_path, poses_dir, json_dir)
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
        # Auto-detect skateboard bbox (no GUI required)
        print("[INFO] No --bbox provided, attempting auto-detection...")
        user_bbox = auto_detect_bbox(video_path)
        if user_bbox is None:
            print("[ERROR] Auto-detection failed. Please provide --bbox 'x1,y1,x2,y2'.")
            print("        Tip: Use the web UI (src/app.py) for interactive selection.")
            sys.exit(1)
        print(f"[INFO] Auto-detected bbox: {user_bbox}")

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
    tracker.init_video(video_path)
    tracker.add_initial_prompt(frame_idx=0, bbox=user_bbox)
    n_mask_frames = tracker.propagate_and_save(masks_dir)
    tracker.cleanup()

    # ------------------------------------------------------------------
    # Save metadata (bbox + counts) for downstream scripts
    # ------------------------------------------------------------------
    metadata = {
        "source": args.video,          # original arg (may be URL)
        "video": os.path.abspath(video_path),
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
                vram = _torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
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
        print("  [FAIL] rtmlib not installed (pip install rtmlib==0.0.15)")
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

    # yt-dlp (optional but needed for YouTube URL input)
    try:
        _ytdlp = importlib.import_module("yt_dlp")
        print(f"  [OK] yt-dlp {_ytdlp.version.__version__}")
    except ImportError:
        print("  [WARN] yt-dlp not installed (needed for YouTube URL support)")

    print("=" * 60)
    if ok:
        print("  All dependencies OK. Ready to run.")
    else:
        print("  Some dependencies missing. Fix the [FAIL] items above.")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
