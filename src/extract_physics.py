"""
Master Orchestration Script: Tracking Extraction (extract_physics)
Extracts SAM 2.1 object masks and DWPose skeletons from source video.
Supports both local (cv2.selectROI) and Colab (get_bbox_colab) bbox input.
"""

import argparse
import os
import sys

# Add parent for imports when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.skateboard_tracker import SkateboardTracker
from src.tracking.skater_pose import SkaterPoseExtractor

# Frame cap for Colab T4 (16GB VRAM)
DEFAULT_FRAME_CAP = 50


def get_user_box(video_path):
    """
    Opens the first frame and lets the user draw a ROI (local only, requires display).
    Returns [x1, y1, x2, y2].
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error reading video for ROI selection.")
        sys.exit(1)

    print("\n---------------------------------------------------------")
    print("A window will open. Draw a box around the SKATEBOARD.")
    print("Press SPACE or ENTER to confirm. Press c to cancel.")
    print("---------------------------------------------------------\n")

    roi = cv2.selectROI("Select Skateboard", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("No selection made. Exiting.")
        sys.exit(1)

    x, y, w, h = roi
    return [x, y, x + w, y + h]


def get_bbox_colab(video_path, bbox_input=None):
    """
    Colab-friendly bbox input. No GUI required.

    Args:
        video_path: Path to video (used to show first frame if ipywidgets available)
        bbox_input: Optional [x1, y1, x2, y2] or "x1,y1,x2,y2" string.
                    If None, displays first frame and prompts for manual input.

    Returns:
        [x1, y1, x2, y2] as list of ints
    """
    import cv2
    from PIL import Image
    import io

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read first frame for bbox selection.")

    if bbox_input is not None:
        if isinstance(bbox_input, (list, tuple)) and len(bbox_input) == 4:
            return [int(x) for x in bbox_input]
        if isinstance(bbox_input, str):
            parts = [int(float(x.strip())) for x in bbox_input.replace("[", "").replace("]", "").split(",")]
            if len(parts) == 4:
                return parts

    # Display first frame for reference
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    try:
        from IPython.display import display

        display(img)
    except Exception:
        pass

    print("First frame displayed above. Provide bbox as [x1, y1, x2, y2] or 'x1,y1,x2,y2'")
    print("Example: 100,200,300,250 (box around object)")
    user_input = input("Enter bbox: ").strip()
    parts = [int(float(x.strip())) for x in user_input.replace("[", "").replace("]", "").split(",")]
    if len(parts) != 4:
        raise ValueError("Bbox must have 4 values: x1, y1, x2, y2")
    return parts


def run_extraction(
    video_path,
    output_dir,
    bbox,
    sam_checkpoint=None,
    sam_config=None,
    frame_load_cap=DEFAULT_FRAME_CAP,
    use_colab_models=True,
):
    """
    Run full tracking extraction pipeline.

    Args:
        video_path: Input video path
        output_dir: Base output directory
        bbox: [x1, y1, x2, y2] for skateboard on frame 0
        sam_checkpoint: Path to SAM 2.1 checkpoint (default: Hiera-Small for Colab)
        sam_config: Path to SAM 2.1 config YAML
        frame_load_cap: Max frames (50 for T4)
        use_colab_models: If True, use Hiera-Small paths
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sam_checkpoint is None:
        sam_checkpoint = os.path.join(
            base_dir, "checkpoints", "sam2.1_hiera_s.pt"
        )
    if sam_config is None:
        sam_config = os.path.join(base_dir, "configs", "sam2.1", "sam2.1_hiera_s.yaml")

    masks_dir = os.path.join(output_dir, "mask_skateboard")
    poses_dir = os.path.join(output_dir, "pose_skater")
    poses_json_dir = os.path.join(output_dir, "pose_json")

    print("\n=== STARTING PASS 1: DWPose ===")
    pose_extractor = SkaterPoseExtractor(mode="lightweight")
    pose_extractor.process_video(
        video_path, poses_dir, output_dir_json=poses_json_dir, frame_load_cap=frame_load_cap
    )

    print("\n=== STARTING PASS 2: SAM 2.1 ===")
    tracker = SkateboardTracker(
        sam_checkpoint, sam_config, frame_load_cap=frame_load_cap
    )
    tracker.init_video(video_path)
    tracker.add_initial_prompt(frame_idx=0, bbox=bbox)
    tracker.propagate_and_save(masks_dir)

    print("\n=== EXTRACTION COMPLETE ===")
    print(f"Output available at: {output_dir}")
    return {"masks_dir": masks_dir, "poses_dir": poses_dir, "poses_json_dir": poses_json_dir}


def main():
    parser = argparse.ArgumentParser(description="Extract Tracking Data for ComfyUI")
    parser.add_argument("-i", "--video", required=True, help="Input video path")
    parser.add_argument("-o", "--output", required=True, help="Output folder base")
    parser.add_argument(
        "--bbox",
        help="Bbox [x1,y1,x2,y2] for Colab/headless. Omit for local GUI selection.",
    )
    parser.add_argument(
        "--sam_checkpoint",
        default=None,
        help="Path to SAM 2.1 checkpoint (default: sam2.1_hiera_s.pt)",
    )
    parser.add_argument(
        "--sam_config",
        default=None,
        help="Path to SAM 2.1 config YAML",
    )
    parser.add_argument(
        "--frame_cap",
        type=int,
        default=DEFAULT_FRAME_CAP,
        help=f"Max frames to process (default: {DEFAULT_FRAME_CAP} for T4)",
    )
    args = parser.parse_args()

    if args.bbox:
        bbox_str = args.bbox.replace("[", "").replace("]", "")
        bbox = [int(float(x)) for x in bbox_str.split(",")]
        if len(bbox) != 4:
            print("Error: bbox must be [x1,y1,x2,y2]")
            sys.exit(1)
    else:
        bbox = get_user_box(args.video)

    run_extraction(
        args.video,
        args.output,
        bbox,
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        frame_load_cap=args.frame_cap,
    )


if __name__ == "__main__":
    main()
