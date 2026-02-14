#!/usr/bin/env python3
"""
app.py - Web Validation Interface for Skate Physics Preserver
==============================================================
Launches a local web server for:
  1. Video upload / YouTube URL input
  2. Automatic skater + skateboard detection  (YOLO ➜ SAM 2.1)
  3. Visual validation of segmentation masks
  4. Full pipeline execution  (DWPose + SAM propagation)

Usage:
  python src/app.py --output output/
  python src/app.py --video input.mp4 --output output/

Opens http://localhost:5000
"""

import argparse
import gc
import json
import os
import re
import sys
import threading
import time
import webbrowser

import cv2
import numpy as np
import torch

from flask import Flask, render_template, jsonify, request

# Make sibling modules importable
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from auto_detect import (
    detect_objects,
    segment_frame,
    segment_with_points,
    create_overlay,
    frame_to_base64,
)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Shared state  (single-user local tool — no DB needed)
# ---------------------------------------------------------------------------
state = {
    # Paths / config
    "video_path": None,
    "output_dir": None,
    "sam_checkpoint": None,
    "sam_config": None,
    "pose_mode": "balanced",
    "device": "cuda",
    # Frame-0 data
    "frame0": None,            # BGR np.ndarray
    "frame0_b64": None,        # base64 JPEG
    "frame_h": 0,
    "frame_w": 0,
    # Detection results
    "detections": [],          # list[dict]
    "masks": {},               # {label: mask_np}
    "overlay_b64": None,       # base64 JPEG
    # Pipeline state
    "status": "idle",          # idle | loading | detecting | detected
                               # | approved | running | complete | error
    "progress": 0,             # 0-100
    "message": "",
    "error": None,
}

# YouTube regex (borrowed from extract_physics.py)
_YT_RE = re.compile(
    r"(https?://)?(www\.)?"
    r"(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)"
    r"[A-Za-z0-9_\-]+"
)


def _is_yt(url: str) -> bool:
    return bool(_YT_RE.match(url))


# ===================================================================
# API routes
# ===================================================================

@app.route("/")
def index():
    """Serve the single-page web UI."""
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    """Return lightweight snapshot of current state."""
    return jsonify({
        "status": state["status"],
        "progress": state["progress"],
        "message": state["message"],
        "error": state["error"],
        "has_video": state["video_path"] is not None,
        "video_path": state["video_path"],
        "has_detections": len(state["detections"]) > 0,
        "detections": state["detections"],
    })


# ------------------------------------------------------------------
# 1.  Load video
# ------------------------------------------------------------------

@app.route("/api/load", methods=["POST"])
def load_video():
    """Accept video file upload or YouTube URL, extract frame 0."""
    state["status"] = "loading"
    state["error"] = None
    state["detections"] = []
    state["masks"] = {}
    state["overlay_b64"] = None

    try:
        video_path = None

        # File upload
        if "video" in request.files:
            f = request.files["video"]
            if f and f.filename:
                upload_dir = os.path.join(state["output_dir"], "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                save_path = os.path.join(upload_dir, f.filename)
                f.save(save_path)
                video_path = save_path

        # YouTube URL
        if not video_path:
            url = request.form.get("url", "").strip()
            if url:
                if _is_yt(url):
                    from extract_physics import download_youtube_video
                    dl_dir = os.path.join(state["output_dir"], "downloads")
                    video_path = download_youtube_video(url, dl_dir)
                else:
                    return jsonify({"error": "Not a valid YouTube URL."}), 400

        # Pre-loaded via CLI
        if not video_path and state["video_path"]:
            video_path = state["video_path"]

        if not video_path:
            return jsonify({"error": "No video provided."}), 400
        if not os.path.isfile(video_path):
            return jsonify({"error": f"File not found: {video_path}"}), 404

        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return jsonify({"error": "Cannot read first frame."}), 500

        h, w = frame.shape[:2]
        state["video_path"] = video_path
        state["frame0"] = frame
        state["frame0_b64"] = frame_to_base64(frame)
        state["frame_h"] = h
        state["frame_w"] = w
        state["status"] = "loaded"

        return jsonify({
            "status": "loaded",
            "frame": state["frame0_b64"],
            "width": w,
            "height": h,
            "video_path": os.path.basename(video_path),
        })
    except Exception as exc:
        state["status"] = "error"
        state["error"] = str(exc)
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# 2.  Auto-detect  (YOLO ➜ SAM)
# ------------------------------------------------------------------

@app.route("/api/detect", methods=["POST"])
def run_detection():
    """Run YOLO + SAM 2.1 auto-detection on frame 0."""
    if state["frame0"] is None:
        return jsonify({"error": "Load a video first."}), 400

    state["status"] = "detecting"
    state["error"] = None

    try:
        frame = state["frame0"]

        # YOLO
        detections = detect_objects(frame, device=state["device"])
        state["detections"] = detections

        if not detections:
            state["status"] = "detected"
            state["overlay_b64"] = state["frame0_b64"]
            return jsonify({
                "status": "detected",
                "detections": [],
                "overlay": state["frame0_b64"],
                "message": ("No skater or skateboard detected automatically. "
                            "Use manual mode to click on them."),
            })

        # SAM — segment each detection
        masks = segment_frame(
            frame, detections,
            sam_checkpoint=state["sam_checkpoint"],
            sam_config=state["sam_config"],
            device=state["device"],
        )
        state["masks"] = masks

        overlay = create_overlay(frame, detections, masks)
        state["overlay_b64"] = frame_to_base64(overlay)
        state["status"] = "detected"

        labels = [d["label"] for d in detections]
        return jsonify({
            "status": "detected",
            "detections": detections,
            "overlay": state["overlay_b64"],
            "message": f"Detected: {', '.join(labels)}. Review the masks and approve.",
        })
    except Exception as exc:
        state["status"] = "error"
        state["error"] = str(exc)
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# 3.  Manual point-click refinement
# ------------------------------------------------------------------

@app.route("/api/refine", methods=["POST"])
def refine():
    """Re-segment using manual point prompts."""
    if state["frame0"] is None:
        return jsonify({"error": "Load a video first."}), 400

    data = request.get_json(force=True)
    points = data.get("points", [])
    if not points:
        return jsonify({"error": "No points provided."}), 400

    state["status"] = "detecting"
    state["error"] = None

    try:
        frame = state["frame0"]

        masks = segment_with_points(
            frame, points,
            sam_checkpoint=state["sam_checkpoint"],
            sam_config=state["sam_config"],
            device=state["device"],
        )
        state["masks"] = masks

        # Derive bboxes from masks
        detections = []
        for label, mask in masks.items():
            ys, xs = np.where(mask > 127)
            if len(ys) > 0:
                detections.append({
                    "label": label,
                    "bbox": [int(xs.min()), int(ys.min()),
                             int(xs.max()), int(ys.max())],
                    "confidence": 1.0,
                    "class_id": -1,
                })
        state["detections"] = detections

        overlay = create_overlay(frame, detections, masks)
        state["overlay_b64"] = frame_to_base64(overlay)
        state["status"] = "detected"

        return jsonify({
            "status": "detected",
            "detections": detections,
            "overlay": state["overlay_b64"],
            "message": f"Manual segmentation complete ({len(masks)} object(s)).",
        })
    except Exception as exc:
        state["status"] = "error"
        state["error"] = str(exc)
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# 4.  Approve & run full pipeline
# ------------------------------------------------------------------

@app.route("/api/approve", methods=["POST"])
def approve():
    """Approve current detections and launch the full extraction pipeline."""
    if not state["detections"]:
        return jsonify({"error": "Nothing to approve — detect first."}), 400

    state["status"] = "running"
    state["progress"] = 0
    state["message"] = "Starting pipeline …"
    state["error"] = None

    t = threading.Thread(target=_run_pipeline, daemon=True)
    t.start()
    return jsonify({"status": "running"})


@app.route("/api/status")
def status():
    """Poll pipeline progress."""
    return jsonify({
        "status": state["status"],
        "progress": state["progress"],
        "message": state["message"],
        "error": state["error"],
    })


# ===================================================================
# Background pipeline
# ===================================================================

def _run_pipeline():
    """Execute DWPose (pass 1) + SAM multi-object propagation (pass 2)."""
    try:
        video_path = state["video_path"]
        output_dir = state["output_dir"]
        device = state["device"]

        masks_skateboard_dir = os.path.join(output_dir, "mask_skateboard")
        masks_skater_dir = os.path.join(output_dir, "mask_skater")
        poses_dir = os.path.join(output_dir, "pose_skater")
        json_dir = os.path.join(output_dir, "pose_json")

        # Map obj_ids ➜ detections
        prompts: dict[int, dict] = {}
        for det in state["detections"]:
            if det["label"] == "skateboard":
                prompts[1] = det
            elif det["label"] == "skater":
                prompts[2] = det

        # ----------------------------------------------------------
        # PASS 1 — DWPose skeleton extraction
        # ----------------------------------------------------------
        state["message"] = "Pass 1/2 — Extracting skater pose (DWPose) …"
        state["progress"] = 5

        from tracking.skater_pose import SkaterPoseExtractor

        pose_ext = SkaterPoseExtractor(
            device=device,
            backend="onnxruntime",
            mode=state["pose_mode"],
        )
        n_pose = pose_ext.process_video(video_path, poses_dir, json_dir)
        pose_ext.cleanup()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        state["progress"] = 40
        state["message"] = f"Pass 1/2 done — {n_pose} pose frames."

        # ----------------------------------------------------------
        # PASS 2 — SAM 2.1 multi-object mask propagation
        # ----------------------------------------------------------
        state["message"] = "Pass 2/2 — Loading SAM 2.1 video predictor …"
        state["progress"] = 45

        from tracking.skateboard_tracker import SkateboardTracker

        tracker = SkateboardTracker(
            checkpoint_path=state["sam_checkpoint"],
            config_path=state["sam_config"],
            device=device,
        )
        tracker.init_video(video_path)

        state["progress"] = 55

        output_dirs: dict[int, str] = {}
        for oid, det in prompts.items():
            tracker.add_initial_prompt(frame_idx=0, bbox=det["bbox"], obj_id=oid)
            if det["label"] == "skateboard":
                output_dirs[oid] = masks_skateboard_dir
            else:
                output_dirs[oid] = masks_skater_dir

        state["message"] = "Propagating masks through video …"
        state["progress"] = 60

        def _prog(frac):
            state["progress"] = 60 + int(frac * 35)

        counts = tracker.propagate_and_save_multi(
            output_dirs=output_dirs,
            progress_callback=_prog,
        )
        tracker.cleanup()

        # ----------------------------------------------------------
        # Metadata
        # ----------------------------------------------------------
        n_mask = max(counts.values()) if counts else 0

        metadata = {
            "source": video_path,
            "video": os.path.abspath(video_path),
            "detections": state["detections"],
            "object_counts": {str(k): v for k, v in counts.items()},
            "mask_frames": n_mask,
            "pose_frames": n_pose,
            "output_dir": os.path.abspath(output_dir),
            "masks_skateboard_dir": os.path.abspath(masks_skateboard_dir),
            "masks_skater_dir": os.path.abspath(masks_skater_dir),
            "poses_dir": os.path.abspath(poses_dir),
            "json_dir": os.path.abspath(json_dir),
        }
        meta_path = os.path.join(output_dir, "tracking_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        state["status"] = "complete"
        state["progress"] = 100
        state["message"] = (
            f"Pipeline complete!  "
            f"Mask frames: {n_mask}  |  Pose frames: {n_pose}  |  "
            f"Output: {output_dir}"
        )
        print(f"\n[DONE] {state['message']}\n")

    except Exception as exc:
        state["status"] = "error"
        state["error"] = str(exc)
        state["message"] = f"Pipeline failed: {exc}"
        import traceback; traceback.print_exc()


# ===================================================================
# Entry point
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Skate Physics Preserver — Web Validation UI"
    )
    ap.add_argument("--video", "-i", default=None,
                    help="Pre-load a video path or YouTube URL")
    ap.add_argument("--output", "-o", required=True,
                    help="Output directory")
    ap.add_argument("--sam-checkpoint",
                    default="./checkpoints/sam2.1_hiera_small.pt")
    ap.add_argument("--sam-config",
                    default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--pose-mode", default="balanced",
                    choices=["performance", "balanced", "lightweight"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true",
                    help="Don't auto-open the browser")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    state.update({
        "output_dir": args.output,
        "sam_checkpoint": args.sam_checkpoint,
        "sam_config": args.sam_config,
        "pose_mode": args.pose_mode,
        "device": args.device,
    })

    # Pre-load video if given on CLI
    if args.video:
        if _is_yt(args.video):
            from extract_physics import download_youtube_video
            dl_dir = os.path.join(args.output, "downloads")
            state["video_path"] = download_youtube_video(args.video, dl_dir)
        else:
            state["video_path"] = os.path.abspath(args.video)

    # Auto-open browser
    if not args.no_browser:
        def _open():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{args.port}")
        threading.Thread(target=_open, daemon=True).start()

    print()
    print("=" * 60)
    print("  Skate Physics Preserver — Web Validation UI")
    print(f"  http://localhost:{args.port}")
    print("=" * 60)
    print()

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
