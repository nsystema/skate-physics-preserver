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
import io
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser

import cv2
import numpy as np
import torch

from flask import Flask, render_template, jsonify, request, send_file, send_from_directory

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
    # Video metadata
    "fps": 0,
    "frame_count": 0,
    "duration": 0,
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
    # Pipeline sub-steps for detailed progress
    "sub_steps": [],           # list[dict] with label, status, detail
    # Stage 2 — Generation
    "comfyui_server": os.environ.get("COMFYUI_SERVER", "127.0.0.1:8188"),
    "comfyui_path": os.environ.get("COMFYUI_PATH", ""),  # path to ComfyUI install dir
    "comfyui_process": None,   # subprocess.Popen if we auto-launched ComfyUI
    "comfyui_autostarted": False,
    "workflow_path": "./workflows/vace_template.json",
    "gen_status": "idle",      # idle | running | complete | error
    "gen_progress": 0,
    "gen_message": "",
    "gen_error": None,
    "gen_sub_steps": [],
    "generated_video_path": None,
    # Stage 3 — Validation
    "val_status": "idle",      # idle | running | complete | error
    "val_progress": 0,
    "val_message": "",
    "val_error": None,
    "val_sub_steps": [],
    "val_results": None,
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
        "fps": state.get("fps", 0),
        "frame_count": state.get("frame_count", 0),
        "duration": state.get("duration", 0),
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
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not ok:
            return jsonify({"error": "Cannot read first frame. The video codec may not be supported (e.g. AV1). Try a different video or re-download in H.264 format."}), 500

        h, w = frame.shape[:2]
        state["video_path"] = video_path
        state["frame0"] = frame
        state["frame0_b64"] = frame_to_base64(frame)
        state["frame_h"] = h
        state["frame_w"] = w
        state["fps"] = fps
        state["frame_count"] = frame_count
        state["duration"] = frame_count / fps if fps > 0 else 0
        state["status"] = "loaded"

        return jsonify({
            "status": "loaded",
            "frame": state["frame0_b64"],
            "width": w,
            "height": h,
            "fps": fps,
            "frame_count": frame_count,
            "duration": frame_count / fps if fps > 0 else 0,
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
        "sub_steps": state.get("sub_steps", []),
        "frame_count": state.get("frame_count", 0),
        "fps": state.get("fps", 0),
        "duration": state.get("duration", 0),
    })


# ------------------------------------------------------------------
# 5.  Output file serving & comparison viewer APIs
# ------------------------------------------------------------------

@app.route("/output/<path:subpath>")
def serve_output(subpath):
    """Serve files from the output directory (masks, poses, original frames)."""
    out_dir = state.get("output_dir")
    if not out_dir:
        return jsonify({"error": "No output directory set"}), 400
    abs_dir = os.path.abspath(out_dir)
    return send_from_directory(abs_dir, subpath)


@app.route("/api/frame/original/<int:idx>")
def get_original_frame(idx):
    """Serve a specific frame from the original video as JPEG (fallback if
    pre-extracted frames are not available)."""
    if not state["video_path"]:
        return jsonify({"error": "No video loaded"}), 400

    # Try pre-extracted frame first
    if state["output_dir"]:
        pre_path = os.path.join(state["output_dir"], "frames_original",
                                f"frame_{idx:05d}.jpg")
        if os.path.isfile(pre_path):
            return send_file(pre_path, mimetype="image/jpeg")

    # Fallback: extract on-the-fly
    cap = cv2.VideoCapture(state["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({"error": f"Cannot read frame {idx}"}), 404

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.route("/api/video/original")
def serve_video_file():
    """Stream the original video file for HTML5 <video> playback."""
    vp = state.get("video_path")
    if not vp or not os.path.isfile(vp):
        return jsonify({"error": "No video loaded"}), 400
    return send_file(os.path.abspath(vp), mimetype="video/mp4")


@app.route("/api/outputs/info")
def outputs_info():
    """Return info about available pipeline outputs (for the comparison viewer)."""
    out_dir = state.get("output_dir", "")
    info = {
        "frame_count": state.get("frame_count", 0),
        "fps": state.get("fps", 0),
        "duration": state.get("duration", 0),
        "outputs": {},
    }
    for name in ["frames_original", "pose_skater", "mask_skateboard", "mask_skater"]:
        d = os.path.join(out_dir, name) if out_dir else ""
        if d and os.path.isdir(d):
            files = sorted(f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg')))
            info["outputs"][name] = {
                "count": len(files),
                "dir": name,
                "first": files[0] if files else None,
                "ext": os.path.splitext(files[0])[1] if files else ".png",
            }
    return jsonify(info)


# ------------------------------------------------------------------
# 6.  ComfyUI server management  (check / auto-start / stop)
# ------------------------------------------------------------------

def _comfyui_is_reachable(server=None):
    """Return True if ComfyUI is responding at the given address."""
    server = server or state["comfyui_server"]
    try:
        import requests as _req
        r = _req.get(f"http://{server}/system_stats", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _start_comfyui(comfyui_path=None, extra_args=None):
    """
    Start ComfyUI as a background subprocess.

    Args:
        comfyui_path: Path to the ComfyUI installation directory
                      (the folder containing main.py).
        extra_args:   Additional CLI args (list of strings).

    Returns:
        subprocess.Popen or None on failure.
    """
    comfyui_path = comfyui_path or state.get("comfyui_path", "")
    if not comfyui_path:
        return None

    main_py = os.path.join(comfyui_path, "main.py")
    if not os.path.isfile(main_py):
        print(f"[ComfyUI] main.py not found at {main_py}")
        return None

    # Build the command
    cmd = [sys.executable, main_py, "--listen", "0.0.0.0", "--lowvram"]

    # Parse host:port to pass --port if non-default
    server = state.get("comfyui_server", "127.0.0.1:8188")
    if ":" in server:
        port = server.split(":")[-1]
        if port != "8188":
            cmd.extend(["--port", port])

    if extra_args:
        cmd.extend(extra_args)

    print(f"[ComfyUI] Auto-starting: {' '.join(cmd)}")
    print(f"[ComfyUI] Working directory: {comfyui_path}")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=comfyui_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # On Windows, CREATE_NEW_PROCESS_GROUP lets us stop it cleanly
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                           if sys.platform == "win32" else 0),
        )
        state["comfyui_process"] = proc
        state["comfyui_autostarted"] = True

        # Start a thread to drain stdout so the pipe doesn't fill up
        def _drain(p):
            try:
                for line in iter(p.stdout.readline, b""):
                    print(f"[ComfyUI] {line.decode(errors='replace').rstrip()}")
            except Exception:
                pass
        threading.Thread(target=_drain, args=(proc,), daemon=True).start()

        return proc
    except Exception as exc:
        print(f"[ComfyUI] Failed to start: {exc}")
        return None


def _wait_for_comfyui(timeout=120, poll_interval=2):
    """
    Block until ComfyUI becomes reachable or *timeout* seconds elapse.

    Returns True if server came up, False on timeout.
    """
    server = state["comfyui_server"]
    deadline = time.time() + timeout
    print(f"[ComfyUI] Waiting for server at {server} (timeout {timeout}s) ...")

    while time.time() < deadline:
        # Check if the process died
        proc = state.get("comfyui_process")
        if proc and proc.poll() is not None:
            print(f"[ComfyUI] Process exited with code {proc.returncode}")
            return False

        if _comfyui_is_reachable(server):
            print(f"[ComfyUI] Server is ready at {server}")
            return True

        time.sleep(poll_interval)

    print(f"[ComfyUI] Timed out after {timeout}s")
    return False


def _stop_comfyui():
    """Gracefully stop a ComfyUI process we auto-started."""
    proc = state.get("comfyui_process")
    if proc is None:
        return

    print("[ComfyUI] Stopping auto-started server ...")
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    except Exception as exc:
        print(f"[ComfyUI] Error stopping: {exc}")
    finally:
        state["comfyui_process"] = None
        state["comfyui_autostarted"] = False
        print("[ComfyUI] Stopped.")


def _ensure_comfyui():
    """
    Make sure ComfyUI is reachable.  Auto-start it if a path is configured
    and the server isn't already running.

    Returns (ok: bool, message: str).
    """
    server = state["comfyui_server"]

    # Already running?
    if _comfyui_is_reachable(server):
        return True, f"ComfyUI reachable at {server}"

    # Can we auto-start?
    comfyui_path = state.get("comfyui_path", "")
    if not comfyui_path:
        return False, (
            f"Cannot reach ComfyUI at {server} and no ComfyUI path is configured. "
            f"Either start ComfyUI manually or set --comfyui-path / COMFYUI_PATH."
        )

    # Auto-start
    proc = _start_comfyui(comfyui_path)
    if proc is None:
        return False, (
            f"Failed to start ComfyUI from {comfyui_path}. "
            f"Check that main.py exists in that directory."
        )

    if _wait_for_comfyui(timeout=120):
        return True, f"ComfyUI auto-started and ready at {server}"
    else:
        _stop_comfyui()
        return False, (
            f"ComfyUI was started but did not become reachable within 120s. "
            f"Check the logs above for errors."
        )


@app.route("/api/comfyui/check", methods=["POST"])
def check_comfyui():
    """Check if ComfyUI server is reachable."""
    data = request.get_json(force=True) if request.is_json else {}
    server = data.get("server", state["comfyui_server"])
    state["comfyui_server"] = server

    try:
        from generate_reskin import ComfyOrchestrator
        orch = ComfyOrchestrator(server)
        if orch.check_server():
            return jsonify({
                "status": "ok",
                "message": f"ComfyUI reachable at {server}",
                "autostarted": state.get("comfyui_autostarted", False),
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Cannot reach ComfyUI at {server}",
                "can_autostart": bool(state.get("comfyui_path")),
            }), 503
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/comfyui/start", methods=["POST"])
def start_comfyui():
    """Manually trigger ComfyUI auto-start from the UI."""
    if _comfyui_is_reachable():
        return jsonify({
            "status": "ok",
            "message": "ComfyUI is already running."
        })

    data = request.get_json(force=True) if request.is_json else {}
    path = data.get("path", state.get("comfyui_path", ""))
    if path:
        state["comfyui_path"] = path

    ok, msg = _ensure_comfyui()
    if ok:
        return jsonify({"status": "ok", "message": msg})
    else:
        return jsonify({"status": "error", "message": msg}), 500


@app.route("/api/comfyui/stop", methods=["POST"])
def stop_comfyui():
    """Stop a ComfyUI server we auto-started."""
    if not state.get("comfyui_autostarted"):
        return jsonify({
            "status": "ok",
            "message": "ComfyUI was not auto-started by this app."
        })
    _stop_comfyui()
    return jsonify({"status": "ok", "message": "ComfyUI stopped."})


# ------------------------------------------------------------------
# 7.  Generate reskin  (Stage 2)
# ------------------------------------------------------------------

@app.route("/api/generate", methods=["POST"])
def start_generation():
    """Start the reskin generation pipeline."""
    if state["status"] != "complete":
        return jsonify({"error": "Stage 1 (extraction) must complete first."}), 400

    data = request.get_json(force=True) if request.is_json else {}
    positive_prompt = data.get("positive_prompt", "").strip()
    negative_prompt = data.get("negative_prompt",
                               "blurry, distorted, deformed, low quality").strip()
    server = data.get("server", state["comfyui_server"])

    if not positive_prompt:
        return jsonify({"error": "Positive prompt is required."}), 400

    state["comfyui_server"] = server
    state["gen_status"] = "running"
    state["gen_progress"] = 0
    state["gen_message"] = "Starting generation..."
    state["gen_error"] = None
    state["gen_sub_steps"] = []
    state["generated_video_path"] = None
    state["_positive_prompt"] = positive_prompt
    state["_negative_prompt"] = negative_prompt

    t = threading.Thread(target=_run_generation, daemon=True)
    t.start()
    return jsonify({"status": "running"})


@app.route("/api/generate/status")
def generation_status():
    """Poll generation progress."""
    return jsonify({
        "status": state["gen_status"],
        "progress": state["gen_progress"],
        "message": state["gen_message"],
        "error": state["gen_error"],
        "sub_steps": state.get("gen_sub_steps", []),
        "generated_video": state.get("generated_video_path"),
    })


# ------------------------------------------------------------------
# 8.  Validate IoU  (Stage 3)
# ------------------------------------------------------------------

@app.route("/api/validate", methods=["POST"])
def start_validation():
    """Start IoU validation."""
    gen_video = state.get("generated_video_path")

    # Accept uploaded video
    if "video" in request.files:
        f = request.files["video"]
        if f and f.filename:
            upload_dir = os.path.join(state["output_dir"], "generated")
            os.makedirs(upload_dir, exist_ok=True)
            save_path = os.path.join(upload_dir, f.filename)
            f.save(save_path)
            gen_video = save_path
            state["generated_video_path"] = gen_video

    if not gen_video or not os.path.isfile(gen_video):
        return jsonify({
            "error": "No generated video available. "
                     "Run generation first or upload a video."
        }), 400

    data = (request.form if request.form
            else (request.get_json(force=True) if request.is_json else {}))
    threshold = float(data.get("threshold", 0.90))

    state["val_status"] = "running"
    state["val_progress"] = 0
    state["val_message"] = "Starting validation..."
    state["val_error"] = None
    state["val_sub_steps"] = []
    state["val_results"] = None
    state["_val_threshold"] = threshold

    t = threading.Thread(target=_run_validation, daemon=True)
    t.start()
    return jsonify({"status": "running"})


@app.route("/api/validate/status")
def validation_status():
    """Poll validation progress."""
    return jsonify({
        "status": state["val_status"],
        "progress": state["val_progress"],
        "message": state["val_message"],
        "error": state["val_error"],
        "sub_steps": state.get("val_sub_steps", []),
        "results": state.get("val_results"),
    })


@app.route("/api/generated/video")
def serve_generated_video():
    """Serve the generated video file."""
    vp = state.get("generated_video_path")
    if not vp or not os.path.isfile(vp):
        return jsonify({"error": "No generated video available"}), 404
    ext = os.path.splitext(vp)[1].lower()
    mime_map = {".mp4": "video/mp4", ".webm": "video/webm",
                ".gif": "image/gif", ".avi": "video/x-msvideo"}
    return send_file(os.path.abspath(vp),
                     mimetype=mime_map.get(ext, "video/mp4"))


# ===================================================================
# Background pipeline
# ===================================================================

def _update_sub(idx, status, detail=""):
    """Update a sub-step in state['sub_steps']."""
    if 0 <= idx < len(state["sub_steps"]):
        state["sub_steps"][idx]["status"] = status
        if detail:
            state["sub_steps"][idx]["detail"] = detail


def _run_pipeline():
    """Execute frame extraction + DWPose (pass 1) + SAM propagation (pass 2)."""
    try:
        video_path = state["video_path"]
        output_dir = state["output_dir"]
        device = state["device"]

        frames_dir = os.path.join(output_dir, "frames_original")
        masks_skateboard_dir = os.path.join(output_dir, "mask_skateboard")
        masks_skater_dir = os.path.join(output_dir, "mask_skater")
        poses_dir = os.path.join(output_dir, "pose_skater")
        json_dir = os.path.join(output_dir, "pose_json")

        # Map obj_ids → detections
        prompts: dict[int, dict] = {}
        for det in state["detections"]:
            if det["label"] == "skateboard":
                prompts[1] = det
            elif det["label"] == "skater":
                prompts[2] = det

        # Initialize sub-steps
        state["sub_steps"] = [
            {"label": "Extract original frames", "status": "pending", "detail": ""},
            {"label": "DWPose skeleton extraction", "status": "pending", "detail": ""},
            {"label": "SAM 2.1 mask propagation", "status": "pending", "detail": ""},
            {"label": "Save metadata", "status": "pending", "detail": ""},
        ]

        # ----------------------------------------------------------
        # STEP 0 — Extract original frames (for comparison viewer)
        # ----------------------------------------------------------
        _update_sub(0, "running", "Reading video …")
        state["message"] = "Extracting original frames for comparison viewer …"
        state["progress"] = 1

        os.makedirs(frames_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(
                os.path.join(frames_dir, f"frame_{idx:05d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            idx += 1
        cap.release()

        state["fps"] = fps
        state["frame_count"] = idx
        state["duration"] = idx / fps if fps > 0 else 0
        state["progress"] = 5
        _update_sub(0, "complete", f"{idx} frames")

        # ----------------------------------------------------------
        # PASS 1 — DWPose skeleton extraction
        # ----------------------------------------------------------
        _update_sub(1, "running", "Loading DWPose model …")
        state["message"] = "Pass 1/2 — Extracting skater pose (DWPose) …"
        state["progress"] = 8

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
        _update_sub(1, "complete", f"{n_pose} poses extracted")
        state["message"] = f"Pass 1/2 done — {n_pose} pose frames."

        # ----------------------------------------------------------
        # PASS 2 — SAM 2.1 multi-object mask propagation
        # ----------------------------------------------------------
        _update_sub(2, "running", "Loading SAM 2.1 model …")
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
        _update_sub(2, "running", "Initializing prompts …")

        output_dirs: dict[int, str] = {}
        for oid, det in prompts.items():
            tracker.add_initial_prompt(frame_idx=0, bbox=det["bbox"], obj_id=oid)
            if det["label"] == "skateboard":
                output_dirs[oid] = masks_skateboard_dir
            else:
                output_dirs[oid] = masks_skater_dir

        state["message"] = "Propagating masks through video …"
        state["progress"] = 60
        _update_sub(2, "running", "Propagating masks …")

        def _prog(frac):
            state["progress"] = 60 + int(frac * 33)
            _update_sub(2, "running", f"{int(frac * 100)}% complete")

        counts = tracker.propagate_and_save_multi(
            output_dirs=output_dirs,
            progress_callback=_prog,
        )
        tracker.cleanup()

        n_mask = max(counts.values()) if counts else 0
        _update_sub(2, "complete", f"{n_mask} mask frames")

        # ----------------------------------------------------------
        # Metadata
        # ----------------------------------------------------------
        _update_sub(3, "running", "Writing metadata …")
        state["progress"] = 95
        state["message"] = "Saving pipeline metadata …"

        metadata = {
            "source": video_path,
            "video": os.path.abspath(video_path),
            "fps": fps,
            "frame_count": state["frame_count"],
            "duration": state["duration"],
            "detections": state["detections"],
            "object_counts": {str(k): v for k, v in counts.items()},
            "mask_frames": n_mask,
            "pose_frames": n_pose,
            "output_dir": os.path.abspath(output_dir),
            "frames_original_dir": os.path.abspath(frames_dir),
            "masks_skateboard_dir": os.path.abspath(masks_skateboard_dir),
            "masks_skater_dir": os.path.abspath(masks_skater_dir),
            "poses_dir": os.path.abspath(poses_dir),
            "json_dir": os.path.abspath(json_dir),
        }
        meta_path = os.path.join(output_dir, "tracking_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        _update_sub(3, "complete", "Done")

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
        # Mark current running sub-step as error
        for ss in state.get("sub_steps", []):
            if ss["status"] == "running":
                ss["status"] = "error"
                ss["detail"] = str(exc)
                break
        import traceback; traceback.print_exc()


# ===================================================================
# Stage 2 — Generation background thread
# ===================================================================

def _update_gen_sub(idx, status, detail=""):
    """Update a sub-step in state['gen_sub_steps']."""
    if 0 <= idx < len(state["gen_sub_steps"]):
        state["gen_sub_steps"][idx]["status"] = status
        if detail:
            state["gen_sub_steps"][idx]["detail"] = detail


def _run_generation():
    """Execute ComfyUI generation pipeline (Stage 2)."""
    try:
        from generate_reskin import ComfyOrchestrator

        server = state["comfyui_server"]
        workflow_path = state["workflow_path"]
        source_video = state["video_path"]
        masks_dir = os.path.join(state["output_dir"], "mask_skateboard")
        poses_dir = os.path.join(state["output_dir"], "pose_skater")
        output_dir = os.path.join(state["output_dir"], "generated")
        positive_prompt = state["_positive_prompt"]
        negative_prompt = state["_negative_prompt"]

        state["gen_sub_steps"] = [
            {"label": "Check ComfyUI connection", "status": "pending", "detail": ""},
            {"label": "Upload assets", "status": "pending", "detail": ""},
            {"label": "Configure & submit job", "status": "pending", "detail": ""},
            {"label": "Generate video", "status": "pending", "detail": ""},
            {"label": "Retrieve output", "status": "pending", "detail": ""},
        ]

        # Step 0: Check connection (auto-start if needed)
        _update_gen_sub(0, "running", "Connecting...")
        state["gen_message"] = "Checking ComfyUI connection..."
        state["gen_progress"] = 2

        orch = ComfyOrchestrator(server)
        if not orch.check_server():
            # Attempt auto-start
            comfyui_path = state.get("comfyui_path", "")
            if comfyui_path:
                _update_gen_sub(0, "running", "Auto-starting ComfyUI...")
                state["gen_message"] = "ComfyUI not running — auto-starting..."
                ok, msg = _ensure_comfyui()
                if not ok:
                    raise RuntimeError(msg)
                # Re-check after auto-start
                if not orch.check_server():
                    raise RuntimeError(
                        f"ComfyUI was started but still not reachable at {server}."
                    )
            else:
                raise RuntimeError(
                    f"Cannot reach ComfyUI at {server} and no ComfyUI path is "
                    f"configured for auto-start. Set --comfyui-path or the "
                    f"COMFYUI_PATH environment variable to enable auto-start, "
                    f"or start ComfyUI manually."
                )
        _update_gen_sub(0, "complete",
                        "Connected" + (" (auto-started)" if state.get("comfyui_autostarted") else ""))
        state["gen_progress"] = 5

        # Validate inputs
        for label, path in [
            ("Source video", source_video),
            ("Masks directory", masks_dir),
            ("Poses directory", poses_dir),
            ("Workflow template", workflow_path),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} not found: {path}")

        # Progress callback
        def on_progress(event, detail, pct):
            if event == "uploading":
                _update_gen_sub(1, "running", detail)
                state["gen_progress"] = 5 + int(pct * 25)
                state["gen_message"] = f"Uploading: {detail}"
            elif event == "injecting":
                _update_gen_sub(1, "complete")
                _update_gen_sub(2, "running", detail)
                state["gen_progress"] = 32
                state["gen_message"] = detail
            elif event == "generating":
                _update_gen_sub(2, "complete")
                _update_gen_sub(3, "running", detail)
                state["gen_progress"] = 35 + int(pct * 55)
                state["gen_message"] = f"Generating: {detail}"
            elif event == "retrieving":
                _update_gen_sub(3, "complete")
                _update_gen_sub(4, "running", detail)
                state["gen_progress"] = 92
                state["gen_message"] = detail
            elif event == "complete":
                _update_gen_sub(4, "complete", detail)
                state["gen_progress"] = 100

        orch.progress_callback = on_progress

        _update_gen_sub(1, "running", "Starting uploads...")
        state["gen_message"] = "Uploading assets to ComfyUI..."

        output_file = orch.execute_v2v(
            workflow_path=workflow_path,
            source_video_path=source_video,
            masks_dir=masks_dir,
            poses_dir=poses_dir,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            output_dir=output_dir,
        )

        state["generated_video_path"] = output_file
        state["gen_status"] = "complete"
        state["gen_progress"] = 100
        state["gen_message"] = (
            f"Generation complete! Output: "
            f"{os.path.basename(output_file) if output_file else 'unknown'}"
        )

        # Mark all steps complete
        for i in range(5):
            if state["gen_sub_steps"][i]["status"] != "complete":
                _update_gen_sub(i, "complete")

        print(f"\n[GEN DONE] {state['gen_message']}\n")

    except Exception as exc:
        state["gen_status"] = "error"
        state["gen_error"] = str(exc)
        state["gen_message"] = f"Generation failed: {exc}"
        for ss in state.get("gen_sub_steps", []):
            if ss["status"] == "running":
                ss["status"] = "error"
                ss["detail"] = str(exc)
                break
        import traceback; traceback.print_exc()


# ===================================================================
# Stage 3 — Validation background thread
# ===================================================================

def _update_val_sub(idx, status, detail=""):
    """Update a sub-step in state['val_sub_steps']."""
    if 0 <= idx < len(state["val_sub_steps"]):
        state["val_sub_steps"][idx]["status"] = status
        if detail:
            state["val_sub_steps"][idx]["detail"] = detail


def _run_validation():
    """Execute IoU validation (Stage 3)."""
    try:
        from evaluate_iou import (
            load_mask_sequence, extract_generated_masks, run_validation,
        )

        gen_video = state["generated_video_path"]
        masks_dir = os.path.join(state["output_dir"], "mask_skateboard")
        threshold = state.get("_val_threshold", 0.90)

        # Find skateboard bbox from detections
        bbox = None
        for det in state.get("detections", []):
            if det["label"] == "skateboard":
                bbox = det["bbox"]
                break
        if not bbox:
            raise ValueError(
                "No skateboard bounding box found from Stage 1 detections."
            )

        state["val_sub_steps"] = [
            {"label": "Load ground-truth masks", "status": "pending", "detail": ""},
            {"label": "Reverse-track generated video (SAM 2.1)", "status": "pending", "detail": ""},
            {"label": "Frame-by-frame IoU comparison", "status": "pending", "detail": ""},
        ]

        # Step 0: Load GT masks
        _update_val_sub(0, "running", "Loading masks...")
        state["val_message"] = "Loading ground-truth masks..."
        state["val_progress"] = 5

        gt_masks = load_mask_sequence(masks_dir)
        _update_val_sub(0, "complete", f"{len(gt_masks)} masks loaded")
        state["val_progress"] = 15

        # Step 1: Reverse-track generated video
        _update_val_sub(1, "running", "Running SAM 2.1 on generated video...")
        state["val_message"] = "Reverse-tracking generated video with SAM 2.1..."
        state["val_progress"] = 20

        gen_masks = extract_generated_masks(
            video_path=gen_video,
            bbox=bbox,
            sam_checkpoint=state["sam_checkpoint"],
            sam_config=state["sam_config"],
            device=state["device"],
        )
        _update_val_sub(1, "complete", f"{len(gen_masks)} masks extracted")
        state["val_progress"] = 75

        # Step 2: IoU comparison
        _update_val_sub(2, "running", "Computing IoU scores...")
        state["val_message"] = "Computing frame-by-frame IoU..."
        state["val_progress"] = 80

        results = run_validation(
            gt_masks, gen_masks, threshold=threshold, verbose=False,
        )
        _update_val_sub(2, "complete",
                        f"{'PASSED' if results['passed'] else 'FAILED'}")

        # Save report
        report_path = os.path.join(state["output_dir"], "iou_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        state["val_results"] = results
        state["val_status"] = "complete"
        state["val_progress"] = 100

        passed_str = "PASSED" if results["passed"] else "FAILED"
        state["val_message"] = (
            f"Validation {passed_str}. "
            f"Avg IoU: {results['avg_iou']:.4f} | "
            f"Min IoU: {results['min_iou']:.4f} | "
            f"Failed frames: {results['failed_count']}/{results['total_frames']}"
        )
        print(f"\n[VAL DONE] {state['val_message']}\n")

    except Exception as exc:
        state["val_status"] = "error"
        state["val_error"] = str(exc)
        state["val_message"] = f"Validation failed: {exc}"
        for ss in state.get("val_sub_steps", []):
            if ss["status"] == "running":
                ss["status"] = "error"
                ss["detail"] = str(exc)
                break
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
    ap.add_argument("--comfyui-server",
                    default=os.environ.get("COMFYUI_SERVER", "127.0.0.1:8188"),
                    help="ComfyUI server address (host:port). "
                         "Also reads COMFYUI_SERVER env var.")
    ap.add_argument("--comfyui-path",
                    default=os.environ.get("COMFYUI_PATH", ""),
                    help="Path to ComfyUI installation directory (contains main.py). "
                         "Enables auto-start when ComfyUI isn't running. "
                         "Also reads COMFYUI_PATH env var.")
    ap.add_argument("--workflow",
                    default="./workflows/vace_template.json",
                    help="ComfyUI workflow template JSON")
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
        "comfyui_server": args.comfyui_server,
        "comfyui_path": args.comfyui_path,
        "workflow_path": args.workflow,
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

    # Register cleanup for auto-started ComfyUI
    import atexit
    atexit.register(_stop_comfyui)

    comfyui_info = ""
    if args.comfyui_path:
        comfyui_info = f"\n  ComfyUI path : {args.comfyui_path}  (auto-start enabled)"
    else:
        comfyui_info = (f"\n  ComfyUI      : {args.comfyui_server}  "
                        f"(auto-start disabled — set --comfyui-path to enable)")

    print()
    print("=" * 60)
    print("  Skate Physics Preserver — Web Validation UI")
    print(f"  http://localhost:{args.port}{comfyui_info}")
    print("=" * 60)
    print()

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
