#!/usr/bin/env python3
"""
generate_reskin.py - Headless ComfyUI Generative Orchestrator (Fun Control)
==============================================================================
Usage:
  python src/generate_reskin.py \\
    --source-video input.mp4 \\
    --masks-dir output/mask_skateboard \\
    --skater-masks-dir output/mask_skater \\
    --poses-dir output/pose_skater \\
    --positive-prompt "cyberpunk samurai riding a neon hoverboard" \\
    --negative-prompt "blurry, distorted, low quality" \\
    --output-dir output/generated

Architecture (Fun Control — Generate + Post-Composite):
  Two-step approach using Wan22FunControlToVideo:
    Step 1 — GENERATE: The model creates a fully new video guided by:
      - control_video: DWPose skeleton frames for motion/pose guidance.
      - text prompt: describes the desired character/skateboard appearance.
    Step 2 — POST-COMPOSITE: The generated character regions are blended
      onto the original video's background using the segmentation masks:
      - Where mask=1 (skater/skateboard): take from generated video
      - Where mask=0 (background): keep from original video

Pipeline:
  1. Reads source video dimensions and calculates VAE-compatible target
     resolution (nearest multiple of 16, capped at 480px short side).
  2. Uploads source video + masks + poses to ComfyUI.
  3. Injects dynamic resolution, unified prompt, and file paths into the
     Fun Control workflow template.
  4. Submits to ComfyUI headless server and monitors via WebSocket.
  5. Downloads the post-composited video on completion.

Requirements:
  - ComfyUI server running locally: python main.py --listen --lowvram
  - Custom nodes: VideoHelperSuite, ComfyUI-GGUF
  - Models: Wan 2.2 Fun 5B Control GGUF (Q4_K_M for 8GB VRAM)
  - VRAM: Optimized for RTX 3070 (8GB), max 81 frames at ~480p.
"""

import argparse
import json
import os
import random
import sys
import uuid

import cv2
import requests
import websocket


# ---------------------------------------------------------------------------
# Node-finding utilities
# ---------------------------------------------------------------------------

def find_node_by_class(workflow, class_type, index=0):
    """
    Locate a node ID by its class_type in the workflow.
    Returns the node ID string.

    Args:
        workflow:   The workflow dict.
        class_type: e.g. 'CLIPTextEncode', 'VHS_LoadVideo'
        index:      Which occurrence (0-based) if multiple exist.
    """
    matches = []
    for node_id, node_data in workflow.items():
        if node_data.get("class_type") == class_type:
            matches.append(node_id)

    if not matches:
        raise ValueError(f"Node class '{class_type}' not found in workflow template.")
    if index >= len(matches):
        raise ValueError(
            f"Only {len(matches)} node(s) of class '{class_type}' found, "
            f"but index {index} was requested."
        )
    return matches[index]


def find_node_by_title(workflow, title):
    """
    Locate a node ID by its _meta.title field.
    More robust than class_type if the user sets custom titles.
    """
    for node_id, node_data in workflow.items():
        meta = node_data.get("_meta", {})
        if meta.get("title", "").lower() == title.lower():
            return node_id
    raise ValueError(f"Node with title '{title}' not found in workflow template.")


def find_node_safe(workflow, class_type=None, title=None, index=0):
    """Try title first, fall back to class_type."""
    if title:
        try:
            return find_node_by_title(workflow, title)
        except ValueError:
            pass
    if class_type:
        try:
            return find_node_by_class(workflow, class_type, index)
        except ValueError:
            pass
    label = title or class_type or "unknown"
    raise ValueError(f"Cannot find node: {label}")


# ---------------------------------------------------------------------------
# Asset upload
# ---------------------------------------------------------------------------

def upload_asset(server_address, file_path, subfolder="", asset_type="input"):
    """
    Upload a file (video/image) to ComfyUI via multipart POST.

    Returns:
        dict with 'name', 'subfolder', 'type' keys.
    """
    url = f"http://{server_address}/upload/image"
    filename = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        files = {"image": (filename, f)}
        data = {"type": asset_type, "overwrite": "true"}
        if subfolder:
            data["subfolder"] = subfolder
        response = requests.post(url, files=files, data=data, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"Upload failed ({response.status_code}): {response.text}")

    result = response.json()
    print(f"[UPLOAD] {filename} -> {result}")
    return result


def upload_image_sequence(server_address, images_dir, subfolder_name, max_frames=0):
    """
    Upload an image sequence directory to ComfyUI.

    Args:
        max_frames: If > 0, only upload the first N frames (avoids uploading
                    hundreds of frames when the model will only use 81).
    """
    files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg"))],
    )
    if not files:
        raise FileNotFoundError(f"No image files found in {images_dir}")

    # Only upload as many frames as the model will actually use
    if max_frames > 0 and len(files) > max_frames:
        print(f"[UPLOAD] Capping upload: {max_frames}/{len(files)} frames from {images_dir}")
        files = files[:max_frames]

    print(f"[UPLOAD] Uploading {len(files)} frames from {images_dir} ...")
    for i, fname in enumerate(files):
        fpath = os.path.join(images_dir, fname)
        upload_asset(server_address, fpath, subfolder=subfolder_name)
        if (i + 1) % 50 == 0:
            print(f"  ... uploaded {i + 1}/{len(files)}")

    print(f"[UPLOAD] Done. {len(files)} frames uploaded to input/{subfolder_name}/")
    return subfolder_name, files[0]  # return subfolder and first filename


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------

class ComfyOrchestrator:
    """
    Manages the lifecycle of a ComfyUI V2V generation job:
    upload -> inject -> queue -> monitor -> retrieve.
    """

    def __init__(self, server_addr="127.0.0.1:8188"):
        self.server_addr = server_addr
        self.client_id = str(uuid.uuid4())
        self.ws = None
        self.progress_callback = None  # Optional: callback(event, detail, pct)

    def connect(self):
        """Establish WebSocket connection to ComfyUI server."""
        ws_url = f"ws://{self.server_addr}/ws?clientId={self.client_id}"
        print(f"[WS] Connecting to {ws_url} ...")
        self.ws = websocket.WebSocket()
        self.ws.settimeout(300)  # 5-minute timeout for long generations
        self.ws.connect(ws_url)
        print("[WS] Connected.")

    def check_server(self):
        """Verify ComfyUI server is reachable."""
        try:
            r = requests.get(f"http://{self.server_addr}/system_stats", timeout=5)
            if r.status_code == 200:
                stats = r.json()
                devices = stats.get("devices", [])
                if devices:
                    vram = devices[0].get("vram_total", 0) / (1024 ** 3)
                    name = devices[0].get("name", "unknown")
                    print(f"[SERVER] ComfyUI online: {name} ({vram:.1f} GB VRAM)")
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        return False

    def _notify(self, event, detail="", pct=0.0):
        """Call progress_callback if set. Events: uploading|injecting|generating|retrieving|complete"""
        if self.progress_callback:
            try:
                self.progress_callback(event, detail, pct)
            except Exception:
                pass

    def execute_v2v(
        self,
        workflow_path,
        source_video_path,
        masks_dir,
        poses_dir,
        skater_prompt,
        skateboard_prompt,
        negative_prompt,
        output_dir,
        ref_image_path=None,
        skater_masks_dir=None,
    ):
        """
        Execute the full V2V pipeline via ComfyUI API (Fun Control).

        Fun Control architecture (Generate + Post-Composite):
          Step 1: Wan22FunControlToVideo generates a fully new video:
            - control_video: DWPose skeleton frames for pose guidance.
            - Text prompt drives the desired character appearance.
          Step 2: ImageCompositeMasked blends generated character regions
            onto the original video background using segmentation masks.

        Args:
            workflow_path:      Path to template JSON.
            source_video_path:  Source MP4.
            masks_dir:          Directory of skateboard mask PNGs.
            poses_dir:          Directory of pose PNGs (used as
                                control_video for Fun Control).
            skater_prompt:      Text prompt for the skater/character.
            skateboard_prompt:  Text prompt for the skateboard/board.
                                Combined with skater_prompt into a unified
                                prompt for the Fun Control model.
            negative_prompt:    Negative prompt.
            output_dir:         Where to copy the result.
            ref_image_path:     Optional reference image for style guidance.
            skater_masks_dir:   Directory of skater mask PNGs (combined with
                                skateboard masks for white-out inpainting).
        """
        # ----- Load workflow template -----
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        print(f"[WORKFLOW] Loaded template: {workflow_path}")
        self._notify("uploading", "Workflow loaded, uploading assets...", 0.05)

        # ----- Read source video metadata for workflow sync -----
        video_fps = 24.0  # safe default
        video_w, video_h = 0, 0
        try:
            cap = cv2.VideoCapture(source_video_path)
            if cap.isOpened():
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"[VIDEO] Native: {video_w}x{video_h} @ {video_fps:.1f} FPS")
        except Exception:
            print("[VIDEO] Could not read metadata, using defaults")

        # ----- Upload assets -----
        print("\n--- Uploading Assets ---")
        self._notify("uploading", "Uploading source video...", 0.08)
        vid_meta = upload_asset(self.server_addr, source_video_path)

        _upload_cap = self.MAX_GEN_FRAMES  # only upload what the model will use

        self._notify("uploading", "Uploading skateboard mask sequence...", 0.15)
        mask_subfolder, _ = upload_image_sequence(
            self.server_addr, masks_dir, "mask_skateboard",
            max_frames=_upload_cap,
        )

        # Upload skater masks if available
        skater_mask_subfolder = None
        if skater_masks_dir and os.path.isdir(skater_masks_dir):
            self._notify("uploading", "Uploading skater mask sequence...", 0.22)
            skater_mask_subfolder, _ = upload_image_sequence(
                self.server_addr, skater_masks_dir, "mask_skater",
                max_frames=_upload_cap,
            )
        else:
            print("[WARN] No skater masks directory provided — skater will NOT be inpainted.")

        # Upload pose sequences (used as control_video for Fun Control)
        pose_subfolder = None
        if poses_dir and os.path.isdir(poses_dir):
            self._notify("uploading", "Uploading pose skeleton sequence...", 0.28)
            pose_subfolder, _ = upload_image_sequence(
                self.server_addr, poses_dir, "pose_skater",
                max_frames=_upload_cap,
            )
        else:
            print("[WARN] No pose directory provided — control_video will be empty.")

        img_meta = None
        if ref_image_path and os.path.isfile(ref_image_path):
            img_meta = upload_asset(self.server_addr, ref_image_path)

        # ----- Inject values into workflow -----
        print("\n--- Injecting Parameters ---")
        self._notify("injecting", "Configuring workflow parameters", 0.32)
        # Read total frame count for dynamic cap alignment
        _total_frames = 0
        try:
            _cap = cv2.VideoCapture(source_video_path)
            if _cap.isOpened():
                _total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            _cap.release()
        except Exception:
            pass
        self._inject_workflow(
            workflow,
            vid_meta=vid_meta,
            img_meta=img_meta,
            mask_subfolder=mask_subfolder,
            skater_prompt=skater_prompt,
            skateboard_prompt=skateboard_prompt,
            negative_prompt=negative_prompt,
            skater_mask_subfolder=skater_mask_subfolder,
            pose_subfolder=pose_subfolder,
            video_fps=video_fps,
            total_video_frames=_total_frames,
            source_width=video_w,
            source_height=video_h,
        )

        # ----- Connect WebSocket -----
        self._notify("generating", "Connecting to ComfyUI...", 0.35)
        self.connect()

        # ----- Queue job -----
        print("\n--- Submitting Job ---")
        prompt_id = self._queue_prompt(workflow)
        print(f"[JOB] Queued with prompt_id: {prompt_id}")
        self._notify("generating", "Job queued, generating...", 0.38)

        # ----- Monitor -----
        print("\n--- Monitoring Generation ---")
        self._monitor_execution(prompt_id)

        # ----- Retrieve output -----
        print("\n--- Retrieving Output ---")
        self._notify("retrieving", "Generation done, retrieving output...", 0.92)
        output_filename = self._retrieve_output(prompt_id, output_dir)

        if self.ws:
            self.ws.close()

        self._notify("complete", str(output_filename or "Done"), 1.0)
        return output_filename

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    # Maximum frames the model should process (Wan 2.2 Fun 5B on 8GB VRAM)
    MAX_GEN_FRAMES = 81

    # ------------------------------------------------------------------
    # Dynamic resolution helper
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_target_dims(width, height, max_short_side=480):
        """
        Calculate VAE-compatible target dimensions preserving aspect ratio.

        Logic:
          - Vertical video (w < h): target_w = max_short_side, scale h.
          - Horizontal video (w >= h): target_h = max_short_side, scale w.
          - Both dimensions rounded to the nearest multiple of 16 (VAE req).

        Example: 1080x1920 (YouTube Short) -> 480x848
        """
        aspect = width / height

        if width < height:
            # Vertical / Short
            target_w = max_short_side
            target_h = max_short_side / aspect
        else:
            # Horizontal / Landscape
            target_h = max_short_side
            target_w = max_short_side * aspect

        # Round to nearest multiple of 16 (required by VAE)
        target_w = round(target_w / 16) * 16
        target_h = round(target_h / 16) * 16

        return int(target_w), int(target_h)

    # ------------------------------------------------------------------
    # Workflow injection  (Fun Control topology)
    # ------------------------------------------------------------------

    def _inject_workflow(
        self, workflow, vid_meta, img_meta, mask_subfolder,
        skater_prompt, skateboard_prompt, negative_prompt,
        skater_mask_subfolder=None, pose_subfolder=None,
        video_fps=24.0,
        total_video_frames=0, source_width=0, source_height=0,
    ):
        """
        Dynamically inject file paths, prompts, dimensions, and FPS into
        the Fun Control workflow JSON.

        Fun Control architecture (Generate + Post-Composite):
          Wan22FunControlToVideo.control_video = DWPose skeleton frames
          for explicit pose guidance.  Text prompt drives appearance.

          Masks are used AFTER generation: ImageCompositeMasked
          blends the generated character onto the original background.
        """

        # --- Compute frame cap (align video + mask + pose loaders) ---
        frame_cap = self.MAX_GEN_FRAMES
        if 0 < total_video_frames < frame_cap:
            frame_cap = total_video_frames
        print(f"  [FRAMES] Capping all loaders to {frame_cap} frames "
              f"(source has {total_video_frames})")

        # --- Calculate target resolution ---
        if source_width > 0 and source_height > 0:
            target_w, target_h = self._calculate_target_dims(
                source_width, source_height
            )
        else:
            # Fallback for unknown dimensions
            target_w, target_h = 480, 848
        print(f"  [RESOLUTION] Source {source_width}x{source_height} -> "
              f"Target {target_w}x{target_h}")

        # --- Source Video Loader ---
        try:
            vid_node = find_node_safe(workflow, "VHS_LoadVideo", "Source Video")
            workflow[vid_node]["inputs"]["video"] = vid_meta["name"]
            workflow[vid_node]["inputs"]["force_rate"] = round(video_fps, 2)
            workflow[vid_node]["inputs"]["frame_load_cap"] = frame_cap
            workflow[vid_node]["inputs"]["force_size"] = "Custom"
            workflow[vid_node]["inputs"]["custom_width"] = target_w
            workflow[vid_node]["inputs"]["custom_height"] = target_h
            print(f"  [OK] Source video -> node {vid_node} "
                  f"(fps={video_fps:.1f}, cap={frame_cap}, "
                  f"{target_w}x{target_h})")
        except ValueError:
            print("  [SKIP] No VHS_LoadVideo node found")

        # --- Skateboard Mask Sequence Loader ---
        for title in ["Mask Sequence (Skateboard)", "Mask Sequence"]:
            try:
                mask_node = find_node_safe(workflow, "VHS_LoadImages", title)
                workflow[mask_node]["inputs"]["directory"] = mask_subfolder
                workflow[mask_node]["inputs"]["image_load_cap"] = frame_cap
                print(f"  [OK] Skateboard mask sequence -> node {mask_node} "
                      f"(cap={frame_cap})")
                break
            except ValueError:
                continue
        else:
            print("  [SKIP] No skateboard mask loader node found")

        # --- Skater Mask Sequence Loader ---
        if skater_mask_subfolder:
            try:
                skater_mask_node = find_node_safe(
                    workflow, "VHS_LoadImages", "Mask Sequence (Skater)"
                )
                workflow[skater_mask_node]["inputs"]["directory"] = (
                    skater_mask_subfolder
                )
                workflow[skater_mask_node]["inputs"]["image_load_cap"] = frame_cap
                print(f"  [OK] Skater mask sequence -> node "
                      f"{skater_mask_node} (cap={frame_cap})")
            except ValueError:
                print("  [SKIP] No skater mask loader node found in workflow")

        # --- Pose Sequence Loader (control_video for Fun Control) ---
        if pose_subfolder:
            try:
                pose_node = find_node_safe(
                    workflow, "VHS_LoadImages", "Pose Sequence (Skater)"
                )
                workflow[pose_node]["inputs"]["directory"] = pose_subfolder
                workflow[pose_node]["inputs"]["image_load_cap"] = frame_cap
                print(f"  [OK] Pose sequence -> node {pose_node} "
                      f"(cap={frame_cap})")
            except ValueError:
                print("  [SKIP] No pose loader node found in workflow")

        # --- Inject target dimensions into ALL ImageScale "Resize *" nodes ---
        for node_id, node_data in workflow.items():
            meta_title = node_data.get("_meta", {}).get("title", "")
            if (meta_title.startswith("Resize ")
                    and node_data.get("class_type") == "ImageScale"):
                node_data["inputs"]["width"] = target_w
                node_data["inputs"]["height"] = target_h
                print(f"  [OK] {meta_title} -> node {node_id} "
                      f"({target_w}x{target_h})")

        # --- Inject dims + length into Wan22FunControlToVideo node ---
        try:
            ctrl_node = find_node_safe(
                workflow, "Wan22FunControlToVideo", "Fun Control Conditioning"
            )
            workflow[ctrl_node]["inputs"]["width"] = target_w
            workflow[ctrl_node]["inputs"]["height"] = target_h
            workflow[ctrl_node]["inputs"]["length"] = frame_cap
            print(f"  [OK] Fun Control dims -> node {ctrl_node} "
                  f"({target_w}x{target_h}, length={frame_cap})")
        except ValueError:
            print("  [SKIP] No Wan22FunControlToVideo node found")

        # --- Reference Image (optional) ---
        if img_meta:
            try:
                img_node = find_node_safe(
                    workflow, "LoadImage", "Reference Image"
                )
                workflow[img_node]["inputs"]["image"] = img_meta["name"]
                print(f"  [OK] Reference image -> node {img_node}")
            except ValueError:
                print("  [SKIP] No reference image node found")

        # --- Unified Prompt (single positive prompt for Fun Control) ---
        # Combine skater + skateboard descriptions into one prompt
        prompt_parts = [p for p in [skater_prompt, skateboard_prompt]
                        if p and p.strip()]
        unified_prompt = ", ".join(prompt_parts) if prompt_parts else (
            "high quality video"
        )

        try:
            pos_node = find_node_safe(
                workflow, "CLIPTextEncode", "Positive Prompt"
            )
            workflow[pos_node]["inputs"]["text"] = unified_prompt
            print(f"  [OK] Unified prompt -> node {pos_node}")
        except ValueError:
            print("  [SKIP] No Positive Prompt node found")

        try:
            neg_node = find_node_safe(
                workflow, "CLIPTextEncode", "Negative Prompt"
            )
            workflow[neg_node]["inputs"]["text"] = negative_prompt
            print(f"  [OK] Negative prompt -> node {neg_node}")
        except ValueError:
            print("  [SKIP] No negative prompt node found")

        # --- Output Video FPS (apply to all VHS_VideoCombine nodes) ---
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "VHS_VideoCombine":
                node_data["inputs"]["frame_rate"] = round(video_fps, 2)
                meta_title = node_data.get("_meta", {}).get("title", "VHS_VideoCombine")
                print(f"  [OK] {meta_title} frame_rate -> node {node_id} "
                      f"({video_fps:.1f}fps)")

        # --- Random Seeds (set unique seed on every sampler node) ---
        for sampler_class in [
            "Wan22FunControlToVideo",
            "KSampler", "KSamplerAdvanced", "SamplerCustom",
        ]:
            for node_id, node_data in workflow.items():
                if node_data.get("class_type") == sampler_class:
                    if "seed" in node_data["inputs"]:
                        node_data["inputs"]["seed"] = random.randint(1, 10**9)
                        title = node_data.get("_meta", {}).get(
                            "title", sampler_class
                        )
                        print(f"  [OK] Random seed -> node {node_id} "
                              f"({title})")

    def _queue_prompt(self, workflow):
        """Submit the workflow to ComfyUI and return the prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        url = f"http://{self.server_addr}/prompt"
        resp = requests.post(url, json=payload, timeout=30)

        if resp.status_code != 200:
            raise RuntimeError(f"Failed to queue prompt: {resp.status_code} {resp.text}")

        result = resp.json()
        if "error" in result:
            raise RuntimeError(f"ComfyUI error: {result['error']}")

        return result["prompt_id"]

    def _monitor_execution(self, prompt_id):
        """Listen on WebSocket for progress and completion."""
        # Track progress floor — monotonically increasing so the UI never
        # jumps backward.  Works for single-pass Fun Control and legacy
        # multi-pass workflows.
        _progress_floor = 0.38
        while True:
            try:
                out = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                print("  [TIMEOUT] WebSocket timed out, checking status...")
                if self._check_history(prompt_id):
                    break
                continue

            if isinstance(out, str):
                msg = json.loads(out)
                msg_type = msg.get("type", "")
                data = msg.get("data", {})

                if msg_type == "progress":
                    value = data.get("value", 0)
                    maximum = data.get("max", 0)
                    if maximum > 0:
                        pct = value / maximum * 100
                        step_frac = value / maximum
                        # Compute candidate progress in the 0.38-0.92 range
                        candidate = 0.38 + step_frac * 0.54
                        # If candidate < floor, we're in a new pass →
                        # remap into the remaining range above the floor
                        if candidate < _progress_floor:
                            remaining = max(0.92 - _progress_floor, 0.01)
                            candidate = _progress_floor + step_frac * remaining
                        _overall_pct = min(candidate, 0.92)
                        # Update floor when a phase completes
                        if value == maximum and _overall_pct > _progress_floor:
                            _progress_floor = _overall_pct
                        print(f"  [PROGRESS] {value}/{maximum} ({pct:.0f}%)", end="\r")
                        self._notify("generating", f"Step {value}/{maximum} ({pct:.0f}%)", _overall_pct)

                elif msg_type == "executing":
                    node = data.get("node", "")
                    if node:
                        print(f"  [EXECUTING] Node: {node}              ")
                    elif data.get("prompt_id") == prompt_id:
                        # node is None -> execution finished
                        print("\n  [DONE] Generation complete.")
                        break

                elif msg_type == "execution_error":
                    error_msg = data.get("exception_message", "Unknown error")
                    print(f"\n  [ERROR] Execution failed: {error_msg}")
                    raise RuntimeError(f"ComfyUI execution error: {error_msg}")

                elif msg_type == "execution_success":
                    if data.get("prompt_id") == prompt_id:
                        print("\n  [DONE] Generation complete.")
                        break

    def _check_history(self, prompt_id):
        """Check if the job completed via history endpoint."""
        try:
            url = f"http://{self.server_addr}/history/{prompt_id}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                history = resp.json()
                return prompt_id in history
        except (requests.RequestException, json.JSONDecodeError):
            pass
        return False

    def _retrieve_output(self, prompt_id, output_dir):
        """
        Fetch ALL output videos from ComfyUI history and copy to output_dir.
        Returns the primary output (the composited video from "Output Video"
        node, or the first video if no match).
        """
        os.makedirs(output_dir, exist_ok=True)

        url = f"http://{self.server_addr}/history/{prompt_id}"
        resp = requests.get(url, timeout=30)
        history = resp.json()

        if prompt_id not in history:
            raise RuntimeError(f"No history found for prompt_id: {prompt_id}")

        outputs = history[prompt_id]["outputs"]
        prompt_info = history[prompt_id].get("prompt", [None, None, {}])
        # prompt_info[2] is the workflow dict if available
        workflow_snapshot = prompt_info[2] if len(prompt_info) > 2 else {}

        # Download ALL output files, track the primary (composited) one
        primary_output = None
        all_downloaded = []

        for node_id, node_output in outputs.items():
            for key in ["gifs", "videos", "images"]:
                if key in node_output:
                    items = node_output[key]
                    if not isinstance(items, list):
                        items = [items]
                    for item in items:
                        if isinstance(item, dict) and "filename" in item:
                            downloaded = self._download_file(item, output_dir)
                            all_downloaded.append((node_id, downloaded))

                            # Identify primary output by node title or
                            # filename prefix
                            node_meta = {}
                            if isinstance(workflow_snapshot, dict):
                                node_meta = workflow_snapshot.get(
                                    node_id, {}
                                ).get("_meta", {})
                            title = node_meta.get("title", "")
                            fname = item.get("filename", "")

                            if (title == "Output Video"
                                    or fname.startswith("skate_reskin")):
                                primary_output = downloaded

        if all_downloaded:
            print(f"[OUTPUT] Downloaded {len(all_downloaded)} output(s)")
            for nid, path in all_downloaded:
                print(f"  Node {nid}: {os.path.basename(path)}")

        if primary_output:
            return primary_output
        if all_downloaded:
            return all_downloaded[0][1]

        print("[WARN] No video output found in history. Check ComfyUI output folder.")
        return None

    def _download_file(self, file_info, output_dir):
        """Download a specific output file from ComfyUI."""
        filename = file_info["filename"]
        subfolder = file_info.get("subfolder", "")
        file_type = file_info.get("type", "output")

        params = {"filename": filename, "subfolder": subfolder, "type": file_type}
        url = f"http://{self.server_addr}/view"
        resp = requests.get(url, params=params, timeout=120)

        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download {filename}: {resp.status_code}")

        output_path = os.path.join(output_dir, filename)
        with open(output_path, "wb") as f:
            f.write(resp.content)

        print(f"[OUTPUT] Saved: {output_path} ({len(resp.content) / 1024:.1f} KB)")
        return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate reskinned video via headless ComfyUI (Fun Control)"
    )
    parser.add_argument("--source-video", required=True, help="Source video path")
    parser.add_argument("--masks-dir", required=True, help="Directory of skateboard mask PNGs")
    parser.add_argument("--skater-masks-dir", default=None,
                        help="Directory of skater mask PNGs (combined with skateboard masks for white-out inpainting)")
    parser.add_argument("--poses-dir", required=True, help="Directory of pose PNGs")
    parser.add_argument(
        "--positive-prompt", default=None,
        help="Unified positive prompt for the entire scene "
             "(e.g., 'cyberpunk samurai riding a neon hoverboard'). "
             "Overrides --skater-prompt / --board-prompt if provided."
    )
    parser.add_argument(
        "--skater-prompt", default="",
        help="Text prompt for the skater/character "
             "(combined with --board-prompt into a unified prompt)"
    )
    parser.add_argument(
        "--board-prompt", default="",
        help="Text prompt for the skateboard/board "
             "(combined with --skater-prompt into a unified prompt)"
    )
    parser.add_argument(
        "--negative-prompt", default="blurry, distorted, low quality, deformed",
        help="Negative prompt"
    )
    parser.add_argument("--ref-image", default=None, help="Optional reference image for style")
    parser.add_argument(
        "--workflow", default="./workflows/fun_control_template.json",
        help="ComfyUI workflow template JSON"
    )
    parser.add_argument("--server", default="127.0.0.1:8188", help="ComfyUI server address")
    parser.add_argument("--output-dir", default="./output/generated", help="Output directory")
    parser.add_argument(
        "--check", action="store_true",
        help="Only check if ComfyUI server is reachable"
    )
    args = parser.parse_args()

    # Build unified prompt from CLI args
    if args.positive_prompt:
        unified_skater = args.positive_prompt
        unified_board = ""
    else:
        unified_skater = args.skater_prompt
        unified_board = args.board_prompt
        if not unified_skater and not unified_board:
            parser.error(
                "Provide --positive-prompt or at least one of "
                "--skater-prompt / --board-prompt."
            )

    orchestrator = ComfyOrchestrator(server_addr=args.server)

    # ------------------------------------------------------------------
    # Server check mode
    # ------------------------------------------------------------------
    if args.check:
        if orchestrator.check_server():
            print("[OK] ComfyUI server is reachable.")
            return 0
        else:
            print(f"[FAIL] Cannot reach ComfyUI at {args.server}")
            print("  Start ComfyUI with: python main.py --listen --lowvram")
            return 1

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    for label, path in [
        ("Source video", args.source_video),
        ("Masks directory", args.masks_dir),
        ("Poses directory", args.poses_dir),
        ("Workflow template", args.workflow),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}")
            sys.exit(1)

    if not orchestrator.check_server():
        print(f"[ERROR] ComfyUI server not reachable at {args.server}")
        print("  Start with: python main.py --listen --lowvram")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    # Display the unified prompt
    if unified_board:
        display_prompt = f"{unified_skater}, {unified_board}"
    else:
        display_prompt = unified_skater

    print("\n" + "=" * 60)
    print("  GENERATIVE RESKINNING (Fun Control)")
    print("=" * 60)
    print(f"  Prompt  : {display_prompt}")
    print(f"  Negative: {args.negative_prompt}")
    print(f"  Server  : {args.server}")
    print()

    output_file = orchestrator.execute_v2v(
        workflow_path=args.workflow,
        source_video_path=args.source_video,
        masks_dir=args.masks_dir,
        poses_dir=args.poses_dir,
        skater_prompt=unified_skater,
        skateboard_prompt=unified_board,
        negative_prompt=args.negative_prompt,
        output_dir=args.output_dir,
        ref_image_path=args.ref_image,
        skater_masks_dir=args.skater_masks_dir,
    )

    print("\n" + "=" * 60)
    print("  GENERATION COMPLETE")
    print("=" * 60)
    if output_file:
        print(f"  Output: {output_file}")
    else:
        print("  [WARN] No output file retrieved. Check ComfyUI output folder.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
