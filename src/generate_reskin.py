#!/usr/bin/env python3
"""
generate_reskin.py - Headless ComfyUI Generative Orchestrator
==============================================================
Usage:
  python src/generate_reskin.py \\
    --source-video input.mp4 \\
    --masks-dir output/mask_skateboard \\
    --poses-dir output/pose_skater \\
    --positive-prompt "cyberpunk samurai riding a neon hoverboard" \\
    --negative-prompt "blurry, distorted, low quality" \\
    --output-dir output/generated

Workflow:
  1. Uploads source video + reference assets to ComfyUI
  2. Loads & mutates a template workflow JSON
  3. Injects prompts, file paths, and random seed
  4. Submits to ComfyUI headless server via HTTP POST
  5. Monitors progress via WebSocket
  6. Downloads the generated video on completion

Requirements:
  - ComfyUI server running locally: python main.py --listen --lowvram
  - Custom nodes: VideoHelperSuite, ComfyUI-GGUF, ComfyUI-WanVideoWrapper
  - Models: Wan 2.1 VACE GGUF (1.3B Q8 for 8GB VRAM)
"""

import argparse
import json
import os
import random
import sys
import uuid

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


def upload_image_sequence(server_address, images_dir, subfolder_name):
    """
    Upload an entire directory of PNGs as an image sequence.
    ComfyUI stores them in input/{subfolder_name}/.
    """
    files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg"))],
    )
    if not files:
        raise FileNotFoundError(f"No image files found in {images_dir}")

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
        positive_prompt,
        negative_prompt,
        output_dir,
        ref_image_path=None,
    ):
        """
        Execute the full V2V pipeline via ComfyUI API.

        Args:
            workflow_path:     Path to template JSON.
            source_video_path: Source MP4.
            masks_dir:         Directory of mask PNGs.
            poses_dir:         Directory of pose PNGs.
            positive_prompt:   Text prompt for generation.
            negative_prompt:   Negative prompt.
            output_dir:        Where to copy the result.
            ref_image_path:    Optional reference image for style guidance.
        """
        # ----- Load workflow template -----
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        print(f"[WORKFLOW] Loaded template: {workflow_path}")
        self._notify("uploading", "Workflow loaded, uploading assets...", 0.05)

        # ----- Upload assets -----
        print("\n--- Uploading Assets ---")
        self._notify("uploading", "Uploading source video...", 0.08)
        vid_meta = upload_asset(self.server_addr, source_video_path)

        self._notify("uploading", "Uploading mask sequence...", 0.15)
        mask_subfolder, _ = upload_image_sequence(
            self.server_addr, masks_dir, "mask_skateboard"
        )

        self._notify("uploading", "Uploading pose sequence...", 0.25)
        pose_subfolder, _ = upload_image_sequence(
            self.server_addr, poses_dir, "pose_skater"
        )

        img_meta = None
        if ref_image_path and os.path.isfile(ref_image_path):
            img_meta = upload_asset(self.server_addr, ref_image_path)

        # ----- Inject values into workflow -----
        print("\n--- Injecting Parameters ---")
        self._notify("injecting", "Configuring workflow parameters", 0.32)
        self._inject_workflow(
            workflow,
            vid_meta=vid_meta,
            img_meta=img_meta,
            mask_subfolder=mask_subfolder,
            pose_subfolder=pose_subfolder,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
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

    def _inject_workflow(
        self, workflow, vid_meta, img_meta, mask_subfolder, pose_subfolder,
        positive_prompt, negative_prompt
    ):
        """Dynamically inject file paths and prompts into the workflow JSON."""

        # --- Source Video Loader ---
        try:
            vid_node = find_node_safe(workflow, "VHS_LoadVideo", "Source Video")
            workflow[vid_node]["inputs"]["video"] = vid_meta["name"]
            print(f"  [OK] Source video -> node {vid_node}")
        except ValueError:
            print("  [SKIP] No VHS_LoadVideo node found")

        # --- Mask Sequence Loader ---
        try:
            mask_node = find_node_safe(workflow, "VHS_LoadImages", "Mask Sequence")
            workflow[mask_node]["inputs"]["directory"] = mask_subfolder
            print(f"  [OK] Mask sequence -> node {mask_node}")
        except ValueError:
            # Try LoadImageSequence as alternative
            try:
                mask_node = find_node_safe(workflow, "LoadImageSequence", "Mask Sequence")
                workflow[mask_node]["inputs"]["directory"] = mask_subfolder
                print(f"  [OK] Mask sequence -> node {mask_node}")
            except ValueError:
                print("  [SKIP] No mask loader node found")

        # --- Pose Sequence Loader ---
        try:
            pose_node = find_node_safe(workflow, "VHS_LoadImages", "Pose Sequence")
            workflow[pose_node]["inputs"]["directory"] = pose_subfolder
            print(f"  [OK] Pose sequence -> node {pose_node}")
        except ValueError:
            try:
                pose_node = find_node_safe(workflow, "LoadImageSequence", "Pose Sequence")
                workflow[pose_node]["inputs"]["directory"] = pose_subfolder
                print(f"  [OK] Pose sequence -> node {pose_node}")
            except ValueError:
                print("  [SKIP] No pose loader node found")

        # --- Reference Image (optional) ---
        if img_meta:
            try:
                img_node = find_node_safe(workflow, "LoadImage", "Reference Image")
                workflow[img_node]["inputs"]["image"] = img_meta["name"]
                print(f"  [OK] Reference image -> node {img_node}")
            except ValueError:
                print("  [SKIP] No reference image node found")

        # --- Text Prompts ---
        try:
            pos_node = find_node_safe(workflow, "CLIPTextEncode", "Positive Prompt", index=0)
            workflow[pos_node]["inputs"]["text"] = positive_prompt
            print(f"  [OK] Positive prompt -> node {pos_node}")
        except ValueError:
            print("  [SKIP] No positive prompt node found")

        try:
            neg_node = find_node_safe(workflow, "CLIPTextEncode", "Negative Prompt", index=1)
            workflow[neg_node]["inputs"]["text"] = negative_prompt
            print(f"  [OK] Negative prompt -> node {neg_node}")
        except ValueError:
            print("  [SKIP] No negative prompt node found")

        # --- Random Seed ---
        for sampler_class in ["KSampler", "WanVACESampler", "KSamplerAdvanced", "SamplerCustom"]:
            try:
                sampler_node = find_node_by_class(workflow, sampler_class)
                if "seed" in workflow[sampler_node]["inputs"]:
                    workflow[sampler_node]["inputs"]["seed"] = random.randint(1, 10**9)
                    print(f"  [OK] Random seed -> node {sampler_node} ({sampler_class})")
                break
            except ValueError:
                continue

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
                        print(f"  [PROGRESS] {value}/{maximum} ({pct:.0f}%)", end="\r")
                        self._notify("generating", f"Step {value}/{maximum} ({pct:.0f}%)", 0.38 + (value / maximum) * 0.54)

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
        Fetch the output video from ComfyUI history and copy to output_dir.
        """
        os.makedirs(output_dir, exist_ok=True)

        url = f"http://{self.server_addr}/history/{prompt_id}"
        resp = requests.get(url, timeout=30)
        history = resp.json()

        if prompt_id not in history:
            raise RuntimeError(f"No history found for prompt_id: {prompt_id}")

        outputs = history[prompt_id]["outputs"]

        # Search for video output (VHS_VideoCombine typically outputs 'gifs')
        for _node_id, node_output in outputs.items():
            for key in ["gifs", "videos", "images"]:
                if key in node_output:
                    items = node_output[key]
                    if not isinstance(items, list):
                        items = [items]
                    for item in items:
                        if isinstance(item, dict) and "filename" in item:
                            return self._download_file(item, output_dir)

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
        description="Generate reskinned video via headless ComfyUI"
    )
    parser.add_argument("--source-video", required=True, help="Source video path")
    parser.add_argument("--masks-dir", required=True, help="Directory of mask PNGs")
    parser.add_argument("--poses-dir", required=True, help="Directory of pose PNGs")
    parser.add_argument(
        "--positive-prompt", "--prompt", required=True,
        help="Positive text prompt (e.g., 'cyberpunk samurai on a neon hoverboard')"
    )
    parser.add_argument(
        "--negative-prompt", default="blurry, distorted, low quality, deformed",
        help="Negative prompt"
    )
    parser.add_argument("--ref-image", default=None, help="Optional reference image for style")
    parser.add_argument(
        "--workflow", default="./workflows/vace_template.json",
        help="ComfyUI workflow template JSON"
    )
    parser.add_argument("--server", default="127.0.0.1:8188", help="ComfyUI server address")
    parser.add_argument("--output-dir", default="./output/generated", help="Output directory")
    parser.add_argument(
        "--check", action="store_true",
        help="Only check if ComfyUI server is reachable"
    )
    args = parser.parse_args()

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
    print("\n" + "=" * 60)
    print("  GENERATIVE RESKINNING")
    print("=" * 60)
    print(f"  Prompt (+): {args.positive_prompt}")
    print(f"  Prompt (-): {args.negative_prompt}")
    print(f"  Server    : {args.server}")
    print()

    output_file = orchestrator.execute_v2v(
        workflow_path=args.workflow,
        source_video_path=args.source_video,
        masks_dir=args.masks_dir,
        poses_dir=args.poses_dir,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        output_dir=args.output_dir,
        ref_image_path=args.ref_image,
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
