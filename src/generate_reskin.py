"""
Generative Reskinning Engine: ComfyUI API client for V2V generation.
Uploads assets, mutates workflow JSON, and executes via HTTP/WebSocket.
"""

import json
import os
import random
import uuid
import urllib.request

import requests

# Optional websocket - install websocket-client for progress tracking
try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


def find_node_by_class(workflow, class_type, index=0):
    """
    Locates the Node ID for a specific class type (e.g., 'VHS_LoadVideo').
    Returns the node ID string.
    """
    matches = []
    for node_id, node_data in workflow.items():
        if node_data.get("class_type") == class_type:
            matches.append(node_id)
    if not matches:
        raise ValueError(f"Node class '{class_type}' not found in workflow.")
    if index >= len(matches):
        raise ValueError(f"Index {index} out of range for class '{class_type}'.")
    return matches[index]


def upload_asset(server_address, file_path, asset_type="input"):
    """
    Uploads a video or image to the ComfyUI server.
    """
    url = f"http://{server_address}/upload/image"
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        files = {"image": (filename, f, "multipart/form-data")}
        data = {"type": asset_type, "overwrite": "true"}
        response = requests.post(url, files=files, data=data)
    if response.status_code != 200:
        raise Exception(f"Upload failed: {response.text}")
    return response.json()


def upload_image_sequence(server_address, dir_path, asset_type="input"):
    """
    ComfyUI may expect image sequences. For mask/pose dirs, we upload as batch.
    Returns list of uploaded filenames for workflow injection.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".png")])
    uploaded = []
    for f in files[:60]:  # Cap for T4
        path = os.path.join(dir_path, f)
        meta = upload_asset(server_address, path, asset_type)
        uploaded.append(meta.get("name", f))
    return uploaded


class ComfyOrchestrator:
    """
    Orchestrates ComfyUI workflow execution via HTTP and WebSocket.
    """

    def __init__(self, server_addr="127.0.0.1:8188"):
        self.server_addr = server_addr
        self.client_id = str(uuid.uuid4())
        self.ws = None
        if HAS_WEBSOCKET:
            try:
                self.ws = websocket.WebSocket()
                self.ws.connect(f"ws://{self.server_addr}/ws?clientId={self.client_id}")
            except Exception as e:
                print(f"WebSocket connect failed (progress tracking disabled): {e}")
                self.ws = None

    def execute_v2v(
        self,
        workflow_template,
        source_video_path,
        mask_dir,
        pose_dir,
        positive_prompt,
        negative_prompt="blur, distortion, artifacts",
        ref_image_path=None,
        frame_load_cap=50,
    ):
        """
        Execute V2V workflow with mask and pose conditioning.

        Args:
            workflow_template: ComfyUI workflow dict (from JSON)
            source_video_path: Path to source video
            mask_dir: Directory with mask PNG sequence
            pose_dir: Directory with pose PNG sequence
            positive_prompt: e.g. "cyberpunk drone, samurai"
            negative_prompt: Negative prompt
            ref_image_path: Optional reference image for object style
            frame_load_cap: Max frames (50 for T4)

        Returns:
            Output filename or path from ComfyUI history
        """
        workflow = json.loads(json.dumps(workflow_template))  # Deep copy

        # 1. Upload assets
        vid_meta = upload_asset(self.server_addr, source_video_path)
        if ref_image_path and os.path.exists(ref_image_path):
            img_meta = upload_asset(self.server_addr, ref_image_path)
        else:
            img_meta = None

        # 2. Mutate workflow - find nodes by class_type
        node_classes = [
            "VHS_LoadVideo",
            "LoadImage",
            "LoadImagesFromDirectory",
            "LoadImagesFromDirectory",
            "CLIPTextEncode",
            "CLIPTextEncode",
        ]
        try:
            vid_loader_id = find_node_by_class(workflow, "VHS_LoadVideo")
            workflow[vid_loader_id]["inputs"]["video"] = vid_meta["name"]
            if "frame_load_cap" in workflow[vid_loader_id].get("inputs", {}):
                workflow[vid_loader_id]["inputs"]["frame_load_cap"] = frame_load_cap
        except ValueError:
            pass

        try:
            img_loader_id = find_node_by_class(workflow, "LoadImage")
            if img_meta:
                workflow[img_loader_id]["inputs"]["image"] = img_meta["name"]
        except ValueError:
            pass

        # Mask and pose dir loaders (if workflow uses LoadImagesFromDirectory)
        try:
            mask_loader_ids = [
                n for n, d in workflow.items()
                if d.get("class_type") == "LoadImagesFromDirectory"
            ]
            if mask_loader_ids and mask_dir:
                workflow[mask_loader_ids[0]]["inputs"]["directory"] = mask_dir
            if len(mask_loader_ids) > 1 and pose_dir:
                workflow[mask_loader_ids[1]]["inputs"]["directory"] = pose_dir
        except (ValueError, IndexError):
            pass

        # Prompts
        try:
            text_nodes = [n for n, d in workflow.items() if d.get("class_type") == "CLIPTextEncode"]
            if len(text_nodes) >= 1:
                workflow[text_nodes[0]]["inputs"]["text"] = positive_prompt
            if len(text_nodes) >= 2:
                workflow[text_nodes[1]]["inputs"]["text"] = negative_prompt
        except (ValueError, IndexError):
            pass

        # Sampler seed
        for nid, ndata in workflow.items():
            if "seed" in ndata.get("inputs", {}):
                ndata["inputs"]["seed"] = random.randint(1, 10**9)
                break

        # 3. Queue execution
        p = {"prompt": workflow, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(
            f"http://{self.server_addr}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req).read())
        prompt_id = resp["prompt_id"]
        print(f"Queued prompt_id: {prompt_id}")

        # 4. Wait for completion (poll or WebSocket)
        if self.ws:
            while True:
                out = self.ws.recv()
                if isinstance(out, str):
                    msg = json.loads(out)
                    if msg.get("type") == "execution_success" and msg.get("data", {}).get(
                        "prompt_id"
                    ) == prompt_id:
                        print("Generation complete.")
                        break
                    if msg.get("type") == "execution_error":
                        print("Execution error:", msg)
                        raise RuntimeError(msg.get("data", {}).get("exception_message", "Unknown error"))
        else:
            # Poll history
            import time
            for _ in range(600):  # 10 min timeout
                time.sleep(1)
                try:
                    with urllib.request.urlopen(
                        f"http://{self.server_addr}/history/{prompt_id}"
                    ) as r:
                        hist = json.loads(r.read())
                    if prompt_id in hist:
                        break
                except Exception:
                    pass

        # 5. Retrieve output
        with urllib.request.urlopen(
            f"http://{self.server_addr}/history/{prompt_id}"
        ) as response:
            history = json.loads(response.read())

        outputs = history.get(prompt_id, {}).get("outputs", {})
        for node_id, out in outputs.items():
            if "gifs" in out:
                return out["gifs"][0].get("filename")
            if "videos" in out:
                return out["videos"][0].get("filename")
            if "images" in out:
                return out["images"][0].get("filename")

        return None


def run_generation(
    server_addr="127.0.0.1:8188",
    workflow_path=None,
    source_video_path=None,
    mask_dir=None,
    pose_dir=None,
    prompt="cyberpunk drone, samurai",
    negative_prompt="blur, distortion",
    ref_image_path=None,
    frame_load_cap=50,
):
    """
    Convenience function to run generation from Colab/script.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if workflow_path is None:
        workflow_path = os.path.join(base_dir, "workflows", "vace_template.json")

    with open(workflow_path) as f:
        workflow = json.load(f)

    orch = ComfyOrchestrator(server_addr)
    return orch.execute_v2v(
        workflow,
        source_video_path,
        mask_dir,
        pose_dir,
        prompt,
        negative_prompt,
        ref_image_path,
        frame_load_cap,
    )
