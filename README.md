# Skate Physics Preserver

**Human-Object Relational Mapping for Video-to-Video Synthesis**

A 100% local pipeline that reskins dynamic human-object interactions (e.g., skateboard tricks) while enforcing strict frame-by-frame physical boundaries. No cloud APIs.

All three stages -- extraction, generation, and validation -- run through a single **local web UI** at `localhost:5000`. Upload a video, review auto-detected masks, approve, enter a creative prompt, generate, validate IoU, and inspect results. Everything is button-driven. A headless CLI is also available for scripted workflows.

Optimized for **RTX 3070 8GB VRAM**.

---

## Architecture

```
input.mp4  (or YouTube URL)
    |
    v
+----------------------------------------------+
|  app.py — Web UI (localhost:5000)            |
|  All 3 stages, button-driven                 |
|                                              |
|  Stage 1: Extract Physics                    |
|    YOLO detect -> SAM 2.1 segment            |
|    -> approve -> DWPose + SAM propagation    |
|                                              |
|  Stage 2: Generate Reskin                    |
|    Enter prompts -> ComfyUI API              |
|    -> live WebSocket progress                |
|                                              |
|  Stage 3: Validate IoU                       |
|    Reverse-track with SAM 2.1                |
|    -> frame-by-frame IoU chart + report      |
|                                              |
|  Results: comparison viewer                  |
|    Original vs generated, pose/mask/overlay  |
+----------------------------------------------+
    |                    |                |
    v                    v                v
output/              output/          output/
  frames_original/     generated/       iou_report.json
  pose_skater/           output.mp4
  mask_skateboard/
  mask_skater/
```

---

## Quick Start (Docker)

Docker eliminates every dependency issue. One build, zero debugging.

> **PowerShell users:** All commands below use backtick (`` ` ``) for line continuation, not backslash (`\`). Backslash will cause parse errors in PowerShell. Bash equivalents are in collapsible sections where needed.

### Prerequisites

| Requirement | Check with | Install |
|---|---|---|
| Docker Desktop (WSL2 backend) | `docker --version` | [docker.com/desktop](https://www.docker.com/products/docker-desktop/) |
| NVIDIA GPU Driver 525+ | `nvidia-smi` | [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx) |
| NVIDIA Container Toolkit | `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` | [see below](#nvidia-container-toolkit) |
| ~15 GB free disk | for the Docker image | |

### 1. Build

```bash
cd skate-physics-preserver
docker compose build          # ~10-15 min first time, cached after
```

### 2. Download SAM2 checkpoint

**PowerShell:**

```powershell
New-Item -ItemType Directory -Force checkpoints
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" `
  -OutFile "checkpoints/sam2.1_hiera_small.pt"
```

<details>
<summary>Bash / Git Bash</summary>

```bash
mkdir checkpoints
curl -L -o checkpoints/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```
</details>

DWPose ONNX models auto-download on first run (~200 MB, cached).

### 3. Verify

```bash
docker compose run --rm pipeline src/extract_physics.py --check
```

### 4. Launch the Web UI

**PowerShell:**

```powershell
docker compose run --rm -p 5000:5000 pipeline src/app.py `
  --output /data/output `
  --sam-checkpoint /data/checkpoints/sam2.1_hiera_small.pt
```

<details>
<summary>Bash / Git Bash</summary>

```bash
docker compose run --rm -p 5000:5000 pipeline src/app.py \
  --output /data/output \
  --sam-checkpoint /data/checkpoints/sam2.1_hiera_small.pt
```
</details>

Open **http://localhost:5000** in your browser. The UI walks you through six steps:

| Step | What happens |
|------|-------------|
| **1. Upload** | Drop a video file or paste a YouTube URL |
| **2. Detect** | YOLO finds skater + skateboard, SAM 2.1 segments them on frame 0. Manual click fallback if needed. |
| **3. Extract** | Approve masks, DWPose skeletons + SAM mask propagation run with live progress |
| **4. Generate** | Enter positive/negative prompts, check ComfyUI connection, click Generate |
| **5. Validate** | Reverse-track the generated video, view per-frame IoU chart and pass/fail result |
| **6. Results** | Side-by-side original vs generated video, frame-by-frame pose/mask/overlay scrubber, IoU report |

> Stages 2 and 3 are optional -- skip either with the UI buttons and go straight to results.

You can also pre-load a video from the command line:

**PowerShell:**

```powershell
docker compose run --rm -p 5000:5000 pipeline src/app.py `
  --video /data/input/skate_clip.mp4 `
  --output /data/output `
  --sam-checkpoint /data/checkpoints/sam2.1_hiera_small.pt
```

<details>
<summary>Bash / Git Bash</summary>

```bash
docker compose run --rm -p 5000:5000 pipeline src/app.py \
  --video /data/input/skate_clip.mp4 \
  --output /data/output \
  --sam-checkpoint /data/checkpoints/sam2.1_hiera_small.pt
```
</details>

YouTube URLs work too: `--video "https://www.youtube.com/watch?v=VIDEO_ID"`

### ComfyUI requirement for Stage 2

The Generate step talks to a ComfyUI server. Start it on your host machine **before** clicking Generate:

```powershell
cd C:\path\to\ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --lowvram
```

The web UI's "Check Connection" button verifies connectivity before you start. Default server address is `127.0.0.1:8188` (change it in the UI if needed).

> On Docker for Windows, the UI resolves `host.docker.internal` automatically. On Linux, use your host's LAN IP or add `--network host`.

---

## Native Setup (Without Docker)

### 1. Environment

**PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 `
  --index-url https://download.pytorch.org/whl/cu124

$env:SAM2_BUILD_CUDA=0
pip install git+https://github.com/facebookresearch/sam2.git@main

pip install rtmlib==0.0.15
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.20.1

pip install opencv-python==4.11.0.86 "av>=12.0.0" websocket-client `
  requests tqdm matplotlib "numpy<2.0" hydra-core==1.3.2 yt-dlp `
  "ultralytics>=8.3.0" "flask>=3.0.0"
```

<details>
<summary>Bash / Linux / Mac</summary>

```bash
python -m venv venv
source venv/bin/activate

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

export SAM2_BUILD_CUDA=0
pip install git+https://github.com/facebookresearch/sam2.git@main

pip install rtmlib==0.0.15
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.20.1

pip install opencv-python==4.11.0.86 "av>=12.0.0" websocket-client \
  requests tqdm matplotlib "numpy<2.0" hydra-core==1.3.2 yt-dlp \
  "ultralytics>=8.3.0" "flask>=3.0.0"
```
</details>

> **Note:** `$env:SAM2_BUILD_CUDA=0` is needed on Windows if you don't have `nvcc` in PATH.
> YOLOv8-nano weights (~6 MB) are auto-downloaded on first run.

### 2. Download checkpoint

**PowerShell:**

```powershell
New-Item -ItemType Directory -Force checkpoints
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" `
  -OutFile "checkpoints/sam2.1_hiera_small.pt"
```

<details>
<summary>Bash / Git Bash</summary>

```bash
mkdir checkpoints
curl -L -o checkpoints/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```
</details>

### 3. Verify

```powershell
python src/extract_physics.py --check
```

### 4. Launch the Web UI

```powershell
# Launch the web UI
python src/app.py --output output/

# Or pre-load a video
python src/app.py --video input.mp4 --output output/

# Or with a YouTube URL
python src/app.py --video "https://www.youtube.com/watch?v=VIDEO_ID" --output output/
```

Open **http://localhost:5000** -- same six-step UI as Docker.

---

## Headless CLI (for scripting)

All three stages can also be run from the command line without the web UI.

**Stage 1 -- Extract:**

```powershell
python src/extract_physics.py `
  --video input.mp4 `
  --output output/ `
  --bbox "120,340,280,410"        # optional, auto-detects if omitted
```

**Stage 2 -- Generate** (ComfyUI must be running):

```powershell
python src/generate_reskin.py `
  --source-video input.mp4 `
  --masks-dir output/mask_skateboard `
  --poses-dir output/pose_skater `
  --positive-prompt "cyberpunk samurai riding a neon hoverboard, cinematic" `
  --output-dir output/generated
```

**Stage 3 -- Validate:**

```powershell
python src/evaluate_iou.py `
  --metadata output/tracking_metadata.json `
  --generated output/generated/output.mp4
```

<details>
<summary>Bash equivalents (replace backtick with backslash)</summary>

In Bash, use `\` instead of `` ` `` for line continuation. Everything else is the same.
</details>

**Output structure:**

```
output/
  frames_original/         # original frames (JPEG)
  pose_skater/             # skeleton overlays (PNG)
  pose_json/               # per-frame keypoint JSON
  mask_skateboard/         # skateboard masks (PNG)
  mask_skater/             # skater masks (PNG)
  tracking_metadata.json
  generated/               # Stage 2 output
    output.mp4
  iou_report.json          # Stage 3 report
```

---

## ComfyUI Setup

Stage 2 (generation) requires a running ComfyUI server with specific custom nodes and models.

### Custom Nodes

Install in ComfyUI's `custom_nodes/` directory:

```bash
cd custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
git clone https://github.com/city96/ComfyUI-GGUF
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper
```

### Models (for 8GB VRAM)

| Model | Path in ComfyUI | Size |
|-------|------|------|
| Wan 2.1 VACE 1.3B GGUF Q8 | `models/unet/wan2.1_vace_1.3B_Q8_0.gguf` | ~1.3 GB |
| UMT5-XXL FP8 | `models/clip/umt5_xxl_fp8_e4m3fn.safetensors` | ~5 GB |
| Wan 2.1 VAE | `models/vae/wan_2.1_vae.safetensors` | ~200 MB |

### Custom Workflow

The template at `workflows/vace_template.json` is the default. To use your own:

1. Create your workflow in ComfyUI's GUI
2. Settings > Dev Mode > "Save (API Format)"
3. Replace `workflows/vace_template.json`
4. Give nodes descriptive `_meta.title` values for reliable injection

The script resolves nodes by `_meta.title` first, then falls back to `class_type`.

### Launch ComfyUI

```powershell
python main.py --listen 0.0.0.0 --port 8188 --lowvram
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3070 8GB | RTX 3090 24GB |
| RAM | 16 GB | 32 GB |
| CUDA Driver | 525+ | 570+ |
| Docker | 24+ (compose v2) | latest |
| OS | Windows 10/11, Linux | Windows 11 / Ubuntu 22.04 |

### 8GB VRAM Strategy

- **SAM 2.1 Hiera-Small** instead of Large (~3 GB vs ~12 GB)
- **DWPose balanced** mode instead of performance
- **Sequential processing** with full VRAM cleanup between passes
- **CPU offloading** for SAM2 frame embeddings and tracking state
- **Wan 2.1 VACE 1.3B GGUF Q8** for generation (~3 GB model)
- **ComfyUI `--lowvram`** flag for aggressive model offloading

---

## Project Structure

```
skate-physics-preserver/
+-- Dockerfile
+-- docker-compose.yml
+-- README.md
+-- requirements.txt
+-- configs/
|   +-- sam2.1/sam2.1_hiera_s.yaml
+-- checkpoints/                    # model weights (host-mounted)
+-- input/                          # source videos (host-mounted)
+-- output/                         # pipeline outputs (host-mounted)
+-- workflows/
|   +-- vace_template.json          # ComfyUI workflow template
+-- src/
    +-- app.py                      # Web UI (all 3 stages)
    +-- auto_detect.py              # YOLO + SAM auto-detection
    +-- extract_physics.py          # CLI: Stage 1 extraction
    +-- generate_reskin.py          # CLI: Stage 2 generation
    +-- evaluate_iou.py             # CLI: Stage 3 validation
    +-- templates/
    |   +-- index.html              # Single-page web app
    +-- tracking/
        +-- skateboard_tracker.py   # SAM 2.1 multi-object wrapper
        +-- skater_pose.py          # DWPose wrapper
```

---

## Troubleshooting

### Docker

**Can't connect to the web UI**
> Open **http://localhost:5000**, not the container IP. Verify the container started (look for the banner in the terminal).

**PowerShell line continuation errors**
> PowerShell uses backtick (`` ` ``) not backslash. Or put the entire command on one line.

**"no matching manifest for windows/amd64"**
> Switch Docker Desktop to Linux containers: right-click tray icon > "Switch to Linux containers".

**GPU not detected in container**
> 1. `nvidia-smi` must work on the host
> 2. Docker Desktop > Settings > Resources > WSL Integration > enable your distro
> 3. Test: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

**"CUDA out of memory"**
> Close other GPU apps. The container shares VRAM with the host.

**"host.docker.internal" not resolving (Linux)**
> Add `--add-host=host.docker.internal:host-gateway` or use `--network host`.

<a id="nvidia-container-toolkit"></a>
**NVIDIA Container Toolkit (Linux)**

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Native Install

**CUDA Out of Memory** -- use `--pose-mode balanced` or `--pose-mode lightweight`, reduce video resolution, or set `SAM2_BUILD_CUDA=0`.

**SAM2 Config Not Found** -- verify with `python -c "import sam2; print(sam2.__file__)"` and check that `configs/sam2.1/sam2.1_hiera_s.yaml` exists in the package directory.

**DWPose Model Download Fails** -- download manually from [rtmlib releases](https://github.com/Tau-J/rtmlib/releases) and place in `~/.cache/rtmlib/`.

**ComfyUI Connection Refused** -- run `python src/generate_reskin.py --check` to test, then start ComfyUI with `python main.py --listen 0.0.0.0 --port 8188 --lowvram`.

---

## License

Research/educational use. See individual model licenses:
- SAM 2.1: Apache 2.0
- DWPose/rtmlib: Apache 2.0
- Wan 2.1: Apache 2.0
