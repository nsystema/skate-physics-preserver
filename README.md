# Skate Physics Preserver

**Human-Object Relational Mapping for Video-to-Video Synthesis**

A 100% local, headless pipeline that reskins dynamic human-object interactions (e.g., skateboard tricks) while enforcing strict frame-by-frame physical boundaries. No cloud APIs. No GUI.

Optimized for **RTX 3070 8GB VRAM**.

---

## Architecture

```
input.mp4
    |
    v
+------------------------------+
|  extract_physics.py          |
|  +- Pass 1: DWPose (~1.5GB) |
|  |  (skeleton extraction)    |
|  |  > VRAM cleanup           |
|  +- Pass 2: SAM 2.1 (~3GB)  |
|     (object mask tracking)   |
+------------------------------+
    |                 |
    v                 v
pose_skater/    mask_skateboard/
(PNG sequence)  (PNG sequence)
    |                 |
    v                 v
+------------------------------+
|  generate_reskin.py          |
|  (ComfyUI headless API)     |
|  Wan 2.1 VACE 1.3B GGUF     |
+------------------------------+
    |
    v
output.mp4
    |
    v
+------------------------------+
|  evaluate_iou.py             |
|  (Reverse-tracking + IoU)    |
|  Target: IoU > 0.90          |
+------------------------------+
```

---

## Docker Setup (Recommended)

Docker eliminates every dependency issue. One build, zero debugging.

### Prerequisites

| Requirement | How to check | Install link |
|---|---|---|
| Docker Desktop (WSL2 backend) | `docker --version` | [docker.com/desktop](https://www.docker.com/products/docker-desktop/) |
| NVIDIA GPU Driver 525+ | `nvidia-smi` | [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx) |
| NVIDIA Container Toolkit | `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` | see below |
| ~15 GB free disk space | for the Docker image | |

**Windows Docker Desktop users:** Make sure "Use the WSL 2 based engine" is ON in Settings > General. GPU passthrough works automatically with recent Docker Desktop versions (4.x+).

**Linux users** -- install NVIDIA Container Toolkit if the test above fails:

```bash
# Ubuntu / Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Step 1: Build the Image

```bash
cd skate-physics-preserver
docker compose build
```

First build takes ~10-15 minutes (downloads PyTorch + CUDA). Subsequent rebuilds use Docker cache and finish in seconds.

### Step 2: Download SAM2 Checkpoint

```bash
# Create the checkpoints folder
mkdir checkpoints

# Download SAM 2.1 Hiera-Small (~150 MB)
curl -L -o checkpoints/sam2.1_hiera_small.pt ^
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

PowerShell alternative:

```powershell
New-Item -ItemType Directory -Force checkpoints
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" `
  -OutFile "checkpoints/sam2.1_hiera_small.pt"
```

DWPose ONNX models are auto-downloaded inside the container on first run (~200 MB, cached in the Docker layer).

### Step 3: Verify Everything Works

```bash
docker compose run --rm pipeline src/extract_physics.py --check
```

Expected output:

```
============================================================
  DEPENDENCY CHECK
============================================================
  [OK] torch 2.6.0
  [OK] CUDA available: NVIDIA GeForce RTX 3070 (8.0 GB)
  [OK] sam2 (SAM 2.1)
  [OK] rtmlib (DWPose)
  [OK] onnxruntime 1.20.1 (providers: ['CUDAExecutionProvider', ...])
  [OK] opencv 4.11.0
  [OK] numpy 1.26.4
============================================================
  All dependencies OK. Ready to run.
============================================================
```

### Step 4: Run the Pipeline

Place your source video in `./input/`, **or pass a YouTube URL directly**:

```
skate-physics-preserver/
  input/
    skate_clip.mp4       <-- your video here (optional if using a URL)
  checkpoints/
    sam2.1_hiera_small.pt
```

**Stage 1 -- Extract tracking data:**

*From a local file:*

```bash
docker compose run --rm pipeline src/extract_physics.py \
  --video /data/input/skate_clip.mp4 \
  --output /data/output \
  --bbox "120,340,280,410" \
  --sam-checkpoint /data/checkpoints/sam2.1_hiera_small.pt
```

*From a YouTube URL:*

```bash
docker compose run --rm pipeline src/extract_physics.py \
  --video "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output /data/output \
  --bbox "120,340,280,410" \
  --sam-checkpoint /data/checkpoints/sam2.1_hiera_small.pt
```

> The video is auto-downloaded to `output/downloads/` via `yt-dlp` (max 1080p, mp4). Short URLs like `https://youtu.be/VIDEO_ID` and `/shorts/` links also work.

> **How to get the bbox:** Open frame 0 in any image viewer and note the pixel coordinates of the skateboard's top-left (x1,y1) and bottom-right (x2,y2) corners. Format: `"x1,y1,x2,y2"`. Tip: use `ffmpeg -i input.mp4 -frames:v 1 frame0.png` to extract the first frame.

Output appears in `./output/` on your host:

```
output/
  mask_skateboard/     <- grayscale mask PNGs
  pose_skater/         <- RGB skeleton PNGs
  pose_json/           <- per-frame keypoint JSON
  tracking_metadata.json
```

**Stage 2 -- Generate reskin (requires ComfyUI running on host):**

Start ComfyUI on your host machine first:

```bash
cd /path/to/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --lowvram
```

Then from the project directory:

```bash
docker compose run --rm pipeline src/generate_reskin.py \
  --source-video /data/input/skate_clip.mp4 \
  --masks-dir /data/output/mask_skateboard \
  --poses-dir /data/output/pose_skater \
  --positive-prompt "cyberpunk samurai riding a neon hoverboard, cinematic" \
  --negative-prompt "blurry, distorted, deformed" \
  --output-dir /data/output/generated \
  --server host.docker.internal:8188
```

> On Linux, replace `host.docker.internal` with your host's LAN IP (e.g., `192.168.1.100`), or add `--network host` to the docker run command.

**Stage 3 -- Validate IoU:**

```bash
docker compose run --rm pipeline src/evaluate_iou.py \
  --metadata /data/output/tracking_metadata.json \
  --generated /data/output/generated/output.mp4 \
  --sam-checkpoint /data/checkpoints/sam2.1_hiera_small.pt
```

### Quick-Reference Commands

```bash
# Build image
docker compose build

# Dependency check
docker compose run --rm pipeline src/extract_physics.py --check

# Check if ComfyUI is reachable
docker compose run --rm pipeline src/generate_reskin.py --check --server host.docker.internal:8188

# Interactive shell inside container (for debugging)
docker compose run --rm --entrypoint bash pipeline

# Run with verbose GPU info
docker compose run --rm --entrypoint nvidia-smi pipeline
```

### Docker Troubleshooting

**"no matching manifest for windows/amd64"**
> Docker Desktop is set to Windows containers. Switch to Linux containers: right-click Docker tray icon > "Switch to Linux containers".

**"could not select device driver" / GPU not detected**
> 1. Verify driver: `nvidia-smi` on host must work.
> 2. Docker Desktop: Settings > Resources > WSL Integration > enable your distro.
> 3. Test GPU: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

**"CUDA out of memory" inside container**
> Close other GPU-heavy apps (games, browsers with HW accel). The container shares your 8 GB VRAM with the host.

**Build fails at "pip install sam2"**
> Transient network error. Re-run `docker compose build` -- Docker layer cache means it resumes where it left off.

**"host.docker.internal" not resolving (Linux)**
> Add `--add-host=host.docker.internal:host-gateway` to your docker run command, or use `--network host`.

---

## Native Setup (Without Docker)

If you prefer a bare-metal install:

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install SAM 2.1 from source
set SAM2_BUILD_CUDA=0
pip install git+https://github.com/facebookresearch/sam2.git@main

# Install rtmlib + ONNX Runtime GPU
pip install rtmlib==0.2.0
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.20.1

# Install remaining dependencies
pip install opencv-python==4.11.0.86 "av>=12.0.0" websocket-client requests tqdm matplotlib "numpy<2.0" hydra-core==1.3.2 yt-dlp
```

> **Note:** Set `SAM2_BUILD_CUDA=0` on Windows if you don't have `nvcc` in PATH.

### 2. Download Model Checkpoints

```bash
mkdir checkpoints
curl -L -o checkpoints/sam2.1_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

### 3. Verify Installation

```bash
python src/extract_physics.py --check
```

### 4. Run the Pipeline

```bash
# Extract tracking from a local file (interactive bbox selection)
python src/extract_physics.py --video input.mp4 --output output/

# Extract tracking from a local file (headless, provide bbox)
python src/extract_physics.py --video input.mp4 --output output/ --bbox "120,340,280,410"

# Extract tracking from a YouTube URL (auto-downloads to output/downloads/)
python src/extract_physics.py --video "https://www.youtube.com/watch?v=VIDEO_ID" --output output/ --bbox "120,340,280,410"

# Generate reskin (ComfyUI must be running)
python src/generate_reskin.py \
  --source-video input.mp4 \
  --masks-dir output/mask_skateboard \
  --poses-dir output/pose_skater \
  --positive-prompt "cyberpunk samurai riding a neon hoverboard, cinematic" \
  --output-dir output/generated

# Validate IoU
python src/evaluate_iou.py \
  --metadata output/tracking_metadata.json \
  --generated output/generated/output.mp4
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3070 8GB | RTX 3090 24GB |
| RAM | 16 GB | 32 GB |
| CUDA Driver | 525+ | 570+ |
| Docker | 24+ (with compose v2) | latest |
| OS | Windows 10/11, Linux | Windows 11 / Ubuntu 22.04 |

### 8GB VRAM Strategy

- **SAM 2.1 Hiera-Small** instead of Large (~3GB vs ~12GB)
- **DWPose balanced** mode instead of performance
- **Sequential processing** with full VRAM cleanup between passes
- **CPU offloading** for SAM2 frame embeddings and tracking state
- **Wan 2.1 VACE 1.3B GGUF Q8** for generation (~3GB model)
- **ComfyUI `--lowvram`** flag for aggressive model offloading

---

## ComfyUI Setup for Generation

### Required Custom Nodes

Install in ComfyUI's `custom_nodes/` directory:

1. **VideoHelperSuite** - Video I/O
   ```bash
   cd custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
   ```

2. **ComfyUI-GGUF** - GGUF model loading
   ```bash
   git clone https://github.com/city96/ComfyUI-GGUF
   ```

3. **ComfyUI-WanVideoWrapper** - Wan 2.1 VACE nodes
   ```bash
   git clone https://github.com/kijai/ComfyUI-WanVideoWrapper
   ```

### Required Models (for 8GB VRAM)

| Model | Path in ComfyUI | Size |
|-------|------|------|
| Wan 2.1 VACE 1.3B GGUF Q8 | `models/unet/wan2.1_vace_1.3B_Q8_0.gguf` | ~1.3 GB |
| UMT5-XXL FP8 | `models/clip/umt5_xxl_fp8_e4m3fn.safetensors` | ~5 GB |
| Wan 2.1 VAE | `models/vae/wan_2.1_vae.safetensors` | ~200 MB |

### Custom Workflow

The template at `workflows/vace_template.json` is a starting point. To use your own:

1. Create your workflow in ComfyUI's GUI
2. Enable Dev Mode (Settings > Dev Mode)
3. Click "Save (API Format)" to export
4. Replace `workflows/vace_template.json`
5. Ensure nodes have descriptive `_meta.title` values for reliable injection

The script finds nodes by `_meta.title` first, then falls back to `class_type`.

---

## Project Structure

```
skate-physics-preserver/
+-- Dockerfile                    # Full GPU-enabled environment
+-- docker-compose.yml            # One-command orchestration
+-- .dockerignore
+-- README.md
+-- requirements.txt
+-- AI_HANDOFF.md                 # Full architecture spec
+-- configs/
|   +-- sam2.1/
|       +-- sam2.1_hiera_s.yaml   # SAM2 Hiera-Small config
+-- checkpoints/                  # (host-mounted) model weights
|   +-- sam2.1_hiera_small.pt
+-- input/                        # (host-mounted) source videos
+-- output/                       # (host-mounted) pipeline outputs
+-- src/
|   +-- __init__.py
|   +-- extract_physics.py        # CLI: tracking orchestrator
|   +-- generate_reskin.py        # CLI: ComfyUI headless client
|   +-- evaluate_iou.py           # CLI: IoU validation
|   +-- tracking/
|       +-- __init__.py
|       +-- skateboard_tracker.py # SAM 2.1 wrapper
|       +-- skater_pose.py        # DWPose wrapper
+-- workflows/
|   +-- vace_template.json        # ComfyUI workflow template
+-- colab_demo.ipynb              # Jupyter demo notebook
```

---

## Troubleshooting (Native Install)

### CUDA Out of Memory
- Use `--pose-mode balanced` or `--pose-mode lightweight`
- Reduce video resolution before processing
- Set `SAM2_BUILD_CUDA=0` (uses PyTorch native ops, slightly less VRAM)
- For ComfyUI: always use `--lowvram` flag

### SAM2 Config Not Found
The tracker uses the SAM2 package's built-in config by default. If you get config errors:
```bash
python -c "import sam2; print(sam2.__file__)"
```
Check that `configs/sam2.1/sam2.1_hiera_s.yaml` exists in the package directory.

### DWPose Model Download Fails
rtmlib downloads ONNX models on first run. If behind a firewall:
1. Download models manually from [rtmlib releases](https://github.com/Tau-J/rtmlib/releases)
2. Place in `~/.cache/rtmlib/`

### ComfyUI Connection Refused
```bash
python src/generate_reskin.py --check
python main.py --listen 0.0.0.0 --port 8188 --lowvram
```

---

## License

Research/educational use. See individual model licenses:
- SAM 2.1: Apache 2.0
- DWPose/rtmlib: Apache 2.0
- Wan 2.1: Apache 2.0
