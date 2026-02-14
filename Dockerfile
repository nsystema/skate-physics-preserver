# =============================================================================
# Skate Physics Preserver - Docker Image
# Target: RTX 3070 8GB | CUDA 12.4 | Python 3.11
# Build:  docker compose build
# =============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        git curl wget ca-certificates \
        # OpenCV runtime deps (headless-safe)
        ffmpeg libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 \
        # Build essentials for pycocotools / SAM2
        build-essential gcc g++ \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# ---------- Python dependencies (layered for caching) ----------
WORKDIR /app

# Layer 1: PyTorch (largest download, changes least)
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Layer 2: SAM 2.1 from source (skip custom CUDA kernels for smaller image)
ENV SAM2_BUILD_CUDA=0
RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/sam2.git@main"

# Layer 3: DWPose + ONNX Runtime GPU
RUN pip install --no-cache-dir rtmlib==0.0.15 && \
    (pip uninstall -y onnxruntime 2>/dev/null || true) && \
    pip install --no-cache-dir onnxruntime-gpu==1.20.1

# Layer 4: Everything else
RUN pip install --no-cache-dir \
    opencv-python-headless==4.11.0.86 \
    "av>=12.0.0" \
    "websocket-client>=1.7.0" \
    "requests>=2.31.0" \
    "numpy>=1.26.4,<2.0" \
    hydra-core==1.3.2 \
    "tqdm>=4.67.0" \
    "matplotlib>=3.9.0" \
    pycocotools==2.0.8 \
    "yt-dlp>=2024.12.0"

# ---------- Project source ----------
COPY configs/ /app/configs/
COPY src/     /app/src/
COPY workflows/ /app/workflows/

# Create mount-point directories
RUN mkdir -p /data/checkpoints /data/input /data/output

# Smoke-test: verify core imports parse without error
RUN python -c "\
import torch, cv2, numpy, requests, websocket; \
from sam2.build_sam import build_sam2_video_predictor; \
from rtmlib import Wholebody; \
print('=== All imports OK ==='); \
print(f'  torch {torch.__version__}  CUDA {torch.version.cuda}'); \
print(f'  numpy {numpy.__version__}'); \
print(f'  cv2   {cv2.__version__}'); \
"

# Default entrypoint: drop into a shell or run a command
ENTRYPOINT ["python"]
CMD ["src/extract_physics.py", "--check"]
