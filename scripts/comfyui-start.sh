#!/usr/bin/env bash
# =============================================================================
# ComfyUI Entrypoint — Downloads models on first start, then launches server.
#
# Models are stored in /comfyui/models/ which should be a Docker volume so
# they persist across container restarts (~8 GB total download on first run).
# =============================================================================
set -euo pipefail

MODELS_DIR="/comfyui/models"

# ---- Model definitions ----
# Each entry: LOCAL_PATH  DOWNLOAD_URL  EXPECTED_SIZE_MB  RENAME_FROM (optional)

declare -A MODELS
declare -A MODEL_URLS
declare -A MODEL_RENAME

# 1. UNET: Wan 2.1 VACE 1.3B GGUF Q8  (~2.3 GB)
MODELS[unet]="$MODELS_DIR/unet/wan2.1_vace_1.3B_Q8_0.gguf"
MODEL_URLS[unet]="https://huggingface.co/samuelchristlie/Wan2.1-VACE-1.3B-GGUF/resolve/main/Wan2.1-VACE-1.3B-Q8_0.gguf"

# 2. CLIP: UMT5-XXL FP8  (~6.7 GB)
MODELS[clip]="$MODELS_DIR/clip/umt5_xxl_fp8_e4m3fn.safetensors"
MODEL_URLS[clip]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

# 3. VAE: Wan 2.1 VAE  (~254 MB)
MODELS[vae]="$MODELS_DIR/vae/wan_2.1_vae.safetensors"
MODEL_URLS[vae]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"


# ---- Download helper ----
download_model() {
    local name="$1"
    local dest="${MODELS[$name]}"
    local url="${MODEL_URLS[$name]}"
    local dir
    dir=$(dirname "$dest")

    if [ -f "$dest" ]; then
        local size_mb
        size_mb=$(du -m "$dest" 2>/dev/null | cut -f1)
        echo "[models] $name: already exists ($size_mb MB) — skipping"
        return 0
    fi

    mkdir -p "$dir"
    echo ""
    echo "============================================================"
    echo "  Downloading: $name"
    echo "  URL:  $url"
    echo "  Dest: $dest"
    echo "============================================================"

    # Use wget with resume support; fall back to curl
    if command -v wget &>/dev/null; then
        wget --continue --progress=bar:force:noscroll -O "$dest" "$url"
    else
        curl -L --retry 3 -C - -o "$dest" "$url"
    fi

    if [ -f "$dest" ]; then
        local size_mb
        size_mb=$(du -m "$dest" 2>/dev/null | cut -f1)
        echo "[models] $name: downloaded ($size_mb MB)"
    else
        echo "[models] ERROR: $name download failed!"
        return 1
    fi
}


# ---- Main ----
echo ""
echo "============================================================"
echo "  ComfyUI — Skate Physics Preserver"
echo "============================================================"
echo ""
echo "[models] Checking models in $MODELS_DIR ..."

ALL_OK=true
for name in unet clip vae; do
    download_model "$name" || ALL_OK=false
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "[ERROR] Some models failed to download. ComfyUI may not work correctly."
    echo "        You can place models manually in the comfyui-models/ volume."
    echo ""
fi

echo ""
echo "[models] Model check complete."
echo ""

# ---- Start ComfyUI ----
echo "============================================================"
echo "  Starting ComfyUI server"
echo "  Listening on 0.0.0.0:8188  (--lowvram for 8GB cards)"
echo "============================================================"
echo ""

exec python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --lowvram \
    "$@"
