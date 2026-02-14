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
get_remote_size() {
    # Return the Content-Length of a URL (follows redirects)
    local url="$1"
    if command -v wget &>/dev/null; then
        wget --spider --server-response "$url" 2>&1 | awk '/Content-Length:/{size=$2} END{print size+0}'
    else
        curl -sIL "$url" 2>/dev/null | awk '/^[Cc]ontent-[Ll]ength:/{size=$2} END{gsub(/\r/,"",size); print size+0}'
    fi
}

download_model() {
    local name="$1"
    local dest="${MODELS[$name]}"
    local url="${MODEL_URLS[$name]}"
    local dir
    dir=$(dirname "$dest")

    if [ -f "$dest" ]; then
        # Validate: compare local size against remote Content-Length
        local local_size remote_size
        local_size=$(stat -c %s "$dest" 2>/dev/null || echo 0)
        remote_size=$(get_remote_size "$url")
        if [ "$remote_size" -gt 0 ] 2>/dev/null && [ "$local_size" -ne "$remote_size" ]; then
            echo "[models] $name: SIZE MISMATCH (local ${local_size} vs remote ${remote_size}) — re-downloading"
            rm -f "$dest"
        else
            local size_mb
            size_mb=$(du -m "$dest" 2>/dev/null | cut -f1)
            echo "[models] $name: already exists ($size_mb MB) — skipping"
            return 0
        fi
    fi

    mkdir -p "$dir"
    echo ""
    echo "============================================================"
    echo "  Downloading: $name"
    echo "  URL:  $url"
    echo "  Dest: $dest"
    echo "============================================================"

    # Download to a temp file first, then rename on success
    local tmp="${dest}.part"
    local rc=0
    if command -v wget &>/dev/null; then
        wget --continue --progress=bar:force:noscroll -O "$tmp" "$url" || rc=$?
    else
        curl -L --retry 3 -C - -o "$tmp" "$url" || rc=$?
    fi

    if [ $rc -ne 0 ] || [ ! -f "$tmp" ]; then
        echo "[models] ERROR: $name download failed (exit code $rc)!"
        rm -f "$tmp"
        return 1
    fi

    mv "$tmp" "$dest"
    local size_mb
    size_mb=$(du -m "$dest" 2>/dev/null | cut -f1)
    echo "[models] $name: downloaded ($size_mb MB)"
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
