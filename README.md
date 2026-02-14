# Skate Physics Preserver

A headless Video-to-Video (V2V) pipeline that preserves spatial integrity during generative human-object interactions (e.g., skateboard → drone reskinning). Optimized for **Google Colab T4** (16GB VRAM).

## Overview

- **Tracking:** SAM 2.1 (Hiera-Small) + DWPose for object masks and pose skeletons
- **Generation:** Wan 2.1 VACE (Q4_K_S GGUF) via ComfyUI
- **Validation:** IoU > 0.90 Zero-Clipping Benchmark

## Colab Setup

1. **Open the notebook:** Upload `colab_demo.ipynb` to [Google Colab](https://colab.research.google.com)
2. **Enable GPU:** Runtime → Change runtime type → T4 GPU
3. **Upload project:** Upload the full `skate-physics-preserver` folder to `/content/` (or clone from your repo)
4. **Run cells** in order: Setup → Install → Models → Tracking → Generation → Validation

### Model Paths (Drive)

Models are stored in Drive for persistence:

- `skate-physics-models/sam2.1_hiera_s.pt` – SAM 2.1 Hiera-Small
- ComfyUI `models/unet/Wan2.1-VACE-14B-Q4_K_S.gguf` – Wan VACE
- ComfyUI `models/vae/wan_2.1_vae.safetensors` – VAE

### BBox Input (Colab)

Colab has no GUI. Provide the skateboard bounding box as `[x1, y1, x2, y2]`:

- **Option A:** Inspect the first frame (displayed in notebook), then set `BBOX = [x1, y1, x2, y2]`
- **Option B:** Use `get_bbox_colab(VIDEO_PATH)` to display the frame and type coordinates when prompted

### Frame Cap

Default `frame_load_cap=50` (~2 seconds @ 24fps) for T4. Increase only if you have more VRAM.

## Local Usage (CLI)

```bash
# Tracking
python -m src.extract_physics -i input.mp4 -o output/ --bbox 100,200,300,280 --frame_cap 50

# Generation (ComfyUI must be running on :8188)
python -m src.generate_reskin --workflow workflows/vace_template.json ...

# Validation
python -m src.evaluate_iou -original output/mask_skateboard -generated output.mp4 -bbox 100,200,300,280
```

## Project Structure

```
skate-physics-preserver/
├── AI_HANDOFF.md           # Full spec
├── README.md
├── requirements.txt
├── colab_demo.ipynb        # End-to-end Colab demo
├── src/
│   ├── tracking/           # SAM 2.1 + DWPose
│   ├── extract_physics.py
│   ├── generate_reskin.py
│   └── evaluate_iou.py
├── configs/sam2.1/         # SAM 2.1 Hiera-Small config
└── workflows/              # ComfyUI VACE template
```

## ComfyUI Workflow

The `workflows/vace_template.json` is a minimal template. For full VACE execution, export a working workflow from ComfyUI (File → Export API) and replace it. Required nodes: `VHS_LoadVideo`, `LoadImage`, `CLIPTextEncode`, `WanVaceToVideo`, `VHS_VideoCombine`.

## License

See project license file.
