"""
auto_detect.py - Automatic Skater + Skateboard Detection
=========================================================
Uses YOLOv8 for object detection and SAM 2.1 Image Predictor for
precise single-frame segmentation.

Pipeline:
  1. YOLO detects 'person' (class 0) and 'skateboard' (class 36)
  2. SAM 2.1 segments each detection on frame 0
  3. Returns annotated overlays for web validation

COCO class IDs:  person = 0,  skateboard = 36
"""

import base64
import gc

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERSON_CLASS = 0
SKATEBOARD_CLASS = 36
CLASS_LABELS = {PERSON_CLASS: "skater", SKATEBOARD_CLASS: "skateboard"}

# BGR colours for overlay: skater = blue-ish, skateboard = orange
OVERLAY_COLORS = {
    "skater": (255, 140, 50),
    "skateboard": (0, 140, 255),
}


# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------

def detect_objects(frame, device="cuda", confidence=0.25, model_size="yolov8n"):
    """
    Run YOLOv8 on a single BGR frame and return person + skateboard hits.

    Returns
    -------
    list[dict]
        Each dict: {label, bbox [x1,y1,x2,y2], confidence, class_id}
        At most one per class (best confidence, with tie-break for skater
        proximity to the skateboard).
    """
    from ultralytics import YOLO

    model = YOLO(f"{model_size}.pt")
    results = model(frame, conf=confidence, device=device, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in CLASS_LABELS:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            detections.append({
                "label": CLASS_LABELS[cls_id],
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(float(box.conf[0]), 3),
                "class_id": cls_id,
            })

    # Free YOLO immediately
    del model, results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Keep the single best detection per class
    best: dict[str, dict] = {}
    for d in detections:
        lab = d["label"]
        if lab not in best or d["confidence"] > best[lab]["confidence"]:
            best[lab] = d

    # If multiple persons, prefer the one closest to the skateboard
    if "skateboard" in best:
        persons = [d for d in detections if d["label"] == "skater"]
        if len(persons) > 1:
            sb = best["skateboard"]["bbox"]
            sb_cx, sb_cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2

            def _dist(p):
                px = (p["bbox"][0] + p["bbox"][2]) / 2
                py = (p["bbox"][1] + p["bbox"][3]) / 2
                return ((px - sb_cx) ** 2 + (py - sb_cy) ** 2) ** 0.5

            best["skater"] = min(persons, key=_dist)

    return list(best.values())


# ---------------------------------------------------------------------------
# SAM 2.1 single-frame segmentation  (Image Predictor)
# ---------------------------------------------------------------------------

def segment_frame(frame, detections, sam_checkpoint, sam_config, device="cuda"):
    """
    Segment detected objects on a single frame using SAM 2.1 Image Predictor.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (frame 0 of the video).
    detections : list[dict]
        Output of ``detect_objects``.
    sam_checkpoint, sam_config : str
        Paths to SAM 2.1 checkpoint and YAML config.
    device : str

    Returns
    -------
    dict[str, np.ndarray]
        ``{label: mask_uint8}`` where mask values are 0 or 255.
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "cuda" and torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam_model = build_sam2(sam_config, sam_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam_model)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(frame_rgb)

    masks: dict[str, np.ndarray] = {}
    for det in detections:
        bbox = np.array(det["bbox"], dtype=np.float32)
        pred_masks, _scores, _logits = predictor.predict(
            box=bbox,
            multimask_output=False,
        )
        masks[det["label"]] = (pred_masks[0].astype(np.uint8)) * 255

    del predictor, sam_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return masks


# ---------------------------------------------------------------------------
# SAM 2.1 point-prompt segmentation  (manual correction)
# ---------------------------------------------------------------------------

def segment_with_points(frame, points, sam_checkpoint, sam_config, device="cuda"):
    """
    Segment objects using manual point prompts (click-to-select fallback).

    Parameters
    ----------
    points : list[dict]
        Each dict: ``{x: int, y: int, label: "skater"|"skateboard"}``.
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "cuda" and torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam_model = build_sam2(sam_config, sam_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam_model)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(frame_rgb)

    # Group points by label
    grouped: dict[str, list] = {}
    for p in points:
        grouped.setdefault(p["label"], []).append([p["x"], p["y"]])

    masks: dict[str, np.ndarray] = {}
    for label, pts in grouped.items():
        point_coords = np.array(pts, dtype=np.float32)
        point_labels = np.ones(len(pts), dtype=np.int32)  # all foreground
        pred_masks, _scores, _logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        masks[label] = (pred_masks[0].astype(np.uint8)) * 255

    del predictor, sam_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return masks


# ---------------------------------------------------------------------------
# Overlay visualisation
# ---------------------------------------------------------------------------

def create_overlay(frame, detections, masks):
    """
    Composite semi-transparent coloured masks + bboxes + labels onto *frame*.

    Returns a BGR image suitable for JPEG encoding.
    """
    overlay = frame.copy()

    for det in detections:
        label = det["label"]
        mask = masks.get(label)
        if mask is None:
            continue

        color = OVERLAY_COLORS.get(label, (128, 128, 128))
        mask_bool = mask > 127

        # Semi-transparent colour fill
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                np.clip(frame[:, :, c] * 0.45 + color[c] * 0.55, 0, 255).astype(np.uint8),
                overlay[:, :, c],
            )

        # Bounding box
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Label pill
        conf = det.get("confidence", 0)
        text = f"{label} {conf:.0%}" if conf else label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(overlay, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
        cv2.putText(
            overlay, text, (x1 + 5, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

    return overlay


def create_points_overlay(frame, points):
    """Draw click markers on the frame for manual-mode feedback."""
    overlay = frame.copy()
    for p in points:
        color = OVERLAY_COLORS.get(p["label"], (128, 128, 128))
        center = (int(p["x"]), int(p["y"]))
        cv2.circle(overlay, center, 10, color, -1)
        cv2.circle(overlay, center, 12, (255, 255, 255), 2)
        cv2.putText(
            overlay, p["label"], (center[0] + 16, center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return overlay


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def frame_to_base64(frame, quality=85):
    """Encode a BGR numpy frame as a base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf).decode("utf-8")


def mask_to_base64(mask):
    """Encode a grayscale mask as base64 PNG."""
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf).decode("utf-8")
