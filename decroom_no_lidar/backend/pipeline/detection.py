"""
YOLOv8-seg object detection module for the Decroom pipeline.
Runs YOLOv8s-seg via ONNX Runtime to detect furniture with instance masks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import exp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore

from .config import PipelineConfig, COCO_TO_PRD

logger = logging.getLogger("decroom-backend.detection")

# COCO-80 class names (YOLOv8 order)
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass
class Detection:
    """A single object detection."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized to input_size
    score: float = 0.0
    category: str = ""
    frame_idx: int = 0
    input_size: Tuple[int, int] = (640, 640)  # (width, height) of the image used for detection
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # HxW binary mask


def _map_category(coco_name: str, cfg: PipelineConfig) -> Optional[str]:
    """Map COCO class name to PRD category, or None if not relevant."""
    # Direct match in categories list
    if coco_name in cfg.categories:
        return coco_name
    # Lookup in mapping
    mapped = COCO_TO_PRD.get(coco_name)
    if mapped and mapped in cfg.categories:
        return mapped
    return None


def _confidence_threshold(mapped: str, cfg: PipelineConfig) -> float:
    """Get confidence threshold for a mapped category."""
    per_class = cfg.detection_class_thresholds.get(mapped)
    if per_class is not None:
        return per_class
    return max(0.0, cfg.detection_min_confidence)


class YoloSegOnnxDetector:
    """YOLOv8-seg ONNX detector with instance segmentation masks."""

    def __init__(self, model_path: Path, input_size: int = 640):
        if ort is None:
            raise ImportError("onnxruntime required")
        if np is None:
            raise ImportError("numpy required")

        self.input_size = input_size
        self.model_path = model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Detector model not found: {model_path}")

        logger.info("Loading YOLOv8-seg model: %s", model_path.name)
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        self.has_masks = len(outputs) > 1

        logger.info(
            "Detector loaded: input=%s, outputs=%s, has_masks=%s",
            self.input_name, self.output_names, self.has_masks,
        )

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], float, Tuple[int, int]]:
        """Resize and normalize image for YOLO input."""
        orig_h, orig_w = image.shape[:2]
        scale = min(self.input_size / orig_w, self.input_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize
        try:
            import cv2
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # Numpy fallback
            y_idx = (np.arange(new_h) * orig_h / new_h).astype(int)
            x_idx = (np.arange(new_w) * orig_w / new_w).astype(int)
            y_idx = np.clip(y_idx, 0, orig_h - 1)
            x_idx = np.clip(x_idx, 0, orig_w - 1)
            resized = image[np.ix_(y_idx, x_idx)]

        # Pad to square
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Normalize and transpose
        tensor = padded.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

        return tensor, (orig_w, orig_h), scale, (pad_x, pad_y)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.45) -> List[int]:
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep

    def _process_mask(
        self,
        mask_coeffs: np.ndarray,
        protos: np.ndarray,
        bbox: Tuple[float, float, float, float],
        img_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Generate instance mask from mask coefficients and prototypes."""
        try:
            # protos shape: (32, mask_h, mask_w)
            # mask_coeffs shape: (32,)
            mask_h, mask_w = protos.shape[1], protos.shape[2]

            # Matrix multiply: coeffs @ protos -> (mask_h, mask_w)
            mask = np.einsum("c,chw->hw", mask_coeffs, protos)

            # Sigmoid
            mask = 1.0 / (1.0 + np.exp(-mask))

            # Crop to bbox region and resize to original image size
            orig_w, orig_h = img_shape
            x1, y1, x2, y2 = bbox

            # Scale bbox to mask resolution
            sx = mask_w / self.input_size
            sy = mask_h / self.input_size
            mx1 = max(0, int(x1 * sx))
            my1 = max(0, int(y1 * sy))
            mx2 = min(mask_w, int(x2 * sx) + 1)
            my2 = min(mask_h, int(y2 * sy) + 1)

            if mx2 <= mx1 or my2 <= my1:
                return None

            cropped = mask[my1:my2, mx1:mx2]

            # Resize to original image size (within bbox)
            try:
                import cv2
                full_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
                bx1 = max(0, int(x1 * orig_w / self.input_size))
                by1 = max(0, int(y1 * orig_h / self.input_size))
                bx2 = min(orig_w, int(x2 * orig_w / self.input_size))
                by2 = min(orig_h, int(y2 * orig_h / self.input_size))
                if bx2 > bx1 and by2 > by1:
                    resized = cv2.resize(cropped, (bx2 - bx1, by2 - by1))
                    full_mask[by1:by2, bx1:bx2] = resized
                return (full_mask > 0.5).astype(np.uint8)
            except ImportError:
                return (mask > 0.5).astype(np.uint8)

        except Exception as e:
            logger.debug("Mask processing failed: %s", e)
            return None

    def _infer(
        self,
        tensor: np.ndarray,
        frame_idx: int,
        original_size: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int],
        cfg: PipelineConfig,
        score_thresh: float = 0.15,
        nms_thresh: float = 0.45,
    ) -> List[Detection]:
        """Run inference on a single preprocessed image."""
        outputs = self.session.run(self.output_names, {self.input_name: tensor})

        # YOLOv8 output: (1, 4+num_classes+32, num_predictions) for seg
        # or (1, 4+num_classes, num_predictions) for det-only
        pred = outputs[0]
        protos = outputs[1] if self.has_masks and len(outputs) > 1 else None

        if pred.ndim == 3:
            pred = pred[0]  # Remove batch dim -> (features, num_preds)

        # Transpose if needed: (features, N) -> (N, features)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        num_preds = pred.shape[0]
        num_classes = 80
        has_mask_coeffs = pred.shape[1] > 4 + num_classes

        # Extract boxes, scores, classes
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:4 + num_classes]
        mask_coeffs = pred[:, 4 + num_classes:] if has_mask_coeffs else None

        # Get best class per prediction
        class_ids = np.argmax(class_scores, axis=1)
        max_scores = class_scores[np.arange(num_preds), class_ids]

        # Score filter
        valid = max_scores > score_thresh
        if not np.any(valid):
            return []

        boxes_xywh = boxes_xywh[valid]
        class_ids = class_ids[valid]
        max_scores = max_scores[valid]
        if mask_coeffs is not None:
            mask_coeffs = mask_coeffs[valid]

        # XYWH -> XYXY
        x = boxes_xywh[:, 0]
        y = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        keep = self._nms(boxes_xyxy, max_scores, nms_thresh)

        # Unpad and unscale coordinates
        pad_x, pad_y = pad
        orig_w, orig_h = original_size

        if protos is not None and protos.ndim == 4:
            protos = protos[0]  # Remove batch dim

        detections = []
        for k in keep:
            coco_id = int(class_ids[k])
            if coco_id >= len(COCO_NAMES):
                continue
            coco_name = COCO_NAMES[coco_id]

            mapped = _map_category(coco_name, cfg)
            if mapped is None:
                continue

            score = float(max_scores[k])
            thresh = _confidence_threshold(mapped, cfg)
            if score < thresh:
                continue

            # Unpad box coordinates
            bx1 = (float(boxes_xyxy[k, 0]) - pad_x) / scale
            by1 = (float(boxes_xyxy[k, 1]) - pad_y) / scale
            bx2 = (float(boxes_xyxy[k, 2]) - pad_x) / scale
            by2 = (float(boxes_xyxy[k, 3]) - pad_y) / scale

            # Clip to image bounds
            bx1 = max(0, min(bx1, orig_w))
            by1 = max(0, min(by1, orig_h))
            bx2 = max(0, min(bx2, orig_w))
            by2 = max(0, min(by2, orig_h))

            if bx2 - bx1 < 5 or by2 - by1 < 5:
                continue

            # Process mask
            mask = None
            if protos is not None and mask_coeffs is not None:
                mask = self._process_mask(
                    mask_coeffs[k],
                    protos,
                    (float(boxes_xyxy[k, 0]), float(boxes_xyxy[k, 1]),
                     float(boxes_xyxy[k, 2]), float(boxes_xyxy[k, 3])),
                    original_size,
                )

            detections.append(Detection(
                bbox=(bx1, by1, bx2, by2),
                score=score,
                category=mapped,
                frame_idx=frame_idx,
                input_size=original_size,
                mask=mask,
            ))

        return detections

    def run(
        self,
        frames,
        cfg: PipelineConfig,
        debug_output: bool = False,
        debug_dir: str = "/tmp",
    ) -> List[Detection]:
        """Run detection on all frames."""
        from .depth import _load_image

        all_detections: List[Detection] = []

        for idx, frame in enumerate(frames):
            try:
                image = _load_image(frame)
                if image is None:
                    continue

                tensor, orig_size, scale, pad = self._preprocess(image)
                dets = self._infer(tensor, idx, orig_size, scale, pad, cfg)
                all_detections.extend(dets)

                if debug_output:
                    _save_debug_overlay(image, dets, idx, debug_dir)

            except Exception as e:
                logger.warning("Detection failed for frame %d: %s", idx, e)

        logger.info("Detection complete: %d detections across %d frames", len(all_detections), len(frames))

        # Log category counts
        cat_counts: Dict[str, int] = {}
        for d in all_detections:
            cat_counts[d.category] = cat_counts.get(d.category, 0) + 1
        if cat_counts:
            logger.info("Detections by category: %s", ", ".join(f"{k}:{v}" for k, v in sorted(cat_counts.items())))

        return all_detections


# Alias for backwards compatibility
YoloOnnxDetector = YoloSegOnnxDetector


def detect_objects(
    frames,
    cfg: PipelineConfig,
    cached_detector: Optional[YoloSegOnnxDetector] = None,
) -> Tuple[List[Detection], Optional[YoloSegOnnxDetector]]:
    """
    Run object detection on all frames.

    Returns:
        Tuple of (detections, detector) for caching
    """
    detector = cached_detector
    if detector is None:
        try:
            detector = YoloSegOnnxDetector(cfg.detector_model)
        except Exception as e:
            logger.error("Failed to load detector: %s", e)
            return [], None

    detections = detector.run(frames, cfg, debug_output=cfg.debug_output)
    return detections, detector


_CATEGORY_COLORS = {
    "bed":        (255, 80, 80),     # red
    "sofa":       (255, 160, 60),    # orange
    "desk":       (60, 180, 255),    # sky blue
    "chair":      (80, 255, 80),     # green
    "table":      (180, 255, 60),    # lime
    "dresser":    (200, 100, 255),   # purple
    "nightstand": (255, 200, 100),   # peach
    "lamp_floor": (255, 255, 80),    # yellow
    "lamp_desk":  (255, 220, 120),   # gold
    "plant":      (60, 200, 60),     # forest green
    "bookshelf":  (100, 140, 255),   # blue
    "monitor":    (0, 255, 255),     # cyan
    "rug":        (200, 150, 100),   # tan
    "poster":     (255, 100, 200),   # pink
}


def _save_debug_overlay(image: np.ndarray, detections: List[Detection], idx: int, debug_dir: str) -> None:
    """Save detection overlay with colored bboxes, labels, and semi-transparent masks."""
    try:
        import cv2

        vis = image.copy()
        if vis.dtype != np.uint8:
            vis = (vis * 255).astype(np.uint8)

        overlay = vis.copy()  # For semi-transparent mask blending

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = _CATEGORY_COLORS.get(det.category, (0, 255, 0))

            # Draw semi-transparent mask if available
            if det.mask is not None:
                mask_resized = det.mask
                mh, mw = mask_resized.shape[:2]
                ih, iw = vis.shape[:2]
                if mh != ih or mw != iw:
                    mask_resized = cv2.resize(mask_resized.astype(np.uint8), (iw, ih),
                                              interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized > 0.5
                overlay[mask_bool] = (
                    (overlay[mask_bool].astype(np.float32) * 0.6 +
                     np.array(color, dtype=np.float32) * 0.4)
                ).astype(np.uint8)

            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{det.category} {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        vis_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{debug_dir}/decroom_det_{idx}.png", vis_bgr)
        logger.debug("Debug overlay saved: %s/decroom_det_%d.png (%d detections)",
                     debug_dir, idx, len(detections))
    except Exception as e:
        logger.debug("Failed to save debug overlay for frame %d: %s", idx, e)
