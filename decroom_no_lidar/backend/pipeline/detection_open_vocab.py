"""
YOLO-World open-vocabulary detector for furniture categories not in COCO80.
Detects: dresser/wardrobe/cabinet, nightstand, floor lamp, desk lamp, rug, poster.
"""
from __future__ import annotations

import logging
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

from .config import PipelineConfig
from .detection import Detection

logger = logging.getLogger("decroom-backend.detection_open_vocab")

# YOLO-World output class index -> PRD category
YOLO_WORLD_TO_PRD = {
    0: "dresser",
    1: "nightstand",
    2: "lamp_floor",
    3: "lamp_desk",
    4: "rug",
    5: "poster",
    6: "poster",
    7: "dresser",
    8: "nightstand",
    9: "lamp_floor",
    10: "lamp_desk",
    11: "rug",
}


class YoloWorldOnnxDetector:
    """YOLO-World ONNX detector for open-vocabulary furniture detection."""

    def __init__(self, model_path: Path, input_size: int = 640):
        if ort is None or np is None:
            self._available = False
            return

        self._available = False
        self.input_size = input_size

        if not model_path.exists():
            logger.warning("YOLO-World model not found: %s", model_path)
            return

        try:
            logger.info("Loading YOLO-World model: %s", model_path.name)
            self.session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            self._available = True
            logger.info("YOLO-World loaded: outputs=%s", self.output_names)
        except Exception as e:
            logger.warning("Failed to load YOLO-World: %s", e)

    @property
    def available(self) -> bool:
        return self._available

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], float, Tuple[int, int]]:
        """Same preprocessing as YOLOv8."""
        orig_h, orig_w = image.shape[:2]
        scale = min(self.input_size / orig_w, self.input_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        try:
            import cv2
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            y_idx = (np.arange(new_h) * orig_h / new_h).astype(int)
            x_idx = (np.arange(new_w) * orig_w / new_w).astype(int)
            resized = image[np.clip(y_idx, 0, orig_h - 1)][:, np.clip(x_idx, 0, orig_w - 1)]

        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        tensor = padded.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
        return tensor, (orig_w, orig_h), scale, (pad_x, pad_y)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.45) -> List[int]:
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
            iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        return keep

    def _infer(
        self,
        tensor: np.ndarray,
        frame_idx: int,
        original_size: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int],
        cfg: PipelineConfig,
        score_thresh: float = 0.20,
        nms_thresh: float = 0.45,
    ) -> List[Detection]:
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        pred = outputs[0]
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        num_preds = pred.shape[0]
        num_classes = pred.shape[1] - 4

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        max_scores = class_scores[np.arange(num_preds), class_ids]

        valid = max_scores > score_thresh
        if not np.any(valid):
            return []

        boxes_xywh = boxes_xywh[valid]
        class_ids = class_ids[valid]
        max_scores = max_scores[valid]

        x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        boxes_xyxy = np.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=1)

        keep = self._nms(boxes_xyxy, max_scores, nms_thresh)

        pad_x, pad_y = pad
        orig_w, orig_h = original_size
        detections = []

        for k in keep:
            cls_id = int(class_ids[k])
            mapped = YOLO_WORLD_TO_PRD.get(cls_id)
            if mapped is None:
                continue

            score = float(max_scores[k])
            thresh = cfg.detection_class_thresholds.get(mapped, score_thresh)
            if score < thresh:
                continue

            bx1 = (float(boxes_xyxy[k, 0]) - pad_x) / scale
            by1 = (float(boxes_xyxy[k, 1]) - pad_y) / scale
            bx2 = (float(boxes_xyxy[k, 2]) - pad_x) / scale
            by2 = (float(boxes_xyxy[k, 3]) - pad_y) / scale

            bx1 = max(0, min(bx1, orig_w))
            by1 = max(0, min(by1, orig_h))
            bx2 = max(0, min(bx2, orig_w))
            by2 = max(0, min(by2, orig_h))

            if bx2 - bx1 < 5 or by2 - by1 < 5:
                continue

            detections.append(Detection(
                bbox=(bx1, by1, bx2, by2),
                score=score,
                category=mapped,
                frame_idx=frame_idx,
                input_size=original_size,
                mask=None,
            ))

        return detections

    def run(self, frames, cfg: PipelineConfig) -> List[Detection]:
        """Run open-vocab detection on all frames."""
        if not self.available:
            return []

        from .depth import _load_image

        all_dets: List[Detection] = []
        for idx, frame in enumerate(frames):
            try:
                image = _load_image(frame)
                if image is None:
                    continue
                tensor, orig_size, scale, pad = self._preprocess(image)
                dets = self._infer(tensor, idx, orig_size, scale, pad, cfg)
                all_dets.extend(dets)
            except Exception as e:
                logger.warning("YOLO-World failed for frame %d: %s", idx, e)

        if all_dets:
            cat_counts: Dict[str, int] = {}
            for d in all_dets:
                cat_counts[d.category] = cat_counts.get(d.category, 0) + 1
            logger.info("YOLO-World detections: %s", ", ".join(f"{k}:{v}" for k, v in sorted(cat_counts.items())))

        return all_dets
