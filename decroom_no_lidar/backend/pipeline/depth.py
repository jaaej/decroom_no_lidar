"""
Depth estimation module for the Decroom pipeline.
Supports Depth Anything V2 (relative and metric) via ONNX Runtime.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore

from .config import PipelineConfig

logger = logging.getLogger("decroom-backend.depth")


class DepthRunner:
    """ONNX-based depth estimator supporting DAv2 metric and relative models."""

    def __init__(self, model_path: Path, model_type: str = "dav2_metric"):
        if ort is None:
            raise ImportError("onnxruntime is required for depth estimation")
        if np is None:
            raise ImportError("numpy is required for depth estimation")

        self.model_type = model_type
        self.model_path = model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Depth model not found: {model_path}")

        logger.info("Loading depth model: %s (type=%s)", model_path.name, model_type)
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        # Get input shape
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # e.g. [1, 3, 518, 518]
        if len(self.input_shape) == 4:
            self.input_h = self.input_shape[2]
            self.input_w = self.input_shape[3]
        else:
            self.input_h = 518
            self.input_w = 518

        # Output name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(
            "Depth model loaded: input=%s %s, output=%s",
            self.input_name, self.input_shape, self.output_name,
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict depth for a single image.

        Args:
            image: HxWx3 uint8 RGB image (or BGR — will be converted)

        Returns:
            HxW float32 depth map (meters for metric, relative otherwise)
        """
        orig_h, orig_w = image.shape[:2]

        # Preprocess: resize, normalize, NCHW
        resized = _resize(image, self.input_w, self.input_h)
        tensor = resized.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std

        # HWC -> NCHW
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        depth = outputs[0]

        # Remove batch/channel dims
        if depth.ndim == 4:
            depth = depth[0, 0]
        elif depth.ndim == 3:
            depth = depth[0]

        # Resize back to original resolution
        depth = _resize_depth(depth, orig_w, orig_h)

        # For metric models, output is already in meters
        # For relative models, invert (DAv2 outputs inverse depth)
        if self.model_type == "dav2":
            # Relative model: convert from disparity to depth
            valid = depth > 1e-6
            out = np.zeros_like(depth)
            out[valid] = 1.0 / depth[valid]
            depth = out

        return depth.astype(np.float32)


def _resize(image: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize image using nearest-neighbor (fast, no OpenCV dependency)."""
    try:
        import cv2
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        pass

    # Fallback: numpy-based nearest-neighbor resize
    orig_h, orig_w = image.shape[:2]
    y_ratio = orig_h / h
    x_ratio = orig_w / w
    y_idx = (np.arange(h) * y_ratio).astype(int)
    x_idx = (np.arange(w) * x_ratio).astype(int)
    y_idx = np.clip(y_idx, 0, orig_h - 1)
    x_idx = np.clip(x_idx, 0, orig_w - 1)
    return image[np.ix_(y_idx, x_idx)]


def _resize_depth(depth: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize depth map."""
    try:
        import cv2
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        pass

    orig_h, orig_w = depth.shape[:2]
    y_ratio = orig_h / h
    x_ratio = orig_w / w
    y_idx = (np.arange(h) * y_ratio).astype(int)
    x_idx = (np.arange(w) * x_ratio).astype(int)
    y_idx = np.clip(y_idx, 0, orig_h - 1)
    x_idx = np.clip(x_idx, 0, orig_w - 1)
    return depth[np.ix_(y_idx, x_idx)]


def estimate_depth(
    frames,
    cfg: PipelineConfig,
    cached_runner: Optional[DepthRunner] = None,
) -> Tuple[List[Optional[np.ndarray]], DepthRunner]:
    """
    Run depth estimation on all frames.

    Args:
        frames: List of uploaded image files (FastAPI UploadFile)
        cfg: Pipeline configuration
        cached_runner: Reuse a previously loaded model

    Returns:
        Tuple of (depth_maps, runner) where depth_maps[i] is HxW float32
    """
    if np is None:
        logger.warning("numpy not available; returning empty depth maps")
        return [None] * len(frames), cached_runner

    runner = cached_runner
    if runner is None:
        try:
            runner = DepthRunner(cfg.depth_model, cfg.depth_model_type)
        except Exception as e:
            logger.error("Failed to load depth model: %s", e)
            return [None] * len(frames), None

    depth_maps: List[Optional[np.ndarray]] = []

    for idx, frame in enumerate(frames):
        try:
            image = _load_image(frame)
            if image is None:
                depth_maps.append(None)
                continue

            depth = runner.predict(image)
            depth_maps.append(depth)

            # Debug output
            if cfg.debug_output and idx < 3:
                _save_debug_depth(depth, idx)

        except Exception as e:
            logger.warning("Depth estimation failed for frame %d: %s", idx, e)
            depth_maps.append(None)

    logger.info("Depth estimation complete: %d/%d valid maps", sum(1 for d in depth_maps if d is not None), len(frames))
    return depth_maps, runner


def _load_image(frame) -> Optional[np.ndarray]:
    """Load image from FastAPI UploadFile or file path."""
    try:
        import cv2

        if hasattr(frame, 'file'):
            # FastAPI UploadFile
            data = frame.file.read()
            frame.file.seek(0)
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif isinstance(frame, (str, Path)):
            img = cv2.imread(str(frame))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    except ImportError:
        pass

    # Fallback: PIL
    try:
        from PIL import Image
        import io

        if hasattr(frame, 'file'):
            data = frame.file.read()
            frame.file.seek(0)
            img = Image.open(io.BytesIO(data)).convert("RGB")
        elif isinstance(frame, (str, Path)):
            img = Image.open(str(frame)).convert("RGB")
        else:
            return None
        return np.array(img)
    except Exception as e:
        logger.warning("Failed to load image: %s", e)
        return None


def _save_debug_depth(depth: np.ndarray, idx: int) -> None:
    """Save depth map as debug PNG."""
    try:
        import cv2
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if valid.size == 0:
            return
        vmin, vmax = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
        if vmax - vmin < 0.01:
            return
        normalized = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imwrite(f"/tmp/decroom_depth_{idx}.png", colored)
    except Exception:
        pass
