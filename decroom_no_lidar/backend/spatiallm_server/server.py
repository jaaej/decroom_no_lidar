"""
SpatialLM GPU Inference Server

Mirrors the official SpatialLM inference.py flow exactly so the model
actually produces structured layout output (instead of degenerate
'Wall(11,11,11...)' token loops).

Run: python server.py
Listens on 127.0.0.1:8100 by default — reach it through the SSH tunnel
(ssh -L 8100:localhost:8100 ...). Set SPATIALLM_HOST=0.0.0.0 to expose it.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spatiallm-server")

app = FastAPI(title="SpatialLM Inference Server", version="1.1.0")

# ── Global state ──
_model = None
_tokenizer = None
_device = None
_code_template: str = ""
_num_bins: int = 1280

MODEL_ID = "manycore-research/SpatialLM1.1-Qwen-0.5B"
# Pinned commits: the tokenizer loads with trust_remote_code, so an
# unpinned load would run whatever code is on the model repo's main branch.
MODEL_REVISION = "b3ae9ad7a8689bc130a242c7f922a8cc290c1b59"
SERVER_DIR = Path(__file__).parent
CODE_TEMPLATE_PATH = SERVER_DIR / "code_template.txt"

DETECT_PROMPT = "Detect walls, doors, windows, boxes."

# ── Grounding DINO globals (Plan 3c) ──
_gdino_model = None
_gdino_processor = None
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GDINO_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
DEFAULT_GDINO_PROMPT = (
    "bed. wardrobe. bookshelf. desk. monitor. chair. "
    "nightstand. dresser. lamp. potted plant. sofa."
)


def load_gdino():
    """Load Grounding DINO model + processor (called once at startup)."""
    global _gdino_model, _gdino_processor
    if _gdino_model is not None:
        return
    logger.info("Loading Grounding DINO: %s", GDINO_MODEL_ID)
    t0 = time.time()
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    _gdino_processor = AutoProcessor.from_pretrained(
        GDINO_MODEL_ID, revision=GDINO_REVISION
    )
    device = _device or ("cuda" if torch.cuda.is_available() else "cpu")
    _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_MODEL_ID, revision=GDINO_REVISION
    ).to(device)
    _gdino_model.eval()
    logger.info("Grounding DINO loaded in %.1fs", time.time() - t0)


def load_model():
    """Load SpatialLM model + code template (called once at startup)."""
    global _model, _tokenizer, _device, _code_template, _num_bins

    if _model is not None:
        return

    logger.info("Loading model: %s", MODEL_ID)
    t0 = time.time()

    from transformers import AutoTokenizer
    from spatiallm.model.spatiallm_qwen import SpatialLMQwenForCausalLM

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    _model = SpatialLMQwenForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
    )

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(_device)
    _model.set_point_backbone_dtype(torch.float32)
    _model.eval()

    # Number of grid bins used for coordinate discretization.
    _num_bins = _model.config.point_config["num_bins"]
    logger.info("Model num_bins = %d", _num_bins)

    # Load the code template the model expects in its prompt.
    if not CODE_TEMPLATE_PATH.exists():
        raise RuntimeError(
            f"code_template.txt not found at {CODE_TEMPLATE_PATH}. Download it: "
            f"curl -O https://raw.githubusercontent.com/manycore-research/SpatialLM/8913c44d84a450c53e9340b13317f8cf7144a738/code_template.txt"
        )
    _code_template = CODE_TEMPLATE_PATH.read_text()

    logger.info("Model loaded in %.1fs on %s", time.time() - t0, _device)


def preprocess_point_cloud(points: np.ndarray, colors: np.ndarray,
                             grid_size: float, num_bins: int) -> torch.Tensor:
    """Mirror of official inference.py preprocess_point_cloud."""
    from spatiallm.pcd.transform import Compose

    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform(
        {
            "name": "pcd",
            "coord": points.copy(),
            "color": colors.copy(),
        }
    )
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]
    # Concat order MUST be [grid_coord, xyz, rgb] — model expects this layout.
    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud], axis=0))


def run_inference(ply_path: str) -> str:
    """Run SpatialLM inference, return language-string layout."""
    from spatiallm import Layout
    from spatiallm.pcd import (
        cleanup_pcd, get_points_and_colors, load_o3d_pcd,
    )

    grid_size = Layout.get_grid_size(_num_bins)

    pcd = load_o3d_pcd(ply_path)
    logger.info("Loaded PLY: %d points", len(pcd.points))

    pcd = cleanup_pcd(pcd, voxel_size=grid_size)
    points, colors = get_points_and_colors(pcd)
    logger.info("After cleanup: %d points", len(points))

    min_extent = np.min(points, axis=0)
    input_pcd = preprocess_point_cloud(points, colors, grid_size, _num_bins)
    input_pcd = input_pcd.to(_device)
    logger.info("Point cloud tensor shape: %s", input_pcd.shape)

    # Build prompt with the reference code template.
    prompt = (
        f"<|point_start|><|point_pad|><|point_end|>{DETECT_PROMPT} "
        f"The reference code is as followed: {_code_template}"
    )

    if _model.config.model_type == "spatiallm_qwen":
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    else:
        conversation = [{"role": "user", "content": prompt}]

    input_ids = _tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    ).to(_device)

    # Streamer-based generate matching official flow.
    from transformers import TextIteratorStreamer
    streamer = TextIteratorStreamer(
        _tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True,
    )
    generate_kwargs = dict(
        input_ids=input_ids,
        point_clouds=input_pcd,
        streamer=streamer,
        max_new_tokens=4096,
        do_sample=True,
        use_cache=True,
        temperature=0.6,
        top_p=0.95,
        top_k=10,
        num_beams=1,
    )
    t = Thread(target=_model.generate, kwargs=generate_kwargs)
    t.start()

    chunks = []
    for chunk in streamer:
        chunks.append(chunk)
    t.join()
    layout_str = "".join(chunks)
    logger.info("Generated %d chars output", len(layout_str))

    # Decode + de-discretize back to meters.
    layout = Layout(layout_str)
    layout.undiscretize_and_unnormalize(num_bins=_num_bins)
    layout.translate(min_extent)
    return layout.to_language_string()


# ── Upload size caps ──
# The backend sends one PLY (<= 500K points, ~7.5 MB) or ~66 keyframes of
# ~300 KB, so these leave plenty of headroom.
MAX_REQUEST_BYTES = 500 * 1024 * 1024
MAX_PLY_BYTES = 200 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 200


class _BodyTooLarge(StarletteHTTPException):
    # An HTTPException so FastAPI's body parser re-raises it as a 413.
    def __init__(self) -> None:
        super().__init__(status_code=413, detail="payload_too_large")


class _SizeGuard:
    """Pure ASGI middleware: rejects oversized POST bodies by declared
    Content-Length and by bytes actually received (covers chunked)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            if not declared.isdigit():
                await JSONResponse(status_code=400, content={"error": "bad_content_length"})(scope, receive, send)
                return
            if int(declared) > MAX_REQUEST_BYTES:
                await JSONResponse(status_code=413, content={"error": "payload_too_large"})(scope, receive, send)
                return
        received = 0

        async def capped_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_REQUEST_BYTES:
                    raise _BodyTooLarge()
            return message

        started = False

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, capped_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await JSONResponse(status_code=413, content={"error": "payload_too_large"})(scope, receive, send)


app.add_middleware(_SizeGuard)


async def _read_capped(upload: UploadFile, limit: int) -> bytes:
    if upload.size is not None and upload.size > limit:
        raise _BodyTooLarge()
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise _BodyTooLarge()
    return data


@app.on_event("startup")
async def startup():
    load_model()    # SpatialLM
    load_gdino()    # Grounding DINO (Plan 3c)


@app.post("/infer")
async def infer(point_cloud: UploadFile = File(...)):
    t0 = time.time()
    content = await _read_capped(point_cloud, MAX_PLY_BYTES)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    logger.info("Received PLY: %.1f KB", len(content) / 1024)

    try:
        result = run_inference(tmp_path)
        elapsed = time.time() - t0
        logger.info("Inference completed in %.1fs", elapsed)
        return JSONResponse(content={
            "output": result,
            "elapsed_s": round(elapsed, 2),
        })
    except Exception as e:
        # Details stay in the server log; they can include local paths.
        logger.error("Inference failed: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": "inference_failed"})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": str(_device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_bins": _num_bins,
        "gdino_loaded": _gdino_model is not None,
    }


# ── Plan 3c: open-vocab RGB detection ──

from typing import List
from fastapi import Form


@app.post("/detect_furniture")
async def detect_furniture(
    images: List[UploadFile] = File(...),
    prompt: str = Form(DEFAULT_GDINO_PROMPT),
    box_threshold: float = Form(0.4),
    text_threshold: float = Form(0.3),
):
    """Run open-vocab detection on a list of JPEGs.

    Returns: {"frames": [{"frame_index": int,
                            "detections": [{"category": str,
                                            "bbox_normalized": [x, y, w, h],
                                            "confidence": float}, ...]}, ...]}
    bbox is (x, y, w, h) in normalized [0,1] image coords.
    """
    from PIL import Image
    import io

    if _gdino_model is None or _gdino_processor is None:
        return JSONResponse(status_code=503, content={"error": "gdino not loaded"})

    if len(images) > MAX_IMAGES:
        return JSONResponse(status_code=413, content={"error": "payload_too_large"})

    t0 = time.time()
    frames_out = []
    for idx, upload in enumerate(images):
        raw = await _read_capped(upload, MAX_IMAGE_BYTES)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        W, H = image.size
        inputs = _gdino_processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(_gdino_model.device)
        with torch.no_grad():
            outputs = _gdino_model(**inputs)
        results = _gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(H, W)],
        )[0]
        dets = []
        for label, score, box in zip(
            results["labels"], results["scores"], results["boxes"]
        ):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            dets.append({
                "category": str(label).lower().strip(),
                "bbox_normalized": [x0 / W, y0 / H,
                                     (x1 - x0) / W, (y1 - y0) / H],
                "confidence": float(score),
            })
        frames_out.append({"frame_index": idx, "detections": dets})

    elapsed = time.time() - t0
    logger.info("Detected on %d frames in %.1fs", len(images), elapsed)
    return JSONResponse(content={"frames": frames_out,
                                  "elapsed_s": round(elapsed, 2)})


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("SPATIALLM_HOST", "127.0.0.1"), port=8100)
