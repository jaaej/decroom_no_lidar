"""
SpatialLM GPU Inference Server

Mirrors the official SpatialLM inference.py flow exactly so the model
actually produces structured layout output (instead of degenerate
'Wall(11,11,11...)' token loops).

Run: python server.py
Listens on port 8100 by default.
"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

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
SERVER_DIR = Path(__file__).parent
CODE_TEMPLATE_PATH = SERVER_DIR / "code_template.txt"

DETECT_PROMPT = "Detect walls, doors, windows, boxes."


def load_model():
    """Load SpatialLM model + code template (called once at startup)."""
    global _model, _tokenizer, _device, _code_template, _num_bins

    if _model is not None:
        return

    logger.info("Loading model: %s", MODEL_ID)
    t0 = time.time()

    from transformers import AutoTokenizer
    from spatiallm.model.spatiallm_qwen import SpatialLMQwenForCausalLM

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = SpatialLMQwenForCausalLM.from_pretrained(
        MODEL_ID,
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
            f"curl -O https://raw.githubusercontent.com/manycore-research/SpatialLM/main/code_template.txt"
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


@app.on_event("startup")
async def startup():
    load_model()


@app.post("/infer")
async def infer(point_cloud: UploadFile = File(...)):
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        content = await point_cloud.read()
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
        logger.error("Inference failed: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
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
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
