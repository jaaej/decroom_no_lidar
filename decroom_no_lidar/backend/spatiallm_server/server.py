"""
SpatialLM GPU Inference Server

Minimal FastAPI server that accepts PLY point cloud uploads
and returns SpatialLM structured layout + object output.

Run: python server.py
Listens on port 8100 by default.
"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spatiallm-server")

app = FastAPI(title="SpatialLM Inference Server", version="1.0.0")

# ── Global model cache ──
_model = None
_tokenizer = None
_device = None

MODEL_ID = "manycore-research/SpatialLM1.1-Qwen-0.5B"


def load_model():
    """Load SpatialLM model (called once at startup)."""
    global _model, _tokenizer, _device

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
    _model.eval()

    # Point cloud encoder must be float32
    if hasattr(_model, "set_point_backbone_dtype"):
        _model.set_point_backbone_dtype(torch.float32)

    logger.info("Model loaded in %.1fs on %s", time.time() - t0, _device)


def preprocess_point_cloud(ply_path: str) -> torch.Tensor:
    """Load PLY, clean up, and convert to model input tensor."""
    from spatiallm.pcd.pcd_loader import cleanup_pcd, get_points_and_colors, load_o3d_pcd
    from spatiallm.pcd.transform import (
        Compose, GridSample, NormalizeColor, PositiveShift,
    )

    grid_size = 32.0 / 1280  # SpatialLM 1.1 uses 1280 bins

    # Load and clean point cloud
    pcd = load_o3d_pcd(ply_path)
    logger.info("Loaded PLY: %d points", len(pcd.points))

    pcd = cleanup_pcd(pcd, voxel_size=grid_size)
    points, colors = get_points_and_colors(pcd)
    logger.info("After cleanup: %d points", len(points))

    # Apply transforms: normalize colors, shift to positive coords, grid sample
    data = {
        "coord": points.astype(np.float32),
        "color": colors.astype(np.float32),
    }

    transform = Compose([
        NormalizeColor(),
        PositiveShift(),
        GridSample(
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            keys=("coord", "color"),
            return_grid_coord=True,
        ),
    ])

    data = transform(data)

    # Build input tensor: [coord, color, grid_coord]
    coord = torch.from_numpy(data["coord"]).float()
    color = torch.from_numpy(data["color"]).float()
    grid_coord = torch.from_numpy(data["grid_coord"]).int()

    # Combine into single tensor expected by model
    # Format: (N, 9) = [x, y, z, r, g, b, gx, gy, gz]
    point_cloud = torch.cat([
        coord,
        color,
        grid_coord.float(),
    ], dim=1).unsqueeze(0)  # Add batch dimension

    return point_cloud


def run_inference(ply_path: str) -> str:
    """Run SpatialLM inference on a PLY file, return structured text output."""
    from spatiallm.layout.layout import Layout

    # Preprocess point cloud
    point_clouds = preprocess_point_cloud(ply_path).to(_device)
    logger.info("Point cloud tensor shape: %s", point_clouds.shape)

    # Build prompt
    prompt = "<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, boxes."
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    input_ids = _tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    ).to(_device)

    # Generate
    with torch.no_grad():
        generate_ids = _model.generate(
            input_ids,
            point_clouds=point_clouds,
            max_new_tokens=4096,
            do_sample=False,
        )

    # Decode output (skip prompt tokens)
    output_ids = generate_ids[:, input_ids.shape[1]:]
    text = _tokenizer.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    logger.info("Generated %d chars output", len(text))
    if len(text) > 200:
        logger.info("Output preview: %s...", text[:200])
    else:
        logger.info("Output: %s", text)

    return text


@app.on_event("startup")
async def startup():
    load_model()


@app.post("/infer")
async def infer(point_cloud: UploadFile = File(...)):
    """Accept a PLY file, run SpatialLM, return structured text."""
    t0 = time.time()

    # Save uploaded PLY to temp file
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
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": str(_device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
