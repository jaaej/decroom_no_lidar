"""
Decroom Backend Server
FastAPI application that receives room photos and returns 3D room models.
"""
from __future__ import annotations

import atexit
import datetime as dt
import glob
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from pipeline.config import PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("decroom-backend")

app = FastAPI(title="Decroom Backend", version="0.1.0")

# Singleton config
cfg = PipelineConfig()

# SpatialLM integration
spatiallm_env = os.getenv("DECROOM_ENABLE_SPATIALLM", "").strip().lower()
if spatiallm_env in {"1", "true", "yes", "on"}:
    cfg.enable_spatiallm = True
elif spatiallm_env in {"0", "false", "no", "off"}:
    cfg.enable_spatiallm = False
spatiallm_mode_env = os.getenv("DECROOM_SPATIALLM_MODE", "").strip().lower()
if spatiallm_mode_env in {"local", "remote"}:
    cfg.spatiallm_mode = spatiallm_mode_env
spatiallm_url_env = os.getenv("DECROOM_SPATIALLM_API_URL", "").strip()
if spatiallm_url_env:
    cfg.spatiallm_api_url = spatiallm_url_env

# ── Bonjour / mDNS service registration ─────────────────────────
# Advertises "_decroom._tcp.local." so the iOS app can discover
# the server automatically without a hardcoded IP address.
_zeroconf = None
_service_info = None

PORT = 8000
RUN_ARCHIVE_ROOT = Path("/tmp/decroom_runs")

_dns_sd_proc: subprocess.Popen | None = None

def _register_bonjour():
    """Register via macOS native dns-sd command (always works, no deps)."""
    global _dns_sd_proc
    try:
        _dns_sd_proc = subprocess.Popen(
            ["dns-sd", "-R", "Decroom Backend", "_decroom._tcp", "local", str(PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Bonjour: registered _decroom._tcp on port %d (dns-sd pid=%d)",
                     PORT, _dns_sd_proc.pid)
    except Exception as e:
        logger.warning("Bonjour registration failed (non-fatal): %s", e)

def _unregister_bonjour():
    global _dns_sd_proc
    if _dns_sd_proc:
        _dns_sd_proc.terminate()
        _dns_sd_proc = None
        logger.info("Bonjour: unregistered")


def _archive_debug_artifacts(
    room_id: str,
    payload: dict,
    metadata_raw: Optional[dict],
    run_started_at: float,
) -> None:
    """Copy /tmp decroom diagnostics to a timestamped run folder."""
    try:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUN_ARCHIVE_ROOT / f"{ts}_{room_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        src_globs = [
            "/tmp/decroom_*.json",
            "/tmp/decroom_*.png",
        ]
        copied = 0
        copied_names = []
        run_end = time.time()
        for g in src_globs:
            for src in glob.glob(g):
                src_path = Path(src)
                if not src_path.is_file():
                    continue
                try:
                    mtime = src_path.stat().st_mtime
                except Exception:
                    continue
                # Keep only artifacts created/updated during this run.
                if mtime < run_started_at - 1.0 or mtime > run_end + 1.0:
                    continue
                dst = run_dir / src_path.name
                shutil.copy2(src_path, dst)
                copied += 1
                copied_names.append(src_path.name)

        if metadata_raw is not None:
            with open(run_dir / "request_metadata.json", "w") as f:
                json.dump(metadata_raw, f, indent=2)
        with open(run_dir / "response_payload.json", "w") as f:
            json.dump(payload, f, indent=2)

        latest = RUN_ARCHIVE_ROOT / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            os.symlink(str(run_dir), str(latest))
        except Exception:
            # Symlink update is best-effort; archive is still valid without it.
            pass

        logger.info(
            "Archived %d debug artifacts to %s (files=%s)",
            copied,
            run_dir,
            ", ".join(sorted(copied_names)) if copied_names else "none",
        )
    except Exception as e:
        logger.warning("Failed to archive debug artifacts: %s", e)


@app.on_event("startup")
async def on_startup():
    _register_bonjour()
    atexit.register(_unregister_bonjour)


@app.post("/api/upload/phone-first")
async def upload_phone_first(
    room_geometry: UploadFile = File(...),
    point_cloud: UploadFile = File(...),
    keyframes_manifest: UploadFile = File(...),
    keyframes: List[UploadFile] = File(...),
    scan_log: UploadFile = File(...),
    yolo_objects: Optional[UploadFile] = File(None),
):
    """Receive a phone-first upload: walls + PLY + keyframes + log.
    Runs SpatialLM on the phone PLY and returns a RoomPayload."""
    room_id = str(uuid.uuid4())[:8]
    run_started_at = time.time()
    logger.info("Phone-first upload received: room_id=%s, %d keyframes",
                room_id, len(keyframes))

    try:
        from pipeline.phone_first_pipeline import (
            PhoneFirstSchemaError,
            parse_payload,
            run_phone_first_pipeline,
        )

        room_geom_bytes = await room_geometry.read()
        ply_bytes = await point_cloud.read()
        manifest_bytes = await keyframes_manifest.read()
        keyframe_bytes = [await kf.read() for kf in keyframes]
        log_bytes = await scan_log.read() if scan_log is not None else b""
        yolo_bytes = await yolo_objects.read() if yolo_objects is not None else b""

        try:
            payload = parse_payload(
                room_geom_bytes, ply_bytes, manifest_bytes,
                keyframe_bytes, log_bytes, yolo_bytes,
            )
        except PhoneFirstSchemaError as e:
            logger.warning("Phone-first payload rejected: %s", e)
            return JSONResponse(
                status_code=400,
                content={"error": "schema_error", "detail": str(e)},
            )

        result = await run_in_threadpool(
            run_phone_first_pipeline, payload, cfg
        )
    except Exception as e:
        logger.exception("Phone-first pipeline failed")
        return JSONResponse(
            status_code=500,
            content={"error": "pipeline_failure", "detail": str(e)},
        )

    _archive_debug_artifacts(room_id, result, None, run_started_at)
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
