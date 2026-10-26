"""
Decroom Backend Server
FastAPI application that receives room photos and returns 3D room models.
"""
from __future__ import annotations

import atexit
import datetime as dt
import glob
import hashlib
import hmac
import json
import logging
import os
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from pipeline.config import PipelineConfig
from pipeline.data_dir import data_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("decroom-backend")

app = FastAPI(title="Decroom Backend", version="0.1.0")

# ── Shared-key auth ─────────────────────────────────────────────
# The phone and this server share a secret key (the "server key" the user
# pastes into the app's Settings). The key itself never goes over the wire:
#   - uploads carry X-Decroom-Auth: "<unix_ts>:<nonce>:<hmac>" where
#     hmac = HMAC-SHA256(key, "decroom-upload:<ts>:<nonce>"), single-use;
#   - GET /health?nonce=N returns proof = HMAC-SHA256(key, "decroom-health:N")
#     so the app can check a discovered server actually holds the key.
# The two message prefixes differ so /health can't be used to mint upload auth.
API_TOKEN_FILE = Path.home() / ".decroom" / "api_token"
AUTH_HEADER = "X-Decroom-Auth"
AUTH_MAX_SKEW_S = 300
NONCE_MAX_LEN = 128


def _load_api_token() -> tuple[str, str, bool]:
    """DECROOM_API_TOKEN env, else ~/.decroom/api_token, else generate one.

    Returns (token, where it came from, whether it was just generated).
    """
    env_token = os.getenv("DECROOM_API_TOKEN", "").strip()
    if env_token:
        return env_token, "the DECROOM_API_TOKEN env var", False
    try:
        file_token = API_TOKEN_FILE.read_text().strip()
        if file_token:
            return file_token, str(API_TOKEN_FILE), False
    except FileNotFoundError:
        pass
    API_TOKEN_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(API_TOKEN_FILE.parent, 0o700)
    token = secrets.token_urlsafe(32)
    fd = os.open(API_TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    return token, str(API_TOKEN_FILE), True


API_TOKEN, API_TOKEN_SOURCE, API_TOKEN_GENERATED = _load_api_token()


def _api_token_fingerprint() -> str:
    """First 8 hex of sha256(key) — safe to show; the app shows the same."""
    return hashlib.sha256(API_TOKEN.encode()).hexdigest()[:8]

_seen_nonces: dict[str, float] = {}
_seen_nonces_lock = threading.Lock()


def _hmac_hex(message: str) -> str:
    return hmac.new(API_TOKEN.encode(), message.encode(), hashlib.sha256).hexdigest()


def _auth_ok(header: Optional[str]) -> bool:
    """Validate a single-use X-Decroom-Auth header."""
    if not header:
        return False
    parts = header.split(":")
    if len(parts) != 3:
        return False
    ts_str, nonce, sig = parts
    try:
        ts = int(ts_str)
    except ValueError:
        return False
    now = time.time()
    if abs(now - ts) > AUTH_MAX_SKEW_S:
        return False
    if not (8 <= len(nonce) <= NONCE_MAX_LEN):
        return False
    expected = _hmac_hex(f"decroom-upload:{ts_str}:{nonce}")
    if not hmac.compare_digest(expected, sig.lower()):
        return False
    with _seen_nonces_lock:
        for n, seen_at in list(_seen_nonces.items()):
            if now - seen_at > 2 * AUTH_MAX_SKEW_S:
                del _seen_nonces[n]
        if nonce in _seen_nonces:
            return False
        _seen_nonces[nonce] = now
    return True


class _BodyTooLarge(StarletteHTTPException):
    # An HTTPException so FastAPI's body parser re-raises it as a 413
    # instead of wrapping it in a generic 400.
    def __init__(self) -> None:
        super().__init__(status_code=413, detail="payload_too_large")


class _UploadGuard:
    """Pure ASGI middleware: auth + body-size cap on every POST, enforced
    before anything reads the body (FastAPI parses multipart forms before
    the endpoint runs, so an endpoint-level check would be too late)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope["headers"]}
        if not _auth_ok(headers.get(AUTH_HEADER.lower())):
            await JSONResponse(status_code=401, content={"error": "unauthorized"})(scope, receive, send)
            return
        declared = headers.get("content-length")
        if declared is not None:
            if not declared.isdigit():
                await JSONResponse(status_code=400, content={"error": "bad_content_length"})(scope, receive, send)
                return
            if int(declared) > MAX_REQUEST_BYTES:
                await JSONResponse(status_code=413, content={"error": "payload_too_large"})(scope, receive, send)
                return

        # Content-Length can be absent (chunked) or wrong, so also count the
        # bytes actually received.
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


app.add_middleware(_UploadGuard)


# ── Upload size caps ────────────────────────────────────────────
# A real scan is ~25 MB (66 keyframes x ~300 KB, PLY capped at 500K points
# = 7.5 MB), so these leave plenty of headroom.
MAX_REQUEST_BYTES = 500 * 1024 * 1024
MAX_PLY_BYTES = 200 * 1024 * 1024
MAX_KEYFRAME_BYTES = 10 * 1024 * 1024
MAX_KEYFRAMES = 200
MAX_JSON_BYTES = 20 * 1024 * 1024


class _PayloadTooLarge(Exception):
    pass


async def _read_capped(upload: UploadFile, limit: int) -> bytes:
    if upload.size is not None and upload.size > limit:
        raise _PayloadTooLarge(upload.filename)
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise _PayloadTooLarge(upload.filename)
    return data

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
    """Copy decroom diagnostics to a timestamped run folder."""
    try:
        run_archive_root = data_dir() / "runs"
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = run_archive_root / f"{ts}_{room_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        src_globs = [
            str(data_dir() / "decroom_*.json"),
            str(data_dir() / "decroom_*.png"),
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

        latest = run_archive_root / "latest"
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
    # The key goes to stdout only, never through `logger` (logs persist).
    if API_TOKEN_GENERATED:
        print(
            f"\nNew Decroom server key (saved to {API_TOKEN_SOURCE}):\n\n"
            f"    {API_TOKEN}\n\n"
            "Paste it into the app: Settings > Server key.\n",
            flush=True,
        )
    else:
        print(
            f"Decroom server key loaded from {API_TOKEN_SOURCE} "
            f"(fingerprint {_api_token_fingerprint()}). "
            f"To see it again: cat {API_TOKEN_FILE}",
            flush=True,
        )


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

        if len(keyframes) > MAX_KEYFRAMES:
            raise _PayloadTooLarge(f"{len(keyframes)} keyframes")
        room_geom_bytes = await _read_capped(room_geometry, MAX_JSON_BYTES)
        ply_bytes = await _read_capped(point_cloud, MAX_PLY_BYTES)
        manifest_bytes = await _read_capped(keyframes_manifest, MAX_JSON_BYTES)
        keyframe_bytes = [await _read_capped(kf, MAX_KEYFRAME_BYTES) for kf in keyframes]
        log_bytes = await _read_capped(scan_log, MAX_JSON_BYTES) if scan_log is not None else b""
        yolo_bytes = await _read_capped(yolo_objects, MAX_JSON_BYTES) if yolo_objects is not None else b""

        # Dump raw keyframe JPEGs for offline use (Marble spike, inspection).
        try:
            kf_root = data_dir() / "keyframes"
            kf_dir = kf_root / room_id
            kf_dir.mkdir(parents=True, exist_ok=True)
            for i, kb in enumerate(keyframe_bytes):
                (kf_dir / f"keyframe_{i:02d}.jpg").write_bytes(kb)
            latest_kf = kf_root / "latest"
            if latest_kf.is_symlink() or latest_kf.exists():
                latest_kf.unlink()
            os.symlink(str(kf_dir), str(latest_kf))
            logger.info("Dumped %d keyframes to %s", len(keyframe_bytes), kf_dir)

            # Prune scan folders older than 7 days.
            cutoff = time.time() - 7 * 86400
            for child in kf_root.iterdir():
                if child.is_symlink() or not child.is_dir():
                    continue
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
        except Exception as e:
            logger.warning("Failed to dump/prune keyframes (non-fatal): %s", e)

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
    except _PayloadTooLarge as e:
        logger.warning("Phone-first upload too large: %s", e)
        return JSONResponse(status_code=413, content={"error": "payload_too_large"})
    except Exception:
        # Details stay in the server log; they can include local paths.
        logger.exception("Phone-first pipeline failed")
        return JSONResponse(
            status_code=500,
            content={"error": "pipeline_failure", "detail": "see server log"},
        )

    _archive_debug_artifacts(room_id, result, None, run_started_at)
    try:
        body = json.dumps(result, allow_nan=False)
    except ValueError:
        logger.error("Phone-first result for %s contains NaN/Infinity", room_id)
        return JSONResponse(
            status_code=500,
            content={"error": "pipeline_failure", "detail": "see server log"},
        )
    return Response(content=body, media_type="application/json")


@app.get("/health")
async def health(nonce: Optional[str] = Query(None, max_length=NONCE_MAX_LEN)):
    if nonce is None:
        return {"status": "ok"}
    return {"status": "ok", "proof": _hmac_hex(f"decroom-health:{nonce}")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
