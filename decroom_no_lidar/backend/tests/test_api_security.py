"""
HTTP-level security tests for main.py: shared-key auth on uploads and the
/health key proof. The pipeline itself is stubbed out.
"""
import hashlib
import hmac
import json
import os
import tempfile
import time
import uuid

os.environ["DECROOM_API_TOKEN"] = "test-server-key"
os.environ.setdefault("DECROOM_DATA_DIR", tempfile.mkdtemp(prefix="decroom-test-"))

import pytest
from fastapi.testclient import TestClient

import main
import pipeline.phone_first_pipeline as pfp
from pipeline.phone_first_pipeline import EXPECTED_COORDINATE_SYSTEM

KEY = b"test-server-key"
UPLOAD = "/api/upload/phone-first"


def _auth_header(ts=None, nonce=None, key=KEY):
    ts = str(int(time.time()) if ts is None else ts)
    nonce = nonce or uuid.uuid4().hex
    sig = hmac.new(key, f"decroom-upload:{ts}:{nonce}".encode(), hashlib.sha256).hexdigest()
    return {"X-Decroom-Auth": f"{ts}:{nonce}:{sig}"}


def _files(geom=None, keyframe_count=1):
    geom = geom if geom is not None else json.dumps({
        "walls": [],
        "floor_height": 0.0,
        "ceiling_height": 2.4,
        "manhattan_yaw": 0.0,
        "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
    }).encode()
    ply = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        b"end_header\n"
    )
    files = [
        ("room_geometry", ("room_geometry.json", geom, "application/json")),
        ("point_cloud", ("point_cloud.ply", ply, "application/octet-stream")),
        ("keyframes_manifest", ("keyframes.json", b'{"frames": []}', "application/json")),
        ("scan_log", ("scan_log.json", b"{}", "application/json")),
    ]
    for i in range(keyframe_count):
        files.append(("keyframes", (f"frame_{i}.jpg", b"\xff\xd8\xff", "image/jpeg")))
    return files


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(pfp, "run_phone_first_pipeline",
                        lambda payload, cfg: {"room": {"id": "stub"}, "objects": []})
    return TestClient(main.app)


def test_upload_without_auth_is_401(client):
    r = client.post(UPLOAD, files=_files())
    assert r.status_code == 401


def test_upload_with_wrong_key_is_401(client):
    r = client.post(UPLOAD, files=_files(), headers=_auth_header(key=b"wrong"))
    assert r.status_code == 401


def test_upload_with_stale_timestamp_is_401(client):
    r = client.post(UPLOAD, files=_files(),
                    headers=_auth_header(ts=int(time.time()) - 3600))
    assert r.status_code == 401


def test_upload_with_valid_auth_succeeds(client):
    r = client.post(UPLOAD, files=_files(), headers=_auth_header())
    assert r.status_code == 200, r.text
    assert r.json()["room"]["id"] == "stub"


def test_auth_header_is_single_use(client):
    headers = _auth_header()
    assert client.post(UPLOAD, files=_files(), headers=headers).status_code == 200
    assert client.post(UPLOAD, files=_files(), headers=headers).status_code == 401


def test_health_without_nonce_has_no_proof(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "proof" not in r.json()


def test_health_proof_is_hmac_of_nonce(client):
    r = client.get("/health", params={"nonce": "abc123"})
    expected = hmac.new(KEY, b"decroom-health:abc123", hashlib.sha256).hexdigest()
    assert r.json()["proof"] == expected


def test_health_proof_cannot_be_replayed_as_upload_auth(client):
    # A rogue server could ask /health for HMAC(key, X) of any X; that must
    # never yield a valid upload signature.
    ts, nonce = str(int(time.time())), uuid.uuid4().hex
    proof = client.get("/health", params={"nonce": f"{ts}:{nonce}"}).json()["proof"]
    r = client.post(UPLOAD, files=_files(),
                    headers={"X-Decroom-Auth": f"{ts}:{nonce}:{proof}"})
    assert r.status_code == 401


def test_health_rejects_oversized_nonce(client):
    r = client.get("/health", params={"nonce": "x" * 500})
    assert r.status_code == 422


def test_too_many_keyframes_is_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_KEYFRAMES", 2)
    r = client.post(UPLOAD, files=_files(keyframe_count=3), headers=_auth_header())
    assert r.status_code == 413


def test_oversized_part_is_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_PLY_BYTES", 16)
    r = client.post(UPLOAD, files=_files(), headers=_auth_header())
    assert r.status_code == 413


def test_oversized_request_is_413_before_parsing(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 100)
    r = client.post(UPLOAD, files=_files(), headers=_auth_header())
    assert r.status_code == 413


def test_chunked_body_over_cap_is_413(client, monkeypatch):
    # A generator body is sent chunked (no Content-Length), so only the
    # received-bytes counter can catch it.
    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 1000)
    body = (b"x" * 400 for _ in range(5))
    r = client.post(UPLOAD, content=body, headers={
        **_auth_header(), "content-type": "multipart/form-data; boundary=abc"})
    assert r.status_code == 413


def test_bad_content_length_is_400(client):
    r = client.post(UPLOAD, content=b"{}", headers={
        **_auth_header(), "content-length": "-5"})
    assert r.status_code == 400


def _run_asgi(headers):
    """Drive main.app directly so we can see whether the body is ever read."""
    import asyncio
    reads, sent = [], []

    async def receive():
        reads.append(1)
        return {"type": "http.request", "body": b"x" * 10, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": UPLOAD, "raw_path": UPLOAD.encode(),
             "query_string": b"", "root_path": "", "server": ("test", 80), "client": ("1.2.3.4", 1),
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]}
    asyncio.run(main.app(scope, receive, send))
    return reads, sent[0]["status"]


def test_unauthenticated_upload_is_rejected_before_body_is_read():
    reads, status = _run_asgi({"content-type": "multipart/form-data; boundary=abc",
                               "content-length": "10"})
    assert status == 401
    assert reads == []
