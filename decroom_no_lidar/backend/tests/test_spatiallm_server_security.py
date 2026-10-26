"""
Size caps and error hygiene for spatiallm_server/server.py. The model is
never loaded (no startup event), so /infer fails inside run_inference.
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
from fastapi.testclient import TestClient

_spec = importlib.util.spec_from_file_location(
    "spatiallm_server_under_test",
    Path(__file__).resolve().parent.parent / "spatiallm_server" / "server.py",
)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)


@pytest.fixture
def client():
    return TestClient(server.app)


def test_infer_oversized_ply_is_413(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_PLY_BYTES", 10)
    r = client.post("/infer", files={"point_cloud": ("p.ply", b"x" * 100)})
    assert r.status_code == 413


def test_infer_chunked_body_over_cap_is_413(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_REQUEST_BYTES", 1000)
    body = (b"x" * 400 for _ in range(5))
    r = client.post("/infer", content=body,
                    headers={"content-type": "multipart/form-data; boundary=abc"})
    assert r.status_code == 413


def test_detect_too_many_images_is_413(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_IMAGES", 1)
    monkeypatch.setattr(server, "_gdino_model", object())
    monkeypatch.setattr(server, "_gdino_processor", object())
    files = [("images", (f"{i}.jpg", b"x")) for i in range(2)]
    r = client.post("/detect_furniture", files=files)
    assert r.status_code == 413


def test_infer_failure_does_not_echo_exception(client, monkeypatch):
    def boom(path):
        raise RuntimeError("/root/secret/path exploded")
    monkeypatch.setattr(server, "run_inference", boom)
    r = client.post("/infer", files={"point_cloud": ("p.ply", b"ply\n")})
    assert r.status_code == 500
    assert "secret" not in r.text


def test_models_are_pinned():
    assert len(server.MODEL_REVISION) == 40
    assert len(server.GDINO_REVISION) == 40
