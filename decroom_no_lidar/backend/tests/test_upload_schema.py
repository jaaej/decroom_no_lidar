"""
Tests that malformed phone-first uploads are rejected with
PhoneFirstSchemaError (-> HTTP 400 -> phone ScanError.backendSchemaError).
"""
import json

import pytest

from pipeline.phone_first_pipeline import (
    EXPECTED_COORDINATE_SYSTEM,
    PhoneFirstSchemaError,
    parse_payload,
)


def _valid_geom_bytes() -> bytes:
    return json.dumps({
        "walls": [],
        "floor_height": 0.0,
        "ceiling_height": 2.4,
        "manhattan_yaw": 0.0,
        "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
    }).encode()


def _valid_ply_bytes() -> bytes:
    return (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 0\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"property uchar red\n"
        b"property uchar green\n"
        b"property uchar blue\n"
        b"end_header\n"
    )


def _valid_manifest_bytes() -> bytes:
    return json.dumps({"frames": []}).encode()


def test_parse_valid_payload_succeeds():
    p = parse_payload(
        _valid_geom_bytes(),
        _valid_ply_bytes(),
        _valid_manifest_bytes(),
        [],
        b"{}",
    )
    assert p.room_geometry["coordinate_system"] == EXPECTED_COORDINATE_SYSTEM


def test_rejects_invalid_json_room_geometry():
    with pytest.raises(PhoneFirstSchemaError, match="room_geometry"):
        parse_payload(
            b"not-json",
            _valid_ply_bytes(),
            _valid_manifest_bytes(),
            [],
            b"{}",
        )


def test_rejects_wrong_coordinate_system():
    geom = json.dumps({
        "walls": [],
        "floor_height": 0.0,
        "ceiling_height": 2.4,
        "manhattan_yaw": 0.0,
        "coordinate_system": "unity_left_handed",
    }).encode()
    with pytest.raises(PhoneFirstSchemaError, match="coordinate_system"):
        parse_payload(geom, _valid_ply_bytes(), _valid_manifest_bytes(), [], b"{}")


def test_rejects_missing_fields():
    geom = json.dumps({
        "walls": [],
        "floor_height": 0.0,
        # ceiling_height missing
        "manhattan_yaw": 0.0,
        "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
    }).encode()
    with pytest.raises(PhoneFirstSchemaError, match="ceiling_height"):
        parse_payload(geom, _valid_ply_bytes(), _valid_manifest_bytes(), [], b"{}")


def test_rejects_malformed_ply():
    with pytest.raises(PhoneFirstSchemaError, match="point_cloud.ply"):
        parse_payload(
            _valid_geom_bytes(),
            b"not-a-ply",
            _valid_manifest_bytes(),
            [],
            b"{}",
        )


def test_rejects_missing_frames_in_manifest():
    manifest = json.dumps({"not_frames": []}).encode()
    with pytest.raises(PhoneFirstSchemaError, match="frames"):
        parse_payload(
            _valid_geom_bytes(), _valid_ply_bytes(), manifest, [], b"{}"
        )


# ── Plan 3b: yolo_objects schema validation ──────────────────────────────────

def test_rejects_malformed_yolo_objects_missing_required_field():
    bad = json.dumps({"objects": [{"category": "bed"}]}).encode()
    with pytest.raises(PhoneFirstSchemaError, match="yolo_objects"):
        parse_payload(
            _valid_geom_bytes(), _valid_ply_bytes(),
            _valid_manifest_bytes(), [], b"{}", bad,
        )


def test_rejects_malformed_yolo_objects_bad_position_shape():
    bad = json.dumps({"objects": [{
        "category": "bed", "position": [1, 2],  # only 2 coords
        "peak_confidence": 0.5, "view_count": 1, "best_frame_index": 0,
    }]}).encode()
    with pytest.raises(PhoneFirstSchemaError, match="position"):
        parse_payload(
            _valid_geom_bytes(), _valid_ply_bytes(),
            _valid_manifest_bytes(), [], b"{}", bad,
        )


def test_rejects_yolo_objects_invalid_json():
    with pytest.raises(PhoneFirstSchemaError, match="yolo_objects"):
        parse_payload(
            _valid_geom_bytes(), _valid_ply_bytes(),
            _valid_manifest_bytes(), [], b"{}", b"{not json",
        )


def test_accepts_empty_yolo_objects_field():
    """Empty/missing yolo_objects bytes is valid (Plan 3a tier devices)."""
    p = parse_payload(
        _valid_geom_bytes(), _valid_ply_bytes(),
        _valid_manifest_bytes(), [], b"{}", b"",
    )
    assert p.yolo_objects == []
