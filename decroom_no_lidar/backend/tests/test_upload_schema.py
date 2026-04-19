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
