"""
Unit tests for phone_first_pipeline. SpatialLM is mocked -- these tests
verify our parsing, category mapping, and envelope derivation.
"""
import json
import struct
from unittest.mock import patch

import pytest

from pipeline.phone_first_pipeline import (
    EXPECTED_COORDINATE_SYSTEM,
    PhoneFirstPayload,
    load_ply_points,
    map_spatiallm_objects_to_room_objects,
    room_envelope_from_walls,
    run_phone_first_pipeline,
)


def _ply_with_n_points(n: int) -> bytes:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode()
    body = b""
    for i in range(n):
        body += struct.pack("<3f", float(i), 0.0, 0.0) + bytes([128, 128, 128])
    return header + body


def test_load_ply_points_parses_position_and_color():
    data = _ply_with_n_points(3)
    pos, col = load_ply_points(data)
    assert pos.shape == (3, 3)
    assert col.shape == (3, 3)
    assert float(pos[0][0]) == 0.0
    assert float(pos[2][0]) == 2.0
    assert int(col[0][0]) == 128


def test_room_envelope_from_walls_produces_nonempty_info():
    walls = [
        {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 3.0, "z": 0.0}},
        {"start": {"x": 3.0, "z": 0.0}, "end": {"x": 3.0, "z": 2.5}},
    ]
    info = room_envelope_from_walls(walls, floor_h=0.0, ceil_h=2.4)
    assert info.width_m >= 3.0
    assert info.length_m >= 2.5
    assert info.height_m >= 2.0
    assert len(info.walls) == 2


def test_map_spatiallm_objects_maps_known_classes():
    objs = [
        {"class_name": "bed", "center": [1, 0, 2], "size": [2, 1, 1.5],
         "rotation_y": 0.0, "confidence": 0.9},
        {"class_name": "unknown_class", "center": [0, 0, 0]},
    ]
    rooms = map_spatiallm_objects_to_room_objects(objs)
    assert len(rooms) == 1
    assert rooms[0].type == "bed"
    assert rooms[0].position.x == 1.0


def test_run_pipeline_returns_roompayload_dict():
    """End-to-end dry run with a mocked SpatialLM. Validates result shape."""
    payload = PhoneFirstPayload(
        room_geometry={
            "walls": [
                {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 3.0, "z": 0.0}},
            ],
            "floor_height": 0.0,
            "ceiling_height": 2.4,
            "manhattan_yaw": 0.0,
            "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        },
        point_cloud_ply=_ply_with_n_points(10),
        keyframes_manifest={"frames": []},
        keyframe_jpegs=[],
        scan_log={"scan_id": "testscan"},
    )

    class MockCfg:
        pass

    def fake_spatiallm(points, cfg):
        return [], [{"class_name": "bed", "center": [1, 0, 2],
                     "size": [2, 1, 1.5], "rotation_y": 0.0, "confidence": 0.9}]

    with patch("pipeline.phone_first_pipeline.run_spatiallm_pipeline",
                side_effect=fake_spatiallm):
        result = run_phone_first_pipeline(payload, MockCfg())

    assert "room" in result
    assert "objects" in result
    assert len(result["objects"]) == 1
    assert result["objects"][0]["type"] == "bed"


def test_zero_objects_returns_empty_list():
    payload = PhoneFirstPayload(
        room_geometry={
            "walls": [
                {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 3.0, "z": 0.0}},
            ],
            "floor_height": 0.0, "ceiling_height": 2.4, "manhattan_yaw": 0.0,
            "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        },
        point_cloud_ply=_ply_with_n_points(10),
        keyframes_manifest={"frames": []},
        keyframe_jpegs=[],
        scan_log={"scan_id": "testscan"},
    )

    with patch("pipeline.phone_first_pipeline.run_spatiallm_pipeline",
                return_value=([], [])):
        result = run_phone_first_pipeline(payload, object())

    assert result["objects"] == []
