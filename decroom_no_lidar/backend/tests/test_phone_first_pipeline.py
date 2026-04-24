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
    dimensions_from_point_cloud,
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


def test_dimensions_from_point_cloud_basic():
    """Percentile bbox of a simple point cloud should approximate span."""
    # Points spanning X from 0 to 3, Z from 0 to 4, with a few outliers
    import numpy as np
    xs = np.concatenate([
        np.random.uniform(0.0, 3.0, 1000),
        np.array([-2.0, 5.0]),  # outliers (should be clipped by percentile)
    ])
    zs = np.concatenate([
        np.random.uniform(0.0, 4.0, 1000),
        np.array([-3.0, 7.0]),  # outliers
    ])
    ys = np.zeros_like(xs)
    positions = np.stack([xs, ys, zs], axis=1).astype(np.float32)

    width, length = dimensions_from_point_cloud(positions, manhattan_yaw=0.0)
    # Percentile-based, so should be close to the bulk spans 3x4, not 7x10
    assert 2.5 <= width <= 3.5, f"width={width}"
    assert 3.5 <= length <= 4.5, f"length={length}"


def test_dimensions_from_point_cloud_empty_returns_zero():
    import numpy as np
    empty = np.zeros((0, 3), dtype=np.float32)
    width, length = dimensions_from_point_cloud(empty, manhattan_yaw=0.0)
    assert width == 0.0 and length == 0.0


def test_room_envelope_prefers_cloud_when_anchors_diverge():
    """If the plane-anchor bbox is 5m x 5m but the cloud says 3m x 3m,
    the cloud's dimensions should win."""
    import numpy as np
    # Walls suggest a 5m × 5m room
    walls = [
        {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 5.0, "z": 0.0}},
        {"start": {"x": 0.0, "z": 5.0}, "end": {"x": 5.0, "z": 5.0}},
    ]
    # But the point cloud is 3m × 3m, centered near origin
    xs = np.random.uniform(0.0, 3.0, 1000)
    zs = np.random.uniform(0.0, 3.0, 1000)
    ys = np.zeros_like(xs)
    positions = np.stack([xs, ys, zs], axis=1).astype(np.float32)

    info = room_envelope_from_walls(
        walls, floor_h=0.0, ceil_h=2.4,
        positions=positions, manhattan_yaw=0.0,
    )
    # Cloud wins. Expect ~3m, not ~5m.
    assert info.width_m < 4.0, f"width={info.width_m}"
    assert info.length_m < 4.0, f"length={info.length_m}"


def test_room_envelope_keeps_anchors_when_agreement():
    """If plane anchors and cloud agree within 20%, use anchor bbox."""
    import numpy as np
    walls = [
        {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 3.0, "z": 0.0}},
        {"start": {"x": 0.0, "z": 3.4}, "end": {"x": 3.0, "z": 3.4}},
    ]
    xs = np.random.uniform(0.1, 2.9, 1000)  # cloud bbox ~2.8
    zs = np.random.uniform(0.1, 3.3, 1000)  # cloud bbox ~3.2
    ys = np.zeros_like(xs)
    positions = np.stack([xs, ys, zs], axis=1).astype(np.float32)

    info = room_envelope_from_walls(
        walls, floor_h=0.0, ceil_h=2.4,
        positions=positions, manhattan_yaw=0.0,
    )
    # Anchors = 3.0 × 3.4, cloud = ~2.8 × ~3.2 — within 20% on both.
    # Expect anchor-derived dims (3.0, 3.4).
    assert 2.9 <= info.width_m <= 3.1
    assert 3.3 <= info.length_m <= 3.5
