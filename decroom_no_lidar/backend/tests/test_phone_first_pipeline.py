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
    dimensions_from_cloud_peaks,  # NEW
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
    # Percentile-based (10/90), so the bbox is ~80% of the bulk span:
    # uniform 0..3 -> ~2.4, uniform 0..4 -> ~3.2. Outliers are clipped.
    assert 2.0 <= width <= 3.0, f"width={width}"
    assert 2.8 <= length <= 3.8, f"length={length}"


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


def test_room_envelope_keeps_anchors_when_tight_agreement():
    """When plane-anchor and cloud bbox agree within 5%, use anchor bbox
    (preserves the clean round numbers ARKit gives). With 10/90 percentile
    the cloud bbox of uniform points 0..3 is roughly 2.4 (0.3..2.7)."""
    import numpy as np
    walls = [
        {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 2.42, "z": 0.0}},
        {"start": {"x": 0.0, "z": 2.42}, "end": {"x": 2.42, "z": 2.42}},
    ]
    xs = np.random.uniform(0.0, 3.0, 2000)  # 10-90 percentile -> ~0.3..2.7 = 2.4
    zs = np.random.uniform(0.0, 3.0, 2000)
    ys = np.zeros_like(xs)
    positions = np.stack([xs, ys, zs], axis=1).astype(np.float32)

    info = room_envelope_from_walls(
        walls, floor_h=0.0, ceil_h=2.4,
        positions=positions, manhattan_yaw=0.0,
    )
    # Anchors say 2.42 x 2.42, cloud says ~2.4 x ~2.4 — agree to <5%.
    # Expect anchor-derived dims.
    assert 2.35 <= info.width_m <= 2.50
    assert 2.35 <= info.length_m <= 2.50


def test_room_envelope_prefers_smaller_when_moderate_disagreement():
    """When they disagree by more than 5% (but less than the old 20%
    threshold), the smaller value wins — because both failure modes
    (ARKit extent bleed, cloud seeing through walls) cause overshoot."""
    import numpy as np
    # Anchor bbox 4.1 x 4.0 (overshoots from ARKit extent bleed)
    walls = [
        {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 4.1, "z": 0.0}},
        {"start": {"x": 0.0, "z": 4.0}, "end": {"x": 4.1, "z": 4.0}},
    ]
    # Cloud bbox ~3.4 x ~3.4 (better reflects the real room)
    xs = np.random.uniform(0.3, 3.7, 2000)
    zs = np.random.uniform(0.3, 3.7, 2000)
    ys = np.zeros_like(xs)
    positions = np.stack([xs, ys, zs], axis=1).astype(np.float32)

    info = room_envelope_from_walls(
        walls, floor_h=0.0, ceil_h=2.4,
        positions=positions, manhattan_yaw=0.0,
    )
    # Width: anchor=4.1, cloud=~3.4, diff=~17% -> smaller wins -> ~3.4
    # Length: anchor=4.0, cloud=~3.4, diff=~15% -> smaller wins -> ~3.4
    assert info.width_m < 3.8, f"width={info.width_m} should prefer smaller"
    assert info.length_m < 3.8, f"length={info.length_m} should prefer smaller"


def test_cloud_peaks_finds_walls_at_density_maxima():
    """Synthetic cloud: dense band at x=0 (west wall) and x=3 (east wall),
    sparse dots from x=3 to x=4 (through-doorway leak). Peaks should land
    at 0 and 3, NOT the bbox 4."""
    import numpy as np
    rng = np.random.default_rng(0)
    # Dense walls: 5000 dots each within ±3 cm of the wall position
    west = rng.normal(0.0, 0.03, 5000)
    east = rng.normal(3.0, 0.03, 5000)
    # Sparse through-door bleed: 300 dots uniformly 3.0..4.0
    bleed = rng.uniform(3.0, 4.0, 300)
    x = np.concatenate([west, east, bleed])
    # Corresponding Z: dense bands at z=0 and z=3.4 so ok_z is True too
    # (the function's ok flag requires peaks on both axes).
    south = rng.normal(0.0, 0.03, 5000)
    north = rng.normal(3.4, 0.03, 5000)
    z_bleed = rng.uniform(0.0, 3.4, 300)
    z = np.concatenate([south, north, z_bleed])
    y = np.zeros_like(x)
    positions = np.stack([x, y, z], axis=1).astype(np.float32)

    width, length, ok = dimensions_from_cloud_peaks(positions, manhattan_yaw=0.0)
    assert ok, "expected peaks to be found"
    assert abs(width - 3.0) < 0.15, f"expected ~3.0, got {width}"


def test_cloud_peaks_returns_not_ok_on_sparse_cloud():
    """Fewer than 1000 points: peaks fallback signal."""
    import numpy as np
    positions = np.random.uniform(0, 3, size=(500, 3)).astype(np.float32)
    width, length, ok = dimensions_from_cloud_peaks(positions, manhattan_yaw=0.0)
    assert not ok


def test_cloud_peaks_returns_not_ok_when_no_clear_peaks():
    """Uniform cloud with no density peaks → fallback."""
    import numpy as np
    rng = np.random.default_rng(1)
    xs = rng.uniform(0, 3, 5000)
    zs = rng.uniform(0, 3, 5000)
    ys = np.zeros_like(xs)
    positions = np.stack([xs, ys, zs], axis=1).astype(np.float32)

    width, length, ok = dimensions_from_cloud_peaks(positions, manhattan_yaw=0.0)
    # Uniform distribution has no strong peaks above 3× median density
    assert not ok or abs(width - 3.0) < 0.3, f"width={width}, ok={ok}"


def test_room_envelope_prefers_peaks_when_they_agree():
    """End-to-end: with dense walls + through-door bleed, envelope width
    should come from peaks, not from anchor bbox or percentile."""
    import numpy as np
    rng = np.random.default_rng(2)
    # Walls far apart, both visible
    west = rng.normal(0.0, 0.03, 5000)
    east = rng.normal(3.0, 0.03, 5000)
    bleed = rng.uniform(3.0, 4.0, 300)
    x = np.concatenate([west, east, bleed])
    # Length walls
    south = rng.normal(0.0, 0.03, 5000)
    north = rng.normal(3.4, 0.03, 5000)
    z = np.concatenate([south, north, rng.uniform(0.0, 3.4, 300)])
    y = np.zeros_like(x)
    positions = np.stack([x, y, z], axis=1).astype(np.float32)

    walls = [  # anchors overshoot to 4.0 x 4.0
        {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 4.0, "z": 0.0}},
        {"start": {"x": 0.0, "z": 4.0}, "end": {"x": 4.0, "z": 4.0}},
    ]
    info = room_envelope_from_walls(
        walls, floor_h=0.0, ceil_h=2.4,
        positions=positions, manhattan_yaw=0.0,
    )
    # Peaks should win, giving ~3.0 × ~3.4 (not 4 × 4)
    assert abs(info.width_m - 3.0) < 0.25, f"width={info.width_m}"
    assert abs(info.length_m - 3.4) < 0.25, f"length={info.length_m}"
