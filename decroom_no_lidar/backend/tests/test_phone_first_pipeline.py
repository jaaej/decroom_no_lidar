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
    PhoneFirstSchemaError,
    dimensions_from_point_cloud,
    load_ply_points,
    map_spatiallm_objects_to_room_objects,
    parse_payload,
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


# ── Plan 3b: yolo_objects payload parsing ────────────────────────────────────

def _minimal_payload_parts():
    """Helper: returns the bytes for the 5 base multipart parts (no yolo_objects)."""
    room_geom = json.dumps({
        "walls": [], "floor_height": 0.0, "ceiling_height": 2.5,
        "manhattan_yaw": 0.0,
        "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
    }).encode()
    ply = b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n"
    manifest = json.dumps({"frames": []}).encode()
    log = b"{}"
    return room_geom, ply, manifest, log


def test_parse_payload_accepts_yolo_objects_json():
    room_geom, ply, manifest, log = _minimal_payload_parts()
    yolo = json.dumps({
        "objects": [
            {"category": "bed", "position": [1, 0, 2],
             "peak_confidence": 0.9, "view_count": 5, "best_frame_index": 3}
        ]
    }).encode()

    payload = parse_payload(room_geom, ply, manifest, [], log, yolo)
    assert len(payload.yolo_objects) == 1
    assert payload.yolo_objects[0]["category"] == "bed"
    assert payload.yolo_objects[0]["position"] == [1, 0, 2]


def test_parse_payload_allows_missing_yolo_objects_as_empty():
    room_geom, ply, manifest, log = _minimal_payload_parts()
    payload = parse_payload(room_geom, ply, manifest, [], log, b"")
    assert payload.yolo_objects == []


def test_parse_payload_rejects_yolo_object_missing_required_field():
    room_geom, ply, manifest, log = _minimal_payload_parts()
    bad = json.dumps({"objects": [{"category": "bed"}]}).encode()
    with pytest.raises(PhoneFirstSchemaError):
        parse_payload(room_geom, ply, manifest, [], log, bad)


def test_parse_payload_rejects_yolo_objects_wrong_position_length():
    room_geom, ply, manifest, log = _minimal_payload_parts()
    bad = json.dumps({"objects": [{
        "category": "bed", "position": [1, 0],  # missing z
        "peak_confidence": 0.5, "view_count": 1, "best_frame_index": 0,
    }]}).encode()
    with pytest.raises(PhoneFirstSchemaError):
        parse_payload(room_geom, ply, manifest, [], log, bad)


def test_parse_payload_rejects_invalid_yolo_objects_json():
    room_geom, ply, manifest, log = _minimal_payload_parts()
    with pytest.raises(PhoneFirstSchemaError):
        parse_payload(room_geom, ply, manifest, [], log, b"{not json")


def test_fixture_yolo_objects_parses_cleanly():
    """Contract fixture: shared between iOS multipart tests and backend
    parser. If either side changes the schema, this fails."""
    import pathlib
    fx = pathlib.Path(__file__).parent / "fixtures" / "phone_first_payload" / "yolo_objects.json"
    raw = fx.read_bytes()

    room_geom, ply, manifest, log = _minimal_payload_parts()
    payload = parse_payload(room_geom, ply, manifest, [], log, raw)
    assert len(payload.yolo_objects) == 3
    assert {o["category"] for o in payload.yolo_objects} == {"bed", "chair", "potted plant"}


# ── Plan 3b Task 15: consensus pipeline wiring ───────────────────────────────

def test_run_pipeline_applies_consensus_and_wall_snapping(monkeypatch):
    """End-to-end: stub SpatialLM to return a bed 0.5m off the z=0 wall,
    send a YOLO bed at the same position, and assert the final RoomPayload
    bed is wall-snapped (z=0) and confidence > 0.5."""
    import numpy as np
    from pipeline import phone_first_pipeline as mod

    def fake_run(points, cfg):
        walls = [{
            "id": "wall_0",
            "start": [0.0, 0.0],
            "end": [5.0, 0.0],
            "has_door": False,
            "has_window": False,
        }]
        objects = [{
            "type": "bed",
            "position": {"x": 2.0, "y": 0.0, "z": 0.5},
            "rotation_y": 0.0,
            "metadata": {"width": 1.6, "length": 2.0, "height": 0.6},
            "confidence": 0.5,
            "id": "slm0",
        }]
        return walls, objects
    monkeypatch.setattr(mod, "run_spatiallm_pipeline", fake_run)
    monkeypatch.setattr(mod, "load_ply_points",
                         lambda b: (np.zeros((1, 3), dtype=np.float32),
                                    np.zeros((1, 3), dtype=np.uint8)))

    payload = mod.PhoneFirstPayload(
        room_geometry={
            "walls": [{"start": {"x": 0.0, "z": 0.0},
                        "end": {"x": 5.0, "z": 0.0}}],
            "floor_height": 0.0,
            "ceiling_height": 2.5,
            "manhattan_yaw": 0.0,
            "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        },
        point_cloud_ply=b"ply\n",
        keyframes_manifest={"frames": []},
        keyframe_jpegs=[],
        scan_log={},
        yolo_objects=[{
            "category": "bed",
            "position": [2.0, 0.0, 0.5],
            "peak_confidence": 0.85,
            "view_count": 4,
            "best_frame_index": 2,
        }],
    )

    result = mod.run_phone_first_pipeline(payload, cfg=None)
    beds = [o for o in result["objects"] if o["type"] == "bed"]
    assert len(beds) == 1
    bed = beds[0]
    # Wall-snapped onto z=0.
    assert abs(bed["position"]["z"]) < 1e-4, f"bed z={bed['position']['z']}"
    # Consensus + YOLO multi-view + plausible size → confidence well above 0.5.
    assert bed["confidence"] > 0.7, f"bed conf={bed['confidence']}"


def test_run_pipeline_propagates_has_door_from_slm_to_phone_walls(monkeypatch):
    """SLM emits a wall with has_door=True. The phone wall closest to that
    SLM wall's midpoint should end up with has_door=True in the final
    RoomPayload."""
    import numpy as np
    from pipeline import phone_first_pipeline as mod

    def fake_run(points, cfg):
        # SLM returns a wall with a door; same span as phone wall.
        walls = [{
            "id": "wall_0",
            "start": [0.0, 0.0],
            "end": [5.0, 0.0],
            "has_door": True,
            "has_window": False,
        }]
        return walls, []
    monkeypatch.setattr(mod, "run_spatiallm_pipeline", fake_run)
    monkeypatch.setattr(mod, "load_ply_points",
                         lambda b: (np.zeros((1, 3), dtype=np.float32),
                                    np.zeros((1, 3), dtype=np.uint8)))

    payload = mod.PhoneFirstPayload(
        room_geometry={
            "walls": [{"start": {"x": 0.0, "z": 0.0},
                        "end": {"x": 5.0, "z": 0.0}}],
            "floor_height": 0.0,
            "ceiling_height": 2.5,
            "manhattan_yaw": 0.0,
            "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        },
        point_cloud_ply=b"ply\n",
        keyframes_manifest={"frames": []},
        keyframe_jpegs=[],
        scan_log={},
        yolo_objects=[],
    )

    result = mod.run_phone_first_pipeline(payload, cfg=None)
    walls = result["room"]["walls"]
    assert any(w.get("has_door") for w in walls), \
        f"no phone wall got has_door from SLM: {walls}"


def test_run_pipeline_keeps_yolo_only_object_when_slm_missed_it(monkeypatch):
    """SLM sees nothing; YOLO sees a bed. Final payload should still
    contain a bed (lifted from YOLO with default dims, agreement=0.5)."""
    import numpy as np
    from pipeline import phone_first_pipeline as mod

    monkeypatch.setattr(mod, "run_spatiallm_pipeline",
                         lambda points, cfg: ([], []))
    monkeypatch.setattr(mod, "load_ply_points",
                         lambda b: (np.zeros((1, 3), dtype=np.float32),
                                    np.zeros((1, 3), dtype=np.uint8)))

    # Realistic 4-wall rectangle so the YOLO room-bounds filter sees
    # a real room polygon — bed at (2, 0, 1.5) falls inside it.
    payload = mod.PhoneFirstPayload(
        room_geometry={
            "walls": [
                {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 5.0, "z": 0.0}},
                {"start": {"x": 0.0, "z": 3.0}, "end": {"x": 5.0, "z": 3.0}},
                {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 0.0, "z": 3.0}},
                {"start": {"x": 5.0, "z": 0.0}, "end": {"x": 5.0, "z": 3.0}},
            ],
            "floor_height": 0.0,
            "ceiling_height": 2.5,
            "manhattan_yaw": 0.0,
            "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        },
        point_cloud_ply=b"ply\n",
        keyframes_manifest={"frames": []},
        keyframe_jpegs=[],
        scan_log={},
        yolo_objects=[{
            "category": "bed",
            "position": [2.0, 0.0, 1.5],
            "peak_confidence": 0.85,
            "view_count": 5,
            "best_frame_index": 3,
        }],
    )

    result = mod.run_phone_first_pipeline(payload, cfg=None)
    beds = [o for o in result["objects"] if o["type"] == "bed"]
    assert len(beds) == 1, "YOLO-only bed should survive into RoomPayload"


def test_run_pipeline_merges_backend_rgb_into_combined_detections(monkeypatch):
    """Plan 3c: when /detect_furniture returns clusters, they merge into
    the same combined-detections list that consensus_merge sees, alongside
    on-device YOLO objects. Final RoomPayload should contain a wardrobe
    even though YOLO COCO-80 has no such category."""
    import numpy as np
    from pipeline import phone_first_pipeline as mod
    from pipeline import keyframe_clustering as kc_mod
    from pipeline import raycast_lift as lift_mod
    from pipeline import rgb_detection as rgb_mod
    from pipeline.raycast_lift import LiftedDetection
    from pipeline.rgb_detection import KeyframeDetection

    fake_2d = [
        KeyframeDetection(frame_index=0, category="wardrobe",
                          bbox_normalized=(0.4, 0.4, 0.2, 0.2),
                          confidence=0.7),
    ]
    fake_lifted = [
        LiftedDetection(frame_index=0, category="wardrobe",
                        bbox_normalized=(0.4, 0.4, 0.2, 0.2),
                        confidence=0.7,
                        position_3d=(0.5, 0.0, 0.0)),
    ]
    monkeypatch.setattr(rgb_mod, "detect_furniture_in_keyframes",
                         lambda *a, **k: fake_2d)
    monkeypatch.setattr(lift_mod, "lift_detections_to_3d",
                         lambda *a, **k: fake_lifted)
    # SLM returns nothing — wardrobe must come from RGB path alone.
    monkeypatch.setattr(mod, "run_spatiallm_pipeline",
                         lambda points, cfg: ([], []))
    monkeypatch.setattr(mod, "load_ply_points",
                         lambda b: (np.zeros((1, 3), dtype=np.float32),
                                    np.zeros((1, 3), dtype=np.uint8)))

    payload = mod.PhoneFirstPayload(
        room_geometry={
            "walls": [{"start": {"x": 0.0, "z": 0.0},
                        "end": {"x": 5.0, "z": 0.0}}],
            "floor_height": 0.0,
            "ceiling_height": 2.5,
            "manhattan_yaw": 0.0,
            "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        },
        point_cloud_ply=b"ply\n",
        keyframes_manifest={"frames": []},
        keyframe_jpegs=[b"fake_jpeg"],
        scan_log={},
        yolo_objects=[{
            "category": "chair",
            "position": [2.0, 0.0, 0.2],   # within the test room's z bounds
            "peak_confidence": 0.9,
            "view_count": 3,
            "best_frame_index": 0,
        }],
    )

    result = mod.run_phone_first_pipeline(payload, cfg=None)
    types = sorted(o["type"] for o in result["objects"])
    assert "wardrobe" in types, f"expected wardrobe in {types}"
    assert "chair" in types, f"expected chair in {types}"


def test_run_pipeline_continues_when_rgb_detection_fails(monkeypatch):
    """Plan 3c: if /detect_furniture errors out (pod down, network), the
    pipeline must still return a valid RoomPayload using YOLO + SLM only."""
    import numpy as np
    from pipeline import phone_first_pipeline as mod
    from pipeline import rgb_detection as rgb_mod

    def boom(*a, **k):
        raise ConnectionError("pod is down")
    monkeypatch.setattr(rgb_mod, "detect_furniture_in_keyframes", boom)
    monkeypatch.setattr(mod, "run_spatiallm_pipeline",
                         lambda points, cfg: ([], []))
    monkeypatch.setattr(mod, "load_ply_points",
                         lambda b: (np.zeros((1, 3), dtype=np.float32),
                                    np.zeros((1, 3), dtype=np.uint8)))

    payload = mod.PhoneFirstPayload(
        room_geometry={
            "walls": [{"start": {"x": 0.0, "z": 0.0},
                        "end": {"x": 5.0, "z": 0.0}}],
            "floor_height": 0.0, "ceiling_height": 2.5,
            "manhattan_yaw": 0.0,
            "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        },
        point_cloud_ply=b"ply\n",
        keyframes_manifest={"frames": []},
        keyframe_jpegs=[b"fake"],
        scan_log={},
        yolo_objects=[],
    )

    # Should not raise.
    result = mod.run_phone_first_pipeline(payload, cfg=None)
    assert isinstance(result, dict)
    assert "objects" in result
