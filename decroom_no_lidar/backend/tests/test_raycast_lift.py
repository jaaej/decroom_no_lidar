"""Tests for 2D bbox center → 3D world position via PLY raycast (Plan 3c Tasks 6 + 8)."""
import numpy as np
import pytest

from pipeline.raycast_lift import (
    KeyframeMeta,
    LiftedDetection,
    keyframe_metas_from_manifest,
    lift_detections_to_3d,
)
from pipeline.rgb_detection import KeyframeDetection


def _synthetic_meta() -> KeyframeMeta:
    """Identity pose, principal point at center, fx=fy=100, 200x200 image."""
    return KeyframeMeta(
        frame_index=0,
        pose_world_from_camera=np.eye(4, dtype=np.float32),
        fx=100.0, fy=100.0, cx=100.0, cy=100.0,
        image_width=200, image_height=200,
    )


def test_centered_bbox_hits_ply_surface():
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1.0, 1.0, 1000).astype(np.float32)
    ys = rng.uniform(-1.0, 1.0, 1000).astype(np.float32)
    zs = np.full(1000, -2.0, dtype=np.float32)
    ply = np.stack([xs, ys, zs], axis=1)

    det = KeyframeDetection(
        frame_index=0, category="bed",
        bbox_normalized=(0.4, 0.4, 0.2, 0.2),  # center at 0.5, 0.5
        confidence=0.8,
    )

    out = lift_detections_to_3d([det], ply, [_synthetic_meta()])
    assert len(out) == 1
    pos = out[0].position_3d
    assert pos is not None
    assert pos[0] == pytest.approx(0.0, abs=0.1)
    assert pos[1] == pytest.approx(0.0, abs=0.1)
    assert pos[2] == pytest.approx(-2.0, abs=0.1)


def test_bbox_aimed_at_empty_space_returns_none():
    # Tight wall at x ∈ [-0.3, 0.3], z = -2.0. Bbox center at image x=180
    # rays out to ~camera x=0.8 at z=-2 — beyond the wall.
    rng = np.random.default_rng(0)
    xs = rng.uniform(-0.3, 0.3, 1000).astype(np.float32)
    ys = rng.uniform(-1.0, 1.0, 1000).astype(np.float32)
    zs = np.full(1000, -2.0, dtype=np.float32)
    ply = np.stack([xs, ys, zs], axis=1)

    det = KeyframeDetection(
        frame_index=0, category="bed",
        bbox_normalized=(0.85, 0.4, 0.1, 0.2),  # center at u=190, v=100
        confidence=0.8,
    )
    out = lift_detections_to_3d([det], ply, [_synthetic_meta()])
    assert len(out) == 1
    assert out[0].position_3d is None


def test_drops_detection_when_frame_meta_missing():
    det = KeyframeDetection(
        frame_index=99,
        category="bed",
        bbox_normalized=(0.5, 0.5, 0.1, 0.1),
        confidence=0.8,
    )
    out = lift_detections_to_3d(
        [det], np.zeros((10, 3), dtype=np.float32), [_synthetic_meta()],
    )
    assert len(out) == 1
    assert out[0].position_3d is None


def test_empty_ply_returns_all_none_positions():
    det = KeyframeDetection(
        frame_index=0, category="bed",
        bbox_normalized=(0.5, 0.5, 0.1, 0.1),
        confidence=0.8,
    )
    out = lift_detections_to_3d(
        [det], np.empty((0, 3), dtype=np.float32), [_synthetic_meta()],
    )
    assert len(out) == 1
    assert out[0].position_3d is None


def test_keyframe_metas_from_manifest_parses_pose_intrinsics():
    manifest = {
        "frames": [
            {
                "filename": "kf_000.jpg",
                "timestamp": 0.0,
                "pose_world_from_camera": [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                ],
                "intrinsics": {
                    "fx": 100.0, "fy": 100.0,
                    "cx": 50.0, "cy": 50.0,
                    "width": 100, "height": 100,
                },
                "detections": [],
            },
            {
                "filename": "kf_001.jpg",
                "timestamp": 1.0,
                "pose_world_from_camera": [
                    0, 1, 0, 5,
                    1, 0, 0, 0,
                    0, 0, -1, -3,
                    0, 0, 0, 1,
                ],
                "intrinsics": {
                    "fx": 200.0, "fy": 200.0,
                    "cx": 100.0, "cy": 75.0,
                    "width": 200, "height": 150,
                },
                "detections": [],
            },
        ]
    }
    metas = keyframe_metas_from_manifest(manifest)
    assert len(metas) == 2
    assert metas[0].frame_index == 0
    assert metas[0].fx == 100.0
    assert metas[1].frame_index == 1
    assert metas[1].image_width == 200
    assert list(metas[1].pose_world_from_camera[0]) == [0, 1, 0, 5]


def test_keyframe_metas_skips_malformed_pose():
    manifest = {
        "frames": [
            {"pose_world_from_camera": [1, 0, 0],  # bad length
             "intrinsics": {"fx": 0, "fy": 0, "cx": 0, "cy": 0,
                             "width": 0, "height": 0}},
            {"pose_world_from_camera": [1, 0, 0, 0,
                                          0, 1, 0, 0,
                                          0, 0, 1, 0,
                                          0, 0, 0, 1],
             "intrinsics": {"fx": 10, "fy": 10, "cx": 5, "cy": 5,
                             "width": 10, "height": 10}},
        ]
    }
    metas = keyframe_metas_from_manifest(manifest)
    assert len(metas) == 1
    assert metas[0].frame_index == 1
