"""Plan 3c: lift 2D bbox centers to 3D world positions via PLY raycast.

For each detection, build the camera ray through the bbox center,
transform to world space, and march along it until the nearest PLY voxel
is within `shell_tolerance` meters. That voxel is the lifted position.

Falls back to position_3d=None when the ray finds no PLY support along
its first `max_m` meters — caller filters those out.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .rgb_detection import KeyframeDetection

logger = logging.getLogger("decroom-backend.raycast_lift")

RAY_STEP_M = 0.05          # 5 cm per step along the ray
RAY_MAX_M = 8.0            # don't search beyond 8 m (room max diagonal)
SHELL_TOLERANCE_M = 0.08   # accept first PLY point within 8 cm of ray


@dataclass
class KeyframeMeta:
    frame_index: int
    pose_world_from_camera: np.ndarray  # 4x4 row-major, world_from_camera
    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int


@dataclass
class LiftedDetection:
    """KeyframeDetection augmented with a 3D world position (or None)."""
    frame_index: int
    category: str
    bbox_normalized: Tuple[float, float, float, float]
    confidence: float
    position_3d: Optional[Tuple[float, float, float]] = None


def _bbox_center_pixel(
    bbox_normalized: Tuple[float, float, float, float],
    width: int, height: int,
) -> Tuple[float, float]:
    x, y, w, h = bbox_normalized
    return ((x + w / 2.0) * width, (y + h / 2.0) * height)


def _ray_in_world(
    pixel_u: float, pixel_v: float, meta: KeyframeMeta,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (origin, direction) in world space.

    ARKit camera looks down -Z in camera space. Direction transforms via
    pose's rotation block (translation has w=0 so it's ignored).
    """
    x_cam = (pixel_u - meta.cx) / meta.fx
    y_cam = (pixel_v - meta.cy) / meta.fy
    dir_cam = np.array([x_cam, y_cam, -1.0, 0.0], dtype=np.float32)
    pose = meta.pose_world_from_camera
    dir_world = (pose @ dir_cam)[:3]
    norm = float(np.linalg.norm(dir_world))
    if norm > 0:
        dir_world = dir_world / norm
    origin_world = pose[:3, 3].astype(np.float32)
    return origin_world, dir_world.astype(np.float32)


def lift_detections_to_3d(
    detections: List[KeyframeDetection],
    ply_positions: np.ndarray,
    keyframe_metas: List[KeyframeMeta],
    step_m: float = RAY_STEP_M,
    max_m: float = RAY_MAX_M,
    shell_tolerance_m: float = SHELL_TOLERANCE_M,
) -> List[LiftedDetection]:
    """Per detection: raycast through bbox center; first PLY voxel within
    shell_tolerance along the ray = lifted 3D position.

    Detections whose frame has no matching meta, or whose ray finds no
    PLY support within shell_tolerance along its first max_m, return
    position_3d=None.
    """
    out: List[LiftedDetection] = []
    if ply_positions is None or len(ply_positions) == 0:
        return [
            LiftedDetection(
                frame_index=d.frame_index, category=d.category,
                bbox_normalized=d.bbox_normalized, confidence=d.confidence,
                position_3d=None,
            )
            for d in detections
        ]

    meta_by_frame = {m.frame_index: m for m in keyframe_metas}
    tree = cKDTree(ply_positions)
    n_steps = int(max_m / step_m)

    for det in detections:
        meta = meta_by_frame.get(det.frame_index)
        if meta is None:
            out.append(LiftedDetection(
                frame_index=det.frame_index, category=det.category,
                bbox_normalized=det.bbox_normalized, confidence=det.confidence,
                position_3d=None,
            ))
            continue

        u, v = _bbox_center_pixel(
            det.bbox_normalized, meta.image_width, meta.image_height,
        )
        origin, direction = _ray_in_world(u, v, meta)

        hit_pos: Optional[Tuple[float, float, float]] = None
        for k in range(1, n_steps + 1):
            sample = origin + direction * (k * step_m)
            dist, idx = tree.query(sample, k=1)
            if dist <= shell_tolerance_m:
                hit = ply_positions[idx]
                hit_pos = (float(hit[0]), float(hit[1]), float(hit[2]))
                break

        out.append(LiftedDetection(
            frame_index=det.frame_index, category=det.category,
            bbox_normalized=det.bbox_normalized, confidence=det.confidence,
            position_3d=hit_pos,
        ))

    n_hits = sum(1 for d in out if d.position_3d is not None)
    logger.info(
        "raycast_lift: %d/%d detections lifted to 3D", n_hits, len(out),
    )
    return out


def keyframe_metas_from_manifest(manifest: dict) -> List[KeyframeMeta]:
    """Parse the keyframes.json dict the phone uploads into KeyframeMeta list.

    Skips entries with malformed pose (length != 16). Pose is stored
    row-major by ScanUploadPayload.encodeKeyframesManifest, so reshape
    to (4,4) directly.
    """
    out: List[KeyframeMeta] = []
    for i, f in enumerate(manifest.get("frames", [])):
        pose_flat = f.get("pose_world_from_camera", [])
        if len(pose_flat) != 16:
            logger.warning("frame %d: bad pose length %d -- skipping", i, len(pose_flat))
            continue
        pose = np.array(pose_flat, dtype=np.float32).reshape(4, 4)
        intr = f.get("intrinsics", {})
        out.append(KeyframeMeta(
            frame_index=i,
            pose_world_from_camera=pose,
            fx=float(intr.get("fx", 0)),
            fy=float(intr.get("fy", 0)),
            cx=float(intr.get("cx", 0)),
            cy=float(intr.get("cy", 0)),
            image_width=int(intr.get("width", 0)),
            image_height=int(intr.get("height", 0)),
        ))
    return out
