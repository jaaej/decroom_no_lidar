"""
Phone-first pipeline: consumes the ScanUploadPayload produced by the iOS app
and returns a RoomPayload. Replaces the heavy SfM/backend-depth/layout/lifting
path from pipeline.py for devices that send phone-built geometry.

Flow:
  1. Parse multipart payload into structured objects
  2. Load phone-provided PLY into a numpy point cloud (ARKit Y-up)
  3. Convert to SpatialLM Z-up coords, re-write PLY, run SpatialLM inference
  4. Parse SpatialLM output → layout + objects (already mapped via SPATIALLM_TO_PRD)
  5. Cross-check SpatialLM walls vs phone walls (keep phone, log disagreement)
  6. Build RoomPayload from phone walls + SpatialLM-derived objects
"""
from __future__ import annotations

import json
import logging
import math
import os
import struct
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .spatiallm_layout import (
    SPATIALLM_TO_PRD,
    arkit_to_spatiallm_coords,
    parse_spatiallm_output,
    run_spatiallm_inference,
    write_colored_ply,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import (
    ObjectMetadata,
    ObjectPosition,
    RoomInfo,
    RoomObject,
    RoomPayload,
    Wall,
)

logger = logging.getLogger("decroom-backend.phone_first")

# Known coordinate system tag the phone sends. Reject anything else.
EXPECTED_COORDINATE_SYSTEM = "arkit_world_right_handed_y_up"

# Wall-disagreement thresholds (logged, not user-facing).
WALL_YAW_DISAGREEMENT_DEG = 15.0
WALL_POS_DISAGREEMENT_M = 0.5


# ── Error type ───────────────────────────────────────────────────────────────

class PhoneFirstSchemaError(ValueError):
    """Raised when the phone payload is malformed. Caller returns HTTP 400."""


# ── Parsed payload container ─────────────────────────────────────────────────

@dataclass
class PhoneFirstPayload:
    room_geometry: Dict[str, Any]
    point_cloud_ply: bytes
    keyframes_manifest: Dict[str, Any]
    keyframe_jpegs: List[bytes]
    scan_log: Dict[str, Any]
    # Plan 3b: deduped 3D-tracked objects from on-device ObjectTracker.
    # Optional in 3a (default empty list); required-shape when present.
    yolo_objects: List[Dict[str, Any]] = field(default_factory=list)


def parse_payload(
    room_geometry: bytes,
    point_cloud_ply: bytes,
    keyframes_manifest: bytes,
    keyframe_jpegs: List[bytes],
    scan_log: bytes,
    yolo_objects: bytes = b"",
) -> PhoneFirstPayload:
    """Parse multipart parts. Raises PhoneFirstSchemaError on any malformity."""
    try:
        geom = json.loads(room_geometry.decode("utf-8"))
    except Exception as e:
        raise PhoneFirstSchemaError(f"room_geometry.json invalid: {e}")

    for field_name in ("walls", "floor_height", "ceiling_height",
                        "manhattan_yaw", "coordinate_system"):
        if field_name not in geom:
            raise PhoneFirstSchemaError(f"room_geometry.json missing {field_name}")

    if geom["coordinate_system"] != EXPECTED_COORDINATE_SYSTEM:
        raise PhoneFirstSchemaError(
            f"unsupported coordinate_system={geom['coordinate_system']!r}"
            f" (expected {EXPECTED_COORDINATE_SYSTEM!r})"
        )

    if len(point_cloud_ply) < 20 or not point_cloud_ply.startswith(b"ply"):
        raise PhoneFirstSchemaError("point_cloud.ply missing or malformed header")

    try:
        manifest = json.loads(keyframes_manifest.decode("utf-8"))
    except Exception as e:
        raise PhoneFirstSchemaError(f"keyframes.json invalid: {e}")
    if "frames" not in manifest:
        raise PhoneFirstSchemaError("keyframes.json missing 'frames'")

    try:
        log = json.loads(scan_log.decode("utf-8")) if scan_log else {}
    except Exception as e:
        raise PhoneFirstSchemaError(f"scan_log.json invalid: {e}")

    yolo_list: List[Dict[str, Any]] = []
    if yolo_objects:
        try:
            yolo_doc = json.loads(yolo_objects.decode("utf-8"))
        except Exception as e:
            raise PhoneFirstSchemaError(f"yolo_objects.json invalid: {e}")
        if not isinstance(yolo_doc, dict) or "objects" not in yolo_doc:
            raise PhoneFirstSchemaError("yolo_objects.json missing 'objects'")
        raw = yolo_doc["objects"]
        if not isinstance(raw, list):
            raise PhoneFirstSchemaError("yolo_objects.objects must be list")
        for i, o in enumerate(raw):
            if not isinstance(o, dict):
                raise PhoneFirstSchemaError(f"yolo_objects[{i}] not an object")
            for required in ("category", "position", "peak_confidence",
                              "view_count", "best_frame_index"):
                if required not in o:
                    raise PhoneFirstSchemaError(
                        f"yolo_objects[{i}] missing {required}"
                    )
            pos = o["position"]
            if not isinstance(pos, list) or len(pos) != 3:
                raise PhoneFirstSchemaError(
                    f"yolo_objects[{i}].position must be [x,y,z]"
                )
        yolo_list = raw

    return PhoneFirstPayload(
        room_geometry=geom,
        point_cloud_ply=point_cloud_ply,
        keyframes_manifest=manifest,
        keyframe_jpegs=keyframe_jpegs,
        scan_log=log,
        yolo_objects=yolo_list,
    )


# ── PLY loader ───────────────────────────────────────────────────────────────

def load_ply_points(ply_bytes: bytes) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Parse the phone's binary-little-endian PLY into (Nx3 float32 positions,
    Nx3 uint8 colors). Raises PhoneFirstSchemaError if malformed.
    """
    if np is None:
        raise PhoneFirstSchemaError("numpy not available on backend")

    header_end = ply_bytes.find(b"end_header\n")
    if header_end < 0:
        raise PhoneFirstSchemaError("PLY missing end_header")
    header = ply_bytes[:header_end].decode("ascii", errors="replace")

    count = None
    for line in header.splitlines():
        if line.startswith("element vertex"):
            parts = line.split()
            if len(parts) >= 3:
                count = int(parts[2])
            break
    if count is None:
        raise PhoneFirstSchemaError("PLY missing element vertex count")

    body = ply_bytes[header_end + len(b"end_header\n"):]
    # 12 bytes xyz (3 × float32) + 3 bytes rgb = 15 bytes per vertex
    expected = count * 15
    if len(body) < expected:
        raise PhoneFirstSchemaError(
            f"PLY body too short: {len(body)} < {expected}"
        )

    positions = np.empty((count, 3), dtype=np.float32)
    colors = np.empty((count, 3), dtype=np.uint8)
    for i in range(count):
        off = i * 15
        positions[i] = struct.unpack_from("<3f", body, off)
        colors[i] = struct.unpack_from("<3B", body, off + 12)

    return positions, colors


# ── Wall disagreement diagnostic ─────────────────────────────────────────────

def check_wall_agreement(
    phone_walls: List[Dict[str, Any]],
    spatiallm_walls: List[Any],
) -> None:
    """Log disagreements (>15° yaw or >0.5m position). Does not mutate results."""
    if not phone_walls or not spatiallm_walls:
        return

    def yaw_of_wall(w: Dict[str, Any]) -> float:
        s, e = w["start"], w["end"]
        return math.atan2(e["x"] - s["x"], e["z"] - s["z"])

    def center_of_wall(w: Dict[str, Any]) -> Tuple[float, float]:
        s, e = w["start"], w["end"]
        return ((s["x"] + e["x"]) / 2, (s["z"] + e["z"]) / 2)

    for pw in phone_walls:
        pyaw = yaw_of_wall(pw)
        pcx, pcz = center_of_wall(pw)
        best_delta_deg = None
        for sw in spatiallm_walls:
            try:
                # spatiallm_walls from parse_spatiallm_output are dicts with
                # "start"/"end" = [x, z] lists in ARKit coords.
                sstart = None
                send = None
                if isinstance(sw, dict):
                    sstart = sw.get("start")
                    send = sw.get("end")
                else:
                    sstart = getattr(sw, "start", None)
                    send = getattr(sw, "end", None)
                if sstart is None or send is None:
                    continue
                sx0, sz0 = float(sstart[0]), float(sstart[1])
                sx1, sz1 = float(send[0]), float(send[1])
                syaw = math.atan2(sx1 - sx0, sz1 - sz0)
                delta_deg = math.degrees(abs(syaw - pyaw))
                if best_delta_deg is None or delta_deg < best_delta_deg:
                    best_delta_deg = delta_deg
            except Exception:
                continue
        if best_delta_deg is not None and best_delta_deg > WALL_YAW_DISAGREEMENT_DEG:
            logger.warning(
                "Phone/SpatialLM wall disagreement %.1f deg > %.1f deg at (%.2f,%.2f) -- keeping phone wall",
                best_delta_deg, WALL_YAW_DISAGREEMENT_DEG, pcx, pcz,
            )


# ── Dimensions from point cloud (more robust than plane-anchor bbox) ─────────

def dimensions_from_point_cloud(
    positions: "np.ndarray",
    manhattan_yaw: float,
    percentile_low: float = 8.0,
    percentile_high: float = 92.0,
) -> Tuple[float, float]:
    """
    Compute (width_m, length_m) from the Manhattan-aligned point cloud
    using percentile bounds. Robust against:
      - A few stray dots (mirror reflections, depth noise at edges) —
        clipped by the percentile
      - ARKit plane-anchor variance — doesn't care about anchors at all
      - Mis-classified plane anchors pushed past real walls — irrelevant
        here, we look at depth-derived surface positions

    Returns (width_along_X_axis, length_along_Z_axis) in the Manhattan
    frame, in meters.
    """
    if np is None or positions is None or len(positions) == 0:
        return (0.0, 0.0)

    # Rotate points into Manhattan frame so the room aligns to X/Z axes.
    # manhattan_yaw is the room's rotation around Y, so we apply -yaw.
    cos_y = math.cos(-manhattan_yaw)
    sin_y = math.sin(-manhattan_yaw)
    x = positions[:, 0]
    z = positions[:, 2]
    xr = x * cos_y - z * sin_y
    zr = x * sin_y + z * cos_y

    x_lo = float(np.percentile(xr, percentile_low))
    x_hi = float(np.percentile(xr, percentile_high))
    z_lo = float(np.percentile(zr, percentile_low))
    z_hi = float(np.percentile(zr, percentile_high))

    width = max(x_hi - x_lo, 0.1)
    length = max(z_hi - z_lo, 0.1)
    return (width, length)


# ── Room envelope from phone walls ───────────────────────────────────────────

def room_envelope_from_walls(
    walls: List[Dict[str, Any]],
    floor_h: float,
    ceil_h: float,
    positions: "np.ndarray" = None,
    manhattan_yaw: float = 0.0,
) -> RoomInfo:
    """Derive width/length/height from phone walls + cross-check via cloud.

    Width/length: use plane-anchor bbox if it agrees with the point cloud
    within 20%; otherwise prefer the point-cloud percentile dimensions
    (they're less affected by ARKit plane-anchor variance).

    Height: unchanged — use (ceiling - floor) if available, else 2.5 m.
    """
    # Plane-anchor bounding box (legacy approach)
    xs: List[float] = []
    zs: List[float] = []
    for w in walls:
        xs.extend([w["start"]["x"], w["end"]["x"]])
        zs.extend([w["start"]["z"], w["end"]["z"]])

    anchor_width = (max(xs) - min(xs)) if xs else 0.0
    anchor_length = (max(zs) - min(zs)) if zs else 0.0

    # Point-cloud percentile-based dimensions
    cloud_width, cloud_length = (0.0, 0.0)
    if positions is not None and len(positions) > 0:
        cloud_width, cloud_length = dimensions_from_point_cloud(
            positions, manhattan_yaw
        )

    # Reconcile: both anchor and cloud can overshoot (ARKit extent-bleed,
    # depth-model seeing through windows/doorways). When they disagree,
    # the smaller value is almost always closer to truth because both
    # failure modes cause overshoot. When they agree tightly, we keep the
    # anchor value for cleaner round numbers matching ARKit's committed
    # walls.
    def reconcile(anchor: float, cloud: float) -> float:
        if anchor <= 0.1:
            return cloud if cloud > 0.1 else 4.0
        if cloud <= 0.1:
            return anchor
        diff = abs(anchor - cloud) / max(anchor, cloud)
        if diff <= 0.05:
            # Tight agreement — use anchor (rounder numbers, matches ARKit)
            logger.info("dimensions reconcile: anchor=%.2f cloud=%.2f agree (%.1f%%)",
                        anchor, cloud, diff * 100)
            return anchor
        # Any meaningful disagreement: prefer the smaller value
        chosen = min(anchor, cloud)
        logger.warning(
            "dimensions reconcile: anchor=%.2f cloud=%.2f diff=%.1f%% -> using %.2f (smaller)",
            anchor, cloud, diff * 100, chosen,
        )
        return chosen

    width = reconcile(anchor_width, cloud_width)
    length = reconcile(anchor_length, cloud_length)
    height = (ceil_h - floor_h) if ceil_h > floor_h else 2.5

    walls_out = [
        Wall(
            id=f"w{i}",
            start=[w["start"]["x"], w["start"]["z"]],
            end=[w["end"]["x"], w["end"]["z"]],
        )
        for i, w in enumerate(walls)
    ]

    return RoomInfo(
        id=str(uuid.uuid4())[:8],
        width_m=max(width, 1.0),
        length_m=max(length, 1.0),
        height_m=max(height, 2.0),
        walls=walls_out,
    )


# ── SpatialLM object mapping ─────────────────────────────────────────────────

def map_spatiallm_objects_to_room_objects(
    spatiallm_objects: List[Any],
) -> List[RoomObject]:
    """
    Map SpatialLM object list into RoomObject list.

    Accepts two shapes:
      - Output of parse_spatiallm_output(): dicts with already-mapped "type"
        (Decroom category), "position" dict, "rotation_y", "metadata" dict.
      - Raw-ish dicts with "class_name", "center", "size", "rotation_y",
        "confidence" that still need mapping through SPATIALLM_TO_PRD.
    """
    out: List[RoomObject] = []

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        val = getattr(obj, key, None)
        if val is None and isinstance(obj, dict):
            val = obj.get(key, default)
        return val if val is not None else default

    for i, obj in enumerate(spatiallm_objects):
        # Decide category: prefer already-mapped "type", else map class_name.
        prd_type = _get(obj, "type")
        if not prd_type:
            cls = _get(obj, "class_name")
            if not cls:
                continue
            prd_type = SPATIALLM_TO_PRD.get(cls)
            if not prd_type:
                logger.debug("SpatialLM class %r not mapped -- skipping", cls)
                continue

        # Position: prefer "position" dict (parse_spatiallm_output),
        # else "center" list.
        pos_dict = _get(obj, "position")
        if isinstance(pos_dict, dict):
            px = float(pos_dict.get("x", 0.0))
            py = float(pos_dict.get("y", 0.0))
            pz = float(pos_dict.get("z", 0.0))
        else:
            center = _get(obj, "center")
            if center is None:
                continue
            px = float(center[0])
            py = float(center[1]) if len(center) > 1 else 0.0
            pz = float(center[2]) if len(center) > 2 else 0.0

        # Size: "metadata" dict (parse_spatiallm_output) or raw "size" list.
        meta = _get(obj, "metadata")
        if isinstance(meta, dict):
            width = float(meta.get("width", 1.0))
            length = float(meta.get("length", 1.0))
            height = float(meta.get("height", 1.0))
        else:
            size = _get(obj, "size", [1.0, 1.0, 1.0])
            width = float(size[0])
            # For raw "size", convention is [x, y, z] (ARKit) → width, height, length.
            height = float(size[1]) if len(size) > 1 else 1.0
            length = float(size[2]) if len(size) > 2 else 1.0

        rotation_y = float(_get(obj, "rotation_y", 0.0) or 0.0)
        confidence_raw = _get(obj, "confidence", 0.5)
        confidence = float(confidence_raw if confidence_raw is not None else 0.5)

        obj_id = _get(obj, "id") or f"o{i}"

        out.append(RoomObject(
            id=obj_id,
            type=prd_type,
            asset_id=f"kenney_{prd_type}",
            position=ObjectPosition(x=px, y=py, z=pz),
            rotation_y=rotation_y,
            scale=1.0,
            confidence=confidence,
            metadata=ObjectMetadata(
                width=width,
                length=length,
                height=height,
                inferred=True,
                support_views=None,
            ),
        ))
    return out


# ── SpatialLM inference on phone PLY ─────────────────────────────────────────

def _run_spatiallm_on_phone_ply(
    positions: "np.ndarray",
    colors: "np.ndarray",
    cfg: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run SpatialLM inference using the phone-provided point cloud.

    Mirrors the relevant steps of run_spatiallm_pipeline() but starts from
    an already-built point cloud (no frames/depth backprojection).

    Returns (layout_dict, objects_list) — same shape as parse_spatiallm_output().
    """
    if np is None:
        raise RuntimeError("numpy not available on backend")

    # Convert to SpatialLM Z-up coords.
    slm_points, coord_offset = arkit_to_spatiallm_coords(positions)

    # Subsample if the phone sent too many points. SpatialLM's voxel encoder
    # crashes (Floating point exception / CUDA kernel failure) on inputs
    # > ~250K. Phone caps at 500K, which is too much; random-subsample to
    # 200K here so the pipeline survives. Override via env if needed.
    max_points = int(os.getenv("DECROOM_SPATIALLM_MAX_POINTS", "200000"))
    if len(slm_points) > max_points:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(len(slm_points), max_points, replace=False)
        slm_points = slm_points[idx]
        colors = colors[idx]
        logger.info(
            "Subsampled point cloud %d -> %d for SpatialLM",
            len(positions), max_points,
        )

    # Set confidence the parser will stamp on each object.
    confidence = float(getattr(cfg, "spatiallm_confidence", 0.85))
    parse_spatiallm_output._confidence = confidence

    ply_path = Path(tempfile.gettempdir()) / "decroom_phone_first_spatiallm.ply"
    write_colored_ply(ply_path, slm_points, colors)

    raw_text = run_spatiallm_inference(ply_path, cfg)
    if raw_text is None:
        raise RuntimeError("SpatialLM: inference returned no output")

    layout, objects = parse_spatiallm_output(raw_text, coord_offset)
    return layout, objects


# ── Main entry point (called from main.py route) ─────────────────────────────

def run_spatiallm_pipeline(points: "np.ndarray", cfg: Any) -> Tuple[List[Any], List[Any]]:
    """
    Thin wrapper that lets tests patch a single symbol.

    Returns (walls_list, objects_list). Walls come from parse_spatiallm_output's
    layout dict's "walls" field.
    """
    # Colors are not essential for SpatialLM inference, but its NormalizeColor
    # transform z-scores per-batch — a uniform grey input has zero std and
    # the divide produces a CUDA SIGFPE / "Floating point exception". Use
    # a small Gaussian noise around 128 so std > 0 without changing what
    # the model "sees".
    n = points.shape[0]
    rng = np.random.default_rng(seed=0)
    colors = np.clip(
        128 + rng.normal(0, 4, size=(n, 3)),
        0, 255,
    ).astype(np.uint8)
    layout, objects = _run_spatiallm_on_phone_ply(points, colors, cfg)
    walls = layout.get("walls", []) if isinstance(layout, dict) else []
    return walls, objects


def run_phone_first_pipeline(
    payload: PhoneFirstPayload,
    cfg: Any,
) -> Dict:
    """Entry point. Returns a dict matching RoomPayload.to_dict()."""
    t0 = time.time()
    logger.info("Phone-first pipeline start (scan_id=%s)",
                 payload.scan_log.get("scan_id", "?"))

    positions, _colors = load_ply_points(payload.point_cloud_ply)
    logger.info("PLY loaded: %d points", positions.shape[0])

    # SpatialLM is optional in Plan 3a — if it fails (GPU tunnel down,
    # model not loaded, network error, etc.), we still return the phone's
    # walls as a valid RoomPayload with empty objects. The iOS app shows
    # the ".noFurnitureDetected" info toast in that case, and the scan
    # saves with walls-only. When SpatialLM is reachable again on a later
    # scan, furniture detection resumes automatically.
    slm_walls: List[Any] = []
    slm_objects: List[Any] = []
    try:
        slm_walls, slm_objects = run_spatiallm_pipeline(points=positions, cfg=cfg)
    except Exception as e:
        logger.warning(
            "SpatialLM inference unavailable — returning walls-only: %s", e
        )

    check_wall_agreement(payload.room_geometry["walls"], slm_walls or [])

    manhattan_yaw = float(payload.room_geometry.get("manhattan_yaw", 0.0))
    room_info = room_envelope_from_walls(
        payload.room_geometry["walls"],
        payload.room_geometry["floor_height"],
        payload.room_geometry["ceiling_height"],
        positions=positions,
        manhattan_yaw=manhattan_yaw,
    )
    objects = map_spatiallm_objects_to_room_objects(slm_objects or [])

    result = RoomPayload(room=room_info, objects=objects).to_dict()
    elapsed = time.time() - t0
    logger.info(
        "Phone-first pipeline done: %d walls, %d objects, %.2fs",
        len(room_info.walls), len(objects), elapsed,
    )
    return result
