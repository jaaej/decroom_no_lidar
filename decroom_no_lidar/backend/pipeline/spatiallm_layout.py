"""
SpatialLM integration for the Decroom pipeline.

Replaces heuristic layout estimation + 3D object lifting with a learned model
that takes a dense colored point cloud and outputs room walls + oriented
3D furniture bounding boxes.

Pipeline:
  1. Back-project depth maps from multiple keyframes → dense colored point cloud
  2. Convert ARKit Y-up → SpatialLM Z-up coordinates
  3. Write PLY file
  4. Run SpatialLM inference (remote GPU or local)
  5. Parse structured output → layout dict + object list
  6. Optionally merge with YOLO-tracked objects
"""
from __future__ import annotations

import logging
import re
import struct
import tempfile
from math import atan2, cos, pi, sin, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("decroom-backend.spatiallm")

try:
    import numpy as np
except ImportError:
    np = None

from .depth import _load_image
from .metadata import ShotMetadata

# ── Category mapping: SpatialLM 59 categories → Decroom 14 categories ──

SPATIALLM_TO_PRD: Dict[str, str] = {
    "bed": "bed",
    "single_bed": "bed",
    "double_bed": "bed",
    "bunk_bed": "bed",
    "sofa": "sofa",
    "l_shaped_sofa": "sofa",
    "armchair": "chair",
    "chair": "chair",
    "office_chair": "chair",
    "dining_chair": "chair",
    "swivel_chair": "chair",
    "stool": "chair",
    "desk": "desk",
    "table": "table",
    "dining_table": "desk",
    "coffee_table": "table",
    "side_table": "nightstand",
    "nightstand": "nightstand",
    "wardrobe": "dresser",
    "dresser": "dresser",
    "chest_of_drawers": "dresser",
    "cabinet": "dresser",
    "tv_stand": "dresser",
    "tv_cabinet": "dresser",
    "bookshelf": "bookshelf",
    "shelf": "bookshelf",
    "monitor": "monitor",
    "tv": "monitor",
    "floor_lamp": "lamp_floor",
    "chandelier": "lamp_floor",
    "desk_lamp": "lamp_desk",
    "table_lamp": "lamp_desk",
    "plant": "plant",
    "potted_plant": "plant",
    "plants": "plant",
    "rug": "rug",
    "carpet": "rug",
    "painting": "poster",
    "picture_frame": "poster",
    "curtain": "poster",
}


# ═══════════════════════════════════════════════════════════════════════
# 1. Dense Colored Point Cloud
# ═══════════════════════════════════════════════════════════════════════

def build_dense_colored_pointcloud(
    frames: List[Any],
    depth_maps: List[Optional[Any]],
    shots: List[ShotMetadata],
    cfg: Any,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Back-project depth maps into a unified colored 3D point cloud.

    Returns (points_Nx3, colors_Nx3) in ARKit Y-up coordinates,
    or (None, None) if insufficient data.
    """
    if np is None:
        return None, None

    max_kf = int(getattr(cfg, "spatiallm_max_keyframes", 30))
    stride = max(1, int(getattr(cfg, "spatiallm_pixel_stride", 4)))
    max_pts = int(getattr(cfg, "spatiallm_max_points", 500_000))
    max_depth = float(getattr(cfg, "spatiallm_max_depth_m", 5.0))
    voxel_size = float(getattr(cfg, "spatiallm_voxel_size_m", 0.02))

    # Select keyframes with spatial diversity
    selected = _select_diverse_keyframes(shots, depth_maps, max_kf)
    if not selected:
        logger.warning("SpatialLM: no valid keyframes for point cloud")
        return None, None

    logger.info("SpatialLM: building dense cloud from %d keyframes", len(selected))

    all_pts: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []

    for fi in selected:
        depth = depth_maps[fi]
        if depth is None:
            continue
        shot = shots[fi]
        if shot.pose is None or len(shot.pose) != 16 or shot.intrinsics is None:
            continue

        h, w = depth.shape[:2]
        intr = shot.intrinsics.scaled_to(w, h)

        # Grid of pixel coordinates
        ys = np.arange(0, h, stride, dtype=np.int64)
        xs = np.arange(0, w, stride, dtype=np.int64)
        if len(xs) == 0 or len(ys) == 0:
            continue

        grid_x, grid_y = np.meshgrid(xs, ys)
        uu = grid_x.reshape(-1)
        vv = grid_y.reshape(-1)
        dd = depth[vv, uu]

        valid = np.isfinite(dd) & (dd > 0.1) & (dd < max_depth)
        if not np.any(valid):
            continue

        uu = uu[valid].astype(np.float64)
        vv = vv[valid].astype(np.float64)
        dd = dd[valid].astype(np.float64)

        # Unproject to camera space
        cam_x = (uu - intr.cx) * dd / max(1e-6, intr.fx)
        cam_y = (vv - intr.cy) * dd / max(1e-6, intr.fy)
        cam_z = dd

        # Transform to world space
        pts_cam = np.stack([cam_x, cam_y, cam_z, np.ones_like(cam_x)], axis=1)
        mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
        pts_world = (mat @ pts_cam.T).T[:, :3]
        all_pts.append(pts_world)

        # Sample RGB color from the frame image
        rgb_img = None
        if fi < len(frames):
            frame = frames[fi]
            if hasattr(frame, "shape") and len(frame.shape) == 3:
                rgb_img = frame
            else:
                rgb_img = _load_image(frame)

        if rgb_img is not None and len(rgb_img.shape) == 3:
            fh, fw = rgb_img.shape[:2]
            sx = fw / w
            sy = fh / h
            fu = np.clip((uu * sx).astype(np.int64), 0, fw - 1)
            fv = np.clip((vv * sy).astype(np.int64), 0, fh - 1)
            colors = rgb_img[fv, fu][:, :3]  # RGB
            all_colors.append(colors.astype(np.uint8))
        else:
            all_colors.append(np.full((len(pts_world), 3), 128, dtype=np.uint8))

    if not all_pts:
        return None, None

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    logger.info("SpatialLM: raw cloud %d points", len(points))

    # Remove outliers: points > 3 sigma from median
    median = np.median(points, axis=0)
    dists = np.linalg.norm(points - median, axis=1)
    cutoff = np.median(dists) + 3.0 * np.std(dists)
    mask = dists < cutoff
    points = points[mask]
    colors = colors[mask]

    # Voxel downsample
    if voxel_size > 0 and len(points) > 0:
        points, colors = _voxel_downsample(points, colors, voxel_size)

    # Subsample to max_pts
    if len(points) > max_pts:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(points), size=max_pts, replace=False)
        points = points[idx]
        colors = colors[idx]

    logger.info("SpatialLM: final cloud %d points", len(points))
    return points, colors


def _select_diverse_keyframes(
    shots: List[ShotMetadata],
    depth_maps: List[Optional[Any]],
    max_kf: int,
    min_dist: float = 0.15,
) -> List[int]:
    """Select keyframe indices with minimum spatial diversity."""
    if np is None:
        return []
    candidates = []
    for i, shot in enumerate(shots):
        if i >= len(depth_maps) or depth_maps[i] is None:
            continue
        if shot.pose is None or len(shot.pose) != 16:
            continue
        candidates.append(i)

    if not candidates:
        return []
    if len(candidates) <= max_kf:
        return candidates

    # Greedy farthest-point sampling
    selected = [candidates[0]]
    positions = {}
    for ci in candidates:
        p = shots[ci].pose
        positions[ci] = np.array([p[3], p[7], p[11]], dtype=np.float64)

    while len(selected) < max_kf:
        best_idx = -1
        best_dist = -1.0
        for ci in candidates:
            if ci in selected:
                continue
            d = min(np.linalg.norm(positions[ci] - positions[si]) for si in selected)
            if d > best_dist:
                best_dist = d
                best_idx = ci
        if best_idx < 0 or best_dist < min_dist * 0.5:
            break
        selected.append(best_idx)

    return sorted(selected)


def _voxel_downsample(
    points: Any, colors: Any, voxel_size: float
) -> Tuple[Any, Any]:
    """Voxel downsample: average points and colors within each cell."""
    keys = np.floor(points / voxel_size).astype(np.int64)
    # Hash voxel keys
    primes = np.array([73856093, 19349663, 83492791], dtype=np.int64)
    hashes = (keys * primes).sum(axis=1)
    unique_hashes, inverse = np.unique(hashes, return_inverse=True)
    n_voxels = len(unique_hashes)

    out_pts = np.zeros((n_voxels, 3), dtype=np.float64)
    out_colors = np.zeros((n_voxels, 3), dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=np.float64)

    np.add.at(out_pts, inverse, points)
    np.add.at(out_colors, inverse, colors.astype(np.float64))
    np.add.at(counts, inverse, 1.0)

    mask = counts > 0
    out_pts[mask] /= counts[mask, None]
    out_colors[mask] /= counts[mask, None]

    return out_pts[mask], np.clip(out_colors[mask], 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════
# 2. Coordinate Conversion
# ═══════════════════════════════════════════════════════════════════════

def arkit_to_spatiallm_coords(
    points: Any,
) -> Tuple[Any, Any]:
    """
    Convert ARKit Y-up to SpatialLM Z-up coordinate system.

    ARKit (X, Y, Z) → SpatialLM (X, -Z, Y)
    Then shift to positive coordinates.

    Returns (shifted_points, offset) where offset is the min that was subtracted.
    """
    out = np.empty_like(points)
    out[:, 0] = points[:, 0]      # X stays X
    out[:, 1] = -points[:, 2]     # Z → -Y (negate)
    out[:, 2] = points[:, 1]      # Y → Z (up)

    offset = out.min(axis=0)
    out -= offset  # make all positive

    return out, offset


def spatiallm_to_arkit_coords(
    slm_points: Any,
    offset: Any,
) -> Any:
    """
    Inverse of arkit_to_spatiallm_coords.

    SpatialLM (X, Y, Z) + offset → ARKit (X, Y, Z)
    """
    # Undo the positive shift
    pts = slm_points + offset

    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]      # X stays X
    out[:, 1] = pts[:, 2]      # SLM Z → ARKit Y (up)
    out[:, 2] = -pts[:, 1]     # SLM Y → ARKit -Z
    return out


# ═══════════════════════════════════════════════════════════════════════
# 3. PLY Writer
# ═══════════════════════════════════════════════════════════════════════

def write_colored_ply(
    path: Path,
    points: Any,
    colors: Any,
) -> None:
    """Write a binary PLY file with XYZ + RGB vertex properties."""
    n = len(points)
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
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack(
                "<fffBBB",
                float(points[i, 0]),
                float(points[i, 1]),
                float(points[i, 2]),
                int(colors[i, 0]),
                int(colors[i, 1]),
                int(colors[i, 2]),
            ))
    logger.info("SpatialLM: wrote PLY %s (%d vertices)", path, n)


# ═══════════════════════════════════════════════════════════════════════
# 4. SpatialLM Inference
# ═══════════════════════════════════════════════════════════════════════

def run_spatiallm_inference(
    ply_path: Path,
    cfg: Any,
) -> Optional[str]:
    """
    Run SpatialLM inference on a PLY file.

    Supports two modes:
    - remote: POST PLY to cfg.spatiallm_api_url
    - local: load model via transformers (requires CUDA)

    Returns raw text output from SpatialLM or None on failure.
    """
    mode = str(getattr(cfg, "spatiallm_mode", "remote"))

    if mode == "remote":
        return _inference_remote(ply_path, cfg)
    elif mode == "local":
        return _inference_local(ply_path, cfg)
    else:
        logger.error("SpatialLM: unknown mode '%s'", mode)
        return None


def _inference_remote(ply_path: Path, cfg: Any) -> Optional[str]:
    """POST PLY file to remote SpatialLM inference server."""
    import requests

    url = str(getattr(cfg, "spatiallm_api_url", "http://localhost:8100/infer"))
    timeout = float(getattr(cfg, "spatiallm_timeout_s", 120.0))

    logger.info("SpatialLM: sending PLY to %s", url)
    try:
        with open(ply_path, "rb") as f:
            resp = requests.post(
                url,
                files={"point_cloud": (ply_path.name, f, "application/octet-stream")},
                timeout=timeout,
            )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("output", data.get("text", data.get("result", "")))
        if not text:
            logger.warning("SpatialLM remote: empty response")
            return None
        logger.info("SpatialLM remote: got %d chars", len(text))
        return text
    except Exception as e:
        logger.error("SpatialLM remote inference failed: %s", e)
        return None


def _inference_local(ply_path: Path, cfg: Any) -> Optional[str]:
    """Run SpatialLM locally using transformers (requires CUDA GPU)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error("SpatialLM local: missing dependency: %s", e)
        return None

    if not torch.cuda.is_available():
        logger.error("SpatialLM local: CUDA not available")
        return None

    model_id = str(getattr(cfg, "spatiallm_model_id",
                            "manycore-research/SpatialLM1.1-Qwen-0.5B"))

    try:
        # Lazy load — cache for reuse across calls
        if not hasattr(_inference_local, "_model"):
            logger.info("SpatialLM local: loading model %s", model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            model.to("cuda")
            model.eval()
            if hasattr(model, "set_point_backbone_dtype"):
                model.set_point_backbone_dtype(torch.float32)
            _inference_local._model = model
            _inference_local._tokenizer = tokenizer

        model = _inference_local._model
        tokenizer = _inference_local._tokenizer

        # Load and preprocess point cloud
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(ply_path))
        # Use SpatialLM's preprocessing if available
        try:
            from spatiallm import preprocess_point_cloud, generate_layout
            from spatiallm.utils import cleanup_pcd, get_points_and_colors

            grid_size = 32.0 / 1280
            pcd = cleanup_pcd(pcd, voxel_size=grid_size)
            points, colors = get_points_and_colors(pcd)
            input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins=1280)

            layout = generate_layout(model, input_pcd, tokenizer, None)
            return layout.to_language_string()
        except ImportError:
            logger.warning("SpatialLM local: spatiallm package not installed, "
                         "falling back to basic inference")
            # Basic inference without spatiallm package
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            prompt = "Detect walls, doors, windows, boxes."
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text

    except Exception as e:
        logger.error("SpatialLM local inference failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════
# 5. Output Parser
# ═══════════════════════════════════════════════════════════════════════

_WALL_RE = re.compile(
    r"wall_\d+=Wall\(\s*"
    r"([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*,\s*"
    r"([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*,\s*"
    r"([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*\)"
)

_BBOX_RE = re.compile(
    r"bbox_\d+=Bbox\(\s*"
    r"(\w+)\s*,\s*"
    r"([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*,\s*"
    r"([+-]?[\d.]+)\s*,\s*"
    r"([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*\)"
)


def parse_spatiallm_output(
    text: str,
    coord_offset: Any,
    room_height_fallback: float = 2.5,
) -> Tuple[Dict, List[Dict]]:
    """
    Parse SpatialLM structured text into layout dict + object list.

    Returns formats matching estimate_layout() and lift_objects() outputs.
    """
    walls_raw = _WALL_RE.findall(text)
    bboxes_raw = _BBOX_RE.findall(text)

    logger.info("SpatialLM parse: %d walls, %d bboxes", len(walls_raw), len(bboxes_raw))

    # ── Parse walls ──
    walls = []
    wall_heights = []
    all_wall_pts_arkit = []

    for i, w in enumerate(walls_raw):
        ax, ay, az, bx, by, bz, height, thickness = [float(v) for v in w]
        # Convert wall endpoints from SpatialLM coords back to ARKit
        slm_a = np.array([[ax, ay, az]])
        slm_b = np.array([[bx, by, bz]])
        arkit_a = spatiallm_to_arkit_coords(slm_a, coord_offset)[0]
        arkit_b = spatiallm_to_arkit_coords(slm_b, coord_offset)[0]

        # Project to XZ floor plane (ARKit convention)
        walls.append({
            "id": f"wall_{i}",
            "start": [round(float(arkit_a[0]), 4), round(float(arkit_a[2]), 4)],
            "end": [round(float(arkit_b[0]), 4), round(float(arkit_b[2]), 4)],
            "has_window": False,
            "has_door": False,
        })
        wall_heights.append(height)
        all_wall_pts_arkit.extend([arkit_a, arkit_b])

    # Compute room dimensions from wall extents
    if all_wall_pts_arkit:
        pts = np.array(all_wall_pts_arkit)
        min_x, min_z = pts[:, 0].min(), pts[:, 2].min()
        max_x, max_z = pts[:, 0].max(), pts[:, 2].max()
        width = max(2.0, min(12.0, float(max_x - min_x)))
        length = max(2.0, min(12.0, float(max_z - min_z)))
        room_center = [
            round(float((min_x + max_x) / 2), 4),
            round(float((min_z + max_z) / 2), 4),
        ]
        floor_height = round(float(pts[:, 1].min()), 4)
    else:
        width = 4.0
        length = 4.0
        room_center = [0.0, 0.0]
        floor_height = 0.0

    room_height = max(2.2, min(3.5,
        float(np.median(wall_heights)) if wall_heights else room_height_fallback
    ))

    layout = {
        "width_m": round(width, 2),
        "length_m": round(length, 2),
        "height_m": round(room_height, 2),
        "floor_height": floor_height,
        "room_center": room_center,
        "room_yaw_rad": 0.0,  # SpatialLM output is already axis-aligned
        "axis_confidence": 0.95,
        "walls": walls,
        "layout_source": "spatiallm",
    }

    # ── Parse bounding boxes ──
    objects = []
    prd_counts: Dict[str, int] = {}

    for match in bboxes_raw:
        class_name = match[0].lower()
        pos_x, pos_y, pos_z = float(match[1]), float(match[2]), float(match[3])
        angle_z = float(match[4])
        scale_x, scale_y, scale_z = float(match[5]), float(match[6]), float(match[7])

        prd_type = SPATIALLM_TO_PRD.get(class_name)
        if prd_type is None:
            logger.debug("SpatialLM: unmapped category '%s', skipping", class_name)
            continue

        # Convert position from SpatialLM to ARKit
        slm_pos = np.array([[pos_x, pos_y, pos_z]])
        arkit_pos = spatiallm_to_arkit_coords(slm_pos, coord_offset)[0]

        # SpatialLM angle_z (rotation around Z-up) maps to rotation_y (around Y-up)
        # Both describe yaw in the horizontal plane
        rotation_y = float(angle_z)

        # Generate unique ID
        count = prd_counts.get(prd_type, 0)
        prd_counts[prd_type] = count + 1
        obj_id = f"slm_{prd_type}_{count}"

        # Place object on floor: y = floor_height + scale_z/2 (half-height above floor)
        # In ARKit, Y is up. SpatialLM scale_z is height (Z-up), which maps to ARKit Y.
        obj_y = floor_height + scale_z / 2.0

        objects.append({
            "id": obj_id,
            "type": prd_type,
            "asset_id": "",
            "position": {
                "x": round(float(arkit_pos[0]), 4),
                "y": round(float(obj_y), 4),
                "z": round(float(arkit_pos[2]), 4),
            },
            "rotation_y": round(rotation_y, 4),
            "scale": 1.0,
            "confidence": float(getattr(parse_spatiallm_output, "_confidence", 0.85)),
            "metadata": {
                "width": round(float(scale_x), 3),
                "length": round(float(scale_y), 3),
                "height": round(float(scale_z), 3),
                "inferred": False,
            },
            "_source": "spatiallm",
        })

    logger.info("SpatialLM parse: %d objects after category mapping", len(objects))
    return layout, objects


# ═══════════════════════════════════════════════════════════════════════
# 6. Merge with YOLO Objects
# ═══════════════════════════════════════════════════════════════════════

def merge_spatiallm_and_yolo_objects(
    slm_objects: List[Dict],
    yolo_objects: List[Dict],
    merge_distance_m: float = 1.0,
) -> List[Dict]:
    """
    Merge SpatialLM objects with YOLO-tracked lifted objects.

    Strategy:
    - For each YOLO object, find nearest SpatialLM object of same type.
    - If distance < merge_distance_m: prefer SpatialLM position (more reliable).
    - Unmatched YOLO objects: keep (SpatialLM may miss them).
    - Unmatched SpatialLM objects: keep (YOLO may miss them).
    """
    if not slm_objects:
        return yolo_objects
    if not yolo_objects:
        return slm_objects

    merged = list(slm_objects)  # Start with all SpatialLM objects
    slm_matched = set()

    for yolo_obj in yolo_objects:
        yolo_type = yolo_obj.get("type", "")
        yolo_pos = yolo_obj.get("position", {})
        yx = float(yolo_pos.get("x", 0))
        yz = float(yolo_pos.get("z", 0))

        best_idx = -1
        best_dist = float("inf")

        for i, slm_obj in enumerate(slm_objects):
            if i in slm_matched:
                continue
            if slm_obj.get("type") != yolo_type:
                continue
            sx = float(slm_obj["position"]["x"])
            sz = float(slm_obj["position"]["z"])
            d = sqrt((yx - sx) ** 2 + (yz - sz) ** 2)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx >= 0 and best_dist < merge_distance_m:
            slm_matched.add(best_idx)
            # SpatialLM position is already in merged, skip YOLO duplicate
        else:
            # No SpatialLM match — keep YOLO object
            merged.append(yolo_obj)

    return merged


# ═══════════════════════════════════════════════════════════════════════
# 7. Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def run_spatiallm_pipeline(
    frames: List[Any],
    depth_maps: List[Optional[Any]],
    shots: List[ShotMetadata],
    cfg: Any,
    yolo_objects: Optional[List[Dict]] = None,
) -> Tuple[Dict, List[Dict]]:
    """
    Run the full SpatialLM pipeline.

    Returns (layout_dict, objects_list) in the same format as
    estimate_layout() + lift_objects().

    Raises RuntimeError if point cloud is too sparse or inference fails.
    """
    min_pts = int(getattr(cfg, "spatiallm_min_pointcloud_size", 50_000))
    confidence = float(getattr(cfg, "spatiallm_confidence", 0.85))
    parse_spatiallm_output._confidence = confidence

    # Step 1: Build dense colored point cloud
    points, colors = build_dense_colored_pointcloud(frames, depth_maps, shots, cfg)
    if points is None or len(points) < min_pts:
        raise RuntimeError(
            f"SpatialLM: point cloud too sparse ({0 if points is None else len(points)} < {min_pts})"
        )

    # Step 2: Convert coordinates
    slm_points, coord_offset = arkit_to_spatiallm_coords(points)

    # Step 3: Write PLY
    ply_path = Path(tempfile.gettempdir()) / "decroom_spatiallm_input.ply"
    write_colored_ply(ply_path, slm_points, colors)

    # Also save a debug copy of the ARKit-frame point cloud
    if bool(getattr(cfg, "debug_output", False)):
        debug_ply = Path(tempfile.gettempdir()) / "decroom_pointcloud.ply"
        write_colored_ply(debug_ply, points, colors)

    # Step 4: Run inference
    raw_text = run_spatiallm_inference(ply_path, cfg)
    if raw_text is None:
        raise RuntimeError("SpatialLM: inference returned no output")

    # Save raw output for debugging
    if bool(getattr(cfg, "debug_output", False)):
        debug_txt = Path(tempfile.gettempdir()) / "decroom_spatiallm_output.txt"
        debug_txt.write_text(raw_text)

    # Step 5: Parse output
    layout, slm_objects = parse_spatiallm_output(raw_text, coord_offset)

    # Step 6: Merge with YOLO objects
    if yolo_objects and bool(getattr(cfg, "spatiallm_merge_with_yolo", True)):
        merge_dist = float(getattr(cfg, "spatiallm_merge_distance_m", 1.0))
        objects = merge_spatiallm_and_yolo_objects(slm_objects, yolo_objects, merge_dist)
    else:
        objects = slm_objects

    logger.info(
        "SpatialLM pipeline: layout=%s, %d walls, %d objects",
        layout.get("layout_source"),
        len(layout.get("walls", [])),
        len(objects),
    )

    return layout, objects
