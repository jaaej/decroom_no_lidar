"""
Shared 3D geometry utilities for the Decroom pipeline.

Provides robust_lift_detection() — the single authoritative function for
lifting a 2D detection to a 3D world-space position. Used by both
tracking.py (for cross-frame matching) and lifting.py (for final placement),
ensuring tracking decisions and final positions are always consistent.

Key improvements over the previous single-point approach:
  - Multi-point back-projection: samples many pixels → back-projects each → median world pos
  - Adaptive percentile depth selection: biases toward nearest surface (the object, not the wall behind)
  - Instance mask support with erosion to kill boundary noise
  - Foreground occluder guardrail
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .detection import Detection
from .metadata import CameraIntrinsics, ShotMetadata

logger = logging.getLogger("decroom-backend.geometry")
_fp_overlay_saved: set = set()
_fp_overlay_limit = 24
_fp_good_overlay_saved = False


def _save_fp_debug_overlay(
    det: Detection,
    img_w: int,
    img_h: int,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
    u: "np.ndarray",
    v: "np.ndarray",
    inside: "np.ndarray",
    source_tag: str,
    title: Optional[str] = None,
) -> None:
    """Save a debug image showing FP UVs vs detection box for mismatch diagnosis."""
    if np is None:
        return
    key = (det.frame_idx, source_tag)
    if key in _fp_overlay_saved:
        return
    if len(_fp_overlay_saved) >= _fp_overlay_limit:
        return
    _fp_overlay_saved.add(key)

    try:
        import cv2
    except ImportError:
        return

    bg_path = f"/tmp/decroom_det_{det.frame_idx}.png"
    img = cv2.imread(bg_path)
    if img is None:
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        img[:, :] = (25, 25, 25)
    else:
        img = cv2.resize(img, (img_w, img_h))

    x1 = int(max(0, min(img_w - 1, bx1)))
    y1 = int(max(0, min(img_h - 1, by1)))
    x2 = int(max(0, min(img_w - 1, bx2)))
    y2 = int(max(0, min(img_h - 1, by2)))
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

    for i in range(len(u)):
        px = int(round(float(u[i])))
        py = int(round(float(v[i])))
        if px < 0 or py < 0 or px >= img_w or py >= img_h:
            continue
        color = (0, 220, 0) if bool(inside[i]) else (0, 0, 255)
        cv2.circle(img, (px, py), 2, color, -1, cv2.LINE_AA)

    inside_count = int(np.sum(inside))
    cv2.putText(
        img,
        title or f"{source_tag} | {det.category} | fp={len(u)} in_bbox={inside_count}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_path = f"/tmp/decroom_fp_overlay_f{det.frame_idx}_{det.category}_{source_tag}.png"
    cv2.imwrite(out_path, img)


def _pixel_space_contract(
    det: Detection,
    shot: ShotMetadata,
    img_w: int,
    img_h: int,
) -> Dict:
    """Describe the pixel-space transform assumptions for FP↔bbox association."""
    det_w, det_h = det.input_size
    if det_w == img_w and det_h == img_h:
        bbox_space = "original_upload_pixels"
    elif det_w == 640 and det_h == 640:
        bbox_space = "detector_input_pixels"
    else:
        bbox_space = "post_letterbox_pixels"

    tf = shot.image_transform
    if tf is None:
        image_transform = {
            "rotate_deg": 0,
            "flip_x": False,
            "flip_y": False,
            "crop": [0, 0, img_w, img_h],
            "resize": [img_w, img_h],
            "letterbox_pad": [0, 0],
        }
    else:
        image_transform = {
            "rotate_deg": int(tf.rotate_deg),
            "flip_x": bool(tf.flip_x),
            "flip_y": bool(tf.flip_y),
            "crop": list(tf.crop),
            "resize": list(tf.resize),
            "letterbox_pad": list(tf.letterbox_pad),
        }

    return {
        "image_transform": image_transform,
        "detector_input_size": [int(det_w), int(det_h)],
        "bbox_space": bbox_space,
    }


# ── Feature-Point-Anchored Geometry (Session 11) ─────────────────────────
# ARKit feature points are metric (from VIO-SLAM). These functions make
# them the PRIMARY geometry source, replacing depth-based back-projection.

def build_global_feature_cloud(
    shots: List[ShotMetadata],
) -> Optional["np.ndarray"]:
    """
    Build an authoritative metric scene point cloud from ARKit feature points.

    Concatenates all feature_points_w across all frames, removes outliers,
    and deduplicates via voxel-grid filter (0.02m resolution).

    Returns:
        np.ndarray of shape (M, 3) in world coordinates, or None if too few.
    """
    if np is None:
        return None

    all_pts = []
    for shot in shots:
        if shot.feature_points_w is None:
            continue
        for fp in shot.feature_points_w:
            if len(fp) == 3:
                all_pts.append(fp)

    if len(all_pts) < 20:
        logger.warning("build_global_feature_cloud: only %d points (need 20+)", len(all_pts))
        return None

    pts = np.array(all_pts, dtype=np.float64)  # (N, 3)

    # Remove NaN / inf
    valid = np.all(np.isfinite(pts), axis=1)
    pts = pts[valid]

    if len(pts) < 20:
        return None

    # Remove extreme outliers: >10m from median
    median_xyz = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - median_xyz, axis=1)
    pts = pts[dists < 10.0]

    if len(pts) < 20:
        return None

    # Voxel-grid deduplication at 0.02m resolution
    # Quantize → find unique voxels → keep one point per voxel
    voxel_size = 0.02
    voxel_coords = np.floor(pts / voxel_size).astype(np.int64)
    # Use a structured array for unique
    voxel_keys = voxel_coords[:, 0] * 1_000_000 + voxel_coords[:, 1] * 1_000 + voxel_coords[:, 2]
    _, unique_idx = np.unique(voxel_keys, return_index=True)
    pts = pts[unique_idx]

    logger.info("build_global_feature_cloud: %d unique points (from %d raw)", len(pts), len(all_pts))
    return pts


def associate_fp_to_detection(
    det: Detection,
    shot: ShotMetadata,
    return_diag: bool = False,
) -> "np.ndarray":
    """
    Find ARKit feature points that project inside a 2D detection bbox.

    Uses portrait-corrected projection (cam_y → u, -cam_x → v) matching
    the pipeline.py X/Y swap fix from Session 10.

    Applies depth consistency filter: only keeps points within the
    [p25 - margin, p75 + margin] depth range to reject background/floor.

    Args:
        return_diag: If True, returns (points, diag_dict) instead of just points.

    Returns:
        np.ndarray of shape (M, 3) — world-space points on the object surface.
        Empty array (0, 3) if no points found.
        If return_diag=True: tuple of (points, {"fp_visible": int, "fp_behind": int})
    """
    empty = np.empty((0, 3), dtype=np.float64) if np is not None else []
    empty_diag = {"fp_visible": 0, "fp_behind": 0, "pixel_space_contract": None}
    global _fp_good_overlay_saved

    def _ret(pts, diag=None):
        if return_diag:
            return pts, diag if diag else empty_diag
        return pts

    if np is None:
        return _ret(empty)
    if shot.feature_points_w is None or not shot.feature_points_w:
        return _ret(empty)
    if shot.intrinsics is None:
        return _ret(empty)

    inv_mat = None
    if shot.pose is not None and len(shot.pose) == 16:
        mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
        try:
            inv_mat = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            inv_mat = None

    intr = shot.intrinsics
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.cx, intr.cy
    img_w, img_h = intr.width, intr.height

    # Detection bbox (in portrait pixel space, same as intrinsics)
    bx1, by1, bx2, by2 = det.bbox

    # Scale bbox from detection resolution to intrinsics resolution
    det_w, det_h = det.input_size
    sx = img_w / det_w
    sy = img_h / det_h
    bx1_s = bx1 * sx
    by1_s = by1 * sy
    bx2_s = bx2 * sx
    by2_s = by2 * sy

    # Expand bbox slightly to compensate for detector box tightness vs sparse
    # ARKit feature points. This improves FP association for thin/partial objects.
    bw = max(1.0, bx2_s - bx1_s)
    bh = max(1.0, by2_s - by1_s)
    pad = max(8.0, 0.08 * max(bw, bh))
    bx1_s = max(0.0, bx1_s - pad)
    by1_s = max(0.0, by1_s - pad)
    bx2_s = min(float(img_w - 1), bx2_s + pad)
    by2_s = min(float(img_h - 1), by2_s + pad)

    # ── Session 12 debug: dump projection details ──
    logger.debug(
        "FP_PROJ frame=%d cat=%s | intr: fx=%.1f fy=%.1f cx=%.1f cy=%.1f img=%dx%d | "
        "det_input=%dx%d bbox=(%.0f,%.0f,%.0f,%.0f) → scaled=(%.1f,%.1f,%.1f,%.1f) | "
        "n_fp=%d",
        det.frame_idx, det.category,
        fx, fy, cx, cy, img_w, img_h,
        det_w, det_h, bx1, by1, bx2, by2,
        bx1_s, by1_s, bx2_s, by2_s,
        len(shot.feature_points_w) if shot.feature_points_w else 0,
    )
    px_contract = _pixel_space_contract(det, shot, img_w=img_w, img_h=img_h)

    # Convert feature points to array
    fp_arr = np.array(shot.feature_points_w, dtype=np.float64)
    if fp_arr.ndim != 2 or fp_arr.shape[1] != 3:
        return _ret(empty)

    # ── Path A: Use client-projected UVs directly (preferred) ─────────
    fp_uv = shot.feature_points_uv
    if fp_uv is not None and len(fp_uv) == len(fp_arr):
        uv_arr = np.array(fp_uv, dtype=np.float64)
        if uv_arr.ndim == 2 and uv_arr.shape[1] == 2:
            u_all = uv_arr[:, 0]
            v_all = uv_arr[:, 1]
            uv_valid = np.isfinite(u_all) & np.isfinite(v_all)
            uv_valid &= (u_all >= 0.0) & (u_all < float(img_w))
            uv_valid &= (v_all >= 0.0) & (v_all < float(img_h))

            n_total = len(fp_arr)
            n_visible = int(np.sum(uv_valid))
            n_behind = n_total - n_visible
            if n_visible == 0:
                return _ret(empty, {
                    "fp_visible": 0,
                    "fp_behind": n_total,
                    "bbox_space": "client_uv",
                    "fp_uv_bounds": None,
                    "in_mask": 0,
                    "pixel_space_contract": px_contract,
                })

            visible_indices = np.where(uv_valid)[0]
            u_vis = u_all[uv_valid]
            v_vis = v_all[uv_valid]
            inside_bbox = (u_vis >= bx1_s) & (u_vis <= bx2_s) & (v_vis >= by1_s) & (v_vis <= by2_s)
            inside = inside_bbox
            bbox_space = "client_uv_bbox"

            # Optional mask-first hit test when mask exists in the same image space.
            if det.mask is not None and getattr(det.mask, "shape", None) is not None:
                mh, mw = det.mask.shape[:2]
                if mh == img_h and mw == img_w:
                    uu = np.clip(np.round(u_vis).astype(np.int64), 0, img_w - 1)
                    vv = np.clip(np.round(v_vis).astype(np.int64), 0, img_h - 1)
                    mask_hits = det.mask[vv, uu] > 0
                    in_mask_count = int(np.sum(mask_hits))
                    if int(np.sum(mask_hits)) > 0:
                        inside = mask_hits
                        bbox_space = "client_uv_mask"
                    else:
                        inside = inside_bbox
                        bbox_space = "client_uv_bbox_fallback"
                        _save_fp_debug_overlay(
                            det=det,
                            img_w=img_w,
                            img_h=img_h,
                            bx1=bx1_s,
                            by1=by1_s,
                            bx2=bx2_s,
                            by2=by2_s,
                            u=u_vis,
                            v=v_vis,
                            inside=mask_hits,
                            source_tag="client_uv_mask_miss",
                            title=f"mask_miss | {det.category} | fp={len(u_vis)} in_mask={in_mask_count}",
                        )
                else:
                    in_mask_count = 0
            else:
                in_mask_count = 0

            # Optional failure-case overlay for fast diagnosis
            if int(np.sum(inside)) == 0 and n_total >= 30:
                _save_fp_debug_overlay(
                    det=det,
                    img_w=img_w,
                    img_h=img_h,
                    bx1=bx1_s, by1=by1_s, bx2=bx2_s, by2=by2_s,
                    u=u_vis, v=v_vis, inside=inside,
                    source_tag="client_uv",
                    title=f"fp_miss | {det.category} | shot_fp={n_total} in_bbox=0",
                )

            if np.sum(inside) == 0:
                return _ret(empty, {
                    "fp_visible": n_visible,
                    "fp_behind": n_behind,
                    "bbox_space": bbox_space,
                    "in_mask": in_mask_count,
                    "fp_uv_bounds": [
                        round(float(np.percentile(u_vis, 5)), 1),
                        round(float(np.percentile(u_vis, 95)), 1),
                        round(float(np.percentile(v_vis, 5)), 1),
                        round(float(np.percentile(v_vis, 95)), 1),
                    ],
                    "pixel_space_contract": px_contract,
                })

            inside_indices = visible_indices[inside]
            inside_world = fp_arr[inside_indices]

            # Optional depth consistency filter when pose is available
            if inv_mat is not None and len(inside_world) >= 4:
                ones = np.ones((len(inside_world), 1), dtype=np.float64)
                inside_h = np.hstack([inside_world, ones])
                inside_cam = (inv_mat @ inside_h.T).T[:, :3]
                inside_depths = -inside_cam[:, 2]
                depth_ok = np.isfinite(inside_depths) & (inside_depths > 0.05)
                inside_depths = inside_depths[depth_ok]
                inside_world = inside_world[depth_ok]
                if len(inside_depths) >= 4:
                    p25 = float(np.percentile(inside_depths, 25))
                    p75 = float(np.percentile(inside_depths, 75))
                    iqr = p75 - p25
                    margin = max(0.2, 0.3 * iqr)
                    lo = p25 - margin
                    hi = p75 + margin
                    keep = (inside_depths >= lo) & (inside_depths <= hi)
                    if np.sum(keep) >= 2:
                        inside_world = inside_world[keep]

            if not _fp_good_overlay_saved and n_total >= 30 and int(np.sum(inside)) > 0:
                _save_fp_debug_overlay(
                    det=det,
                    img_w=img_w,
                    img_h=img_h,
                    bx1=bx1_s,
                    by1=by1_s,
                    bx2=bx2_s,
                    by2=by2_s,
                    u=u_vis,
                    v=v_vis,
                    inside=inside,
                    source_tag="client_uv_good",
                    title=f"good_ref | {det.category} | shot_fp={n_total} in_bbox={int(np.sum(inside))}",
                )
                _fp_good_overlay_saved = True

            return _ret(inside_world, {
                "fp_visible": n_visible,
                "fp_behind": n_behind,
                "bbox_space": bbox_space,
                "in_mask": in_mask_count,
                "fp_uv_bounds": [
                    round(float(np.percentile(u_vis, 5)), 1),
                    round(float(np.percentile(u_vis, 95)), 1),
                    round(float(np.percentile(v_vis, 5)), 1),
                    round(float(np.percentile(v_vis, 95)), 1),
                ],
                "pixel_space_contract": px_contract,
            })

    # ── Path B: Legacy backend reprojection fallback ───────────────────
    if inv_mat is None:
        return _ret(empty, {
            "fp_visible": 0,
            "fp_behind": len(fp_arr),
            "bbox_space": "legacy_reproject_no_pose",
            "in_mask": 0,
            "pixel_space_contract": px_contract,
        })

    # Transform all feature points to camera space at once
    ones = np.ones((len(fp_arr), 1), dtype=np.float64)
    fp_h = np.hstack([fp_arr, ones])  # (N, 4)
    pts_cam = (inv_mat @ fp_h.T).T[:, :3]  # (N, 3)

    # ARKit camera: -Z is forward. Visible points have pts_cam[:, 2] < 0.
    depth = -pts_cam[:, 2]  # positive for visible points
    visible = depth > 0.05
    n_total = len(fp_arr)
    n_visible = int(np.sum(visible))
    n_behind = n_total - n_visible
    if n_visible == 0:
        return _ret(empty, {
            "fp_visible": 0,
            "fp_behind": n_total,
            "bbox_space": "legacy_reproject",
            "in_mask": 0,
            "pixel_space_contract": px_contract,
        })

    # Portrait projection
    cam_x = pts_cam[visible, 0]
    cam_y = pts_cam[visible, 1]
    d_vis = depth[visible]
    u = fx * cam_y / d_vis + cx
    v = fy * (-cam_x) / d_vis + cy

    inside = (u >= bx1_s) & (u <= bx2_s) & (v >= by1_s) & (v <= by2_s)
    if int(np.sum(inside)) == 0 and n_total >= 30:
        _save_fp_debug_overlay(
            det=det,
            img_w=img_w,
            img_h=img_h,
            bx1=bx1_s, by1=by1_s, bx2=bx2_s, by2=by2_s,
            u=u, v=v, inside=inside,
            source_tag="legacy_reproject",
            title=f"fp_miss_legacy | {det.category} | shot_fp={n_total} in_bbox=0",
        )
    if np.sum(inside) == 0:
        return _ret(empty, {
            "fp_visible": n_visible,
            "fp_behind": n_behind,
            "bbox_space": "legacy_reproject",
            "in_mask": 0,
            "fp_uv_bounds": [
                round(float(np.percentile(u, 5)), 1),
                round(float(np.percentile(u, 95)), 1),
                round(float(np.percentile(v, 5)), 1),
                round(float(np.percentile(v, 95)), 1),
            ],
            "pixel_space_contract": px_contract,
        })

    visible_indices = np.where(visible)[0]
    inside_indices = visible_indices[inside]
    inside_depths = d_vis[inside]
    inside_world = fp_arr[inside_indices]

    if len(inside_depths) >= 4:
        p25 = float(np.percentile(inside_depths, 25))
        p75 = float(np.percentile(inside_depths, 75))
        iqr = p75 - p25
        margin = max(0.2, 0.3 * iqr)
        depth_lo = p25 - margin
        depth_hi = p75 + margin
        depth_ok = (inside_depths >= depth_lo) & (inside_depths <= depth_hi)
        if np.sum(depth_ok) >= 2:
            inside_world = inside_world[depth_ok]

    if not _fp_good_overlay_saved and n_total >= 30 and int(np.sum(inside)) > 0:
        _save_fp_debug_overlay(
            det=det,
            img_w=img_w,
            img_h=img_h,
            bx1=bx1_s,
            by1=by1_s,
            bx2=bx2_s,
            by2=by2_s,
            u=u,
            v=v,
            inside=inside,
            source_tag="legacy_reproject_good",
            title=f"good_ref_legacy | {det.category} | shot_fp={n_total} in_bbox={int(np.sum(inside))}",
        )
        _fp_good_overlay_saved = True

    return _ret(inside_world, {
        "fp_visible": n_visible,
        "fp_behind": n_behind,
        "bbox_space": "legacy_reproject",
        "in_mask": 0,
        "fp_uv_bounds": [
            round(float(np.percentile(u, 5)), 1),
            round(float(np.percentile(u, 95)), 1),
            round(float(np.percentile(v, 5)), 1),
            round(float(np.percentile(v, 95)), 1),
        ],
        "pixel_space_contract": px_contract,
    })


@dataclass
class LiftResult:
    """Result of lifting a 2D detection to 3D world space."""
    world_pos: Tuple[float, float, float]  # (x, y, z) in world coordinates
    depth_value: float                      # representative depth used (meters)
    valid_sample_count: int                 # how many depth pixels passed filters
    depth_mad: float                        # median absolute deviation of depths (meters)
    camera_yaw: Optional[float]             # camera forward direction yaw (radians)


def _shrunk_bbox_mask(roi_h: int, roi_w: int, shrink: float = 0.20) -> "np.ndarray":
    """Create a boolean mask that is the bbox shrunk by *shrink* fraction on each side."""
    mask = np.zeros((roi_h, roi_w), dtype=bool)
    margin_y = int(roi_h * shrink)
    margin_x = int(roi_w * shrink)
    # Ensure at least 1 pixel remains
    y1 = min(margin_y, max(0, roi_h // 2 - 1))
    y2 = max(roi_h - margin_y, min(roi_h, roi_h // 2 + 1))
    x1 = min(margin_x, max(0, roi_w // 2 - 1))
    x2 = max(roi_w - margin_x, min(roi_w, roi_w // 2 + 1))
    mask[y1:y2, x1:x2] = True
    return mask


def robust_lift_detection(
    det: Detection,
    depth_map: "np.ndarray",
    shot: ShotMetadata,
) -> Optional[LiftResult]:
    """
    Lift a 2D detection to a 3D world-space position using multi-point
    back-projection with adaptive near-surface depth selection.

    Args:
        det:       A Detection with bbox, category, mask (optional)
        depth_map: H×W depth array for this frame (in meters, already scaled)
        shot:      ShotMetadata with pose (4×4 as 16 floats) + intrinsics

    Returns:
        LiftResult or None if lifting failed (bad depth, no pose, etc.)
    """
    if np is None:
        return None

    # ── validate inputs ──────────────────────────────────────────────
    if shot.pose is None or shot.intrinsics is None:
        return None
    if len(shot.pose) != 16:
        return None

    h, w = depth_map.shape[:2]
    intr = shot.intrinsics.scaled_to(w, h)

    # ── scale bbox from detection resolution to depth-map resolution ─
    x1, y1, x2, y2 = det.bbox
    det_w, det_h = det.input_size
    sx = w / det_w
    sy = h / det_h
    bx1 = int(max(0, x1 * sx))
    by1 = int(max(0, y1 * sy))
    bx2 = int(min(w, x2 * sx))
    by2 = int(min(h, y2 * sy))

    if bx2 <= bx1 or by2 <= by1:
        return None

    roi_h = by2 - by1
    roi_w = bx2 - bx1

    # ── build sample mask ────────────────────────────────────────────
    if det.mask is not None:
        mask_h, mask_w = det.mask.shape[:2]
        if mask_h == h and mask_w == w:
            sample_mask = det.mask[by1:by2, bx1:bx2] > 0
            # Erode 2px to remove boundary noise (depth edges are unreliable)
            try:
                import cv2
                kernel = np.ones((3, 3), dtype=np.uint8)
                eroded = cv2.erode(sample_mask.astype(np.uint8), kernel, iterations=1)
                # Only use eroded if it still has enough pixels
                if np.sum(eroded) >= 10:
                    sample_mask = eroded > 0
            except ImportError:
                pass  # no cv2 — use raw mask
        else:
            # Mask resolution doesn't match depth map — resize
            try:
                import cv2
                resized = cv2.resize(det.mask.astype(np.uint8), (w, h),
                                     interpolation=cv2.INTER_NEAREST)
                sample_mask = resized[by1:by2, bx1:bx2] > 0
            except ImportError:
                sample_mask = _shrunk_bbox_mask(roi_h, roi_w)
    else:
        sample_mask = _shrunk_bbox_mask(roi_h, roi_w)

    # ── extract valid depth samples ──────────────────────────────────
    depth_roi = depth_map[by1:by2, bx1:bx2]
    valid = np.isfinite(depth_roi) & (depth_roi > 0.1) & (depth_roi < 15.0) & sample_mask

    valid_count = int(np.sum(valid))
    if valid_count < 10:
        return None

    valid_depths = depth_roi[valid]

    # ── foreground occluder guardrail ────────────────────────────────
    # If p10 is < 50% of p25, a thin foreground object (arm, person) is
    # likely present — discard the bottom 10 % of depth samples.
    p10 = float(np.percentile(valid_depths, 10))
    p25 = float(np.percentile(valid_depths, 25))
    if p25 > 0.3 and p10 < 0.5 * p25:
        cutoff = float(np.percentile(valid_depths, 10))
        keep = valid_depths >= cutoff
        if np.sum(keep) >= 10:
            valid_depths = valid_depths[keep]
            # Also need to filter the pixel coordinates correspondingly
            # We'll rebuild ys/xs below from filtered set

    # ── adaptive percentile depth selection ──────────────────────────
    p25 = float(np.percentile(valid_depths, 25))
    p50 = float(np.percentile(valid_depths, 50))

    if p50 > 0.2 and (p50 - p25) / p50 > 0.15:
        # Bimodal: object surface + wall behind → use p25 (nearest surface)
        depth_cutoff = p25
    else:
        # Unimodal: use p35 (slightly conservative)
        depth_cutoff = float(np.percentile(valid_depths, 35))

    depth_mad = float(np.median(np.abs(valid_depths - p50)))

    # ── multi-point back-projection ──────────────────────────────────
    # Get pixel coordinates of all valid samples
    ys_local, xs_local = np.where(
        np.isfinite(depth_roi) & (depth_roi > 0.1) & (depth_roi < 15.0) & sample_mask
    )

    # Filter to only depths that passed the occluder guardrail
    # (recompute from the depth values at these positions)
    pixel_depths = depth_roi[ys_local, xs_local]
    if p25 > 0.3 and p10 < 0.5 * p25:
        keep_mask = pixel_depths >= float(np.percentile(pixel_depths, 10))
        ys_local = ys_local[keep_mask]
        xs_local = xs_local[keep_mask]
        pixel_depths = pixel_depths[keep_mask]

    if len(pixel_depths) < 10:
        return None

    # Select only pixels near the object surface (within adaptive percentile range)
    # Keep depths up to the adaptive cutoff + a small margin
    margin = max(0.3, depth_cutoff * 0.15)
    near_surface = pixel_depths <= (depth_cutoff + margin)
    if np.sum(near_surface) < 5:
        # Fall back to all samples if surface filter is too aggressive
        near_surface = np.ones(len(pixel_depths), dtype=bool)

    ys_sel = ys_local[near_surface]
    xs_sel = xs_local[near_surface]
    depths_sel = pixel_depths[near_surface]

    # Subsample to ~200 pixels for consistency
    rng = np.random.RandomState(42)
    if len(depths_sel) > 200:
        idx = rng.choice(len(depths_sel), size=200, replace=False)
        ys_sel = ys_sel[idx]
        xs_sel = xs_sel[idx]
        depths_sel = depths_sel[idx]

    # Convert local ROI coords to full image coords
    us = (xs_sel + bx1).astype(np.float64)  # pixel x in depth map
    vs = (ys_sel + by1).astype(np.float64)  # pixel y in depth map
    ds = depths_sel.astype(np.float64)

    # Back-project each pixel to camera space
    cam_xs = (us - intr.cx) * ds / intr.fx
    cam_ys = (vs - intr.cy) * ds / intr.fy
    cam_zs = ds

    # Transform all points to world space in one matrix multiply
    mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
    pts_cam = np.stack([cam_xs, cam_ys, cam_zs, np.ones_like(cam_xs)], axis=1)  # (N, 4)
    pts_world = (mat @ pts_cam.T).T[:, :3]  # (N, 3)

    # Anchor = median of world coordinates (per axis)
    world_x = float(np.median(pts_world[:, 0]))
    world_y = float(np.median(pts_world[:, 1]))
    world_z = float(np.median(pts_world[:, 2]))

    # ── camera yaw ───────────────────────────────────────────────────
    cam_fwd_x = -mat[0, 2]
    cam_fwd_z = -mat[2, 2]
    camera_yaw = float(np.arctan2(cam_fwd_x, cam_fwd_z))

    logger.debug(
        "robust_lift: %s frame=%d mask=%s samples=%d/%d depth_p25=%.3f p50=%.3f "
        "selected=%.3f mad=%.3f world=(%.3f,%.3f,%.3f)",
        det.category, det.frame_idx,
        "instance" if (det.mask is not None) else "shrunk_bbox",
        len(depths_sel), valid_count,
        p25, p50, depth_cutoff, depth_mad,
        world_x, world_y, world_z,
    )

    return LiftResult(
        world_pos=(world_x, world_y, world_z),
        depth_value=depth_cutoff,
        valid_sample_count=valid_count,
        depth_mad=depth_mad,
        camera_yaw=camera_yaw,
    )
