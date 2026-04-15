"""
Room layout estimation for the Decroom pipeline.

Session 11: Feature-point-first architecture.
  Priority 1: ARKit feature points (metric) → floor RANSAC + PCA rectangle
  Priority 2: Percentile-based feature-point layout (simpler fallback)
  Priority 3: Plane-anchor prior (floor/walls from ARKit planes)
  Priority 4: Camera-trajectory prior (safety net when FP is weak)
  Priority 5: Depth-based RANSAC walls (legacy, only if FP insufficient)
"""
from __future__ import annotations

import logging
from math import atan2, cos, pi, sin, sqrt
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .config import PipelineConfig
from .metadata import CameraIntrinsics, ShotMetadata

logger = logging.getLogger("decroom-backend.layout")


def estimate_layout(
    depth_maps: List[Optional["np.ndarray"]],
    shots: List[ShotMetadata],
    cfg: PipelineConfig,
    mode: str = "guided",
    feature_cloud: Optional["np.ndarray"] = None,
) -> Dict:
    """
    Estimate room layout.

    Session 11: Feature-point-first hierarchy.
      Priority 1: Feature-point floor RANSAC + PCA bounding rectangle
      Priority 2: Percentile-based feature-point layout
      Priority 3: Plane-anchor prior
      Priority 4: Camera trajectory prior
      Priority 5: Depth-based RANSAC walls (legacy fallback)

    Returns:
        dict with keys: width_m, length_m, height_m, walls, floor_height,
                        room_center, layout_source, axis_confidence
    """
    if np is None:
        logger.warning("numpy not available, returning default layout")
        return _default_layout()

    camera_extent_x, camera_extent_z = _camera_path_extent_xz(shots)
    base_diag = {
        "floor_ransac_inliers": 0,
        "floor_ransac_rmse": None,
        "wall_fit_residual": None,
        "num_near_floor_points": 0,
        "camera_path_extent_xz": {
            "width": round(camera_extent_x, 4),
            "length": round(camera_extent_z, 4),
        },
        "layout_fallback_reason": None,
    }
    plane_layout = _plane_anchor_layout(shots)
    if plane_layout is not None:
        p_w, p_l, p_h, p_floor, p_center, p_yaw, p_conf = plane_layout
        base_diag["plane_prior_available"] = True
        base_diag["plane_prior"] = {
            "width_m": round(float(p_w), 4),
            "length_m": round(float(p_l), 4),
            "height_m": round(float(p_h), 4),
            "floor_height": round(float(p_floor), 4),
            "room_center": [round(float(p_center[0]), 4), round(float(p_center[1]), 4)],
            "room_yaw_rad": round(float(p_yaw), 4),
            "axis_confidence": round(float(p_conf), 4),
        }
    else:
        base_diag["plane_prior_available"] = False
        base_diag["plane_prior"] = None
    fallback_reason = None

    # ── Priority 1: Feature-point floor + PCA rectangle ──────────────
    if feature_cloud is not None and len(feature_cloud) >= 50:
        fp_result, fp_diag = _fp_floor_and_rectangle(feature_cloud, shots, cfg)
        if fp_result is not None:
            width, length, height, floor_h, room_center, room_yaw, axis_conf = fp_result
            source_label = "feature_points_pca"
            axis_conf_fp = float(axis_conf)
            axis_conf_plane = None
            if plane_layout is not None:
                p_w, p_l, _, _, p_center, p_yaw, p_conf = plane_layout
                axis_conf_plane = float(p_conf)
                min_ratio = float(getattr(cfg, "layout_plane_regularization_min_ratio", 0.88))
                blend = float(getattr(cfg, "layout_plane_regularization_blend", 0.35))
                blend = max(0.0, min(0.7, blend))
                width_under = width < (p_w * min_ratio)
                length_under = length < (p_l * min_ratio)
                fp_diag["plane_regularization_checked"] = True
                fp_diag["plane_regularization_trigger"] = bool(width_under or length_under)
                fp_diag["axis_confidence_fp"] = round(axis_conf_fp, 4)
                fp_diag["axis_confidence_plane"] = round(axis_conf_plane, 4)
                # Use plane yaw when plane confidence exceeds PCA confidence
                if axis_conf_plane > axis_conf_fp and axis_conf_plane > 0.5:
                    room_yaw = float(p_yaw)
                    fp_diag["yaw_source"] = "plane_anchor"
                    fp_diag["plane_yaw_rad"] = round(float(p_yaw), 4)
                elif axis_conf_plane > 0.4:
                    # Blend yaw toward plane estimate
                    yaw_blend = 0.5
                    s = (1.0 - yaw_blend) * sin(2.0 * room_yaw) + yaw_blend * sin(2.0 * float(p_yaw))
                    c = (1.0 - yaw_blend) * cos(2.0 * room_yaw) + yaw_blend * cos(2.0 * float(p_yaw))
                    room_yaw = 0.5 * atan2(s, c)
                    fp_diag["yaw_source"] = "pca+plane_blended"
                    fp_diag["plane_yaw_rad"] = round(float(p_yaw), 4)
                if width_under or length_under:
                    reg_w = max(width, (1.0 - blend) * width + blend * float(p_w))
                    reg_l = max(length, (1.0 - blend) * length + blend * float(p_l))
                    c_alpha = max(0.0, min(0.4, float(getattr(cfg, "layout_plane_center_blend", 0.2))))
                    cx = (1.0 - c_alpha) * float(room_center[0]) + c_alpha * float(p_center[0])
                    cz = (1.0 - c_alpha) * float(room_center[1]) + c_alpha * float(p_center[1])
                    width = reg_w
                    length = reg_l
                    room_center = (cx, cz)
                    source_label = "feature_points_pca+plane_regularized"
                    fp_diag["plane_regularization_blend"] = round(blend, 4)
                    fp_diag["plane_regularization_min_ratio"] = round(min_ratio, 4)
                    fp_diag["plane_regularized_width_m"] = round(float(width), 4)
                    fp_diag["plane_regularized_length_m"] = round(float(length), 4)
                    fp_diag["plane_regularized_center"] = [round(float(cx), 4), round(float(cz), 4)]
            if axis_conf_plane is not None:
                axis_conf = max(float(axis_conf_fp), float(axis_conf_plane))
            width = max(2.0, min(12.0, width))
            length = max(2.0, min(12.0, length))
            height = max(2.2, min(3.5, height))
            walls = _generate_walls_centered(width, length, room_center, yaw=room_yaw)
            layout = {
                "width_m": round(width, 2),
                "length_m": round(length, 2),
                "height_m": round(height, 2),
                "floor_height": round(floor_h, 4),
                "room_center": [round(room_center[0], 4), round(room_center[1], 4)],
                "room_yaw_rad": round(room_yaw, 4),
                "axis_confidence": round(axis_conf, 3),
                "axis_confidence_fp": round(axis_conf_fp, 3),
                "axis_confidence_plane": round(axis_conf_plane, 3) if axis_conf_plane is not None else None,
                "walls": walls,
                "layout_source": source_label,
            }
            layout.update(base_diag)
            layout.update(fp_diag)
            logger.info("Layout (FP-PCA): %.1fm x %.1fm x %.1fm, floor=%.3f",
                         width, length, height, floor_h)
            _debug_layout(layout)
            return layout
        fallback_reason = fp_diag.get("layout_fallback_reason", "feature_points_pca_failed")
    elif feature_cloud is None:
        fallback_reason = "feature_cloud_missing"
    else:
        fallback_reason = f"feature_cloud_too_sparse(n={len(feature_cloud)})"

    # ── Priority 2: Percentile-based FP layout ───────────────────────
    fp_layout = _feature_point_layout(shots, cfg)
    if fp_layout is not None:
        fp_w, fp_l, fp_floor, fp_center = fp_layout
        fp_w = max(2.0, min(12.0, fp_w))
        fp_l = max(2.0, min(12.0, fp_l))
        walls = _generate_walls_centered(fp_w, fp_l, fp_center)
        layout = {
            "width_m": round(fp_w, 2),
            "length_m": round(fp_l, 2),
            "height_m": 2.5,
            "floor_height": round(fp_floor, 4),
            "room_center": [round(fp_center[0], 4), round(fp_center[1], 4)],
            "axis_confidence": 0.55,
            "walls": walls,
            "layout_source": "feature_points_percentile",
            "layout_fallback_reason": fallback_reason,
        }
        layout.update(base_diag)
        layout["layout_fallback_reason"] = fallback_reason
        logger.info("Layout (FP-percentile): %.1fm x %.1fm, floor=%.3f", fp_w, fp_l, fp_floor)
        _debug_layout(layout)
        return layout
    if fallback_reason is None:
        fallback_reason = "feature_points_percentile_unavailable"
    else:
        fallback_reason = f"{fallback_reason};feature_points_percentile_unavailable"

    # ── Priority 3: Plane-anchor prior (ARKit floor/walls) ────────────
    if plane_layout is not None:
        p_w, p_l, p_h, p_floor, p_center, p_yaw, p_conf = plane_layout
        walls = _generate_walls_centered(p_w, p_l, p_center, yaw=p_yaw)
        layout = {
            "width_m": round(p_w, 2),
            "length_m": round(p_l, 2),
            "height_m": round(p_h, 2),
            "floor_height": round(p_floor, 4),
            "room_center": [round(p_center[0], 4), round(p_center[1], 4)],
            "room_yaw_rad": round(p_yaw, 4),
            "axis_confidence": round(p_conf, 3),
            "axis_confidence_fp": None,
            "axis_confidence_plane": round(p_conf, 3),
            "walls": walls,
            "layout_source": "plane_anchor_prior",
            "layout_fallback_reason": fallback_reason,
        }
        layout.update(base_diag)
        layout["layout_fallback_reason"] = fallback_reason
        logger.info(
            "Layout (plane prior): %.1fm x %.1fm x %.1fm, floor=%.3f, center=(%.2f, %.2f), yaw=%.1f°",
            p_w, p_l, p_h, p_floor, p_center[0], p_center[1], p_yaw * 180.0 / pi
        )
        _debug_layout(layout)
        return layout
    if fallback_reason is None:
        fallback_reason = "plane_anchor_prior_unavailable"
    else:
        fallback_reason = f"{fallback_reason};plane_anchor_prior_unavailable"

    # ── Priority 4: Camera trajectory prior (non-LiDAR safety net) ───
    traj_layout = _trajectory_layout(shots)
    if traj_layout is not None:
        t_w, t_l, t_floor, t_center, t_yaw, t_conf = traj_layout
        walls = _generate_walls_centered(t_w, t_l, t_center, yaw=t_yaw)
        layout = {
            "width_m": round(t_w, 2),
            "length_m": round(t_l, 2),
            "height_m": 2.6,
            "floor_height": round(t_floor, 4),
            "room_center": [round(t_center[0], 4), round(t_center[1], 4)],
            "room_yaw_rad": round(t_yaw, 4),
            "axis_confidence": round(t_conf, 3),
            "walls": walls,
            "layout_source": "camera_trajectory_prior",
            "layout_fallback_reason": fallback_reason,
        }
        layout.update(base_diag)
        layout["layout_fallback_reason"] = fallback_reason
        logger.info(
            "Layout (trajectory prior): %.1fm x %.1fm, center=(%.2f, %.2f), yaw=%.1f°",
            t_w, t_l, t_center[0], t_center[1], t_yaw * 180.0 / pi
        )
        _debug_layout(layout)
        return layout
    if fallback_reason is None:
        fallback_reason = "camera_trajectory_prior_unavailable"
    else:
        fallback_reason = f"{fallback_reason};camera_trajectory_prior_unavailable"

    # ── Priority 5: Depth-based RANSAC (legacy fallback) ─────────────
    logger.info("Feature points insufficient, falling back to depth-based layout")
    points = _build_point_cloud(depth_maps, shots)

    if points is None or len(points) < 50:
        logger.warning("Insufficient points for layout (%s), using defaults",
                        len(points) if points is not None else 0)
        layout = _default_layout()
        layout.update(base_diag)
        layout["layout_fallback_reason"] = (
            f"{fallback_reason};depth_ransac_points_insufficient(n={len(points) if points is not None else 0})"
            if fallback_reason
            else f"depth_ransac_points_insufficient(n={len(points) if points is not None else 0})"
        )
        _debug_layout(layout)
        return layout

    gravity = _get_gravity_vector(shots)
    if gravity is not None:
        points = _align_to_gravity(points, gravity)

    room_bounds = _ransac_walls(points)
    if room_bounds is None:
        room_bounds = _pca_room_bounds(points)
    if room_bounds is None:
        layout = _default_layout()
        layout.update(base_diag)
        layout["layout_fallback_reason"] = (
            f"{fallback_reason};depth_bounds_fit_failed"
            if fallback_reason
            else "depth_bounds_fit_failed"
        )
        _debug_layout(layout)
        return layout

    x_min, x_max, z_min, z_max, y_min, y_max = room_bounds
    width = max(2.0, min(12.0, x_max - x_min))
    length = max(2.0, min(12.0, z_max - z_min))
    height = max(2.2, min(3.0, y_max - y_min))
    walls = _generate_walls(width, length)
    layout = {
        "width_m": round(width, 2),
        "length_m": round(length, 2),
        "height_m": round(height, 2),
        "floor_height": round(y_min, 4),
        "room_center": [0.0, 0.0],
        "axis_confidence": 0.5,
        "walls": walls,
        "layout_source": "depth_ransac",
        "layout_fallback_reason": fallback_reason,
    }
    layout.update(base_diag)
    layout["layout_fallback_reason"] = fallback_reason
    logger.info("Layout (depth fallback): %.1fm x %.1fm x %.1fm", width, length, height)
    _debug_layout(layout)
    return layout


def _fp_floor_and_rectangle(
    cloud: "np.ndarray",
    shots: List[ShotMetadata],
    cfg: PipelineConfig,
) -> Tuple[
    Optional[Tuple[float, float, float, float, Tuple[float, float], float, float]],
    Dict,
]:
    """
    Session 11: Primary layout from feature-point cloud.

    Step 1: Floor plane via RANSAC with gravity constraint.
    Step 2: Room boundary via PCA bounding rectangle on XZ hull.

    Returns:
        ((width, length, height, floor_height, (center_x, center_z), yaw_rad, axis_confidence), diag)
        or (None, diag) on failure.
    """
    diag = {
        "floor_ransac_inliers": 0,
        "floor_ransac_rmse": None,
        "wall_fit_residual": None,
        "num_near_floor_points": 0,
        "layout_fallback_reason": None,
    }
    if len(cloud) < 50:
        diag["layout_fallback_reason"] = f"not_enough_feature_cloud_points(n={len(cloud)})"
        return None, diag

    # ── Step 1: Floor plane detection ──
    gravity = _get_gravity_vector(shots)
    if gravity is None:
        gravity = np.array([0.0, -1.0, 0.0])

    # RANSAC: find plane whose normal is close to gravity (i.e., horizontal plane)
    rng = np.random.RandomState(42)
    best_inliers = 0
    best_floor_y = None
    best_plane_p0 = None
    best_plane_normal = None
    best_inlier_mask = None
    n = len(cloud)
    dist_thresh = 0.08  # 8cm tolerance for floor plane

    for _ in range(100):
        # Sample 3 random points
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = cloud[idx]

        # Fit plane
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            continue
        normal = normal / norm_len

        # Check gravity alignment (plane normal ≈ gravity direction)
        if abs(float(np.dot(normal, gravity))) < 0.95:
            continue

        # Count inliers
        dists = np.abs(np.dot(cloud - p0, normal))
        n_inliers = int(np.sum(dists < dist_thresh))

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_plane_p0 = p0
            best_plane_normal = normal
            # Floor Y = median Y of inlier points
            inlier_mask = dists < dist_thresh
            best_inlier_mask = inlier_mask
            best_floor_y = float(np.median(cloud[inlier_mask, 1]))

    if best_floor_y is None:
        # Fallback: floor = min(Y) + small margin
        y_vals = cloud[:, 1]
        best_floor_y = float(np.percentile(y_vals, 5))
        diag["layout_fallback_reason"] = "floor_ransac_failed_using_y_p5"
        logger.info("Floor RANSAC failed, using Y p5: %.3f", best_floor_y)
    elif best_plane_p0 is not None and best_plane_normal is not None and best_inlier_mask is not None:
        floor_dists = np.abs(np.dot(cloud - best_plane_p0, best_plane_normal))
        inlier_dists = floor_dists[best_inlier_mask]
        if len(inlier_dists) > 0:
            rmse = float(np.sqrt(np.mean(inlier_dists * inlier_dists)))
            diag["floor_ransac_rmse"] = round(rmse, 5)
    diag["floor_ransac_inliers"] = int(best_inliers)

    floor_height = best_floor_y
    logger.info("Floor height: %.3f (RANSAC inliers=%d/%d)", floor_height, best_inliers, n)

    # ── Step 2: Room boundary from near-floor points ──
    # Keep points within 0.5m above floor (furniture tops are higher, ignore those)
    near_floor = (cloud[:, 1] >= floor_height - 0.2) & (cloud[:, 1] <= floor_height + 0.5)
    near_floor_count = int(np.sum(near_floor))
    diag["num_near_floor_points"] = near_floor_count
    floor_pts = cloud[near_floor]

    if len(floor_pts) < 20:
        # Use all points if not enough near floor
        floor_pts = cloud
        if diag["layout_fallback_reason"] is None:
            diag["layout_fallback_reason"] = f"near_floor_sparse(n={near_floor_count})"

    # Project to XZ plane
    xz = floor_pts[:, [0, 2]]  # (M, 2)

    # PCA to find principal axes of the room
    center_xz = np.mean(xz, axis=0)
    centered = xz - center_xz
    cov = np.cov(centered.T)  # 2×2
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx_sort = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    # Project onto principal axes
    projected = centered @ eigenvectors  # (M, 2)

    # Room extent: 5-95 percentile + margin for walls beyond visible FPs
    p5_0, p95_0 = float(np.percentile(projected[:, 0], 5)), float(np.percentile(projected[:, 0], 95))
    p5_1, p95_1 = float(np.percentile(projected[:, 1], 5)), float(np.percentile(projected[:, 1], 95))

    extent_0 = p95_0 - p5_0
    extent_1 = p95_1 - p5_1
    if extent_0 < 0.15 or extent_1 < 0.15:
        diag["layout_fallback_reason"] = f"rect_extent_too_small(e0={extent_0:.3f},e1={extent_1:.3f})"
        return None, diag

    # Constrained room-fit: keep FP-driven envelope as primary signal while
    # enforcing camera-path containment and capping expansion.
    fp_margin = max(0.05, float(getattr(cfg, "layout_fp_margin_m", 0.2)))
    cam_wall_pad = max(0.2, float(getattr(cfg, "layout_camera_wall_pad_m", 0.55)))
    cam_min_pad = max(0.05, float(getattr(cfg, "layout_camera_min_pad_m", 0.2)))
    max_growth = max(1.05, float(getattr(cfg, "layout_extent_max_growth_ratio", 1.55)))

    cam_proj = []
    for shot in shots:
        if shot.pose is None or len(shot.pose) != 16:
            continue
        try:
            mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
        except Exception:
            continue
        cam_xz = np.array([float(mat[0, 3]), float(mat[2, 3])], dtype=np.float64)
        cam_proj.append((cam_xz - center_xz) @ eigenvectors)

    cam_extent_0 = 0.0
    cam_extent_1 = 0.0
    cam_center_0 = 0.0
    cam_center_1 = 0.0
    if cam_proj:
        cam_arr = np.array(cam_proj, dtype=np.float64)
        cam_p5_0 = float(np.percentile(cam_arr[:, 0], 5))
        cam_p95_0 = float(np.percentile(cam_arr[:, 0], 95))
        cam_p5_1 = float(np.percentile(cam_arr[:, 1], 5))
        cam_p95_1 = float(np.percentile(cam_arr[:, 1], 95))
        cam_extent_0 = max(0.0, cam_p95_0 - cam_p5_0)
        cam_extent_1 = max(0.0, cam_p95_1 - cam_p5_1)
        cam_center_0 = float(np.median(cam_arr[:, 0]))
        cam_center_1 = float(np.median(cam_arr[:, 1]))

    width, width_diag = _constrained_axis_extent(
        observed_extent=extent_0,
        camera_extent=cam_extent_0,
        fp_margin_m=fp_margin,
        camera_wall_pad_m=cam_wall_pad,
        camera_min_pad_m=cam_min_pad,
        max_growth_ratio=max_growth,
    )
    length, length_diag = _constrained_axis_extent(
        observed_extent=extent_1,
        camera_extent=cam_extent_1,
        fp_margin_m=fp_margin,
        camera_wall_pad_m=cam_wall_pad,
        camera_min_pad_m=cam_min_pad,
        max_growth_ratio=max_growth,
    )

    # Height: Y range
    y_vals = cloud[:, 1]
    y_lo = float(np.percentile(y_vals, 5))
    y_hi = float(np.percentile(y_vals, 95))
    height = y_hi - y_lo

    obs_center_0 = 0.5 * (p5_0 + p95_0)
    obs_center_1 = 0.5 * (p5_1 + p95_1)
    blend_alpha = float(getattr(cfg, "layout_center_blend_alpha", 0.25))
    blend_alpha = max(0.0, min(0.5, blend_alpha))
    if len(cam_proj) >= 4 and (cam_extent_0 > 0.25 or cam_extent_1 > 0.25):
        center_0 = (1.0 - blend_alpha) * obs_center_0 + blend_alpha * cam_center_0
        center_1 = (1.0 - blend_alpha) * obs_center_1 + blend_alpha * cam_center_1
    else:
        center_0 = obs_center_0
        center_1 = obs_center_1

    room_center_vec = center_xz + (eigenvectors @ np.array([center_0, center_1], dtype=np.float64))
    room_center = (float(room_center_vec[0]), float(room_center_vec[1]))
    principal = eigenvectors[:, 0]
    room_yaw = float(atan2(float(principal[0]), float(principal[1])))

    # Rectangle fit residual on PCA axes. Values near 0 mean points are
    # tightly explained by the fitted room rectangle.
    over0 = np.maximum(0.0, p5_0 - projected[:, 0]) + np.maximum(0.0, projected[:, 0] - p95_0)
    over1 = np.maximum(0.0, p5_1 - projected[:, 1]) + np.maximum(0.0, projected[:, 1] - p95_1)
    rect_err = np.sqrt(over0 * over0 + over1 * over1)
    diag["wall_fit_residual"] = round(float(np.sqrt(np.mean(rect_err * rect_err))), 5)
    diag["room_fit_mode"] = "constrained_feature_cloud"
    diag["room_fit_camera_pose_count"] = int(len(cam_proj))
    diag["room_fit_observed_extent"] = {
        "axis0": round(float(extent_0), 4),
        "axis1": round(float(extent_1), 4),
    }
    diag["room_fit_camera_extent"] = {
        "axis0": round(float(cam_extent_0), 4),
        "axis1": round(float(cam_extent_1), 4),
    }
    diag["room_fit_final_extent"] = {
        "axis0": round(float(width), 4),
        "axis1": round(float(length), 4),
    }
    diag["room_fit_growth"] = {
        "axis0": round(float(width_diag["growth_ratio"]), 4),
        "axis1": round(float(length_diag["growth_ratio"]), 4),
    }
    diag["room_fit_growth_capped"] = bool(width_diag["capped"] or length_diag["capped"])

    floor_conf = float(best_inliers) / float(max(1, n))
    elongation = float(eigenvalues[0] / (eigenvalues[1] + 1e-8))
    axis_structure = min(1.0, max(0.0, (elongation - 1.0) / 3.0))
    point_conf = min(1.0, float(len(floor_pts)) / 250.0)
    axis_confidence = max(
        0.4,
        min(
            0.95,
            0.35 + 0.35 * floor_conf + 0.2 * axis_structure + 0.1 * point_conf,
        ),
    )

    logger.info(
        "FP-PCA layout: %.1fm × %.1fm, height=%.1fm, center=(%.2f, %.2f), "
        "floor=%.3f, n_floor_pts=%d, cam_extent=(%.2f, %.2f)",
        width, length, height, room_center[0], room_center[1],
        floor_height, len(floor_pts), cam_extent_0, cam_extent_1
    )

    return (width, length, height, floor_height, room_center, room_yaw, axis_confidence), diag


def _feature_point_layout(shots: List[ShotMetadata], cfg: PipelineConfig) -> Optional[tuple]:
    """
    Percentile-based room size from ARKit feature points (mid-level fallback).

    Returns (width, length, floor_height, (center_x, center_z)) or None.
    """
    all_pts = []
    for shot in shots:
        if shot.feature_points_w is None:
            continue
        for fp in shot.feature_points_w:
            if len(fp) == 3:
                all_pts.append(fp)

    if len(all_pts) < 20:
        logger.info("Feature-point layout: only %d points (need 20+), skipping", len(all_pts))
        return None

    arr = np.array(all_pts, dtype=np.float64)
    arr_x = arr[:, 0]
    arr_y = arr[:, 1]
    arr_z = arr[:, 2]

    # Use 5-95 percentile to reject outliers
    x_lo, x_hi = float(np.percentile(arr_x, 5)), float(np.percentile(arr_x, 95))
    z_lo, z_hi = float(np.percentile(arr_z, 5)), float(np.percentile(arr_z, 95))

    obs_width = max(0.0, x_hi - x_lo)
    obs_length = max(0.0, z_hi - z_lo)
    fp_margin = max(0.05, float(getattr(cfg, "layout_fp_margin_m", 0.2)))
    cam_wall_pad = max(0.2, float(getattr(cfg, "layout_camera_wall_pad_m", 0.55)))
    cam_min_pad = max(0.05, float(getattr(cfg, "layout_camera_min_pad_m", 0.2)))
    max_growth = max(1.05, float(getattr(cfg, "layout_extent_max_growth_ratio", 1.55)))

    cam_extent_x, cam_extent_z = _camera_path_extent_xz(shots)
    width, _ = _constrained_axis_extent(
        observed_extent=obs_width,
        camera_extent=cam_extent_x,
        fp_margin_m=fp_margin,
        camera_wall_pad_m=cam_wall_pad,
        camera_min_pad_m=cam_min_pad,
        max_growth_ratio=max_growth,
    )
    length, _ = _constrained_axis_extent(
        observed_extent=obs_length,
        camera_extent=cam_extent_z,
        fp_margin_m=fp_margin,
        camera_wall_pad_m=cam_wall_pad,
        camera_min_pad_m=cam_min_pad,
        max_growth_ratio=max_growth,
    )

    # Floor height = Y p5 (lowest points = floor)
    floor_height = float(np.percentile(arr_y, 5))
    center_x = float(np.median(arr_x))
    center_z = float(np.median(arr_z))
    blend_alpha = float(getattr(cfg, "layout_center_blend_alpha", 0.25))
    blend_alpha = max(0.0, min(0.5, blend_alpha))
    cam_x = []
    cam_z = []
    for shot in shots:
        if shot.pose is None or len(shot.pose) != 16:
            continue
        try:
            mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
        except Exception:
            continue
        cam_x.append(float(mat[0, 3]))
        cam_z.append(float(mat[2, 3]))
    if len(cam_x) >= 4 and (cam_extent_x > 0.25 or cam_extent_z > 0.25):
        center_x = (1.0 - blend_alpha) * center_x + blend_alpha * float(np.median(np.array(cam_x, dtype=np.float64)))
        center_z = (1.0 - blend_alpha) * center_z + blend_alpha * float(np.median(np.array(cam_z, dtype=np.float64)))

    # Clamp
    width = max(2.0, min(12.0, width))
    length = max(2.0, min(12.0, length))

    logger.info("Feature-point layout (percentile): %.1fm × %.1fm, floor=%.3f, "
                "center=(%.2f, %.2f) from %d pts",
                width, length, floor_height, center_x, center_z, len(all_pts))
    return (width, length, floor_height, (center_x, center_z))


def _constrained_axis_extent(
    observed_extent: float,
    camera_extent: float,
    fp_margin_m: float,
    camera_wall_pad_m: float,
    camera_min_pad_m: float,
    max_growth_ratio: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Constrain axis extent with:
    - observed feature-point span (+ small margin),
    - camera-path containment (+ minimum clearance),
    - capped growth to avoid runaway room inflation.
    """
    obs = max(0.0, float(observed_extent))
    cam = max(0.0, float(camera_extent))
    obs_target = obs + 2.0 * max(0.0, fp_margin_m)
    cam_preferred = cam + 2.0 * max(0.0, camera_wall_pad_m)
    cam_required = cam + 2.0 * max(0.0, camera_min_pad_m)
    max_allowed = max(obs_target * max(1.0, max_growth_ratio), cam_required)
    uncapped = max(obs_target, cam_preferred)
    extent = min(uncapped, max_allowed)
    extent = max(extent, obs_target, cam_required)
    growth_ratio = extent / max(obs, 1e-6)
    return extent, {
        "growth_ratio": float(growth_ratio),
        "capped": 1.0 if uncapped > max_allowed else 0.0,
    }


def _default_layout() -> Dict:
    """Return default room layout."""
    w, l = 4.0, 4.0
    return {
        "width_m": w,
        "length_m": l,
        "height_m": 2.5,
        "axis_confidence": 0.25,
        "walls": _generate_walls(w, l, yaw=None),
    }


def _camera_path_extent_xz(shots: List[ShotMetadata]) -> Tuple[float, float]:
    """Return camera trajectory extent on X/Z axes (p95-p5), in meters."""
    xs: List[float] = []
    zs: List[float] = []
    for shot in shots:
        if shot.pose is None or len(shot.pose) != 16:
            continue
        try:
            mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
        except Exception:
            continue
        xs.append(float(mat[0, 3]))
        zs.append(float(mat[2, 3]))

    if not xs or not zs:
        return (0.0, 0.0)

    x_extent = float(np.percentile(np.array(xs, dtype=np.float64), 95) -
                     np.percentile(np.array(xs, dtype=np.float64), 5))
    z_extent = float(np.percentile(np.array(zs, dtype=np.float64), 95) -
                     np.percentile(np.array(zs, dtype=np.float64), 5))
    return (max(0.0, x_extent), max(0.0, z_extent))


def _trajectory_layout(
    shots: List[ShotMetadata],
) -> Optional[Tuple[float, float, float, Tuple[float, float], float, float]]:
    """
    Estimate a conservative room prior from camera trajectory only.

    Used as a fallback when feature-point layout is weak/unavailable.
    Returns (width, length, floor_height, (center_x, center_z), yaw, axis_confidence).
    """
    poses = []
    yaws = []
    for shot in shots:
        if shot.pose is None or len(shot.pose) != 16:
            continue
        mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
        poses.append(mat[:3, 3])
        # Camera forward in world (ARKit: -Z column)
        fwd_x = -float(mat[0, 2])
        fwd_z = -float(mat[2, 2])
        yaws.append(float(atan2(fwd_x, fwd_z)))

    if len(poses) < 2:
        return None

    pos = np.array(poses, dtype=np.float64)
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    x_lo, x_hi = float(np.percentile(x, 5)), float(np.percentile(x, 95))
    z_lo, z_hi = float(np.percentile(z, 5)), float(np.percentile(z, 95))
    spread_x = max(0.0, x_hi - x_lo)
    spread_z = max(0.0, z_hi - z_lo)

    # Expand camera path to a plausible room envelope.
    width = spread_x + 2.2
    length = spread_z + 2.2
    width = max(3.0, min(10.0, width))
    length = max(3.0, min(10.0, length))

    center = (float(np.median(x)), float(np.median(z)))
    floor_height = float(np.percentile(y, 5) - 1.4)

    # Yaw prior from camera trajectory PCA (if moved), else camera heading median.
    yaw = 0.0
    if len(pos) >= 3 and (spread_x > 0.2 or spread_z > 0.2):
        xz = pos[:, [0, 2]]
        xz_center = np.mean(xz, axis=0)
        centered = xz - xz_center
        cov = np.cov(centered.T)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(-vals)
        vec = vecs[:, idx[0]]
        yaw = float(atan2(float(vec[0]), float(vec[1])))
    elif yaws:
        sin_med = float(np.median(np.sin(yaws)))
        cos_med = float(np.median(np.cos(yaws)))
        yaw = float(atan2(sin_med, cos_med))

    yaw_bins = set()
    for yv in yaws:
        a = yv % (2.0 * pi)
        yaw_bins.add(int(a / (pi / 6.0)))  # 12 bins
    yaw_cov = len(yaw_bins) / 12.0
    motion = min(1.0, sqrt(spread_x * spread_x + spread_z * spread_z) / 2.0)
    shot_conf = min(1.0, len(poses) / 12.0)
    axis_conf = 0.25 + 0.3 * yaw_cov + 0.25 * motion + 0.2 * shot_conf
    axis_conf = max(0.25, min(0.75, axis_conf))

    return (width, length, floor_height, center, yaw, axis_conf)


def _plane_anchor_layout(
    shots: List[ShotMetadata],
) -> Optional[Tuple[float, float, float, float, Tuple[float, float], float, float]]:
    """
    Estimate room layout from ARKit plane anchors uploaded by the client.

    Uses horizontal anchors for floor/footprint and vertical anchors for axis prior.
    Returns (width, length, height, floor_height, (center_x, center_z), yaw, axis_confidence).
    """
    anchors_by_id: Dict[str, object] = {}
    for shot_idx, shot in enumerate(shots):
        if not shot.plane_anchors:
            continue
        for anchor_idx, anchor in enumerate(shot.plane_anchors):
            key = anchor.anchor_id or f"{shot_idx}_{anchor_idx}"
            anchors_by_id[key] = anchor

    if len(anchors_by_id) < 2:
        return None

    floor_polys: List["np.ndarray"] = []
    floor_y: List[float] = []
    wall_normals: List[Tuple["np.ndarray", float]] = []
    wall_centers: List["np.ndarray"] = []
    wall_height_cands: List[float] = []

    for anchor in anchors_by_id.values():
        transform = getattr(anchor, "transform", None)
        if transform is None or len(transform) != 16:
            continue
        try:
            mat = np.array(transform, dtype=np.float64).reshape(4, 4)
        except Exception:
            continue

        center = getattr(anchor, "center", None)
        if center is None or len(center) != 3:
            center = [0.0, 0.0, 0.0]
        extent = getattr(anchor, "extent", None)
        if extent is None or len(extent) != 3:
            extent = [0.0, 0.0, 0.0]

        c_local = np.array([float(center[0]), float(center[1]), float(center[2]), 1.0], dtype=np.float64)
        c_world = mat @ c_local
        wx, wy, wz = float(c_world[0]), float(c_world[1]), float(c_world[2])

        ex = abs(float(extent[0]))
        ey = abs(float(extent[1]))
        ez = abs(float(extent[2]))

        normal = mat[:3, 1]
        n_norm = float(np.linalg.norm(normal))
        if n_norm < 1e-6:
            continue
        normal = normal / n_norm
        up_dot = abs(float(normal[1]))

        alignment = str(getattr(anchor, "alignment", "unknown")).lower()
        is_horizontal = alignment == "horizontal" or up_dot > 0.82
        is_vertical = alignment == "vertical" or up_dot < 0.35

        if is_horizontal:
            floor_y.append(wy)
            if ex > 0.05 and ez > 0.05:
                hx = ex * 0.5
                hz = ez * 0.5
                local_corners = np.array([
                    [center[0] - hx, center[1], center[2] - hz, 1.0],
                    [center[0] + hx, center[1], center[2] - hz, 1.0],
                    [center[0] + hx, center[1], center[2] + hz, 1.0],
                    [center[0] - hx, center[1], center[2] + hz, 1.0],
                ], dtype=np.float64)
                world_corners = (mat @ local_corners.T).T
                floor_polys.append(world_corners[:, [0, 2]])

        if is_vertical:
            wall_centers.append(np.array([wx, wz], dtype=np.float64))
            nxz = np.array([float(normal[0]), float(normal[2])], dtype=np.float64)
            nxz_norm = float(np.linalg.norm(nxz))
            if nxz_norm > 1e-6:
                weight = max(ex, ez, 0.2)
                wall_normals.append((nxz / nxz_norm, weight))

            h_cand = max(ey, ex, ez)
            if h_cand > 1.5:
                wall_height_cands.append(h_cand)

    if not floor_polys and len(wall_centers) < 2:
        return None

    room_yaw = 0.0
    axis_consistency = 0.0
    if wall_normals:
        s2 = 0.0
        c2 = 0.0
        w_sum = 0.0
        for nxz, w in wall_normals:
            yaw = float(atan2(float(nxz[0]), float(nxz[1])))
            s2 += w * sin(2.0 * yaw)
            c2 += w * cos(2.0 * yaw)
            w_sum += w
        if w_sum > 1e-6:
            room_yaw = 0.5 * atan2(s2, c2)
            axis_consistency = min(1.0, max(0.0, sqrt(s2 * s2 + c2 * c2) / w_sum))
    elif floor_polys:
        floor_pts = np.vstack(floor_polys)
        if len(floor_pts) >= 4:
            centered = floor_pts - np.mean(floor_pts, axis=0)
            cov = np.cov(centered.T)
            vals, vecs = np.linalg.eigh(cov)
            idx = np.argsort(-vals)
            principal = vecs[:, idx[0]]
            room_yaw = float(atan2(float(principal[0]), float(principal[1])))
            elong = float(vals[idx[0]] / (vals[idx[1]] + 1e-8))
            axis_consistency = min(1.0, max(0.0, (elong - 1.0) / 3.0))

    axis0 = np.array([sin(room_yaw), cos(room_yaw)], dtype=np.float64)
    axis1 = np.array([-axis0[1], axis0[0]], dtype=np.float64)

    support_pts = []
    if floor_polys:
        support_pts.extend([p for poly in floor_polys for p in poly])
    support_pts.extend(wall_centers)
    if len(support_pts) < 4:
        return None
    pts = np.array(support_pts, dtype=np.float64)
    proj0 = pts @ axis0
    proj1 = pts @ axis1

    span0 = float(np.percentile(proj0, 95) - np.percentile(proj0, 5))
    span1 = float(np.percentile(proj1, 95) - np.percentile(proj1, 5))
    margin = 0.45 if floor_polys else 1.0
    width = max(2.4, min(12.0, span0 + margin))
    length = max(2.4, min(12.0, span1 + margin))

    center0 = float(np.median(proj0))
    center1 = float(np.median(proj1))
    center_xz = axis0 * center0 + axis1 * center1
    room_center = (float(center_xz[0]), float(center_xz[1]))

    if floor_y:
        floor_height = float(np.percentile(np.array(floor_y, dtype=np.float64), 20))
    else:
        pose_y = []
        for shot in shots:
            if shot.pose is not None and len(shot.pose) == 16:
                mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
                pose_y.append(float(mat[1, 3]))
        floor_height = (float(np.percentile(pose_y, 5)) - 1.4) if pose_y else 0.0

    if wall_height_cands:
        height = float(np.median(np.array(wall_height_cands, dtype=np.float64)))
    else:
        pose_y = []
        for shot in shots:
            if shot.pose is not None and len(shot.pose) == 16:
                mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
                pose_y.append(float(mat[1, 3]))
        if pose_y:
            cam_above_floor = float(np.percentile(pose_y, 95) - floor_height)
            height = cam_above_floor + 1.0
        else:
            height = 2.6
    height = max(2.2, min(3.5, height))

    n_floor = len(floor_polys)
    n_walls = len(wall_centers)
    n_total = len(anchors_by_id)
    anchor_cov = min(1.0, n_total / 16.0)
    wall_cov = min(1.0, n_walls / 8.0)
    floor_cov = min(1.0, n_floor / 6.0)
    axis_conf = 0.45 + 0.2 * axis_consistency + 0.2 * wall_cov + 0.1 * floor_cov + 0.05 * anchor_cov
    axis_conf = max(0.45, min(0.9, axis_conf))

    logger.info(
        "Plane-anchor prior: anchors=%d (floor=%d wall=%d), span=(%.2f, %.2f), floor=%.3f",
        n_total, n_floor, n_walls, span0, span1, floor_height
    )
    return (width, length, height, floor_height, room_center, room_yaw, axis_conf)


def _build_point_cloud(
    depth_maps: List[Optional["np.ndarray"]],
    shots: List[ShotMetadata],
) -> Optional["np.ndarray"]:
    """Back-project depth maps to 3D points using camera intrinsics."""
    all_points = []

    for idx, depth in enumerate(depth_maps):
        if depth is None:
            continue

        # Get camera intrinsics
        intr = None
        if idx < len(shots) and shots[idx].intrinsics is not None:
            intr = shots[idx].intrinsics
        if intr is None:
            # Default intrinsics
            h, w = depth.shape[:2]
            from .metadata import CameraIntrinsics
            intr = CameraIntrinsics(
                fx=w * 0.8, fy=w * 0.8,
                cx=w / 2, cy=h / 2,
                width=w, height=h,
            )

        h, w = depth.shape[:2]
        intr_scaled = intr.scaled_to(w, h)

        # Subsample for performance (every 8th pixel)
        step = 8
        ys = np.arange(0, h, step)
        xs = np.arange(0, w, step)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        d = depth[yy, xx].astype(np.float64)

        # Filter invalid depths — cap at 8m to reject distant hallway/outdoor points
        # (Depth Anything V2 systematically overestimates far surfaces)
        valid = np.isfinite(d) & (d > 0.1) & (d < 8.0)
        if np.sum(valid) < 10:
            continue

        xx_valid = xx[valid].astype(np.float64)
        yy_valid = yy[valid].astype(np.float64)
        d_valid = d[valid]

        # Back-project: (u - cx) * depth / fx, etc.
        px = (xx_valid - intr_scaled.cx) * d_valid / intr_scaled.fx
        py = (yy_valid - intr_scaled.cy) * d_valid / intr_scaled.fy
        pz = d_valid

        pts = np.stack([px, py, pz], axis=1)  # (N, 3)

        # Apply camera pose if available
        if idx < len(shots) and shots[idx].pose is not None:
            pose = shots[idx].pose
            if pose is not None and len(pose) == 16:
                mat = np.array(pose, dtype=np.float64).reshape(4, 4)
                # Transform points: P_world = T @ P_camera
                ones = np.ones((pts.shape[0], 1), dtype=np.float64)
                pts_h = np.hstack([pts, ones])  # (N, 4)
                pts_world = (mat @ pts_h.T).T[:, :3]  # (N, 3)
                pts = pts_world

        all_points.append(pts)

    if not all_points:
        return None

    return np.vstack(all_points)


def _get_gravity_vector(shots: List[ShotMetadata]) -> Optional["np.ndarray"]:
    """Extract average gravity direction from shot metadata."""
    gravities = []
    for s in shots:
        if s.gravity_w is not None and len(s.gravity_w) == 3:
            g = np.array(s.gravity_w, dtype=np.float64)
            norm = np.linalg.norm(g)
            if norm > 0.5:
                gravities.append(g / norm)
    if gravities:
        avg = np.mean(gravities, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0.5:
            return avg / norm
    return None


def _align_to_gravity(points: "np.ndarray", gravity: "np.ndarray") -> "np.ndarray":
    """Rotate points so that gravity aligns with -Y axis."""
    target = np.array([0, -1, 0], dtype=np.float64)

    # Already aligned?
    dot = float(np.dot(gravity, target))
    if abs(dot) > 0.999:
        if dot < 0:
            # Gravity is pointing UP, flip
            points[:, 1] = -points[:, 1]
        return points

    # Rotation axis = cross(gravity, target)
    axis = np.cross(gravity, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return points
    axis = axis / axis_norm

    angle = float(np.arccos(np.clip(dot, -1, 1)))

    # Rodrigues rotation
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float64)

    R = np.eye(3) + sin(angle) * K + (1 - cos(angle)) * (K @ K)
    return (R @ points.T).T


def _ransac_walls(
    points: "np.ndarray",
    n_iter: int = 200,
    dist_thresh: float = 0.1,
    min_inliers_frac: float = 0.05,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    Detect perpendicular wall planes using RANSAC.

    Looks for vertical planes in X and Z directions (Manhattan world assumption).
    Returns (x_min, x_max, z_min, z_max, y_min, y_max) or None.
    """
    if len(points) < 100:
        return None

    rng = np.random.RandomState(42)

    # Separate into floor/ceiling vs walls based on Y distribution
    y_vals = points[:, 1]
    y_med = float(np.median(y_vals))

    # Focus on points near wall height (middle 60% of Y range)
    y_range = float(np.percentile(y_vals, 98)) - float(np.percentile(y_vals, 2))
    wall_mask = np.abs(y_vals - y_med) < y_range * 0.4
    wall_points = points[wall_mask]

    if len(wall_points) < 50:
        wall_points = points

    # Try to find planes along X axis (walls perpendicular to X)
    x_walls = _find_wall_positions(wall_points[:, 0], rng, n_iter, dist_thresh,
                                    min_inliers_frac)
    # Try to find planes along Z axis (walls perpendicular to Z)
    z_walls = _find_wall_positions(wall_points[:, 2], rng, n_iter, dist_thresh,
                                    min_inliers_frac)

    if x_walls is None or z_walls is None:
        logger.debug("RANSAC: failed to find wall pairs (x=%s, z=%s)", x_walls, z_walls)
        return None

    x_min, x_max = x_walls
    z_min, z_max = z_walls

    # Y bounds from full point cloud (Session 10: tighter percentiles)
    y_vals = points[:, 1]
    y_min = float(np.percentile(y_vals, 5))
    y_max = float(np.percentile(y_vals, 95))

    logger.info("RANSAC walls: X=[%.2f, %.2f], Z=[%.2f, %.2f], Y=[%.2f, %.2f]",
                x_min, x_max, z_min, z_max, y_min, y_max)

    return (x_min, x_max, z_min, z_max, y_min, y_max)


def _find_wall_positions(
    coords: "np.ndarray",
    rng,
    n_iter: int,
    dist_thresh: float,
    min_inliers_frac: float,
) -> Optional[Tuple[float, float]]:
    """Find two opposite wall positions along a single axis using RANSAC."""
    n = len(coords)
    min_inliers = max(10, int(n * min_inliers_frac))

    # Histogram approach: find peaks in coordinate distribution
    # This is more robust than RANSAC for axis-aligned walls
    hist_bins = 50
    hist, edges = np.histogram(coords, bins=hist_bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # Smooth histogram
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64) / 9
    if len(hist) > len(kernel):
        smoothed = np.convolve(hist.astype(np.float64), kernel, mode="same")
    else:
        smoothed = hist.astype(np.float64)

    # Find peaks (local maxima above mean)
    mean_val = float(np.mean(smoothed))
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            if smoothed[i] > mean_val * 0.5:
                peaks.append((float(centers[i]), float(smoothed[i])))

    # Also check endpoints
    if len(smoothed) > 0 and smoothed[0] > mean_val * 0.5:
        peaks.insert(0, (float(centers[0]), float(smoothed[0])))
    if len(smoothed) > 1 and smoothed[-1] > mean_val * 0.5:
        peaks.append((float(centers[-1]), float(smoothed[-1])))

    if len(peaks) < 2:
        # Fallback: use tighter percentile range (Session 10: 5/95 instead of 2/98)
        p5 = float(np.percentile(coords, 5))
        p95 = float(np.percentile(coords, 95))
        if p95 - p5 > 1.5:  # At least 1.5m
            logger.debug("Wall positions: percentile fallback [%.2f, %.2f] (no peaks)", p5, p95)
            return (p5, p95)
        return None

    # Pick the two most separated peaks
    peaks.sort(key=lambda p: p[0])
    best_pair = None
    best_span = 0
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            span = peaks[j][0] - peaks[i][0]
            if span > best_span and span > 1.5:
                best_span = span
                best_pair = (peaks[i][0], peaks[j][0])

    if best_pair is not None:
        logger.debug("Wall positions: histogram peaks [%.2f, %.2f]", best_pair[0], best_pair[1])
        return best_pair

    # Final fallback: tighter percentiles (Session 10: 5/95 instead of 2/98)
    p5 = float(np.percentile(coords, 5))
    p95 = float(np.percentile(coords, 95))
    if p95 - p5 > 1.5:
        logger.debug("Wall positions: percentile final fallback [%.2f, %.2f]", p5, p95)
        return (p5, p95)

    return None


def _pca_room_bounds(points: "np.ndarray") -> Optional[Tuple[float, float, float, float, float, float]]:
    """Fallback: estimate room bounds using PCA on XZ plane."""
    if len(points) < 20:
        return None

    # Use X, Z coordinates (floor plane)
    xz = points[:, [0, 2]]

    # PCA to find principal axes
    mean_xz = np.mean(xz, axis=0)
    centered = xz - mean_xz

    try:
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except Exception:
        return None

    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project points onto principal axes
    projected = centered @ eigenvectors

    # Room extent: 5-95 percentile (Session 10: tighter than 2-98)
    p1_min = float(np.percentile(projected[:, 0], 5))
    p1_max = float(np.percentile(projected[:, 0], 95))
    p2_min = float(np.percentile(projected[:, 1], 5))
    p2_max = float(np.percentile(projected[:, 1], 95))

    extent1 = p1_max - p1_min
    extent2 = p2_max - p2_min

    # Y bounds (Session 10: tighter percentiles)
    y_vals = points[:, 1]
    y_min = float(np.percentile(y_vals, 5))
    y_max = float(np.percentile(y_vals, 95))

    # Map back to world coordinates for min/max
    x_vals = points[:, 0]
    z_vals = points[:, 2]
    x_min = float(np.percentile(x_vals, 5))
    x_max = float(np.percentile(x_vals, 95))
    z_min = float(np.percentile(z_vals, 5))
    z_max = float(np.percentile(z_vals, 95))

    logger.info("PCA room bounds: X=[%.2f, %.2f], Z=[%.2f, %.2f], PCA extents=%.2f, %.2f",
                x_min, x_max, z_min, z_max, extent1, extent2)

    return (x_min, x_max, z_min, z_max, y_min, y_max)


def _compute_pose_yaw(shots: List[ShotMetadata]) -> Optional[float]:
    """Compute average camera yaw from pose matrices."""
    yaws = []
    for s in shots:
        if s.pose is not None and len(s.pose) == 16:
            mat = np.array(s.pose, dtype=np.float64).reshape(4, 4)
            # Camera forward is -Z in camera space
            forward = mat[:3, :3] @ np.array([0, 0, -1], dtype=np.float64)
            yaw = atan2(float(forward[0]), float(forward[2]))
            yaws.append(yaw)

    if not yaws:
        return None

    # Circular mean
    sin_sum = sum(sin(y) for y in yaws)
    cos_sum = sum(cos(y) for y in yaws)
    avg_yaw = atan2(sin_sum, cos_sum)
    return avg_yaw


def _generate_walls_centered(
    width: float,
    length: float,
    room_center: Tuple[float, float],
    yaw: Optional[float] = None,
) -> List[Dict]:
    """Generate 4 walls centered at room_center (XZ coords), optionally rotated."""
    cx, cz = room_center
    hw = width / 2
    hl = length / 2
    local_corners = [
        (-hw, -hl),
        (hw, -hl),
        (hw, hl),
        (-hw, hl),
    ]
    if yaw is not None:
        c, s = cos(yaw), sin(yaw)
    else:
        c, s = 1.0, 0.0
    corners = []
    for lx, lz in local_corners:
        wx = cx + lx * c - lz * s
        wz = cz + lx * s + lz * c
        corners.append((wx, wz))
    walls = []
    for i in range(4):
        j = (i + 1) % 4
        walls.append({
            "id": f"wall_{i}",
            "start": [round(corners[i][0], 3), round(corners[i][1], 3)],
            "end": [round(corners[j][0], 3), round(corners[j][1], 3)],
            "has_window": False,
            "has_door": i == 0,
        })
    return walls


def _generate_walls(
    width: float,
    length: float,
    yaw: Optional[float] = None,
) -> List[Dict]:
    """Generate 4 walls as a closed rectangle, optionally rotated by yaw."""
    hw = width / 2
    hl = length / 2

    # Corners: CCW from front-left
    corners = [
        (-hw, -hl),  # front-left
        (hw, -hl),   # front-right
        (hw, hl),    # back-right
        (-hw, hl),   # back-left
    ]

    # Apply rotation if provided
    if yaw is not None:
        c, s = cos(yaw), sin(yaw)
        rotated = []
        for x, z in corners:
            rx = x * c - z * s
            rz = x * s + z * c
            rotated.append((rx, rz))
        corners = rotated

    walls = []
    for i in range(4):
        j = (i + 1) % 4
        wall = {
            "id": f"wall_{i}",
            "start": [round(corners[i][0], 3), round(corners[i][1], 3)],
            "end": [round(corners[j][0], 3), round(corners[j][1], 3)],
            "has_window": False,
            "has_door": i == 0,  # Front wall has door by default
        }
        walls.append(wall)

    return walls


def _debug_layout(layout: Dict) -> None:
    """Write debug layout info to /tmp with quick-glance diagnostics."""
    try:
        import json

        # Add quick-glance diagnostic summary at the top of the report
        w = layout.get("width_m", 0)
        l = layout.get("length_m", 0)
        h = layout.get("height_m", 0)
        ac = layout.get("axis_confidence", 0)
        src = layout.get("layout_source", "unknown")
        fb = layout.get("layout_fallback_reason")
        cam_ext = layout.get("camera_path_extent_xz", {})

        issues = []
        if w < 2.5 or l < 2.5:
            issues.append(f"SMALL_ROOM({w:.1f}x{l:.1f}m)")
        if w > 10 or l > 10:
            issues.append(f"LARGE_ROOM({w:.1f}x{l:.1f}m)")
        if h < 2.2 or h > 3.5:
            issues.append(f"ODD_HEIGHT({h:.1f}m)")
        if ac < 0.5:
            issues.append(f"LOW_AXIS_CONF({ac:.2f})")
        if src == "default":
            issues.append("DEFAULT_LAYOUT(all_priors_failed)")
        if fb:
            issues.append(f"FALLBACK({fb[:60]})")

        layout["_diagnostic_summary"] = {
            "issues": issues if issues else ["NONE"],
            "layout_source": src,
            "axis_confidence": round(ac, 3),
            "room_dims": f"{w:.1f}m x {l:.1f}m x {h:.1f}m",
            "camera_extent_xz": cam_ext,
            "snap_threshold": 0.62,
            "snap_will_apply": ac >= 0.62,
        }

        with open("/tmp/decroom_wall_report.json", "w") as f:
            json.dump(layout, f, indent=2)
        logger.debug("Layout debug written to /tmp/decroom_wall_report.json")
    except Exception:
        pass
