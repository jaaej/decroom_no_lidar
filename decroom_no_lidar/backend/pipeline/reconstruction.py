"""Reconstruction helpers for geometry-first backend upgrades.

This module provides lightweight, dependency-free building blocks that improve
multi-view consistency without introducing heavy external SfM stacks:
  - Pose-path refinement hook (translation smoothing + diagnostics)
  - Sparse multi-view depth fusion for layout assistance
  - Ray-based track triangulation priors (multi-view object positioning)
  - Unified reconstruction report artifact for run-level diagnostics
"""
from __future__ import annotations

import copy
import json
import logging
from math import sqrt
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .metadata import ShotMetadata
from .tracking import Track

logger = logging.getLogger("decroom-backend.reconstruction")


def _structured_unique_points(points: "np.ndarray", voxel_size: float) -> "np.ndarray":
    """Voxel-deduplicate Nx3 points and return one point per occupied voxel."""
    if np is None or len(points) == 0:
        return points

    vox = np.floor(points / max(1e-6, float(voxel_size))).astype(np.int64)
    rec = np.core.records.fromarrays(
        [vox[:, 0], vox[:, 1], vox[:, 2]],
        names="x,y,z",
        formats="i8,i8,i8",
    )
    _, unique_idx = np.unique(rec, return_index=True)
    return points[np.sort(unique_idx)]


def _path_diagnostics(path_xyz: "np.ndarray") -> Dict:
    """Compute path-level motion diagnostics for Nx3 camera positions."""
    if np is None or len(path_xyz) < 2:
        return {
            "path_length_m": 0.0,
            "step_median_m": 0.0,
            "step_p90_m": 0.0,
            "step_max_m": 0.0,
            "accel_median_m": 0.0,
            "accel_p90_m": 0.0,
            "extent_x_m": 0.0,
            "extent_z_m": 0.0,
        }

    diffs = path_xyz[1:] - path_xyz[:-1]
    steps = np.linalg.norm(diffs, axis=1)

    if len(diffs) >= 2:
        acc = diffs[1:] - diffs[:-1]
        acc_norm = np.linalg.norm(acc, axis=1)
        accel_median = float(np.median(acc_norm))
        accel_p90 = float(np.percentile(acc_norm, 90))
    else:
        accel_median = 0.0
        accel_p90 = 0.0

    extent_x = float(np.max(path_xyz[:, 0]) - np.min(path_xyz[:, 0]))
    extent_z = float(np.max(path_xyz[:, 2]) - np.min(path_xyz[:, 2]))

    return {
        "path_length_m": round(float(np.sum(steps)), 4),
        "step_median_m": round(float(np.median(steps)), 4),
        "step_p90_m": round(float(np.percentile(steps, 90)), 4),
        "step_max_m": round(float(np.max(steps)), 4),
        "accel_median_m": round(accel_median, 4),
        "accel_p90_m": round(accel_p90, 4),
        "extent_x_m": round(extent_x, 4),
        "extent_z_m": round(extent_z, 4),
    }


def _window_mean(arr: "np.ndarray", radius: int) -> "np.ndarray":
    """Simple centered moving average for Nx3 arrays."""
    if np is None or len(arr) == 0 or radius <= 0:
        return arr.copy()

    out = np.zeros_like(arr)
    n = len(arr)
    for i in range(n):
        a = max(0, i - radius)
        b = min(n, i + radius + 1)
        out[i] = np.mean(arr[a:b], axis=0)
    return out


def refine_camera_poses(
    shots: List[ShotMetadata],
    cfg,
) -> Tuple[List[ShotMetadata], Dict]:
    """
    Optionally smooth pose translations and return diagnostics.

    Note: rotation matrices are intentionally left unchanged for safety.
    """
    refined = [copy.deepcopy(s) for s in shots]

    diag = {
        "pose_refinement_enabled": bool(getattr(cfg, "enable_pose_path_smoothing", False)),
        "pose_refinement_applied": False,
        "shots_total": int(len(shots)),
        "shots_with_pose": 0,
        "suspicious_pose_jumps": 0,
        "jump_threshold_m": float(getattr(cfg, "pose_jump_threshold_m", 1.2)),
        "path_before": {},
        "path_after": {},
        "fallback_reason": None,
    }

    if np is None:
        diag["fallback_reason"] = "numpy_missing"
        return refined, diag

    pose_indices: List[int] = []
    mats: List["np.ndarray"] = []
    positions: List["np.ndarray"] = []

    for i, shot in enumerate(refined):
        if shot.pose is None or len(shot.pose) != 16:
            continue
        try:
            mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
        except Exception:
            continue
        mats.append(mat)
        pose_indices.append(i)
        positions.append(mat[:3, 3])

    diag["shots_with_pose"] = len(pose_indices)
    if len(positions) < 2:
        diag["fallback_reason"] = "insufficient_pose_count"
        return refined, diag

    pos_arr = np.array(positions, dtype=np.float64)
    diag["path_before"] = _path_diagnostics(pos_arr)

    diffs = pos_arr[1:] - pos_arr[:-1]
    steps = np.linalg.norm(diffs, axis=1)
    jump_thr = float(getattr(cfg, "pose_jump_threshold_m", 1.2))
    diag["suspicious_pose_jumps"] = int(np.sum(steps > jump_thr))

    if not bool(getattr(cfg, "enable_pose_path_smoothing", False)):
        diag["fallback_reason"] = "pose_smoothing_disabled"
        diag["path_after"] = diag["path_before"]
        return refined, diag

    if len(pos_arr) < 3:
        diag["fallback_reason"] = "insufficient_pose_count_for_smoothing"
        diag["path_after"] = diag["path_before"]
        return refined, diag

    radius = max(1, int(getattr(cfg, "pose_smoothing_window", 2)))
    max_adjust = max(0.0, float(getattr(cfg, "pose_max_adjustment_m", 0.08)))

    smoothed = _window_mean(pos_arr, radius=radius)
    delta = smoothed - pos_arr
    delta_norm = np.linalg.norm(delta, axis=1)
    scale = np.ones_like(delta_norm)
    large = delta_norm > max_adjust
    scale[large] = max_adjust / np.maximum(delta_norm[large], 1e-9)
    adjusted = pos_arr + delta * scale[:, None]

    for j, shot_idx in enumerate(pose_indices):
        mat = mats[j].copy()
        mat[:3, 3] = adjusted[j]
        refined[shot_idx].pose = mat.reshape(-1).tolist()

    diag["pose_refinement_applied"] = True
    diag["path_after"] = _path_diagnostics(adjusted)
    diag["smoothing_window_radius"] = radius
    diag["max_adjustment_m"] = round(max_adjust, 4)
    diag["mean_adjustment_m"] = round(float(np.mean(np.linalg.norm(adjusted - pos_arr, axis=1))), 4)

    return refined, diag


def fuse_multiview_geometry(
    depth_maps: List[Optional["np.ndarray"]],
    shots: List[ShotMetadata],
    cfg,
    feature_cloud: Optional["np.ndarray"] = None,
) -> Tuple[Optional["np.ndarray"], Dict]:
    """Build a sparse fused 3D cloud from depth maps + refined poses."""
    diag = {
        "fusion_enabled": bool(getattr(cfg, "enable_sparse_depth_fusion", True)),
        "frames_considered": int(len(depth_maps)),
        "frames_used": 0,
        "raw_depth_points": 0,
        "fused_depth_points": 0,
        "fused_total_points": 0,
        "fusion_stride_px": int(getattr(cfg, "fusion_depth_stride_px", 16)),
        "fusion_voxel_size_m": float(getattr(cfg, "fusion_voxel_size_m", 0.05)),
        "include_feature_cloud": bool(getattr(cfg, "fusion_include_feature_cloud", False)),
        "fallback_reason": None,
    }

    if np is None:
        diag["fallback_reason"] = "numpy_missing"
        return None, diag

    if not bool(getattr(cfg, "enable_sparse_depth_fusion", True)):
        diag["fallback_reason"] = "fusion_disabled"
        return feature_cloud, diag

    stride = max(4, int(getattr(cfg, "fusion_depth_stride_px", 16)))
    max_pf = max(100, int(getattr(cfg, "fusion_max_points_per_frame", 900)))
    max_depth = max(1.0, float(getattr(cfg, "fusion_max_depth_m", 8.0)))
    voxel = max(0.01, float(getattr(cfg, "fusion_voxel_size_m", 0.05)))

    rng = np.random.RandomState(int(getattr(cfg, "random_seed", 42)))
    all_pts: List["np.ndarray"] = []

    for fi, depth in enumerate(depth_maps):
        if depth is None:
            continue
        if fi >= len(shots):
            continue
        shot = shots[fi]
        if shot.pose is None or len(shot.pose) != 16 or shot.intrinsics is None:
            continue

        h, w = depth.shape[:2]
        intr = shot.intrinsics.scaled_to(w, h)

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

        if len(dd) > max_pf:
            idx = rng.choice(len(dd), size=max_pf, replace=False)
            uu = uu[idx]
            vv = vv[idx]
            dd = dd[idx]

        cam_x = (uu - intr.cx) * dd / max(1e-6, intr.fx)
        cam_y = (vv - intr.cy) * dd / max(1e-6, intr.fy)
        cam_z = dd

        pts_cam = np.stack([cam_x, cam_y, cam_z, np.ones_like(cam_x)], axis=1)
        mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
        pts_world = (mat @ pts_cam.T).T[:, :3]

        all_pts.append(pts_world)
        diag["frames_used"] += 1
        diag["raw_depth_points"] += int(len(pts_world))

    if not all_pts:
        if feature_cloud is not None:
            diag["fused_total_points"] = int(len(feature_cloud))
            diag["fallback_reason"] = "no_valid_depth_points"
            return feature_cloud, diag
        diag["fallback_reason"] = "no_valid_depth_points"
        return None, diag

    depth_cloud = np.concatenate(all_pts, axis=0)
    depth_fused = _structured_unique_points(depth_cloud, voxel)
    diag["fused_depth_points"] = int(len(depth_fused))

    include_fc = bool(getattr(cfg, "fusion_include_feature_cloud", False))
    if include_fc and feature_cloud is not None and len(feature_cloud) > 0:
        combined = np.concatenate([depth_fused, feature_cloud], axis=0)
        fused = _structured_unique_points(combined, voxel)
    else:
        fused = depth_fused

    diag["fused_total_points"] = int(len(fused))
    return fused, diag


def _solve_rays_least_squares(
    ray_origins: "np.ndarray",
    ray_dirs: "np.ndarray",
) -> Tuple[Optional["np.ndarray"], float, float]:
    """Find point minimizing squared distances to rays."""
    if np is None or len(ray_origins) < 2:
        return None, float("inf"), float("inf")

    I = np.eye(3, dtype=np.float64)
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)

    for o, d in zip(ray_origins, ray_dirs):
        d = d / max(1e-9, np.linalg.norm(d))
        P = I - np.outer(d, d)
        A += P
        b += P @ o

    try:
        cond = float(np.linalg.cond(A))
    except Exception:
        cond = float("inf")

    if not np.isfinite(cond) or cond > 1e9:
        return None, cond, float("inf")

    try:
        p = np.linalg.solve(A, b)
    except Exception:
        return None, cond, float("inf")

    dists = []
    for o, d in zip(ray_origins, ray_dirs):
        d = d / max(1e-9, np.linalg.norm(d))
        v = p - o
        perp = v - np.dot(v, d) * d
        dists.append(float(np.linalg.norm(perp)))
    median_err = float(np.median(dists)) if dists else float("inf")

    return p, cond, median_err


def triangulate_tracks_from_rays(
    tracks: List[Track],
    shots: List[ShotMetadata],
    cfg,
) -> Tuple[Dict[int, Dict], Dict]:
    """
    Triangulate one 3D center prior per track from multi-view detection-center rays.

    Returns:
      priors_by_track_id and diagnostics dict
    """
    priors: Dict[int, Dict] = {}
    diag = {
        "triangulation_enabled": bool(getattr(cfg, "enable_track_ray_triangulation", True)),
        "tracks_total": int(len(tracks)),
        "tracks_attempted": 0,
        "tracks_with_prior": 0,
        "tracks_rejected": 0,
        "rejected_reasons": {},
        "per_track": [],
        "min_views": int(getattr(cfg, "triangulation_min_views", 2)),
    }

    if np is None:
        diag["rejected_reasons"]["numpy_missing"] = len(tracks)
        return priors, diag

    if not bool(getattr(cfg, "enable_track_ray_triangulation", True)):
        diag["rejected_reasons"]["triangulation_disabled"] = len(tracks)
        return priors, diag

    min_views = max(2, int(getattr(cfg, "triangulation_min_views", 2)))
    max_cond = float(getattr(cfg, "triangulation_max_condition", 1e6))
    max_ray_err = float(getattr(cfg, "triangulation_max_ray_error_m", 0.45))

    for track in tracks:
        per_frame_best = {}
        for det in track.detections:
            prev = per_frame_best.get(det.frame_idx)
            if prev is None or det.score > prev.score:
                per_frame_best[det.frame_idx] = det

        origins = []
        dirs = []
        used_frames = []

        for fi, det in sorted(per_frame_best.items()):
            if fi >= len(shots):
                continue
            shot = shots[fi]
            if shot.pose is None or len(shot.pose) != 16 or shot.intrinsics is None:
                continue

            mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
            intr = shot.intrinsics

            x1, y1, x2, y2 = det.bbox
            det_w, det_h = det.input_size
            u_det = 0.5 * (x1 + x2)
            v_det = 0.5 * (y1 + y2)
            u = u_det * intr.width / max(1.0, float(det_w))
            v = v_det * intr.height / max(1.0, float(det_h))

            dx = (u - intr.cx) / max(1e-6, intr.fx)
            dy = (v - intr.cy) / max(1e-6, intr.fy)
            dz = 1.0
            d_cam = np.array([dx, dy, dz], dtype=np.float64)
            d_cam /= max(1e-9, np.linalg.norm(d_cam))

            R = mat[:3, :3]
            d_world = R @ d_cam
            d_world /= max(1e-9, np.linalg.norm(d_world))

            origins.append(mat[:3, 3].copy())
            dirs.append(d_world)
            used_frames.append(int(fi))

        diag["tracks_attempted"] += 1

        if len(origins) < min_views:
            reason = "insufficient_views"
            diag["tracks_rejected"] += 1
            diag["rejected_reasons"][reason] = diag["rejected_reasons"].get(reason, 0) + 1
            diag["per_track"].append({
                "track_id": int(track.track_id),
                "category": str(track.category),
                "status": "rejected",
                "reason": reason,
                "views": int(len(origins)),
            })
            continue

        ray_origins = np.array(origins, dtype=np.float64)
        ray_dirs = np.array(dirs, dtype=np.float64)

        p, cond, med_err = _solve_rays_least_squares(ray_origins, ray_dirs)
        if p is None:
            reason = "ill_conditioned_system"
            diag["tracks_rejected"] += 1
            diag["rejected_reasons"][reason] = diag["rejected_reasons"].get(reason, 0) + 1
            diag["per_track"].append({
                "track_id": int(track.track_id),
                "category": str(track.category),
                "status": "rejected",
                "reason": reason,
                "views": int(len(origins)),
                "condition_number": None if not np.isfinite(cond) else float(round(cond, 3)),
            })
            continue

        if cond > max_cond:
            reason = "condition_too_high"
            diag["tracks_rejected"] += 1
            diag["rejected_reasons"][reason] = diag["rejected_reasons"].get(reason, 0) + 1
            diag["per_track"].append({
                "track_id": int(track.track_id),
                "category": str(track.category),
                "status": "rejected",
                "reason": reason,
                "views": int(len(origins)),
                "condition_number": float(round(cond, 3)),
            })
            continue

        if med_err > max_ray_err:
            reason = "ray_residual_too_high"
            diag["tracks_rejected"] += 1
            diag["rejected_reasons"][reason] = diag["rejected_reasons"].get(reason, 0) + 1
            diag["per_track"].append({
                "track_id": int(track.track_id),
                "category": str(track.category),
                "status": "rejected",
                "reason": reason,
                "views": int(len(origins)),
                "median_ray_error_m": float(round(med_err, 4)),
                "condition_number": float(round(cond, 3)),
            })
            continue

        priors[int(track.track_id)] = {
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
            "views": int(len(origins)),
            "median_ray_error_m": float(round(med_err, 4)),
            "condition_number": float(round(cond, 3)),
            "frames": used_frames,
        }
        diag["tracks_with_prior"] += 1
        diag["per_track"].append({
            "track_id": int(track.track_id),
            "category": str(track.category),
            "status": "accepted",
            "views": int(len(origins)),
            "median_ray_error_m": float(round(med_err, 4)),
            "condition_number": float(round(cond, 3)),
        })

    return priors, diag


def summarize_shot_coverage(shots: List[ShotMetadata]) -> Dict:
    """Compute scan-level metadata quality summaries."""
    total = len(shots)
    fp_counts = [len(s.feature_points_w) if s.feature_points_w else 0 for s in shots]
    fp_uv_counts = [len(s.feature_points_uv) if s.feature_points_uv else 0 for s in shots]
    shots_with_pose = sum(1 for s in shots if s.pose is not None and len(s.pose) == 16)
    shots_with_intr = sum(1 for s in shots if s.intrinsics is not None)
    shots_with_planes = sum(1 for s in shots if s.plane_anchors)
    texture_scores = []
    blur_scores = []
    edge_densities = []
    matchability = []
    capture_steps = []
    capture_yaw_deltas = []
    for s in shots:
        if s.quality is not None:
            texture_scores.append(float(s.quality.texture_score))
            blur_scores.append(float(s.quality.blur_score))
            edge_densities.append(float(s.quality.edge_density))
            matchability.append(float(s.quality.matchability_seed))
        if s.capture_step_m is not None:
            capture_steps.append(float(s.capture_step_m))
        if s.capture_yaw_delta_deg is not None:
            capture_yaw_deltas.append(float(s.capture_yaw_delta_deg))

    if not fp_counts:
        fp_counts = [0]
    if not fp_uv_counts:
        fp_uv_counts = [0]

    def _arr_stats(vals: List[float]) -> Dict:
        if not vals:
            return {
                "min": 0.0,
                "median": 0.0,
                "p90": 0.0,
                "max": 0.0,
            }
        if np is None:
            vv = sorted(vals)
            mid = vv[len(vv) // 2]
            p90_idx = int(0.9 * max(0, len(vv) - 1))
            return {
                "min": round(float(vv[0]), 4),
                "median": round(float(mid), 4),
                "p90": round(float(vv[p90_idx]), 4),
                "max": round(float(vv[-1]), 4),
            }
        arr = np.array(vals, dtype=np.float64)
        return {
            "min": round(float(np.min(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    return {
        "shots_total": int(total),
        "shots_with_pose": int(shots_with_pose),
        "shots_with_intrinsics": int(shots_with_intr),
        "shots_with_plane_anchors": int(shots_with_planes),
        "fp_per_shot": {
            "min": int(min(fp_counts)),
            "median": float(round(float(np.median(fp_counts)) if np is not None else 0.0, 2)),
            "p90": float(round(float(np.percentile(fp_counts, 90)) if np is not None else 0.0, 2)),
            "max": int(max(fp_counts)),
            "shots_lt_10": int(sum(1 for c in fp_counts if c < 10)),
            "shots_eq_0": int(sum(1 for c in fp_counts if c == 0)),
        },
        "fp_uv_per_shot": {
            "min": int(min(fp_uv_counts)),
            "median": float(round(float(np.median(fp_uv_counts)) if np is not None else 0.0, 2)),
            "p90": float(round(float(np.percentile(fp_uv_counts, 90)) if np is not None else 0.0, 2)),
            "max": int(max(fp_uv_counts)),
            "shots_lt_10": int(sum(1 for c in fp_uv_counts if c < 10)),
            "shots_eq_0": int(sum(1 for c in fp_uv_counts if c == 0)),
        },
        "shot_quality": {
            "texture_score": _arr_stats(texture_scores),
            "blur_score": _arr_stats(blur_scores),
            "edge_density": _arr_stats(edge_densities),
            "matchability_seed": _arr_stats(matchability),
        },
        "capture_step_m": _arr_stats(capture_steps),
        "capture_yaw_delta_deg": _arr_stats(capture_yaw_deltas),
    }


def save_reconstruction_report(report: Dict, path: str = "/tmp/decroom_reconstruction_report.json") -> None:
    """Persist reconstruction diagnostics for run triage."""
    try:
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Reconstruction report saved to %s", path)
    except Exception as e:
        logger.warning("Failed to save reconstruction report: %s", e)


def build_reconstruction_report(
    *,
    sfm_diag: Optional[Dict] = None,
    pose_diag: Dict,
    fusion_diag: Dict,
    triangulation_diag: Dict,
    shot_summary: Dict,
    layout: Dict,
    detections_count: int,
    geometry_detections_count: Optional[int] = None,
    tracks_count: int,
    objects_count: int,
    depth_role: str,
    depth_model_type: str,
    depth_reliability: float,
    tracking_diag: Optional[Dict] = None,
    geometry_frames_used: Optional[List[int]] = None,
    geometry_frames_dropped: Optional[List[int]] = None,
    geometry_drop_reasons: Optional[Dict] = None,
    suppressed_inferred: Optional[List[Dict]] = None,
) -> Dict:
    """Compose a single reconstruction diagnostics payload."""
    room_w = float(layout.get("width_m", 0.0))
    room_l = float(layout.get("length_m", 0.0))
    room_diag = sqrt(max(0.0, room_w * room_w + room_l * room_l))
    tracking_diag = tracking_diag or {}
    geometry_frames_used = geometry_frames_used or []
    geometry_frames_dropped = geometry_frames_dropped or []
    geometry_drop_reasons = geometry_drop_reasons or {}
    suppressed_inferred = suppressed_inferred or []

    failure_scores: Dict[str, float] = {}
    sfm_info = sfm_diag or {}
    if bool(sfm_info.get("sfm_attempted", False)) and not bool(sfm_info.get("sfm_applied", False)):
        failure_scores["sfm_fallback"] = failure_scores.get("sfm_fallback", 0.0) + 3.0
    reg_ratio = float(sfm_info.get("registration_ratio", 0.0) or 0.0)
    if reg_ratio > 0 and reg_ratio < 0.25:
        failure_scores["weak_sfm_registration"] = (
            failure_scores.get("weak_sfm_registration", 0.0) + 2.0
        )
    fp_stats = shot_summary.get("fp_per_shot", {})
    shots_total = max(1, int(shot_summary.get("shots_total", 0)))
    low_fp_ratio = float(fp_stats.get("shots_lt_10", 0)) / float(shots_total)
    zero_fp_ratio = float(fp_stats.get("shots_eq_0", 0)) / float(shots_total)
    if low_fp_ratio >= 0.30:
        failure_scores["sparse_feature_points"] = (
            failure_scores.get("sparse_feature_points", 0.0) + 2.0
        )
    if zero_fp_ratio >= 0.10:
        failure_scores["feature_point_dropouts"] = (
            failure_scores.get("feature_point_dropouts", 0.0) + 1.5
        )
    frag_rates = tracking_diag.get("fragmentation_rate_by_category", {}) or {}
    if frag_rates:
        frag_max = max(float(v) for v in frag_rates.values())
        if frag_max >= 0.65:
            failure_scores["track_fragmentation"] = (
                failure_scores.get("track_fragmentation", 0.0) + 2.0
            )
    if int(objects_count) == 0:
        failure_scores["no_objects_output"] = failure_scores.get("no_objects_output", 0.0) + 3.0
    if geometry_frames_dropped:
        drop_ratio = float(len(geometry_frames_dropped)) / float(
            max(1, len(geometry_frames_used) + len(geometry_frames_dropped))
        )
        if drop_ratio >= 0.25:
            failure_scores["geometry_frame_drops"] = (
                failure_scores.get("geometry_frame_drops", 0.0) + 1.5
            )
    if int(len(suppressed_inferred)) >= 2:
        failure_scores["context_inference_suppressed"] = (
            failure_scores.get("context_inference_suppressed", 0.0) + 0.8
        )

    ranked_failures = sorted(
        [{"mode": k, "score": round(float(v), 3)} for k, v in failure_scores.items()],
        key=lambda x: (-x["score"], x["mode"]),
    )
    primary_failure_mode = ranked_failures[0]["mode"] if ranked_failures else "none"

    return {
        "summary": {
            "detections_total": int(detections_count),
            "geometry_detections_total": int(
                geometry_detections_count if geometry_detections_count is not None else detections_count
            ),
            "tracks_total": int(tracks_count),
            "objects_total": int(objects_count),
            "depth_role": str(depth_role),
            "depth_model_type": str(depth_model_type),
            "depth_reliability": round(float(depth_reliability), 3),
            "layout_source": str(layout.get("layout_source", "unknown")),
            "axis_confidence": round(float(layout.get("axis_confidence", 0.0)), 3),
            "axis_confidence_for_lifting": round(float(layout.get("axis_confidence_for_lifting", layout.get("axis_confidence", 0.0))), 3),
            "axis_snap_enabled": bool(layout.get("axis_snap_enabled", True)),
            "room_diag_m": round(room_diag, 4),
            "primary_failure_mode": primary_failure_mode,
            "primary_failure_mode_ranked": ranked_failures[:3],
            "failure_mode_contributions": ranked_failures,
        },
        "shot_coverage": shot_summary,
        "tracking": tracking_diag,
        "sfm_ba": sfm_diag or {},
        "sfm_apply_quality_gate": (sfm_info or {}).get("sfm_apply_quality_gate", {}),
        "pose_refinement": pose_diag,
        "fusion": fusion_diag,
        "track_triangulation": triangulation_diag,
        "geometry": {
            "geometry_frames_used": [int(i) for i in geometry_frames_used],
            "geometry_frames_dropped": [int(i) for i in geometry_frames_dropped],
            "geometry_drop_reasons": geometry_drop_reasons,
        },
        "suppressed_inferred_objects": suppressed_inferred,
        "primary_failure_mode_ranked": ranked_failures[:3],
        "layout": {
            "width_m": float(layout.get("width_m", 0.0)),
            "length_m": float(layout.get("length_m", 0.0)),
            "height_m": float(layout.get("height_m", 0.0)),
            "layout_source": str(layout.get("layout_source", "unknown")),
            "layout_fallback_reason": layout.get("layout_fallback_reason"),
            "floor_ransac_inliers": layout.get("floor_ransac_inliers"),
            "floor_ransac_rmse": layout.get("floor_ransac_rmse"),
            "wall_fit_residual": layout.get("wall_fit_residual"),
            "num_near_floor_points": layout.get("num_near_floor_points"),
            "camera_path_extent_xz": layout.get("camera_path_extent_xz"),
            "axis_confidence_for_lifting": layout.get("axis_confidence_for_lifting"),
            "axis_snap_enabled": layout.get("axis_snap_enabled"),
            "axis_snap_disable_reason": layout.get("axis_snap_disable_reason"),
            "axis_snap_camera_extent_diag_m": layout.get("axis_snap_camera_extent_diag_m"),
            "axis_snap_camera_extent_ratio": layout.get("axis_snap_camera_extent_ratio"),
        },
    }
