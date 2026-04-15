"""
Main pipeline orchestrator for the Decroom backend.
Coordinates: depth estimation → detection → tracking → layout → 3D lifting.

Session 6 fixes included:
- Reprojection: 0.5m absolute OR 30% relative agreement, non-visible points kept
- Room size: 2-98 percentile, 3.0m minimum clamp
- Depth scaling using median ratio between metric and relative depth
"""
from __future__ import annotations

import json
import logging
import time
from math import pi, sqrt
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .config import PipelineConfig
from .depth import DepthRunner, estimate_depth
from .detection import Detection, YoloSegOnnxDetector, detect_objects
from .detection_open_vocab import YoloWorldOnnxDetector
from .geometry import build_global_feature_cloud
from .layout import estimate_layout
from .lifting import lift_objects, get_lift_diagnostics
from .spatiallm_layout import run_spatiallm_pipeline
from .metadata import CameraIntrinsics, ShotMetadata, parse_metadata_json
from .tracking import track_detections, save_tracking_overlays, get_tracking_diagnostics
from .reconstruction import (
    build_reconstruction_report,
    fuse_multiview_geometry,
    refine_camera_poses,
    save_reconstruction_report,
    summarize_shot_coverage,
    triangulate_tracks_from_rays,
)
from .sfm import run_sfm_bundle_adjustment

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import RoomPayload, RoomInfo, Wall, RoomObject, ObjectPosition, ObjectMetadata

logger = logging.getLogger("decroom-backend.pipeline")

# Cached ONNX runners to avoid expensive model/session reload on every request.
_cache_lock = Lock()
_cached_depth_runner: Optional[DepthRunner] = None
_cached_detector: Optional[YoloSegOnnxDetector] = None
_cached_world_detector: Optional[YoloWorldOnnxDetector] = None
_cached_world_detector_path: Optional[Path] = None


def run_pipeline(
    frames,
    metadata_raw: Optional[dict] = None,
    cfg: Optional[PipelineConfig] = None,
) -> Dict:
    """
    Run the full Decroom pipeline on uploaded frames.

    Args:
        frames: List of uploaded image files (FastAPI UploadFile or file paths)
        metadata_raw: Optional JSON metadata from iOS app
        cfg: Pipeline configuration (uses defaults if None)

    Returns:
        RoomPayload dict ready for JSON serialization
    """
    t0 = time.time()
    if cfg is None:
        cfg = PipelineConfig()
    frames = list(frames)

    logger.info("=" * 60)
    logger.info("DECROOM PIPELINE START — %d frames", len(frames))
    logger.info("=" * 60)

    global _scale_diagnostics
    _scale_diagnostics = {}

    # Parse metadata
    mode = "guided"
    shots: List[ShotMetadata] = []
    if metadata_raw:
        try:
            mode, shots = parse_metadata_json(metadata_raw)
            logger.info("Mode: %s, Shots: %d", mode, len(shots))
        except Exception as e:
            logger.warning("Failed to parse metadata: %s", e)

    # Pad shots list to match frame count
    while len(shots) < len(frames):
        shots.append(ShotMetadata())

    # ── Step -1: Optional SfM + BA pose solve ────────────────
    if cfg.enable_spatiallm:
        logger.info("── Step -1: SfM SKIPPED (SpatialLM enabled, using ARKit poses) ──")
        sfm_diag = {"sfm_enabled": False, "sfm_attempted": False, "sfm_applied": False,
                     "fallback_reason": "skipped_spatiallm_enabled", "aligned_poses": 0}
    else:
        logger.info("── Step -1: SfM + BA (optional) ──")
        t_sfm = time.time()
        shots, sfm_diag = run_sfm_bundle_adjustment(frames, shots, cfg)
        logger.info(
            "SfM+BA: %.1fs, attempted=%s, applied=%s, aligned_poses=%d, reason=%s",
            time.time() - t_sfm,
            sfm_diag.get("sfm_attempted", False),
            sfm_diag.get("sfm_applied", False),
            sfm_diag.get("aligned_poses", 0),
            sfm_diag.get("fallback_reason"),
        )

    # ── Step 0: Pose-path refinement hook ─────────────────────
    logger.info("── Step 0: Pose Refinement ──")
    t_pose = time.time()
    shots, pose_diag = refine_camera_poses(shots, cfg)
    logger.info(
        "Pose refinement: %.1fs, applied=%s, shots_with_pose=%d/%d",
        time.time() - t_pose,
        pose_diag.get("pose_refinement_applied", False),
        pose_diag.get("shots_with_pose", 0),
        pose_diag.get("shots_total", len(shots)),
    )
    frames, shots, pose_diag = _reject_pose_jump_frames(frames, shots, pose_diag, cfg)
    geometry_keep_indices = pose_diag.get("geometry_keep_indices")
    if not isinstance(geometry_keep_indices, list):
        geometry_keep_indices = list(range(len(frames)))
    geometry_keep_indices = [
        int(i) for i in geometry_keep_indices if isinstance(i, int) and 0 <= int(i) < len(frames)
    ]
    if not geometry_keep_indices:
        geometry_keep_indices = list(range(len(frames)))
    geometry_drop_indices = sorted(
        int(i) for i in pose_diag.get("geometry_dropped_frames", [])
        if isinstance(i, int) and 0 <= int(i) < len(frames)
    )
    geometry_idx_set = set(geometry_keep_indices)
    geometry_shots = [shots[i] for i in geometry_keep_indices]

    # ── Step 1: Depth Estimation ──────────────────────────────
    logger.info("── Step 1: Depth Estimation ──")
    t1 = time.time()
    global _cached_depth_runner
    with _cache_lock:
        cached_depth_runner = _cached_depth_runner

    depth_stride = int(getattr(cfg, "video_depth_stride", 1))
    if len(frames) > 60 and depth_stride > 1:
        # Video mode: stride depth estimation for efficiency, interpolate gaps
        depth_indices = list(range(0, len(frames), depth_stride))
        depth_frames_subset = [frames[i] for i in depth_indices]
        depth_maps_subset, depth_runner = estimate_depth(
            depth_frames_subset, cfg, cached_runner=cached_depth_runner
        )
        # Fill gaps: assign nearest computed depth to skipped frames
        depth_maps = [None] * len(frames)
        for i, di in enumerate(depth_indices):
            depth_maps[di] = depth_maps_subset[i]
        for i in range(len(frames)):
            if depth_maps[i] is None:
                nearest = min(depth_indices, key=lambda di: abs(di - i))
                depth_maps[i] = depth_maps[nearest]
        logger.info("Video depth stride=%d: computed %d/%d, interpolated rest",
                    depth_stride, len(depth_indices), len(frames))
    else:
        depth_maps, depth_runner = estimate_depth(frames, cfg, cached_runner=cached_depth_runner)

    if depth_runner is not None:
        with _cache_lock:
            _cached_depth_runner = depth_runner
    dm_shapes = [d.shape if d is not None else None for d in depth_maps]
    logger.info("Depth: %.1fs, %d/%d valid maps, shapes=%s",
                time.time() - t1,
                sum(1 for d in depth_maps if d is not None),
                len(frames),
                dm_shapes[:3])

    # ── Step 2: Object Detection ──────────────────────────────
    logger.info("── Step 2: Object Detection ──")
    t2 = time.time()

    # Log frame sizes
    frame_shapes = [(f.shape if hasattr(f, 'shape') else 'unknown') for f in frames[:3]]
    logger.info("Frame shapes (first 3): %s", frame_shapes)

    # YOLOv8-seg (COCO80)
    global _cached_detector
    with _cache_lock:
        cached_detector = _cached_detector
    detections, detector = detect_objects(frames, cfg, cached_detector=cached_detector)
    if detector is not None:
        with _cache_lock:
            _cached_detector = detector

    # YOLO-World (open vocabulary)
    ov_dets = _run_open_vocab(frames, cfg)
    if ov_dets:
        detections.extend(ov_dets)
        logger.info("Combined detections: %d (COCO) + %d (YOLO-World) = %d",
                     len(detections) - len(ov_dets), len(ov_dets), len(detections))

    logger.info("Detection: %.1fs, %d total detections", time.time() - t2, len(detections))
    if not detections:
        logger.warning("No detections found; continuing geometry-first room reconstruction")

    geometry_depth_maps = [depth_maps[i] for i in geometry_keep_indices]

    # ── Step 2b: Build Global Feature Cloud (Session 11: PRIMARY geometry) ──
    logger.info("── Step 2b: Feature Cloud ──")
    feature_cloud = build_global_feature_cloud(geometry_shots)
    fc_size = len(feature_cloud) if feature_cloud is not None else 0
    logger.info(
        "Feature cloud: %d metric points (geometry_frames=%d/%d, dropped=%d)",
        fc_size,
        len(geometry_keep_indices),
        len(frames),
        len(geometry_drop_indices),
    )

    # ── Step 2c: Depth Scaling ──────────────────────────────────
    # ALWAYS scale depth maps to metric when using relative depth model.
    # Even when feature cloud is the primary geometry source (≥200 pts),
    # individual objects with few in-bbox feature points (<5) fall back to
    # depth_backproject. Without scaling, raw model output (e.g. 0.16) is
    # used as meters, placing objects at the camera instead of in the room.
    depth_role = "refinement_only"
    depth_reliability = 1.0
    if fc_size >= 200:
        logger.info("Feature cloud sufficient (%d pts), depth is secondary "
                     "(dimensions + per-object fallback)", fc_size)
    else:
        logger.info("Feature cloud sparse (%d pts), depth is primary fallback", fc_size)
        depth_role = "primary_fallback"

    # Always scale relative depth maps — needed for both roles
    if cfg.depth_model_type != "dav2_metric" and np is not None:
        depth_maps = _scale_relative_depths(depth_maps, shots)
        depth_reliability = 1.0
    elif cfg.depth_model_type == "dav2_metric" and np is not None:
        metric_diag = _metric_depth_sanity(depth_maps, shots)
        depth_maps, metric_diag = _apply_metric_fp_scale_calibration(depth_maps, metric_diag, cfg)
        _scale_diagnostics.update(metric_diag)
        depth_reliability = float(metric_diag.get("metric_depth_reliability", 0.65))

    _scale_diagnostics["depth_role"] = depth_role
    _scale_diagnostics["depth_model_type"] = cfg.depth_model_type

    # ── Step 2d: Sparse Multi-View Fusion ──────────────────────
    logger.info("── Step 2d: Multi-View Fusion ──")
    t2d = time.time()
    fused_cloud, fusion_diag = fuse_multiview_geometry(
        geometry_depth_maps,
        geometry_shots,
        cfg,
        feature_cloud=feature_cloud,
    )
    fused_size = len(fused_cloud) if fused_cloud is not None else 0
    logger.info(
        "Fusion: %.1fs, fused_points=%d (depth_points=%d, frames_used=%d)",
        time.time() - t2d,
        fused_size,
        int(fusion_diag.get("fused_depth_points", 0)),
        int(fusion_diag.get("frames_used", 0)),
    )

    layout_feature_cloud = feature_cloud
    if bool(getattr(cfg, "enable_fused_layout_assist", True)):
        min_fc = int(getattr(cfg, "min_feature_cloud_for_layout", 80))
        fc_sparse = feature_cloud is None or len(feature_cloud) < min_fc
        if fc_sparse and fused_cloud is not None and len(fused_cloud) >= 50:
            layout_feature_cloud = fused_cloud
            logger.info(
                "Layout assist: using fused cloud (%d pts) instead of sparse feature cloud (%d pts)",
                len(fused_cloud),
                len(feature_cloud) if feature_cloud is not None else 0,
            )

    # ── Step 3: Object Tracking (3D-aware, FP-first) ──────────
    logger.info("── Step 3: Object Tracking ──")
    t3 = time.time()
    geometry_detections = [d for d in detections if int(d.frame_idx) in geometry_idx_set]
    detections_dropped_for_geometry = max(0, len(detections) - len(geometry_detections))
    if detections_dropped_for_geometry > 0:
        logger.info(
            "Geometry-stage frame filter: %d/%d detections retained (%d dropped)",
            len(geometry_detections),
            len(detections),
            detections_dropped_for_geometry,
        )

    tracks = track_detections(
        geometry_detections,
        frames=frames,
        depth_maps=depth_maps,
        shots=shots,
        enable_tracklet_stitching=bool(getattr(cfg, "enable_tracklet_stitching", True)),
        stitch_max_gap_frames=int(getattr(cfg, "tracklet_stitch_max_gap_frames", 8)),
        stitch_world_dist_factor=float(getattr(cfg, "tracklet_stitch_world_dist_factor", 1.35)),
        stitch_min_score=float(getattr(cfg, "tracklet_stitch_min_score", 0.25)),
        stitch_max_passes=int(getattr(cfg, "tracklet_stitch_max_passes", 6)),
        enable_track_duplicate_merge=bool(getattr(cfg, "enable_track_duplicate_merge", True)),
        duplicate_iou_min=float(getattr(cfg, "track_duplicate_iou_min", 0.22)),
        duplicate_center_max=float(getattr(cfg, "track_duplicate_center_max", 0.5)),
        duplicate_world_dist_factor=float(getattr(cfg, "track_duplicate_world_dist_factor", 0.9)),
        duplicate_min_score=float(getattr(cfg, "track_duplicate_min_score", 0.2)),
        duplicate_max_passes=int(getattr(cfg, "track_duplicate_max_passes", 8)),
        appearance_weight=float(getattr(cfg, "track_appearance_weight", 0.25)),
        high_similarity_relaxation=float(getattr(cfg, "track_high_similarity_relaxation", 1.18)),
        fragmentation_alert_threshold=float(
            getattr(cfg, "tracking_fragmentation_alert_threshold", 0.65)
        ),
        new_track_min_score_by_cat=getattr(cfg, "track_new_track_min_score_by_cat", {}),
        sfm_registration_ratio=float(sfm_diag.get("registration_ratio", 0.0) or 0.0),
        sfm_alignment_rmse_m=(
            float(sfm_diag.get("alignment_rmse_m"))
            if sfm_diag.get("alignment_rmse_m") is not None
            else None
        ),
        monitor_weak_sfm_relaxation=float(
            getattr(cfg, "track_monitor_weak_sfm_relaxation", 1.25)
        ),
        enable_singleton_merge=bool(getattr(cfg, "enable_singleton_merge", True)),
        singleton_merge_world_dist_factor=float(
            getattr(cfg, "singleton_merge_world_dist_factor", 1.5)
        ),
        singleton_merge_min_appearance=float(
            getattr(cfg, "singleton_merge_min_appearance", 0.70)
        ),
        singleton_merge_high_conf_threshold=float(
            getattr(cfg, "singleton_merge_high_conf_threshold", 0.80)
        ),
        stitch_2d_fallback_max_gap=int(
            getattr(cfg, "tracklet_stitch_2d_fallback_max_gap", 6)
        ),
    )
    tracking_diag = get_tracking_diagnostics()
    logger.info("Tracking: %.1fs, %d tracks from %d detections",
                time.time() - t3, len(tracks), len(geometry_detections))

    # Save tracking debug overlays
    if cfg.debug_output and tracks:
        save_tracking_overlays(tracks, frames, debug_dir="/tmp")

    # ── Steps 4-5: Layout + Lifting (SpatialLM or heuristic) ──
    spatiallm_used = False
    layout = None
    objects = []
    triangulation_diag: Dict = {
        "triangulation_enabled": False,
        "tracks_total": len(tracks),
        "tracks_attempted": 0,
        "tracks_with_prior": 0,
        "tracks_rejected": 0,
        "rejected_reasons": {"not_run": len(tracks)},
        "per_track": [],
    }

    if cfg.enable_spatiallm:
        logger.info("── Step 4/5: SpatialLM Layout + Objects ──")
        t_slm = time.time()
        try:
            slm_layout, slm_objects = run_spatiallm_pipeline(
                frames, depth_maps, geometry_shots, cfg,
                yolo_objects=None,  # merge handled below if needed
            )
            layout = slm_layout
            objects = slm_objects
            spatiallm_used = True
            logger.info(
                "SpatialLM: %.1fs, %d walls, %d objects, layout_source=%s",
                time.time() - t_slm,
                len(layout.get("walls", [])),
                len(objects),
                layout.get("layout_source"),
            )
        except Exception as e:
            import traceback
            logger.warning("SpatialLM failed: %s\n%s", e, traceback.format_exc())
            if not cfg.spatiallm_fallback_on_failure:
                raise
            logger.info("Falling back to heuristic pipeline")

    if not spatiallm_used:
        # ── Step 4: Room Layout (FP-first, heuristic) ──────────────
        logger.info("── Step 4: Room Layout ──")
        t4 = time.time()
        layout = estimate_layout(
            geometry_depth_maps,
            geometry_shots,
            cfg,
            mode,
            feature_cloud=layout_feature_cloud,
        )
        logger.info("Layout: %.1fs, %.1fm x %.1fm x %.1fm (source=%s, floor=%.3f, axis_conf=%.2f)",
                    time.time() - t4, layout["width_m"], layout["length_m"], layout["height_m"],
                    layout.get("layout_source", "unknown"), layout.get("floor_height", 0.0),
                    float(layout.get("axis_confidence", 0.5)))

    room_width = layout["width_m"]
    room_length = layout["length_m"]
    room_height = layout["height_m"]
    walls = layout["walls"]
    floor_height = layout.get("floor_height", 0.0)
    room_center = layout.get("room_center", [0.0, 0.0])
    layout_source = layout.get("layout_source", "unknown")
    axis_confidence = float(layout.get("axis_confidence", 0.5))
    axis_confidence_for_lifting = axis_confidence

    # Disable axis snapping when retained camera motion is too small relative
    # to room scale; otherwise weak yaw can force wrong object orientations.
    cam_extent = layout.get("camera_path_extent_xz") or {}
    try:
        cam_extent_w = float(cam_extent.get("width", 0.0))
    except Exception:
        cam_extent_w = 0.0
    try:
        cam_extent_l = float(cam_extent.get("length", 0.0))
    except Exception:
        cam_extent_l = 0.0
    cam_extent_diag = sqrt(max(0.0, cam_extent_w * cam_extent_w + cam_extent_l * cam_extent_l))
    room_diag = sqrt(max(0.0, room_width * room_width + room_length * room_length))
    cam_extent_ratio = cam_extent_diag / max(1e-6, room_diag)
    min_extent_m = max(0.1, float(getattr(cfg, "axis_snap_min_camera_extent_m", 0.9)))
    min_extent_ratio = max(0.01, float(getattr(cfg, "axis_snap_min_camera_extent_ratio", 0.16)))
    axis_snap_enabled = (cam_extent_diag >= min_extent_m) and (cam_extent_ratio >= min_extent_ratio)
    if not axis_snap_enabled and not spatiallm_used:
        axis_confidence_for_lifting = 0.0
        layout["axis_snap_enabled"] = False
        layout["axis_snap_disable_reason"] = (
            f"camera_extent_too_small(diag={cam_extent_diag:.3f}m,"
            f"ratio={cam_extent_ratio:.3f},min={min_extent_m:.2f}m/{min_extent_ratio:.2f})"
        )
        logger.warning(
            "Axis snap disabled for this run: camera_extent_diag=%.3fm ratio=%.3f "
            "(required >= %.2fm and >= %.2f of room diag)",
            cam_extent_diag,
            cam_extent_ratio,
            min_extent_m,
            min_extent_ratio,
        )
    else:
        layout["axis_snap_enabled"] = True
        layout["axis_snap_disable_reason"] = None
    layout["axis_snap_camera_extent_diag_m"] = round(cam_extent_diag, 4)
    layout["axis_snap_camera_extent_ratio"] = round(cam_extent_ratio, 4)
    layout["axis_confidence_for_lifting"] = round(axis_confidence_for_lifting, 3)

    # ── Layout sanity cross-check: compare room dims to camera trajectory ──
    cam_positions = []
    for s in shots:
        if s.pose is not None and len(s.pose) == 16:
            mat = np.array(s.pose, dtype=np.float64).reshape(4, 4)
            cam_positions.append(mat[:3, 3])
    if len(cam_positions) >= 2:
        cam_arr = np.array(cam_positions)
        cam_span_x = float(np.max(cam_arr[:, 0]) - np.min(cam_arr[:, 0]))
        cam_span_z = float(np.max(cam_arr[:, 2]) - np.min(cam_arr[:, 2]))
        cam_diag = sqrt(cam_span_x ** 2 + cam_span_z ** 2)
        room_diag_check = sqrt(room_width ** 2 + room_length ** 2)
        if cam_diag > room_diag_check * 1.5:
            logger.warning(
                "⚠ LAYOUT SANITY: camera trajectory (%.1fm diag) exceeds room (%.1fm diag) by %.0f%%",
                cam_diag, room_diag_check, (cam_diag / room_diag_check - 1) * 100)
        elif room_diag_check > cam_diag * 5:
            logger.warning(
                "⚠ LAYOUT SANITY: room (%.1fm diag) is %.0fx camera trajectory (%.1fm diag)",
                room_diag_check, room_diag_check / max(0.01, cam_diag), cam_diag)
        else:
            logger.info("Layout sanity OK: room_diag=%.1fm, cam_diag=%.1fm",
                        room_diag_check, cam_diag)

    # ── Steps 5a/5b: Triangulation + Lifting (heuristic, skipped if SpatialLM succeeded) ──
    if not spatiallm_used:
        triangulation_priors: Dict[int, Dict] = {}
        if tracks:
            logger.info("── Step 5a: Track Triangulation ──")
            t5a = time.time()
            triangulation_priors, triangulation_diag = triangulate_tracks_from_rays(tracks, shots, cfg)
            logger.info(
                "Track triangulation: %.1fs, priors=%d/%d tracks",
                time.time() - t5a,
                len(triangulation_priors),
                len(tracks),
            )

        logger.info("── Step 5b: 3D Lifting ──")
        t5 = time.time()
        objects = lift_objects(
            tracks, depth_maps, shots,
            room_width, room_length, room_height,
            walls, cfg, global_points=None,
            feature_cloud=feature_cloud,
            floor_height=floor_height,
            room_center=tuple(room_center) if room_center else (0.0, 0.0),
            axis_confidence=axis_confidence_for_lifting,
            depth_reliability=depth_reliability,
            triangulation_priors=triangulation_priors,
        )
        logger.info("Lifting: %.1fs, %d objects", time.time() - t5, len(objects))

    # Scan quality gate: large objects dropped as depth outliers.
    # Default behavior is soft-fail (warning payload) to avoid blocking
    # iterative capture; hard-fail can be re-enabled by config/env.
    quality_warnings: List[Dict] = []
    if cfg.enable_large_depth_outlier_gate:
        lift_diag = get_lift_diagnostics()
        dropped = lift_diag.get("dropped_depth_outliers", [])
        gate_categories = set(cfg.large_depth_outlier_categories or [])
        dropped_large = [
            d for d in dropped
            if str(d.get("category", "")) in gate_categories
        ]
        if dropped_large:
            cats = sorted(set(str(d.get("category", "?")) for d in dropped_large))
            detail = ", ".join(cats)
            msg = (
                "SCAN_QUALITY_GATE_FAILED: dropped depth-outlier large objects "
                f"({detail}). Rescan with wider coverage and more textured views."
            )
            if bool(getattr(cfg, "scan_quality_gate_hard_fail", False)):
                raise RuntimeError(msg)
            logger.warning("Soft quality gate warning: %s", msg)
            quality_warnings.append({
                "code": "scan_quality_gate_failed",
                "message": msg,
                "dropped_large_categories": cats,
                "dropped_large_count": int(len(dropped_large)),
            })

    # ── Step 7: Reprojection Filtering ────────────────────────
    if len(objects) > 0:
        objects = _reprojection_filter(
            objects,
            depth_maps,
            shots,
            cfg.depth_model_type,
            enable_metric_filter=cfg.enable_metric_reprojection_filter,
        )

    # ── Build Payload (via schema.py for validation) ──────────
    schema_walls = []
    for w in walls:
        schema_walls.append(Wall(
            id=w.get("id", ""),
            start=w.get("start", [0, 0]),
            end=w.get("end", [0, 0]),
            has_window=w.get("has_window", False),
            has_door=w.get("has_door", False),
        ))

    schema_objects = []
    include_debug_meta = bool(getattr(cfg, "debug_output", False))
    for o in objects:
        pos = o.get("position", {})
        meta = o.get("metadata", {})
        pos_unc = meta.get("position_uncertainty_m") if include_debug_meta else None
        ori_unc = meta.get("orientation_uncertainty_deg") if include_debug_meta else None
        support_views = meta.get("support_views") if include_debug_meta else None
        schema_objects.append(RoomObject(
            id=o.get("id", ""),
            type=o.get("type", ""),
            asset_id=o.get("asset_id", ""),
            position=ObjectPosition(
                x=pos.get("x", 0), y=pos.get("y", 0), z=pos.get("z", 0)
            ),
            rotation_y=o.get("rotation_y", 0),
            scale=o.get("scale", 1.0),
            confidence=o.get("confidence", 0),
            metadata=ObjectMetadata(
                width=meta.get("width", 1),
                length=meta.get("length", 1),
                height=meta.get("height", 1),
                position_uncertainty_m=(
                    float(pos_unc) if pos_unc is not None else None
                ),
                orientation_uncertainty_deg=(
                    float(ori_unc) if ori_unc is not None else None
                ),
                support_views=(
                    int(support_views) if support_views is not None else None
                ),
                inferred=(
                    bool(meta.get("inferred")) if meta.get("inferred") is not None else None
                ),
                suppressed_reason=(
                    str(meta.get("suppressed_reason"))
                    if meta.get("suppressed_reason") is not None
                    else None
                ),
            ),
        ))

    room_payload = RoomPayload(
        room=RoomInfo(
            id="room_0",
            width_m=round(room_width, 2),
            length_m=round(room_length, 2),
            height_m=round(room_height, 2),
            walls=schema_walls,
        ),
        objects=schema_objects,
    )
    payload = room_payload.to_dict()
    if quality_warnings:
        payload["warnings"] = quality_warnings

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — %.1fs total, %d objects", elapsed, len(objects))
    logger.info("=" * 60)

    # Debug output
    _save_debug_payload(payload)
    _save_debug_room_map(objects, walls, room_width, room_length, room_height)
    if bool(getattr(cfg, "save_reconstruction_report", True)):
        shot_summary = summarize_shot_coverage(shots)
        lift_diag = get_lift_diagnostics()
        recon_report = build_reconstruction_report(
            sfm_diag=sfm_diag,
            pose_diag=pose_diag,
            fusion_diag=fusion_diag,
            triangulation_diag=triangulation_diag,
            shot_summary=shot_summary,
            layout=layout,
            detections_count=len(detections),
            geometry_detections_count=len(geometry_detections),
            tracks_count=len(tracks),
            objects_count=len(objects),
            depth_role=depth_role,
            depth_model_type=cfg.depth_model_type,
            depth_reliability=depth_reliability,
            tracking_diag=tracking_diag,
            geometry_frames_used=geometry_keep_indices,
            geometry_frames_dropped=geometry_drop_indices,
            geometry_drop_reasons=pose_diag.get("geometry_drop_reasons", {}),
            suppressed_inferred=lift_diag.get("suppressed_inferred_objects", []),
        )
        save_reconstruction_report(recon_report)

    return payload


def _reject_pose_jump_frames(
    frames: List,
    shots: List[ShotMetadata],
    pose_diag: Dict,
    cfg: PipelineConfig,
) -> Tuple[List, List[ShotMetadata], Dict]:
    """
    Mark discontinuous pose segments for geometry-only stages.

    Detection/depth still run on all frames. This gate only filters frames used
    for geometry-sensitive stages (feature cloud, fusion, tracking, layout,
    triangulation, lifting).
    """
    pose_diag = dict(pose_diag or {})
    pose_diag["pose_jump_rejection_enabled"] = bool(
        getattr(cfg, "enable_pose_jump_rejection", True)
    )
    pose_diag["pose_jump_rejection_applied"] = False
    pose_diag["pose_jump_rejection_dropped_frames"] = []
    pose_diag["geometry_keep_indices"] = [int(i) for i in range(len(frames))]
    pose_diag["geometry_dropped_frames"] = []
    pose_diag["geometry_drop_reasons"] = {}
    pose_diag["geometry_filter_scope"] = "tracking_layout_lifting_only"

    if not pose_diag["pose_jump_rejection_enabled"]:
        pose_diag["pose_jump_rejection_reason"] = "disabled"
        return frames, shots, pose_diag
    if np is None:
        pose_diag["pose_jump_rejection_reason"] = "numpy_missing"
        return frames, shots, pose_diag

    n = min(len(frames), len(shots))
    if n < 3:
        pose_diag["pose_jump_rejection_reason"] = "insufficient_frame_count"
        return frames, shots, pose_diag

    valid_idx: List[int] = []
    valid_pos: List["np.ndarray"] = []
    valid_ts: List[Optional[float]] = []
    valid_yaw: List[Optional[float]] = []
    for i in range(n):
        shot = shots[i]
        if shot.pose is None or len(shot.pose) != 16:
            continue
        try:
            mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
        except Exception:
            continue
        valid_idx.append(i)
        valid_pos.append(mat[:3, 3])
        cam_fwd_x = -float(mat[0, 2])
        cam_fwd_z = -float(mat[2, 2])
        valid_yaw.append(float(np.arctan2(cam_fwd_x, cam_fwd_z)))
        try:
            ts = float(getattr(shot, "timestamp", 0.0))
            valid_ts.append(ts if ts > 0.0 else None)
        except Exception:
            valid_ts.append(None)

    if len(valid_pos) < 3:
        pose_diag["pose_jump_rejection_reason"] = "insufficient_pose_count"
        return frames, shots, pose_diag

    pos_arr = np.array(valid_pos, dtype=np.float64)
    steps = np.linalg.norm(pos_arr[1:] - pos_arr[:-1], axis=1)
    if len(steps) == 0:
        pose_diag["pose_jump_rejection_reason"] = "insufficient_step_count"
        return frames, shots, pose_diag

    median_step = float(np.median(steps))
    base_thr = float(
        getattr(cfg, "pose_jump_rejection_threshold_m", getattr(cfg, "pose_jump_threshold_m", 1.2))
    )
    min_ratio = max(1.0, float(getattr(cfg, "pose_jump_rejection_min_ratio", 8.0)))
    adaptive_thr = max(base_thr, median_step * min_ratio)

    max_speed = max(0.1, float(getattr(cfg, "pose_jump_rejection_max_speed_mps", 1.6)))
    hard_thr = max(base_thr, float(getattr(cfg, "pose_jump_rejection_hard_threshold_m", 3.0)))
    hybrid_step_thr = float(getattr(cfg, "pose_jump_rejection_hybrid_step_m", 0.75))
    hybrid_yaw_thr = float(getattr(cfg, "pose_jump_rejection_hybrid_yaw_deg", 95.0))
    hybrid_step_thr = max(0.1, hybrid_step_thr)
    hybrid_yaw_thr = max(1.0, hybrid_yaw_thr)

    jump_locs: List[int] = []
    jump_pairs: List[Dict] = []
    for j, step in enumerate(steps):
        dt: Optional[float] = None
        speed: Optional[float] = None
        ts0 = valid_ts[j] if j < len(valid_ts) else None
        ts1 = valid_ts[j + 1] if (j + 1) < len(valid_ts) else None
        if ts0 is not None and ts1 is not None and ts1 > ts0:
            dt = float(ts1 - ts0)
            if dt > 1e-4:
                speed = float(step / dt)

        yaw_delta_deg = 0.0
        y0 = valid_yaw[j] if j < len(valid_yaw) else None
        y1 = valid_yaw[j + 1] if (j + 1) < len(valid_yaw) else None
        if y0 is not None and y1 is not None:
            d = float(np.arctan2(np.sin(y1 - y0), np.cos(y1 - y0)))
            yaw_delta_deg = abs(d) * 180.0 / pi
        shot_to_idx = valid_idx[j + 1] if (j + 1) < len(valid_idx) else None
        if shot_to_idx is not None and 0 <= shot_to_idx < len(shots):
            cap_yaw = getattr(shots[shot_to_idx], "capture_yaw_delta_deg", None)
            if cap_yaw is not None:
                try:
                    yaw_delta_deg = max(float(yaw_delta_deg), abs(float(cap_yaw)))
                except Exception:
                    pass

        speed_gate = (speed is not None and speed > max_speed and float(step) > adaptive_thr)
        hard_gate = float(step) > hard_thr
        hybrid_gate = float(step) > hybrid_step_thr and float(yaw_delta_deg) > hybrid_yaw_thr

        if not speed_gate and not hard_gate and not hybrid_gate:
            continue

        jump_locs.append(int(j))
        jump_pairs.append({
            "from_frame": int(valid_idx[j]),
            "to_frame": int(valid_idx[j + 1]),
            "step_m": round(float(step), 4),
            "yaw_delta_deg": round(float(yaw_delta_deg), 2),
            "dt_s": round(float(dt), 3) if dt is not None else None,
            "speed_mps": round(float(speed), 3) if speed is not None else None,
            "speed_gate": bool(speed_gate),
            "hard_gate": bool(hard_gate),
            "hybrid_gate": bool(hybrid_gate),
        })

    pose_diag["pose_jump_rejection_base_threshold_m"] = round(base_thr, 4)
    pose_diag["pose_jump_rejection_median_step_m"] = round(median_step, 4)
    pose_diag["pose_jump_rejection_min_ratio"] = round(min_ratio, 4)
    pose_diag["pose_jump_rejection_threshold_m"] = round(adaptive_thr, 4)
    pose_diag["pose_jump_rejection_max_speed_mps"] = round(max_speed, 4)
    pose_diag["pose_jump_rejection_hard_threshold_m"] = round(hard_thr, 4)
    pose_diag["pose_jump_rejection_hybrid_step_m"] = round(hybrid_step_thr, 4)
    pose_diag["pose_jump_rejection_hybrid_yaw_deg"] = round(hybrid_yaw_thr, 3)
    pose_diag["pose_jump_rejection_jump_pairs"] = jump_pairs
    pose_diag["shots_total_before_jump_rejection"] = int(n)

    min_jumps = max(1, int(getattr(cfg, "pose_jump_rejection_min_jumps", 2)))
    if len(jump_locs) < min_jumps:
        pose_diag["pose_jump_rejection_reason"] = "jump_count_below_min"
        pose_diag["shots_total_after_jump_rejection"] = int(n)
        return frames, shots, pose_diag

    # Split into contiguous valid-pose segments and keep the largest.
    segments: List[List[int]] = []
    seg_start = 0
    for j in jump_locs:
        seg_end = j + 1
        segment = valid_idx[seg_start:seg_end]
        if segment:
            segments.append(segment)
        seg_start = seg_end
    tail = valid_idx[seg_start:]
    if tail:
        segments.append(tail)
    if not segments:
        pose_diag["pose_jump_rejection_reason"] = "no_valid_segments"
        pose_diag["shots_total_after_jump_rejection"] = int(n)
        return frames, shots, pose_diag

    kept_segment = max(segments, key=len)
    min_keep = max(3, int(getattr(cfg, "pose_jump_rejection_min_keep_frames", 8)))
    if len(kept_segment) < min_keep:
        pose_diag["pose_jump_rejection_reason"] = "largest_segment_too_short"
        pose_diag["shots_total_after_jump_rejection"] = int(n)
        return frames, shots, pose_diag

    keep_set = set(kept_segment)
    dropped = [i for i in range(n) if i not in keep_set]
    if not dropped:
        pose_diag["pose_jump_rejection_reason"] = "no_frames_dropped"
        pose_diag["shots_total_after_jump_rejection"] = int(n)
        return frames, shots, pose_diag

    pose_diag["pose_jump_rejection_applied"] = True
    pose_diag["pose_jump_rejection_reason"] = "largest_segment_kept"
    pose_diag["pose_jump_rejection_kept_segment"] = {
        "start_frame": int(kept_segment[0]),
        "end_frame": int(kept_segment[-1]),
        "length": int(len(kept_segment)),
    }
    pose_diag["pose_jump_rejection_dropped_frames"] = [int(i) for i in dropped]
    pose_diag["shots_total_after_jump_rejection"] = int(len(keep_set))
    pose_diag["geometry_keep_indices"] = sorted(int(i) for i in keep_set)
    pose_diag["geometry_dropped_frames"] = [int(i) for i in dropped]
    pose_diag["geometry_drop_reasons"] = {
        "pose_jump_discontinuity": int(len(dropped))
    }

    logger.warning(
        "Pose-jump geometry filter applied: kept %d/%d frames (segment %d..%d), dropped=%s, threshold=%.2fm",
        len(keep_set),
        n,
        kept_segment[0],
        kept_segment[-1],
        dropped,
        adaptive_thr,
    )

    return frames, shots, pose_diag


def _run_open_vocab(frames, cfg: PipelineConfig) -> List[Detection]:
    """Run YOLO-World open-vocabulary detector if model is available."""
    try:
        global _cached_world_detector, _cached_world_detector_path

        model_path = cfg.model_root / "detector" / "yolo_world_v2_s.onnx"
        if not model_path.exists():
            # Try alternative names
            for name in ["yolov8s-worldv2.onnx", "yolo_world_s.onnx"]:
                alt = cfg.model_root / "detector" / name
                if alt.exists():
                    model_path = alt
                    break

        with _cache_lock:
            needs_reload = (
                _cached_world_detector is None
                or _cached_world_detector_path != model_path
                or (
                    _cached_world_detector is not None
                    and not _cached_world_detector.available
                    and model_path.exists()
                )
            )
            if needs_reload:
                _cached_world_detector = YoloWorldOnnxDetector(model_path)
                _cached_world_detector_path = model_path
            detector = _cached_world_detector

        if detector is None:
            return []
        if not detector.available:
            return []

        return detector.run(frames, cfg)
    except Exception as e:
        logger.debug("YOLO-World not available: %s", e)
        return []


def _build_global_point_cloud(
    depth_maps: List[Optional["np.ndarray"]],
    shots: List[ShotMetadata],
) -> Optional["np.ndarray"]:
    """Build a global 3D point cloud from all depth maps."""
    if np is None:
        return None

    all_points = []
    for idx, depth in enumerate(depth_maps):
        if depth is None:
            continue

        intr = None
        if idx < len(shots) and shots[idx].intrinsics is not None:
            intr = shots[idx].intrinsics
        if intr is None:
            h, w = depth.shape[:2]
            intr = CameraIntrinsics(
                fx=w * 0.8, fy=w * 0.8,
                cx=w / 2, cy=h / 2,
                width=w, height=h,
            )

        h, w = depth.shape[:2]
        intr = intr.scaled_to(w, h)

        # Subsample heavily for memory
        step = 16
        ys = np.arange(0, h, step)
        xs = np.arange(0, w, step)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        d = depth[yy, xx].astype(np.float64)
        valid = np.isfinite(d) & (d > 0.1) & (d < 20.0)
        if np.sum(valid) < 5:
            continue

        xx_v = xx[valid].astype(np.float64)
        yy_v = yy[valid].astype(np.float64)
        d_v = d[valid]

        px = (xx_v - intr.cx) * d_v / intr.fx
        py = (yy_v - intr.cy) * d_v / intr.fy
        pz = d_v

        pts = np.stack([px, py, pz], axis=1)

        # Transform to world coordinates
        if idx < len(shots) and shots[idx].pose is not None:
            pose = shots[idx].pose
            if pose is not None and len(pose) == 16:
                mat = np.array(pose, dtype=np.float64).reshape(4, 4)
                ones = np.ones((pts.shape[0], 1), dtype=np.float64)
                pts_h = np.hstack([pts, ones])
                pts = (mat @ pts_h.T).T[:, :3]

        all_points.append(pts)

    if not all_points:
        return None

    cloud = np.vstack(all_points)
    logger.info("Global point cloud: %d points", len(cloud))
    return cloud


# Global dict to hold scale recovery diagnostics for the cluster report
_scale_diagnostics: Dict = {}


def get_scale_diagnostics() -> Dict:
    """Return scale recovery diagnostics for the cluster report."""
    return dict(_scale_diagnostics)


def _metric_depth_sanity(
    depth_maps: List[Optional["np.ndarray"]],
    shots: List[ShotMetadata],
) -> Dict:
    """
    Evaluate whether metric depth agrees with ARKit metric points.

    This does not rescale depth. It only measures consistency and returns a
    reliability score used to soften depth-based filtering when needed.
    """
    ratios: List[float] = []
    total_fp = 0
    uv_points = 0
    legacy_points = 0
    per_shot_usage: List[Dict] = []

    for idx, shot in enumerate(shots):
        shot_entry = {
            "shot_idx": idx,
            "scale_used_uv": False,
            "total_feature_points": int(len(shot.feature_points_w)) if shot.feature_points_w else 0,
            "accepted_ratios": 0,
            "projection_uv_points": 0,
            "projection_legacy_points": 0,
            "scale_fallback_reason": None,
        }
        if shot.feature_points_w is None or not shot.feature_points_w:
            shot_entry["scale_fallback_reason"] = "no_feature_points"
            per_shot_usage.append(shot_entry)
            continue
        if shot.pose is None:
            shot_entry["scale_fallback_reason"] = "no_pose"
            per_shot_usage.append(shot_entry)
            continue
        if idx >= len(depth_maps) or depth_maps[idx] is None:
            shot_entry["scale_fallback_reason"] = "no_depth_map"
            per_shot_usage.append(shot_entry)
            continue
        if shot.intrinsics is None:
            shot_entry["scale_fallback_reason"] = "no_intrinsics"
            per_shot_usage.append(shot_entry)
            continue

        depth = depth_maps[idx]
        h, w = depth.shape[:2]
        intr_s = shot.intrinsics.scaled_to(w, h)
        try:
            mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
            mat_inv = np.linalg.inv(mat)
        except (ValueError, np.linalg.LinAlgError):
            shot_entry["scale_fallback_reason"] = "invalid_pose_matrix"
            per_shot_usage.append(shot_entry)
            continue

        use_client_uv = (
            shot.feature_points_uv is not None
            and len(shot.feature_points_uv) == len(shot.feature_points_w)
        )
        uv_arr = None
        if use_client_uv:
            uv_arr = np.array(shot.feature_points_uv, dtype=np.float64)
            if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
                use_client_uv = False
                uv_arr = None
        shot_entry["scale_used_uv"] = bool(use_client_uv)
        if not use_client_uv:
            shot_entry["scale_fallback_reason"] = "missing_or_invalid_uv"

        for fp_i, fp in enumerate(shot.feature_points_w):
            if len(fp) != 3:
                continue
            total_fp += 1

            pt_w = np.array([fp[0], fp[1], fp[2], 1.0], dtype=np.float64)
            pt_c = mat_inv @ pt_w
            real_d = -float(pt_c[2])
            if real_d <= 0.2 or real_d > 15.0:
                continue

            if use_client_uv and uv_arr is not None:
                u = float(uv_arr[fp_i, 0])
                v = float(uv_arr[fp_i, 1])
                if not np.isfinite(u) or not np.isfinite(v):
                    continue
                ui, vi = int(round(u)), int(round(v))
                uv_points += 1
                shot_entry["projection_uv_points"] += 1
            else:
                cam_x = float(pt_c[0])
                cam_y = float(pt_c[1])
                u = intr_s.fx * cam_y / real_d + intr_s.cx
                v = intr_s.fy * (-cam_x) / real_d + intr_s.cy
                ui, vi = int(round(u)), int(round(v))
                legacy_points += 1
                shot_entry["projection_legacy_points"] += 1

            if ui < 2 or ui >= w - 2 or vi < 2 or vi >= h - 2:
                continue
            patch = depth[vi - 1:vi + 2, ui - 1:ui + 2]
            valid = np.isfinite(patch) & (patch > 0.001)
            if np.sum(valid) < 1:
                continue
            model_d = float(np.median(patch[valid]))
            if model_d <= 0.001:
                continue

            ratios.append(real_d / model_d)
            shot_entry["accepted_ratios"] += 1

        if shot_entry["accepted_ratios"] == 0 and shot_entry["scale_fallback_reason"] is None:
            shot_entry["scale_fallback_reason"] = "no_valid_projection_or_depth_samples"
        per_shot_usage.append(shot_entry)

    if not ratios:
        return {
            "scale_method": "metric_direct",
            "scale_factor": 1.0,
            "scale_num_ratios": 0,
            "scale_ratio_p10": None,
            "scale_ratio_p50": None,
            "scale_ratio_p90": None,
            "scale_inlier_depth_range": None,
            "scale_projection_uv_points": uv_points,
            "scale_projection_legacy_points": legacy_points,
            "scale_total_fp": total_fp,
            "scale_per_shot": per_shot_usage,
            "scale_fallback_reason": "metric_no_valid_fp_ratio",
            "metric_depth_reliability": 0.45,
        }

    arr = np.array(ratios, dtype=np.float64)
    p10 = float(np.percentile(arr, 10))
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    spread = p90 - p10

    center_err = abs(p50 - 1.0)
    reliability = 1.0
    reliability -= min(1.0, center_err / 0.45) * 0.7
    reliability -= min(1.0, spread / 1.25) * 0.3
    reliability = max(0.0, min(1.0, reliability))

    fallback_reason = None if reliability >= 0.45 else "metric_depth_fp_mismatch"
    return {
        "scale_method": "metric_direct",
        "scale_factor": 1.0,
        "scale_num_ratios": len(ratios),
        "scale_ratio_p10": round(p10, 4),
        "scale_ratio_p50": round(p50, 4),
        "scale_ratio_p90": round(p90, 4),
        "scale_ratio_median": round(float(np.median(arr)), 4),
        "scale_ratio_iqr": round(iqr, 4),
        "scale_ratio_q1": round(q1, 4),
        "scale_ratio_q3": round(q3, 4),
        "scale_inlier_depth_range": None,
        "scale_projection_uv_points": uv_points,
        "scale_projection_legacy_points": legacy_points,
        "scale_total_fp": total_fp,
        "scale_per_shot": per_shot_usage,
        "scale_fallback_reason": fallback_reason,
        "metric_depth_reliability": round(reliability, 3),
    }


def _apply_metric_fp_scale_calibration(
    depth_maps: List[Optional["np.ndarray"]],
    metric_diag: Dict,
    cfg: PipelineConfig,
) -> Tuple[List[Optional["np.ndarray"]], Dict]:
    """
    Apply optional feature-point calibration to metric depth maps.

    For metric models, ratio = real_depth / model_depth should be near 1.0.
    If the median ratio is consistently biased, we correct model depth as:
        depth_calibrated = depth_model * ratio_median
    """
    diag = dict(metric_diag or {})
    diag["metric_scale_applied"] = False
    diag["metric_scale_factor"] = 1.0
    diag["metric_scale_reason"] = None
    diag["metric_scale_maps_updated"] = 0

    if np is None:
        diag["metric_scale_reason"] = "numpy_missing"
        return depth_maps, diag
    if not bool(getattr(cfg, "enable_metric_fp_scale_calibration", True)):
        diag["metric_scale_reason"] = "disabled"
        return depth_maps, diag

    n_ratios = int(diag.get("scale_num_ratios") or 0)
    p10 = diag.get("scale_ratio_p10")
    p50 = diag.get("scale_ratio_p50")
    p90 = diag.get("scale_ratio_p90")
    iqr = diag.get("scale_ratio_iqr")

    if p50 is None:
        diag["metric_scale_reason"] = "missing_ratio_median"
        return depth_maps, diag

    min_ratios = max(20, int(getattr(cfg, "metric_fp_scale_min_ratios", 120)))
    min_factor = float(getattr(cfg, "metric_fp_scale_min_factor", 0.45))
    max_factor = float(getattr(cfg, "metric_fp_scale_max_factor", 1.85))
    max_iqr = float(getattr(cfg, "metric_fp_scale_max_iqr", 0.35))
    max_spread = float(getattr(cfg, "metric_fp_scale_max_spread", 1.0))

    factor = float(p50)
    if n_ratios < min_ratios:
        diag["metric_scale_reason"] = f"insufficient_ratios(n={n_ratios},min={min_ratios})"
        return depth_maps, diag
    if not np.isfinite(factor) or factor <= 0.0:
        diag["metric_scale_reason"] = "invalid_scale_factor"
        return depth_maps, diag
    if factor < min_factor or factor > max_factor:
        diag["metric_scale_reason"] = (
            f"scale_factor_out_of_range({factor:.3f} not in [{min_factor:.2f},{max_factor:.2f}])"
        )
        return depth_maps, diag
    if iqr is not None and float(iqr) > max_iqr:
        diag["metric_scale_reason"] = f"ratio_iqr_too_high({float(iqr):.3f}>{max_iqr:.3f})"
        return depth_maps, diag
    if p10 is not None and p90 is not None:
        spread = float(p90) - float(p10)
        if spread > max_spread:
            diag["metric_scale_reason"] = f"ratio_spread_too_high({spread:.3f}>{max_spread:.3f})"
            return depth_maps, diag

    scaled = 0
    calibrated_maps: List[Optional["np.ndarray"]] = []
    for d in depth_maps:
        if d is None:
            calibrated_maps.append(None)
            continue
        calibrated_maps.append(d * factor)
        scaled += 1

    # Recompute post-calibration reliability in the same style as _metric_depth_sanity.
    calibrated_rel = diag.get("metric_depth_reliability")
    if p10 is not None and p90 is not None and p50 is not None:
        p10_c = float(p10) / max(1e-6, factor)
        p50_c = float(p50) / max(1e-6, factor)
        p90_c = float(p90) / max(1e-6, factor)
        center_err = abs(p50_c - 1.0)
        spread_c = max(0.0, p90_c - p10_c)
        rel_c = 1.0
        rel_c -= min(1.0, center_err / 0.45) * 0.7
        rel_c -= min(1.0, spread_c / 1.25) * 0.3
        calibrated_rel = round(max(0.0, min(1.0, rel_c)), 3)
        diag["scale_ratio_p10_calibrated"] = round(p10_c, 4)
        diag["scale_ratio_p50_calibrated"] = round(p50_c, 4)
        diag["scale_ratio_p90_calibrated"] = round(p90_c, 4)

    diag["metric_depth_reliability_raw"] = diag.get("metric_depth_reliability")
    diag["metric_depth_reliability"] = calibrated_rel
    diag["metric_depth_reliability_calibrated"] = calibrated_rel
    diag["metric_scale_applied"] = True
    diag["metric_scale_factor"] = round(factor, 4)
    diag["metric_scale_reason"] = "fp_ratio_median"
    diag["metric_scale_maps_updated"] = int(scaled)
    diag["scale_method"] = "metric_fp_calibrated"
    diag["scale_factor"] = round(factor, 4)

    logger.info(
        "Metric FP scale calibration: applied factor=%.4f using %d ratios (%d maps updated)",
        factor,
        n_ratios,
        scaled,
    )
    return calibrated_maps, diag


def _scale_depth_from_feature_points(
    depth_maps: List[Optional["np.ndarray"]],
    shots: List[ShotMetadata],
) -> Optional[float]:
    """
    Compute depth scale factor using ARKit feature points as ground truth.

    ARKit provides world-space feature points (real meters) via VIO/SLAM.
    For each visible feature point:
      1. Transform world point → camera space using inverse pose
      2. Project to pixel coordinates using intrinsics
      3. Read the model's depth at that pixel
      4. Compute ratio = real_depth / model_depth
    The median of all ratios is the depth scale factor.

    CRITICAL: ARKit camera convention is -Z forward. When transforming
    world → camera via pose_inv, visible points get negative Z.
    Our depth pipeline uses positive depth (cam_z = +depth) so we must
    negate: real_d = -pt_c[2].

    Returns scale factor, or None if not enough data.
    """
    global _scale_diagnostics
    _scale_diagnostics = {}

    ratios: List[float] = []
    real_depths_used: List[float] = []
    model_depths_used: List[float] = []
    total_fp = 0
    rejected_behind = 0
    rejected_outside = 0
    rejected_invalid_depth = 0
    rejected_missing_projection = 0
    accepted = 0
    uv_projection_points = 0
    legacy_projection_points = 0
    per_shot_usage: List[Dict] = []

    for idx, shot in enumerate(shots):
        shot_entry = {
            "shot_idx": idx,
            "scale_used_uv": False,
            "total_feature_points": int(len(shot.feature_points_w)) if shot.feature_points_w else 0,
            "accepted_ratios": 0,
            "projection_uv_points": 0,
            "projection_legacy_points": 0,
            "scale_fallback_reason": None,
        }
        if shot.feature_points_w is None or not shot.feature_points_w:
            shot_entry["scale_fallback_reason"] = "no_feature_points"
            per_shot_usage.append(shot_entry)
            continue
        if shot.pose is None:
            shot_entry["scale_fallback_reason"] = "no_pose"
            per_shot_usage.append(shot_entry)
            continue
        if idx >= len(depth_maps) or depth_maps[idx] is None:
            shot_entry["scale_fallback_reason"] = "no_depth_map"
            per_shot_usage.append(shot_entry)
            continue

        depth = depth_maps[idx]
        h, w = depth.shape[:2]

        # Camera intrinsics
        intr = shot.intrinsics
        if intr is None:
            shot_entry["scale_fallback_reason"] = "no_intrinsics"
            per_shot_usage.append(shot_entry)
            continue
        intr_s = intr.scaled_to(w, h)

        # Pose: camera-to-world. Inverse = world-to-camera.
        try:
            mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
            mat_inv = np.linalg.inv(mat)
        except (ValueError, np.linalg.LinAlgError):
            shot_entry["scale_fallback_reason"] = "invalid_pose_matrix"
            per_shot_usage.append(shot_entry)
            continue

        # If client UVs are present for this shot, use UV-only mapping.
        # Do NOT mix in legacy reprojection for this shot; that would reintroduce
        # projection-convention mismatch.
        use_client_uv = (
            shot.feature_points_uv is not None
            and len(shot.feature_points_uv) == len(shot.feature_points_w)
        )
        uv_arr = None
        if use_client_uv:
            uv_arr = np.array(shot.feature_points_uv, dtype=np.float64)
            if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
                use_client_uv = False
                uv_arr = None
        shot_entry["scale_used_uv"] = bool(use_client_uv)
        if not use_client_uv:
            shot_entry["scale_fallback_reason"] = "missing_or_invalid_uv"

        for fp_i, fp in enumerate(shot.feature_points_w):
            if len(fp) != 3:
                continue
            total_fp += 1

            # World point → camera space
            pt_w = np.array([fp[0], fp[1], fp[2], 1.0])
            pt_c = mat_inv @ pt_w

            # ARKit camera: -Z forward. Visible points have pt_c[2] < 0.
            # Our depth convention: positive depth forward.
            # So real_d = -pt_c[2] for visible points.
            real_d = -float(pt_c[2])
            if real_d <= 0.2 or real_d > 15.0:
                rejected_behind += 1
                continue  # Behind camera or too far

            ui = -1
            vi = -1
            if use_client_uv and uv_arr is not None:
                u = float(uv_arr[fp_i, 0])
                v = float(uv_arr[fp_i, 1])
                if not np.isfinite(u) or not np.isfinite(v):
                    rejected_outside += 1
                    continue
                ui, vi = int(round(u)), int(round(v))
                uv_projection_points += 1
                shot_entry["projection_uv_points"] += 1
            else:
                # Legacy reprojection fallback when client UVs are not available.
                cam_x = float(pt_c[0])
                cam_y = float(pt_c[1])
                u = intr_s.fx * cam_y / real_d + intr_s.cx
                v = intr_s.fy * (-cam_x) / real_d + intr_s.cy
                ui, vi = int(round(u)), int(round(v))
                legacy_projection_points += 1
                shot_entry["projection_legacy_points"] += 1

            if ui < 2 or ui >= w - 2 or vi < 2 or vi >= h - 2:
                rejected_outside += 1
                continue  # Outside image

            # Read model depth (average small patch for robustness)
            patch = depth[vi - 1:vi + 2, ui - 1:ui + 2]
            valid = np.isfinite(patch) & (patch > 0.001)
            if np.sum(valid) < 1:
                rejected_invalid_depth += 1
                continue

            model_d = float(np.median(patch[valid]))
            if model_d < 0.001:
                rejected_invalid_depth += 1
                continue

            ratios.append(real_d / model_d)
            real_depths_used.append(real_d)
            model_depths_used.append(model_d)
            accepted += 1
            shot_entry["accepted_ratios"] += 1

        if shot_entry["accepted_ratios"] == 0 and shot_entry["scale_fallback_reason"] is None:
            shot_entry["scale_fallback_reason"] = "no_valid_projection_or_depth_samples"
        per_shot_usage.append(shot_entry)

    if uv_projection_points == 0 and legacy_projection_points == 0:
        rejected_missing_projection = total_fp

    logger.info(
        "Feature-point scale: total_fp=%d accepted=%d "
        "projection(uv=%d legacy=%d) rejected(behind=%d outside=%d invalid_depth=%d missing_projection=%d)",
        total_fp,
        accepted,
        uv_projection_points,
        legacy_projection_points,
        rejected_behind,
        rejected_outside,
        rejected_invalid_depth,
        rejected_missing_projection,
    )

    if len(ratios) < 10:
        logger.info("Feature-point depth scale: only %d ratios (need 10+), skipping",
                     len(ratios))
        _scale_diagnostics = {
            "scale_method": "none",
            "scale_factor": None,
            "scale_num_ratios": len(ratios),
            "scale_ratio_p10": None,
            "scale_ratio_p50": None,
            "scale_ratio_p90": None,
            "scale_inlier_depth_range": None,
            "scale_total_fp": total_fp,
            "scale_projection_uv_points": uv_projection_points,
            "scale_projection_legacy_points": legacy_projection_points,
            "scale_rejected_behind": rejected_behind,
            "scale_rejected_outside": rejected_outside,
            "scale_rejected_invalid_depth": rejected_invalid_depth,
            "scale_rejected_missing_projection": rejected_missing_projection,
            "scale_per_shot": per_shot_usage,
            "scale_fallback_reason": f"insufficient_ratios(n={len(ratios)})",
        }
        return None

    # Outlier rejection: use IQR to remove extreme ratios
    arr = np.array(ratios)
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q3 - q1
    mask = (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)
    if np.sum(mask) < 5:
        scale = float(np.median(arr))
        n_inliers = len(ratios)
        inlier_mask = np.ones_like(arr, dtype=bool)
    else:
        scale = float(np.median(arr[mask]))
        n_inliers = int(np.sum(mask))
        inlier_mask = mask

    scale = max(0.05, min(50.0, scale))

    real_arr = np.array(real_depths_used)
    model_arr = np.array(model_depths_used)
    inlier_real = real_arr[inlier_mask]
    if len(inlier_real) == 0:
        inlier_real = real_arr

    p10 = float(np.percentile(arr, 10))
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    inlier_depth_range = [
        round(float(np.min(inlier_real)), 4),
        round(float(np.max(inlier_real)), 4),
    ] if len(inlier_real) > 0 else None

    _scale_diagnostics = {
        "scale_method": "feature_points",
        "scale_factor": round(scale, 4),
        "scale_num_ratios": len(ratios),
        "scale_num_inliers": n_inliers,
        "scale_ratio_p10": round(p10, 4),
        "scale_ratio_p50": round(p50, 4),
        "scale_ratio_p90": round(p90, 4),
        "scale_ratio_median": round(float(np.median(arr)), 4),
        "scale_ratio_iqr": round(iqr, 4),
        "scale_ratio_q1": round(q1, 4),
        "scale_ratio_q3": round(q3, 4),
        "scale_inlier_depth_range": inlier_depth_range,
        "scale_feature_depth_median": round(float(np.median(real_arr)), 4),
        "scale_model_depth_median": round(float(np.median(model_arr)), 4),
        "scale_total_fp": total_fp,
        "scale_projection_uv_points": uv_projection_points,
        "scale_projection_legacy_points": legacy_projection_points,
        "scale_rejected_behind": rejected_behind,
        "scale_rejected_outside": rejected_outside,
        "scale_rejected_invalid_depth": rejected_invalid_depth,
        "scale_rejected_missing_projection": rejected_missing_projection,
        "scale_per_shot": per_shot_usage,
        "scale_fallback_reason": None,
    }

    logger.info("Feature-point depth scale: %.3f (from %d/%d ratios, IQR=[%.2f, %.2f], "
                "real_depth_median=%.3f, model_depth_median=%.4f)",
                scale, n_inliers, len(ratios), q1, q3,
                float(np.median(real_arr)), float(np.median(model_arr)))
    return scale


def _scale_relative_depths(
    depth_maps: List[Optional["np.ndarray"]],
    shots: List[ShotMetadata],
) -> List[Optional["np.ndarray"]]:
    """
    Scale relative depth maps to real-world meters.

    Strategy 1 (preferred): Use ARKit feature points as ground-truth depth anchors.
    Each feature point is in real-world meters from VIO/SLAM. Project them into
    the camera, compare with model depth, and compute the scale.

    Strategy 2 (fallback): Use camera baseline spread as a proxy for scene scale.
    This fails when the user barely translates (guided capture = rotate in place).
    """
    if np is None:
        return depth_maps

    global _scale_diagnostics

    # ── Strategy 1: Feature-point depth calibration ──
    fp_scale = _scale_depth_from_feature_points(depth_maps, shots)
    if fp_scale is not None:
        # Validate: need enough inliers. IQR filtering already happened
        # inside _scale_depth_from_feature_points (1.5×IQR outlier rejection).
        # The returned scale is the median of inliers, which is robust.
        # Only reject if we have very few ratios (unreliable).
        diag = get_scale_diagnostics()
        n_inliers = diag.get("scale_num_inliers", 0)
        n_ratios = diag.get("scale_num_ratios", 0)
        if n_inliers >= 20:
            logger.info("Feature-point scale accepted: %.3f (n_ratios=%d, n_inliers=%d)",
                         fp_scale, n_ratios, n_inliers)
            scaled = []
            for dm in depth_maps:
                scaled.append(dm * fp_scale if dm is not None else None)
            return scaled
        else:
            logger.warning("Feature-point scale REJECTED: only %d inliers (need 20+)",
                           n_inliers)
            _scale_diagnostics["scale_method"] = "feature_points_rejected"
            _scale_diagnostics["scale_fallback_reason"] = (
                f"feature_points_inliers_too_low(n={n_inliers},required=20)"
            )

    # ── Strategy 2: Camera baseline fallback ──
    logger.info("Feature-point scaling unavailable or rejected, falling back to camera baseline")

    cam_data = []
    for idx, s in enumerate(shots):
        if s.pose is not None and len(s.pose) == 16:
            mat = np.array(s.pose, dtype=np.float64).reshape(4, 4)
            cam_data.append((mat[:3, 3], idx))

    if len(cam_data) < 2:
        logger.info("Cannot estimate depth scale (need >= 2 poses), using raw values")
        _scale_diagnostics["scale_fallback_reason"] = "camera_baseline_insufficient_poses"
        return depth_maps

    positions = np.array([c[0] for c in cam_data])
    real_dists = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d > 0.05:
                real_dists.append(d)

    if not real_dists:
        logger.info("All cameras at same position, using raw depth values")
        _scale_diagnostics["scale_fallback_reason"] = "camera_baseline_no_spread"
        return depth_maps

    max_real_spread = float(np.max(real_dists))
    median_real_dist = float(np.median(real_dists))
    logger.info("Camera spread: median=%.3fm, max=%.3fm (%d pairs)",
                median_real_dist, max_real_spread, len(real_dists))

    # For guided capture (rotate in place), camera spread is tiny (0.1–0.3m).
    # Use max spread instead of median, and assume the user is ~1.5m from walls
    # (typical for standing in a room). Scale so center depth ≈ max_spread * 3
    # (user diameter ≈ 1/3 of room).
    center_depths = []
    for depth in depth_maps:
        if depth is not None:
            h, w = depth.shape[:2]
            ch, cw = h // 2, w // 2
            rh, rw = max(1, h // 10), max(1, w // 10)
            center_region = depth[ch - rh:ch + rh, cw - rw:cw + rw]
            valid = np.isfinite(center_region) & (center_region > 0)
            if np.sum(valid) > 5:
                center_depths.append(float(np.median(center_region[valid])))

    if not center_depths:
        logger.info("No valid center depths, using raw values")
        _scale_diagnostics["scale_fallback_reason"] = "camera_baseline_no_center_depths"
        return depth_maps

    median_center_depth = float(np.median(center_depths))

    if median_center_depth > 0.01:
        # If cameras barely moved, boost the scale estimate.
        # Typical room: user stands ~1.5m from walls, median depth to wall ~2m.
        # Camera spread during rotation ≈ 0.1–0.3m.
        # Use max spread * 5 as estimated typical wall distance (heuristic).
        if max_real_spread < 0.5:
            # Guided capture: cameras barely translate
            estimated_wall_dist = max(1.5, max_real_spread * 5.0)
            scale = estimated_wall_dist / median_center_depth
            logger.info("Depth scale (guided heuristic): %.3f (est_wall=%.3fm, depth_median=%.3f)",
                        scale, estimated_wall_dist, median_center_depth)
        else:
            scale = median_real_dist / median_center_depth
            logger.info("Depth scale (baseline): %.3f (real_median=%.3fm, depth_median=%.3f)",
                        scale, median_real_dist, median_center_depth)

        scale = max(0.1, min(20.0, scale))

        # Record camera-baseline diagnostics
        if "scale_method" not in _scale_diagnostics or _scale_diagnostics.get("scale_method") in (
            "none", "feature_points_rejected"
        ):
            _scale_diagnostics.update({
                "scale_method": "camera_baseline",
                "scale_factor": round(scale, 4),
                "scale_camera_spread_max": round(max_real_spread, 4),
                "scale_camera_spread_median": round(median_real_dist, 4),
                "scale_center_depth_median": round(median_center_depth, 4),
                "scale_fallback_reason": _scale_diagnostics.get("scale_fallback_reason"),
            })

        scaled = []
        for dm in depth_maps:
            scaled.append(dm * scale if dm is not None else None)
        return scaled

    _scale_diagnostics["scale_fallback_reason"] = "camera_baseline_invalid_center_depth"
    return depth_maps


def _reprojection_filter(
    objects: List[Dict],
    depth_maps: List[Optional["np.ndarray"]],
    shots: List[ShotMetadata],
    depth_model_type: str = "dav2",
    enable_metric_filter: bool = True,
) -> List[Dict]:
    """
    Filter objects using reprojection consistency.

    Session 8 fix: DISABLED for relative depth models.
    With relative depth (non-metric), the scale factor is estimated from camera
    baselines and is often wrong. This causes reprojection to reject valid objects
    (e.g., keeping only 1/5). Clustering + MAX_INSTANCES handle duplicates instead.

    Only enabled for metric depth models where depth values are in true meters.
    """
    # Session 8: Skip for relative depth — scale is too unreliable
    if depth_model_type != "dav2_metric":
        logger.info("Reprojection filter: SKIPPED (relative depth model, scale unreliable)")
        return objects
    if not enable_metric_filter:
        logger.info("Reprojection filter: SKIPPED (metric filter disabled by config)")
        return objects

    if np is None:
        return objects

    if not shots or not depth_maps:
        return objects

    filtered = []
    skipped_fp = 0
    for obj in objects:
        if obj.get("_position_source") == "feature_points":
            filtered.append(obj)
            skipped_fp += 1
            continue

        votes_agree = 0
        votes_disagree = 0
        ox = obj["position"]["x"]
        oy = obj["position"]["y"]
        oz = obj["position"]["z"]

        for frame_idx, depth in enumerate(depth_maps):
            if depth is None or frame_idx >= len(shots):
                continue

            intr = shots[frame_idx].intrinsics
            pose = shots[frame_idx].pose

            if intr is None or pose is None or len(pose) != 16:
                continue

            mat = np.array(pose, dtype=np.float64).reshape(4, 4)
            img_h, img_w = depth.shape[:2]
            intr_s = intr.scaled_to(img_w, img_h)

            # Project object position into this camera.
            # Keep the same ARKit + portrait convention used by
            # feature-point projection in this module:
            #   - ARKit visible points have negative camera Z
            #   - depth in our pipeline is positive forward depth (-Z)
            #   - portrait mapping is cam_y -> u and -cam_x -> v
            try:
                mat_inv = np.linalg.inv(mat)
            except np.linalg.LinAlgError:
                continue

            pt_world = np.array([ox, oy, oz, 1.0])
            pt_cam = mat_inv @ pt_world

            cam_x = float(pt_cam[0])
            cam_y = float(pt_cam[1])
            projected_depth = -float(pt_cam[2])

            # Behind camera? Skip (non-visible = keep)
            if projected_depth <= 0.1:
                continue

            u = intr_s.fx * cam_y / projected_depth + intr_s.cx
            v = intr_s.fy * (-cam_x) / projected_depth + intr_s.cy

            # Outside image? Non-visible = keep (Session 6: don't reject)
            if u < 0 or u >= img_w or v < 0 or v >= img_h:
                continue

            # Compare projected depth with actual depth at that pixel
            actual_depth = float(depth[int(v), int(u)])

            if not np.isfinite(actual_depth) or actual_depth <= 0:
                continue

            abs_diff = abs(actual_depth - projected_depth)
            relative_diff = abs_diff / max(actual_depth, projected_depth)

            # Session 6 agreement: 0.5m absolute OR 30% relative
            if abs_diff <= 0.5 or relative_diff <= 0.30:
                votes_agree += 1
            else:
                votes_disagree += 1

        # Keep object if majority agrees or if too few votes
        total_votes = votes_agree + votes_disagree
        if total_votes < 2 or votes_agree >= votes_disagree:
            filtered.append(obj)
        else:
            logger.debug("Reprojection filtered: %s at (%.2f, %.2f, %.2f) — %d/%d agree",
                         obj.get("type", "?"), ox, oy, oz, votes_agree, total_votes)

    n_removed = len(objects) - len(filtered)
    if skipped_fp > 0:
        logger.info("Reprojection filter: skipped %d FP-anchored objects", skipped_fp)
    if n_removed > 0:
        logger.info("Reprojection filter: kept %d/%d objects (removed %d)",
                     len(filtered), len(objects), n_removed)
    return filtered


def _empty_payload() -> Dict:
    """Return an empty room payload."""
    return {
        "room": {
            "id": "room_0",
            "width_m": 4.0,
            "length_m": 4.0,
            "height_m": 2.5,
            "walls": [
                {"id": "wall_0", "start": [-2, -2], "end": [2, -2], "has_window": False, "has_door": True},
                {"id": "wall_1", "start": [2, -2], "end": [2, 2], "has_window": False, "has_door": False},
                {"id": "wall_2", "start": [2, 2], "end": [-2, 2], "has_window": False, "has_door": False},
                {"id": "wall_3", "start": [-2, 2], "end": [-2, -2], "has_window": False, "has_door": False},
            ],
        },
        "objects": [],
    }


def _save_debug_payload(payload: Dict) -> None:
    """Write full payload to /tmp for debugging."""
    try:
        with open("/tmp/decroom_payload.json", "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Debug payload saved to /tmp/decroom_payload.json")
    except Exception:
        pass


def _save_debug_room_map(
    objects: List[Dict],
    walls: List[Dict],
    room_width: float,
    room_length: float,
    room_height: float,
) -> None:
    """
    Save a top-down debug visualization of the room: walls + final objects.
    Output: /tmp/decroom_room_map.png

    This shows EXACTLY what the iOS app will render — only the objects
    that survived tracking, clustering, and filtering.
    """
    try:
        import cv2

        # Image size and margin
        img_size = 800
        margin = 60

        # Find coordinate bounds from walls and objects
        all_x = []
        all_z = []
        for w in walls:
            s = w.get("start", [0, 0])
            e = w.get("end", [0, 0])
            all_x.extend([s[0], e[0]])
            all_z.extend([s[1], e[1]])
        for o in objects:
            pos = o.get("position", {})
            all_x.append(pos.get("x", 0))
            all_z.append(pos.get("z", 0))

        if not all_x or not all_z:
            return

        min_x, max_x = min(all_x), max(all_x)
        min_z, max_z = min(all_z), max(all_z)

        # Add padding
        range_x = max(max_x - min_x, 0.5)
        range_z = max(max_z - min_z, 0.5)
        pad = max(range_x, range_z) * 0.1
        min_x -= pad
        max_x += pad
        min_z -= pad
        max_z += pad
        range_x = max_x - min_x
        range_z = max_z - min_z

        draw_w = img_size - 2 * margin
        draw_h = img_size - 2 * margin
        scale = min(draw_w / range_x, draw_h / range_z)

        def to_px(x: float, z: float) -> Tuple[int, int]:
            px = int(margin + (x - min_x) * scale)
            py = int(margin + (z - min_z) * scale)
            return (px, py)

        # White background
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

        # Draw walls as thick dark lines
        for w in walls:
            s = w.get("start", [0, 0])
            e = w.get("end", [0, 0])
            p1 = to_px(s[0], s[1])
            p2 = to_px(e[0], e[1])
            cv2.line(img, p1, p2, (40, 40, 40), 3)

        # Category colors (BGR for OpenCV)
        cat_colors = {
            "bed":        (80, 80, 255),
            "sofa":       (60, 160, 255),
            "desk":       (255, 180, 60),
            "chair":      (80, 255, 80),
            "table":      (60, 255, 180),
            "dresser":    (255, 100, 200),
            "nightstand": (100, 200, 255),
            "lamp_floor": (80, 255, 255),
            "lamp_desk":  (120, 220, 255),
            "plant":      (60, 200, 60),
            "bookshelf":  (255, 140, 100),
            "monitor":    (255, 255, 0),
            "rug":        (100, 150, 200),
            "poster":     (200, 100, 255),
        }

        for o in objects:
            pos = o.get("position", {})
            ox = pos.get("x", 0)
            oz = pos.get("z", 0)
            cat = o.get("category", o.get("type", "?"))
            meta = o.get("metadata", {})
            w_m = meta.get("width", 0.5)
            l_m = meta.get("length", 0.5)

            color = cat_colors.get(cat, (128, 128, 128))
            center = to_px(ox, oz)

            # Draw object as a filled rectangle (approximate footprint)
            half_w = int(w_m * scale / 2)
            half_l = int(l_m * scale / 2)
            half_w = max(half_w, 6)
            half_l = max(half_l, 6)

            tl = (center[0] - half_w, center[1] - half_l)
            br = (center[0] + half_w, center[1] + half_l)

            # Semi-transparent fill
            overlay = img.copy()
            cv2.rectangle(overlay, tl, br, color, -1)
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
            cv2.rectangle(img, tl, br, color, 2)

            # Label
            obj_id = o.get("id", cat)
            score = o.get("score", o.get("confidence", 0))
            label = f"{obj_id} ({score:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            label_y = max(tl[1] - 4, th + 4)
            cv2.putText(img, label, (tl[0], label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Title
        title = f"Room: {room_width:.1f}m x {room_length:.1f}m x {room_height:.1f}m  |  {len(objects)} objects"
        cv2.putText(img, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Scale bar
        bar_m = 1.0  # 1 meter
        bar_px = int(bar_m * scale)
        bar_x = img_size - margin - bar_px
        bar_y = img_size - 20
        cv2.line(img, (bar_x, bar_y), (bar_x + bar_px, bar_y), (0, 0, 0), 2)
        cv2.putText(img, "1m", (bar_x + bar_px // 2 - 10, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        cv2.imwrite("/tmp/decroom_room_map.png", img)
        logger.info("Debug room map saved to /tmp/decroom_room_map.png")
    except Exception as e:
        logger.debug("Failed to save room map: %s", e)
