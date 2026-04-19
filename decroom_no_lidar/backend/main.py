"""
Decroom Backend Server
FastAPI application that receives room photos and returns 3D room models.
"""
from __future__ import annotations

import atexit
import datetime as dt
import glob
import json
import logging
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from pipeline.config import PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("decroom-backend")

app = FastAPI(title="Decroom Backend", version="0.1.0")

# Singleton config
cfg = PipelineConfig()
gate_env = os.getenv("DECROOM_ENABLE_LARGE_DEPTH_OUTLIER_GATE", "").strip().lower()
if gate_env in {"1", "true", "yes", "on"}:
    cfg.enable_large_depth_outlier_gate = True
elif gate_env in {"0", "false", "no", "off"}:
    cfg.enable_large_depth_outlier_gate = False
metric_scale_env = os.getenv("DECROOM_ENABLE_METRIC_FP_SCALE_CALIBRATION", "").strip().lower()
if metric_scale_env in {"1", "true", "yes", "on"}:
    cfg.enable_metric_fp_scale_calibration = True
elif metric_scale_env in {"0", "false", "no", "off"}:
    cfg.enable_metric_fp_scale_calibration = False
metric_min_ratios_env = os.getenv("DECROOM_METRIC_FP_SCALE_MIN_RATIOS", "").strip()
if metric_min_ratios_env:
    try:
        cfg.metric_fp_scale_min_ratios = max(20, int(metric_min_ratios_env))
    except Exception:
        logger.warning("Invalid DECROOM_METRIC_FP_SCALE_MIN_RATIOS: %s", metric_min_ratios_env)
metric_min_factor_env = os.getenv("DECROOM_METRIC_FP_SCALE_MIN_FACTOR", "").strip()
if metric_min_factor_env:
    try:
        cfg.metric_fp_scale_min_factor = max(0.1, float(metric_min_factor_env))
    except Exception:
        logger.warning("Invalid DECROOM_METRIC_FP_SCALE_MIN_FACTOR: %s", metric_min_factor_env)
metric_max_factor_env = os.getenv("DECROOM_METRIC_FP_SCALE_MAX_FACTOR", "").strip()
if metric_max_factor_env:
    try:
        cfg.metric_fp_scale_max_factor = max(0.2, float(metric_max_factor_env))
    except Exception:
        logger.warning("Invalid DECROOM_METRIC_FP_SCALE_MAX_FACTOR: %s", metric_max_factor_env)
metric_max_iqr_env = os.getenv("DECROOM_METRIC_FP_SCALE_MAX_IQR", "").strip()
if metric_max_iqr_env:
    try:
        cfg.metric_fp_scale_max_iqr = max(0.01, float(metric_max_iqr_env))
    except Exception:
        logger.warning("Invalid DECROOM_METRIC_FP_SCALE_MAX_IQR: %s", metric_max_iqr_env)
metric_max_spread_env = os.getenv("DECROOM_METRIC_FP_SCALE_MAX_SPREAD", "").strip()
if metric_max_spread_env:
    try:
        cfg.metric_fp_scale_max_spread = max(0.01, float(metric_max_spread_env))
    except Exception:
        logger.warning("Invalid DECROOM_METRIC_FP_SCALE_MAX_SPREAD: %s", metric_max_spread_env)
gate_hard_env = os.getenv("DECROOM_SCAN_QUALITY_GATE_HARD_FAIL", "").strip().lower()
if gate_hard_env in {"1", "true", "yes", "on"}:
    cfg.scan_quality_gate_hard_fail = True
elif gate_hard_env in {"0", "false", "no", "off"}:
    cfg.scan_quality_gate_hard_fail = False
cats_env = os.getenv("DECROOM_LARGE_DEPTH_OUTLIER_CATEGORIES", "").strip()
if cats_env:
    cfg.large_depth_outlier_categories = [
        c.strip() for c in cats_env.split(",") if c.strip()
    ]
sfm_env = os.getenv("DECROOM_ENABLE_SFM_BA", "").strip().lower()
if sfm_env in {"1", "true", "yes", "on"}:
    cfg.enable_sfm_ba = True
elif sfm_env in {"0", "false", "no", "off"}:
    cfg.enable_sfm_ba = False
matcher_env = os.getenv("DECROOM_SFM_MATCHER", "").strip().lower()
if matcher_env in {"sequential", "exhaustive"}:
    cfg.sfm_matcher = matcher_env
colmap_bin_env = os.getenv("DECROOM_SFM_COLMAP_BIN", "").strip()
if colmap_bin_env:
    cfg.sfm_colmap_binary = colmap_bin_env
pycolmap_env = os.getenv("DECROOM_ENABLE_PYCOLMAP", "").strip().lower()
if pycolmap_env in {"1", "true", "yes", "on"}:
    cfg.enable_pycolmap_backend = True
elif pycolmap_env in {"0", "false", "no", "off"}:
    cfg.enable_pycolmap_backend = False
prefer_cli_env = os.getenv("DECROOM_SFM_PREFER_CLI", "").strip().lower()
if prefer_cli_env in {"1", "true", "yes", "on"}:
    cfg.sfm_prefer_colmap_cli = True
elif prefer_cli_env in {"0", "false", "no", "off"}:
    cfg.sfm_prefer_colmap_cli = False
sfm_req_ratio_env = os.getenv("DECROOM_SFM_REQUIRED_REGISTRATION_RATIO", "").strip()
if sfm_req_ratio_env:
    try:
        cfg.sfm_required_registration_ratio = max(0.1, min(0.95, float(sfm_req_ratio_env)))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_REQUIRED_REGISTRATION_RATIO: %s", sfm_req_ratio_env)
sfm_req_aligned_env = os.getenv("DECROOM_SFM_REQUIRED_ALIGNED_POSES", "").strip()
if sfm_req_aligned_env:
    try:
        cfg.sfm_required_aligned_poses = max(3, int(sfm_req_aligned_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_REQUIRED_ALIGNED_POSES: %s", sfm_req_aligned_env)
tracklet_stitch_env = os.getenv("DECROOM_ENABLE_TRACKLET_STITCHING", "").strip().lower()
if tracklet_stitch_env in {"1", "true", "yes", "on"}:
    cfg.enable_tracklet_stitching = True
elif tracklet_stitch_env in {"0", "false", "no", "off"}:
    cfg.enable_tracklet_stitching = False
tracklet_gap_env = os.getenv("DECROOM_TRACKLET_STITCH_MAX_GAP_FRAMES", "").strip()
if tracklet_gap_env:
    try:
        cfg.tracklet_stitch_max_gap_frames = max(1, int(tracklet_gap_env))
    except Exception:
        logger.warning("Invalid DECROOM_TRACKLET_STITCH_MAX_GAP_FRAMES: %s", tracklet_gap_env)
tracklet_dist_env = os.getenv("DECROOM_TRACKLET_STITCH_WORLD_DIST_FACTOR", "").strip()
if tracklet_dist_env:
    try:
        cfg.tracklet_stitch_world_dist_factor = max(1.0, float(tracklet_dist_env))
    except Exception:
        logger.warning("Invalid DECROOM_TRACKLET_STITCH_WORLD_DIST_FACTOR: %s", tracklet_dist_env)
tracklet_score_env = os.getenv("DECROOM_TRACKLET_STITCH_MIN_SCORE", "").strip()
if tracklet_score_env:
    try:
        cfg.tracklet_stitch_min_score = float(tracklet_score_env)
    except Exception:
        logger.warning("Invalid DECROOM_TRACKLET_STITCH_MIN_SCORE: %s", tracklet_score_env)
tracklet_passes_env = os.getenv("DECROOM_TRACKLET_STITCH_MAX_PASSES", "").strip()
if tracklet_passes_env:
    try:
        cfg.tracklet_stitch_max_passes = max(1, int(tracklet_passes_env))
    except Exception:
        logger.warning("Invalid DECROOM_TRACKLET_STITCH_MAX_PASSES: %s", tracklet_passes_env)
dup_merge_env = os.getenv("DECROOM_ENABLE_TRACK_DUPLICATE_MERGE", "").strip().lower()
if dup_merge_env in {"1", "true", "yes", "on"}:
    cfg.enable_track_duplicate_merge = True
elif dup_merge_env in {"0", "false", "no", "off"}:
    cfg.enable_track_duplicate_merge = False
dup_iou_env = os.getenv("DECROOM_TRACK_DUPLICATE_IOU_MIN", "").strip()
if dup_iou_env:
    try:
        cfg.track_duplicate_iou_min = max(0.05, min(0.95, float(dup_iou_env)))
    except Exception:
        logger.warning("Invalid DECROOM_TRACK_DUPLICATE_IOU_MIN: %s", dup_iou_env)
dup_center_env = os.getenv("DECROOM_TRACK_DUPLICATE_CENTER_MAX", "").strip()
if dup_center_env:
    try:
        cfg.track_duplicate_center_max = max(0.05, min(1.2, float(dup_center_env)))
    except Exception:
        logger.warning("Invalid DECROOM_TRACK_DUPLICATE_CENTER_MAX: %s", dup_center_env)
dup_world_env = os.getenv("DECROOM_TRACK_DUPLICATE_WORLD_DIST_FACTOR", "").strip()
if dup_world_env:
    try:
        cfg.track_duplicate_world_dist_factor = max(0.5, min(2.0, float(dup_world_env)))
    except Exception:
        logger.warning("Invalid DECROOM_TRACK_DUPLICATE_WORLD_DIST_FACTOR: %s", dup_world_env)
dup_score_env = os.getenv("DECROOM_TRACK_DUPLICATE_MIN_SCORE", "").strip()
if dup_score_env:
    try:
        cfg.track_duplicate_min_score = float(dup_score_env)
    except Exception:
        logger.warning("Invalid DECROOM_TRACK_DUPLICATE_MIN_SCORE: %s", dup_score_env)
dup_passes_env = os.getenv("DECROOM_TRACK_DUPLICATE_MAX_PASSES", "").strip()
if dup_passes_env:
    try:
        cfg.track_duplicate_max_passes = max(1, int(dup_passes_env))
    except Exception:
        logger.warning("Invalid DECROOM_TRACK_DUPLICATE_MAX_PASSES: %s", dup_passes_env)
sfm_prefilter_env = os.getenv("DECROOM_SFM_ENABLE_KEYFRAME_PREFILTER", "").strip().lower()
if sfm_prefilter_env in {"1", "true", "yes", "on"}:
    cfg.sfm_enable_keyframe_prefilter = True
elif sfm_prefilter_env in {"0", "false", "no", "off"}:
    cfg.sfm_enable_keyframe_prefilter = False
sfm_kf_trans_env = os.getenv("DECROOM_SFM_KEYFRAME_MIN_TRANSLATION_M", "").strip()
if sfm_kf_trans_env:
    try:
        cfg.sfm_keyframe_min_translation_m = max(0.01, float(sfm_kf_trans_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_KEYFRAME_MIN_TRANSLATION_M: %s", sfm_kf_trans_env)
sfm_kf_yaw_env = os.getenv("DECROOM_SFM_KEYFRAME_MIN_YAW_DEG", "").strip()
if sfm_kf_yaw_env:
    try:
        cfg.sfm_keyframe_min_yaw_deg = max(0.5, float(sfm_kf_yaw_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_KEYFRAME_MIN_YAW_DEG: %s", sfm_kf_yaw_env)
sfm_kf_stride_env = os.getenv("DECROOM_SFM_KEYFRAME_FORCE_STRIDE", "").strip()
if sfm_kf_stride_env:
    try:
        cfg.sfm_keyframe_force_stride = max(1, int(sfm_kf_stride_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_KEYFRAME_FORCE_STRIDE: %s", sfm_kf_stride_env)
sfm_kf_sharp_env = os.getenv("DECROOM_SFM_KEYFRAME_SHARPNESS_MIN", "").strip()
if sfm_kf_sharp_env:
    try:
        cfg.sfm_keyframe_sharpness_min = max(1.0, float(sfm_kf_sharp_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_KEYFRAME_SHARPNESS_MIN: %s", sfm_kf_sharp_env)
sfm_kf_max_env = os.getenv("DECROOM_SFM_KEYFRAME_MAX_FRAMES", "").strip()
if sfm_kf_max_env:
    try:
        cfg.sfm_keyframe_max_frames = max(8, int(sfm_kf_max_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_KEYFRAME_MAX_FRAMES: %s", sfm_kf_max_env)
sfm_kf_min_env = os.getenv("DECROOM_SFM_KEYFRAME_MIN_FRAMES", "").strip()
if sfm_kf_min_env:
    try:
        cfg.sfm_keyframe_min_frames = max(4, int(sfm_kf_min_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_KEYFRAME_MIN_FRAMES: %s", sfm_kf_min_env)
sfm_enhance_env = os.getenv("DECROOM_SFM_ENABLE_IMAGE_ENHANCEMENT", "").strip().lower()
if sfm_enhance_env in {"1", "true", "yes", "on"}:
    cfg.sfm_enable_image_enhancement = True
elif sfm_enhance_env in {"0", "false", "no", "off"}:
    cfg.sfm_enable_image_enhancement = False
sfm_enhance_contrast_env = os.getenv("DECROOM_SFM_ENHANCE_CONTRAST", "").strip()
if sfm_enhance_contrast_env:
    try:
        cfg.sfm_enhance_contrast = max(1.0, float(sfm_enhance_contrast_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_ENHANCE_CONTRAST: %s", sfm_enhance_contrast_env)
sfm_enhance_sharp_env = os.getenv("DECROOM_SFM_ENHANCE_SHARPNESS", "").strip()
if sfm_enhance_sharp_env:
    try:
        cfg.sfm_enhance_sharpness = max(1.0, float(sfm_enhance_sharp_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_ENHANCE_SHARPNESS: %s", sfm_enhance_sharp_env)
sfm_sift_feats_env = os.getenv("DECROOM_SFM_SIFT_MAX_NUM_FEATURES", "").strip()
if sfm_sift_feats_env:
    try:
        cfg.sfm_sift_max_num_features = max(2000, int(sfm_sift_feats_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_SIFT_MAX_NUM_FEATURES: %s", sfm_sift_feats_env)
sfm_sift_peak_env = os.getenv("DECROOM_SFM_SIFT_PEAK_THRESHOLD", "").strip()
if sfm_sift_peak_env:
    try:
        cfg.sfm_sift_peak_threshold = max(1e-4, float(sfm_sift_peak_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_SIFT_PEAK_THRESHOLD: %s", sfm_sift_peak_env)
sfm_match_guided_env = os.getenv("DECROOM_SFM_MATCHING_GUIDED", "").strip().lower()
if sfm_match_guided_env in {"1", "true", "yes", "on"}:
    cfg.sfm_matching_guided = True
elif sfm_match_guided_env in {"0", "false", "no", "off"}:
    cfg.sfm_matching_guided = False
sfm_map_min_matches_env = os.getenv("DECROOM_SFM_MAPPER_MIN_NUM_MATCHES", "").strip()
if sfm_map_min_matches_env:
    try:
        cfg.sfm_mapper_min_num_matches = max(4, int(sfm_map_min_matches_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_MAPPER_MIN_NUM_MATCHES: %s", sfm_map_min_matches_env)
sfm_map_init_inliers_env = os.getenv("DECROOM_SFM_MAPPER_INIT_MIN_NUM_INLIERS", "").strip()
if sfm_map_init_inliers_env:
    try:
        cfg.sfm_mapper_init_min_num_inliers = max(8, int(sfm_map_init_inliers_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_MAPPER_INIT_MIN_NUM_INLIERS: %s", sfm_map_init_inliers_env)
sfm_map_abs_inliers_env = os.getenv("DECROOM_SFM_MAPPER_ABS_POSE_MIN_NUM_INLIERS", "").strip()
if sfm_map_abs_inliers_env:
    try:
        cfg.sfm_mapper_abs_pose_min_num_inliers = max(6, int(sfm_map_abs_inliers_env))
    except Exception:
        logger.warning("Invalid DECROOM_SFM_MAPPER_ABS_POSE_MIN_NUM_INLIERS: %s", sfm_map_abs_inliers_env)
pose_jump_reject_env = os.getenv("DECROOM_ENABLE_POSE_JUMP_REJECTION", "").strip().lower()
if pose_jump_reject_env in {"1", "true", "yes", "on"}:
    cfg.enable_pose_jump_rejection = True
elif pose_jump_reject_env in {"0", "false", "no", "off"}:
    cfg.enable_pose_jump_rejection = False
pose_jump_thr_env = os.getenv("DECROOM_POSE_JUMP_REJECTION_THRESHOLD_M", "").strip()
if pose_jump_thr_env:
    try:
        cfg.pose_jump_rejection_threshold_m = float(pose_jump_thr_env)
    except Exception:
        logger.warning("Invalid DECROOM_POSE_JUMP_REJECTION_THRESHOLD_M: %s", pose_jump_thr_env)
pose_jump_ratio_env = os.getenv("DECROOM_POSE_JUMP_REJECTION_MIN_RATIO", "").strip()
if pose_jump_ratio_env:
    try:
        cfg.pose_jump_rejection_min_ratio = float(pose_jump_ratio_env)
    except Exception:
        logger.warning("Invalid DECROOM_POSE_JUMP_REJECTION_MIN_RATIO: %s", pose_jump_ratio_env)
pose_jump_min_jumps_env = os.getenv("DECROOM_POSE_JUMP_REJECTION_MIN_JUMPS", "").strip()
if pose_jump_min_jumps_env:
    try:
        cfg.pose_jump_rejection_min_jumps = max(1, int(pose_jump_min_jumps_env))
    except Exception:
        logger.warning("Invalid DECROOM_POSE_JUMP_REJECTION_MIN_JUMPS: %s", pose_jump_min_jumps_env)
pose_jump_speed_env = os.getenv("DECROOM_POSE_JUMP_REJECTION_MAX_SPEED_MPS", "").strip()
if pose_jump_speed_env:
    try:
        cfg.pose_jump_rejection_max_speed_mps = float(pose_jump_speed_env)
    except Exception:
        logger.warning("Invalid DECROOM_POSE_JUMP_REJECTION_MAX_SPEED_MPS: %s", pose_jump_speed_env)
pose_jump_hard_env = os.getenv("DECROOM_POSE_JUMP_REJECTION_HARD_THRESHOLD_M", "").strip()
if pose_jump_hard_env:
    try:
        cfg.pose_jump_rejection_hard_threshold_m = float(pose_jump_hard_env)
    except Exception:
        logger.warning("Invalid DECROOM_POSE_JUMP_REJECTION_HARD_THRESHOLD_M: %s", pose_jump_hard_env)
# SpatialLM integration
spatiallm_env = os.getenv("DECROOM_ENABLE_SPATIALLM", "").strip().lower()
if spatiallm_env in {"1", "true", "yes", "on"}:
    cfg.enable_spatiallm = True
elif spatiallm_env in {"0", "false", "no", "off"}:
    cfg.enable_spatiallm = False
spatiallm_mode_env = os.getenv("DECROOM_SPATIALLM_MODE", "").strip().lower()
if spatiallm_mode_env in {"local", "remote"}:
    cfg.spatiallm_mode = spatiallm_mode_env
spatiallm_url_env = os.getenv("DECROOM_SPATIALLM_API_URL", "").strip()
if spatiallm_url_env:
    cfg.spatiallm_api_url = spatiallm_url_env

layout_fp_margin_env = os.getenv("DECROOM_LAYOUT_FP_MARGIN_M", "").strip()
if layout_fp_margin_env:
    try:
        cfg.layout_fp_margin_m = max(0.05, float(layout_fp_margin_env))
    except Exception:
        logger.warning("Invalid DECROOM_LAYOUT_FP_MARGIN_M: %s", layout_fp_margin_env)
layout_cam_wall_pad_env = os.getenv("DECROOM_LAYOUT_CAMERA_WALL_PAD_M", "").strip()
if layout_cam_wall_pad_env:
    try:
        cfg.layout_camera_wall_pad_m = max(0.2, float(layout_cam_wall_pad_env))
    except Exception:
        logger.warning("Invalid DECROOM_LAYOUT_CAMERA_WALL_PAD_M: %s", layout_cam_wall_pad_env)
layout_cam_min_pad_env = os.getenv("DECROOM_LAYOUT_CAMERA_MIN_PAD_M", "").strip()
if layout_cam_min_pad_env:
    try:
        cfg.layout_camera_min_pad_m = max(0.05, float(layout_cam_min_pad_env))
    except Exception:
        logger.warning("Invalid DECROOM_LAYOUT_CAMERA_MIN_PAD_M: %s", layout_cam_min_pad_env)
layout_growth_env = os.getenv("DECROOM_LAYOUT_EXTENT_MAX_GROWTH_RATIO", "").strip()
if layout_growth_env:
    try:
        cfg.layout_extent_max_growth_ratio = max(1.05, float(layout_growth_env))
    except Exception:
        logger.warning("Invalid DECROOM_LAYOUT_EXTENT_MAX_GROWTH_RATIO: %s", layout_growth_env)
layout_center_blend_env = os.getenv("DECROOM_LAYOUT_CENTER_BLEND_ALPHA", "").strip()
if layout_center_blend_env:
    try:
        cfg.layout_center_blend_alpha = max(0.0, min(0.5, float(layout_center_blend_env)))
    except Exception:
        logger.warning("Invalid DECROOM_LAYOUT_CENTER_BLEND_ALPHA: %s", layout_center_blend_env)

# ── Bonjour / mDNS service registration ─────────────────────────
# Advertises "_decroom._tcp.local." so the iOS app can discover
# the server automatically without a hardcoded IP address.
_zeroconf = None
_service_info = None

PORT = 8000
RUN_ARCHIVE_ROOT = Path("/tmp/decroom_runs")

_dns_sd_proc: subprocess.Popen | None = None

def _register_bonjour():
    """Register via macOS native dns-sd command (always works, no deps)."""
    global _dns_sd_proc
    try:
        _dns_sd_proc = subprocess.Popen(
            ["dns-sd", "-R", "Decroom Backend", "_decroom._tcp", "local", str(PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Bonjour: registered _decroom._tcp on port %d (dns-sd pid=%d)",
                     PORT, _dns_sd_proc.pid)
    except Exception as e:
        logger.warning("Bonjour registration failed (non-fatal): %s", e)

def _unregister_bonjour():
    global _dns_sd_proc
    if _dns_sd_proc:
        _dns_sd_proc.terminate()
        _dns_sd_proc = None
        logger.info("Bonjour: unregistered")


def _archive_debug_artifacts(
    room_id: str,
    payload: dict,
    metadata_raw: Optional[dict],
    run_started_at: float,
) -> None:
    """Copy /tmp decroom diagnostics to a timestamped run folder."""
    try:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUN_ARCHIVE_ROOT / f"{ts}_{room_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        src_globs = [
            "/tmp/decroom_*.json",
            "/tmp/decroom_*.png",
        ]
        copied = 0
        copied_names = []
        run_end = time.time()
        for g in src_globs:
            for src in glob.glob(g):
                src_path = Path(src)
                if not src_path.is_file():
                    continue
                try:
                    mtime = src_path.stat().st_mtime
                except Exception:
                    continue
                # Keep only artifacts created/updated during this run.
                if mtime < run_started_at - 1.0 or mtime > run_end + 1.0:
                    continue
                dst = run_dir / src_path.name
                shutil.copy2(src_path, dst)
                copied += 1
                copied_names.append(src_path.name)

        if metadata_raw is not None:
            with open(run_dir / "request_metadata.json", "w") as f:
                json.dump(metadata_raw, f, indent=2)
        with open(run_dir / "response_payload.json", "w") as f:
            json.dump(payload, f, indent=2)

        latest = RUN_ARCHIVE_ROOT / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            os.symlink(str(run_dir), str(latest))
        except Exception:
            # Symlink update is best-effort; archive is still valid without it.
            pass

        logger.info(
            "Archived %d debug artifacts to %s (files=%s)",
            copied,
            run_dir,
            ", ".join(sorted(copied_names)) if copied_names else "none",
        )
    except Exception as e:
        logger.warning("Failed to archive debug artifacts: %s", e)


@app.on_event("startup")
async def on_startup():
    _register_bonjour()
    atexit.register(_unregister_bonjour)


@app.post("/api/upload")
async def upload_room(
    frames: List[UploadFile] = File(...),
    metadata: Optional[UploadFile] = File(None),
):
    """
    Receive room photos and optional metadata, run the pipeline,
    and return a RoomPayload JSON.
    """
    room_id = str(uuid.uuid4())[:8]
    logger.info("Upload received: %d frames, room_id=%s", len(frames), room_id)

    # Server-side frame count limit (defense against malicious/runaway clients)
    max_upload_frames = 300  # generous ceiling above max_frames=200 video + headroom
    if len(frames) > max_upload_frames:
        return JSONResponse(
            status_code=413,
            content={"error": "too_many_frames", "detail": f"Maximum {max_upload_frames} frames allowed, got {len(frames)}"},
        )

    # Parse metadata JSON
    metadata_raw = None
    if metadata is not None:
        try:
            meta_bytes = await metadata.read()
            metadata_raw = json.loads(meta_bytes)
            logger.info("Metadata parsed: %d bytes", len(meta_bytes))
        except Exception as e:
            logger.warning("Failed to parse metadata: %s", e)

    # Import pipeline lazily to avoid slow startup
    from pipeline.pipeline import run_pipeline

    run_started_at = time.time()
    try:
        result = await run_in_threadpool(
            run_pipeline,
            frames,
            metadata_raw,
            cfg,
        )
    except RuntimeError as e:
        message = str(e)
        if message.startswith("SCAN_QUALITY_GATE_FAILED"):
            logger.warning("Upload rejected by scan quality gate: %s", message)
            error_payload = {
                "error": "scan_quality_gate_failed",
                "detail": message,
            }
            _archive_debug_artifacts(room_id, error_payload, metadata_raw, run_started_at)
            return JSONResponse(status_code=422, content=error_payload)
        raise
    _archive_debug_artifacts(room_id, result, metadata_raw, run_started_at)

    return JSONResponse(content=result)


@app.post("/api/upload/phone-first")
async def upload_phone_first(
    room_geometry: UploadFile = File(...),
    point_cloud: UploadFile = File(...),
    keyframes_manifest: UploadFile = File(...),
    keyframes: List[UploadFile] = File(...),
    scan_log: UploadFile = File(...),
):
    """Receive a phone-first upload: walls + PLY + keyframes + log.
    Runs SpatialLM on the phone PLY and returns a RoomPayload."""
    room_id = str(uuid.uuid4())[:8]
    run_started_at = time.time()
    logger.info("Phone-first upload received: room_id=%s, %d keyframes",
                room_id, len(keyframes))

    try:
        from pipeline.phone_first_pipeline import (
            PhoneFirstSchemaError,
            parse_payload,
            run_phone_first_pipeline,
        )

        room_geom_bytes = await room_geometry.read()
        ply_bytes = await point_cloud.read()
        manifest_bytes = await keyframes_manifest.read()
        keyframe_bytes = [await kf.read() for kf in keyframes]
        log_bytes = await scan_log.read() if scan_log is not None else b""

        try:
            payload = parse_payload(
                room_geom_bytes, ply_bytes, manifest_bytes,
                keyframe_bytes, log_bytes,
            )
        except PhoneFirstSchemaError as e:
            logger.warning("Phone-first payload rejected: %s", e)
            return JSONResponse(
                status_code=400,
                content={"error": "schema_error", "detail": str(e)},
            )

        result = await run_in_threadpool(
            run_phone_first_pipeline, payload, cfg
        )
    except Exception as e:
        logger.exception("Phone-first pipeline failed")
        return JSONResponse(
            status_code=500,
            content={"error": "pipeline_failure", "detail": str(e)},
        )

    _archive_debug_artifacts(room_id, result, None, run_started_at)
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
