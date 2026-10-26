from dataclasses import dataclass, field
import logging
from pathlib import Path
import shutil
from typing import Dict, List

from .data_dir import data_dir

_config_logger = logging.getLogger("decroom-backend.config")


def _resolve_depth_model() -> Path:
    """Auto-resolve best available depth model, preferring metric."""
    models_dir = Path(__file__).resolve().parent.parent / "models" / "depth"
    candidates = [
        ("depth_anything_v2_metric_hypersim_vitb.onnx", "dav2_metric"),
        ("depth_anything_v2_metric_hypersim_vitl.onnx", "dav2_metric"),
        ("depth_anything_v2_metric_vkitti_vitb.onnx", "dav2_metric"),
        ("depth_anything_v2_vitb.onnx", "dav2"),
        ("depth_anything_v2_base.onnx", "dav2"),
        ("video_depth_anything_small.onnx", "vda"),
    ]
    for filename, _ in candidates:
        path = models_dir / filename
        if path.exists() and path.stat().st_size > 1000:
            _config_logger.info("Auto-selected depth model: %s", filename)
            return path
    return models_dir / candidates[0][0]


def _resolve_depth_model_type(model_path: Path) -> str:
    """Infer model type from filename."""
    name = model_path.name.lower()
    if "metric" in name or "hypersim" in name or "vkitti" in name:
        return "dav2_metric"
    elif "depth_anything_v2" in name:
        return "dav2"
    else:
        return "vda"


def _resolve_colmap_binary() -> str:
    """Prefer Homebrew COLMAP when present, else fallback to PATH lookup."""
    brew_colmap = Path("/opt/homebrew/bin/colmap")
    if brew_colmap.exists():
        return str(brew_colmap)
    if shutil.which("colmap") is not None:
        return "colmap"
    return str(brew_colmap)


# Canonical COCO -> PRD category mapping
COCO_TO_PRD = {
    "bed": "bed",
    "chair": "chair",
    "bench": "chair",
    "couch": "sofa",
    "sofa": "sofa",
    "dining table": "desk",
    "potted plant": "plant",
    "tv": "monitor",
    "laptop": "monitor",
    "book": "bookshelf",
    "vase": "plant",
    "clock": "poster",
    "refrigerator": "dresser",
    "suitcase": "dresser",
    "backpack": "dresser",
}


@dataclass
class PipelineConfig:
    """Configuration for the PRD pipeline steps."""

    model_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "models"
    )
    depth_model: Path = field(
        default_factory=lambda: _resolve_depth_model()
    )
    depth_model_type: str = "auto"
    detector_model: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "models/detector/yolov8s-seg.onnx"
    )
    layout_model: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "models/layout/horizonnet.pth"
    )

    frame_downscale: float = 1.0
    max_frames: int = 96
    random_seed: int = 42
    debug_output: bool = True
    enable_metric_reprojection_filter: bool = False
    enable_metric_fp_scale_calibration: bool = True
    metric_fp_scale_min_ratios: int = 120
    metric_fp_scale_min_factor: float = 0.45
    metric_fp_scale_max_factor: float = 1.85
    metric_fp_scale_max_iqr: float = 0.35
    metric_fp_scale_max_spread: float = 1.0
    enable_large_depth_outlier_gate: bool = True
    scan_quality_gate_hard_fail: bool = False
    large_depth_outlier_categories: List[str] = field(default_factory=lambda: [
        "bed", "sofa", "desk", "table", "bookshelf",
    ])
    detection_min_confidence: float = 0.15
    detection_class_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "bed": 0.15,
        "desk": 0.20,
        "chair": 0.15,
        "table": 0.20,
        "dresser": 0.40,
        "nightstand": 0.30,
        "lamp_floor": 0.30,
        "lamp_desk": 0.30,
        "plant": 0.30,
        "bookshelf": 0.25,
        "monitor": 0.20,
        "sofa": 0.15,
        "rug": 0.25,
        "poster": 0.40,
    })
    object_merge_distance_m: float = 0.9
    object_merge_distance_by_type: Dict[str, float] = field(default_factory=lambda: {
        "bed": 1.8,
        "sofa": 1.5,
        "desk": 1.1,
        "dresser": 1.0,
        "bookshelf": 1.0,
        "monitor": 0.80,
        "chair": 0.65,
        "table": 1.0,
        "nightstand": 0.75,
        "lamp_floor": 0.75,
        "lamp_desk": 0.6,
        "plant": 0.7,
        "poster": 0.9,
    })
    object_merge_size_ratio: float = 0.6
    cluster_same_frame_iou_duplicate_min: float = 0.55
    cluster_same_frame_containment_min: float = 0.82
    cluster_small_object_crossframe_only: bool = True

    categories: List[str] = field(default_factory=lambda: [
        "bed", "desk", "chair", "table", "dresser", "nightstand",
        "lamp_floor", "lamp_desk", "plant", "bookshelf", "monitor",
        "sofa", "rug", "poster",
    ])

    # Reconstruction upgrades
    # Pose-path smoothing is opt-in to avoid changing production behavior
    # without explicit validation.
    enable_pose_path_smoothing: bool = False
    pose_smoothing_window: int = 2
    pose_max_adjustment_m: float = 0.08
    pose_jump_threshold_m: float = 1.2
    enable_pose_jump_rejection: bool = True
    pose_jump_rejection_threshold_m: float = 1.2
    pose_jump_rejection_min_ratio: float = 8.0
    pose_jump_rejection_min_jumps: int = 2
    pose_jump_rejection_min_keep_frames: int = 8
    pose_jump_rejection_max_speed_mps: float = 1.6
    pose_jump_rejection_hard_threshold_m: float = 3.0
    pose_jump_rejection_hybrid_step_m: float = 0.75
    pose_jump_rejection_hybrid_yaw_deg: float = 95.0
    axis_snap_min_camera_extent_m: float = 0.9
    axis_snap_min_camera_extent_ratio: float = 0.16
    layout_fp_margin_m: float = 0.2
    layout_camera_wall_pad_m: float = 0.55
    layout_camera_min_pad_m: float = 0.2
    layout_extent_max_growth_ratio: float = 1.55
    layout_center_blend_alpha: float = 0.25

    # Sparse depth fusion (multi-view geometry support)
    enable_sparse_depth_fusion: bool = True
    fusion_depth_stride_px: int = 16
    fusion_max_points_per_frame: int = 900
    fusion_voxel_size_m: float = 0.05
    fusion_max_depth_m: float = 5.0  # was 8.0 — clamp to 5m to reject depth through windows/hallways
    fusion_include_feature_cloud: bool = False
    enable_fused_layout_assist: bool = True
    min_feature_cloud_for_layout: int = 80

    # Multi-view track triangulation priors
    enable_track_ray_triangulation: bool = True
    triangulation_min_views: int = 2
    triangulation_max_condition: float = 1e6
    triangulation_max_ray_error_m: float = 0.45

    # Tracklet stitching (post-tracking fragmentation recovery)
    enable_tracklet_stitching: bool = True
    tracklet_stitch_max_gap_frames: int = 12  # was 8
    tracklet_stitch_world_dist_factor: float = 1.35
    tracklet_stitch_min_score: float = 0.25
    tracklet_stitch_max_passes: int = 6
    tracklet_stitch_2d_fallback_max_gap: int = 6
    enable_singleton_merge: bool = True
    singleton_merge_world_dist_factor: float = 1.5
    singleton_merge_min_appearance: float = 0.70
    singleton_merge_high_conf_threshold: float = 0.80

    # Duplicate-overlap track merge (same object split into concurrent tracks)
    enable_track_duplicate_merge: bool = True
    track_duplicate_iou_min: float = 0.22
    track_duplicate_center_max: float = 0.5
    track_duplicate_world_dist_factor: float = 0.9
    track_duplicate_min_score: float = 0.2
    track_duplicate_max_passes: int = 8
    track_appearance_weight: float = 0.25
    track_high_similarity_relaxation: float = 1.18
    tracking_fragmentation_alert_threshold: float = 0.65
    track_new_track_min_score_by_cat: Dict[str, float] = field(default_factory=lambda: {
        "monitor": 0.45,
        "chair": 0.40,
        "bookshelf": 0.30,
        "dresser": 0.30,
    })
    track_monitor_weak_sfm_relaxation: float = 1.25

    # Feature-point usability weighting (shot-level quality + UV spread).
    fp_usability_min_uv_area_ratio: float = 0.02
    fp_usability_low_quality_weight: float = 0.45
    fp_usability_high_quality_weight: float = 1.15

    # Diagnostics artifact
    save_reconstruction_report: bool = True

    # Optional full SfM + BA stage (COLMAP CLI-backed)
    enable_sfm_ba: bool = True
    sfm_colmap_binary: str = field(default_factory=_resolve_colmap_binary)
    sfm_min_frames: int = 8
    sfm_matcher: str = "sequential"  # sequential | exhaustive
    sfm_sequential_overlap: int = 10
    sfm_use_gpu: bool = False
    sfm_max_image_size: int = 1600
    sfm_camera_model: str = "SIMPLE_PINHOLE"
    sfm_required_registration_ratio: float = 0.25
    sfm_required_aligned_poses: int = 4
    sfm_workspace_root: str = field(default_factory=lambda: str(data_dir() / "sfm"))
    sfm_keep_workspace: bool = False
    enable_pycolmap_backend: bool = True
    sfm_prefer_colmap_cli: bool = True
    sfm_enable_keyframe_prefilter: bool = True
    sfm_keyframe_min_translation_m: float = 0.04
    sfm_keyframe_min_yaw_deg: float = 3.0
    sfm_keyframe_force_stride: int = 4
    sfm_keyframe_sharpness_min: float = 14.0
    sfm_keyframe_max_frames: int = 120
    sfm_keyframe_min_frames: int = 14
    sfm_enable_image_enhancement: bool = True
    sfm_enhance_contrast: float = 1.14
    sfm_enhance_sharpness: float = 1.18
    sfm_enhance_autocontrast_cutoff: float = 0.5
    sfm_sift_max_num_features: int = 12000
    sfm_sift_peak_threshold: float = 0.004
    sfm_sift_edge_threshold: float = 10.0
    sfm_matching_guided: bool = False
    sfm_matching_max_num_matches: int = 32768
    sfm_mapper_min_num_matches: int = 10
    sfm_mapper_init_min_num_inliers: int = 24
    sfm_mapper_abs_pose_min_num_inliers: int = 18
    sfm_mapper_abs_pose_min_inlier_ratio: float = 0.1
    # Loop detection for video walk-around captures
    sfm_enable_loop_detection: bool = True
    sfm_loop_detection_period: int = 10
    sfm_loop_detection_num_images: int = 50

    # Depth stride for video (>60 frames)
    video_depth_stride: int = 2

    sfm_enable_match_graph_prefilter: bool = True
    sfm_graph_temporal_window: int = 5
    sfm_graph_min_inliers: int = 45
    sfm_graph_min_inlier_ratio: float = 0.22
    sfm_graph_min_component_size: int = 3
    sfm_partial_apply_enabled: bool = True
    sfm_partial_apply_min_registered: int = 4
    sfm_partial_apply_min_aligned_poses: int = 3
    sfm_partial_apply_max_rmse_m: float = 0.22
    sfm_apply_max_rmse_m: float = 0.35

    # SfM recovery pass — re-register unregistered frames with relaxed thresholds
    sfm_enable_recovery_pass: bool = True
    sfm_recovery_min_unregistered: int = 3
    sfm_recovery_registration_ratio_trigger: float = 0.6
    sfm_recovery_abs_pose_min_num_inliers: int = 12
    sfm_recovery_abs_pose_min_inlier_ratio: float = 0.06
    sfm_recovery_max_reproj_error_px: float = 2.0
    # Sparse capture adaptive graph prefilter
    sfm_sparse_yaw_threshold_deg: float = 45.0
    sfm_graph_min_inliers_sparse: int = 25
    sfm_graph_min_inlier_ratio_sparse: float = 0.12
    sfm_graph_temporal_window_sparse: int = 8

    # SpatialLM integration (replaces heuristic layout + lifting when enabled)
    enable_spatiallm: bool = True
    spatiallm_mode: str = "remote"  # "local" | "remote"
    spatiallm_api_url: str = "http://localhost:8100/infer"
    spatiallm_timeout_s: float = 120.0
    spatiallm_model_id: str = "manycore-research/SpatialLM1.1-Qwen-0.5B"
    spatiallm_max_keyframes: int = 30
    spatiallm_pixel_stride: int = 4
    spatiallm_max_points: int = 500_000
    spatiallm_voxel_size_m: float = 0.02
    spatiallm_max_depth_m: float = 5.0
    spatiallm_merge_with_yolo: bool = True
    spatiallm_merge_distance_m: float = 1.0
    spatiallm_fallback_on_failure: bool = True
    spatiallm_min_pointcloud_size: int = 10_000
    spatiallm_confidence: float = 0.85

    # False-positive reflection filter
    enable_reflection_filter: bool = True
    reflection_room_margin_m: float = 0.5
    reflection_min_fp_density: int = 3
    reflection_fp_radius_m: float = 0.5
    reflection_max_confidence: float = 0.6
    # Wall-mounted misdetection suppression for floor-standing categories
    enable_wall_mounted_suppression: bool = True
    wall_mounted_min_height_above_floor_m: float = 0.8
    wall_mounted_max_wall_dist_m: float = 0.3

    def __post_init__(self) -> None:
        self.model_root = self.model_root.resolve()
        if not self.depth_model.is_absolute():
            self.depth_model = self.model_root / self.depth_model
        if not self.detector_model.is_absolute():
            self.detector_model = self.model_root / self.detector_model
        if not self.layout_model.is_absolute():
            self.layout_model = self.model_root / self.layout_model
        if self.depth_model_type == "auto":
            self.depth_model_type = _resolve_depth_model_type(self.depth_model)
