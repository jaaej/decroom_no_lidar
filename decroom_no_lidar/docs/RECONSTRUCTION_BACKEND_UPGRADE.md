# Decroom Reconstruction Backend Upgrade

## Objective
Improve non-LiDAR room reconstruction accuracy by making geometry estimation robust even when object detections are sparse or missing.

## Implemented in this change set

### 0) Optional full SfM + BA stage (COLMAP CLI)
- Files:
  - `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/pipeline/sfm.py`
  - `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/pipeline/pipeline.py`
  - `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/main.py`
- Functions:
  - `run_sfm_bundle_adjustment(...)`
- Behavior:
  - Materializes frame images to a temporary workspace.
  - Runs COLMAP `feature_extractor -> matcher -> mapper -> model_converter`.
  - Parses reconstructed camera poses and aligns them to ARKit metric coordinates
    via Umeyama similarity transform.
  - Applies aligned camera poses back to `ShotMetadata.pose`.
  - Fails open (non-blocking): if COLMAP is missing or registration quality is
    low, pipeline falls back to ARKit poses.
  - Additional backend path: when COLMAP CLI is unavailable, the stage can run
    directly through `pycolmap` bindings (`extract_features`, matcher, and
    `incremental_mapping`) and then apply the same ARKit-frame alignment logic.

### 1) Geometry-first pipeline execution
- File: `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/pipeline/pipeline.py`
- Change:
  - Removed early return when detections are empty.
  - Pipeline now always runs geometry/layout stages and returns room envelope/walls.
  - Object branch is additive (`detection -> tracking -> lifting`) instead of blocking room reconstruction.

### 2) Pose refinement hook (safe, opt-in)
- File: `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/pipeline/reconstruction.py`
- Function: `refine_camera_poses(...)`
- Behavior:
  - Computes motion diagnostics from shot poses.
  - Optional translation smoothing with bounded max adjustment (`enable_pose_path_smoothing=False` by default).
  - Leaves rotation untouched for safety.

### 3) Sparse multi-view depth fusion
- File: `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/pipeline/reconstruction.py`
- Function: `fuse_multiview_geometry(...)`
- Behavior:
  - Back-projects sparse depth samples from all frames into world coordinates.
  - Voxel-deduplicates fused points.
  - Can assist layout when ARKit feature cloud is sparse (`enable_fused_layout_assist=True`).

### 4) Multi-view ray triangulation priors for objects
- Files:
  - `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/pipeline/reconstruction.py`
  - `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/pipeline/lifting.py`
- Functions:
  - `triangulate_tracks_from_rays(...)`
  - `lift_objects(..., triangulation_priors=...)`
- Behavior:
  - Triangulates track center from multi-view bbox-center rays.
  - Rejects unstable triangulations via condition number and median ray residual checks.
  - Uses accepted priors as fallback source (`position_source="triangulated_rays"`) before depth backprojection.

### 5) Unified reconstruction diagnostics artifact
- File: `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/pipeline/reconstruction.py`
- Output: `/tmp/decroom_reconstruction_report.json`
- Includes:
  - Shot/feature coverage summary
  - Pose diagnostics (before/after path stats)
  - Fusion diagnostics
  - Triangulation diagnostics
  - Layout diagnostics and high-level run summary

### 6) Empty detection diagnostics continuity
- File: `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/pipeline/tracking.py`
- Change:
  - Writes an empty tracking report when no detections are found.

## New configuration knobs
- File: `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/pipeline/config.py`
- Added:
  - SfM/BA flags: `enable_sfm_ba`, `sfm_colmap_binary`, `sfm_min_frames`, `sfm_matcher`, `sfm_sequential_overlap`, `sfm_use_gpu`, `sfm_max_image_size`, `sfm_camera_model`, `sfm_required_registration_ratio`, `sfm_required_aligned_poses`, `sfm_workspace_root`, `sfm_keep_workspace`
  - SfM backend selection flags: `enable_pycolmap_backend`, `sfm_prefer_colmap_cli`
  - Pose refinement flags: `enable_pose_path_smoothing`, `pose_smoothing_window`, `pose_max_adjustment_m`, `pose_jump_threshold_m`
  - Fusion flags: `enable_sparse_depth_fusion`, `fusion_depth_stride_px`, `fusion_max_points_per_frame`, `fusion_voxel_size_m`, `fusion_max_depth_m`, `fusion_include_feature_cloud`, `enable_fused_layout_assist`, `min_feature_cloud_for_layout`
  - Triangulation flags: `enable_track_ray_triangulation`, `triangulation_min_views`, `triangulation_max_condition`, `triangulation_max_ray_error_m`
  - Diagnostics flag: `save_reconstruction_report`

## Runtime toggles (environment)
- File: `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/main.py`
- Added:
  - `DECROOM_ENABLE_SFM_BA` (`on/off`)
  - `DECROOM_SFM_MATCHER` (`sequential|exhaustive`)
  - `DECROOM_SFM_COLMAP_BIN` (path/binary name)
  - `DECROOM_ENABLE_PYCOLMAP` (`on/off`)
  - `DECROOM_SFM_PREFER_CLI` (`on/off`)

## Design rationale
- Geometry and room layout must not be coupled to object detections.
- Depth-only fallback is fragile for low-feature objects; triangulated multi-view priors provide a stronger fallback.
- Instrumentation must quantify whether failure came from pose, coverage, fusion, tracking, or layout.

## Validation checklist (recommended)
1. Run upload with no detections and verify:
   - Room envelope still returned
   - `/tmp/decroom_tracking_report.json` exists (empty tracks)
   - `/tmp/decroom_reconstruction_report.json` exists
2. Run normal scan and verify:
   - `position_source` includes `feature_points` and/or `triangulated_rays` before `depth_backproject`
   - Triangulation accept/reject reasons are populated
   - Layout source and residuals are present

## Technical references
- Schonberger, J. L. & Frahm, J.-M. *Structure-from-Motion Revisited (COLMAP)*.
- Hartley, R. & Zisserman, A. *Multiple View Geometry in Computer Vision*.
- Triggs, B. et al. *Bundle Adjustment — A Modern Synthesis*.
- Fischler, M. & Bolles, R. *Random Sample Consensus (RANSAC)*.
- Curless, B. & Levoy, M. *A Volumetric Method for Building Complex Models from Range Images*.
- Newcombe, R. et al. *KinectFusion*.

Notes:
- Current implementation is a low-risk upgrade path using existing dependencies.
- Full global optimization beyond incremental SfM/BA (stronger global BA loops / dense MVS integration) is the next phase for maximum accuracy.
