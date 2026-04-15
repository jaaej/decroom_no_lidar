# Decroom Implementation Log & Reference Sheet

> Last updated: Session 26 (April 5, 2026)
> Total source lines: ~6,200 across 28 files

---

## Session 26: Reconstruction Chain Stabilization (SfM + FP + Tracking + Fallbacks)

Implemented from the stabilization plan with universal (room-agnostic) behavior.

### iOS capture contract and policy feedback
- Added per-shot metadata fields in upload payload:
  - `quality.{texture_score, blur_score, edge_density, matchability_seed}`
  - `capture_step_m`
  - `capture_yaw_delta_deg`
- Added dynamic pre-capture policy feedback:
  - overlap continuity scoring
  - large camera-step detection with intermediate-shot prompt
  - low-texture warning before capture commit
- Files:
  - `decroom_no_lidar/ContentView.swift`
  - `decroom_no_lidar/CoverageEngine.swift`
  - `decroom_no_lidar/ScanGuidanceOverlayView.swift`

### Backend SfM/Tracking/Clustering diagnostics and guardrails
- Added SfM match-graph prefilter and partial-apply policy controls.
- Added tracking fragmentation diagnostics + appearance-aware association terms.
- Added same-frame cluster-merge guardrails and small-object cross-frame preference.
- Added object uncertainty/support metadata in lifting diagnostics:
  - `position_uncertainty_m`
  - `orientation_uncertainty_deg`
  - `support_views`
- Files:
  - `backend/pipeline/sfm.py`
  - `backend/pipeline/tracking.py`
  - `backend/pipeline/lifting.py`
  - `backend/pipeline/reconstruction.py`
  - `backend/pipeline/config.py`
  - `backend/pipeline/metadata.py`

### Payload/schema plumbing
- Extended backend schema metadata model with optional uncertainty/support fields.
- Added debug-mode pass-through from pipeline payload assembly.
- Files:
  - `backend/schema.py`
  - `backend/pipeline/pipeline.py`

### Validation status
- Python syntax validation: pass (`python3 -m py_compile backend/main.py backend/schema.py backend/pipeline/*.py`).
- `xcodebuild` invocation starts but hangs without diagnostics in this environment; Swift compile validation should be run locally in Xcode UI.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Reference](#file-reference)
3. [Critical Bug Fixes (Sessions 1-8)](#critical-bug-fixes)
4. [Session-by-Session Changelog](#session-by-session-changelog)
5. [Known Gotchas & Invariants](#known-gotchas--invariants)
6. [Model Files](#model-files)
7. [Configuration Reference](#configuration-reference)

---

## Architecture Overview

```
iOS App (SwiftUI + ARKit + RealityKit + SceneKit)
    |
    | POST /api/upload (frames[] + metadata.json)
    v
FastAPI Backend (Python 3.10+)
    |
    ├── Step 1: Depth Estimation (Depth Anything V2, ONNX)
    ├── Step 2: Object Detection (YOLOv8-seg + YOLO-World, ONNX)
    ├── Step 3: Multi-Object Tracking (ByteTrack)
    ├── Step 4: Room Layout (RANSAC + PCA fallback)
    ├── Step 5: 3D Lifting (depth→3D position, clustering, wall snapping)
    └── Returns: RoomPayload JSON
            |
            v
    iOS App renders 3D room (SceneKit + Kenney USDZ models)
```

**AR Capture Flow:**
1. User opens Capture tab → ARKit session starts
2. 20 green sphere reticles appear around camera (8 level + 6 up + 6 down)
3. User aims at reticles → highlighted yellow when aimed → tap Capture
4. Reticle animates out (flash white → shrink → disappear)
5. CompassOverlayView shows coverage progress
6. After 8+ captures with good coverage → "Upload Now" enabled
7. Backend processes → returns RoomPayload → navigate to RoomPreviewPage

---

## File Reference

### iOS Swift Files (15 files, 2,580 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `ContentView.swift` | 867 | Main capture UI: AR view, guided capture logic, reticle aiming, async image capture, wireframe mesh overlay, RoomPreviewPage |
| `RoomSceneView.swift` | 532 | SceneKit 3D room renderer: walls, floor, USDZ furniture, fallback geometry, camera positioning |
| `ARReticleManager.swift` | 223 | RealityKit reticle manager: placement with yaw offset, aim detection, highlight animation, capture animation |
| `CoverageEngine.swift` | 460 | Angular coverage tracking: 24-bin grid, 20 reticle positions (8+6+6), guidance suggestions, yaw gap center |
| `ScanGuidanceOverlayView.swift` | 96 | Checklist overlay + directional chevron pointing at uncovered areas |
| `CompassOverlayView.swift` | 180 | 24-bin compass strip overlay: yaw bins, direction indicator with hysteresis, coverage badge |
| `RoomModels.swift` | 114 | Data models: RoomPayload, RoomEnvelope, Wall, RoomObject, ObjectPosition, ObjectMetadata |
| `UploadService.swift` | 111 | HTTP multipart upload to backend: frames + metadata JSON |
| `RoomCardView.swift` | 97 | Home screen card: isometric 3D preview via RoomSceneView + warm cream info bar |
| `HomeView.swift` | 94 | Home screen: blue/purple gradient, room card list, empty state |
| `CaptureFlowView.swift` | 72 | Capture navigation: shared RecordingProcessor, save room sheet, inline title |
| `RoomStore.swift` | 63 | Persistent room storage: UserDefaults-backed, SavedRoom model |
| `RecordingProcessor.swift` | 59 | State machine: idle → extracting → uploading → finished/failed |
| `MainTabView.swift` | 33 | Tab bar: Home + Capture tabs |
| `VideoRecorderView.swift` | 29 | Stub for future video recording feature |
| `decroom_no_lidarApp.swift` | 20 | App entry point: RoomStore environment object |

### Backend Python Files (13 files, 3,146 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `pipeline/lifting.py` | 1521 | 3D object lifting: depth→position, clustering, wall snapping, dimension estimation, diagnostic reporting |
| `pipeline/layout.py` | 480 | Room layout: RANSAC wall detection, PCA fallback, point cloud construction |
| `pipeline/pipeline.py` | 471 | Pipeline orchestrator: runs all steps, builds RoomPayload via schema.py |
| `pipeline/detection.py` | 438 | YOLOv8-seg COCO detection + COCO→PRD category mapping |
| `pipeline/depth.py` | 258 | Depth Anything V2 ONNX inference: metric and relative modes |
| `pipeline/detection_open_vocab.py` | 227 | YOLO-World open-vocabulary detection for non-COCO furniture |
| `pipeline/tracking.py` | 159 | ByteTrack multi-object tracking across frames |
| `pipeline/config.py` | 129 | Pipeline configuration: model paths, thresholds, merge distances, COCO→PRD map |
| `schema.py` | 97 | Pydantic-free data models with `to_dict()`: RoomPayload, RoomInfo, Wall, RoomObject |
| `pipeline/metadata.py` | 69 | Camera metadata parsing: intrinsics, poses, gravity |
| `main.py` | 67 | FastAPI app: /api/upload endpoint, /health, uvicorn runner |
| `pipeline/__init__.py` | 4 | Package init |
| `__init__.py` | 0 | Package init |

---

## Critical Bug Fixes (Sessions 1-7)

### Session 1-2: Foundation & Basic Detection

| # | Bug | Fix | File | Line(s) |
|---|-----|-----|------|---------|
| 1 | Backend wouldn't start | Added `uvicorn.run(app, host="0.0.0.0", port=8000)` | `main.py` | 65-67 |
| 2 | Empty pipeline results | Implemented full 5-step pipeline with proper orchestration | `pipeline.py` | All |

### Session 3: ARKit Image Orientation (CRITICAL)

| # | Bug | Fix | File | Line(s) |
|---|-----|-----|------|---------|
| 3 | Sideways images sent to YOLO (catastrophic) | Apply `.oriented(.right)` to CIImage before creating CGImage | `ContentView.swift` | `captureAsync()` |
| 4 | Intrinsics wrong after rotation | Swap: new_fx=old_fy, new_fy=old_fx, new_cx=old_cy, new_cy=old_width-old_cx | `ContentView.swift` | `captureAsync()` |

### Session 4: Detection & Depth Quality

| # | Bug | Fix | File | Line(s) |
|---|-----|-----|------|---------|
| 5 | Confidence threshold too high | Lowered from 0.25 to 0.15 | `config.py` | 83 |
| 6 | "wardrobe" not detected | Added COCO→PRD mappings (refrigerator→dresser, suitcase→dresser, etc.) | `config.py` | 40-56 |
| 7 | "dining table" lost | Mapped "dining table"→"desk" in COCO_TO_PRD | `config.py` | 46 |
| 8 | Blur detection on main thread | Moved to `Task.detached` background thread | `ContentView.swift` | `captureAsync()` |
| 9 | Darkness detection on main thread | Moved to `Task.detached` background thread | `ContentView.swift` | `captureAsync()` |

### Session 5: Depth & Layout Estimation

| # | Bug | Fix | File | Line(s) |
|---|-----|-----|------|---------|
| 10 | Reprojection filtering too aggressive | Changed to 0.5m absolute OR 30% relative | `pipeline.py` | Reprojection section |
| 11 | Non-visible points rejected | Kept non-visible points instead of filtering | `pipeline.py` | Reprojection section |
| 12 | Room min too small (2.5m) | Raised minimum to 3.0m | `layout.py` | Room extent clamping |
| 13 | Room percentile too narrow (5-95) | Changed to 2-98 percentile | `layout.py` | Point cloud extent |
| 14 | Wall RANSAC fails on similar normals | Added PCA fallback | `layout.py` | `_pca_fallback()` |

### Session 6: SceneKit Rendering & Object Clustering (CRITICAL)

| # | Bug | Fix | File | Line(s) |
|---|-----|-----|------|---------|
| 15 | **Cross-pattern walls (4 sessions!)** | Changed `atan2(dx, dz)` → `atan2(dz, dx)` for wall rotation | `RoomSceneView.swift` | 221 |
| 16 | Wall rotation direction wrong | Negate angle: `wallNode.eulerAngles.y = -angle` | `RoomSceneView.swift` | 222 |
| 17 | Furniture rotation wrong | Negate: `eulerAngles.y = -Float(obj.rotation_y)` | `RoomSceneView.swift` | 356 |
| 18 | Camera looking at wrong center | Compute center from actual wall coordinates, not assumed positive | `RoomSceneView.swift` | 141-151 |
| 19 | USDZ models wrong size | Measure bounding box, use median scale factor for uniform scaling | `RoomSceneView.swift` | 397-418 |
| 20 | USDZ models floating | Reposition after scaling so bottom sits on y=0 | `RoomSceneView.swift` | 422-424 |
| 21 | Object merge too tight (0.18m) | Increased to 0.8m default, 2.0m for beds, 1.5m for sofas | `config.py` | 100-111 |
| 22 | Duplicate beds after clustering | Added max instance limits per category | `lifting.py` | 26-41 |
| 23 | Size check fails for large furniture | Skip size similarity for bed, sofa, dresser, bookshelf | `lifting.py` | 44-45 |
| 24 | Pose yaw rotates walls incorrectly | Guard: skip if \|yaw\| > pi/4 | `layout.py` | Yaw guard |

### Session 7: AR Capture & UI Restoration

| # | Bug | Fix | File | Line(s) |
|---|-----|-----|------|---------|
| 25 | Reticle aim detection offset | Add initial camera yaw to reticle angles in `placeReticles()` | `ARReticleManager.swift` | 78-92 |
| 26 | aimThreshold too tight (0.25 rad) | Increased to 0.35 radians (~20 degrees) | `ARReticleManager.swift` | 42 |
| 27 | Camera shakes on capture | `captureAsync()` runs image processing on background thread | `ContentView.swift` | `captureAsync()` |
| 28 | Only 15/20 captures reachable | Fixed aim offset + larger threshold = all 20 reachable | `ARReticleManager.swift` | 42, 78-92 |
| 29 | Can capture without aiming | Button disabled when `aimedReticleIndex == nil` | `ContentView.swift` | 115 |
| 30 | No capture animation | 3-phase animation: flash white → shrink → disable | `ARReticleManager.swift` | 167-192 |
| 31 | No aimed highlight | Yellow glow + 1.4x scale when aimed | `ARReticleManager.swift` | 150-164 |
| 32 | CompassOverlayView was dead code | Restored full 24-bin compass with hysteresis direction indicator | `CompassOverlayView.swift` | All |
| 33 | Compass not wired in | Added to GuidedCaptureView ZStack | `ContentView.swift` | 58-69 |
| 34 | HomeView lost gradient design | Restored blue/purple gradient (#5B6BF5→#7B8BFF) | `HomeView.swift` | 13-16 |
| 35 | RoomCardView lost isometric previews | Restored cards with RoomSceneView + cream info bar | `RoomCardView.swift` | All |
| 36 | Preview not separate page | Added RoomPreviewPage as navigation destination | `ContentView.swift` | 320-356 |
| 37 | schema.py not integrated | Pipeline builds payload via `RoomPayload.to_dict()` | `pipeline.py` | Schema section |
| 38 | Dead `_apply_pose_orientation()` | Removed (was bug contributor for cross-pattern walls) | `layout.py` | Deleted |
| 39 | Unused COCO_TO_PRD import | Removed from lifting.py (only used in detection.py & config.py) | `lifting.py` | Import section |
| 40 | ONNX models missing (iCloud deletion) | Downloaded: depth_anything_v2_vitb (371MB), yolov8s-seg (45MB), yolov8s-worldv2 (48MB) | `models/` | N/A |
| 41 | Depth model name not in candidates | Added `depth_anything_v2_vitb.onnx` to auto-resolve list | `config.py` | 16 |
| 42 | Screen flash overlay missing | Added white overlay with opacity animation on capture | `ContentView.swift` | 82-87 |
| 43 | `import SwiftUI` at wrong position | Moved from bottom to top of file | `RoomModels.swift` | 1 |
| 44 | 24 reticles but only 20 max photos | Changed to 20 reticles (8 level + 6 up + 6 down) | `CoverageEngine.swift` | 14-17 |

### Session 8: Object Duplication & Orientation Fixes

| # | Bug | Fix | File | Line(s) |
|---|-----|-----|------|---------|
| 45 | 9 monitors from 1 real object | Raised merge distances: monitor 0.6→1.5m, default 0.8→1.5m, all types increased | `config.py` | 100-117 |
| 46 | Size similarity check too strict | Lowered merge_size_ratio 0.7→0.5, added ALL categories to SKIP_SIZE_CHECK | `lifting.py` | 27-48, `config.py` 117 |
| 47 | MAX_INSTANCES too generous | Reduced: monitor 4→2, chair 6→4, nightstand 4→2, table 3→2 | `lifting.py` | 27-43 |
| 48 | Furniture facing down / random rotation | ALL orientations snapped to cardinal (0°, 90°, 180°, 270°) via `_snap_to_cardinal()` | `lifting.py` | `_snap_to_cardinal()`, `_estimate_orientation()` |
| 49 | Wall angle not used for wall items when available | Wall types ALWAYS try wall_angle first, then view_angle, both snapped | `lifting.py` | `_estimate_orientation()` |
| 50 | No second-pass dedup after clustering | Added `_second_pass_dedup()` with 2x merge radius for same-category survivors | `lifting.py` | `_second_pass_dedup()` |
| 51 | IoU tracking fails across wide viewpoints | Added bbox center distance fallback + lowered IoU threshold 0.3→0.2 | `tracking.py` | `_bbox_center_dist()`, `track_detections()` |
| 52 | desk/table/chair had no explicit orientation | Added per-category orientation rules, all snapped to cardinal | `lifting.py` | `_estimate_orientation()` |
| 53 | Reprojection filter killed 4/5 valid objects | Skipped entirely for relative depth models (scale unreliable) | `pipeline.py` | `_reprojection_filter()` |
| 54 | Second-pass dedup merged genuinely different objects | Only activates when category count exceeds MAX_INSTANCES | `lifting.py` | `_second_pass_dedup()` |
| 55 | Monitor dim 1.3m deep → appeared face-down | Per-category DIMENSION_RANGES clamping (monitor depth max 0.3m) | `lifting.py` | `_sanity_check_dimensions()` |
| 56 | Room dims 2x too large (7.3m vs 3.5m actual) | Depth scale calibration using ARKit camera positions instead of hardcoded target | `pipeline.py` | `_scale_relative_depths()` |
| 57 | Reticle aim offset — highlights at screen edge, not center | Replaced angle-comparison with 3D dot-product from camera pos to reticle world pos | `ARReticleManager.swift` | `updateAimedReticle()` |

---

## Session-by-Session Changelog

### Session 1-2: Initial Build
- Bootstrapped iOS app: SwiftUI tab layout, AR session, basic capture
- Built FastAPI backend skeleton with upload endpoint
- Wrote first 4 pipeline modules: depth.py, detection.py, detection_open_vocab.py, tracking.py

### Session 3: ARKit Orientation Fix
- **Critical discovery**: ARKit `capturedImage` is ALWAYS landscape-right regardless of device orientation
- Applied `.oriented(.right)` to CIImage and swapped intrinsics (fx↔fy, cx↔cy)
- Without this fix, YOLO received sideways images → zero useful detections

### Session 4: Detection Tuning
- Lowered confidence thresholds to 0.15
- Added comprehensive COCO→PRD category mappings
- Moved blur/darkness quality checks to background thread

### Session 5: Layout & Depth
- Implemented RANSAC wall plane detection with PCA fallback
- Fixed reprojection filtering (was rejecting valid points)
- Raised room minimum from 2.5m to 3.0m, widened percentile range

### Session 6: SceneKit Rendering (Cross-Pattern Fix)
- **Root cause found**: `atan2(dx, dz)` instead of `atan2(dz, dx)` for wall rotation
  - SCNBox width extends along local X axis
  - To align with direction vector (dx, dz): angle = `atan2(dz, dx)` (from +X axis)
  - Must negate for SceneKit: `eulerAngles.y = -angle`
- Fixed camera centering from actual wall coordinates
- Added USDZ model scaling (median of per-axis scales) and y=0 repositioning
- Tuned object merge distances per category
- Added post-clustering instance limits

### Session 7: AR Capture & Disaster Recovery
- **iCloud eviction destroyed files** → restored CompassOverlayView, HomeView gradient, RoomCardView
- Fixed reticle aim detection by offsetting angles by initial camera yaw
- Increased aimThreshold from 0.25 to 0.35 rad
- Added 3-phase capture animation (flash → shrink → hide)
- Made preview a separate navigation page (RoomPreviewPage)
- Re-downloaded all 3 ONNX models (depth + detector + open-vocab)
- Integrated schema.py back into pipeline

### Session 8: Object Duplication, Orientation, Depth Scale, Reticle Aiming (CRITICAL)
- **Root cause of 9 monitors**: Merge distance was 0.6m but depth noise shifts same object 0.5-1.0m across frames. Each unique position became a new cluster.
  - Raised ALL merge distances: default 0.8→1.5m, monitor 0.6→1.5m, bed 2.0→2.5m, etc.
  - Added ALL categories to SKIP_SIZE_CHECK (depth noise makes dimensions unreliable across viewpoints)
  - Lowered object_merge_size_ratio from 0.7 to 0.5
  - Reduced MAX_INSTANCES: monitor 4→2, chair 6→4, nightstand 4→2
  - Second-pass dedup: now only activates when category exceeds MAX_INSTANCES (was wrongly merging all same-category objects)
- **Root cause of "facing down" furniture**: `_viewing_angle()` returned raw atan2 values without snapping, producing arbitrary angles like 1.2 rad or -0.7 rad
  - Added `_snap_to_cardinal()` function — ALL furniture snaps to nearest 0°, 90°, 180°, or 270°
  - Most residential furniture IS axis-aligned; continuous angles from noisy depth are always wrong
  - Each category now has explicit orientation rules (wall items use wall angle, chairs face desks, beds align to walls, sofas face center)
- **Reprojection filter was killing valid objects** (kept 1/5 in real test): Depth scale from relative model is estimated and unreliable. Projected depth vs depth-map disagreement caused all objects to fail.
  - Reprojection filter now SKIPPED entirely for non-metric depth models (`depth_model_type != "dav2_metric"`)
- **Monitor appeared face-down due to dimension distortion**: Depth noise made monitor 1.3m deep (should be 0.2m), stretching the USDZ model into a flat slab.
  - Added per-category DIMENSION_RANGES clamping in `_sanity_check_dimensions()` (e.g. monitor depth max 0.3m)
- **Room dimensions 2x too large** (7.3m×6.4m×4.0m vs actual 3.53m×3.05m×2.36m): `_scale_relative_depths()` used hardcoded `target_depth = 2.5m` assumption. For small rooms where median depth is ~1.2m, this doubled everything.
  - Rewrote `_scale_relative_depths()` to use ARKit camera positions (real meters from VIO/SLAM) to calibrate the depth scale
- **Reticle aiming was offset/inconsistent**: Used abstract yaw/pitch angle comparison which broke when camera moved from initial anchor position. Reticle at screen edge would highlight, reticle at center would not.
  - Replaced angle comparison with 3D dot-product: compute direction from current camera world position to each reticle's world position, compare against camera forward vector
  - Threshold reduced from 0.35 to 0.25 rad (~14°) — more precise with exact 3D calculation
  - Fixed `updateState()` which had mismatched grid-cell vs entity indices (was silently failing)
- **Tracking improvement**: Added bbox center distance as secondary matching criterion
  - Same object across wide viewpoints has low IoU but similar center position
  - Lowered IoU threshold from 0.3 to 0.2, added center_dist_threshold = 0.5
- **Swift concurrency fix**: Added `nonisolated` to 4 static methods on `GuidedARFrameRecorder` that are pure computation called from `Task.detached`

### Session 9: Diagnostic Logging, Wireframe Mesh, Directional Guidance

**Diagnostic Logging (per-object placement diagnostics):**
- Added `return_diag` parameter to `associate_fp_to_detection()` in `geometry.py` — returns `fp_visible` and `fp_behind` counts
- Enriched `per_frame_fp` in cluster report from tuples to dicts: `{frame, shot_fp, in_bbox, bbox, image_size, intrinsics_image_size, fp_visible, fp_behind}`
- Added `_detection_details` array per object: `{frame_idx, bbox, score, category, image_size, tracking_quality}`
- Added `depth_median`, `depth_mad`, `depth_sample_count`, `tracking_quality_per_frame` to report
- Propagated `tracking_quality` from `ShotMetadata` through `_lift_single_track()`
- Updated `_merge_cluster()` to concatenate `_detection_details` from all cluster members
- Report file: `/tmp/decroom_cluster_report.json`

**Wireframe Mesh Overlay (capture-region visualization):**
- First attempt: Solid filled triangles with `UnlitMaterial(0.15 opacity)` → appeared as opaque blue blob
- Second attempt: `arView.debugOptions = [.showSceneUnderstanding]` → showed ALL mesh everywhere, not just captured regions
- **Final approach**: Per-capture `snapshotMesh()` with wireframe line geometry
  - Extract unique edges from each `ARMeshAnchor`'s triangle mesh
  - Build thin 2mm quad strips (2 triangles per edge) to simulate wireframe lines
  - `UnlitMaterial(cyan, 40% opacity)` on quads
  - Cap at 8000 edges per anchor for performance
  - Only includes mesh anchors visible at time of capture; builds incrementally

**Directional Guidance (scan guidance overlay):**
- Created `ScanGuidanceOverlayView.swift`: 4-item checklist + directional chevron
  - Checklist: photo count, walked around, multiple angles, captured detail
  - Chevron: points at largest uncovered yaw arc
- Added `largestGapCenterYaw: Float?` to `CoverageEngine` — center of biggest uncovered arc
- **Arrow ping-pong fix**: Initially arrow always showed when coverage incomplete. Each capture split the biggest gap, flipping the arrow to point the other way. Fixed: chevron only appears when `largestGap > π/2` (90°), with 14° dead zone

| # | Bug | Fix | File |
|---|-----|-----|------|
| 58 | Solid blue blob mesh overlay | Switched from filled triangles to wireframe edge quads (2mm wide) | `ContentView.swift` |
| 59 | `debugOptions = [.showSceneUnderstanding]` shows all mesh | Reverted to per-capture `snapshotMesh()` approach | `ContentView.swift` |
| 60 | Arrow ping-pongs between left/right | Only show chevron when largest gap > 90°, dead zone 14° | `ScanGuidanceOverlayView.swift` |
| 61 | No directional guidance for scanning | Added `largestGapCenterYaw` to CoverageEngine, chevron at screen edge | `CoverageEngine.swift`, `ScanGuidanceOverlayView.swift` |
| 62 | Cluster report missing detection details | Added bbox, image_size, intrinsics, fp_visible/behind, tracking_quality | `lifting.py`, `geometry.py` |
| 63 | Backend doesn't hot-reload | Must kill & restart `python main.py` for code changes | `main.py` |

---

## Known Gotchas & Invariants

### ARKit
- `capturedImage` is **always landscape-right** (sensor orientation). Must apply `.oriented(.right)`.
- Intrinsics must be swapped after rotation: `new_fx=old_fy, new_fy=old_fx, new_cx=old_cy, new_cy=old_width-old_cx`
- Camera forward direction: `-transform.columns.2` (negative Z in camera space)
- Camera position: `transform.columns.3`
- Yaw: `atan2(fwdX, fwdZ)`, normalized to [0, 2pi)

### SceneKit
- `SCNBox` width extends along **local X axis**
- Wall rotation: `atan2(dz, dx)` — angle from +X to direction vector. **NOT** `atan2(dx, dz)`.
- `eulerAngles.y`: positive = clockwise from above. Negate the math angle: `eulerAngles.y = -angle`
- Backend `rotation_y` uses CCW positive (math convention). Negate for SceneKit: `eulerAngles.y = -Float(obj.rotation_y)`
- Camera must be centered on actual wall geometry center, not assumed positive coordinates

### RealityKit
- Reticle yaw angles from CoverageEngine are abstract (0 to 2pi). Must offset by initial camera yaw in `placeReticles()`.
- `AnchorEntity(world:)` places in world coordinates — position at camera origin, reticles on sphere around it.
- **Reticle aiming MUST use 3D world positions, NOT abstract angles.** When the camera moves from the anchor position, stored yaw/pitch no longer matches the actual direction to reticles. Use dot product of camera forward vs (reticle world pos - camera world pos).
- `entity.position` is local to parent anchor. World position = `anchor.position(relativeTo: nil) + entity.position`.
- **RealityKit has NO native wireframe mode.** To show wireframe mesh: extract triangle edges, build thin quad strips, apply UnlitMaterial.
- `debugOptions = [.showSceneUnderstanding]` shows ALL reconstructed mesh globally — NOT suitable for per-region visualization.
- Per-capture mesh: iterate `frame.anchors` as `ARMeshAnchor`, snapshot only those present at capture time. Track IDs in a Set to update (not duplicate) on subsequent captures.

### Backend Pipeline
- Depth Anything V2 relative model needs depth scaling via camera baselines (auto-handled)
- RANSAC can fail when planes have similar normals → always needs PCA fallback
- **Object merge distances must be LARGE** (1.5m+ default) — depth noise in guided capture shifts same object 0.5-1.0m across frames
- **Skip size similarity for ALL categories** — depth noise distorts dimensions differently per viewpoint
- **All orientations MUST snap to cardinal** (0, 90, 180, 270 deg) — continuous angles from noisy depth are ALWAYS wrong for residential furniture
- Second-pass dedup at 2x merge radius catches near-miss clusters
- IoU tracking alone fails for guided capture (wide viewpoints) — center distance fallback needed
- Pose yaw guard: if |yaw| > pi/4, skip rotation (noisy pose data)
- MAX_INSTANCES should be conservative (monitor=2, not 4) — better to miss a real duplicate than show 9 fake ones

### Project Management
- **NEVER move the project directory** — causes catastrophic Xcode reference breakage
- iCloud eviction can silently replace binary files with stubs — verify model file sizes
- Kenney USDZ models are CC0 from kenney.nl/assets/furniture-kit

---

## Model Files

| Model | File | Size | Purpose |
|-------|------|------|---------|
| Depth Anything V2 | `models/depth/depth_anything_v2_vitb.onnx` | 371 MB | Relative monocular depth estimation |
| YOLOv8s-seg | `models/detector/yolov8s-seg.onnx` | 45 MB | COCO 80-class instance segmentation |
| YOLO-World v2 | `models/detector/yolov8s-worldv2.onnx` | 48 MB | Open-vocabulary furniture detection |

### Kenney USDZ Models (13 files, CC0 license)

`kenney_bed.usdz`, `kenney_bookshelf.usdz`, `kenney_chair.usdz`, `kenney_desk.usdz`,
`kenney_dresser.usdz`, `kenney_lamp_desk.usdz`, `kenney_lamp_floor.usdz`,
`kenney_monitor.usdz`, `kenney_nightstand.usdz`, `kenney_plant.usdz`,
`kenney_rug.usdz`, `kenney_sofa.usdz`, `kenney_table.usdz`

---

## Configuration Reference

### CoverageEngine (iOS)
| Parameter | Value | Notes |
|-----------|-------|-------|
| `yawBins` | 24 | Fine grid for tracking (15 deg each) |
| `pitchBins` | 3 | Level, up (+25 deg), down (-25 deg) |
| `maximumShots` | 20 | Matches reticle count |
| `minimumShots` | 8 | Minimum for "good coverage" |
| `maxAllowedYawGap` | 90 deg | Largest gap between consecutive captures |

### ARReticleManager (iOS) — Session 8 updated
| Parameter | Value | Notes |
|-----------|-------|-------|
| `reticleDistance` | 2.0m | Distance from camera to place spheres |
| `aimThreshold` | **0.25 rad** | **~14 deg** (was 0.35/~20°). Now uses exact 3D dot-product, so tighter threshold works better |
| Sphere radius | 0.07m | Visual size of each reticle sphere |
| Aimed highlight | Yellow, 1.4x scale | Visual feedback when aimed |
| Aim method | **3D dot-product** | Direction from camera position to reticle world position vs camera forward vector (was abstract yaw/pitch angle comparison) |

### Reticle Layout (20 positions)
| Tier | Pitch | Count | Yaw Spacing |
|------|-------|-------|-------------|
| Level | 0 deg | 8 | Every 45 deg |
| Up | +25 deg | 6 | Every 60 deg |
| Down | -25 deg | 6 | Every 60 deg |

### PipelineConfig (Backend) — Session 8 updated
| Parameter | Value | Notes |
|-----------|-------|-------|
| `detection_min_confidence` | 0.15 | Lowered from 0.25 (Session 4) |
| `object_merge_distance_m` | **1.5m** | Default (was 0.8m, raised in Session 8) |
| `object_merge_distance_by_type.bed` | **2.5m** | Was 2.0m |
| `object_merge_distance_by_type.sofa` | **2.0m** | Was 1.5m |
| `object_merge_distance_by_type.desk` | **1.8m** | Was 1.0m |
| `object_merge_distance_by_type.monitor` | **1.5m** | Was 0.6m (root cause of 9-monitor bug) |
| `object_merge_distance_by_type.chair` | **1.0m** | Was 0.5m |
| `object_merge_size_ratio` | **0.5** | Was 0.7 (too strict with depth noise) |
| `SKIP_SIZE_CHECK` | **All categories** | Was only bed/sofa/dresser/bookshelf |
| Room minimum | 3.0m | Raised from 2.5m |
| Room percentile | 2-98 | Widened from 5-95 |
| Second-pass dedup radius | 2x merge distance | New in Session 8 |
| Orientation snapping | Cardinal (0/90/180/270 deg) | New in Session 8 |

### COCO to PRD Category Mapping
```
bed → bed          chair → chair       bench → chair
couch → sofa       sofa → sofa         dining table → desk
potted plant → plant  tv → monitor     laptop → monitor
book → bookshelf   vase → plant        clock → poster
refrigerator → dresser  suitcase → dresser  backpack → dresser
```

### Instance Limits — REMOVED (Session 8 late)
No hard instance caps. Rooms vary widely (dorms, shared spaces, large homes).
Deduplication relies solely on merge distance clustering — if two detections
are further apart than the merge distance, they are treated as distinct objects.

### Diagnostic Debug Output (Session 9)
All debug files written to `/tmp/`:
| File | Contents |
|------|----------|
| `decroom_cluster_report.json` | Per-object: position, detections (bbox, score, category, image_size, tracking_quality), per-frame FP (fp_visible, fp_behind, intrinsics_image_size), depth stats (median, MAD, sample_count) |
| `decroom_payload.json` | Final RoomPayload sent to iOS (room + objects) |
| `decroom_tracking_report.json` | ByteTrack output: track_id, category, n_detections, frames, world_pos |
| `decroom_wall_report.json` | Room dimensions, wall corners, layout_source |

Key diagnostic questions answered by cluster report:
1. **What was detected?** `detections[].bbox` + `image_size` + `score` + `category`
2. **How was 3D position computed?** `position_source` (feature_points vs depth_backproject) + `fp_count`/`fp_variance` + `depth_median`/`depth_mad`/`depth_sample_count`
3. **Was tracking reliable?** `tracking_quality_per_frame` + `per_frame_fp[].fp_visible`/`fp_behind`

---

## Session 14: Reconstruction Backend Upgrade (Geometry-First)

### Core pipeline changes
- Removed detection-dependent early return from `backend/pipeline/pipeline.py`.
  - Room geometry/layout now runs even when detections are empty.
- Added `backend/pipeline/reconstruction.py` with:
  - `refine_camera_poses()` (optional pose translation smoothing + diagnostics)
  - `fuse_multiview_geometry()` (sparse multi-view depth fusion)
  - `triangulate_tracks_from_rays()` (multi-view object-center priors)
  - `build_reconstruction_report()` + `save_reconstruction_report()`

### Object placement robustness
- Extended `backend/pipeline/lifting.py`:
  - New optional `triangulation_priors` input to `lift_objects()`
  - New fallback source `position_source = "triangulated_rays"` before depth backprojection
  - Added triangulation diagnostics fields on objects (`triangulation_views`, `triangulation_median_ray_error_m`)

### Tracking/report continuity
- Updated `backend/pipeline/tracking.py`:
  - Writes empty `/tmp/decroom_tracking_report.json` when no detections are found.

### New config options
- Added reconstruction tuning fields in `backend/pipeline/config.py`:
  - pose smoothing controls
  - sparse fusion controls
  - fused-layout assist controls
  - track ray-triangulation controls
  - reconstruction report toggle

### New diagnostics artifact
- Added `/tmp/decroom_reconstruction_report.json` with:
  - shot coverage summary
  - pose diagnostics
  - fusion diagnostics
  - track triangulation accept/reject diagnostics
  - layout diagnostics and run summary

### Additional documentation
- Added architecture + references doc:
  - `docs/RECONSTRUCTION_BACKEND_UPGRADE.md`

### Session 15: Optional Full SfM/BA Integration (COLMAP)
- Added new module `backend/pipeline/sfm.py`:
  - Materializes upload frames to image files for SfM.
  - Runs COLMAP CLI pipeline (`feature_extractor`, `matcher`, `mapper`, `model_converter`).
  - Parses reconstructed camera poses.
  - Aligns SfM poses to ARKit metric frame using Umeyama similarity transform.
  - Applies aligned camera poses back into `ShotMetadata.pose`.
  - Non-blocking fallback when COLMAP is unavailable or quality thresholds are not met.
- Integrated SfM stage into `backend/pipeline/pipeline.py` as Step `-1` (before pose refinement).
- Extended reconstruction report to include `sfm_ba` diagnostics.
- Added SfM/BA config options in `backend/pipeline/config.py`.
- Added backend env toggles in `backend/main.py`:
  - `DECROOM_ENABLE_SFM_BA`
  - `DECROOM_SFM_MATCHER`
  - `DECROOM_SFM_COLMAP_BIN`

### Session 16: pycolmap Backend Path + Local Install
- Installed `pycolmap` in both:
  - project venv: `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/.venv-sfm`
  - system Python (used by backend runtime)
- Added dual SfM backend logic in `backend/pipeline/sfm.py`:
  - prefers COLMAP CLI when available
  - falls back to `pycolmap` when CLI is not installed (or when preferred)
- Added frame materialization fallback path (Pillow/OpenCV) for SfM input image export.
- Added config toggles:
  - `enable_pycolmap_backend`
  - `sfm_prefer_colmap_cli`
- Added env toggles in `backend/main.py`:
  - `DECROOM_ENABLE_PYCOLMAP`
  - `DECROOM_SFM_PREFER_CLI`
- Homebrew/COLMAP CLI install was blocked by missing admin privileges on this macOS account.

### Session 17: Placement Stability Hardening (Pose-Jump Rejection + Stricter 2D Fallback)
- Added pose-jump rejection controls in `backend/pipeline/config.py`:
  - `enable_pose_jump_rejection`
  - `pose_jump_rejection_threshold_m`
  - `pose_jump_rejection_min_ratio`
  - `pose_jump_rejection_min_jumps`
  - `pose_jump_rejection_min_keep_frames`
- Added backend env parsing in `backend/main.py`:
  - `DECROOM_ENABLE_POSE_JUMP_REJECTION`
  - `DECROOM_POSE_JUMP_REJECTION_THRESHOLD_M`
  - `DECROOM_POSE_JUMP_REJECTION_MIN_RATIO`
  - `DECROOM_POSE_JUMP_REJECTION_MIN_JUMPS`
- Added `_reject_pose_jump_frames(...)` in `backend/pipeline/pipeline.py`:
  - Runs after pose refinement and before depth/detection/tracking/layout.
  - Detects discontinuities from pose step outliers (absolute + relative threshold).
  - Keeps the largest contiguous pose-consistent segment when discontinuities are severe.
  - Writes explicit rejection diagnostics into `pose_refinement` report fields.
- Tightened 2D fallback matching in `backend/pipeline/tracking.py`:
  - Added bbox area-ratio gating and frame-gap gating.
  - When a 3D match exists but fails distance threshold, 2D fallback now requires
    stricter IoU + center-distance + size consistency.
  - Added richer assignment diagnostics (`center_dist`, `area_ratio`, `frame_gap`,
    fallback thresholds/mode) to `/tmp/decroom_tracking_report.json`.

### Session 18: Pose-Jump Rejection Softening + Context Rescue Diagnostics
- Updated pose-jump rejection in `backend/pipeline/pipeline.py` to reduce false drops:
  - Uses per-step timestamp-derived speed gating.
  - Rejects jump pairs only when step exceeds adaptive threshold and is either:
    - implausibly fast (`speed > pose_jump_rejection_max_speed_mps`), or
    - absolutely extreme (`step > pose_jump_rejection_hard_threshold_m`).
- Added new tuning fields in `backend/pipeline/config.py`:
  - `pose_jump_rejection_max_speed_mps`
  - `pose_jump_rejection_hard_threshold_m`
- Added env parsing in `backend/main.py`:
  - `DECROOM_POSE_JUMP_REJECTION_MAX_SPEED_MPS`
  - `DECROOM_POSE_JUMP_REJECTION_HARD_THRESHOLD_M`
- Extended context rescue in `backend/pipeline/lifting.py`:
  - Keeps rescued sparse objects through stability filtering via `_rescue_context` marker.
  - Writes rescue diagnostics to `/tmp/decroom_cluster_report.json`:
    - summary count (`rescued_context_total`)
    - rescue list (`rescued_context`)
    - per-object fields (`rescue_context`, `rescue_reason`)

### Session 19: Default Runtime Policy + Orientation/Survival Robustness
- Changed defaults in `backend/pipeline/config.py`:
  - `enable_pose_jump_rejection = False` (off by default)
  - `enable_sfm_ba = True` (on by default)
  - Added axis-snap safety thresholds:
    - `axis_snap_min_camera_extent_m`
    - `axis_snap_min_camera_extent_ratio`
- Updated `backend/pipeline/pipeline.py`:
  - Computes retained camera-path extent from layout diagnostics.
  - Disables axis-snap input to lifting (`axis_confidence_for_lifting = 0`) when
    camera extent is too small relative to room scale.
  - Records run-level axis-snap gating fields on layout diagnostics:
    - `axis_snap_enabled`
    - `axis_snap_disable_reason`
    - `axis_snap_camera_extent_diag_m`
    - `axis_snap_camera_extent_ratio`
    - `axis_confidence_for_lifting`
- Updated `backend/pipeline/lifting.py` observation filtering:
  - Added category-aware single-view survival rule for `monitor`/`chair` when
    feature-point support is strong (`single_view_fp_strong`).
  - Added per-object diagnostics fields in cluster report:
    - `obs_required`
    - `obs_rule`
- Updated `backend/pipeline/reconstruction.py` report:
  - Includes `axis_confidence_for_lifting` and axis-snap gating status/reason in
    summary/layout sections.

### Session 20: Tracking Fragmentation + Orientation + SfM Robustness
- Updated `backend/pipeline/lifting.py`:
  - Stability filter now treats `position_source="triangulated_rays"` as a
    dedicated multi-view source (no depth-MAD gating).
  - Added fragmentation rescue path for high-singleton runs to recover a small
    number of likely true positives (`fragmentation_singleton_recovery`).
  - Tightened desk context rescue:
    - monitor anchors must have stronger support (`n_detections` or `fp_count`)
    - monitor cluster compactness checks (radius + pairwise distance) before
      inferring a desk proxy.
  - Orientation logic now avoids axis snapping for weak sources:
    - `camera_last_resort` uses unsnapped angle
    - when `pca_ambiguous` is observed, downstream fallback orientation skips
      axis snapping.
  - Cluster report summary now includes:
    - `triangulated_positioned`
    - `context_positioned`
- Updated `backend/pipeline/sfm.py`:
  - pycolmap path now performs a recovery attempt when primary setup fails:
    - retry with `SIMPLE_RADIAL` + `exhaustive` matching.
  - Added attempt-level fallback diagnostics in `fallback_reason` for easier
    SfM failure triage.

### Session 21: Constrained Room Fit (Best-Case Upgrade #1)
- Updated `backend/pipeline/layout.py`:
  - Replaced fixed FP room inflation (`extent * 1.25 + 0.3`) with a constrained
    extent solver on both:
    - FP-PCA primary path (`feature_points_pca`)
    - FP-percentile fallback path (`feature_points_percentile`)
  - New constrained fit combines:
    - observed feature-point span (+ small margin),
    - camera-path containment (+ configurable camera wall padding),
    - capped growth ratio to prevent oversized room envelopes.
  - Added center stabilization by blending FP center with camera-path median
    center when camera coverage is sufficient.
  - Added diagnostics fields to layout report:
    - `room_fit_mode`
    - `room_fit_camera_pose_count`
    - `room_fit_observed_extent`
    - `room_fit_camera_extent`
    - `room_fit_final_extent`
    - `room_fit_growth`
    - `room_fit_growth_capped`
- Updated `backend/pipeline/config.py` with tunables:
  - `layout_fp_margin_m`
  - `layout_camera_wall_pad_m`
  - `layout_camera_min_pad_m`
  - `layout_extent_max_growth_ratio`
  - `layout_center_blend_alpha`
- Updated `backend/main.py` env support:
  - `DECROOM_LAYOUT_FP_MARGIN_M`
  - `DECROOM_LAYOUT_CAMERA_WALL_PAD_M`
  - `DECROOM_LAYOUT_CAMERA_MIN_PAD_M`
  - `DECROOM_LAYOUT_EXTENT_MAX_GROWTH_RATIO`
  - `DECROOM_LAYOUT_CENTER_BLEND_ALPHA`

### Session 22: Tracklet Stitching + SfM Keyframe Prefilter (Best-Case #2/#3)
- Updated `backend/pipeline/tracking.py`:
  - Added post-tracking `tracklet stitching` pass to recover fragmented tracks.
  - Stitch logic prefers 3D continuity and uses strict 2D endpoint fallback only
    for short gaps.
  - Added stitch diagnostics into `/tmp/decroom_tracking_report.json`:
    - `summary.n_stitches`
    - `stitches[]` with source/target ids, gap, score, reason, and metrics.
- Updated `backend/pipeline/pipeline.py`:
  - Tracking now receives stitch settings from config.
- Updated `backend/pipeline/config.py`:
  - Added stitch controls:
    - `enable_tracklet_stitching`
    - `tracklet_stitch_max_gap_frames`
    - `tracklet_stitch_world_dist_factor`
    - `tracklet_stitch_min_score`
    - `tracklet_stitch_max_passes`
  - Added SfM prefilter controls:
    - `sfm_enable_keyframe_prefilter`
    - `sfm_keyframe_min_translation_m`
    - `sfm_keyframe_min_yaw_deg`
    - `sfm_keyframe_force_stride`
    - `sfm_keyframe_sharpness_min`
    - `sfm_keyframe_max_frames`
    - `sfm_keyframe_min_frames`
- Updated `backend/pipeline/sfm.py`:
  - Added keyframe selection stage before SfM:
    - sharpness filtering (blur suppression),
    - motion/yaw novelty gating,
    - minimum/maximum selected frame constraints.
  - Added keyframe diagnostics fields in SfM report (counts + sharpness means).
  - Added COLMAP CLI retry strategy mirroring pycolmap fallback:
    - primary (`cfg` camera/matcher),
    - fallback (`SIMPLE_RADIAL + exhaustive`).
  - Added attempt fallback from prefiltered set to full frame list when needed.
- Updated `backend/main.py` env support:
  - `DECROOM_ENABLE_TRACKLET_STITCHING`
  - `DECROOM_TRACKLET_STITCH_MAX_GAP_FRAMES`
  - `DECROOM_TRACKLET_STITCH_WORLD_DIST_FACTOR`
  - `DECROOM_TRACKLET_STITCH_MIN_SCORE`
  - `DECROOM_TRACKLET_STITCH_MAX_PASSES`
  - `DECROOM_SFM_ENABLE_KEYFRAME_PREFILTER`
  - `DECROOM_SFM_KEYFRAME_MIN_TRANSLATION_M`
  - `DECROOM_SFM_KEYFRAME_MIN_YAW_DEG`
  - `DECROOM_SFM_KEYFRAME_FORCE_STRIDE`
  - `DECROOM_SFM_KEYFRAME_SHARPNESS_MIN`
  - `DECROOM_SFM_KEYFRAME_MAX_FRAMES`
  - `DECROOM_SFM_KEYFRAME_MIN_FRAMES`

### Session 23: COLMAP 3.13 CLI Option Compatibility Fix
- Updated `backend/pipeline/sfm.py` COLMAP CLI arguments to match `colmap 3.13.0`:
  - `feature_extractor`:
    - replaced `--SiftExtraction.use_gpu` with `--FeatureExtraction.use_gpu`
  - `sequential_matcher` / `exhaustive_matcher`:
    - replaced `--SiftMatching.use_gpu` with `--FeatureMatching.use_gpu`
- Result:
  - Fixes SfM hard-fail caused by unrecognized CLI options and allows CLI SfM
    to proceed on Homebrew COLMAP 3.13 builds.

### Session 24: Scan Quality Gate Soft-Fail Mode
- Updated `backend/pipeline/pipeline.py` quality-gate handling:
  - Large depth-outlier gate now defaults to **soft-fail**:
    - does not raise `RuntimeError` / HTTP 422 by default
    - appends warning payload under top-level `warnings[]`
  - Preserves strict mode via config flag:
    - when `scan_quality_gate_hard_fail=true`, behavior remains hard-fail.
- Updated `backend/pipeline/config.py`:
  - Added `scan_quality_gate_hard_fail` (default `False`).
- Updated `backend/main.py` env support:
  - `DECROOM_SCAN_QUALITY_GATE_HARD_FAIL`

### Session 25: Duplicate-Track Merge Controls + SfM Best-Candidate Selection
- Updated `backend/pipeline/tracking.py`:
  - Added duplicate-overlap merge pass (same-category concurrent track collapse)
    before stitch pass.
  - Added diagnostics in `/tmp/decroom_tracking_report.json`:
    - `summary.n_duplicate_merges`
    - `duplicate_merges[]`
- Updated `backend/pipeline/config.py`:
  - Added duplicate merge knobs:
    - `enable_track_duplicate_merge`
    - `track_duplicate_iou_min`
    - `track_duplicate_center_max`
    - `track_duplicate_world_dist_factor`
    - `track_duplicate_min_score`
    - `track_duplicate_max_passes`
- Updated `backend/pipeline/pipeline.py`:
  - Wired duplicate merge settings into `track_detections(...)`.
- Updated `backend/pipeline/sfm.py`:
  - SfM no longer stops at the first successful backend/set attempt.
  - Collects candidate reconstructions (prefilter/full, CLI/pycolmap), selects
    best by registration ratio/registered images, and records `backend_attempts`.
  - Prevents low-ratio prefilter successes from blocking full-frame retries.
- Updated `backend/main.py` env support:
  - `DECROOM_ENABLE_TRACK_DUPLICATE_MERGE`
  - `DECROOM_TRACK_DUPLICATE_IOU_MIN`
  - `DECROOM_TRACK_DUPLICATE_CENTER_MAX`
  - `DECROOM_TRACK_DUPLICATE_WORLD_DIST_FACTOR`
  - `DECROOM_TRACK_DUPLICATE_MIN_SCORE`
  - `DECROOM_TRACK_DUPLICATE_MAX_PASSES`
  - `DECROOM_SFM_REQUIRED_REGISTRATION_RATIO`
  - `DECROOM_SFM_REQUIRED_ALIGNED_POSES`

### Session 26: SfM Default Runtime Tuning (No Per-Run Env Required)
- Updated `backend/pipeline/config.py` SfM defaults:
  - `sfm_colmap_binary` now auto-resolves to `/opt/homebrew/bin/colmap` when
    available, then falls back to `colmap` in PATH.
  - `sfm_required_registration_ratio` default lowered to `0.25`.
  - `sfm_required_aligned_poses` default lowered to `4`.
- Result:
  - You no longer need to export SfM env vars each run for standard local usage.
  - Env vars still override defaults when needed for stricter validation.

### Session 27: Orientation Stability + Context Desk Recovery + Clutter Suppression
- Updated `backend/pipeline/lifting.py` orientation logic:
  - Added category-aware near-wall axis snap thresholds for `monitor`, `desk`,
    and `chair` so slight axis-confidence shortfalls do not skip snapping.
  - PCA orientation now accepts a per-object snap threshold and records this in
    `orient_trace` (`min=` diagnostics).
- Updated singleton/fragmentation filtering in `backend/pipeline/lifting.py`:
  - Tightened single-view keep thresholds for monitors (`score/fp`).
  - Tightened fragmentation rescue thresholds, especially for `monitor` and
    `dresser`, to suppress low-quality singleton clutter restores.
- Updated context rescue in `backend/pipeline/lifting.py`:
  - Added desk inference fallback from `monitor + chair` tabletop geometry when
    `desk/table` class is missed.
  - New rescue reason: `desk_inferred_from_monitor_chair_geometry`.

### Session 28: Metric Depth FP Calibration (Applied, Guarded)
- Updated `backend/pipeline/pipeline.py` metric-depth path:
  - Added `_apply_metric_fp_scale_calibration(...)` after `_metric_depth_sanity(...)`.
  - When diagnostics are stable enough, metric depth is corrected by:
    - `depth_calibrated = depth_model * scale_ratio_p50`
  - Added guardrails before applying:
    - minimum ratio count
    - scale-factor bounds
    - max IQR
    - max p90-p10 spread
  - Emits diagnostics:
    - `metric_scale_applied`
    - `metric_scale_factor`
    - `metric_scale_reason`
    - `metric_scale_maps_updated`
    - calibrated ratio quantiles
    - `metric_depth_reliability_raw` and calibrated reliability
  - Sets `scale_method = metric_fp_calibrated` and `scale_factor` when applied.
- Updated `backend/pipeline/config.py`:
  - `enable_metric_fp_scale_calibration`
  - `metric_fp_scale_min_ratios`
  - `metric_fp_scale_min_factor`
  - `metric_fp_scale_max_factor`
  - `metric_fp_scale_max_iqr`
  - `metric_fp_scale_max_spread`
- Updated `backend/main.py` env support:
  - `DECROOM_ENABLE_METRIC_FP_SCALE_CALIBRATION`
  - `DECROOM_METRIC_FP_SCALE_MIN_RATIOS`
  - `DECROOM_METRIC_FP_SCALE_MIN_FACTOR`
  - `DECROOM_METRIC_FP_SCALE_MAX_FACTOR`
  - `DECROOM_METRIC_FP_SCALE_MAX_IQR`
  - `DECROOM_METRIC_FP_SCALE_MAX_SPREAD`

### Session 29: SfM Robustness Upgrade (Low-Texture / Low-Contrast Scenes)
- Updated `backend/pipeline/sfm.py` image materialization:
  - Added optional SfM-only image enhancement (autocontrast + contrast + sharpness)
    before writing COLMAP frames.
- Updated `backend/pipeline/sfm.py` COLMAP CLI options:
  - Higher SIFT feature budget and lower peak threshold.
  - Guided matching + larger max match cap.
  - More permissive mapper inlier/match thresholds.
- Updated `backend/pipeline/sfm.py` pycolmap options:
  - Mirrors CLI-side SIFT/matching/mapper permissive settings when supported by
    installed pycolmap bindings (`hasattr`-guarded).
- Updated `backend/pipeline/config.py`:
  - Added SfM enhancement + SIFT/matching/mapper tuning fields.
- Updated `backend/main.py` env support:
  - `DECROOM_SFM_ENABLE_IMAGE_ENHANCEMENT`
  - `DECROOM_SFM_ENHANCE_CONTRAST`
  - `DECROOM_SFM_ENHANCE_SHARPNESS`
  - `DECROOM_SFM_SIFT_MAX_NUM_FEATURES`
  - `DECROOM_SFM_SIFT_PEAK_THRESHOLD`
  - `DECROOM_SFM_MATCHING_GUIDED`
  - `DECROOM_SFM_MAPPER_MIN_NUM_MATCHES`
  - `DECROOM_SFM_MAPPER_INIT_MIN_NUM_INLIERS`
  - `DECROOM_SFM_MAPPER_ABS_POSE_MIN_NUM_INLIERS`

### Session 30: SfM CLI Compatibility Hotfix
- Updated `backend/pipeline/sfm.py` matcher command construction:
  - `--SiftMatching.guided_matching` is now only appended when explicitly enabled.
  - Prevents COLMAP builds that do not support this flag from hard-failing matcher.
- Updated `backend/pipeline/config.py` default:
  - `sfm_matching_guided` now defaults to `False` for broader CLI compatibility.

### Session 31: COLMAP Matcher Option Fix + Duplicate-Bed Suppression
- Updated `backend/pipeline/sfm.py` COLMAP matcher options for 3.13:
  - Replaced unsupported `--SiftMatching.max_num_matches` with
    `--FeatureMatching.max_num_matches`.
  - Replaced guided option to `--FeatureMatching.guided_matching` when enabled.
- Updated `backend/pipeline/lifting.py`:
  - Added weak duplicate-bed suppression:
    - when a strong bed exists, weak bed candidates (`low n_detections + low fp`)
      are dropped pre-filter.
  - Added `bed` relaxed near-wall axis snap threshold (0.58) to reduce
    unsnapped near-threshold bed orientation drift.

### Session 32: Recurrence Stabilization (Geometry Continuity + Fragmentation Control)
- Updated iOS capture continuity + metadata contract:
  - `decroom_no_lidar/CoverageEngine.swift`
    - Overlap policy reweighted to continuity-first:
      - `0.15*texture + 0.45*yaw + 0.40*step`
    - Added continuity hard block:
      - block capture when `yawDelta > 105° && step > 0.45m` (after first 2 captures)
    - Added continuity soft warning:
      - warn when `yawDelta > 80° && step > 0.30m`
  - `decroom_no_lidar/ContentView.swift`
    - Uploads new per-shot fields:
      - `quality.overexposure_ratio`
      - `quality.motion_overlap_score`
      - `sequence_break_flag`
- Updated backend metadata parsing:
  - `backend/pipeline/metadata.py`
    - Parses:
      - `quality.overexposure_ratio`
      - `quality.motion_overlap_score`
      - `sequence_break_flag`
- Updated pose-jump policy to geometry-only filtering:
  - `backend/pipeline/pipeline.py`
    - Pose rejection no longer removes frames from detection/depth.
    - Added geometry-stage frame mask diagnostics:
      - `geometry_keep_indices`
      - `geometry_dropped_frames`
      - `geometry_drop_reasons`
    - Added hybrid discontinuity trigger:
      - `(step > 0.75m && yaw_delta > 95°) || (step > hard_threshold) || (speed_gate)`
    - Geometry stages now use filtered frame set:
      - feature cloud / fusion / tracking / layout / triangulation / lifting
    - Detection/depth still run on all uploaded frames.
- Updated tracking anti-fragmentation controls:
  - `backend/pipeline/tracking.py`
    - Added per-category new-track minimum confidence gate.
    - Added weak-SfM monitor tolerance relaxation.
    - Extended 2D fallback gap allowance when appearance similarity is high.
    - Added explicit rejection logging:
      - low score
      - no viable match candidates
      - candidates existed but not selected
    - Tracking report now includes:
      - `rejections`
      - `rejection_reason_counts`
      - `fragmentation_severity_by_category`
- Updated SfM/diagnostic quality gates and reporting:
  - `backend/pipeline/config.py`
    - `enable_pose_jump_rejection = True` (default)
    - `sfm_min_frames = 8` (default)
    - `sfm_apply_max_rmse_m = 0.35`
    - added hybrid pose-jump thresholds:
      - `pose_jump_rejection_hybrid_step_m`
      - `pose_jump_rejection_hybrid_yaw_deg`
  - `backend/pipeline/sfm.py`
    - Added/kept quality gate reporting and frame-set diagnostics:
      - `sfm_apply_quality_gate`
      - `sfm_frames_used_indices`
      - `sfm_graph_component_size`
      - fixed `pruned_frames` index handling
- Updated layout regularization with plane prior:
  - `backend/pipeline/layout.py`
    - Always computes plane prior when available.
    - When FP-PCA envelope is undersized versus plane prior, applies upward regularization.
    - Emits blended source:
      - `feature_points_pca+plane_regularized`
    - Exposes both confidences:
      - `axis_confidence_fp`
      - `axis_confidence_plane`
- Updated lifting uncertainty suppression and orientation fallback:
  - `backend/pipeline/lifting.py`
    - Tightened desk context rescue anchors (requires stronger multi-frame monitor evidence).
    - Suppresses high-uncertainty context-inferred objects from final payload by default.
    - Adds suppression diagnostics:
      - `suppressed_inferred_objects_total`
      - `suppressed_inferred_objects`
    - For low-support `camera_last_resort`, outputs neutral orientation with high uncertainty.
    - Adds payload metadata flags:
      - `inferred`
      - `suppressed_reason`
- Updated schema and reconstruction report interfaces:
  - `backend/schema.py`
    - `ObjectMetadata` now supports:
      - `inferred`
      - `suppressed_reason`
  - `backend/pipeline/reconstruction.py`
    - Adds run-level fields:
      - `geometry_frames_used`
      - `geometry_frames_dropped`
      - `geometry_drop_reasons`
      - `sfm_apply_quality_gate`
      - `primary_failure_mode_ranked` (top-3)
      - `suppressed_inferred_objects`
