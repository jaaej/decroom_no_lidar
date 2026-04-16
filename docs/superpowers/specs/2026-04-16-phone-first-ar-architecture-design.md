# Phone-First AR Architecture — Design Spec

**Date**: 2026-04-16
**Author**: jaejeon (with Claude)
**Status**: Design — awaiting implementation plan

## Context

The Decroom pipeline has been iterated on many times to fix recurring failures:
- COLMAP SfM fragments (19+ components from 49 frames → bad camera poses)
- PCA layout aligns to camera path instead of actual walls (room rotated 45°)
- Monocular depth noise causes duplicate objects and bunched placement
- Text-based error messages ("hold steady") are ambiguous and unhelpful

Every fix has patched symptoms rather than the root cause. The root cause: **the architecture ignores the rich real-time 3D data that ARKit already produces**. The phone has been treated as a dumb camera while the backend tries to reconstruct 3D from 2D images — work ARKit has already solved.

This spec redesigns the pipeline around the principle: **phone builds the room, backend adds semantics**.

## Goals

- **Reliability** — every scan produces a usable result. Failures become graceful degradation, never "nothing."
- **Accuracy** — walls from ARKit plane anchors (ground truth). Furniture placement from learned models on good data.
- **User experience** — user sees the room being built around them in AR during scanning. Visual-first guidance replaces text warnings.

## Non-Goals

- Supporting devices older than iPhone X (A11). Fallback exists but isn't the focus.
- Real-time furniture placement on-device (SpatialLM stays on backend GPU).
- Replacing SpatialLM with a different semantic model in this iteration.
- Addressing the security issues from the 2026-04-16 security audit (tracked separately).

## Architecture

### Before

```
Phone: capture frames → upload 40-200 JPEGs (~50MB)
Backend: COLMAP SfM → Depth Anything → YOLO → ByteTrack → PCA layout → lifting → SpatialLM
Return: RoomPayload JSON
```

### After

```
Phone (real-time, 30fps):
  ARKit → metric poses, feature points, plane anchors
  Core ML Depth Pro → on-device metric depth (~5fps)
  On-device fusion → room geometry + dense point cloud
  Live AR grid shell + reticle → visual guidance

Upload (~5MB):
  room_geometry.json + point_cloud.ply + ~10 keyframes + per-scan log

Backend (~5-10s):
  SpatialLM → semantic labels + furniture bounding boxes
  YOLO → refinement on keyframes
  Cross-check: SpatialLM walls vs phone walls (prefer phone)
  Return: enriched RoomPayload
```

Device tiers determine which path runs:

| Tier | Devices | Path |
|------|---------|------|
| `.full` | iPhone 12+ (A14+) | Phone-first path, Core ML depth at 5-10fps |
| `.reduced` | iPhone XR/11 (A12-A13) | Phone-first path, Core ML depth at 1-2fps |
| `.fallback` | iPhone X or older | Existing backend-heavy pipeline, unchanged |

## Components

### New: iOS

**`DepthRunner.swift`**
Wraps Core ML Depth Pro. Input `CVPixelBuffer`, output `MLMultiArray` metric depth map. Handles thermal throttling (drops to 2fps when frame time > 400ms). Model (~150MB) downloaded on first launch, cached in app support directory.

**`RoomGeometryBuilder.swift`**
Consumes `ARPlaneAnchor` updates. Merges coplanar planes into wall segments. Clusters plane normals for Manhattan frame snapping. Outputs walls as `[(start: CGPoint, end: CGPoint)]` on XZ plane plus floor height. Replaces `layout.py` for phone-first path.

**`PointCloudAccumulator.swift`**
Given depth map + camera pose + RGB, unprojects depth pixels to world space and voxel-downsamples at 2cm. Bounded memory (500K points max). Exports binary PLY blob on demand for upload.

**`ARRoomPreviewView.swift`**
`ARView` with RealityKit entities implementing the grid shell + reticle + brightness gradient + emerging furniture. See the "Visual Design" section below for full spec.

**`KeyframeSelector.swift`**
Farthest-point sampling in camera position space. Selects ~10 diverse keyframes for upload (not for geometry — geometry is solved on-device).

**`DeviceCapabilities.swift`**
Detects chip generation, ARKit capabilities, returns `.full | .reduced | .fallback` tier. Read by `CaptureFlowView` to choose path.

**`ScanLogger.swift`**
Writes per-scan JSON log to `~/Documents/scan_logs/{timestamp}_{room_id}.json`. Retention: last 50 scans. Accessible from Settings → Debug → Scan Logs with export/share. Included in upload payload for backend correlation.

**`YoloRunner.swift`** (NEW)
Wraps YOLOv8-seg Core ML model (~15MB). Runs at 1-2fps during capture. Outputs `[YoloDetection]` with category, confidence, 2D mask. Feeds into both keyframe selection (semantic richness) and post-upload consensus check.

**`ObjectTracker.swift`** (NEW — on-device, simple)
Maintains unique object instances across frames using nearest-neighbor in 3D position space (derived from YOLO mask centroid + phone depth). Not a full ByteTrack — just "same object seen twice" deduplication. Outputs a per-object best-frame index for keyframe selection.

**`ObjectCorrectionView.swift`** (NEW)
Overlay on `RoomSceneView` showing confidence indicators (pulse for medium, "?" for low). Tap handler opens action sheet (Delete / Rotate 90° / Move). Corrections update the `RoomPayload` in-place and persist with save.

### Modified: iOS

**`UploadService.swift`**
New payload schema:
- `room_geometry.json` (~50KB) — walls, floor height, Manhattan yaw
- `point_cloud.ply` (~2MB binary) — 200K colored points from phone
- `keyframes/` (~3MB, 10 JPEGs + per-frame intrinsics/pose)
- `scan_log.json` (~100KB)

Total: ~5-6MB (10x reduction from current).

**`CaptureFlowView.swift`**
Reads device tier. Launches `ARRoomPreviewView` for `.full`/`.reduced`, falls back to current `VideoRecorderView` for `.fallback`.

**`VideoRecorderView.swift`**
Kept for fallback tier devices. No changes required.

### Modified: Backend

**`pipeline/spatiallm_layout.py`**
- `build_dense_colored_pointcloud()` removed when phone provides PLY directly. Keep for fallback path.
- Add `load_phone_pointcloud(ply_bytes)` to accept pre-built cloud.
- Add `verify_geometry_agreement(phone_walls, spatiallm_walls)` — logs when they disagree by > 15° yaw. Phone walls win.
- Add `apply_wall_snapping(objects, walls, cfg)` — post-process SpatialLM output with semantic rules.
- Add `merge_with_yolo_detections(slm_objects, yolo_detections, cfg)` — dual-model consensus logic. Collapses duplicates, flags unverified detections with confidence scores.
- Add `compute_object_confidence(obj, yolo_match, view_count, size_plausible)` — per-object confidence score.

**`pipeline/pipeline.py`**
New conditional branch: if upload includes `room_geometry.json`, skip Steps -1 (SfM), 0 (pose refinement), 4 (layout), 5 (lifting). Use phone geometry + SpatialLM output directly. YOLO still runs for detection refinement.

**`main.py`**
Accept new multipart fields (`room_geometry`, `point_cloud`, `scan_log`). Validate schema. Archive scan_log alongside existing debug artifacts.

### Removed (phone-first path only)

- COLMAP SfM — already disabled for SpatialLM path.
- PCA layout estimation in `layout.py`.
- Depth backprojection in `lifting.py`.
- Track triangulation from rays.
- Depth scale calibration (phone depth is already metric).
- ~3,000 lines of heuristic code.

These remain for `.fallback` tier devices.

## Data Flow

```
Capture (phone, real-time):
  ARFrame → camera.transform → pose tracker (30fps)
                 → rawFeaturePoints → feature cloud
                 → planeAnchors → RoomGeometryBuilder
                                       ↓
                                   walls[] + floor_y
                                       ↓
                           Grid shell solidifies in AR

  Every 200ms:
    capturedImage → DepthRunner (Core ML) → depth map
                                                 ↓
                                          PointCloudAccumulator
                                                 ↓
                                          200K-point cloud grows
                                                 ↓
                                          Dense voxels dim→bright in AR

  Every ~2s:
    Keyframe trigger if new coverage detected → save JPEG + metadata

User taps "Done":
  → KeyframeSelector picks 10 diverse frames
  → PointCloudAccumulator exports PLY
  → RoomGeometryBuilder exports walls JSON
  → ScanLogger writes log
  → UploadService POSTs compact payload

Backend:
  → Validate schema
  → SpatialLM inference on PLY → walls + furniture bboxes
  → Cross-check SpatialLM walls vs phone walls (phone wins on disagreement)
  → YOLO on keyframes (refine categories)
  → Map SpatialLM 59 cats → Decroom 14 cats
  → Build RoomPayload

Response:
  → Phone renders 3D preview in SceneKit
  → User: Save Room / Rescan
```

## Visual Design

### Layer 1: Grid Shell (canvas)

- `RealityKit` entity: wireframe cube, 3m × 3m × 3m, cyan lines at 20cm spacing
- Centered on user at scan start
- Each detected `ARPlaneAnchor`:
  - Replace corresponding section of grid with solid wall material
  - Material: semi-transparent white (30% opacity initially)
  - Preserves Manhattan alignment from ARKit
- Unscanned faces remain wireframe

### Layer 2: Reticle (guide)

- Floating world-anchored cyan ring, 20cm diameter
- Position: centroid of nearest still-wireframe region
- Updates every 500ms as coverage evolves
- States:
  - Default: steady glow
  - User not facing: pulse brighter (1.2x intensity, 2Hz)
  - User centered on it: ring fills clockwise over 2s → captures complete → jumps to next region
- Haptic pulse (`.medium`) when centered in view

### Layer 3: Brightness Gradient (progress)

- Each wall section starts at 30% opacity
- Additional depth points falling on that wall → opacity increases
- Cap at 100% at ~500 points per wall
- Visual effect: walls "thicken" as you rescan

### Layer 4: Furniture Emergence

- Received from backend after upload, but visible during upload as ghost boxes from YOLO on-device (stretch goal)
- Fade in over 0.5s at detected position
- Colored box per category, USDZ model replaces box after backend returns

### Failure States (visual-first)

| Problem | Visual State |
|---------|--------------|
| ARKit tracking limited | AR view desaturates to grayscale, grid pauses animation, haptic tick |
| Too dark | Grid tints red in dark regions, brighter in lit regions |
| Moving too fast | Grid edges blur with motion streaks, sharper at rest |
| No textured features | Reticle gravitates to nearest textured area with pulsing outline |
| Memory pressure | Silent — accumulator drops oldest voxels, no user interruption |

Only permitted text strings during capture:
- **Scan complete**: "All walls captured! Tap Done." (green)
- **Unrecoverable**: "AR not supported on this device." (red, once)

## Semantic Reliability Layer

The phone-first architecture fixes the *geometric* causes of misplacement (wrong poses, noisy depth, PCA misalignment). But semantic problems — duplicate detections, wrong orientation, category confusion — need explicit downstream handling. Four mechanisms:

### 1. Dual-Model Consensus (YOLO on-device + SpatialLM backend)

A single model's output is a guess; agreement between two models is evidence.

**On-device**: Run YOLOv8-seg Core ML at 1-2fps during capture. For each detection:
- Record category, confidence, 2D mask
- Unproject mask centroid using phone's depth map → 3D position
- Track unique instances across frames using simple nearest-neighbor assignment

**On return from backend**: Compare YOLO local detections with SpatialLM output:

| YOLO says | SpatialLM says | Outcome |
|-----------|----------------|---------|
| Bed at (2, 1.5) | Bed at (2.1, 1.4) | ✅ High confidence, keep SpatialLM's oriented bbox |
| Bed at (2, 1.5) | Nothing | ⚠️ Use YOLO position, default orientation |
| Nothing | Bed at (2, 1.5) | ⚠️ Medium confidence, keep with "?" flag |
| 1 bed total | 2 beds at same position | ❌ SpatialLM hallucination — collapse to 1 at merged centroid |

Directly addresses "2-3 beds from 1 bed" — SpatialLM duplicates are caught by YOLO's single detection.

### 2. Semantic Wall-Snapping Rules

Universal priors applied as post-processing after SpatialLM returns:

| Category | Rule |
|----------|------|
| bed, dresser, wardrobe, bookshelf, sofa | Snap to nearest wall within 0.8m, rotate facing = wall normal |
| desk | Snap to wall within 0.8m, or corner if two walls are equally close |
| nightstand | Snap to wall within 0.5m (tighter — nightstands are against walls) |
| monitor, TV | Orient perpendicular to nearest wall (facing into room) |
| chair, stool, table (coffee/dining) | Freestanding, no snap |
| lamp_floor, plant, rug | Freestanding, no snap |

Applied backend-side after `parse_spatiallm_output()` in `spatiallm_layout.py`. Uses phone's `walls[]` as hard constraints. Directly addresses "bed in middle of room."

### 3. Smart Keyframe Selection (detect-then-select)

Selection criteria prioritize semantic richness, not just spatial diversity:

1. **Spatial coverage**: cover full 360° yaw via farthest-point sampling (geometric)
2. **Object coverage**: for each unique YOLO-detected object, include the frame with the highest confidence (semantic)
3. **Quality**: drop frames with blur score < 120 or fewer than 5 ARKit feature points
4. **Cap**: 10-15 keyframes total

YOLO runs on-device during scanning (already added for consensus). It tracks unique object instances, so when selecting keyframes the phone knows "frame 34 had the clearest view of the bed, frame 67 had the clearest view of the desk." Backend gets the best view of each object.

### 4. Per-Object Confidence + User Correction UI

Every final object has a confidence score:

```
confidence = (agreement_score × 0.5) + (multi_view_score × 0.3) + (size_plausibility × 0.2)
```

Where:
- `agreement_score`: 1.0 if YOLO and SpatialLM agree, 0.5 if only one, 0.3 for SpatialLM duplicate-collapse
- `multi_view_score`: 1.0 if object seen in 3+ keyframes, scales down for fewer
- `size_plausibility`: 1.0 if dimensions within typical range for category, lower otherwise

In the 3D preview, confidence is visualized:
- **High (>0.8)**: solid USDZ, no indicator
- **Medium (0.5-0.8)**: subtle pulse animation
- **Low (<0.5)**: translucent + small "?" icon

Tap any object to open an action sheet:
- **Delete** (wrong detection — most common)
- **Rotate 90°** (orientation wrong)
- **Move** (drag to new position — stretch goal)

Corrections persist with the saved room and are included in scan logs (signal for improving the models over time).

## Upload & Post-Capture UX

The room shell exists on the phone before upload begins, so there is **no blocking loader** during the backend trip. The user transitions directly into an interactive 3D preview and watches furniture populate.

### Sequence

| Time | State |
|------|-------|
| t=0 | User taps "Done". Smooth transition from AR capture → orbital 3D preview. |
| t=0-0.3s | Walls + floor render as solid textured surfaces (from phone-side geometry). Empty but fully interactive (rotate, pan, zoom). |
| t=0.1s | Background upload begins. Subtle status strip at top: thin progress line + "Finding furniture..." |
| t=5-10s | Progress line fills as backend processes. Status may update to "Placing furniture..." once. No other UI changes. |
| t=10-15s | Backend returns RoomPayload. Progress strip fades out. |
| t=10-15s+ | Furniture fades in one item at a time. Each: 0.4s fade from ghost shape → solid USDZ. 0.15s stagger between items. |
| t=complete | "Save Room" / "Rescan" buttons appear. Full interactivity. |

### Failure States

- **Upload fails** (no network, timeout): progress strip turns amber, text becomes "Tap to retry." Room shell remains interactive. User can save an empty room or retry upload.
- **Backend timeout** (> 60s): same as upload fail. Non-blocking.
- **No furniture detected** (SpatialLM returns 0 objects): room saved with empty object list. Valid state.

### Principles

- Never block user interaction with a spinner
- Progress is additive (furniture appears), never subtractive (walls disappearing)
- The room always exists from t=0, even if backend never returns

## Error Handling

Graceful degradation chain:

```
Full success: Room + furniture + interactive preview
Depth model fails to load: Room from plane anchors only (less dense cloud, functional)
Backend fails: Room shell only (walls, no furniture) + "Retry detection" button
ARKit fails: Guided error (can't recover — fundamental)
```

Per-failure details in Section 4 of the brainstorming output (reproduced visually above).

## Logging

Every scan writes `~/Documents/scan_logs/{timestamp}_{room_id}.json`:

```json
{
  "scan_id": "8af3c921",
  "started_at": "2026-04-16T14:22:01Z",
  "completed_at": "2026-04-16T14:24:17Z",
  "device": {"model": "iPhone 15 Pro", "ios": "17.2", "chip": "A17 Pro", "tier": "full"},
  "arkit": {
    "tracking_losses": [{"t": 34.2, "duration_s": 1.1, "reason": "excessive_motion"}],
    "plane_anchors_total": 7,
    "plane_types": {"wall": 4, "floor": 1, "ceiling": 2}
  },
  "depth": {
    "model": "depth_pro_coreml_v1",
    "frames_processed": 312,
    "avg_inference_ms": 187,
    "thermal_throttles": 0
  },
  "coverage": {
    "final_yaw_bins_filled": 23,
    "final_yaw_bins_total": 24,
    "unscanned_regions": ["ceiling_southwest_corner"]
  },
  "upload": {
    "payload_size_kb": 4821,
    "backend_latency_ms": 8340,
    "spatiallm_objects": 7,
    "spatiallm_walls_match_arkit": true
  },
  "events": [ {"t": 0.0, "event": "scan_started"}, ... ]
}
```

- Retention: last 50 scans
- Accessible in Settings → Debug → Scan Logs
- Exportable/shareable
- Included in upload payload; backend archives alongside `/tmp/decroom_runs/{timestamp}_{room_id}/`

## Testing

Three tiers (expanded in brainstorming Section 5):

1. **Unit tests**: `RoomGeometryBuilder.merge()`, `PointCloudAccumulator.voxelize()`, `KeyframeSelector.isDiverse()`, `DeviceCapabilities.tier()`, coordinate conversions.
2. **ARWorldMap replay**: capture real-room `ARWorldMap` fixtures, replay in tests, assert geometry output.
3. **Manual QA matrix**: device matrix (iPhone 15 Pro / 13 / XR) × room matrix (bedroom / living / empty / dim / textured), per-test checklist.
4. **Log-driven regression**: compare scan logs across versions to catch silent regressions.

## Performance Targets

| Stage | Target |
|-------|--------|
| Scan duration (user-facing) | 30-90s |
| Upload size | < 6 MB |
| Upload duration (WiFi) | < 10s |
| Backend processing | < 15s |
| Total scan → preview | < 2 min |
| Depth model inference | < 300ms per frame (A14+) |
| Memory footprint during scan | < 500 MB |

## Risks and Open Questions

**Risk**: Core ML Depth Pro model size (~150MB) — significant app download.
*Mitigation*: Download on first launch, cache, show progress. Could be bundled if size matters less than cold start.

**Risk**: Plane anchor quality varies by lighting and texture.
*Mitigation*: Grid shell visual guidance nudges users to textured areas. Fallback to depth-only geometry if plane anchors remain sparse.

**Risk**: Backend SpatialLM downtime would stop furniture detection entirely.
*Mitigation*: Room shell renders without backend. "Retry detection" button. Future: on-device YOLO as stretch (already explored, deferred).

**Open question**: What's the minimum coverage threshold to allow "Done"?
*Current answer*: User can tap Done anytime; coverage percentage shown. No hard gate.

**Open question**: How to merge phone walls and SpatialLM walls when they agree but differ slightly (e.g., 3° off)?
*Current answer*: Always prefer phone walls (ARKit is ground truth). SpatialLM disagreement is logged as a diagnostic signal.

**Open question**: Does Core ML Depth Pro include its own coordinate frame conversion, or must we handle it?
*Current answer*: TBD during implementation. Apple's model targets generic images, so we provide the portrait-oriented CVPixelBuffer directly.

## Implementation Phases

Phase 1 — Foundation (Week 1):
- `DeviceCapabilities.swift` tier detection
- `ScanLogger.swift` JSON log writer
- Core ML Depth Pro integration + download flow

Phase 2 — On-device Geometry (Week 2):
- `RoomGeometryBuilder.swift` from plane anchors
- `PointCloudAccumulator.swift` with voxelization
- `KeyframeSelector.swift` smart selection (spatial + semantic)

Phase 3 — On-device Semantics (Week 2-3):
- `YoloRunner.swift` Core ML YOLOv8 integration
- `ObjectTracker.swift` simple NN instance tracking
- Integrate YOLO output into keyframe selection

Phase 4 — Visual Capture (Week 3-4):
- `ARRoomPreviewView.swift` grid shell + reticle + brightness + furniture
- Failure state visuals
- Replace `VideoRecorderView` for `.full`/`.reduced` tiers

Phase 5 — Upload & Backend (Week 4):
- New `UploadService` payload schema (includes YOLO detections)
- Backend accepts `room_geometry` + `point_cloud` + `yolo_detections`
- `verify_geometry_agreement()` cross-check
- `apply_wall_snapping()` semantic rules
- `merge_with_yolo_detections()` dual-model consensus
- Short-circuit SfM/layout/lifting when phone geometry present

Phase 6 — Correction UI + Testing (Week 5):
- `ObjectCorrectionView.swift` confidence indicators + action sheet
- ARWorldMap fixture capture
- Device/room QA matrix
- Log viewer in debug menu
- Performance profiling + thermal optimization

## Dependencies

- **Apple**: ARKit 6+ (iOS 16+), Core ML 7+ (iOS 17+), RealityKit 2+
- **Apple models**: Depth Pro via Core ML (Apache 2.0, from apple/ml-depth-pro GitHub)
- **On-device models**: YOLOv8-seg Core ML export (~15MB, AGPL), bundled with app
- **Existing backend**: SpatialLM (unchanged), YOLO (still used for backend refinement), FastAPI

No new backend dependencies required.
