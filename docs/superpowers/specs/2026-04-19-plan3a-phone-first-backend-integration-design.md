# Plan 3a — Phone-First Backend Integration (MVP)

**Date**: 2026-04-19
**Author**: jaejeon (with Claude)
**Status**: Design — awaiting implementation plan
**Supersedes**: the "Plan 3" bullet list in `memory/plan_status.md` and Phase 5 of `docs/superpowers/specs/2026-04-16-phone-first-ar-architecture-design.md`
**Related plans**: Plan 1 (merged to `main`), Plan 2 (on `feature/phone-first-visual`, visual UI abandoned)

## Context

Plans 1 and 2 built the entire phone-side foundation of the phone-first architecture:
- Plan 1: `DeviceCapabilities`, `ScanLogger`, `DepthRunner` + model, `RoomGeometryBuilder`, `PointCloudAccumulator`, `KeyframeSelector` — merged to `main`.
- Plan 2: `DeviceInfo.current()`, `Intrinsics+ARCamera`, `PixelBufferRGB`, `ObjectTracker`, `YoloRunner` + model, `ScanSession` coordinator — on `feature/phone-first-visual`.

**None of it is wired into the live scan.** `VideoRecorderView` still uploads JPEG-only payloads to the old backend (`main.py` → COLMAP → backend depth → PCA layout → lifting → SpatialLM). The phone-first upside — 10× smaller upload, no SfM fragmentation, ARKit plane anchors as ground truth — is gated behind the unfinished wiring.

Plan 2 also ran an experimental AR visual UI (grid shell + reticle, then panorama mosaic, then five more variations), all of which failed. The visual scanning UX is deferred indefinitely; `VideoRecorderView` remains the capture UX.

Plan 3a is the smallest change that unlocks the phone-first architecture: wire `ScanSession` into `VideoRecorderView`, replace the upload payload, accept it on the backend, and retire the old pipeline. Semantic polish (wall-snapping rules, YOLO/SpatialLM consensus, per-object confidence) is explicitly deferred to a future Plan 3b.

## Goals

- Deliver the phone-first architecture end-to-end on a user scan today.
- Keep `VideoRecorderView`'s capture UX unchanged — same Start/Done/Upload flow, same coverage arc. The only new user-visible surfaces are (a) the device-tier gate on unsupported phones at launch, and (b) the error alerts defined in the error taxonomy. The happy-path capture screen is identical to today.
- Hard cutover: every device that can scan uses phone-first. No fallback to the old backend pipeline.
- Every failure mode produces a specific, actionable error message. No silent failures, no generic "something went wrong."

## Non-Goals

- No new scanning UX. `VideoRecorderView` stays.
- No on-device YOLO consensus (deferred to 3b).
- No server-side wall-snapping or per-object confidence (deferred to 3b).
- No support for iPhone X and older (`.fallback` tier). Those devices get a "not supported" message at launch.
- No app-size reduction work. The ~72 MB of bundled Core ML models stays as-is.

## Architecture

### User-facing experience (unchanged)

`VideoRecorderView` stays exactly as today: Start Scan button, live camera, coverage arc, Done button, Upload button. Users do not see a new screen or different copy.

### Under the hood

```
PHONE (during scan)
───────────────────
VideoRecorderView (unchanged UX)
    │
    ▼
GuidedARFrameRecorder (existing capture logic)
    │
    owns & forwards frames to ─▶
    │
    ▼
ScanSession (Plan 2 component, now live):
    ├─ ARKit plane anchors   → RoomGeometryBuilder → walls + floor/ceiling
    ├─ Captured image        → DepthRunner (metric) → depth map (meters)
    │                           PointCloudAccumulator (voxel 2cm) → PLY
    ├─ Captured image        → YoloRunner → detections (logged only in 3a)
    │                           ObjectTracker (logged only in 3a)
    ├─ Camera pose stream    → KeyframeSelector → top-10 JPEGs + metadata
    └─ Event stream          → ScanLogger → scan_log.json

User taps Done:
    ScanSession.finalize() → ScanUploadPayload {
        room_geometry.json  (~50 KB)
        point_cloud.ply     (~2 MB, binary, XYZRGB)
        keyframes/ × 10     (~3 MB, JPEGs + manifest)
        scan_log.json       (~100 KB)
        device_info
    }
    Total: ~5–6 MB (down from ~50 MB)
    │
    ▼ POST /api/upload/phone-first (multipart)

BACKEND
────────
main.py route phone_first_upload():
    1. Parse multipart, validate schema
    2. Load PLY, walls, keyframes, log
    3. SpatialLM inference on phone-provided PLY (~5–10s)
    4. Compare SpatialLM walls vs phone walls; log disagreement, KEEP phone walls
    5. Map SpatialLM 59 categories → Decroom 14 categories
    6. Build RoomPayload (same schema as today)

  DELETED ENTIRELY:
    - COLMAP / Structure-from-Motion
    - Backend depth estimation
    - PCA layout
    - Lifting / 2D→3D backprojection
    - _INSTANCE_CAPS hardcoded limits
    - Old /api/upload route

    │
    ▼

PHONE (response)
────────────────
RoomSceneView renders RoomPayload (unchanged UX)
```

### Device tier policy

| Tier | Devices | Behavior |
|---|---|---|
| `.full` | iPhone 12+ (A14+) | Phone-first path, depth at ~5 fps |
| `.reduced` | iPhone XR/11 (A12–A13) | Phone-first path, depth at ~2 fps |
| `.fallback` | iPhone X and older | Scan button disabled at app launch. Message: "This iPhone doesn't support Decroom's scanning. Minimum: iPhone XS." |

**No fallback path exists.** `.fallback` tier does not get the old backend pipeline as a safety net — it simply can't scan. This is the direct consequence of deleting the old pipeline.

## Components

### On-device (iOS)

1. **`DepthModel.mlpackage` swap** — replace bundled model with `depth_anything_v2_metric_hypersim_small` converted to Core ML via `coremltools`. Indoor-scene-tuned, outputs meters directly. Same ~50 MB, same iPhone speed as current relative variant.

2. **`DepthRunner.swift` update** — remove "relative" language in comments + docstring. No behavior change.

3. **`PointCloudAccumulator.swift` simplification** — delete the scale-calibration math that multiplied relative depth against ARKit feature-point scale. Depth is now in meters at ingress.

4. **`ScanSession.swift` finalize method** — add `finalize() -> ScanUploadPayload`. Serializes geometry to JSON, point cloud to binary PLY, keyframes with per-frame pose/intrinsics, scan log.

5. **`GuidedARFrameRecorder` → `ScanSession` wiring** — recorder owns a `ScanSession` instance. On each `ARFrame` update, after its existing processing, forwards the frame to `ScanSession.update(frame:)`. On stop, asks `ScanSession.finalize()` for the payload.

6. **`UploadService.uploadPhoneFirst(_:) async throws -> RoomPayload`** — new method. Multipart POST to `/api/upload/phone-first`. Same Bonjour-first, fallback-URL resolution as today. 60-second timeout (down from 1800s — no SfM to wait on).

7. **`UploadService.upload(images:)` + `uploadFromDisk(jpegPaths:)` deletion** — no remaining callers after cutover.

8. **`ScanError` enum (new)** — one case per named failure mode. Each case carries a user-facing `message` and a technical `detail` (logged, not shown). See "Error taxonomy" below.

9. **Device tier gate** — at app launch (likely in `CaptureFlowView` or `MainTabView`), if `DeviceCapabilities.currentTier == .fallback`, disable the scan button and display the tier-unsupported message. Applies to all scan modes.

10. **Plan 2 dead-code deletion** — remove `ARRoomPreviewView.swift`, `ARRoomPreviewContainer.swift`, `ARRoomPreviewViewModel.swift`, `GridShellEntity.swift`, `CaptureReticle.swift`, `PanoramaBuffer.swift`, plus their associated tests (`ARRoomPreviewViewModelTests`, `CaptureReticleTests`).

### Backend

11. **`POST /api/upload/phone-first` route (new)** in `main.py`. Parses multipart body: `room_geometry`, `point_cloud`, `keyframes[]`, `scan_log`. Schema validation — malformed uploads return 400 with a specific error code the phone maps to `.backendSchemaError`.

12. **`pipeline/phone_first_pipeline.py` (new module)**:
    1. Parse + validate payload
    2. Load PLY, walls, keyframes, log
    3. Invoke SpatialLM on phone-provided PLY
    4. Cross-check SpatialLM walls vs phone walls; if disagreement > 15° yaw or > 0.5 m, log diagnostic, keep phone walls
    5. Map SpatialLM 59 categories → Decroom 14 (reuse existing mapping)
    6. Build `RoomPayload` (existing schema — phone is an invisible implementation detail to the client)

13. **Old pipeline deletion** — remove from `pipeline/`:
    - COLMAP/SfM wrapper and dependencies
    - `build_dense_colored_pointcloud()` and backend depth estimation
    - `layout.py` PCA logic
    - `lifting.py` 2D→3D backprojection
    - `_INSTANCE_CAPS` dict
    - Old `/api/upload` route

## Data shapes

### `room_geometry.json`

```json
{
  "walls": [
    {"start": {"x": 0.00, "z": 0.00}, "end": {"x": 3.20, "z": 0.00}},
    {"start": {"x": 3.20, "z": 0.00}, "end": {"x": 3.20, "z": 2.50}}
  ],
  "floor_height": 0.00,
  "ceiling_height": 2.40,
  "manhattan_yaw": 0.0,
  "coordinate_system": "arkit_world_right_handed_y_up"
}
```

All distances in meters. XZ is the floor plane. `coordinate_system` is a literal tag — the backend validates it matches and rejects unknown values.

### `point_cloud.ply`

Binary PLY, little-endian, vertices only:

```
ply
format binary_little_endian 1.0
element vertex <N>
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
<N × 15 bytes binary>
```

Same coordinate frame as `room_geometry.json`. N ≈ 200,000 after voxel dedup at 2 cm.

### `keyframes/`

10 JPEGs (`frame_00.jpg` … `frame_09.jpg`) + one manifest `keyframes.json`:

```json
{
  "frames": [
    {
      "filename": "frame_00.jpg",
      "timestamp": 1.234,
      "pose_world_from_camera": [[...4x4 row-major float...]],
      "intrinsics": {
        "fx": 1500.0, "fy": 1500.0,
        "cx": 720.0,  "cy": 960.0,
        "width": 1440, "height": 1920
      },
      "detections": []
    }
  ]
}
```

`detections` stays empty in 3a — YOLO output is captured only in `scan_log.json`. Populating it is a 3b task needed for consensus.

### `scan_log.json`

Existing `ScanLogger` schema from Plan 1 — device info, events, coverage stats, thermal events. Backend archives alongside debug artifacts for correlation.

### Response: `RoomPayload`

Existing schema. Phone is invisible to `RoomSceneView` — it still consumes the same JSON it does today.

## Data flow timing

| Stage | Runs on | Frequency |
|---|---|---|
| ARSession delegate | MainActor | 30 fps |
| `ScanSession.update(frame:)` entry | MainActor | 30 fps (fast, only reads frame metadata) |
| `DepthRunner.infer()` | mlQueue (background) | ~5 fps (every ~6 ARFrames) |
| `YoloRunner.infer()` | mlQueue (background) | ~2 fps (every ~15 ARFrames) |
| `PointCloudAccumulator.addDepthMap()` | MainActor (hop-back) | ~5 fps |
| `@Published` state updates | MainActor | as needed |
| Payload serialization at finalize | Task.detached (background) | once |
| Multipart body build + upload | Task.detached (background) | once |

Invariants:
- No ML inference on MainActor
- Every result hops back to MainActor before touching `@Published`
- `PointCloudAccumulator` is MainActor-only (no locks)

## Phone-side validation before upload

On Done tap, before building the payload:

| Check | Threshold | Failure error |
|---|---|---|
| Walls detected | ≥ 1 | `.noWallsDetected` |
| Yaw coverage | ≥ 12 of 24 bins (50%) | `.insufficientCoverage` |
| Payload serializes without error | — | `.payloadBuildFailed` |

These are cheap checks that prevent obviously-broken uploads and give users a specific "resume scan" action instead of a generic server-side rejection.

## Error taxonomy

Every named failure mode. Each `ScanError` case has a stable user-facing message (asserted by tests) and a logged technical detail.

| Error | Fires at | User message | Retry UX |
|---|---|---|---|
| `.unsupportedDevice` | App launch | "This iPhone doesn't support Decroom's scanning. Minimum: iPhone XS." | Scan button disabled |
| `.depthModelLoadFailed` | Scan start | "Can't start scan: the depth engine isn't available. Please reinstall the app." | No retry; scan button stays disabled until next launch |
| `.arKitTrackingLimited` | During scan | Yellow banner: "Lost tracking — slow down, more light" | Auto-clears when tracking resumes |
| `.arKitSessionFailed` | During scan | "ARKit stopped working. Restart the app." | No retry |
| `.noWallsDetected` | Tap Done | "No walls found. Point at the walls of your room and keep scanning." | "Resume scan" button |
| `.insufficientCoverage` | Tap Done | "Only part of the room was captured. Rotate 360° and try again." | "Resume scan" button |
| `.payloadBuildFailed` | Tap Done | "Couldn't save scan data. Check phone storage and try again." | "Retry" button |
| `.uploadNoServer` | Upload | "Can't reach the Decroom server. Check WiFi and tap Retry." | "Retry" button |
| `.uploadTimeout` | Upload | "Server took too long. Try again in a moment." | "Retry" button |
| `.backendSchemaError` | Upload response 4xx | "Scan data was rejected. Please report this. Scan ID: <id>" | No retry; shows scan ID |
| `.backendServerError` | Upload response 5xx | "Server error. Try again in a moment." | "Retry" button |
| `.noFurnitureDetected` | Response (not error) | Info toast: "Scan complete — no furniture found." | N/A — scan saves |

YOLO model load failures are **not fatal** in 3a (YOLO output is log-only). Logged as a warning, scan proceeds. This is deliberate — 3b depends on YOLO, so 3b upgrades this to a fatal error.

## Testing

### Layer 1: Swift unit tests

| Suite | Coverage |
|---|---|
| `DepthRunnerTests` (updated) | Metric output shape + value range sanity |
| `PointCloudAccumulatorTests` (updated) | Depth at ingress is already meters, no calibration step |
| `ScanUploadPayloadTests` (new) | `room_geometry.json` serialization; PLY bytes parse round-trip; keyframe manifest JSON shape |
| `ScanSessionFinalizeTests` (new) | `finalize()` returns non-empty payload; zero walls → throws `.noWallsDetected`; 5/24 bins → throws `.insufficientCoverage` |
| `ScanErrorTests` (new) | Every `ScanError` case has non-empty `message`; stable message strings (prevents silent UX drift) |
| `UploadServicePhoneFirstTests` (new) | Multipart body structure, boundary math, timeout config. Mocked `URLSession`, no real network |
| `DeviceTierGateTests` (new) | `.fallback` → scan button disabled + correct message |

### Layer 2: Python unit tests (backend)

| Module | Coverage |
|---|---|
| `test_phone_first_pipeline.py` (new) | Fixture PLY + walls + mocked SpatialLM returning known bboxes → correct `RoomPayload`. Wall disagreement > 15° logged + phone walls kept. Zero SpatialLM objects → valid `RoomPayload` with empty `objects` |
| `test_upload_schema.py` (new) | Malformed payload (missing PLY, bad JSON, wrong coord system tag) → 400 with specific error code phone maps to `.backendSchemaError` |
| `test_category_mapping.py` (new) | SpatialLM 59 categories → Decroom 14 mapping is exhaustive (every category maps to something or to "ignored") |

### Layer 3: Contract tests (phone ↔ backend)

Shared fixture at `tests/fixtures/phone_first_payload/` — a realistic payload captured once from a real scan, checked into the repo.

- Swift: `UploadServicePhoneFirstTests` posts fixture body to a mock server, verifies body bytes.
- Python: `test_phone_first_pipeline.py` loads the same fixture, verifies parser accepts it.

If either side changes schema, one of these breaks at CI — drift caught immediately.

### Layer 4: Manual QA (physical device required)

Documented in `docs/superpowers/qa/` with filename format `YYYY-MM-DD-plan3a-manual-qa.md` (date filled in when QA is performed), following the pattern of Plan 2's QA doc. Filled in from an actual physical-device scan. Pre-merge requirement.

**Happy-path:**
- Launch on supported device → scan button enabled
- Launch on `.fallback` device (iPhone X if available) → scan button disabled with correct message
- Full scan of a real room → upload succeeds → 3D preview renders
- `scan_log.json` present in `~/Documents/scan_logs/` after scan
- Thermal state during 60s scan stays `nominal` or `fair`

**Error-path — each deliberately triggered:**

| Error | How to trigger |
|---|---|
| `.depthModelLoadFailed` | Dev build with renamed `DepthModel.mlpackage` |
| `.arKitTrackingLimited` | Cover camera or move quickly |
| `.noWallsDetected` | Scan pitch-black room, tap Done |
| `.insufficientCoverage` | Tap Done after 2 seconds of scanning |
| `.uploadNoServer` | Backend off, tap Upload |
| `.uploadTimeout` | Backend with forced 120s sleep |
| `.backendSchemaError` | Dev build that corrupts `room_geometry.json` pre-upload |
| `.backendServerError` | Backend feature flag returning 500 |
| `.noFurnitureDetected` | Scan empty room |

### Out of scope for automation

- SceneKit rendering correctness of the 3D preview
- Depth model accuracy in real scenes (model eval, tracked separately)
- SpatialLM accuracy (correlated via scan_log diffs over time)
- Network conditions beyond reachable / unreachable

## Migration + deletion plan

Every commit is buildable and testable. No broken-in-between states.

### Phase A — Depth model swap (no behavior change, `ScanSession` still dead)

1. `feat: swap DepthModel to metric indoor variant` — replace `.mlpackage`, update `DepthRunner` comments, update `DepthRunnerTests`.
2. `refactor: remove scale calibration from PointCloudAccumulator` — depth is meters now. Update `PointCloudAccumulatorTests`.

### Phase B — ScanSession wiring (parallel, silent)

3. `feat: wire ScanSession into GuidedARFrameRecorder` — recorder owns session, forwards frames. New `ScanSessionFinalizeTests`.
4. `feat: add ScanUploadPayload builder + serialization` — `finalize()` returns full payload. New `ScanUploadPayloadTests`.

### Phase C — New upload method (dormant)

5. `feat: add UploadService.uploadPhoneFirst` — mocked tests, contract fixture at `tests/fixtures/phone_first_payload/`. New `UploadServicePhoneFirstTests`.

### Phase D — Backend (parallel work)

6. `feat: add phone_first_pipeline + /api/upload/phone-first route` — new endpoint + pipeline module. New `test_phone_first_pipeline.py`, `test_upload_schema.py`, `test_category_mapping.py`.

### Phase E — Cutover (the one scary commit)

7. `feat: switch VideoRecorderView to phone-first upload` — recorder calls `uploadPhoneFirst`. Add `ScanError` enum + phone-side validation. Update `RecordingProcessor` for new response path. **Manual on-device QA required before proceeding.**

### Phase F — UX plumbing

8. `feat: device tier gate at app launch` — `.fallback` → scan button disabled.
9. `feat: wire ScanError alerts throughout scan flow` — every error case a specific alert. Deliberate trigger of each in manual QA.

### Phase G — Delete obsolete iOS code

10. `refactor: delete UploadService.upload + uploadFromDisk` — compile verifies no callers.
11. `refactor: delete Plan 2 dead code` — all six Plan-2 visual files + associated tests. ~2,000 lines out.

### Phase H — Delete obsolete backend code

12. `refactor: remove old backend pipeline` — COLMAP/SfM, backend depth, PCA layout, lifting, `_INSTANCE_CAPS`.
13. `refactor: remove old /api/upload route` — phone-first is the only route.

### Phase I — QA + merge

14. `docs: plan 3a manual QA results` — filled-in checklist from actual physical-device scan.
15. Merge `feature/phone-first-visual` → `main` with `--no-ff`.
16. Delete the feature branch.

### Rollback paths

| Rolling back from | Impact | Recovery |
|---|---|---|
| Commits 1–6 | None — app still on old upload | `git revert <sha>` |
| Commit 7 (cutover) | Scans fail if new backend has bug | Revert on phone → old upload returns. Backend still has old route (Phase H not done yet) |
| Commits 10–13 (deletions) | Can't cleanly revert without also reverting 10–13 | Revert cleanups first, then revert cutover |
| Worst case | 3a dead, `main` untouched | Delete branch, start over |

**Ordering property**: Phase D (backend accepts new payload) commits *before* Phase E (phone sends new payload). Phase G (phone stops using old path) commits *before* Phase H (backend deletes old route). Eliminates deployment-order bugs.

**Deployment timing note**: commit order is not the same as deploy order. The backend GPU server (RunPod) must be running the Phase-D code before Phase E ships to test devices. In practice: merge Phase D backend changes, deploy to RunPod (follow `memory/plan_status.md` SpatialLM server bring-up steps), then only after that, land the Phase E phone commit. If backend deployment is delayed, Phase E stays unmerged — no pressure to cut over until backend is demonstrably live.

### Branch strategy

- Continue on `feature/phone-first-visual`. Optional rename to `feature/phone-first-backend` for clarity.
- One merge to `main` at the end of Phase I.
- Plan 2's dead code never touches `main` — it's deleted in Phase G before merge.

## Risks and open questions

**Risk**: Metric depth model Core ML conversion could fail (unsupported ops, shape issues).
*Mitigation*: `coremltools` supports Depth Anything V2 architecture. If conversion fails, fall back to the current relative variant + existing calibration code. ~1 hour to revert Phase A.

**Risk**: Metric indoor model is less accurate than expected in unusual scenes (dim, extreme wide-angle, reflective).
*Mitigation*: `scan_log.json` records per-scan depth statistics. Patterns become visible over real scans. Can swap to a different metric variant in a future minor commit without re-planning.

**Risk**: SpatialLM gives unusable results on phone-built PLY because PLY density/distribution differs from what SpatialLM was trained on.
*Mitigation*: Keep a representative test scan's PLY as a regression fixture. Run it through SpatialLM at every backend change. If results degrade, we tune point cloud generation (voxel size, max point count) without changing the overall architecture.

**Risk**: iPhone X/older users who scan today lose scanning after Phase F.
*Mitigation*: User has explicitly chosen "no fallback." If user traffic is non-negligible, alternative is keeping the old path alive for `.fallback` tier only — but that contradicts the design decision here.

**Open question**: Is the 60-second upload timeout safe? SpatialLM inference is ~5–10s on the GPU, plus network round-trip. 60s should be plenty, but worth measuring in the first week after cutover.

**Open question**: Should `scan_log.json` include raw Depth Runner statistics (avg inference time, thermal events) by default? Current `ScanLogger` schema from Plan 1 supports it but not all fields are populated. Keeping defaults as-is in 3a; expand in 3b if a pattern emerges.

**Open question**: Semantic wall-snapping is deferred to 3b. Some beds/dressers will appear detached from walls in 3a. Acceptable as MVP, might surface in QA as "furniture floating in the room." Worth noting so expectations are calibrated.

## Dependencies

- Apple: ARKit 6+ (iOS 16+), Core ML 7+ (iOS 17+), RealityKit 2+ (unchanged from Plan 1)
- Open-source models: Depth Anything V2 Metric (Hypersim fine-tune, Apache 2.0), YOLOv8s-seg (AGPL-3.0 — commercial licensing concern for future release, not blocker for 3a)
- Backend: existing SpatialLM 1.1 Qwen 0.5B on RunPod, FastAPI, Python 3
- No new third-party dependencies

## Out of scope (explicitly — tracked for Plan 3b and beyond)

- YOLO/SpatialLM consensus logic (dual-model detection merge)
- Semantic wall-snapping rules (beds → walls, nightstands → walls, etc.)
- Per-object confidence scoring + confidence-based visual indicators
- Object correction UI (tap to delete/rotate/move)
- Furniture library browser (manual Kenney model placement)
- AR scanning visual layer (grid shell, reticle, emerging furniture) — permanently deferred unless LiDAR becomes a supported path
- App size reduction (dropping either or both Core ML models)
- Security audit fixes (auth, HTTPS, hardcoded IP) — tracked separately in `memory/security_audit.md`
