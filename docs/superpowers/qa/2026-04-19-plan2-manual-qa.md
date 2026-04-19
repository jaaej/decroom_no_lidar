# Plan 2 Manual QA — Adapted for MVP (2026-04-19)

## Context

Plan 2's original QA checklist (Task 11) tested the visual AR experience: wireframe grid shell, capture reticle, solidifying walls. All 6 visual UI approaches were tried and failed (see `visual_ui_failures.md`). The app was reverted to route all devices through `VideoRecorderView` (commit `0682d32`).

This adapted QA validates what's actually shipping: the functional MVP with Plan 2's semantic/foundation code compiled in but visual AR views unused.

## Automated Verification (2026-04-19)

### Device Build
- **Result: PASS** — `xcodebuild build -destination 'generic/platform=iOS'` succeeds
- Both Core ML models compile: DepthModel.mlpackage, YOLOv8s-seg.mlpackage

### Test Suite
- **Result: PASS** — 56/56 tests pass on iPhone 17 Pro simulator
- Plan 1 tests (35): DeviceCapabilities (15), ScanLogger (4), DepthRunner (3), RoomGeometryBuilder (5), PointCloudAccumulator (4), KeyframeSelector (4)
- Plan 2 tests (21): DeviceInfoCurrent (3), IntrinsicsFromARCamera (3), ObjectTracker (6), ARRoomPreviewViewModel (5), CaptureReticle (3), existing baseline (1)
- No regressions from Plan 2 additions

## Adapted Device Checklist

Items adapted from the original Task 11 checklist:

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | App launches without crash | **Needs on-device test** | Build succeeds; requires physical device for runtime |
| 2 | Video capture tab → VideoRecorderView loads | **Needs on-device test** | All devices route here per commit 0682d32 |
| 3 | Coverage arc shows progress during recording | **Needs on-device test** | CoverageArcView renders filled/unfilled bins |
| 4 | Recording + upload flow completes | **Needs on-device test** | End-to-end with backend |
| 5 | No crash from dead AR code at compile time | **PASS** | Build succeeds; AR files compile but are unreferenced |
| 6 | Core ML models don't bloat app excessively | **PASS** | DepthModel ~50MB + YOLOv8s ~22MB = ~72MB added |
| 7 | Thermal behavior acceptable | **Needs on-device test** | ML models are not invoked in current flow (dead code) |

## Dead Code Inventory

These files were added in Plan 2 but are NOT wired into the active navigation flow:

| File | Purpose | Why Dead |
|------|---------|----------|
| `ARRoomPreviewView.swift` | Visual AR scanning UI | All visual approaches failed |
| `ARRoomPreviewContainer.swift` | UIViewRepresentable bridge | Depends on ARRoomPreviewView |
| `ARRoomPreviewViewModel.swift` | View logic (tested, 5 tests) | View not used |
| `GridShellEntity.swift` | Wireframe cube | Visual approach abandoned |
| `CaptureReticle.swift` | Guide reticle (tested, 3 tests) | Visual approach abandoned |
| `PanoramaBuffer.swift` | Equirectangular projection | Distortion issues |
| `ScanSession.swift` | Coordinator for all components | Only referenced by ARRoomPreviewView |
| `YoloRunner.swift` | YOLO inference wrapper | Only called by ScanSession |
| `YoloDetection.swift` | Detection struct | Only used by YoloRunner |
| `PixelBufferRGB.swift` | YUV→BGRA converter | Only called by ScanSession |

**Active Plan 2 code** (used by tests, will be consumed by Plan 3):
- `DeviceInfo+Current.swift` — factory for device info
- `Intrinsics+ARCamera.swift` — portrait-aware intrinsics
- `ObjectTracker.swift` — 3D instance tracking logic
- `YOLOv8s-seg.mlpackage` — bundled model (ready for Plan 3)

## Recommendations

1. **On-device test items 1-4** require manual testing with a physical iPhone — cannot be automated from CLI
2. **Dead code cleanup** is optional before merge. The dead files add ~72MB (models) and compile cleanly. They may be useful reference for Plan 3/4 or a future visual approach
3. **Core ML models are bundled but unused** — they add to app size but don't affect runtime. Consider removing YOLOv8s-seg.mlpackage if app size matters before Plan 3

## Decision Needed

Before Task 12 (merge), decide:
- **Option A**: Merge as-is — dead code ships, Plan 3 can revive it
- **Option B**: Remove dead AR view files, keep foundation code + models
- **Option C**: Remove dead AR view files AND unused models, re-add in Plan 3
