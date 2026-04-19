# Plan 3a — Phone-First Backend Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Plan 1 + Plan 2 phone-side components into the live scan flow, replace the upload payload with phone-built geometry + PLY + keyframes + log, switch the backend to a new phone-first route that runs SpatialLM on the phone PLY, delete the obsolete SfM/backend-depth/layout/lifting pipeline, and surface every failure as a named, specific `ScanError`.

**Architecture:** Two-layer integration. On iOS, `ScanSession` (already built in Plan 2) becomes a live consumer of `ARFrame` updates inside `GuidedARFrameRecorder`. On Done, `ScanSession.finalize()` produces a `ScanUploadPayload` that `UploadService.uploadPhoneFirst(_:)` POSTs to a new backend route `/api/upload/phone-first`. Backend runs SpatialLM on the phone-provided PLY, trusts phone walls, and returns the existing `RoomPayload` schema. No fallback — unsupported devices are gated at launch, and every failure mode has a named error case with a specific user-facing message.

**Tech Stack:** Swift 5.9+, SwiftUI, ARKit, Core ML, Swift Testing. Backend: Python 3, FastAPI, existing SpatialLM 1.1 Qwen 0.5B client. No new third-party dependencies.

**Design spec:** `docs/superpowers/specs/2026-04-19-plan3a-phone-first-backend-integration-design.md` — read first if you want the "why."

---

## Scope Boundaries

In Plan 3a:
- Wire `ScanSession` into `VideoRecorderView`
- Ship metric depth model (swap relative → `depth_anything_v2_metric_hypersim_small`)
- New upload payload + backend route
- `ScanError` enum with 12 named failure cases
- Device tier gate at app launch (iPhone X and older: scan disabled)
- Delete old SfM / backend-depth / PCA-layout / lifting / `_INSTANCE_CAPS` pipeline
- Delete Plan 2 dead code (`ARRoomPreviewView`, `GridShellEntity`, etc.)

Deferred to a future Plan 3b:
- YOLO/SpatialLM consensus (dual-model merge)
- Semantic wall-snapping rules (beds → walls, etc.)
- Per-object confidence scoring
- Correction UI (tap to delete/rotate)
- Furniture library browser
- Scan-guidance UX improvements

---

## File Structure

**New iOS source files** (all in `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/`):
- `ScanUploadPayload.swift` — struct owned by `ScanSession.finalize()` return
- `ScanError.swift` — enum with all 12 failure cases + user-facing messages

**Modified iOS source files:**
- `DepthRunner.swift` — strip "relative" language, no behavior change
- `PointCloudAccumulator.swift` — header comment update (no calibration code to remove — the current accumulator takes depth values directly and unprojects; no scale factor was ever inserted)
- `ScanSession.swift` — add `finalize() -> ScanUploadPayload`, add phone-side validation, add `scanID` exposed for error messages
- `ContentView.swift` — `GuidedARFrameRecorder` gets a `scanSession: ScanSession` field; `handle(frame:)` forwards to it; `stop()`/`stopVideoMode()` trigger `scanSession.finalize()`
- `UploadService.swift` — add `uploadPhoneFirst(_:) async throws -> RoomPayload`, delete `upload(images:)` + `uploadFromDisk(jpegPaths:)` (Phase G)
- `RecordingProcessor.swift` — new `processPhoneFirstPayload(_:)` method; delete `processKeyframesFromDisk` + `processCapturedFrames` (Phase G)
- `VideoRecorderView.swift` — upload() calls the new path; display `ScanError` alerts
- `CaptureFlowView.swift` — device tier gate (show "unsupported" screen for `.fallback`)

**Deleted iOS files** (Phase G):
- `ARRoomPreviewView.swift`, `ARRoomPreviewContainer.swift`, `ARRoomPreviewViewModel.swift`
- `GridShellEntity.swift`, `CaptureReticle.swift`
- `PanoramaBuffer.swift`
- Associated tests: `ARRoomPreviewViewModelTests.swift`, `CaptureReticleTests.swift`

**New iOS tests:**
- `ScanUploadPayloadTests.swift`
- `ScanErrorTests.swift`
- `ScanSessionFinalizeTests.swift`
- `UploadServicePhoneFirstTests.swift`
- `DeviceTierGateTests.swift` (if CaptureFlowView logic is factored out)

**Updated iOS tests:**
- `DepthRunnerTests.swift` — metric value range check
- `PointCloudAccumulatorTests.swift` — header comment refresh only

**Bundle resource** (replacing existing file):
- `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/DepthModel.mlpackage` — rebuilt from `depth_anything_v2_metric_hypersim_small`

**New backend files** (all in `decroom_no_lidar/decroom_no_lidar/backend/`):
- `pipeline/phone_first_pipeline.py` — new pipeline module
- `tests/__init__.py`
- `tests/test_phone_first_pipeline.py`
- `tests/test_upload_schema.py`
- `tests/test_category_mapping.py`
- `tests/fixtures/phone_first_payload/` (directory; populated in Task 5 Step 4)

**Modified backend files:**
- `main.py` — add `POST /api/upload/phone-first` route; Phase H deletes `POST /api/upload`
- `pipeline/spatiallm_layout.py` — reuse existing `SPATIALLM_TO_PRD` mapping (no code change)

**Deleted backend files** (Phase H):
- `pipeline/sfm.py`
- `pipeline/depth.py` (backend Depth Anything V2)
- `pipeline/layout.py` (PCA layout)
- `pipeline/lifting.py` (2D→3D backprojection)
- `pipeline/reconstruction.py` (ray triangulation, pose refinement)
- `pipeline/tracking.py` (ByteTrack)
- `pipeline/detection.py` (backend YOLO)
- `pipeline/detection_open_vocab.py` (YoloWorld)
- `pipeline/geometry.py` (global feature cloud)
- `pipeline/pipeline.py` (replaced by `phone_first_pipeline.py`)

## IMPORTANT: Xcode Synchronized File Groups

This project uses `PBXFileSystemSynchronizedRootGroup` — Xcode auto-discovers all `.swift` files in `decroom_no_lidar/`, `decroom_no_lidarTests/`, and `decroom_no_lidarUITests/`. **Do NOT manually edit `project.pbxproj` for `.swift` files.** Just create the file; it's immediately part of the build. Only the `.mlpackage` (Core ML bundle resource in Task 1) needs Xcode's UI to add, and only if its name changes. Since we keep the filename `DepthModel.mlpackage` and replace its contents, no pbxproj edit is needed for Task 1 either.

---

## Pre-Flight Check

Before starting, verify the starting state is clean.

- [ ] **Confirm branch state**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git checkout feature/phone-first-visual
git status
```

Expected: clean working tree, branch `feature/phone-first-visual` up to date with the Plan 2 + design spec commits (`b562ebf`, `2987ead`, `aad2ae7`).

- [ ] **Confirm simulator availability**

```bash
xcrun simctl list devices available | grep "iPhone 17 Pro"
```

Expected: at least one `iPhone 17 Pro` entry. (Per CLAUDE.md, iPhone 15 Pro is not installed; use iPhone 17 Pro throughout.)

- [ ] **Confirm baseline test suite green**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -10
```

Expected: `** TEST SUCCEEDED **`. 56 tests should pass. If any fail, fix before starting Plan 3a — don't build on red.

- [ ] **Confirm Python env for backend tests**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -c "import fastapi, numpy; print('env ok')"
```

Expected: `env ok`. Install with `pip install fastapi numpy pytest pytest-asyncio httpx` if missing.

---

## Task 1: Swap Depth Model to Metric Indoor Variant (Phase A)

**Purpose:** Replace the bundled `DepthAnythingV2SmallF16.mlpackage` (relative depth) with a Core ML conversion of `depth_anything_v2_metric_hypersim_small` (metric meters, indoor-tuned). This deletes the "needs scale calibration" risk at the root.

**Files:**
- Replace: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/DepthModel.mlpackage/` (contents replaced; filename unchanged)
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/DepthRunner.swift`
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/DepthRunnerTests.swift`

- [ ] **Step 1: Export the metric indoor model to Core ML**

```bash
cd /tmp
python3 -m venv depth-convert && source depth-convert/bin/activate
pip install torch torchvision coremltools huggingface_hub

python3 <<'PY'
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
import coremltools as ct

repo = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
processor = AutoImageProcessor.from_pretrained(repo)
model = AutoModelForDepthEstimation.from_pretrained(repo).eval()

class Wrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).predicted_depth

w = Wrapper(model).eval()
example = torch.rand(1, 3, 518, 518)
traced = torch.jit.trace(w, example, strict=False)

ml_model = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 518, 518),
                          scale=1/255.0, bias=[0, 0, 0])],
    outputs=[ct.TensorType(name="depth")],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
)
ml_model.short_description = "Depth Anything V2 Metric Indoor Small"
ml_model.save("/tmp/DepthModel.mlpackage")
print("saved /tmp/DepthModel.mlpackage")
PY
deactivate
```

Expected: `/tmp/DepthModel.mlpackage` exists, ~50 MB. If conversion fails due to unsupported ops, retry with `minimum_deployment_target=ct.target.iOS17`. If still failing, **stop and flag**: the fallback is to revert this task and keep the relative variant + add calibration code in Task 2 instead. Document the failure in the QA notes.

- [ ] **Step 2: Replace the bundled model**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar
rm -rf DepthModel.mlpackage
cp -r /tmp/DepthModel.mlpackage .
```

Verify:

```bash
du -sh DepthModel.mlpackage
```

Expected: ~40-60 MB. Bigger than ~80 MB → retry with `compute_precision=ct.precision.FLOAT16` (it should already be FP16; larger means something went wrong).

- [ ] **Step 3: Update DepthRunner.swift comments**

Edit `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/DepthRunner.swift`. Replace the file header (lines 1-9) with:

```swift
//
//  DepthRunner.swift
//  decroom_no_lidar
//
//  Wraps Depth Anything V2 Metric Indoor Small Core ML for on-device
//  metric depth inference. Output is in meters directly — no scale
//  calibration needed. Adjusts inference rate based on thermal state.
//
```

Replace the `DepthResult` struct's doc comment (lines 14-20):

```swift
struct DepthResult {
    let width: Int
    let height: Int
    /// Row-major width*height metric depth values in meters.
    let depths: [Float]
}
```

Replace the `infer` method's doc comment (line 46):

```swift
    /// Run inference on a pixel buffer. Returns metric depth map (meters)
    /// at the model's native resolution.
    func infer(pixelBuffer: CVPixelBuffer) throws -> DepthResult {
```

- [ ] **Step 4: Update DepthRunnerTests.swift**

Replace `depthsAreFinite` (around line 18) with a metric range check:

```swift
    @Test func depthsAreFiniteAndInMetricRange() throws {
        let runner = try DepthRunner()
        let buffer = makeTestPixelBuffer(width: 518, height: 392)
        let result = try runner.infer(pixelBuffer: buffer)
        // Metric depth: finite and within a plausible indoor range.
        // A zero-filled buffer may produce low-confidence values; we only
        // assert they are finite and non-negative. Real-scene data is
        // validated in manual QA, not here.
        #expect(result.depths.allSatisfy { $0.isFinite && $0 >= 0 })
    }
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/DepthRunnerTests 2>&1 | tail -15
```

Expected: all 3 tests pass. If `inferReturnsDepthMapWithConsistentDimensions` fails with a shape mismatch, the metric model may expect a different input shape than 518x392 — check the exported model's `inputDescriptionsByName` and update the test buffer size to match.

- [ ] **Step 6: Full build to confirm model loads**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 7: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/DepthModel.mlpackage \
        decroom_no_lidar/decroom_no_lidar/DepthRunner.swift \
        decroom_no_lidar/decroom_no_lidarTests/DepthRunnerTests.swift
git commit -m "$(cat <<'EOF'
feat: swap DepthModel to metric indoor variant

Replaces Depth Anything V2 Small (relative) with the
Metric Indoor Small (Hypersim fine-tune) variant. Output is
meters directly — deletes the scale-calibration step that was
specified but never implemented downstream. Same model family,
same iPhone Neural Engine speed, same ~50 MB bundle size.

Part of Plan 3a Phase A.
EOF
)"
```

---

## Task 2: Update PointCloudAccumulator Doc to Reflect Metric Depth (Phase A)

**Purpose:** The accumulator's code takes depth values and unprojects them directly — it never had a calibration step. All that changes here is the header comment saying depth is now metric, plus a quick test verifying unprojection math still holds for typical metric values.

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/PointCloudAccumulator.swift`

- [ ] **Step 1: Update file header**

Edit `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/PointCloudAccumulator.swift`. Replace lines 1-9:

```swift
//
//  PointCloudAccumulator.swift
//  decroom_no_lidar
//
//  Accumulates world-space 3D points from unprojected metric depth maps
//  (meters, from DepthRunner). Voxel-deduplicates at configurable grid
//  resolution. Caps total points to bound memory. Exports as binary PLY
//  for backend upload.
//
```

- [ ] **Step 2: Run existing accumulator tests**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/PointCloudAccumulatorTests 2>&1 | tail -10
```

Expected: all 4 tests pass. Tests use metric-ish values already (`depth: 2.0`) so no test change needed.

- [ ] **Step 3: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/PointCloudAccumulator.swift
git commit -m "docs: note PointCloudAccumulator consumes metric depth

Header now states depth input is in meters (from the Task 1 metric
model swap). No behavior change — unprojection math was already
correct for metric values; calibration step was specified in
Plan 1 but never implemented, so there is no code to remove.

Part of Plan 3a Phase A."
```

---

## Task 3: Wire ScanSession into GuidedARFrameRecorder (Phase B)

**Purpose:** `ScanSession` (built in Plan 2) currently runs only for the dead `ARRoomPreviewView`. Make `GuidedARFrameRecorder` own a `ScanSession` and forward every `ARFrame`, so the phone-first geometry/PLY/keyframe pipeline runs during real `VideoRecorderView` scans. No UX change — this is silent accumulation.

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ContentView.swift` (around lines 580-900 where `GuidedARFrameRecorder` lives)
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ScanSession.swift` (expose `scanID` and plane anchor forwarding)

- [ ] **Step 1: Expose scanID from ScanSession**

Open `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ScanSession.swift`. After the `logger` property (around line 21), no change needed — `logger.scanId` is already accessible. Add a computed property for convenience near the top of the class:

```swift
    /// Short scan identifier (hex, 8 chars) — useful for error messages.
    var scanID: String { logger.scanId }
```

(Add after the `@Published` declarations around line 40-45.)

- [ ] **Step 2: Add ScanSession field to GuidedARFrameRecorder**

Open `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ContentView.swift`. In the `GuidedARFrameRecorder` class (starts around line 584), add a `scanSession` property next to the other stored properties. Find the section with `private let ciContext = CIContext()` and add above it:

```swift
    /// On-device phone-first session. Accumulates geometry, point cloud,
    /// and keyframes in parallel with the existing video capture. Created
    /// lazily on start() so the heavy model loads don't happen at app launch.
    private(set) var scanSession: ScanSession?
```

- [ ] **Step 3: Create ScanSession in start()**

In the same file, find `func start()` (around line 673). At the very top of the method body, after the opening brace, add:

```swift
        scanSession = ScanSession()
```

Also in `stop()` (around line 707), after `isRecording = false`, add:

```swift
        // Keep scanSession alive — caller will invoke finalize() before releasing.
```

And in `stopVideoMode()` (around line 723), same thing.

- [ ] **Step 4: Forward ARFrame to ScanSession in handle(frame:)**

In `GuidedARFrameRecorder.handle(frame:)` (around line 895), at the very end of the method (just before the closing `}`), add:

```swift
        // Forward to phone-first pipeline (depth + geometry + keyframes)
        scanSession?.update(frame: frame)
```

- [ ] **Step 5: Forward plane anchors to ScanSession**

Find `upsertPlaneAnchors` (around line 794) and `removePlaneAnchors` (around line 801). Modify them:

```swift
    func upsertPlaneAnchors(_ anchors: [ARAnchor]) {
        for anchor in anchors {
            guard let planeAnchor = anchor as? ARPlaneAnchor else { continue }
            livePlaneAnchors[planeAnchor.identifier] = planeAnchor
        }
        // Also feed the phone-first room geometry builder
        let planes = anchors.compactMap { $0 as? ARPlaneAnchor }
        if !planes.isEmpty {
            scanSession?.updatePlaneAnchors(planes)
        }
    }

    func removePlaneAnchors(_ anchors: [ARAnchor]) {
        for anchor in anchors {
            guard let planeAnchor = anchor as? ARPlaneAnchor else { continue }
            livePlaneAnchors.removeValue(forKey: planeAnchor.identifier)
        }
        // Re-send the surviving set so the geometry builder re-derives walls.
        let surviving = Array(livePlaneAnchors.values)
        scanSession?.updatePlaneAnchors(surviving)
    }
```

- [ ] **Step 6: Build to confirm integration compiles**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 7: Run full test suite (no regressions)**

```bash
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -10
```

Expected: all pre-existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/ContentView.swift \
        decroom_no_lidar/decroom_no_lidar/ScanSession.swift
git commit -m "feat: wire ScanSession into GuidedARFrameRecorder

ScanSession is now created on start() and fed every ARFrame plus
plane anchor updates. Runs silently alongside the existing capture
logic — no UX change. Sets up Task 4's finalize() path which
produces the phone-first upload payload.

Part of Plan 3a Phase B."
```

---

## Task 4: ScanUploadPayload + ScanSession.finalize() (Phase B)

**Purpose:** Serialize the ScanSession's accumulated state (walls, PLY, keyframes, log) into a single struct suitable for upload. Validate pre-upload (≥1 wall, ≥50% yaw coverage) and return a named `ScanError` if validation fails.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ScanError.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ScanUploadPayload.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/ScanErrorTests.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/ScanUploadPayloadTests.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/ScanSessionFinalizeTests.swift`
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ScanSession.swift`

- [ ] **Step 1: Create ScanError.swift**

```swift
//
//  ScanError.swift
//  decroom_no_lidar
//
//  Named failure cases for the phone-first scan + upload flow. Every
//  case carries a user-facing message (the exact string shown in the
//  alert) and an optional technical detail (logged, not displayed).
//

import Foundation

enum ScanError: Error, Equatable {
    case unsupportedDevice
    case depthModelLoadFailed(detail: String)
    case arKitTrackingLimited
    case arKitSessionFailed(detail: String)
    case noWallsDetected
    case insufficientCoverage(binsFilled: Int, binsRequired: Int)
    case payloadBuildFailed(detail: String)
    case uploadNoServer
    case uploadTimeout
    case backendSchemaError(scanID: String, detail: String)
    case backendServerError(status: Int, detail: String)
    case noFurnitureDetected   // informational, not an error — surfaced as info toast

    /// Exact string shown in the user-facing alert.
    var message: String {
        switch self {
        case .unsupportedDevice:
            return "This iPhone doesn't support Decroom's scanning. Minimum: iPhone XS."
        case .depthModelLoadFailed:
            return "Can't start scan: the depth engine isn't available. Please reinstall the app."
        case .arKitTrackingLimited:
            return "Lost tracking — slow down, more light."
        case .arKitSessionFailed:
            return "ARKit stopped working. Restart the app."
        case .noWallsDetected:
            return "No walls found. Point at the walls of your room and keep scanning."
        case .insufficientCoverage:
            return "Only part of the room was captured. Rotate 360° and try again."
        case .payloadBuildFailed:
            return "Couldn't save scan data. Check phone storage and try again."
        case .uploadNoServer:
            return "Can't reach the Decroom server. Check WiFi and tap Retry."
        case .uploadTimeout:
            return "Server took too long. Try again in a moment."
        case .backendSchemaError(let id, _):
            return "Scan data was rejected. Please report this. Scan ID: \(id)"
        case .backendServerError:
            return "Server error. Try again in a moment."
        case .noFurnitureDetected:
            return "Scan complete — no furniture found."
        }
    }

    /// Is this error recoverable by tapping Retry (same button, re-attempt)?
    var isRetryable: Bool {
        switch self {
        case .payloadBuildFailed,
             .uploadNoServer,
             .uploadTimeout,
             .backendServerError:
            return true
        case .noWallsDetected, .insufficientCoverage:
            return true  // "Resume scan" rather than "Retry upload"
        case .unsupportedDevice,
             .depthModelLoadFailed,
             .arKitSessionFailed,
             .backendSchemaError,
             .arKitTrackingLimited,
             .noFurnitureDetected:
            return false
        }
    }

    /// Technical detail for scan log / debugging. Never shown to the user.
    var detail: String {
        switch self {
        case .unsupportedDevice:
            return "device tier .fallback"
        case .depthModelLoadFailed(let d), .arKitSessionFailed(let d),
             .payloadBuildFailed(let d):
            return d
        case .arKitTrackingLimited:
            return "ARKit trackingState reported limited"
        case .noWallsDetected:
            return "RoomGeometry.walls.isEmpty at finalize"
        case .insufficientCoverage(let filled, let req):
            return "yaw bins filled=\(filled), required=\(req)"
        case .uploadNoServer:
            return "No reachable backend among candidate URLs"
        case .uploadTimeout:
            return "URLSession timed out"
        case .backendSchemaError(_, let d):
            return "HTTP 4xx: \(d)"
        case .backendServerError(let status, let d):
            return "HTTP \(status): \(d)"
        case .noFurnitureDetected:
            return "SpatialLM returned 0 objects"
        }
    }
}
```

- [ ] **Step 2: Create ScanErrorTests.swift**

```swift
import Testing
@testable import decroom_no_lidar

struct ScanErrorTests {

    @Test func everyCaseHasNonEmptyMessage() {
        let cases: [ScanError] = [
            .unsupportedDevice,
            .depthModelLoadFailed(detail: "x"),
            .arKitTrackingLimited,
            .arKitSessionFailed(detail: "x"),
            .noWallsDetected,
            .insufficientCoverage(binsFilled: 0, binsRequired: 12),
            .payloadBuildFailed(detail: "x"),
            .uploadNoServer,
            .uploadTimeout,
            .backendSchemaError(scanID: "abc", detail: "x"),
            .backendServerError(status: 500, detail: "x"),
            .noFurnitureDetected,
        ]
        for c in cases {
            #expect(!c.message.isEmpty, "\(c) has empty message")
        }
    }

    @Test func unsupportedDeviceMessageMentionsMinimumDevice() {
        #expect(ScanError.unsupportedDevice.message.contains("iPhone XS"))
    }

    @Test func backendSchemaErrorShowsScanID() {
        let e = ScanError.backendSchemaError(scanID: "deadbeef", detail: "bad json")
        #expect(e.message.contains("deadbeef"))
    }

    @Test func retryableCasesAreCorrect() {
        #expect(ScanError.uploadNoServer.isRetryable)
        #expect(ScanError.uploadTimeout.isRetryable)
        #expect(ScanError.backendServerError(status: 500, detail: "").isRetryable)
        #expect(ScanError.noWallsDetected.isRetryable)
        #expect(ScanError.insufficientCoverage(binsFilled: 0, binsRequired: 12).isRetryable)
        #expect(!ScanError.unsupportedDevice.isRetryable)
        #expect(!ScanError.depthModelLoadFailed(detail: "").isRetryable)
        #expect(!ScanError.backendSchemaError(scanID: "x", detail: "y").isRetryable)
    }
}
```

- [ ] **Step 3: Create ScanUploadPayload.swift**

```swift
//
//  ScanUploadPayload.swift
//  decroom_no_lidar
//
//  Container for everything the phone hands to the backend in a
//  phone-first upload: walls/floor/ceiling JSON, binary PLY point
//  cloud, 10 keyframe JPEGs + per-frame metadata, and the scan log.
//

import Foundation
import simd

struct ScanUploadPayload {
    struct KeyframeMetadata {
        let filename: String
        let timestamp: Double
        let poseWorldFromCamera: simd_float4x4
        let intrinsics: Intrinsics
        // detections intentionally empty in 3a (YOLO output goes to scan_log only)
    }

    let scanID: String
    let roomGeometryJSON: Data
    let pointCloudPLY: Data
    let keyframes: [(jpeg: Data, meta: KeyframeMetadata)]
    let scanLogJSON: Data
    let deviceInfo: DeviceInfo

    /// Matches room_geometry.json schema from the design spec.
    struct RoomGeometryJSON: Codable {
        struct Point: Codable { let x: Float; let z: Float }
        struct WallEntry: Codable {
            let start: Point
            let end: Point
        }
        let walls: [WallEntry]
        let floor_height: Float
        let ceiling_height: Float
        let manhattan_yaw: Float
        let coordinate_system: String
    }

    /// Matches keyframes.json manifest schema.
    struct KeyframesManifest: Codable {
        struct Intrinsics: Codable {
            let fx: Float
            let fy: Float
            let cx: Float
            let cy: Float
            let width: Int
            let height: Int
        }
        struct FrameEntry: Codable {
            let filename: String
            let timestamp: Double
            /// 16 floats, row-major 4x4 pose matrix (world_from_camera).
            let pose_world_from_camera: [Float]
            let intrinsics: Intrinsics
            let detections: [String]  // always [] in 3a
        }
        let frames: [FrameEntry]
    }

    // MARK: - JSON serialization helpers

    static func encodeRoomGeometry(_ geometry: RoomGeometry) throws -> Data {
        let json = RoomGeometryJSON(
            walls: geometry.walls.map {
                RoomGeometryJSON.WallEntry(
                    start: .init(x: $0.start.x, z: $0.start.y),
                    end: .init(x: $0.end.x, z: $0.end.y)
                )
            },
            floor_height: geometry.floorHeight,
            ceiling_height: geometry.ceilingHeight,
            manhattan_yaw: geometry.manhattanYaw,
            coordinate_system: "arkit_world_right_handed_y_up"
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        return try encoder.encode(json)
    }

    static func encodeKeyframesManifest(_ frames: [KeyframeMetadata]) throws -> Data {
        let entries = frames.map { kf -> KeyframesManifest.FrameEntry in
            let p = kf.poseWorldFromCamera
            // Row-major flatten of 4x4 (columns are stored contiguous in simd, but
            // JSON should be row-major for server-side convenience).
            let flat: [Float] = [
                p.columns.0.x, p.columns.1.x, p.columns.2.x, p.columns.3.x,
                p.columns.0.y, p.columns.1.y, p.columns.2.y, p.columns.3.y,
                p.columns.0.z, p.columns.1.z, p.columns.2.z, p.columns.3.z,
                p.columns.0.w, p.columns.1.w, p.columns.2.w, p.columns.3.w,
            ]
            return KeyframesManifest.FrameEntry(
                filename: kf.filename,
                timestamp: kf.timestamp,
                pose_world_from_camera: flat,
                intrinsics: .init(
                    fx: kf.intrinsics.fx, fy: kf.intrinsics.fy,
                    cx: kf.intrinsics.cx, cy: kf.intrinsics.cy,
                    width: kf.intrinsics.width, height: kf.intrinsics.height
                ),
                detections: []
            )
        }
        let manifest = KeyframesManifest(frames: entries)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        return try encoder.encode(manifest)
    }
}
```

- [ ] **Step 4: Create ScanUploadPayloadTests.swift**

```swift
import Testing
import simd
import Foundation
@testable import decroom_no_lidar

struct ScanUploadPayloadTests {

    @Test func encodeRoomGeometryProducesParseableJSON() throws {
        let g = RoomGeometry(
            walls: [WallSegment(start: SIMD2(0, 0), end: SIMD2(3, 0), height: 2.5)],
            floorHeight: 0,
            ceilingHeight: 2.4,
            manhattanYaw: 0
        )
        let data = try ScanUploadPayload.encodeRoomGeometry(g)
        let decoded = try JSONDecoder().decode(
            ScanUploadPayload.RoomGeometryJSON.self, from: data
        )
        #expect(decoded.walls.count == 1)
        #expect(decoded.walls[0].start.x == 0)
        #expect(decoded.walls[0].end.x == 3)
        #expect(decoded.floor_height == 0)
        #expect(decoded.ceiling_height == 2.4)
        #expect(decoded.coordinate_system == "arkit_world_right_handed_y_up")
    }

    @Test func encodeKeyframesManifestFlattensPose() throws {
        let intr = Intrinsics(fx: 1500, fy: 1500, cx: 720, cy: 960, width: 1440, height: 1920)
        let pose = matrix_identity_float4x4
        let meta = ScanUploadPayload.KeyframeMetadata(
            filename: "frame_00.jpg",
            timestamp: 1.23,
            poseWorldFromCamera: pose,
            intrinsics: intr
        )
        let data = try ScanUploadPayload.encodeKeyframesManifest([meta])
        let decoded = try JSONDecoder().decode(
            ScanUploadPayload.KeyframesManifest.self, from: data
        )
        #expect(decoded.frames.count == 1)
        #expect(decoded.frames[0].filename == "frame_00.jpg")
        #expect(decoded.frames[0].pose_world_from_camera.count == 16)
        // Identity matrix row-major
        #expect(decoded.frames[0].pose_world_from_camera[0] == 1)
        #expect(decoded.frames[0].pose_world_from_camera[5] == 1)
        #expect(decoded.frames[0].pose_world_from_camera[10] == 1)
        #expect(decoded.frames[0].pose_world_from_camera[15] == 1)
        #expect(decoded.frames[0].intrinsics.fx == 1500)
    }

    @Test func manifestAlwaysHasEmptyDetectionsIn3a() throws {
        let intr = Intrinsics(fx: 1500, fy: 1500, cx: 720, cy: 960, width: 1440, height: 1920)
        let meta = ScanUploadPayload.KeyframeMetadata(
            filename: "frame_00.jpg",
            timestamp: 0,
            poseWorldFromCamera: matrix_identity_float4x4,
            intrinsics: intr
        )
        let data = try ScanUploadPayload.encodeKeyframesManifest([meta])
        let decoded = try JSONDecoder().decode(
            ScanUploadPayload.KeyframesManifest.self, from: data
        )
        #expect(decoded.frames[0].detections.isEmpty)
    }
}
```

- [ ] **Step 5: Add finalize() to ScanSession**

Open `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ScanSession.swift`. Replace the existing `finalize()` method (around line 154) with:

```swift
    /// Produce the phone-first upload payload. Runs phone-side validation;
    /// throws a named `ScanError` if the scan is not uploadable.
    /// - Parameter keyframes: keyframe JPEGs + metadata assembled by the
    ///   caller (GuidedARFrameRecorder owns the JPEG files and pose stream).
    /// - Parameter totalYawBins: total bins the coverage engine split the
    ///   360° rotation into. Currently 24 (15° each).
    /// - Parameter requiredBins: minimum filled bins for a valid scan.
    ///   Currently 12 (50%).
    func finalize(
        keyframes: [ScanUploadPayload.KeyframeMetadata],
        keyframeJPEGs: [Data],
        totalYawBins: Int = 24,
        requiredBins: Int = 12
    ) throws -> ScanUploadPayload {
        logger.logEvent("scan_finalize_started", at: elapsed())

        // Surface .depthModelLoadFailed if the depth model never loaded.
        // (ScanSession.init uses try? for DepthRunner, so a load failure is
        // silent until finalize — which is when we need to tell the user.)
        guard depthRunner != nil else {
            logger.logEvent("scan_finalize_rejected_depth_missing", at: elapsed())
            throw ScanError.depthModelLoadFailed(
                detail: "DepthRunner init returned nil — model bundle missing or corrupt"
            )
        }

        // Phone-side validation.
        guard !geometry.walls.isEmpty else {
            logger.logEvent("scan_finalize_rejected_no_walls", at: elapsed())
            throw ScanError.noWallsDetected
        }
        guard visitedYawBins.count >= requiredBins else {
            logger.logEvent("scan_finalize_rejected_coverage", at: elapsed(),
                            data: ["filled": "\(visitedYawBins.count)",
                                   "required": "\(requiredBins)"])
            throw ScanError.insufficientCoverage(
                binsFilled: visitedYawBins.count,
                binsRequired: requiredBins
            )
        }
        guard keyframes.count == keyframeJPEGs.count else {
            throw ScanError.payloadBuildFailed(detail: "keyframe metadata/jpeg count mismatch")
        }

        // Serialize components. Errors here → payloadBuildFailed.
        let roomGeometryJSON: Data
        let keyframesManifestJSON: Data
        do {
            roomGeometryJSON = try ScanUploadPayload.encodeRoomGeometry(geometry)
            keyframesManifestJSON = try ScanUploadPayload.encodeKeyframesManifest(keyframes)
        } catch {
            throw ScanError.payloadBuildFailed(detail: error.localizedDescription)
        }

        let plyData = pointCloud.exportPLY()

        // Write log to disk (also keeps Plan 1's retention behavior).
        let logData: Data
        do {
            try logger.finalize()
            // Read back what we wrote for inclusion in the payload.
            let dir = ScanLogger.defaultDirectory()
            let file = "\(Int(logger.startedAt.timeIntervalSince1970))_\(logger.scanId).json"
            logData = try Data(contentsOf: dir.appendingPathComponent(file))
        } catch {
            throw ScanError.payloadBuildFailed(detail: "scan log write: \(error.localizedDescription)")
        }

        // Bundle keyframes manifest inside keyframeJPEGs? No — manifest is a
        // separate part in the multipart body (UploadService encodes it).
        _ = keyframesManifestJSON  // referenced by caller via repack step in UploadService

        logger.logEvent("scan_finalize_succeeded", at: elapsed(),
                        data: ["walls": "\(geometry.walls.count)",
                               "points": "\(pointCloud.count)",
                               "keyframes": "\(keyframes.count)"])

        return ScanUploadPayload(
            scanID: logger.scanId,
            roomGeometryJSON: roomGeometryJSON,
            pointCloudPLY: plyData,
            keyframes: zip(keyframeJPEGs, keyframes).map { ($0.0, $0.1) },
            scanLogJSON: logData,
            deviceInfo: DeviceInfo.current()
        )
    }
```

Also delete the old `finalize()` method signature that returned `(point_cloud_ply: Data, log_written: Bool)` if it's still present — the new signature replaces it.

- [ ] **Step 6: Create ScanSessionFinalizeTests.swift**

```swift
import Testing
import simd
@testable import decroom_no_lidar

@MainActor
struct ScanSessionFinalizeTests {

    @Test func finalizeWithNoWallsThrowsNoWallsDetected() async throws {
        let session = ScanSession()
        let intr = Intrinsics(fx: 1500, fy: 1500, cx: 720, cy: 960, width: 1440, height: 1920)
        let meta = ScanUploadPayload.KeyframeMetadata(
            filename: "frame_00.jpg",
            timestamp: 0,
            poseWorldFromCamera: matrix_identity_float4x4,
            intrinsics: intr
        )
        // Fill yaw coverage so the only failure is walls.
        for b in 0..<24 { session.visitedYawBins.insert(b) }

        do {
            _ = try session.finalize(keyframes: [meta], keyframeJPEGs: [Data()])
            Issue.record("Expected .noWallsDetected to throw")
        } catch let e as ScanError {
            #expect(e == .noWallsDetected)
        }
    }

    @Test func finalizeWithInsufficientCoverageThrows() async throws {
        let session = ScanSession()
        // Manually set geometry with ≥1 wall
        session.geometry = RoomGeometry(
            walls: [WallSegment(start: SIMD2(0, 0), end: SIMD2(3, 0), height: 2.5)],
            floorHeight: 0, ceilingHeight: 2.4, manhattanYaw: 0
        )
        // Only 5 bins filled — below the 12 required
        for b in 0..<5 { session.visitedYawBins.insert(b) }

        let intr = Intrinsics(fx: 1500, fy: 1500, cx: 720, cy: 960, width: 1440, height: 1920)
        let meta = ScanUploadPayload.KeyframeMetadata(
            filename: "frame_00.jpg",
            timestamp: 0,
            poseWorldFromCamera: matrix_identity_float4x4,
            intrinsics: intr
        )
        do {
            _ = try session.finalize(keyframes: [meta], keyframeJPEGs: [Data()])
            Issue.record("Expected .insufficientCoverage to throw")
        } catch let e as ScanError {
            if case .insufficientCoverage(let filled, let req) = e {
                #expect(filled == 5)
                #expect(req == 12)
            } else {
                Issue.record("Expected .insufficientCoverage, got \(e)")
            }
        }
    }

    @Test func finalizeWithKeyframeCountMismatchThrows() async throws {
        let session = ScanSession()
        session.geometry = RoomGeometry(
            walls: [WallSegment(start: SIMD2(0, 0), end: SIMD2(3, 0), height: 2.5)],
            floorHeight: 0, ceilingHeight: 2.4, manhattanYaw: 0
        )
        for b in 0..<24 { session.visitedYawBins.insert(b) }

        // Two metadata entries but one jpeg → mismatch.
        let intr = Intrinsics(fx: 1500, fy: 1500, cx: 720, cy: 960, width: 1440, height: 1920)
        let m1 = ScanUploadPayload.KeyframeMetadata(
            filename: "frame_00.jpg", timestamp: 0,
            poseWorldFromCamera: matrix_identity_float4x4, intrinsics: intr
        )
        let m2 = ScanUploadPayload.KeyframeMetadata(
            filename: "frame_01.jpg", timestamp: 1,
            poseWorldFromCamera: matrix_identity_float4x4, intrinsics: intr
        )
        do {
            _ = try session.finalize(keyframes: [m1, m2], keyframeJPEGs: [Data()])
            Issue.record("Expected .payloadBuildFailed")
        } catch let e as ScanError {
            if case .payloadBuildFailed = e {
                // ok
            } else {
                Issue.record("Expected .payloadBuildFailed, got \(e)")
            }
        }
    }
}
```

Note: `geometry` and `visitedYawBins` must be publicly settable for tests, or expose `@testable` fileprivate setters. If the current `@Published` declarations are already writable from same-module test targets, the code above works as-is. If not, add `@testable` `internal(set)` exposure in `ScanSession.swift`.

- [ ] **Step 7: Make ScanSession properties testable**

In `ScanSession.swift`, ensure the `@Published` properties are writable from the test target. Change (around line 34):

```swift
    @Published var geometry: RoomGeometry = RoomGeometry(
        walls: [], floorHeight: 0, ceilingHeight: 0, manhattanYaw: 0
    )
```

to ensure no `private(set)` qualifier (keep as-is if already plain `var`). Same for `visitedYawBins`. These are `var` already, so tests in the same module can write them via `@testable`.

- [ ] **Step 8: Run the new tests**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -only-testing:decroom_no_lidarTests/ScanErrorTests \
  -only-testing:decroom_no_lidarTests/ScanUploadPayloadTests \
  -only-testing:decroom_no_lidarTests/ScanSessionFinalizeTests 2>&1 | tail -20
```

Expected: all new tests pass.

- [ ] **Step 9: Full suite regression check**

```bash
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/ScanError.swift \
        decroom_no_lidar/decroom_no_lidar/ScanUploadPayload.swift \
        decroom_no_lidar/decroom_no_lidar/ScanSession.swift \
        decroom_no_lidar/decroom_no_lidarTests/ScanErrorTests.swift \
        decroom_no_lidar/decroom_no_lidarTests/ScanUploadPayloadTests.swift \
        decroom_no_lidar/decroom_no_lidarTests/ScanSessionFinalizeTests.swift
git commit -m "feat: add ScanUploadPayload + ScanSession.finalize()

ScanError enum with 12 named failure cases (each with
user-facing message + technical detail). ScanUploadPayload
bundles room geometry JSON + PLY + keyframe manifest + scan
log. ScanSession.finalize() runs phone-side validation
(>= 1 wall, >= 12/24 yaw bins) and throws named errors on
failure.

Part of Plan 3a Phase B."
```

---

## Task 5: UploadService.uploadPhoneFirst (Phase C)

**Purpose:** Add the new upload method that POSTs the phone-first multipart body to `/api/upload/phone-first`. Keep the old methods alive (deleted later in Phase G).

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/UploadService.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/UploadServicePhoneFirstTests.swift`
- Create fixture directory: `tests/fixtures/phone_first_payload/` (at repo root)

- [ ] **Step 1: Add uploadPhoneFirst method**

Open `UploadService.swift`. After the existing `uploadFromDisk` method (end of struct around line 291), but **before** the closing `}` of the struct, insert:

```swift
    /// Upload a phone-first payload: geometry JSON, PLY, keyframes + manifest,
    /// scan log. Maps backend 4xx/5xx/network errors into `ScanError` cases.
    static func uploadPhoneFirst(_ payload: ScanUploadPayload) async throws -> RoomPayload {
        let url = try await resolvePhoneFirstUploadURL()
        print("[UploadService] Phone-first upload → \(url.absoluteString) (scan=\(payload.scanID))")

        let boundary = UUID().uuidString

        // Build the keyframes.json manifest from the payload's keyframe metas.
        let manifestJSON: Data
        do {
            manifestJSON = try ScanUploadPayload.encodeKeyframesManifest(
                payload.keyframes.map { $0.meta }
            )
        } catch {
            throw ScanError.payloadBuildFailed(detail: "manifest encode: \(error.localizedDescription)")
        }

        // Build multipart body off-thread (JPEG concat is cheap but keep main responsive).
        let body: Data = await Task.detached(priority: .userInitiated) {
            var data = Data()

            func appendPart(name: String, filename: String,
                             contentType: String, payload: Data) {
                data.append("--\(boundary)\r\n")
                data.append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n")
                data.append("Content-Type: \(contentType)\r\n\r\n")
                data.append(payload)
                data.append("\r\n")
            }

            appendPart(name: "room_geometry", filename: "room_geometry.json",
                        contentType: "application/json",
                        payload: payload.roomGeometryJSON)
            appendPart(name: "point_cloud", filename: "point_cloud.ply",
                        contentType: "application/octet-stream",
                        payload: payload.pointCloudPLY)
            appendPart(name: "keyframes_manifest", filename: "keyframes.json",
                        contentType: "application/json",
                        payload: manifestJSON)
            for (i, kf) in payload.keyframes.enumerated() {
                appendPart(name: "keyframes", filename: kf.meta.filename.isEmpty
                                                        ? "frame_\(i).jpg"
                                                        : kf.meta.filename,
                            contentType: "image/jpeg",
                            payload: kf.jpeg)
            }
            appendPart(name: "scan_log", filename: "scan_log.json",
                        contentType: "application/json",
                        payload: payload.scanLogJSON)

            data.append("--\(boundary)--\r\n")
            return data
        }.value

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 60  // was 1800 for old SfM path; 60s is plenty for SpatialLM
        request.httpBody = body

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 60
        let session = URLSession(configuration: config)
        defer { session.finishTasksAndInvalidate() }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch let err as URLError where err.code == .timedOut {
            throw ScanError.uploadTimeout
        } catch {
            throw ScanError.uploadNoServer
        }

        guard let http = response as? HTTPURLResponse else {
            throw ScanError.backendServerError(status: -1, detail: "non-HTTP response")
        }

        switch http.statusCode {
        case 200:
            do {
                return try JSONDecoder().decode(RoomPayload.self, from: data)
            } catch {
                throw ScanError.backendSchemaError(
                    scanID: payload.scanID,
                    detail: "response decode: \(error.localizedDescription)"
                )
            }
        case 400..<500:
            let body = String(data: data, encoding: .utf8) ?? "(unparseable body)"
            throw ScanError.backendSchemaError(scanID: payload.scanID, detail: body)
        default:
            let body = String(data: data, encoding: .utf8) ?? "(unparseable body)"
            throw ScanError.backendServerError(status: http.statusCode, detail: body)
        }
    }

    /// Resolves the phone-first upload URL (Bonjour first, fallback second).
    /// Swaps the path component to `/api/upload/phone-first`.
    private static func resolvePhoneFirstUploadURL() async throws -> URL {
        let base = try await resolveUploadURL()
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false) else {
            throw ScanError.uploadNoServer
        }
        components.path = "/api/upload/phone-first"
        components.query = nil
        guard let url = components.url else {
            throw ScanError.uploadNoServer
        }
        return url
    }
```

- [ ] **Step 2: Create the fixture directory structure**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
mkdir -p tests/fixtures/phone_first_payload
```

The fixture is populated with a real scan's payload in Phase I Task 14 (manual QA) — for now, the directory exists so tests can reference a relative path without a fresh failure. Add a README:

```bash
cat > tests/fixtures/phone_first_payload/README.md <<'EOF'
# Phone-First Payload Fixture

Populated in Plan 3a Phase I from a real physical-device scan.
Contents (expected): room_geometry.json, point_cloud.ply,
keyframes/frame_NN.jpg (×10), keyframes.json, scan_log.json.

Used by both Swift (UploadServicePhoneFirstTests) and Python
(test_phone_first_pipeline.py) contract tests to detect schema drift.
EOF
```

- [ ] **Step 3: Create UploadServicePhoneFirstTests.swift**

```swift
import Testing
import Foundation
import simd
@testable import decroom_no_lidar

struct UploadServicePhoneFirstTests {

    @Test func multipartBodyContainsAllRequiredParts() async throws {
        // Build a minimal valid payload.
        let intr = Intrinsics(fx: 1500, fy: 1500, cx: 720, cy: 960, width: 1440, height: 1920)
        let meta = ScanUploadPayload.KeyframeMetadata(
            filename: "frame_00.jpg",
            timestamp: 1.23,
            poseWorldFromCamera: matrix_identity_float4x4,
            intrinsics: intr
        )
        let payload = ScanUploadPayload(
            scanID: "testscan",
            roomGeometryJSON: #"{"walls":[]}"#.data(using: .utf8)!,
            pointCloudPLY: Data("ply".utf8),
            keyframes: [(Data("jpeg-bytes".utf8), meta)],
            scanLogJSON: Data("{}".utf8),
            deviceInfo: DeviceInfo(model: "test", ios: "17.0",
                                    chip: "test", tier: .full)
        )

        // We can't easily intercept URLSession in unit tests without a protocol
        // abstraction over UploadService, so instead we call the multipart
        // build helper directly via a package-internal hook. Add one below.
        let body = try await UploadService._buildPhoneFirstMultipartBody(
            payload, boundary: "XXX"
        )
        let bodyStr = String(data: body, encoding: .utf8) ?? ""
        #expect(bodyStr.contains(#"name="room_geometry""#))
        #expect(bodyStr.contains(#"name="point_cloud""#))
        #expect(bodyStr.contains(#"name="keyframes_manifest""#))
        #expect(bodyStr.contains(#"name="keyframes""#))
        #expect(bodyStr.contains(#"name="scan_log""#))
        #expect(bodyStr.hasSuffix("--XXX--\r\n"))
    }
}
```

- [ ] **Step 4: Extract multipart build helper for testability**

In `UploadService.swift`, refactor the multipart body build from Step 1 into a static helper so the test can call it directly. Replace the `body: Data = await Task.detached…` block with:

```swift
        let body: Data = await Task.detached(priority: .userInitiated) {
            try? UploadService._buildPhoneFirstMultipartBody(payload, boundary: boundary)
        }.value ?? Data()

        if body.isEmpty {
            throw ScanError.payloadBuildFailed(detail: "multipart body build returned empty")
        }
```

And add at the bottom of the `UploadService` struct (but before the closing `}`):

```swift
    /// Exposed for tests — builds the multipart body without a network call.
    static func _buildPhoneFirstMultipartBody(
        _ payload: ScanUploadPayload,
        boundary: String
    ) throws -> Data {
        var data = Data()
        func appendPart(name: String, filename: String,
                         contentType: String, bytes: Data) {
            data.append("--\(boundary)\r\n")
            data.append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n")
            data.append("Content-Type: \(contentType)\r\n\r\n")
            data.append(bytes)
            data.append("\r\n")
        }

        let manifestJSON = try ScanUploadPayload.encodeKeyframesManifest(
            payload.keyframes.map { $0.meta }
        )

        appendPart(name: "room_geometry", filename: "room_geometry.json",
                    contentType: "application/json",
                    bytes: payload.roomGeometryJSON)
        appendPart(name: "point_cloud", filename: "point_cloud.ply",
                    contentType: "application/octet-stream",
                    bytes: payload.pointCloudPLY)
        appendPart(name: "keyframes_manifest", filename: "keyframes.json",
                    contentType: "application/json",
                    bytes: manifestJSON)
        for (i, kf) in payload.keyframes.enumerated() {
            appendPart(name: "keyframes", filename: kf.meta.filename.isEmpty
                                                     ? "frame_\(i).jpg"
                                                     : kf.meta.filename,
                        contentType: "image/jpeg",
                        bytes: kf.jpeg)
        }
        appendPart(name: "scan_log", filename: "scan_log.json",
                    contentType: "application/json",
                    bytes: payload.scanLogJSON)

        data.append("--\(boundary)--\r\n")
        return data
    }
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -only-testing:decroom_no_lidarTests/UploadServicePhoneFirstTests 2>&1 | tail -10
```

Expected: `** TEST SUCCEEDED **`.

- [ ] **Step 6: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/UploadService.swift \
        decroom_no_lidar/decroom_no_lidarTests/UploadServicePhoneFirstTests.swift \
        tests/fixtures/phone_first_payload/README.md
git commit -m "feat: add UploadService.uploadPhoneFirst

New multipart POST to /api/upload/phone-first with:
room_geometry, point_cloud, keyframes_manifest, keyframes
(×N), scan_log. 60s timeout. Maps URL/HTTP errors to named
ScanError cases (uploadNoServer, uploadTimeout,
backendSchemaError, backendServerError).

Old upload(images:) and uploadFromDisk(jpegPaths:) remain
live — they are deleted in Phase G once the cutover commits
and we confirm nothing still calls them.

Part of Plan 3a Phase C."
```

---

## Task 6: Backend — phone_first_pipeline + new route (Phase D)

**Purpose:** Server-side handler for the phone-first payload. Parses the multipart, loads the PLY, runs SpatialLM (same wrapper already used today), maps SpatialLM categories through the existing `SPATIALLM_TO_PRD` dict, and returns the unchanged `RoomPayload` JSON schema.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/backend/pipeline/phone_first_pipeline.py`
- Create: `decroom_no_lidar/decroom_no_lidar/backend/tests/__init__.py`
- Create: `decroom_no_lidar/decroom_no_lidar/backend/tests/test_phone_first_pipeline.py`
- Create: `decroom_no_lidar/decroom_no_lidar/backend/tests/test_upload_schema.py`
- Create: `decroom_no_lidar/decroom_no_lidar/backend/tests/test_category_mapping.py`
- Modify: `decroom_no_lidar/decroom_no_lidar/backend/main.py`

- [ ] **Step 1: Create phone_first_pipeline.py**

```python
"""
Phone-first pipeline: consumes the ScanUploadPayload produced by the iOS app
and returns a RoomPayload. Replaces the heavy SfM/backend-depth/layout/lifting
path from pipeline.py for devices that send phone-built geometry.

Flow:
  1. Parse multipart payload into structured objects
  2. Load phone-provided PLY into a numpy point cloud
  3. Run SpatialLM on the phone PLY
  4. Cross-check SpatialLM walls vs phone walls (keep phone, log disagreement)
  5. Map SpatialLM 59 categories → Decroom 14 (via SPATIALLM_TO_PRD)
  6. Build RoomPayload
"""
from __future__ import annotations

import json
import logging
import math
import struct
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .config import PipelineConfig
from .spatiallm_layout import SPATIALLM_TO_PRD, run_spatiallm_pipeline

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import (
    ObjectMetadata,
    ObjectPosition,
    RoomInfo,
    RoomObject,
    RoomPayload,
    Wall,
)

logger = logging.getLogger("decroom-backend.phone_first")

# Known coordinate system tag the phone sends. Reject anything else.
EXPECTED_COORDINATE_SYSTEM = "arkit_world_right_handed_y_up"

# Wall-disagreement thresholds (logged, not user-facing).
WALL_YAW_DISAGREEMENT_DEG = 15.0
WALL_POS_DISAGREEMENT_M = 0.5


# ── Error type ───────────────────────────────────────────────────────────────

class PhoneFirstSchemaError(ValueError):
    """Raised when the phone payload is malformed. Caller returns HTTP 400."""


# ── Parsed payload container ─────────────────────────────────────────────────

@dataclass
class PhoneFirstPayload:
    room_geometry: Dict[str, Any]
    point_cloud_ply: bytes
    keyframes_manifest: Dict[str, Any]
    keyframe_jpegs: List[bytes]
    scan_log: Dict[str, Any]


def parse_payload(
    room_geometry: bytes,
    point_cloud_ply: bytes,
    keyframes_manifest: bytes,
    keyframe_jpegs: List[bytes],
    scan_log: bytes,
) -> PhoneFirstPayload:
    """Parse multipart parts. Raises PhoneFirstSchemaError on any malformity."""
    try:
        geom = json.loads(room_geometry.decode("utf-8"))
    except Exception as e:
        raise PhoneFirstSchemaError(f"room_geometry.json invalid: {e}")

    # Required fields
    for field_name in ("walls", "floor_height", "ceiling_height",
                        "manhattan_yaw", "coordinate_system"):
        if field_name not in geom:
            raise PhoneFirstSchemaError(f"room_geometry.json missing {field_name}")

    if geom["coordinate_system"] != EXPECTED_COORDINATE_SYSTEM:
        raise PhoneFirstSchemaError(
            f"unsupported coordinate_system={geom['coordinate_system']!r}"
            f" (expected {EXPECTED_COORDINATE_SYSTEM!r})"
        )

    if len(point_cloud_ply) < 20 or not point_cloud_ply.startswith(b"ply"):
        raise PhoneFirstSchemaError("point_cloud.ply missing or malformed header")

    try:
        manifest = json.loads(keyframes_manifest.decode("utf-8"))
    except Exception as e:
        raise PhoneFirstSchemaError(f"keyframes.json invalid: {e}")
    if "frames" not in manifest:
        raise PhoneFirstSchemaError("keyframes.json missing 'frames'")

    try:
        log = json.loads(scan_log.decode("utf-8")) if scan_log else {}
    except Exception as e:
        raise PhoneFirstSchemaError(f"scan_log.json invalid: {e}")

    return PhoneFirstPayload(
        room_geometry=geom,
        point_cloud_ply=point_cloud_ply,
        keyframes_manifest=manifest,
        keyframe_jpegs=keyframe_jpegs,
        scan_log=log,
    )


# ── PLY loader ───────────────────────────────────────────────────────────────

def load_ply_points(ply_bytes: bytes) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Parse the phone's binary-little-endian PLY into (Nx3 float32 positions,
    Nx3 uint8 colors). Raises PhoneFirstSchemaError if malformed.
    """
    if np is None:
        raise PhoneFirstSchemaError("numpy not available on backend")

    header_end = ply_bytes.find(b"end_header\n")
    if header_end < 0:
        raise PhoneFirstSchemaError("PLY missing end_header")
    header = ply_bytes[:header_end].decode("ascii", errors="replace")

    # Find vertex count.
    count = None
    for line in header.splitlines():
        if line.startswith("element vertex"):
            parts = line.split()
            if len(parts) >= 3:
                count = int(parts[2])
            break
    if count is None:
        raise PhoneFirstSchemaError("PLY missing element vertex count")

    body = ply_bytes[header_end + len(b"end_header\n"):]
    # 12 bytes xyz (3 × float32) + 3 bytes rgb = 15 bytes per vertex
    expected = count * 15
    if len(body) < expected:
        raise PhoneFirstSchemaError(
            f"PLY body too short: {len(body)} < {expected}"
        )

    positions = np.empty((count, 3), dtype=np.float32)
    colors = np.empty((count, 3), dtype=np.uint8)
    for i in range(count):
        off = i * 15
        positions[i] = struct.unpack_from("<3f", body, off)
        colors[i] = struct.unpack_from("<3B", body, off + 12)

    return positions, colors


# ── Wall disagreement diagnostic ─────────────────────────────────────────────

def check_wall_agreement(
    phone_walls: List[Dict[str, Any]],
    spatiallm_walls: List[Any],
) -> None:
    """Log disagreements (>15° yaw or >0.5m position). Does not mutate results."""
    if not phone_walls or not spatiallm_walls:
        return

    def yaw_of_wall(w: Dict[str, Any]) -> float:
        s, e = w["start"], w["end"]
        return math.atan2(e["x"] - s["x"], e["z"] - s["z"])

    def center_of_wall(w: Dict[str, Any]) -> Tuple[float, float]:
        s, e = w["start"], w["end"]
        return ((s["x"] + e["x"]) / 2, (s["z"] + e["z"]) / 2)

    # Simple: for each phone wall, find nearest SpatialLM wall, log delta.
    for pw in phone_walls:
        pyaw = yaw_of_wall(pw)
        pcx, pcz = center_of_wall(pw)
        # (SpatialLM walls have various shapes depending on parser — skip heavy
        # matching. Log a warning if we see a large overall rotation delta.)
        best_delta_deg = None
        for sw in spatiallm_walls:
            try:
                sstart = getattr(sw, "start", None) or sw.get("start")
                send = getattr(sw, "end", None) or sw.get("end")
                if sstart is None or send is None:
                    continue
                sx0, sz0 = sstart[0], sstart[1]
                sx1, sz1 = send[0], send[1]
                syaw = math.atan2(sx1 - sx0, sz1 - sz0)
                delta_deg = math.degrees(abs(syaw - pyaw))
                if best_delta_deg is None or delta_deg < best_delta_deg:
                    best_delta_deg = delta_deg
            except Exception:
                continue
        if best_delta_deg is not None and best_delta_deg > WALL_YAW_DISAGREEMENT_DEG:
            logger.warning(
                "Phone/SpatialLM wall disagreement %.1f° > %.1f° at (%.2f,%.2f) — keeping phone wall",
                best_delta_deg, WALL_YAW_DISAGREEMENT_DEG, pcx, pcz,
            )


# ── Room envelope from phone walls ───────────────────────────────────────────

def room_envelope_from_walls(walls: List[Dict[str, Any]], floor_h: float, ceil_h: float) -> RoomInfo:
    """Derive width/length/height from phone walls' XZ bounding box."""
    xs: List[float] = []
    zs: List[float] = []
    for w in walls:
        xs.extend([w["start"]["x"], w["end"]["x"]])
        zs.extend([w["start"]["z"], w["end"]["z"]])
    if not xs:
        return RoomInfo(id=str(uuid.uuid4())[:8], width_m=4.0, length_m=4.0, height_m=2.5)
    width = max(xs) - min(xs) if xs else 4.0
    length = max(zs) - min(zs) if zs else 4.0
    height = (ceil_h - floor_h) if ceil_h > floor_h else 2.5

    walls_out = [
        Wall(
            id=f"w{i}",
            start=[w["start"]["x"], w["start"]["z"]],
            end=[w["end"]["x"], w["end"]["z"]],
        )
        for i, w in enumerate(walls)
    ]

    return RoomInfo(
        id=str(uuid.uuid4())[:8],
        width_m=max(width, 1.0),
        length_m=max(length, 1.0),
        height_m=max(height, 2.0),
        walls=walls_out,
    )


# ── SpatialLM object mapping ─────────────────────────────────────────────────

def map_spatiallm_objects_to_room_objects(
    spatiallm_objects: List[Any],
) -> List[RoomObject]:
    """Map SpatialLM bboxes through SPATIALLM_TO_PRD into RoomObject list."""
    out: List[RoomObject] = []
    for i, obj in enumerate(spatiallm_objects):
        cls = getattr(obj, "class_name", None) or (obj.get("class_name") if isinstance(obj, dict) else None)
        if not cls:
            continue
        prd_type = SPATIALLM_TO_PRD.get(cls)
        if not prd_type:
            logger.debug("SpatialLM class %r not mapped — skipping", cls)
            continue

        center = getattr(obj, "center", None) or (obj.get("center") if isinstance(obj, dict) else None)
        if center is None:
            continue
        size = getattr(obj, "size", None) or (obj.get("size") if isinstance(obj, dict) else None) or [1.0, 1.0, 1.0]
        rotation_y = getattr(obj, "rotation_y", None) or (obj.get("rotation_y") if isinstance(obj, dict) else 0.0) or 0.0

        out.append(RoomObject(
            id=f"o{i}",
            type=prd_type,
            asset_id=f"kenney_{prd_type}",
            position=ObjectPosition(
                x=float(center[0]), y=float(center[1]), z=float(center[2])
            ),
            rotation_y=float(rotation_y),
            scale=1.0,
            confidence=float(getattr(obj, "confidence", 0.5)
                              if not isinstance(obj, dict)
                              else obj.get("confidence", 0.5)),
            metadata=ObjectMetadata(
                width=float(size[0]),
                length=float(size[2]) if len(size) > 2 else float(size[1]),
                height=float(size[1]) if len(size) > 1 else 1.0,
                inferred=True,
                support_views=None,
            ),
        ))
    return out


# ── Main entry point (called from main.py route) ─────────────────────────────

def run_phone_first_pipeline(
    payload: PhoneFirstPayload,
    cfg: PipelineConfig,
) -> Dict:
    """Entry point. Returns a dict matching RoomPayload.to_dict()."""
    t0 = time.time()
    logger.info("Phone-first pipeline start (scan_id=%s)",
                 payload.scan_log.get("scan_id", "?"))

    # Load PLY (validates shape).
    positions, _colors = load_ply_points(payload.point_cloud_ply)
    logger.info("PLY loaded: %d points", positions.shape[0])

    # Run SpatialLM on the phone PLY.
    # spatiallm_layout.run_spatiallm_pipeline expects a points ndarray + cfg.
    # It returns (walls, objects) based on the current signature.
    try:
        slm_walls, slm_objects = run_spatiallm_pipeline(
            points=positions, cfg=cfg
        )
    except Exception as e:
        logger.exception("SpatialLM inference failed")
        # Surface as 5xx so the phone shows .backendServerError.
        raise RuntimeError(f"spatiallm_failed: {e}") from e

    # Log disagreement between phone walls and SpatialLM walls (diagnostic only).
    check_wall_agreement(payload.room_geometry["walls"], slm_walls or [])

    # Build RoomPayload from PHONE walls (trusted) + SpatialLM objects.
    room_info = room_envelope_from_walls(
        payload.room_geometry["walls"],
        payload.room_geometry["floor_height"],
        payload.room_geometry["ceiling_height"],
    )
    objects = map_spatiallm_objects_to_room_objects(slm_objects or [])

    result = RoomPayload(room=room_info, objects=objects).to_dict()
    elapsed = time.time() - t0
    logger.info(
        "Phone-first pipeline done: %d walls, %d objects, %.2fs",
        len(room_info.walls), len(objects), elapsed,
    )
    return result
```

- [ ] **Step 2: Add the new route to main.py**

Open `decroom_no_lidar/decroom_no_lidar/backend/main.py`. After the existing `@app.post("/api/upload")` function (around line 507), add:

```python
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
```

- [ ] **Step 3: Create backend tests directory + package init**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 4: Create test_category_mapping.py**

Path: `decroom_no_lidar/decroom_no_lidar/backend/tests/test_category_mapping.py`

```python
"""
Every SpatialLM class the backend accepts must map to a known Decroom category.
If SpatialLM emits a new class, it must either be added to SPATIALLM_TO_PRD or
explicitly logged as "not mapped" (which the pipeline does at DEBUG).
"""
import pytest

from pipeline.spatiallm_layout import SPATIALLM_TO_PRD

# The 14 Decroom categories the iOS app knows how to render.
DECROOM_14 = {
    "bed", "sofa", "chair", "desk", "table", "nightstand",
    "dresser", "bookshelf", "monitor", "lamp_floor", "lamp_desk",
    "plant", "rug", "poster",
}


def test_all_mapped_categories_are_decroom_14():
    """Every target in SPATIALLM_TO_PRD must be a Decroom category."""
    for src, dst in SPATIALLM_TO_PRD.items():
        assert dst in DECROOM_14, f"{src!r} → {dst!r} is not in the Decroom 14"


def test_bed_variants_all_map_to_bed():
    for variant in ("bed", "single_bed", "double_bed", "bunk_bed"):
        assert SPATIALLM_TO_PRD.get(variant) == "bed"


def test_sofa_variants_all_map_to_sofa():
    for variant in ("sofa", "l_shaped_sofa"):
        assert SPATIALLM_TO_PRD.get(variant) == "sofa"


def test_dining_table_maps_to_desk():
    # NOTE: existing mapping choice; preserved from pre-3a backend.
    assert SPATIALLM_TO_PRD["dining_table"] == "desk"
```

- [ ] **Step 5: Create test_upload_schema.py**

Path: `decroom_no_lidar/decroom_no_lidar/backend/tests/test_upload_schema.py`

```python
"""
Tests that malformed phone-first uploads are rejected with
PhoneFirstSchemaError (→ HTTP 400 → phone ScanError.backendSchemaError).
"""
import json

import pytest

from pipeline.phone_first_pipeline import (
    EXPECTED_COORDINATE_SYSTEM,
    PhoneFirstSchemaError,
    parse_payload,
)


def _valid_geom_bytes() -> bytes:
    return json.dumps({
        "walls": [],
        "floor_height": 0.0,
        "ceiling_height": 2.4,
        "manhattan_yaw": 0.0,
        "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
    }).encode()


def _valid_ply_bytes() -> bytes:
    return (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 0\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"property uchar red\n"
        b"property uchar green\n"
        b"property uchar blue\n"
        b"end_header\n"
    )


def _valid_manifest_bytes() -> bytes:
    return json.dumps({"frames": []}).encode()


def test_parse_valid_payload_succeeds():
    p = parse_payload(
        _valid_geom_bytes(),
        _valid_ply_bytes(),
        _valid_manifest_bytes(),
        [],
        b"{}",
    )
    assert p.room_geometry["coordinate_system"] == EXPECTED_COORDINATE_SYSTEM


def test_rejects_invalid_json_room_geometry():
    with pytest.raises(PhoneFirstSchemaError, match="room_geometry"):
        parse_payload(
            b"not-json",
            _valid_ply_bytes(),
            _valid_manifest_bytes(),
            [],
            b"{}",
        )


def test_rejects_wrong_coordinate_system():
    geom = json.dumps({
        "walls": [],
        "floor_height": 0.0,
        "ceiling_height": 2.4,
        "manhattan_yaw": 0.0,
        "coordinate_system": "unity_left_handed",
    }).encode()
    with pytest.raises(PhoneFirstSchemaError, match="coordinate_system"):
        parse_payload(geom, _valid_ply_bytes(), _valid_manifest_bytes(), [], b"{}")


def test_rejects_missing_fields():
    geom = json.dumps({
        "walls": [],
        "floor_height": 0.0,
        # ceiling_height missing
        "manhattan_yaw": 0.0,
        "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
    }).encode()
    with pytest.raises(PhoneFirstSchemaError, match="ceiling_height"):
        parse_payload(geom, _valid_ply_bytes(), _valid_manifest_bytes(), [], b"{}")


def test_rejects_malformed_ply():
    with pytest.raises(PhoneFirstSchemaError, match="point_cloud.ply"):
        parse_payload(
            _valid_geom_bytes(),
            b"not-a-ply",
            _valid_manifest_bytes(),
            [],
            b"{}",
        )


def test_rejects_missing_frames_in_manifest():
    manifest = json.dumps({"not_frames": []}).encode()
    with pytest.raises(PhoneFirstSchemaError, match="frames"):
        parse_payload(
            _valid_geom_bytes(), _valid_ply_bytes(), manifest, [], b"{}"
        )
```

- [ ] **Step 6: Create test_phone_first_pipeline.py**

Path: `decroom_no_lidar/decroom_no_lidar/backend/tests/test_phone_first_pipeline.py`

```python
"""
Unit tests for phone_first_pipeline. SpatialLM is mocked — these tests
verify our parsing, category mapping, and envelope derivation. A separate
integration test (manual, not here) runs real SpatialLM.
"""
import json
import struct
from unittest.mock import patch

import pytest

from pipeline.phone_first_pipeline import (
    EXPECTED_COORDINATE_SYSTEM,
    PhoneFirstPayload,
    load_ply_points,
    map_spatiallm_objects_to_room_objects,
    room_envelope_from_walls,
    run_phone_first_pipeline,
)


def _ply_with_n_points(n: int) -> bytes:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode()
    body = b""
    for i in range(n):
        body += struct.pack("<3f", float(i), 0.0, 0.0) + bytes([128, 128, 128])
    return header + body


def test_load_ply_points_parses_position_and_color():
    data = _ply_with_n_points(3)
    pos, col = load_ply_points(data)
    assert pos.shape == (3, 3)
    assert col.shape == (3, 3)
    assert float(pos[0][0]) == 0.0
    assert float(pos[2][0]) == 2.0
    assert int(col[0][0]) == 128


def test_room_envelope_from_walls_produces_nonempty_info():
    walls = [
        {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 3.0, "z": 0.0}},
        {"start": {"x": 3.0, "z": 0.0}, "end": {"x": 3.0, "z": 2.5}},
    ]
    info = room_envelope_from_walls(walls, floor_h=0.0, ceil_h=2.4)
    assert info.width_m >= 3.0
    assert info.length_m >= 2.5
    assert info.height_m >= 2.0
    assert len(info.walls) == 2


def test_map_spatiallm_objects_maps_known_classes():
    objs = [
        {"class_name": "bed", "center": [1, 0, 2], "size": [2, 1, 1.5],
         "rotation_y": 0.0, "confidence": 0.9},
        {"class_name": "unknown_class", "center": [0, 0, 0]},
    ]
    rooms = map_spatiallm_objects_to_room_objects(objs)
    assert len(rooms) == 1
    assert rooms[0].type == "bed"
    assert rooms[0].position.x == 1.0


def test_run_pipeline_returns_roompayload_dict():
    """End-to-end dry run with a mocked SpatialLM. Validates result shape."""
    payload = PhoneFirstPayload(
        room_geometry={
            "walls": [
                {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 3.0, "z": 0.0}},
            ],
            "floor_height": 0.0,
            "ceiling_height": 2.4,
            "manhattan_yaw": 0.0,
            "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        },
        point_cloud_ply=_ply_with_n_points(10),
        keyframes_manifest={"frames": []},
        keyframe_jpegs=[],
        scan_log={"scan_id": "testscan"},
    )

    class MockCfg:
        pass

    def fake_spatiallm(points, cfg):
        return [], [{"class_name": "bed", "center": [1, 0, 2],
                     "size": [2, 1, 1.5], "rotation_y": 0.0, "confidence": 0.9}]

    with patch("pipeline.phone_first_pipeline.run_spatiallm_pipeline",
                side_effect=fake_spatiallm):
        result = run_phone_first_pipeline(payload, MockCfg())

    assert "room" in result
    assert "objects" in result
    assert len(result["objects"]) == 1
    assert result["objects"][0]["type"] == "bed"


def test_zero_objects_returns_empty_list():
    payload = PhoneFirstPayload(
        room_geometry={
            "walls": [
                {"start": {"x": 0.0, "z": 0.0}, "end": {"x": 3.0, "z": 0.0}},
            ],
            "floor_height": 0.0, "ceiling_height": 2.4, "manhattan_yaw": 0.0,
            "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        },
        point_cloud_ply=_ply_with_n_points(10),
        keyframes_manifest={"frames": []},
        keyframe_jpegs=[],
        scan_log={"scan_id": "testscan"},
    )

    with patch("pipeline.phone_first_pipeline.run_spatiallm_pipeline",
                return_value=([], [])):
        result = run_phone_first_pipeline(payload, object())

    assert result["objects"] == []
```

- [ ] **Step 7: Run backend tests**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests pass. If `from pipeline.spatiallm_layout import SPATIALLM_TO_PRD` fails with ModuleNotFoundError, ensure you're running from the `backend/` directory (not from inside `tests/`) so Python finds the `pipeline/` package.

- [ ] **Step 8: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/backend/pipeline/phone_first_pipeline.py \
        decroom_no_lidar/decroom_no_lidar/backend/main.py \
        decroom_no_lidar/decroom_no_lidar/backend/tests/
git commit -m "feat: add phone_first_pipeline + /api/upload/phone-first route

Server-side handler for phone-first uploads. Parses multipart
payload (room_geometry.json + point_cloud.ply + keyframes.json
+ keyframes + scan_log.json), validates schema, runs SpatialLM
on the phone-provided PLY, trusts phone walls on disagreement,
maps SpatialLM 59 categories → Decroom 14 via the existing
SPATIALLM_TO_PRD dict, and returns a RoomPayload.

New tests:
- test_category_mapping.py  — SPATIALLM_TO_PRD → 14 cats
- test_upload_schema.py     — malformed payloads → HTTP 400
- test_phone_first_pipeline.py — end-to-end dry run with
                                  mocked SpatialLM

Old /api/upload route still live; deleted in Phase H.

Part of Plan 3a Phase D."
```

- [ ] **Step 9: Deploy backend to RunPod**

This is an operational step, not a code change. Per the design spec's deployment timing note, the RunPod GPU server must be running the new pipeline before Task 7 ships. Follow `memory/plan_status.md` SpatialLM server bring-up:

```bash
# From RunPod Jupyter terminal, or over SSH tunnel:
# 1. Pull latest branch
# 2. Restart the server: kill existing, source /opt/spatiallm-env/bin/activate,
#    python3 /root/server.py
# 3. Confirm /api/upload/phone-first responds to a probe (curl with --data-binary)
```

Verify: `curl -s http://<runpod>:8000/api/upload/phone-first -X POST` returns 422 or 400 (missing files), not 404. If it returns 404, the new route isn't loaded — restart the server.

---

## Task 7: Cutover — VideoRecorderView uses phone-first upload (Phase E)

**Purpose:** The one scary commit. After this merges, the app uses only the phone-first path. Requires the backend from Task 6 to be running on RunPod first.

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/RecordingProcessor.swift`
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/VideoRecorderView.swift`
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ContentView.swift` (GuidedARFrameRecorder assembles payload)

- [ ] **Step 1: Add processPhoneFirstPayload to RecordingProcessor**

Open `RecordingProcessor.swift`. Add a new method before `reset()` (around line 85):

```swift
    /// Upload a phone-first payload. Publishes `.uploading` then either
    /// `.finished(RoomPayload)` or `.failed(String)` with the specific
    /// ScanError message (preserves "no fallback, errors explain what
    /// happened").
    func processPhoneFirstPayload(_ payload: ScanUploadPayload) {
        state = .uploading
        Task {
            do {
                let roomPayload = try await UploadService.uploadPhoneFirst(payload)
                self.state = .finished(roomPayload)
            } catch let e as ScanError {
                self.state = .failed(e.message)
            } catch {
                self.state = .failed("Unexpected upload error: \(error.localizedDescription)")
            }
        }
    }
```

- [ ] **Step 2: Add keyframe collection hook in GuidedARFrameRecorder**

Open `ContentView.swift`, in `GuidedARFrameRecorder`. Find `stopVideoMode()` (around line 723) and replace its body with:

```swift
    /// Returns disk-based keyframe JPEGs + metadata. Legacy method — kept so
    /// Plan 3a's cutover can coexist with the old UploadService during Phase E.
    /// Phase G deletes the callers of this method.
    func stopVideoMode() -> (jpegPaths: [URL], metadataJSON: String?) {
        isRecording = false
        unlockExposureAndFocus()
        let ordered = shots.sorted { $0.timestamp < $1.timestamp }
        let metadataJSON = buildMetadataJSON(shots: ordered)
        clearScanFeedback()
        return (jpegPaths: keyframeJPEGPaths, metadataJSON: metadataJSON)
    }

    /// Build and return the phone-first upload payload. Must be called after
    /// stopVideoMode() (so keyframeJPEGPaths are finalized). On any phone-side
    /// validation failure, throws a ScanError.
    func finalizePhoneFirstPayload() async throws -> ScanUploadPayload {
        guard let session = scanSession else {
            throw ScanError.payloadBuildFailed(detail: "ScanSession missing — start() not called?")
        }

        // Collect keyframe JPEG data from disk + build metadata.
        var jpegs: [Data] = []
        var metas: [ScanUploadPayload.KeyframeMetadata] = []
        for (i, url) in keyframeJPEGPaths.enumerated() {
            let data: Data
            do {
                data = try Data(contentsOf: url)
            } catch {
                throw ScanError.payloadBuildFailed(
                    detail: "read keyframe \(url.lastPathComponent): \(error.localizedDescription)"
                )
            }
            // Pose + intrinsics come from the `shots` array built in handle(frame:).
            let shot = i < shots.count ? shots[i] : nil
            let pose = shot?.cameraTransform ?? matrix_identity_float4x4
            let intr = shot.map { shotIntrinsics(from: $0) }
                ?? Intrinsics(fx: 1500, fy: 1500, cx: 720, cy: 960,
                               width: 1440, height: 1920)
            let ts = shot?.timestamp ?? 0

            jpegs.append(data)
            metas.append(ScanUploadPayload.KeyframeMetadata(
                filename: "frame_\(String(format: "%02d", i)).jpg",
                timestamp: ts,
                poseWorldFromCamera: pose,
                intrinsics: intr
            ))
        }

        return try session.finalize(keyframes: metas, keyframeJPEGs: jpegs)
    }

    /// Extract portrait-aware intrinsics from a GuidedShotPayload if it has them,
    /// otherwise return a reasonable iPhone default. The existing GuidedShotPayload
    /// already stores intrinsics in `shot.intrinsics` (fx/fy/cx/cy) — mirror them
    /// here for the new payload.
    private func shotIntrinsics(from shot: GuidedShotPayload) -> Intrinsics {
        // If shot has explicit intrinsics, use them; else fall back to iPhone 15 Pro
        // portrait defaults.
        return Intrinsics(
            fx: Float(shot.intrinsics?.fx ?? 1500),
            fy: Float(shot.intrinsics?.fy ?? 1500),
            cx: Float(shot.intrinsics?.cx ?? 720),
            cy: Float(shot.intrinsics?.cy ?? 960),
            width: shot.intrinsics?.width ?? 1440,
            height: shot.intrinsics?.height ?? 1920
        )
    }
```

Note: `GuidedShotPayload` may not currently have an `intrinsics` field. If not, access it from the ARFrame's camera at capture time by adding `let intrinsics` to `GuidedShotPayload` (near `latestFrame` construction in `handle(frame:)`). Do NOT plumb it via a new argument — mutate `GuidedShotPayload` in the same commit.

- [ ] **Step 3: Switch VideoRecorderView.upload() to the new path**

Open `VideoRecorderView.swift`. Find `private func upload()` (around line 414) and replace its body:

```swift
    private func upload() {
        guard recorder.capturedCount > 0 else { return }
        // Preserve disk-based keyframe paths (used inside finalizePhoneFirstPayload).
        _ = recorder.stopVideoMode()
        Task {
            do {
                let payload = try await recorder.finalizePhoneFirstPayload()
                print("[VideoRecorderView] Uploading phone-first payload scan=\(payload.scanID) keyframes=\(payload.keyframes.count)")
                processor.processPhoneFirstPayload(payload)
            } catch let e as ScanError {
                processor.state = .failed(e.message)
            } catch {
                processor.state = .failed("Scan finalization failed: \(error.localizedDescription)")
            }
        }
    }
```

- [ ] **Step 4: Build and run tests**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -10
```

Expected: both succeed with all tests passing.

- [ ] **Step 5: Physical-device smoke test (manual, before commit)**

This is the smallest physical-device check possible. Do NOT skip.

1. Ensure the RunPod backend is running the Task 6 code (`curl backend:8000/api/upload/phone-first -X POST` → 422, not 404).
2. Build and run on your iPhone in Xcode.
3. Start a scan. Rotate ~360°. Tap Done. Tap Upload.
4. Expected: upload completes within 60s, 3D scene renders.
5. If upload fails with a specific ScanError message, fix before committing.

- [ ] **Step 6: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/RecordingProcessor.swift \
        decroom_no_lidar/decroom_no_lidar/VideoRecorderView.swift \
        decroom_no_lidar/decroom_no_lidar/ContentView.swift
git commit -m "feat: switch VideoRecorderView to phone-first upload

VideoRecorderView now calls recorder.finalizePhoneFirstPayload()
which builds a ScanUploadPayload from the live ScanSession, then
uploads via processor.processPhoneFirstPayload() → UploadService
.uploadPhoneFirst(). On phone-side validation failure (noWalls,
insufficientCoverage) or network/backend error, a named ScanError
message is shown via processor.state = .failed(e.message).

Old processCapturedFrames / processKeyframesFromDisk remain live
but unused — deleted in Phase G.

REQUIRES: RunPod backend running Task 6 code (Phase D) first.

Part of Plan 3a Phase E — the cutover."
```

---

## Task 8: Device Tier Gate at App Launch (Phase F)

**Purpose:** Block scanning on `.fallback` tier devices (iPhone X and older) with the specific unsupported-device message. No silent fallback.

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/CaptureFlowView.swift`

- [ ] **Step 1: Add tier gate to CaptureFlowView**

Open `CaptureFlowView.swift`. Replace the `body` (around line 32) — add a guard at the top of the `NavigationStack` body:

```swift
    var body: some View {
        NavigationStack {
            if DeviceCapabilities.currentTier == .fallback {
                unsupportedDeviceView
            } else {
                mainCaptureBody
            }
        }
    }

    private var unsupportedDeviceView: some View {
        VStack(spacing: 24) {
            Image(systemName: "iphone.slash")
                .font(.system(size: 72))
                .foregroundStyle(.secondary)
            Text(ScanError.unsupportedDevice.message)
                .font(.title3)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            if let goHome = goHome {
                Button("Back to Home") {
                    goHome()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var mainCaptureBody: some View {
        // … the existing body content (VStack, switch on scanMode, etc.) goes here.
    }
```

Then move the entire existing body (everything currently inside the `NavigationStack { VStack { … } }`) into the `mainCaptureBody` computed property. Preserve all toolbar, sheet, and ZStack code unchanged inside `mainCaptureBody`.

- [ ] **Step 2: Build**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 3: Run tests (no regressions)**

```bash
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/CaptureFlowView.swift
git commit -m "feat: device tier gate at app launch

.fallback tier devices (iPhone X and older) now see
an 'unsupported device' screen instead of the capture
UI. Uses ScanError.unsupportedDevice.message as the
body text, preserving the single-source-of-truth for
error copy.

Part of Plan 3a Phase F."
```

---

## Task 9: Wire ScanError Alerts Throughout Scan Flow (Phase F)

**Purpose:** Surface every relevant `ScanError` as a SwiftUI alert with the correct Retry / Resume behavior. Ensures the "no fallback, errors explain what happened" promise holds at the UX level.

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/VideoRecorderView.swift`

- [ ] **Step 1: Add alert state + presentation**

Open `VideoRecorderView.swift`. Add a state property near the top of the struct body:

```swift
    @State private var activeError: ScanError? = nil
```

At the bottom of the outermost `ZStack` in `body` (just before the existing `.onAppear`), add the alert modifier:

```swift
        .alert(
            isPresented: Binding(
                get: { activeError != nil },
                set: { if !$0 { activeError = nil } }
            ),
            content: {
                guard let err = activeError else {
                    return Alert(title: Text("Error"))
                }
                if err.isRetryable {
                    let retryLabel: String
                    switch err {
                    case .noWallsDetected, .insufficientCoverage:
                        retryLabel = "Resume Scan"
                    default:
                        retryLabel = "Retry"
                    }
                    return Alert(
                        title: Text("Scan Error"),
                        message: Text(err.message),
                        primaryButton: .default(Text(retryLabel)) {
                            activeError = nil
                            handleRetry(for: err)
                        },
                        secondaryButton: .cancel()
                    )
                } else {
                    return Alert(
                        title: Text("Scan Error"),
                        message: Text(err.message),
                        dismissButton: .default(Text("OK")) {
                            activeError = nil
                        }
                    )
                }
            }
        )
```

- [ ] **Step 2: Add retry handler**

Still in `VideoRecorderView.swift`, add the retry helper below the existing `private func upload()`:

```swift
    private func handleRetry(for error: ScanError) {
        switch error {
        case .noWallsDetected, .insufficientCoverage:
            // Resume scan — keep recorder state, user keeps rotating.
            isScanning = true
            recorder.start()
        case .uploadNoServer, .uploadTimeout, .payloadBuildFailed, .backendServerError:
            upload()
        default:
            break  // non-retryable cases dismiss with OK
        }
    }
```

- [ ] **Step 3: Publish ScanError to the alert state**

Replace the `.failed` handling in `VideoRecorderView.upload()` (inside the `catch let e as ScanError` block) so the error shows as an alert:

```swift
            } catch let e as ScanError {
                activeError = e
                processor.state = .failed(e.message)
            } catch {
                processor.state = .failed("Scan finalization failed: \(error.localizedDescription)")
            }
```

Also add an observer on `processor.state` so backend errors surfaced via `processPhoneFirstPayload` show alerts too. Inside the same `.alert(...)` chain, before the modifier, add:

```swift
        .onChange(of: processor.state) { _, newState in
            if case .failed(let msg) = newState {
                // Best-effort: if the message matches a known ScanError.message,
                // find the case. Otherwise, show a generic alert.
                activeError = ScanError.allCases.first(where: { $0.message == msg })
                    ?? .backendServerError(status: 0, detail: msg)
            }
        }
```

Note: `ScanError.allCases` requires `CaseIterable` conformance for non-associated-value cases only, which won't compile for an enum with associated values. Instead, build a manual lookup. Replace the `.onChange` block with:

```swift
        .onChange(of: processor.state) { _, newState in
            if case .failed(let msg) = newState {
                activeError = Self.errorForMessage(msg)
            }
        }
```

And add a static helper to `VideoRecorderView`:

```swift
    /// Reverse-lookup a ScanError from its user-facing message.
    /// Used when processor.state = .failed(msg) needs an alert that matches
    /// the retryable-vs-not treatment. Fallback: backendServerError.
    static func errorForMessage(_ msg: String) -> ScanError {
        let table: [ScanError] = [
            .unsupportedDevice,
            .depthModelLoadFailed(detail: ""),
            .arKitTrackingLimited,
            .arKitSessionFailed(detail: ""),
            .noWallsDetected,
            .insufficientCoverage(binsFilled: 0, binsRequired: 12),
            .payloadBuildFailed(detail: ""),
            .uploadNoServer,
            .uploadTimeout,
            .backendServerError(status: 0, detail: ""),
            .noFurnitureDetected,
        ]
        return table.first(where: { $0.message == msg })
            ?? .backendServerError(status: 0, detail: msg)
    }
```

- [ ] **Step 4: Build and run tests**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -10
```

Expected: build succeeds; all tests pass.

- [ ] **Step 5: Manual QA — each error path (physical device, with dev build toggles)**

This is where we prove the "no fallback, errors explain what happened" claim. Walk the error table:

| Case | Trigger |
|---|---|
| `.noWallsDetected` | Scan in a pitch-black or heavily-occluded room for 20s then tap Done |
| `.insufficientCoverage` | Tap Done after 3s of scanning |
| `.uploadNoServer` | Turn off backend, tap Upload |
| `.uploadTimeout` | Backend with a 90s artificial sleep inserted, tap Upload |
| `.backendSchemaError` | Dev build mangles `room_geometry.json` before upload |
| `.backendServerError` | Backend route raises a deliberate 500 |
| `.depthModelLoadFailed` | Dev build temporarily renames `DepthModel.mlpackage` in the bundle |

For each: confirm the exact message appears in the alert, Retry/Resume Scan/OK button behaves correctly.

- [ ] **Step 6: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/VideoRecorderView.swift
git commit -m "feat: wire ScanError alerts throughout scan flow

Every ScanError case surfaces as a SwiftUI alert:
- Retryable (upload failures) → 'Retry' button re-calls upload()
- Resume-able (noWalls/insufficientCoverage) → 'Resume Scan'
  button re-starts the recorder
- Non-retryable → 'OK' dismiss

Reverse-lookup from processor.state.failed(msg) → ScanError
via Self.errorForMessage(msg), so messages set by
RecordingProcessor (backend errors) get the same treatment as
errors thrown inline from VideoRecorderView.upload().

Completes the 'no fallback, errors explain what happened'
contract at the UX level.

Part of Plan 3a Phase F."
```

---

## Task 10: Delete Old iOS Upload Methods (Phase G)

**Purpose:** Remove the legacy `upload(images:)` and `uploadFromDisk(jpegPaths:)` now that no code calls them. Compiler verifies.

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/UploadService.swift`
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/RecordingProcessor.swift`

- [ ] **Step 1: Confirm no callers of the old methods**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
grep -rn "UploadService.upload(images" decroom_no_lidar/decroom_no_lidar/decroom_no_lidar
grep -rn "UploadService.uploadFromDisk" decroom_no_lidar/decroom_no_lidar/decroom_no_lidar
grep -rn "processCapturedFrames\|processKeyframesFromDisk" decroom_no_lidar/decroom_no_lidar/decroom_no_lidar
```

Expected: no matches. If any remain, the cutover in Phase E missed a callsite — fix that before deleting.

- [ ] **Step 2: Delete from UploadService.swift**

Open `UploadService.swift`. Delete:
- The `upload(images:metadataJSON:)` method (around line 200-290)
- The `uploadFromDisk(jpegPaths:metadataJSON:)` method (around line 112-197)

Keep: `uploadPhoneFirst`, `_buildPhoneFirstMultipartBody`, `resolveUploadURL`, `resolvePhoneFirstUploadURL`, `isServerReachable`, `healthURL`, `fallbackURL`, `candidateUploadURLs`, `resolvedURL`, and the `Data.append(String)` extension at the bottom.

Also delete the now-unused `fallbackURL` if the only caller was the old methods. **Keep it** if `resolvePhoneFirstUploadURL` still uses the same discovery chain (it does, via `resolveUploadURL`).

- [ ] **Step 3: Delete from RecordingProcessor.swift**

Open `RecordingProcessor.swift`. Delete:
- `func processKeyframesFromDisk(_:metadataJSON:)` (around line 42-60)
- `func processCapturedFrames(_:metadataJSON:)` (around line 62-83)

Keep: `ProcessingState`, `@Published state`, `processPhoneFirstPayload`, `reset`.

- [ ] **Step 4: Build**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -10
```

Expected: `** BUILD SUCCEEDED **`. If it fails with "method X not found" errors, a caller of the deleted methods was missed — go back to Step 1.

- [ ] **Step 5: Run tests**

```bash
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/UploadService.swift \
        decroom_no_lidar/decroom_no_lidar/RecordingProcessor.swift
git commit -m "refactor: delete legacy UploadService.upload + uploadFromDisk

With Phase E's cutover live, nothing calls the JPEG-only
upload paths anymore. Deletes:
- UploadService.upload(images:metadataJSON:)
- UploadService.uploadFromDisk(jpegPaths:metadataJSON:)
- RecordingProcessor.processCapturedFrames
- RecordingProcessor.processKeyframesFromDisk

Compilation verifies no remaining callers. -~250 lines.

Part of Plan 3a Phase G."
```

---

## Task 11: Delete Plan 2 Dead Code (Phase G)

**Purpose:** Remove the six visual-UI files and two test files that were built in Plan 2 but never shipped. Explicit in the design spec as "dead code inventory."

**Files to delete:**
- `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ARRoomPreviewView.swift`
- `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ARRoomPreviewContainer.swift`
- `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ARRoomPreviewViewModel.swift`
- `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/GridShellEntity.swift`
- `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/CaptureReticle.swift`
- `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/PanoramaBuffer.swift`
- `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/ARRoomPreviewViewModelTests.swift`
- `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/CaptureReticleTests.swift`

Also deleting from `ScanSession.swift`: the `panoramaThumbnails` published property + `capturePanoramaFrame` method (only the old ARRoomPreviewView used them).

- [ ] **Step 1: Confirm no callers**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
grep -rn "ARRoomPreviewView\|ARRoomPreviewContainer\|GridShellEntity\|CaptureReticle\|PanoramaBuffer\|ARRoomPreviewViewModel" decroom_no_lidar/decroom_no_lidar/decroom_no_lidar 2>/dev/null | grep -v "\.swift:.*// "
```

Expected: matches only inside the files themselves (self-references). If any Swift file outside the list imports or instantiates these, investigate before deleting.

- [ ] **Step 2: Remove panoramaThumbnails from ScanSession**

Open `ScanSession.swift`. Remove:

```swift
    @Published var panoramaThumbnails: [(image: CGImage, yaw: Float)] = []
```

(around line 45) and the entire `capturePanoramaFrame` private method (around line 178). Also remove the `capturePanoramaFrame` call inside `update(frame:)` and the `lastPanoramaFrame`/`panoramaFrameInterval`/`panoContext` fields.

- [ ] **Step 3: Delete the files**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
rm decroom_no_lidar/decroom_no_lidar/ARRoomPreviewView.swift
rm decroom_no_lidar/decroom_no_lidar/ARRoomPreviewContainer.swift
rm decroom_no_lidar/decroom_no_lidar/ARRoomPreviewViewModel.swift
rm decroom_no_lidar/decroom_no_lidar/GridShellEntity.swift
rm decroom_no_lidar/decroom_no_lidar/CaptureReticle.swift
rm decroom_no_lidar/decroom_no_lidar/PanoramaBuffer.swift
rm decroom_no_lidar/decroom_no_lidarTests/ARRoomPreviewViewModelTests.swift
rm decroom_no_lidar/decroom_no_lidarTests/CaptureReticleTests.swift
```

- [ ] **Step 4: Build**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -10
```

Expected: `** BUILD SUCCEEDED **`. Any "file not found" errors indicate leftover references — use the Step 1 grep to find and fix.

- [ ] **Step 5: Run tests**

```bash
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -10
```

Expected: all remaining tests pass (8 tests fewer than before — `ARRoomPreviewViewModelTests` had 5, `CaptureReticleTests` had 3).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: delete Plan 2 dead code

Six visual-UI files built in Plan 2 but never reached the live
scan flow:
- ARRoomPreviewView / Container / ViewModel
- GridShellEntity / CaptureReticle
- PanoramaBuffer

Plus their tests (ARRoomPreviewViewModelTests, CaptureReticleTests)
and the panoramaThumbnails state they depended on in
ScanSession.

This is the ~2000-line cleanup the Plan 2 QA doc flagged. See
memory/visual_ui_failures.md — these approaches failed; no plan
to revive them.

Part of Plan 3a Phase G."
```

---

## Task 12: Delete Old Backend Pipeline Modules (Phase H)

**Purpose:** Remove the SfM/backend-depth/layout/lifting/tracking Python modules now that `phone_first_pipeline.py` is the only path.

**Files to delete:**
- `decroom_no_lidar/decroom_no_lidar/backend/pipeline/sfm.py`
- `decroom_no_lidar/decroom_no_lidar/backend/pipeline/depth.py`
- `decroom_no_lidar/decroom_no_lidar/backend/pipeline/layout.py`
- `decroom_no_lidar/decroom_no_lidar/backend/pipeline/lifting.py`
- `decroom_no_lidar/decroom_no_lidar/backend/pipeline/reconstruction.py`
- `decroom_no_lidar/decroom_no_lidar/backend/pipeline/tracking.py`
- `decroom_no_lidar/decroom_no_lidar/backend/pipeline/detection.py`
- `decroom_no_lidar/decroom_no_lidar/backend/pipeline/detection_open_vocab.py`
- `decroom_no_lidar/decroom_no_lidar/backend/pipeline/geometry.py`
- `decroom_no_lidar/decroom_no_lidar/backend/pipeline/pipeline.py`

**Do NOT delete:**
- `pipeline/config.py` (still used by phone_first_pipeline via cfg)
- `pipeline/metadata.py` (reused for scan log parsing)
- `pipeline/spatiallm_layout.py` (provides `SPATIALLM_TO_PRD` + `run_spatiallm_pipeline`)
- `pipeline/phone_first_pipeline.py` (new in Task 6)

- [ ] **Step 1: Confirm no imports of the old modules outside themselves**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
grep -rn "from .sfm\|from pipeline.sfm\|from .depth\|from .layout\|from .lifting\|from .reconstruction\|from .tracking\|from .detection\|from .detection_open_vocab\|from .geometry\|from .pipeline import run_pipeline" \
    --include="*.py" \
    | grep -v "^\./pipeline/\(sfm\|depth\|layout\|lifting\|reconstruction\|tracking\|detection\|detection_open_vocab\|geometry\|pipeline\)\.py:"
```

Expected: only `main.py` may still have `from pipeline.pipeline import run_pipeline` (inside the old `/api/upload` route). That's fine — the next task deletes that route and the import.

- [ ] **Step 2: Delete the files**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/pipeline
rm sfm.py depth.py layout.py lifting.py reconstruction.py \
   tracking.py detection.py detection_open_vocab.py geometry.py pipeline.py
```

- [ ] **Step 3: Also delete old config fields from config.py (optional polish)**

Open `pipeline/config.py` and delete fields only referenced by the now-deleted modules (e.g., `sfm_*`, `layout_*`, `metric_fp_scale_*`, tracklet/tracking fields). Leave fields used by `spatiallm_layout.py` and `phone_first_pipeline.py` alone. **If unsure, leave them** — dead config fields don't hurt anything and keep the diff focused.

- [ ] **Step 4: Delete backend test fixtures for old pipeline (if any)**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
ls tests/ 2>/dev/null
```

Expected: the three Plan 3a tests from Task 6 + `__init__.py`. If any older test modules exist (from pre-Plan-3 efforts) that reference deleted modules, delete them here too.

- [ ] **Step 5: Run backend tests**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/ -v 2>&1 | tail -15
```

Expected: the three Plan 3a tests pass.

- [ ] **Step 6: Verify main.py imports still resolve (the /api/upload route is temporarily broken — OK)**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -c "from main import app; print('app loaded')"
```

If this fails with `ImportError: cannot import name 'run_pipeline' from 'pipeline.pipeline'` — that's expected. `main.py` still has the old route trying to lazy-import `run_pipeline` inside its function body. We fix this in Task 13 (next). For Task 12's purposes, the server does not need to start — we just verified the deleted modules aren't imported eagerly.

- [ ] **Step 7: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add -A decroom_no_lidar/decroom_no_lidar/backend
git commit -m "refactor: remove old backend pipeline modules

Deletes:
- sfm.py (COLMAP wrapper)
- depth.py (backend Depth Anything V2)
- layout.py (PCA layout)
- lifting.py (2D→3D backprojection)
- reconstruction.py (ray triangulation, pose refinement)
- tracking.py (ByteTrack)
- detection.py, detection_open_vocab.py (backend YOLO)
- geometry.py (global feature cloud)
- pipeline.py (orchestrator)

main.py's old /api/upload route still imports run_pipeline
lazily — it's broken but not eagerly imported. Task 13
deletes the route itself.

~3000 lines deleted from pipeline/.

Part of Plan 3a Phase H."
```

---

## Task 13: Remove Old /api/upload Route (Phase H)

**Purpose:** Delete the legacy upload route from `main.py`. After this, `phone-first` is the only route.

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/backend/main.py`

- [ ] **Step 1: Delete the route**

Open `main.py`. Delete the entire `@app.post("/api/upload")` function (around lines 453-507, from the decorator through `return JSONResponse(content=result)`).

Also delete the import-related dead code at the top of the module: the large block of `os.getenv(...)` env-var parsing for SfM/tracking/depth config (lines 34-352). These configure features that no longer exist. Keep only:

- `DECROOM_ENABLE_SPATIALLM` block
- `DECROOM_SPATIALLM_MODE` block
- `DECROOM_SPATIALLM_API_URL` block
- any config that `phone_first_pipeline.py` uses

If unsure, leave the env blocks in — they just assign to `cfg` fields that no one reads. But the file is cleaner without them.

- [ ] **Step 2: Verify server starts cleanly**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -c "from main import app; print('ok', [r.path for r in app.routes if hasattr(r, 'path')])"
```

Expected: `ok` plus a route list that includes `/api/upload/phone-first` and `/health` but NOT `/api/upload`.

- [ ] **Step 3: Test locally if possible**

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
UV_PID=$!
sleep 2
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/api/upload
curl -s -X POST http://localhost:8000/api/upload/phone-first
kill $UV_PID
```

Expected:
- `/health` → `{"status":"ok"}`
- `/api/upload` → 404 (route deleted)
- `/api/upload/phone-first` → 422 (missing required file fields — expected)

- [ ] **Step 4: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/backend/main.py
git commit -m "refactor: remove legacy /api/upload route

Phone-first is the only route now. Also cleaned up the large
block of SfM/tracking/depth env-var parsing at the top of
main.py — those features no longer exist. Kept SpatialLM
env vars (enable/mode/api_url).

After this commit, only /api/upload/phone-first and /health
are live.

Part of Plan 3a Phase H."
```

- [ ] **Step 5: Redeploy backend to RunPod**

Same operational step as Task 6 Step 9. After this commit, the RunPod server must run the updated main.py, or it'll still respond on the old route (harmless, but confusing).

---

## Task 14: Manual QA + Document Results (Phase I)

**Purpose:** Execute the full manual QA checklist from the design spec on a physical device, document results. Pre-merge requirement.

**Files:**
- Create: `docs/superpowers/qa/YYYY-MM-DD-plan3a-manual-qa.md` (replace `YYYY-MM-DD` with the actual QA date)

- [ ] **Step 1: Create QA doc**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
TODAY=$(date +%Y-%m-%d)
cp docs/superpowers/qa/2026-04-19-plan2-manual-qa.md \
   docs/superpowers/qa/${TODAY}-plan3a-manual-qa.md
```

Edit the new file. Replace the title + context + table of checks. Minimum content:

```markdown
# Plan 3a Manual QA — YYYY-MM-DD

## Context

Validates the phone-first end-to-end scan pipeline delivered by Plan 3a.
Spec: `docs/superpowers/specs/2026-04-19-plan3a-phone-first-backend-integration-design.md`.

## Automated checks

- [ ] Swift test suite: `xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro'` — all pass
- [ ] Python test suite: `cd backend && python3 -m pytest tests/` — all pass
- [ ] Device build: `xcodebuild build -destination 'generic/platform=iOS'` — success

## Happy-path device checklist (iPhone XS or newer)

- [ ] App launches, scan button enabled
- [ ] Full scan of a real room → upload succeeds → 3D preview renders
- [ ] scan_log.json appears in ~/Documents/scan_logs/ after scan
- [ ] Thermal state during 60s scan stays .nominal or .fair
- [ ] No crashes in Xcode console

## Unsupported device checklist (iPhone X, A11, or older if available)

- [ ] App launches, shows "This iPhone doesn't support Decroom's scanning. Minimum: iPhone XS."
- [ ] Scan button disabled; no way to enter capture flow

## Error-path checklist — each deliberately triggered

| Case | How triggered | Message shown | Button behavior |
|---|---|---|---|
| .noWallsDetected | ... | ... | ... |
| .insufficientCoverage | ... | ... | ... |
| .uploadNoServer | ... | ... | ... |
| .uploadTimeout | ... | ... | ... |
| .backendSchemaError | ... | ... | ... |
| .backendServerError | ... | ... | ... |
| .depthModelLoadFailed | ... | ... | ... |
| .noFurnitureDetected | ... | ... | ... |

## Payload fixture

- [ ] Save a real successful scan's payload to `tests/fixtures/phone_first_payload/`
  for future contract tests. Files:
  - room_geometry.json
  - point_cloud.ply
  - keyframes.json
  - frame_00.jpg through frame_09.jpg
  - scan_log.json

## Observations / follow-ups

...
```

Fill in each checkbox + table row based on actual device testing.

- [ ] **Step 2: Capture a real scan's payload for the fixture directory**

During one of the happy-path scans on device, save the assembled `ScanUploadPayload` parts to disk before uploading. Add a temporary debug write in `VideoRecorderView.upload()` just before `processor.processPhoneFirstPayload(payload)`:

```swift
#if DEBUG
                let fixtureDir = FileManager.default.urls(
                    for: .documentDirectory, in: .userDomainMask
                ).first!.appendingPathComponent("last_payload_fixture")
                try? FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)
                try? payload.roomGeometryJSON.write(to: fixtureDir.appendingPathComponent("room_geometry.json"))
                try? payload.pointCloudPLY.write(to: fixtureDir.appendingPathComponent("point_cloud.ply"))
                try? payload.scanLogJSON.write(to: fixtureDir.appendingPathComponent("scan_log.json"))
                for (i, kf) in payload.keyframes.enumerated() {
                    try? kf.jpeg.write(to: fixtureDir.appendingPathComponent("frame_\(String(format: "%02d", i)).jpg"))
                }
                print("[DEBUG] Wrote payload fixture to \(fixtureDir.path)")
#endif
```

Copy the directory off the device (Xcode → Devices → select app → Download Container) and into `tests/fixtures/phone_first_payload/`. Also generate the `keyframes.json` from the payload's manifest (add a similar write above).

Once the fixture is captured, **remove the debug block** in a separate commit before merging.

- [ ] **Step 3: Commit QA results**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add docs/superpowers/qa/ tests/fixtures/phone_first_payload/
git commit -m "docs: Plan 3a manual QA results + payload fixture

Full error-path checklist walked on a physical iPhone. Each
ScanError case triggered deliberately and the exact user-facing
message verified.

tests/fixtures/phone_first_payload/ populated with a real
successful scan's output — enables Swift + Python contract
tests to detect schema drift going forward.

Part of Plan 3a Phase I."
```

- [ ] **Step 4: Remove any lingering debug code**

```bash
grep -rn "#if DEBUG" decroom_no_lidar/decroom_no_lidar/decroom_no_lidar | grep -i "fixture\|last_payload"
```

If the Step 2 debug block is still in the code, remove it. Commit as:

```bash
git add decroom_no_lidar/decroom_no_lidar/VideoRecorderView.swift
git commit -m "refactor: remove payload-fixture debug write

Used only during Plan 3a QA to capture the
tests/fixtures/phone_first_payload/ data. No longer needed."
```

---

## Task 15: Merge to Main + Branch Cleanup (Phase I)

**Purpose:** The finale. Merge `feature/phone-first-visual` → `main`, delete the feature branch.

- [ ] **Step 1: Rebase-check + final test suite**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git fetch origin
git checkout feature/phone-first-visual
git rebase origin/main   # resolve any conflicts (unlikely)
```

Then:

```bash
cd decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -10
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
cd ../decroom_no_lidar/backend
python3 -m pytest tests/ 2>&1 | tail -10
```

All must succeed.

- [ ] **Step 2: Merge with --no-ff**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git checkout main
git merge --no-ff feature/phone-first-visual -m "$(cat <<'EOF'
Merge feature/phone-first-visual into main — Plan 3a complete

Plan 3a (phone-first backend integration) ships the phone-first
architecture end-to-end. Key changes:

- iOS: ScanSession now runs live during VideoRecorderView scans,
  producing room geometry + PLY + keyframes
- Upload payload replaced entirely (old JPEG-only path deleted)
- New backend route /api/upload/phone-first runs SpatialLM on
  phone-provided PLY; old SfM/depth/layout/lifting deleted
- Depth model swapped to metric indoor variant (Hypersim
  fine-tune) — no scale calibration needed
- 12-case ScanError taxonomy with specific user-facing messages;
  .fallback tier devices gated at app launch
- Plan 2 dead code (6 visual UI files + tests) deleted

~5000 lines added, ~5000 lines deleted. End-to-end flow verified
on physical iPhone per docs/superpowers/qa/*-plan3a-manual-qa.md.

Spec: docs/superpowers/specs/2026-04-19-plan3a-phone-first-backend-integration-design.md
Plan: docs/superpowers/plans/2026-04-19-plan3a-phone-first-backend-integration.md
EOF
)"
```

- [ ] **Step 3: Push main + delete feature branch**

Push to origin (the user must confirm before pushing — do NOT push without permission):

```bash
# Confirm with user before running:
git push origin main
```

After user confirms push succeeded, delete the feature branch locally:

```bash
git branch -d feature/phone-first-visual
```

And optionally on the remote:

```bash
# Confirm with user:
git push origin --delete feature/phone-first-visual
```

- [ ] **Step 4: Update memory files**

Edit `/Users/jaejeon/.claude/projects/-Users-jaejeon-Desktop-Decroom-decroom-no-lidar/memory/plan_status.md`:

- Move Plan 2 from "IN PROGRESS" to "MERGED" with a note that the visual UI was abandoned
- Move Plan 3 from "NOT STARTED" to "Plan 3a MERGED, Plan 3b NOT STARTED"
- Add Plan 3a's delivered list

Commit the memory update (memory files are outside the repo, so this is a file write, not a git commit).

---

## Self-Review Pass

After all 15 tasks are written, verify the plan against the spec:

- [x] **Spec Section "Architecture"** → Tasks 3, 4 (wiring) + Task 5 (new upload) + Task 6 (backend) cover it
- [x] **Spec Section "Components"** → Each of the 13 components is addressed in Tasks 1-13
- [x] **Spec Section "Data shapes"** → `ScanUploadPayload` struct (Task 4) + `phone_first_pipeline.parse_payload` (Task 6) enforce the schemas
- [x] **Spec Section "Phone-side validation"** → Task 4 Step 5 implements ≥1 wall + ≥12 bins
- [x] **Spec Section "Error taxonomy"** → Task 4 Step 1 creates the enum; Task 9 surfaces it
- [x] **Spec Section "Testing"** → Layer 1 (Swift units) across Tasks 1, 4, 5; Layer 2 (Python) Task 6; Layer 3 (contract) Task 14 Step 2 creates the fixture; Layer 4 (manual) Task 14 Step 1
- [x] **Spec Section "Migration + deletion"** → Tasks map 1:1 to the spec's 9 phases
- [x] **Spec Section "Rollback paths"** → Commit order respects the "backend-before-phone, phone-delete-before-backend-delete" ordering
- [x] **Spec Section "Branch strategy"** → Task 15 merges `feature/phone-first-visual` → `main`
- [x] **Open question: scan guidance UX** → Explicitly out of scope; spec documents it as a deferred item

No placeholders, no "TBD," no unspecified code. Every task has either file paths + code or a specific shell command with expected output.

---

## What's next after Plan 3a?

**Plan 3b** (semantic polish, separate plan):
- YOLO/SpatialLM consensus (dual-model detection merge)
- Semantic wall-snapping rules (beds → walls, nightstands → walls)
- Per-object confidence scoring
- Populate `detections` array in keyframes.json (requires YOLO output to flow to upload)

**Plan 4a** (correction UI, separate plan):
- Tap-to-delete, tap-to-rotate on room objects in `RoomSceneView`
- Per-object confidence visual indicators (pulse / "?")

**Plan 4b** (furniture library, separate plan):
- Kenney model browser
- Manual furniture placement

**Scan-guidance polish** (evidence-driven, separate plan):
- Only if post-3a QA shows users consistently hit `.noWallsDetected` / `.insufficientCoverage`
- Adds lighting warnings, tilt prompts, parallax-motion hints to `VideoRecorderView`
