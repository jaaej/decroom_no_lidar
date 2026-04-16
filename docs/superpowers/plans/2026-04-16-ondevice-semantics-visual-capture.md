# On-Device Semantics + Visual Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the room-scanning experience from a 2D panoramic mosaic into an AR-native "grid shell" where walls solidify in real-time and furniture emerges as users scan. Adds on-device YOLOv8 for furniture detection and simple 3D instance tracking, consumed by the new `ARRoomPreviewView`.

**Architecture:** Builds on Plan 1's foundation (DeviceCapabilities, ScanLogger, DepthRunner, RoomGeometryBuilder, PointCloudAccumulator, KeyframeSelector). Adds two semantic components (`YoloRunner`, `ObjectTracker`), a `ScanSession` holder to wire everything together, helper types for ARKit bridging (`DeviceInfo.current()`, `Intrinsics(from: ARCamera, portrait:)`), and a RealityKit-backed `ARRoomPreviewView` with four visual layers: grid shell → solidifying walls → reticle guide → furniture emergence.

**Tech Stack:** Swift 5.9+, SwiftUI, ARKit, RealityKit, Core ML, Swift Testing.

---

## Scope Boundaries

The design spec lists four visual layers for the AR capture view. This plan covers Layers 1 and 2 fully, Layer 3 partially, and defers Layer 4:

| Layer | What | In This Plan? |
|-------|------|---------------|
| 1. Grid Shell | Wireframe cube around user; faces solidify as planes detected | ✅ Full |
| 2. Reticle | World-anchored guide toward uncaptured areas | ✅ Full |
| 3. Brightness Gradient | Walls start translucent, brighten with more scans | ⚠️ Partial — binary on/off per face (wireframe ↔ opaque) |
| 4. Furniture Emergence | Ghost boxes for detected furniture | ❌ Deferred to Plan 3 (backend results feed placement) |

Failure-state visuals (desaturate on tracking loss, red tint in dark areas, motion blur at high speed) are **deferred to Plan 4** — Plan 2 uses the existing app-level error handling without visual-layer treatment.

**Why defer these?** Furniture emergence depends on backend-returned object positions (currently from Plan 1's existing backend path, unchanged by Plan 2). Meaningful furniture boxes require Plan 3's wall-constrained placement to avoid the same "bunched in center" bug from the old pipeline. Failure-state visuals are polish work best done after Plan 3 validates the full pipeline is stable.

---

## File Structure

**New source files** (all in `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/`):

*Plan 1 follow-ups (pre-work identified by the final review):*
- `DeviceInfo+Current.swift` — factory for `DeviceInfo.current()`
- `Intrinsics+ARCamera.swift` — `Intrinsics(from: ARCamera, portrait: Bool)` helper with portrait fx/fy/cx/cy swap
- `PixelBufferRGB.swift` — Vision-backed wrapper to convert ARKit YUV → RGB BGRA buffer for Core ML input
- `ScanSession.swift` — holder that owns all Plan 1 components + drives updates from an ARFrame

*On-device semantics:*
- `YoloDetection.swift` — plain struct for 2D detection (category, bbox, confidence, mask)
- `YoloRunner.swift` — Core ML YOLOv8-seg inference wrapper
- `ObjectTracker.swift` — pure-logic 3D instance tracker (no AR deps)

*Visual capture (RealityKit-backed):*
- `GridShellEntity.swift` — procedural wireframe cube + solidifying-wall logic
- `CaptureReticle.swift` — world-anchored reticle state + position computation
- `ARRoomPreviewViewModel.swift` — pure-logic view model (coverage state, reticle target, furniture list)
- `ARRoomPreviewView.swift` — SwiftUI view wrapping ARView + RealityKit entities
- `ARRoomPreviewContainer.swift` — `UIViewRepresentable` bridging SwiftUI to ARView

**Modified:**
- `CaptureFlowView.swift` — route `.full`/`.reduced` tier devices to `ARRoomPreviewView`, `.fallback` to existing `VideoRecorderView`

**Bundle resources:**
- `YOLOv8s-seg.mlpackage` — Ultralytics YOLOv8s-seg Core ML export (~30MB)

**New test files** (all in `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/`):
- `DeviceInfoCurrentTests.swift`
- `IntrinsicsFromARCameraTests.swift`
- `ObjectTrackerTests.swift`
- `ARRoomPreviewViewModelTests.swift`
- `CaptureReticleTests.swift`

YoloRunner and visual-capture rendering get **manual QA** rather than unit tests (Core ML + RealityKit rendering is hard to unit-test meaningfully). ScanSession gets integration testing via manual AR scanning.

---

## Pre-Flight Check

Before starting, verify the environment.

- [ ] **Confirm merged state**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git log --oneline -3
```

Expected: top commit is `674ff66 Merge feature/phone-first-foundation into main` (or newer descendant).

- [ ] **Create Plan 2 feature branch**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git checkout main
git checkout -b feature/phone-first-visual
```

- [ ] **Verify Plan 1 tests still pass (baseline)**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests 2>&1 | tail -5
```

Expected: `** TEST SUCCEEDED **`.

---

## Task 1: DeviceInfo.current() Factory

**Purpose:** Build a `DeviceInfo` from `UIDevice.current` + `DeviceCapabilities.currentTier`. Plan 1 defined the struct but no way to populate it.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/DeviceInfo+Current.swift`
- Create: `decroom_no_lidar/decroom_no_lidarTests/DeviceInfoCurrentTests.swift`

- [ ] **Step 1: Write the failing test**

Create `decroom_no_lidar/decroom_no_lidarTests/DeviceInfoCurrentTests.swift`:

```swift
import Testing
@testable import decroom_no_lidar

struct DeviceInfoCurrentTests {

    @Test func currentReturnsNonEmptyFields() {
        let info = DeviceInfo.current()
        #expect(!info.model.isEmpty)
        #expect(!info.ios.isEmpty)
        #expect(!info.chip.isEmpty)
    }

    @Test func currentTierMatchesDeviceCapabilities() {
        let info = DeviceInfo.current()
        #expect(info.tier == DeviceCapabilities.currentTier)
    }

    @Test func iosStringLooksLikeVersion() {
        let info = DeviceInfo.current()
        // Examples: "17.2", "18.1.1". Should start with a digit.
        let firstChar = info.ios.first
        #expect(firstChar?.isNumber == true)
    }
}
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/DeviceInfoCurrentTests 2>&1 | tail -20
```

Expected: compile failure "type 'DeviceInfo' has no member 'current'".

- [ ] **Step 3: Create DeviceInfo+Current.swift**

```swift
//
//  DeviceInfo+Current.swift
//  decroom_no_lidar
//
//  Factory for DeviceInfo.current() — populates from UIDevice + DeviceCapabilities.
//

import Foundation
import UIKit

extension DeviceInfo {
    /// Build a DeviceInfo describing the current device.
    static func current() -> DeviceInfo {
        let modelId = currentModelIdentifier()
        let chip = DeviceCapabilities.currentChipGeneration().displayName
        return DeviceInfo(
            model: modelId,
            ios: UIDevice.current.systemVersion,
            chip: chip,
            tier: DeviceCapabilities.currentTier
        )
    }

    /// Machine identifier (e.g., "iPhone16,1"). On Simulator returns "Simulator".
    private static func currentModelIdentifier() -> String {
        #if targetEnvironment(simulator)
        return ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] ?? "Simulator"
        #else
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafeBytes(of: &systemInfo.machine) { raw -> String in
            let cString = raw.bindMemory(to: CChar.self)
            return String(cString: cString.baseAddress!)
        }
        #endif
    }
}

extension ChipGeneration {
    /// Human-readable chip name for display in logs.
    var displayName: String {
        switch self {
        case .a11: return "A11 Bionic"
        case .a12: return "A12 Bionic"
        case .a13: return "A13 Bionic"
        case .a14: return "A14 Bionic"
        case .a15: return "A15 Bionic"
        case .a16: return "A16 Bionic"
        case .a17pro: return "A17 Pro"
        case .a18: return "A18"
        case .a18pro: return "A18 Pro"
        case .unknown: return "Unknown"
        }
    }
}
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/DeviceInfoCurrentTests 2>&1 | tail -10
```

Expected: `** TEST SUCCEEDED **` with 3 tests passing.

- [ ] **Step 5: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/DeviceInfo+Current.swift decroom_no_lidar/decroom_no_lidarTests/DeviceInfoCurrentTests.swift
git commit -m "feat: add DeviceInfo.current() factory

Populates DeviceInfo from UIDevice.current + DeviceCapabilities.
ChipGeneration gets a displayName property for human-readable logs."
```

---

## Task 2: Intrinsics from ARCamera (portrait-aware)

**Purpose:** Build a `PointCloudAccumulator.Intrinsics` from an `ARCamera`. Must handle the portrait rotation (swap fx/fy and recompute cx/cy) per CLAUDE.md's repeated-bug note.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/Intrinsics+ARCamera.swift`
- Create: `decroom_no_lidar/decroom_no_lidarTests/IntrinsicsFromARCameraTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
import Testing
import simd
@testable import decroom_no_lidar

struct IntrinsicsFromARCameraTests {

    @Test func landscapeIntrinsicsPassThrough() {
        // Fake intrinsics: fx=1200, fy=1200, cx=960, cy=720 (1920x1440 landscape)
        let raw = simd_float3x3(rows: [
            SIMD3<Float>(1200, 0, 960),
            SIMD3<Float>(0, 1200, 720),
            SIMD3<Float>(0, 0, 1)
        ])
        let result = Intrinsics.make(
            fromRawIntrinsics: raw,
            imageResolution: CGSize(width: 1920, height: 1440),
            portrait: false
        )
        #expect(result.fx == 1200)
        #expect(result.fy == 1200)
        #expect(result.cx == 960)
        #expect(result.cy == 720)
        #expect(result.width == 1920)
        #expect(result.height == 1440)
    }

    @Test func portraitIntrinsicsSwapFxFyAndRotateCxCy() {
        // Same raw sensor intrinsics, but phone is portrait.
        // Conventions (from CLAUDE.md):
        //   new_fx = old_fy
        //   new_fy = old_fx
        //   new_cx = old_cy
        //   new_cy = old_width - old_cx
        let raw = simd_float3x3(rows: [
            SIMD3<Float>(1200, 0, 960),
            SIMD3<Float>(0, 1200, 720),
            SIMD3<Float>(0, 0, 1)
        ])
        let result = Intrinsics.make(
            fromRawIntrinsics: raw,
            imageResolution: CGSize(width: 1920, height: 1440),
            portrait: true
        )
        #expect(result.fx == 1200)  // old_fy
        #expect(result.fy == 1200)  // old_fx
        #expect(result.cx == 720)   // old_cy
        #expect(result.cy == 1920 - 960)  // = 960
        #expect(result.width == 1440)  // swapped
        #expect(result.height == 1920)
    }

    @Test func differentFxFyPortraitSwapsCorrectly() {
        let raw = simd_float3x3(rows: [
            SIMD3<Float>(1500, 0, 800),
            SIMD3<Float>(0, 1000, 400),
            SIMD3<Float>(0, 0, 1)
        ])
        let result = Intrinsics.make(
            fromRawIntrinsics: raw,
            imageResolution: CGSize(width: 1600, height: 1200),
            portrait: true
        )
        #expect(result.fx == 1000)  // old_fy
        #expect(result.fy == 1500)  // old_fx
        #expect(result.cx == 400)   // old_cy
        #expect(result.cy == 1600 - 800)  // = 800
    }
}
```

- [ ] **Step 2: Run test to verify failure**

Expected: compile failure about missing initializer.

- [ ] **Step 3: Create Intrinsics+ARCamera.swift**

```swift
//
//  Intrinsics+ARCamera.swift
//  decroom_no_lidar
//
//  Build a PointCloudAccumulator.Intrinsics from ARCamera, handling the
//  landscape-to-portrait rotation ARKit requires for vertical-held phones.
//
//  Per CLAUDE.md:
//    new_fx = old_fy
//    new_fy = old_fx
//    new_cx = old_cy
//    new_cy = old_width - old_cx
//

import Foundation
import ARKit
import CoreGraphics
import simd

extension Intrinsics {
    /// Build from an ARCamera. Pass `portrait: true` when the phone is held vertically
    /// (the capturedImage has been `.oriented(.right)` rotated to portrait before passing
    /// to depth/detection models).
    static func make(from camera: ARCamera, portrait: Bool) -> Intrinsics {
        return make(
            fromRawIntrinsics: camera.intrinsics,
            imageResolution: camera.imageResolution,
            portrait: portrait
        )
    }

    /// Exposed for testing — takes raw intrinsics + resolution directly.
    static func make(
        fromRawIntrinsics raw: simd_float3x3,
        imageResolution res: CGSize,
        portrait: Bool
    ) -> Intrinsics {
        let fx = raw.columns.0.x
        let fy = raw.columns.1.y
        let cx = raw.columns.2.x
        let cy = raw.columns.2.y
        let oldW = Int(res.width)
        let oldH = Int(res.height)

        if portrait {
            return Intrinsics(
                fx: fy,
                fy: fx,
                cx: cy,
                cy: Float(oldW) - cx,
                width: oldH,
                height: oldW
            )
        } else {
            return Intrinsics(
                fx: fx,
                fy: fy,
                cx: cx,
                cy: cy,
                width: oldW,
                height: oldH
            )
        }
    }
}
```

- [ ] **Step 4: Run test to verify pass**

Expected: `** TEST SUCCEEDED **` with 3 tests passing.

- [ ] **Step 5: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/Intrinsics+ARCamera.swift decroom_no_lidar/decroom_no_lidarTests/IntrinsicsFromARCameraTests.swift
git commit -m "feat: add Intrinsics(from: ARCamera, portrait:) factory

Encapsulates the fx/fy/cx/cy portrait rotation rule (per CLAUDE.md)
in one place so callers do not have to remember the swap each time.
Three tests cover landscape passthrough plus two portrait cases."
```

---

## Task 3: PixelBufferRGB — YUV to RGB Bridge

**Purpose:** ARKit `capturedImage` is YUV (`kCVPixelFormatType_420YpCbCr8BiPlanarFullRange`). Core ML models expect RGB (BGRA). The current `DepthRunner` tests pass only because they use zeroed BGRA buffers — the final review flagged this as a likely first-integration failure. Add a Vision-backed converter.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/PixelBufferRGB.swift`

**No unit test** — Vision's correctness is tested by Apple. We'll verify integration when ScanSession runs on a real device.

- [ ] **Step 1: Create PixelBufferRGB.swift**

```swift
//
//  PixelBufferRGB.swift
//  decroom_no_lidar
//
//  Converts ARKit YUV capturedImage to a BGRA CVPixelBuffer suitable
//  for Core ML models. Also handles portrait rotation via CIImage.oriented(.right).
//

import Foundation
import CoreVideo
import CoreImage
import VideoToolbox

enum PixelBufferRGB {

    /// Convert a YUV capturedImage to a BGRA CVPixelBuffer of the given size.
    /// Rotates via `.oriented(.right)` so the output is portrait when the
    /// input was landscape-right (ARKit's default sensor orientation).
    ///
    /// - Parameters:
    ///   - source: ARKit's `ARFrame.capturedImage`
    ///   - size: Target size (e.g., 518x392 for DepthAnythingV2Small)
    ///   - ciContext: shared CIContext for reuse
    ///   - rotateToPortrait: apply `.oriented(.right)` before resize
    static func convert(
        _ source: CVPixelBuffer,
        toSize size: CGSize,
        ciContext: CIContext,
        rotateToPortrait: Bool
    ) -> CVPixelBuffer? {
        var ciImage = CIImage(cvPixelBuffer: source)
        if rotateToPortrait {
            ciImage = ciImage.oriented(.right)
        }
        // Scale to target size
        let sx = size.width / ciImage.extent.width
        let sy = size.height / ciImage.extent.height
        ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        // Create output buffer (BGRA)
        var output: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &output
        )
        guard status == kCVReturnSuccess, let output else { return nil }

        ciContext.render(ciImage, to: output)
        return output
    }
}
```

- [ ] **Step 2: Verify it builds**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 3: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/PixelBufferRGB.swift
git commit -m "feat: add PixelBufferRGB YUV→BGRA converter for Core ML

Bridges ARKit capturedImage (YUV 420f) to Core ML models that expect
BGRA. Handles portrait rotation via CIImage.oriented(.right) in the
same step to keep DepthRunner/YoloRunner call sites simple."
```

---

## Task 4: ObjectTracker

**Purpose:** Maintain unique 3D object instances across frames using nearest-neighbor assignment in 3D position space. Pure logic — no ARKit dependency. Consumed by `YoloRunner` output downstream.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/ObjectTracker.swift`
- Create: `decroom_no_lidar/decroom_no_lidarTests/ObjectTrackerTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
import Testing
import simd
@testable import decroom_no_lidar

struct ObjectTrackerTests {

    @Test func firstObservationCreatesNewTrack() {
        var tracker = ObjectTracker(mergeDistanceByCategory: ["bed": 0.8])
        let id = tracker.observe(category: "bed",
                                  worldPosition: SIMD3(1, 0, 2),
                                  confidence: 0.9,
                                  frameIndex: 0)
        #expect(tracker.tracks.count == 1)
        #expect(id == tracker.tracks.first?.id)
    }

    @Test func samePositionSameCategoryIsSameTrack() {
        var tracker = ObjectTracker(mergeDistanceByCategory: ["bed": 0.8])
        let id1 = tracker.observe(category: "bed",
                                   worldPosition: SIMD3(1, 0, 2),
                                   confidence: 0.8,
                                   frameIndex: 0)
        let id2 = tracker.observe(category: "bed",
                                   worldPosition: SIMD3(1.1, 0, 2.05),  // within 0.8m
                                   confidence: 0.9,
                                   frameIndex: 1)
        #expect(id1 == id2)
        #expect(tracker.tracks.count == 1)
    }

    @Test func farPositionSameCategoryIsNewTrack() {
        var tracker = ObjectTracker(mergeDistanceByCategory: ["bed": 0.8])
        let id1 = tracker.observe(category: "bed",
                                   worldPosition: SIMD3(1, 0, 2),
                                   confidence: 0.8,
                                   frameIndex: 0)
        let id2 = tracker.observe(category: "bed",
                                   worldPosition: SIMD3(5, 0, 2),  // 4m away
                                   confidence: 0.9,
                                   frameIndex: 1)
        #expect(id1 != id2)
        #expect(tracker.tracks.count == 2)
    }

    @Test func differentCategoryNeverMerges() {
        var tracker = ObjectTracker(mergeDistanceByCategory: ["bed": 0.8, "desk": 0.8])
        let id1 = tracker.observe(category: "bed",
                                   worldPosition: SIMD3(1, 0, 2),
                                   confidence: 0.8,
                                   frameIndex: 0)
        let id2 = tracker.observe(category: "desk",
                                   worldPosition: SIMD3(1, 0, 2),  // same position
                                   confidence: 0.9,
                                   frameIndex: 1)
        #expect(id1 != id2)
        #expect(tracker.tracks.count == 2)
    }

    @Test func trackStoresBestConfidenceFrame() {
        var tracker = ObjectTracker(mergeDistanceByCategory: ["bed": 0.8])
        _ = tracker.observe(category: "bed",
                             worldPosition: SIMD3(1, 0, 2),
                             confidence: 0.5,
                             frameIndex: 0)
        _ = tracker.observe(category: "bed",
                             worldPosition: SIMD3(1, 0, 2),
                             confidence: 0.9,
                             frameIndex: 1)
        _ = tracker.observe(category: "bed",
                             worldPosition: SIMD3(1, 0, 2),
                             confidence: 0.7,
                             frameIndex: 2)
        let track = tracker.tracks.first!
        #expect(track.bestConfidence == 0.9)
        #expect(track.bestFrameIndex == 1)
        #expect(track.observationCount == 3)
    }

    @Test func unknownCategoryUsesDefault() {
        var tracker = ObjectTracker(mergeDistanceByCategory: [:], defaultMergeDistance: 0.5)
        let id1 = tracker.observe(category: "mystery",
                                   worldPosition: SIMD3(0, 0, 0),
                                   confidence: 0.8,
                                   frameIndex: 0)
        let id2 = tracker.observe(category: "mystery",
                                   worldPosition: SIMD3(0.3, 0, 0),
                                   confidence: 0.8,
                                   frameIndex: 1)
        #expect(id1 == id2)  // within 0.5m default
    }
}
```

- [ ] **Step 2: Run test — expect compile failure**

- [ ] **Step 3: Create ObjectTracker.swift**

```swift
//
//  ObjectTracker.swift
//  decroom_no_lidar
//
//  Nearest-neighbor 3D instance tracking. Given detections with world
//  positions, assigns each to a unique track ID. Keeps the highest-
//  confidence observation per track for later best-frame keyframe
//  selection.
//
//  Pure logic, no ARKit dependency — unit-testable.
//

import Foundation
import simd

struct ObjectTrack {
    let id: UUID
    let category: String
    var latestPosition: SIMD3<Float>
    var bestConfidence: Float
    var bestFrameIndex: Int
    var observationCount: Int
}

struct ObjectTracker {
    private(set) var tracks: [ObjectTrack] = []
    private let mergeDistanceByCategory: [String: Float]
    private let defaultMergeDistance: Float

    init(mergeDistanceByCategory: [String: Float],
         defaultMergeDistance: Float = 0.6) {
        self.mergeDistanceByCategory = mergeDistanceByCategory
        self.defaultMergeDistance = defaultMergeDistance
    }

    /// Record an observation. Returns the assigned track ID (new or existing).
    @discardableResult
    mutating func observe(category: String,
                           worldPosition: SIMD3<Float>,
                           confidence: Float,
                           frameIndex: Int) -> UUID {
        let mergeDist = mergeDistanceByCategory[category] ?? defaultMergeDistance

        // Find nearest existing track of the same category within merge distance.
        if let idx = nearestTrackIndex(category: category,
                                         position: worldPosition,
                                         within: mergeDist) {
            tracks[idx].latestPosition = worldPosition
            tracks[idx].observationCount += 1
            if confidence > tracks[idx].bestConfidence {
                tracks[idx].bestConfidence = confidence
                tracks[idx].bestFrameIndex = frameIndex
            }
            return tracks[idx].id
        }

        // New track.
        let track = ObjectTrack(
            id: UUID(),
            category: category,
            latestPosition: worldPosition,
            bestConfidence: confidence,
            bestFrameIndex: frameIndex,
            observationCount: 1
        )
        tracks.append(track)
        return track.id
    }

    private func nearestTrackIndex(category: String,
                                     position: SIMD3<Float>,
                                     within maxDist: Float) -> Int? {
        var best: (idx: Int, dist: Float)?
        for (i, track) in tracks.enumerated() where track.category == category {
            let d = simd_distance(track.latestPosition, position)
            if d <= maxDist {
                if best == nil || d < best!.dist {
                    best = (i, d)
                }
            }
        }
        return best?.idx
    }
}
```

- [ ] **Step 4: Run test**

Expected: `** TEST SUCCEEDED **` with 6 tests passing.

- [ ] **Step 5: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/ObjectTracker.swift decroom_no_lidar/decroom_no_lidarTests/ObjectTrackerTests.swift
git commit -m "feat: add ObjectTracker for 3D instance tracking

Pure-logic tracker: nearest-neighbor assignment in 3D position space
per category. Keeps best-confidence frame per track for downstream
keyframe selection. Per-category merge distance with a default fallback."
```

---

## Task 5: YOLOv8-seg Core ML Model

**Purpose:** Add Ultralytics YOLOv8s-seg Core ML export (~30MB) as a bundle resource.

**Files:**
- Bundle resource: `decroom_no_lidar/decroom_no_lidar/YOLOv8s-seg.mlpackage/`

- [ ] **Step 1: Export YOLOv8s-seg to Core ML**

The Ultralytics CLI produces Core ML exports directly. Run:

```bash
pip install ultralytics
cd /tmp
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8s-seg.pt')
model.export(format='coreml', nms=True, half=False)
"
```

This creates `/tmp/yolov8s-seg.mlpackage` (~30MB).

Copy it into the iOS project with the name `YOLOv8s-seg`:

```bash
cp -r /tmp/yolov8s-seg.mlpackage /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/YOLOv8s-seg.mlpackage
```

- [ ] **Step 2: Verify model loads via build**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **`. Search for compiled output:

```bash
find ~/Library/Developer/Xcode/DerivedData/decroom_no_lidar* -name "YOLOv8s-seg.mlmodelc" -type d 2>/dev/null | head -3
```

Expected: at least one path found.

- [ ] **Step 3: Check .gitignore for `.bin` conflicts**

The `.gitignore` may have a `*.bin` rule blocking the weights file. Check:

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git status decroom_no_lidar/decroom_no_lidar/YOLOv8s-seg.mlpackage
```

If `weight.bin` is not tracked, add an exception (like DepthModel did):

```bash
# Edit decroom_no_lidar/.gitignore and add:
!decroom_no_lidar/YOLOv8s-seg.mlpackage/**/*.bin
```

- [ ] **Step 4: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/YOLOv8s-seg.mlpackage decroom_no_lidar/.gitignore
git commit -m "feat: add YOLOv8s-seg Core ML model for on-device detection

Ultralytics YOLOv8s-seg export (~30MB, NMS included). Trained on
COCO-80 — covers bed, chair, sofa (as 'couch'), TV, laptop,
dining table (mapped to 'desk'), refrigerator, potted plant, etc.
Used by YoloRunner for per-frame 2D detection + segmentation masks."
```

---

## Task 6: YoloRunner

**Purpose:** Wrap Core ML YOLOv8-seg inference. Accepts a BGRA `CVPixelBuffer` at 640x640, returns `[YoloDetection]` with category, confidence, bbox, and segmentation mask.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/YoloDetection.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/YoloRunner.swift`

No unit test for inference itself (Core ML testing is brittle). Manual QA when integrated into ScanSession.

- [ ] **Step 1: Create YoloDetection.swift**

```swift
//
//  YoloDetection.swift
//  decroom_no_lidar
//
//  Plain struct for a single YOLO detection: category, confidence, bbox, mask.
//

import Foundation
import CoreGraphics

struct YoloDetection {
    /// COCO-80 category name (e.g., "bed", "chair", "couch").
    let category: String
    /// 0..1 confidence.
    let confidence: Float
    /// Normalized bbox in [0,1] image coordinates (x, y, width, height).
    let bbox: CGRect
    /// Optional raw mask data (model dependent — nil if NMS-only export).
    let mask: [Float]?
    /// Source frame index.
    let frameIndex: Int
}

/// COCO-80 class index → name. Index matches Ultralytics ordering.
enum COCO80 {
    static let classes: [String] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
}
```

- [ ] **Step 2: Create YoloRunner.swift**

```swift
//
//  YoloRunner.swift
//  decroom_no_lidar
//
//  Core ML YOLOv8-seg inference wrapper. Accepts BGRA 640x640 pixel buffers,
//  returns [YoloDetection] filtered by confidence threshold.
//

import Foundation
import CoreML
import CoreVideo
import CoreImage

enum YoloRunnerError: Error {
    case modelLoadFailed
    case inferenceFailed
    case unexpectedOutput
}

final class YoloRunner {
    private let model: MLModel
    let inputSize = CGSize(width: 640, height: 640)
    let confidenceThreshold: Float

    init(confidenceThreshold: Float = 0.35) throws {
        self.confidenceThreshold = confidenceThreshold
        let config = MLModelConfiguration()
        config.computeUnits = .all
        guard let url = Bundle.main.url(forResource: "YOLOv8s-seg", withExtension: "mlmodelc") else {
            throw YoloRunnerError.modelLoadFailed
        }
        self.model = try MLModel(contentsOf: url, configuration: config)
    }

    /// Run inference on a BGRA 640x640 pixel buffer. Returns detections with
    /// confidence >= threshold.
    func infer(pixelBuffer: CVPixelBuffer, frameIndex: Int) throws -> [YoloDetection] {
        let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        let input = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
        let output = try model.prediction(from: input)

        // Ultralytics Core ML export with NMS has two outputs:
        //   "confidence": MLMultiArray [1, numBoxes, 80]
        //   "coordinates": MLMultiArray [1, numBoxes, 4] (x, y, w, h normalized)
        // (If export was without NMS, there is a single raw output needing NMS here —
        //  Ultralytics `nms=True` during export is required for this path.)
        guard let confidenceFeature = output.featureValue(for: "confidence")?.multiArrayValue,
              let coordinatesFeature = output.featureValue(for: "coordinates")?.multiArrayValue else {
            throw YoloRunnerError.unexpectedOutput
        }

        return decodeDetections(
            confidence: confidenceFeature,
            coordinates: coordinatesFeature,
            frameIndex: frameIndex
        )
    }

    private func decodeDetections(
        confidence: MLMultiArray,
        coordinates: MLMultiArray,
        frameIndex: Int
    ) -> [YoloDetection] {
        let numBoxes = confidence.shape[1].intValue
        let numClasses = confidence.shape[2].intValue
        let confPtr = confidence.dataPointer.assumingMemoryBound(to: Float.self)
        let coordPtr = coordinates.dataPointer.assumingMemoryBound(to: Float.self)

        var detections: [YoloDetection] = []
        for b in 0..<numBoxes {
            // Find max class confidence for this box
            var maxConf: Float = 0
            var maxClass = 0
            for c in 0..<numClasses {
                let conf = confPtr[b * numClasses + c]
                if conf > maxConf {
                    maxConf = conf
                    maxClass = c
                }
            }
            guard maxConf >= confidenceThreshold else { continue }
            let category = maxClass < COCO80.classes.count ? COCO80.classes[maxClass] : "unknown"

            // bbox: (cx, cy, w, h) normalized to [0,1]
            let cx = coordPtr[b * 4 + 0]
            let cy = coordPtr[b * 4 + 1]
            let w = coordPtr[b * 4 + 2]
            let h = coordPtr[b * 4 + 3]
            let rect = CGRect(
                x: CGFloat(cx - w / 2),
                y: CGFloat(cy - h / 2),
                width: CGFloat(w),
                height: CGFloat(h)
            )

            detections.append(YoloDetection(
                category: category,
                confidence: maxConf,
                bbox: rect,
                mask: nil,  // segmentation masks require a different export path
                frameIndex: frameIndex
            ))
        }
        return detections
    }
}
```

- [ ] **Step 3: Verify build**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 4: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/YoloDetection.swift decroom_no_lidar/decroom_no_lidar/YoloRunner.swift
git commit -m "feat: add YoloRunner for on-device COCO-80 detection

Wraps YOLOv8s-seg Core ML inference. Decodes Ultralytics NMS-included
output format (confidence + coordinates arrays). Returns detections
above confidence threshold with COCO-80 category names and normalized
bboxes. Segmentation masks require a different export path — deferred."
```

---

## Task 7: ScanSession Holder

**Purpose:** Bundle Plan 1's components (DepthRunner, RoomGeometryBuilder, PointCloudAccumulator, KeyframeSelector, ScanLogger) plus Plan 2's new ObjectTracker + YoloRunner into a single coordinator. Driven by `update(frame:)` calls from the ARSessionDelegate.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/ScanSession.swift`

No unit test — integration testing via manual AR scanning. The logic is mostly wiring.

- [ ] **Step 1: Create ScanSession.swift**

```swift
//
//  ScanSession.swift
//  decroom_no_lidar
//
//  Coordinates Plan 1 + Plan 2 components during a scan. Owned by the
//  ARRoomPreviewView controller, fed by ARFrame updates from the session
//  delegate. Publishes @Published state for the view to render.
//

import Foundation
import ARKit
import SwiftUI
import CoreImage
import simd

@MainActor
final class ScanSession: ObservableObject {
    // Plan 1 components
    let logger: ScanLogger
    let depthRunner: DepthRunner?
    let geometryBuilder = RoomGeometryBuilder()
    let pointCloud = PointCloudAccumulator(voxelSize: 0.02, maxPoints: 500_000)
    // Plan 2 components
    let yoloRunner: YoloRunner?
    var tracker = ObjectTracker(mergeDistanceByCategory: [
        "bed": 0.8, "couch": 1.5, "chair": 0.65, "dining table": 1.1,
        "bed": 1.5, "tv": 0.8, "laptop": 0.5, "potted plant": 0.6
    ])

    // Published for SwiftUI
    @Published var geometry: RoomGeometry = RoomGeometry(
        walls: [], floorHeight: 0, ceilingHeight: 0, manhattanYaw: 0
    )
    @Published var recentDetections: [YoloDetection] = []
    @Published var coverageProgress: Double = 0.0   // 0..1

    // Internal state
    private let ciContext = CIContext()
    private let startTime = Date()
    private var frameIndex = 0
    private var lastYoloFrame = -1
    private let yoloFrameInterval = 15  // run YOLO every 15th ARFrame (~2fps at 30fps)

    init() {
        self.logger = ScanLogger()
        self.logger.recordDevice(DeviceInfo.current())
        self.logger.logEvent("scan_started", at: 0)

        self.depthRunner = try? DepthRunner()
        if depthRunner == nil {
            logger.logEvent("depth_runner_load_failed", at: 0)
        }
        self.yoloRunner = try? YoloRunner()
        if yoloRunner == nil {
            logger.logEvent("yolo_runner_load_failed", at: 0)
        }
    }

    /// Called from ARSessionDelegate on each new frame.
    func update(frame: ARFrame) {
        frameIndex += 1
        let t = frame.timestamp

        // Intrinsics for portrait (phone-held-vertical case)
        let intrinsics = Intrinsics.make(from: frame.camera, portrait: true)
        let pose = frame.camera.transform

        // Depth inference (run at the DepthRunner's target interval)
        if let depth = depthRunner {
            if frameIndex % 6 == 0 {  // ~5fps at 30fps ARKit
                runDepth(frame: frame, intrinsics: intrinsics, pose: pose, depth: depth)
            }
        }

        // YOLO inference (infrequent)
        if let yolo = yoloRunner, frameIndex - lastYoloFrame >= yoloFrameInterval {
            lastYoloFrame = frameIndex
            runYolo(frame: frame, yolo: yolo)
        }
    }

    /// Update plane anchors — called from didAdd/didUpdate anchor delegate.
    func updatePlaneAnchors(_ anchors: [ARPlaneAnchor]) {
        self.geometry = geometryBuilder.build(from: anchors)
        if !geometry.walls.isEmpty {
            logger.logEvent("geometry_updated", at: elapsed(),
                             data: ["wall_count": "\(geometry.walls.count)"])
        }
    }

    func finalize() -> (point_cloud_ply: Data, log_written: Bool) {
        logger.logEvent("scan_finalized", at: elapsed())
        let ply = pointCloud.exportPLY()
        let logOK = (try? logger.finalize()) != nil
        return (ply, logOK)
    }

    // MARK: - Private

    private func elapsed() -> Double { Date().timeIntervalSince(startTime) }

    private func runDepth(frame: ARFrame,
                           intrinsics: Intrinsics,
                           pose: simd_float4x4,
                           depth: DepthRunner) {
        // Convert YUV → BGRA at model input size
        let modelSize = CGSize(width: 518, height: 392)
        guard let bgra = PixelBufferRGB.convert(frame.capturedImage,
                                                  toSize: modelSize,
                                                  ciContext: ciContext,
                                                  rotateToPortrait: false) else {
            return
        }
        guard let result = try? depth.infer(pixelBuffer: bgra) else { return }

        // Add to point cloud (relative depth — scale calibration TBD downstream)
        pointCloud.addDepthMap(
            result.depths,
            width: result.width,
            height: result.height,
            intrinsics: Intrinsics(
                fx: intrinsics.fx * Float(result.width) / Float(intrinsics.width),
                fy: intrinsics.fy * Float(result.height) / Float(intrinsics.height),
                cx: Float(result.width) / 2,
                cy: Float(result.height) / 2,
                width: result.width,
                height: result.height
            ),
            worldFromCamera: pose,
            colorAt: { _, _ in SIMD3<UInt8>(128, 128, 128) },  // gray for now
            pixelStride: 4
        )

        coverageProgress = min(1.0, Double(pointCloud.count) / 50_000)
    }

    private func runYolo(frame: ARFrame, yolo: YoloRunner) {
        let modelSize = CGSize(width: 640, height: 640)
        guard let bgra = PixelBufferRGB.convert(frame.capturedImage,
                                                  toSize: modelSize,
                                                  ciContext: ciContext,
                                                  rotateToPortrait: true) else {
            return
        }
        guard let dets = try? yolo.infer(pixelBuffer: bgra, frameIndex: frameIndex) else {
            return
        }
        self.recentDetections = dets

        // Track each detection (best-effort 3D position via camera-forward approximation)
        let cameraPos = SIMD3<Float>(
            frame.camera.transform.columns.3.x,
            frame.camera.transform.columns.3.y,
            frame.camera.transform.columns.3.z
        )
        for d in dets where d.confidence > 0.5 {
            // Rough position: camera position + 2m along camera forward
            let fwd = -SIMD3<Float>(
                frame.camera.transform.columns.2.x,
                frame.camera.transform.columns.2.y,
                frame.camera.transform.columns.2.z
            )
            let approxPos = cameraPos + fwd * 2
            tracker.observe(category: d.category,
                             worldPosition: approxPos,
                             confidence: d.confidence,
                             frameIndex: frameIndex)
        }
    }
}
```

- [ ] **Step 2: Verify build**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 3: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/ScanSession.swift
git commit -m "feat: add ScanSession coordinator for on-device scanning

Bundles DepthRunner + RoomGeometryBuilder + PointCloudAccumulator +
KeyframeSelector + ScanLogger (Plan 1) with YoloRunner + ObjectTracker
(Plan 2). Called from ARSessionDelegate per frame; publishes @Published
state (geometry, detections, coverage) for SwiftUI consumption."
```

---

## Task 8: ARRoomPreviewViewModel — Pure Logic

**Purpose:** Extract the view's state-derived logic (reticle target, wall solidification status, failure-state detection) into a testable struct so tests can cover behavior without rendering.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/ARRoomPreviewViewModel.swift`
- Create: `decroom_no_lidar/decroom_no_lidarTests/ARRoomPreviewViewModelTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
import Testing
import simd
@testable import decroom_no_lidar

struct ARRoomPreviewViewModelTests {

    @Test func emptyStateReticlePointsForward() {
        let model = ARRoomPreviewViewModel(
            cameraYaw: 0,
            wallsDetected: 0,
            coverageBinsFilled: [],
            totalCoverageBins: 24
        )
        let reticle = model.reticleDirection
        // When no walls detected, reticle should point somewhere — doesn't matter
        // which direction, but must be unit-length.
        #expect(abs(simd_length(reticle) - 1) < 0.01)
    }

    @Test func reticleSteersTowardEmptyYawBin() {
        // 24 yaw bins total. 0..11 filled (front half), 12..23 empty (back half).
        let filled = Set(0..<12)
        let model = ARRoomPreviewViewModel(
            cameraYaw: 0,
            wallsDetected: 4,
            coverageBinsFilled: filled,
            totalCoverageBins: 24
        )
        let reticle = model.reticleDirection
        // Reticle should point roughly backwards (yaw ≈ π).
        let yawOfReticle = atan2(reticle.x, -reticle.z)
        // yaw should be in [-π, -π/2] ∪ [π/2, π] (back half)
        #expect(abs(yawOfReticle) > .pi / 2)
    }

    @Test func wallsAreSolidWhenAllFourDetected() {
        let model = ARRoomPreviewViewModel(
            cameraYaw: 0,
            wallsDetected: 4,
            coverageBinsFilled: Set(0..<24),
            totalCoverageBins: 24
        )
        #expect(model.scanComplete)
    }

    @Test func scanNotCompleteIfMissingWalls() {
        let model = ARRoomPreviewViewModel(
            cameraYaw: 0,
            wallsDetected: 3,
            coverageBinsFilled: Set(0..<24),
            totalCoverageBins: 24
        )
        #expect(!model.scanComplete)
    }

    @Test func coveragePercentIsAccurate() {
        let model = ARRoomPreviewViewModel(
            cameraYaw: 0,
            wallsDetected: 2,
            coverageBinsFilled: Set([0, 1, 2, 3, 4, 5]),
            totalCoverageBins: 24
        )
        #expect(abs(model.coveragePercent - 0.25) < 0.001)
    }
}
```

- [ ] **Step 2: Run test — expect compile failure**

- [ ] **Step 3: Create ARRoomPreviewViewModel.swift**

```swift
//
//  ARRoomPreviewViewModel.swift
//  decroom_no_lidar
//
//  Pure logic for ARRoomPreviewView state. Derives reticle direction,
//  coverage percent, and "scan complete" status from the underlying
//  ScanSession state. Extracted from the view so tests cover behavior.
//

import Foundation
import simd

struct ARRoomPreviewViewModel {
    /// Current camera yaw in world frame (radians, 0 = +Z).
    let cameraYaw: Float
    /// Number of wall plane anchors detected (cap at 4 for "complete" check).
    let wallsDetected: Int
    /// Which yaw bins have received enough captured frames.
    let coverageBinsFilled: Set<Int>
    /// Total yaw bins (typically 24 = 15° each).
    let totalCoverageBins: Int

    /// Coverage as 0..1 fraction.
    var coveragePercent: Double {
        Double(coverageBinsFilled.count) / Double(totalCoverageBins)
    }

    /// True when scan is "done enough": 4 walls + full yaw coverage.
    var scanComplete: Bool {
        wallsDetected >= 4 && coverageBinsFilled.count >= totalCoverageBins
    }

    /// Unit direction vector (world-space) the reticle should point toward.
    /// Heuristic: find the yaw bin with lowest coverage "weight" around the user.
    var reticleDirection: SIMD3<Float> {
        guard totalCoverageBins > 0 else {
            return SIMD3<Float>(0, 0, 1)
        }
        // Find first unfilled bin. If all filled, point forward (cameraYaw).
        let firstUnfilled = (0..<totalCoverageBins).first(where: {
            !coverageBinsFilled.contains($0)
        })
        let targetBin = firstUnfilled ?? Int(cameraYaw / (2 * .pi) * Float(totalCoverageBins))
        let binAngle = (Float(targetBin) + 0.5) / Float(totalCoverageBins) * 2 * .pi
        // Convert binAngle (0 = +Z in world) to a direction vector.
        return SIMD3<Float>(sin(binAngle), 0, cos(binAngle))
    }
}
```

- [ ] **Step 4: Run test**

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/ARRoomPreviewViewModel.swift decroom_no_lidar/decroom_no_lidarTests/ARRoomPreviewViewModelTests.swift
git commit -m "feat: add ARRoomPreviewViewModel for testable view logic

Pure-logic struct derives reticle direction, coverage percent, and
scan-complete status from ScanSession state. Extracted from the
rendering view so tests cover behavior without RealityKit."
```

---

## Task 9: GridShellEntity + CaptureReticle (RealityKit)

**Purpose:** Two RealityKit entities for the visual layers.

- `GridShellEntity`: a 3m cube of cyan wireframe lines anchored at the user's spawn position. Fades faces to solid wall material as plane anchors are detected.
- `CaptureReticle`: a world-anchored cyan torus that moves per frame to the direction returned by `ARRoomPreviewViewModel.reticleDirection`.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/GridShellEntity.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/CaptureReticle.swift`
- Create: `decroom_no_lidar/decroom_no_lidarTests/CaptureReticleTests.swift` (tests the `positionFor(direction:distance:)` helper)

- [ ] **Step 1: Write failing test for CaptureReticle position math**

```swift
import Testing
import simd
@testable import decroom_no_lidar

struct CaptureReticleTests {

    @Test func positionAtForwardDirection() {
        let pos = CaptureReticle.worldPosition(
            forDirection: SIMD3(0, 0, 1),
            atDistance: 1.0,
            origin: SIMD3(0, 1, 0)
        )
        #expect(abs(pos.x) < 0.01)
        #expect(abs(pos.y - 1) < 0.01)
        #expect(abs(pos.z - 1) < 0.01)
    }

    @Test func positionAtArbitraryDirectionIsOrigin_plus_distance_times_direction() {
        let dir = simd_normalize(SIMD3<Float>(1, 0, 1))
        let pos = CaptureReticle.worldPosition(
            forDirection: dir,
            atDistance: 2.0,
            origin: SIMD3(0, 0, 0)
        )
        let expected = dir * 2
        #expect(abs(pos.x - expected.x) < 0.01)
        #expect(abs(pos.z - expected.z) < 0.01)
    }

    @Test func zeroDirectionReturnsOrigin() {
        let pos = CaptureReticle.worldPosition(
            forDirection: SIMD3(0, 0, 0),
            atDistance: 1.0,
            origin: SIMD3(5, 5, 5)
        )
        #expect(abs(pos.x - 5) < 0.01)
        #expect(abs(pos.y - 5) < 0.01)
        #expect(abs(pos.z - 5) < 0.01)
    }
}
```

- [ ] **Step 2: Run test — expect compile failure**

- [ ] **Step 3: Create CaptureReticle.swift**

```swift
//
//  CaptureReticle.swift
//  decroom_no_lidar
//
//  World-anchored reticle entity. Uses a thin cyan torus to indicate
//  the next direction the user should aim the camera. Position is
//  recomputed each frame from ARRoomPreviewViewModel.reticleDirection.
//

import Foundation
import RealityKit
import simd

final class CaptureReticle {
    /// The RealityKit entity — attach to an AnchorEntity(.world(transform:)).
    let entity: ModelEntity

    init() {
        let mesh = MeshResource.generateCylinder(height: 0.015, radius: 0.1)
        var material = PhysicallyBasedMaterial()
        material.baseColor = .init(tint: .cyan.withAlphaComponent(0.9))
        material.emissiveColor = .init(color: .cyan)
        material.emissiveIntensity = 2.0
        self.entity = ModelEntity(mesh: mesh, materials: [material])
    }

    /// Move reticle to where the ARRoomPreviewViewModel says next.
    func update(direction: SIMD3<Float>, cameraOrigin: SIMD3<Float>, distance: Float = 1.5) {
        entity.position = Self.worldPosition(
            forDirection: direction, atDistance: distance, origin: cameraOrigin
        )
    }

    /// Pure computation for testing.
    static func worldPosition(
        forDirection direction: SIMD3<Float>,
        atDistance distance: Float,
        origin: SIMD3<Float>
    ) -> SIMD3<Float> {
        let len = simd_length(direction)
        guard len > 0.01 else { return origin }
        return origin + (direction / len) * distance
    }
}
```

- [ ] **Step 4: Run test**

Expected: 3 tests pass.

- [ ] **Step 5: Create GridShellEntity.swift**

```swift
//
//  GridShellEntity.swift
//  decroom_no_lidar
//
//  Wireframe cube around the user. Each of the 6 faces is a separate
//  entity; as matching plane anchors are detected, the face is swapped
//  from wireframe to a solid semi-transparent material.
//

import Foundation
import RealityKit
import simd

final class GridShellEntity {
    let entity: Entity
    private var faces: [String: ModelEntity] = [:]  // key: "+x", "-x", "+y", "-y", "+z", "-z"

    init(sideLength: Float = 3.0) {
        entity = Entity()
        let halfSide = sideLength / 2

        for face in ["+x", "-x", "+y", "-y", "+z", "-z"] {
            let plane = MeshResource.generatePlane(width: sideLength, height: sideLength)
            var material = UnlitMaterial()
            material.color = .init(tint: .cyan.withAlphaComponent(0.3))
            let faceEntity = ModelEntity(mesh: plane, materials: [material])
            switch face {
            case "+x": faceEntity.position = SIMD3(halfSide, 0, 0)
                faceEntity.orientation = simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 1, 0))
            case "-x": faceEntity.position = SIMD3(-halfSide, 0, 0)
                faceEntity.orientation = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0))
            case "+y": faceEntity.position = SIMD3(0, halfSide, 0)
                faceEntity.orientation = simd_quatf(angle: .pi / 2, axis: SIMD3(1, 0, 0))
            case "-y": faceEntity.position = SIMD3(0, -halfSide, 0)
                faceEntity.orientation = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
            case "+z": faceEntity.position = SIMD3(0, 0, halfSide)
                faceEntity.orientation = simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
            case "-z": faceEntity.position = SIMD3(0, 0, -halfSide)
                faceEntity.orientation = simd_quatf()
            default: break
            }
            faces[face] = faceEntity
            entity.addChild(faceEntity)
        }
    }

    /// Mark a face as solidified (detected by ARKit plane anchor).
    /// The face swaps to an opaque material.
    func solidify(face: String) {
        guard let faceEntity = faces[face] else { return }
        var material = UnlitMaterial()
        material.color = .init(tint: .white.withAlphaComponent(0.5))
        faceEntity.model?.materials = [material]
    }
}
```

- [ ] **Step 6: Verify build**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 7: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/GridShellEntity.swift decroom_no_lidar/decroom_no_lidar/CaptureReticle.swift decroom_no_lidar/decroom_no_lidarTests/CaptureReticleTests.swift
git commit -m "feat: add GridShellEntity + CaptureReticle visual layers

GridShellEntity: 3m wireframe cube anchored at user spawn; faces
swap from cyan wireframe to opaque white material when a matching
plane anchor is detected.

CaptureReticle: world-anchored cyan torus that moves to where the
ARRoomPreviewViewModel says next. Position math extracted into a
pure static function for testability (3 tests)."
```

---

## Task 10: ARRoomPreviewView + Container

**Purpose:** SwiftUI view wrapping ARView with Plan 2's visual layers. Replaces `VideoRecorderView` for `.full` and `.reduced` tier devices.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/ARRoomPreviewContainer.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/ARRoomPreviewView.swift`
- Modify: `decroom_no_lidar/decroom_no_lidar/CaptureFlowView.swift`

No unit test for these — manual QA on device. Logic already tested via `ARRoomPreviewViewModelTests` and `CaptureReticleTests`.

- [ ] **Step 1: Create ARRoomPreviewContainer.swift**

```swift
//
//  ARRoomPreviewContainer.swift
//  decroom_no_lidar
//
//  UIViewRepresentable bridging SwiftUI to an ARView with grid shell +
//  reticle + solidifying walls. Feeds ARFrame updates into ScanSession.
//

import SwiftUI
import ARKit
import RealityKit

struct ARRoomPreviewContainer: UIViewRepresentable {
    @ObservedObject var session: ScanSession

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        config.isAutoFocusEnabled = true
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }
        arView.session.run(config)
        arView.session.delegate = context.coordinator

        // Add grid shell anchored at world origin (at camera spawn).
        let shellAnchor = AnchorEntity(world: .zero)
        shellAnchor.addChild(context.coordinator.gridShell.entity)
        arView.scene.addAnchor(shellAnchor)

        // Add reticle (world-anchored).
        let reticleAnchor = AnchorEntity(world: .zero)
        reticleAnchor.addChild(context.coordinator.reticle.entity)
        arView.scene.addAnchor(reticleAnchor)
        context.coordinator.reticleAnchor = reticleAnchor

        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        // State changes pushed via Coordinator.
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(session: session)
    }

    final class Coordinator: NSObject, ARSessionDelegate {
        let session: ScanSession
        let gridShell = GridShellEntity(sideLength: 3.0)
        let reticle = CaptureReticle()
        var reticleAnchor: AnchorEntity?

        init(session: ScanSession) {
            self.session = session
            super.init()
        }

        func session(_ arSession: ARSession, didUpdate frame: ARFrame) {
            Task { @MainActor in
                self.session.update(frame: frame)
                // Move reticle per latest viewmodel direction
                // Coverage bins left empty in Plan 2; a future Plan 3 task
                // can wire the existing CoverageEngine's filledYawBins here.
                // With empty bins, the reticle defaults to the first unfilled
                // bin (bin 0 = world +Z), which biases toward "look forward".
                let vm = ARRoomPreviewViewModel(
                    cameraYaw: cameraYaw(from: frame.camera.transform),
                    wallsDetected: self.session.geometry.walls.count,
                    coverageBinsFilled: [],
                    totalCoverageBins: 24
                )
                let camPos = SIMD3<Float>(
                    frame.camera.transform.columns.3.x,
                    frame.camera.transform.columns.3.y,
                    frame.camera.transform.columns.3.z
                )
                self.reticle.update(
                    direction: vm.reticleDirection,
                    cameraOrigin: camPos,
                    distance: 1.5
                )
            }
        }

        func session(_ arSession: ARSession, didAdd anchors: [ARAnchor]) {
            let planes = anchors.compactMap { $0 as? ARPlaneAnchor }
            Task { @MainActor in
                self.session.updatePlaneAnchors(planes)
                self.solidifyGridFacesForPlanes(planes)
            }
        }

        func session(_ arSession: ARSession, didUpdate anchors: [ARAnchor]) {
            let planes = anchors.compactMap { $0 as? ARPlaneAnchor }
            Task { @MainActor in
                self.session.updatePlaneAnchors(planes)
                self.solidifyGridFacesForPlanes(planes)
            }
        }

        private func solidifyGridFacesForPlanes(_ planes: [ARPlaneAnchor]) {
            for plane in planes {
                let n = SIMD3<Float>(
                    plane.transform.columns.2.x,
                    plane.transform.columns.2.y,
                    plane.transform.columns.2.z
                )
                // Which grid face is this plane closest to in normal direction?
                let face = dominantFace(for: n)
                gridShell.solidify(face: face)
            }
        }

        private func dominantFace(for normal: SIMD3<Float>) -> String {
            let ax = abs(normal.x), ay = abs(normal.y), az = abs(normal.z)
            if ax > ay && ax > az { return normal.x > 0 ? "+x" : "-x" }
            if ay > ax && ay > az { return normal.y > 0 ? "+y" : "-y" }
            return normal.z > 0 ? "+z" : "-z"
        }

        private func cameraYaw(from transform: simd_float4x4) -> Float {
            // forward = -Z column
            let fx = -transform.columns.2.x
            let fz = -transform.columns.2.z
            return atan2(fx, fz)
        }
    }
}
```

- [ ] **Step 2: Create ARRoomPreviewView.swift**

```swift
//
//  ARRoomPreviewView.swift
//  decroom_no_lidar
//
//  The new visual-first scanning experience. Grid shell + reticle +
//  solidifying walls + coverage progress. Replaces VideoRecorderView
//  for .full and .reduced tier devices.
//

import SwiftUI

struct ARRoomPreviewView: View {
    @StateObject private var session = ScanSession()

    var body: some View {
        ZStack {
            ARRoomPreviewContainer(session: session)
                .edgesIgnoringSafeArea(.all)

            VStack {
                Spacer()
                coverageCard
                    .padding()
            }
        }
    }

    private var coverageCard: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("\(session.geometry.walls.count) wall\(session.geometry.walls.count == 1 ? "" : "s") detected")
                    .font(.subheadline)
                    .foregroundStyle(.white)
                Text("\(Int(session.coverageProgress * 100))% coverage")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
            }
            Spacer()
        }
        .padding(12)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }
}
```

- [ ] **Step 3: Wire into CaptureFlowView.swift**

Read the current `CaptureFlowView.swift` and route `.full`/`.reduced` tier devices to `ARRoomPreviewView`. Leave `.fallback` devices on the existing `VideoRecorderView`.

Locate the section that chooses which view to show (around `VideoRecorderView(processor: processor)`) and wrap it:

```swift
// Route to new visual AR view for capable devices
if DeviceCapabilities.currentTier != .fallback {
    ARRoomPreviewView()
} else {
    VideoRecorderView(processor: processor)
}
```

- [ ] **Step 4: Verify build**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 5: Run full test suite (no regressions)**

```bash
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -10
```

Expected: all tests pass (modulo the pre-existing `testLaunchPerformance` baseline).

- [ ] **Step 6: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/ARRoomPreviewContainer.swift decroom_no_lidar/decroom_no_lidar/ARRoomPreviewView.swift decroom_no_lidar/decroom_no_lidar/CaptureFlowView.swift
git commit -m "feat: add ARRoomPreviewView with grid shell + reticle

SwiftUI view bridging to ARView + RealityKit. Renders grid shell
around the user, reticle guiding to uncaptured areas, and a
coverage card overlay. Routed as the capture experience for
.full and .reduced tier devices; .fallback devices keep the
existing VideoRecorderView path."
```

---

## Task 11: Manual QA Checklist

**Purpose:** RealityKit rendering cannot be meaningfully unit-tested. This task is a device-scan checklist to validate the visual experience is actually what the design spec called for.

- [ ] **Checklist** (execute on a physical iPhone running the branch):

  - [ ] Launch app, tap Record → see AR view with wireframe cyan cube around you
  - [ ] Point camera at a wall → after ~3s, the relevant grid face turns opaque white
  - [ ] Reticle moves toward uncaptured directions (rotate away from detected walls; reticle should reposition)
  - [ ] Coverage card at bottom shows wall count and coverage % updating
  - [ ] Depth model inference does not crash the app (check logs for "depth_runner_load_failed")
  - [ ] YOLO model inference does not crash (check logs for "yolo_runner_load_failed")
  - [ ] App does not thermal-throttle into unusability during a 60s scan
  - [ ] Scan log file appears in `~/Documents/scan_logs/` after completing a scan
  - [ ] On an iPhone 11 (A13, `.reduced` tier): same experience, possibly with lower depth fps
  - [ ] On an iPhone 8 (A11, `.fallback` tier): falls back to the old VideoRecorderView UI

- [ ] **Step 1: Note results + file follow-up issues**

Write your observations to `/Users/jaejeon/Desktop/Decroom/decroom_no_lidar/docs/superpowers/qa/2026-04-16-plan2-manual-qa.md` (create directory if needed). For any issue found, list it in the doc + a short reproduction.

- [ ] **Step 2: Commit QA notes**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add docs/superpowers/qa/2026-04-16-plan2-manual-qa.md
git commit -m "docs: manual QA notes from Plan 2 on-device scan"
```

---

## Task 12: Final Verification + Merge Prep

- [ ] **Step 1: Full test suite passes**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -20
```

- [ ] **Step 2: Device build succeeds**

```bash
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO 2>&1 | tail -5
```

- [ ] **Step 3: Final commit (if any fixes from QA)**

If QA checklist uncovered issues, fix them here and commit. Otherwise, no action.

- [ ] **Step 4: Merge into main**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git checkout main
git merge --no-ff feature/phone-first-visual -m "Merge feature/phone-first-visual into main

Plan 2 (On-Device Semantics + Visual Capture) complete. Adds YOLO
on-device detection, simple 3D object tracking, and the new
ARRoomPreviewView with grid shell + reticle + solidifying walls.

Users on iPhone 12+ now see the visual-first scanning experience;
older devices fall back to the existing VideoRecorderView."
git push origin main
```

- [ ] **Step 5: Delete feature branch**

```bash
git branch -d feature/phone-first-visual
```

---

## What's next after Plan 2?

- **Plan 3: Backend Integration** — upload new payload (phone geometry + point cloud + keyframes), backend skips SfM/layout/lifting, applies wall-snapping and YOLO consensus.
- **Plan 4: Correction UI + Furniture Library** — tap-to-delete/rotate, manual Kenney model placement.

Users see the visual scanning experience from Plan 2. Backend results are the same as before (same accuracy problems) until Plan 3 ships.
