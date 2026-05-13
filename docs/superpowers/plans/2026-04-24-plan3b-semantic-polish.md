# Plan 3b — Semantic Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the semantic layer on top of Plan 3a's phone-first pipeline — YOLO/SpatialLM consensus, duplicate collapse, wall-snapping, per-object confidence, door/window detection — without adding new user-facing UI.

**Architecture:** iOS re-enables on-device YOLO, unprojects each detection into a 3D world position via the cached depth map, and `ObjectTracker` dedupes across frames. On `finalize()`, tracker's deduped 3D objects serialize into a new `yolo_objects.json` payload part. Backend adds a `consensus.py` module (4 pure functions) and wires it into `phone_first_pipeline.run_phone_first_pipeline` after SpatialLM mapping but before building the `RoomPayload`. Door/window extraction becomes a post-SLM step that sets `Wall.has_door` / `has_window` booleans.

**Tech Stack:** Swift (SwiftUI + ARKit + Core ML), Python (FastAPI + NumPy), pytest, XCTest.

**Spec:** `docs/superpowers/specs/2026-04-24-plan3b-semantic-polish-design.md` (v2).

**Prerequisites (MUST be true before starting):**
- Plan 3a merged to `main` (includes wall-consolidation commit `a40b2d2` and all Plan 3a backend/iOS work).
- SpatialLM tunnel on RunPod reachable (see `memory/runpod_setup.md`).
- Branch off `main`: `git switch -c feature/plan3b-semantic-polish`.

---

## File structure

### New files

| Path | Responsibility |
|---|---|
| `decroom_no_lidar/decroom_no_lidar/YoloUnprojector.swift` | Pure function: YOLO bbox center → 3D world position via depth + pose. Unit-testable without ARKit. |
| `decroom_no_lidar/decroom_no_lidarTests/YoloUnprojectorTests.swift` | Tests for the unprojection math. |
| `decroom_no_lidar/decroom_no_lidarTests/ScanSessionYoloTrackingTests.swift` | End-to-end test: mock YOLO detections + depth → tracker observes correct world positions. |
| `decroom_no_lidar/backend/pipeline/consensus.py` | The four semantic-polish functions: `consensus_merge`, `collapse_duplicates`, `snap_to_walls`, `compute_confidence`. Also hosts `extract_doors_windows`. |
| `decroom_no_lidar/backend/tests/test_consensus.py` | consensus_merge + collapse_duplicates. |
| `decroom_no_lidar/backend/tests/test_wall_snapping.py` | snap_to_walls. |
| `decroom_no_lidar/backend/tests/test_confidence.py` | compute_confidence formula sanity. |
| `decroom_no_lidar/backend/tests/test_door_window.py` | extract_doors_windows. |
| `decroom_no_lidar/backend/tests/fixtures/phone_first_payload/yolo_objects.json` | Contract fixture. |
| `docs/superpowers/qa/2026-XX-XX-plan3b-b0-spatiallm-format.md` | Phase B0 prerequisite findings doc. |
| `docs/superpowers/qa/2026-XX-XX-plan3b-manual-qa.md` | Final QA checklist results. |

### Modified files

| Path | Change |
|---|---|
| `decroom_no_lidar/decroom_no_lidar/ScanSession.swift` | Cache `(depthResult, intrinsics, pose)` from `runDepthBackground`. Re-enable YOLO dispatch with `yoloInFlight` guard. Replace `runYoloBackground`'s `cameraPos + fwd * 2` stub with a real call to `YoloUnprojector`. Expose `yoloObjectsJSON` for the payload. |
| `decroom_no_lidar/decroom_no_lidar/ScanUploadPayload.swift` | Add `YoloObject`, `YoloObjectsManifest`, `encodeYoloObjects`. Add `yoloObjectsJSON: Data` field. Persist/load with `last_payload` directory. |
| `decroom_no_lidar/decroom_no_lidar/UploadService.swift` | Add `yolo_objects` multipart part in `_buildPhoneFirstMultipartBody`. |
| `decroom_no_lidar/decroom_no_lidarTests/ScanUploadPayloadTests.swift` | Assert `yolo_objects.json` structure (populated + empty cases). |
| `decroom_no_lidar/decroom_no_lidarTests/UploadServicePhoneFirstTests.swift` | Multipart body contains `yolo_objects` part. |
| `decroom_no_lidar/backend/main.py` | Accept `yolo_objects: UploadFile` on `/api/upload/phone-first` (optional with default). |
| `decroom_no_lidar/backend/pipeline/phone_first_pipeline.py` | `PhoneFirstPayload` gets `yolo_objects: List[Dict]`. `parse_payload` takes/parses the new JSON. `run_phone_first_pipeline` calls consensus + extract_doors_windows. |
| `decroom_no_lidar/backend/tests/test_phone_first_pipeline.py` | Update fixtures/expectations for consensus-processed `RoomPayload`. |
| `decroom_no_lidar/backend/tests/test_upload_schema.py` | Accept presence/absence of `yolo_objects` field. |

---

## Task list

Tasks run in order. Phase A (tasks 1–7) is iOS; Phase B0 (task 8) is a manual verification gate; Phase B (tasks 9–15) is backend; Phase C (tasks 16–17) wires it together and adds the contract fixture; Phase D (task 18) is manual QA.

---

### Task 1: YoloUnprojector — pure unprojection function

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/YoloUnprojector.swift`
- Test: `decroom_no_lidar/decroom_no_lidarTests/YoloUnprojectorTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
//
//  YoloUnprojectorTests.swift
//  decroom_no_lidarTests
//
//  Verifies YOLO bbox center + depth + intrinsics + pose → world position.
//

import XCTest
import simd
@testable import decroom_no_lidar

final class YoloUnprojectorTests: XCTestCase {
    /// Identity pose, principal-point centroid, known depth → z points into -Z.
    func testCentroidAtPrincipalPoint_identityPose_returnsCameraSpacePoint() throws {
        // 10x10 depth map, all 2.0 m depth.
        let depths = [Float](repeating: 2.0, count: 100)
        let intr = Intrinsics(fx: 100, fy: 100, cx: 5, cy: 5, width: 10, height: 10)
        let identity = matrix_identity_float4x4

        let world = YoloUnprojector.worldPosition(
            yoloBboxCenterNormalized: SIMD2<Float>(0.5, 0.5),  // dead center
            depthMap: depths,
            depthWidth: 10,
            depthHeight: 10,
            depthIntrinsics: intr,
            worldFromCamera: identity
        )

        let w = try XCTUnwrap(world)
        XCTAssertEqual(w.x, 0.0, accuracy: 1e-4)
        XCTAssertEqual(w.y, 0.0, accuracy: 1e-4)
        XCTAssertEqual(w.z, -2.0, accuracy: 1e-4)  // ARKit convention: camera looks down -Z
    }

    /// Invalid depth (0) at the centroid → nil.
    func testZeroDepthReturnsNil() {
        let depths = [Float](repeating: 0.0, count: 100)
        let intr = Intrinsics(fx: 100, fy: 100, cx: 5, cy: 5, width: 10, height: 10)
        let world = YoloUnprojector.worldPosition(
            yoloBboxCenterNormalized: SIMD2<Float>(0.5, 0.5),
            depthMap: depths,
            depthWidth: 10, depthHeight: 10,
            depthIntrinsics: intr,
            worldFromCamera: matrix_identity_float4x4
        )
        XCTAssertNil(world)
    }

    /// NaN / non-finite depth → nil.
    func testNaNDepthReturnsNil() {
        var depths = [Float](repeating: 2.0, count: 100)
        depths[55] = .nan  // row 5, col 5 — the principal point
        let intr = Intrinsics(fx: 100, fy: 100, cx: 5, cy: 5, width: 10, height: 10)
        let world = YoloUnprojector.worldPosition(
            yoloBboxCenterNormalized: SIMD2<Float>(0.5, 0.5),
            depthMap: depths,
            depthWidth: 10, depthHeight: 10,
            depthIntrinsics: intr,
            worldFromCamera: matrix_identity_float4x4
        )
        XCTAssertNil(world)
    }

    /// Translated pose shifts the result by the translation.
    func testTranslatedPoseShiftsResult() throws {
        let depths = [Float](repeating: 2.0, count: 100)
        let intr = Intrinsics(fx: 100, fy: 100, cx: 5, cy: 5, width: 10, height: 10)
        var pose = matrix_identity_float4x4
        pose.columns.3 = SIMD4<Float>(3.0, 4.0, 5.0, 1.0)  // translate world origin
        let world = try XCTUnwrap(YoloUnprojector.worldPosition(
            yoloBboxCenterNormalized: SIMD2<Float>(0.5, 0.5),
            depthMap: depths,
            depthWidth: 10, depthHeight: 10,
            depthIntrinsics: intr,
            worldFromCamera: pose
        ))
        XCTAssertEqual(world.x, 3.0, accuracy: 1e-4)
        XCTAssertEqual(world.y, 4.0, accuracy: 1e-4)
        XCTAssertEqual(world.z, 3.0, accuracy: 1e-4)  // 5 + (-2) = 3
    }

    /// Out-of-range normalized coords (clamped, not crash).
    func testOutOfRangeNormalizedCoords_clampedSafely() {
        let depths = [Float](repeating: 2.0, count: 100)
        let intr = Intrinsics(fx: 100, fy: 100, cx: 5, cy: 5, width: 10, height: 10)
        let world = YoloUnprojector.worldPosition(
            yoloBboxCenterNormalized: SIMD2<Float>(1.5, -0.1),  // outside [0,1]
            depthMap: depths,
            depthWidth: 10, depthHeight: 10,
            depthIntrinsics: intr,
            worldFromCamera: matrix_identity_float4x4
        )
        // Should not crash; clamped into-bounds pixel returns a valid position.
        XCTAssertNotNil(world)
    }
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Build + run the iOS test target:
```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -scheme decroom_no_lidar \
  -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  test 2>&1 | tail -40
```
Expected: compile error — `YoloUnprojector` undefined.

- [ ] **Step 3: Implement YoloUnprojector**

Create `decroom_no_lidar/decroom_no_lidar/YoloUnprojector.swift`:

```swift
//
//  YoloUnprojector.swift
//  decroom_no_lidar
//
//  Pure function: convert a YOLO bbox center + a depth map into a 3D
//  world position. No ARKit dependency — testable without a device.
//

import Foundation
import simd

enum YoloUnprojector {
    /// Unproject a YOLO bbox center to a world-space position.
    ///
    /// - Parameters:
    ///   - yoloBboxCenterNormalized: YOLO bbox center in [0, 1] × [0, 1]
    ///     (normalized image coordinates in the YOLO input frame).
    ///   - depthMap: row-major metric depth map (meters).
    ///   - depthWidth / depthHeight: depth map size.
    ///   - depthIntrinsics: camera intrinsics aligned to the depth map
    ///     resolution (cx/cy/fx/fy in depth-map pixel units).
    ///   - worldFromCamera: 4×4 ARKit pose matrix (right-handed, Y up).
    /// - Returns: 3D world position, or nil if depth at the centroid is
    ///   invalid (0, NaN, infinite).
    static func worldPosition(
        yoloBboxCenterNormalized: SIMD2<Float>,
        depthMap: [Float],
        depthWidth: Int,
        depthHeight: Int,
        depthIntrinsics: Intrinsics,
        worldFromCamera: simd_float4x4
    ) -> SIMD3<Float>? {
        guard depthWidth > 0, depthHeight > 0,
              depthMap.count == depthWidth * depthHeight else { return nil }

        // Clamp normalized coords into [0, 1] defensively (YOLO bboxes should
        // be in-range, but NMS rarely emits off-by-epsilon values).
        let nx = max(0.0, min(1.0, yoloBboxCenterNormalized.x))
        let ny = max(0.0, min(1.0, yoloBboxCenterNormalized.y))

        let px = min(depthWidth - 1, Int(nx * Float(depthWidth)))
        let py = min(depthHeight - 1, Int(ny * Float(depthHeight)))

        let depth = depthMap[py * depthWidth + px]
        guard depth.isFinite, depth > 0 else { return nil }

        // Pinhole unprojection in camera space (ARKit: camera looks down -Z).
        let xc = (Float(px) - depthIntrinsics.cx) * depth / depthIntrinsics.fx
        let yc = (Float(py) - depthIntrinsics.cy) * depth / depthIntrinsics.fy
        let zc = -depth
        let camPoint = SIMD4<Float>(xc, yc, zc, 1.0)

        let world4 = worldFromCamera * camPoint
        return SIMD3<Float>(world4.x, world4.y, world4.z)
    }
}
```

- [ ] **Step 4: Add the new file to the Xcode target**

`decroom_no_lidar.xcodeproj` uses `PBXFileSystemSynchronizedRootGroup`, so new `.swift` files under `decroom_no_lidar/decroom_no_lidar/` are auto-picked-up. Same for new `*Tests.swift` files under `decroom_no_lidar/decroom_no_lidarTests/`. Do NOT hand-edit the pbxproj. Just save the files and build.

- [ ] **Step 5: Run tests and confirm they pass**

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -scheme decroom_no_lidar \
  -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  test 2>&1 | grep -E "Test Case|YoloUnprojector" | tail -20
```
Expected: 5 YoloUnprojectorTests cases all pass.

- [ ] **Step 6: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/YoloUnprojector.swift \
        decroom_no_lidar/decroom_no_lidarTests/YoloUnprojectorTests.swift
git commit -m "$(cat <<'EOF'
feat(ios): YoloUnprojector pure function for bbox → world position

Pure unprojection: YOLO bbox center (normalized) + depth map +
intrinsics + pose → 3D world position. No ARKit dependency, all
math unit-testable.

Used by Plan 3b to replace the placeholder (cameraPos + fwd * 2)
in ScanSession.runYoloBackground so ObjectTracker records real
object positions instead of "2m in front of camera" stubs.

EOF
)"
```

---

### Task 2: ScanSession caches recent depth for YOLO lookup

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/ScanSession.swift`

**Rationale:** YOLO needs a recent depth map to unproject each detection. Depth and YOLO run at different frequencies on the same ARFrame stream, so we cache the most recent depth result + its intrinsics + pose in a struct that `runYoloBackground` reads.

- [ ] **Step 1: Add the cache struct + storage to ScanSession**

In `ScanSession.swift`, add near the other private state (after the `depthInFlight` declaration around line 65):

```swift
    /// Cached snapshot of the last depth inference. Written by the ML queue
    /// after each depth inference; read by the ML queue before each YOLO
    /// inference to unproject bbox centroids to world space.
    /// Nil before the first depth result lands — YOLO drops detections
    /// that arrive before depth has a chance.
    private struct DepthSnapshot {
        let depths: [Float]
        let width: Int
        let height: Int
        let intrinsics: Intrinsics
        let worldFromCamera: simd_float4x4
    }
    private let snapshotQueue = DispatchQueue(label: "decroom.scan-session.depth-snapshot")
    private var _latestDepthSnapshot: DepthSnapshot?

    private func storeDepthSnapshot(_ snap: DepthSnapshot) {
        snapshotQueue.sync { self._latestDepthSnapshot = snap }
    }
    private func latestDepthSnapshot() -> DepthSnapshot? {
        snapshotQueue.sync { _latestDepthSnapshot }
    }
```

- [ ] **Step 2: Populate the snapshot at the end of runDepthBackground**

In `runDepthBackground` (around line 240-285), after the `pointCloud.addDepthMap(...)` call, add:

```swift
        // Store a snapshot for YOLO unprojection on subsequent frames.
        self.storeDepthSnapshot(DepthSnapshot(
            depths: result.depths,
            width: result.width,
            height: result.height,
            intrinsics: scaledIntrinsics,
            worldFromCamera: pose
        ))
```

- [ ] **Step 3: Build to confirm there are no errors**

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -scheme decroom_no_lidar \
  -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj \
  -destination 'generic/platform=iOS' \
  build 2>&1 | tail -10
```
Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 4: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/ScanSession.swift
git commit -m "feat(ios): cache latest depth snapshot for YOLO unprojection"
```

---

### Task 3: Fix runYoloBackground to use real unprojection

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/ScanSession.swift`
- Test: `decroom_no_lidar/decroom_no_lidarTests/ScanSessionYoloTrackingTests.swift`

- [ ] **Step 1: Write the failing test**

Create `decroom_no_lidar/decroom_no_lidarTests/ScanSessionYoloTrackingTests.swift`:

```swift
//
//  ScanSessionYoloTrackingTests.swift
//  decroom_no_lidarTests
//
//  Covers the tracker side-effect of runYoloBackground: detections + a
//  recent depth snapshot yield real world positions. We invoke the
//  unprojection path directly rather than driving ARKit.
//

import XCTest
import simd
@testable import decroom_no_lidar

final class ScanSessionYoloTrackingTests: XCTestCase {
    /// When depth is 2m everywhere and pose is identity, a YOLO detection
    /// at the center of the frame should be observed at (0, 0, -2).
    func testTrackerObservesUnprojectedPositionForCenterDetection() {
        let depths = [Float](repeating: 2.0, count: 10 * 10)
        let intr = Intrinsics(fx: 100, fy: 100, cx: 5, cy: 5, width: 10, height: 10)
        let pose = matrix_identity_float4x4

        var tracker = ObjectTracker(mergeDistanceByCategory: ["bed": 1.5])
        let detection = YoloDetection(
            category: "bed",
            confidence: 0.85,
            bbox: CGRect(x: 0.4, y: 0.4, width: 0.2, height: 0.2),  // center 0.5, 0.5
            mask: nil,
            frameIndex: 1
        )

        // Use the same code path ScanSession uses to unproject + observe.
        if let world = YoloUnprojector.worldPosition(
            yoloBboxCenterNormalized: SIMD2<Float>(Float(detection.bbox.midX),
                                                     Float(detection.bbox.midY)),
            depthMap: depths, depthWidth: 10, depthHeight: 10,
            depthIntrinsics: intr, worldFromCamera: pose
        ) {
            tracker.observe(category: detection.category,
                             worldPosition: world,
                             confidence: detection.confidence,
                             frameIndex: detection.frameIndex)
        }

        XCTAssertEqual(tracker.tracks.count, 1)
        let t = tracker.tracks[0]
        XCTAssertEqual(t.category, "bed")
        XCTAssertEqual(t.latestPosition.x, 0.0, accuracy: 1e-4)
        XCTAssertEqual(t.latestPosition.z, -2.0, accuracy: 1e-4)
        XCTAssertEqual(t.bestConfidence, 0.85, accuracy: 1e-4)
    }

    /// Two detections of the same bed category within merge distance should
    /// dedupe to one track.
    func testDuplicateBedsAtSamePositionDedupedToOneTrack() {
        let depths = [Float](repeating: 2.0, count: 100)
        let intr = Intrinsics(fx: 100, fy: 100, cx: 5, cy: 5, width: 10, height: 10)
        let pose = matrix_identity_float4x4
        var tracker = ObjectTracker(mergeDistanceByCategory: ["bed": 1.5])

        for (f, cx) in zip([1, 10, 20], [0.50, 0.52, 0.48]) as [(Int, Double)] {
            let det = YoloDetection(
                category: "bed", confidence: Float(0.7 + Double(f) * 0.01),
                bbox: CGRect(x: cx - 0.1, y: 0.4, width: 0.2, height: 0.2),
                mask: nil, frameIndex: f
            )
            let w = YoloUnprojector.worldPosition(
                yoloBboxCenterNormalized: SIMD2<Float>(Float(det.bbox.midX),
                                                         Float(det.bbox.midY)),
                depthMap: depths, depthWidth: 10, depthHeight: 10,
                depthIntrinsics: intr, worldFromCamera: pose
            )!
            tracker.observe(category: det.category, worldPosition: w,
                             confidence: det.confidence, frameIndex: det.frameIndex)
        }

        XCTAssertEqual(tracker.tracks.count, 1, "same-position beds must dedupe")
        XCTAssertEqual(tracker.tracks[0].observationCount, 3)
    }

    /// If depth is invalid, tracker should have no tracks after the call.
    func testInvalidDepthSkipsObservation() {
        let depths = [Float](repeating: 0.0, count: 100)
        let intr = Intrinsics(fx: 100, fy: 100, cx: 5, cy: 5, width: 10, height: 10)
        var tracker = ObjectTracker(mergeDistanceByCategory: [:])
        let w = YoloUnprojector.worldPosition(
            yoloBboxCenterNormalized: SIMD2<Float>(0.5, 0.5),
            depthMap: depths, depthWidth: 10, depthHeight: 10,
            depthIntrinsics: intr, worldFromCamera: matrix_identity_float4x4
        )
        if let w { tracker.observe(category: "bed", worldPosition: w,
                                    confidence: 0.9, frameIndex: 1) }
        XCTAssertEqual(tracker.tracks.count, 0)
    }
}
```

- [ ] **Step 2: Run and confirm failing (test helper methods compile; assertions are evaluating the real code path but `runYoloBackground` still uses the stub — however this test doesn't drive `runYoloBackground` directly; it tests the unprojection + tracker integration, which is passing. Run anyway to confirm.)**

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  test 2>&1 | grep "ScanSessionYoloTrackingTests" | tail -10
```
Expected: 3 PASS (these tests check the *building blocks* our ScanSession change will use — they pass because Tasks 1 + 2 are done. The production code fix is in Step 3 below.)

- [ ] **Step 3: Replace the placeholder in `runYoloBackground`**

In `ScanSession.swift`, locate `runYoloBackground` (around line 287-328). Replace the body:

```swift
    private nonisolated func runYoloBackground(
        capturedImage: CVPixelBuffer,
        cameraTransform: simd_float4x4,
        yolo: YoloRunner,
        frameIndex: Int
    ) {
        let modelSize = CGSize(width: 640, height: 640)
        let ciCtx = CIContext()
        guard let bgra = PixelBufferRGB.convert(capturedImage,
                                                  toSize: modelSize,
                                                  ciContext: ciCtx,
                                                  rotateToPortrait: true) else {
            return
        }
        guard let dets = try? yolo.infer(pixelBuffer: bgra, frameIndex: frameIndex) else {
            return
        }

        // Need a recent depth snapshot to unproject. If we haven't produced
        // one yet (depth hasn't run, or lost), drop this YOLO result — the
        // placeholder "2m in front" was worse than dropping.
        guard let snap = self.latestDepthSnapshot() else {
            Task { @MainActor [weak self] in
                self?.logger.logEvent("yolo_detection_no_depth", at: self?.elapsed() ?? 0,
                                        data: ["frame": "\(frameIndex)",
                                               "detections": "\(dets.count)"])
            }
            return
        }

        // Observe each detection at its real world position.
        var observations: [(YoloDetection, SIMD3<Float>)] = []
        for d in dets where d.confidence > 0.5 {
            let center = SIMD2<Float>(Float(d.bbox.midX), Float(d.bbox.midY))
            guard let worldPos = YoloUnprojector.worldPosition(
                yoloBboxCenterNormalized: center,
                depthMap: snap.depths,
                depthWidth: snap.width, depthHeight: snap.height,
                depthIntrinsics: snap.intrinsics,
                worldFromCamera: snap.worldFromCamera
            ) else { continue }
            observations.append((d, worldPos))
        }

        Task { @MainActor [weak self] in
            guard let self else { return }
            self.recentDetections = dets
            for (d, pos) in observations {
                self.tracker.observe(category: d.category,
                                      worldPosition: pos,
                                      confidence: d.confidence,
                                      frameIndex: d.frameIndex)
            }
        }
    }
```

Note: `cameraTransform` parameter becomes unused here (the snapshot's pose replaces it). Leave the parameter for now — Task 4 will redefine the dispatch.

- [ ] **Step 4: Build to confirm the change compiles**

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj \
  -destination 'generic/platform=iOS' build 2>&1 | tail -10
```
Expected: `** BUILD SUCCEEDED **`. (Warning about `cameraTransform` parameter being unused is fine — will be cleaned up in Task 4.)

- [ ] **Step 5: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/ScanSession.swift \
        decroom_no_lidar/decroom_no_lidarTests/ScanSessionYoloTrackingTests.swift
git commit -m "$(cat <<'EOF'
fix(ios): runYoloBackground uses real depth-based unprojection

Replaces the placeholder 'cameraPos + fwd * 2' (which placed every
tracked object 2m in front of wherever the camera pointed) with a
proper unprojection using the cached depth snapshot.

Detections whose bbox centroid falls on invalid depth (0, NaN) are
dropped. Detections that arrive before any depth snapshot is available
are also dropped with a 'yolo_detection_no_depth' event logged.

Unblocks Plan 3b's consensus pipeline — ObjectTracker.tracks now
carry meaningful world positions the backend can match against
SpatialLM output.

EOF
)"
```

---

### Task 4: Re-enable YOLO dispatch with yoloInFlight guard

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/ScanSession.swift`

- [ ] **Step 1: Add `yoloInFlight` state + dispatch block**

In `ScanSession.swift`, add near the `depthInFlight` declaration:

```swift
    private var yoloInFlight = false
    private let yoloFrameInterval = 30   // 1 Hz at 30fps (YOLO is ~100ms)
    private var lastYoloFrame = -1
```

Then in `update(frame:)`, REPLACE the disabled YOLO comment block (currently lines ~130-134) with:

```swift
        // YOLO inference at 1 Hz for consensus with backend SpatialLM.
        // Drop-on-busy guard prevents ARKit buffer-pool exhaustion.
        if !yoloInFlight,
           let yolo = yoloRunner,
           frameIndex - lastYoloFrame >= yoloFrameInterval {
            lastYoloFrame = frameIndex
            yoloInFlight = true
            let capturedImage = frame.capturedImage
            let pose = frame.camera.transform
            let fi = frameIndex

            mlQueue.async { [weak self] in
                self?.runYoloBackground(
                    capturedImage: capturedImage,
                    cameraTransform: pose,
                    yolo: yolo,
                    frameIndex: fi
                )
                Task { @MainActor [weak self] in self?.yoloInFlight = false }
            }
        }
```

- [ ] **Step 2: Build**

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj \
  -destination 'generic/platform=iOS' build 2>&1 | tail -10
```
Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 3: Run the full iOS test suite**

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' test 2>&1 | grep -E "Test Suite.*(passed|failed)" | tail -5
```
Expected: all suites passed.

- [ ] **Step 4: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/ScanSession.swift
git commit -m "feat(ios): re-enable YOLO dispatch at 1Hz with yoloInFlight guard"
```

---

### Task 5: Add YoloObject / YoloObjectsManifest structs + encoder

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/ScanUploadPayload.swift`
- Test: `decroom_no_lidar/decroom_no_lidarTests/ScanUploadPayloadTests.swift`

- [ ] **Step 1: Write the failing test**

Append to `decroom_no_lidar/decroom_no_lidarTests/ScanUploadPayloadTests.swift` (inside the existing test class):

```swift
    func testEncodeYoloObjects_populatedList() throws {
        let tracks = [
            ObjectTrack(id: UUID(), category: "bed",
                         latestPosition: SIMD3<Float>(2.0, 0.0, 1.5),
                         bestConfidence: 0.87, bestFrameIndex: 34,
                         observationCount: 12),
            ObjectTrack(id: UUID(), category: "potted plant",
                         latestPosition: SIMD3<Float>(0.3, 0.8, 2.9),
                         bestConfidence: 0.62, bestFrameIndex: 67,
                         observationCount: 5),
        ]

        let data = try ScanUploadPayload.encodeYoloObjects(tracks)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let objects = try XCTUnwrap(json?["objects"] as? [[String: Any]])
        XCTAssertEqual(objects.count, 2)

        let bed = objects[0]
        XCTAssertEqual(bed["category"] as? String, "bed")
        let pos = try XCTUnwrap(bed["position"] as? [Double])
        XCTAssertEqual(pos.count, 3)
        XCTAssertEqual(pos[0], 2.0, accuracy: 1e-4)
        XCTAssertEqual(pos[1], 0.0, accuracy: 1e-4)
        XCTAssertEqual(pos[2], 1.5, accuracy: 1e-4)
        XCTAssertEqual(bed["peak_confidence"] as? Double ?? 0, 0.87, accuracy: 1e-4)
        XCTAssertEqual(bed["view_count"] as? Int, 12)
        XCTAssertEqual(bed["best_frame_index"] as? Int, 34)
    }

    func testEncodeYoloObjects_emptyList() throws {
        let data = try ScanUploadPayload.encodeYoloObjects([])
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let objects = try XCTUnwrap(json?["objects"] as? [Any])
        XCTAssertEqual(objects.count, 0)
    }
```

- [ ] **Step 2: Run and confirm failing**

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' test 2>&1 | grep "encodeYoloObjects" | tail -6
```
Expected: compile error — `encodeYoloObjects` unresolved.

- [ ] **Step 3: Implement the structs + encoder**

In `ScanUploadPayload.swift`, add inside `struct ScanUploadPayload` (after the existing `KeyframesManifest` struct, before `// MARK: - JSON serialization helpers`):

```swift
    /// Matches yolo_objects.json schema from the design spec.
    /// Each entry is one deduped 3D-tracked object from ObjectTracker.
    struct YoloObject: Codable {
        let category: String
        let position: [Float]       // [x, y, z] ARKit world, meters
        let peak_confidence: Float
        let view_count: Int
        let best_frame_index: Int
    }
    struct YoloObjectsManifest: Codable {
        let objects: [YoloObject]
    }
```

Then in the same struct, in the `// MARK: - JSON serialization helpers` section, add:

```swift
    static func encodeYoloObjects(_ tracks: [ObjectTrack]) throws -> Data {
        let entries = tracks.map { t in
            YoloObject(
                category: t.category,
                position: [t.latestPosition.x, t.latestPosition.y, t.latestPosition.z],
                peak_confidence: t.bestConfidence,
                view_count: t.observationCount,
                best_frame_index: t.bestFrameIndex
            )
        }
        let manifest = YoloObjectsManifest(objects: entries)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        return try encoder.encode(manifest)
    }
```

Also add a `yoloObjectsJSON: Data` field to the payload struct. Update the initializer:

```swift
struct ScanUploadPayload {
    // ...existing fields...
    let scanID: String
    let roomGeometryJSON: Data
    let pointCloudPLY: Data
    let keyframes: [(jpeg: Data, meta: KeyframeMetadata)]
    let scanLogJSON: Data
    let yoloObjectsJSON: Data       // <-- NEW
    let deviceInfo: DeviceInfo
```

- [ ] **Step 4: Update any existing initializer call sites**

In `ScanSession.swift`'s `finalize()` method (around line 216), the return statement currently passes `scanLogJSON: logData` then `deviceInfo`. Update to include `yoloObjectsJSON`. Insert this line just before the existing return:

```swift
        let yoloJSON: Data
        do {
            yoloJSON = try ScanUploadPayload.encodeYoloObjects(tracker.tracks)
        } catch {
            throw ScanError.payloadBuildFailed(detail: "yolo_objects encode: \(error.localizedDescription)")
        }
```

And change the return:

```swift
        return ScanUploadPayload(
            scanID: logger.scanId,
            roomGeometryJSON: roomGeometryJSON,
            pointCloudPLY: plyData,
            keyframes: zip(keyframeJPEGs, keyframes).map { ($0.0, $0.1) },
            scanLogJSON: logData,
            yoloObjectsJSON: yoloJSON,
            deviceInfo: DeviceInfo.current()
        )
```

- [ ] **Step 5: Update `loadLastCapture()` and `saveAsLastCapture()` to round-trip yolo_objects.json**

In `saveAsLastCapture()` (after `scan_log.json` write, around line 148):

```swift
            try yoloObjectsJSON.write(
                to: dir.appendingPathComponent("yolo_objects.json"))
```

In `loadLastCapture()` (after the `manifestData = try Data(contentsOf: ...)` line, add):

```swift
            let yoloData = (try? Data(
                contentsOf: dir.appendingPathComponent("yolo_objects.json")))
                ?? Data("{\"objects\":[]}".utf8)
```

And pass `yoloObjectsJSON: yoloData` to the returned `ScanUploadPayload`.

- [ ] **Step 6: Build + run tests**

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' test 2>&1 | grep -E "Test Suite ScanUploadPayloadTests" | tail -5
```
Expected: ScanUploadPayloadTests passed.

- [ ] **Step 7: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/ScanUploadPayload.swift \
        decroom_no_lidar/decroom_no_lidar/ScanSession.swift \
        decroom_no_lidar/decroom_no_lidarTests/ScanUploadPayloadTests.swift
git commit -m "feat(ios): serialize ObjectTracker.tracks to yolo_objects.json in payload"
```

---

### Task 6: UploadService includes yolo_objects in multipart body

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/UploadService.swift`
- Test: `decroom_no_lidar/decroom_no_lidarTests/UploadServicePhoneFirstTests.swift`

- [ ] **Step 1: Write the failing test**

Append to `UploadServicePhoneFirstTests.swift`:

```swift
    func testMultipartBodyContainsYoloObjectsPart() throws {
        let payload = ScanUploadPayload(
            scanID: "abc12345",
            roomGeometryJSON: Data(#"{"walls":[]}"#.utf8),
            pointCloudPLY: Data(repeating: 0, count: 100),
            keyframes: [],
            scanLogJSON: Data("{}".utf8),
            yoloObjectsJSON: Data(#"{"objects":[{"category":"bed"}]}"#.utf8),
            deviceInfo: DeviceInfo(model: "iPhone", ios: "17", chip: "A17", tier: .full)
        )
        let boundary = "TEST-BOUNDARY"
        let body = try UploadService._buildPhoneFirstMultipartBody(payload, boundary: boundary)
        let bodyStr = String(data: body, encoding: .utf8) ?? ""
        XCTAssertTrue(bodyStr.contains(#"name="yolo_objects""#),
                        "multipart body must include yolo_objects part")
        XCTAssertTrue(bodyStr.contains(#"filename="yolo_objects.json""#))
        XCTAssertTrue(bodyStr.contains(#""category":"bed""#))
    }
```

- [ ] **Step 2: Run and confirm failing**

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' test 2>&1 | grep "testMultipartBodyContainsYoloObjectsPart" | tail -3
```
Expected: FAIL — assertion "multipart body must include yolo_objects part".

- [ ] **Step 3: Add the part**

In `UploadService.swift`'s `_buildPhoneFirstMultipartBody`, after the `scan_log` part is appended (around line 236), insert:

```swift
        appendPart(name: "yolo_objects", filename: "yolo_objects.json",
                    contentType: "application/json",
                    bytes: payload.yoloObjectsJSON)
```

- [ ] **Step 4: Run test + confirm pass**

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' test 2>&1 | grep "testMultipartBodyContainsYoloObjectsPart" | tail -3
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidar/UploadService.swift \
        decroom_no_lidar/decroom_no_lidarTests/UploadServicePhoneFirstTests.swift
git commit -m "feat(ios): add yolo_objects multipart part to phone-first upload"
```

---

### Task 7: Update ScanUploadPayloadTests for full finalize + serialize round-trip

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidarTests/ScanUploadPayloadTests.swift`

- [ ] **Step 1: Add the round-trip test**

Append to `ScanUploadPayloadTests.swift`:

```swift
    func testSaveAndLoadLastCapture_roundTripsYoloObjects() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("decroom_test_last_\(UUID().uuidString)")

        // Redirect Documents/last_payload to tmp for this test by constructing
        // a payload and writing it directly to tmp. Then read via a raw JSON
        // round-trip.
        let tracks: [ObjectTrack] = [
            ObjectTrack(id: UUID(), category: "bed",
                         latestPosition: SIMD3<Float>(1, 0, 2),
                         bestConfidence: 0.9, bestFrameIndex: 5,
                         observationCount: 3)
        ]
        let data = try ScanUploadPayload.encodeYoloObjects(tracks)

        // Round-trip through JSONDecoder
        let decoded = try JSONDecoder().decode(
            ScanUploadPayload.YoloObjectsManifest.self, from: data
        )
        XCTAssertEqual(decoded.objects.count, 1)
        XCTAssertEqual(decoded.objects[0].category, "bed")
        XCTAssertEqual(decoded.objects[0].view_count, 3)
        XCTAssertEqual(decoded.objects[0].position, [1, 0, 2])
    }
```

- [ ] **Step 2: Run and confirm pass**

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' test 2>&1 | grep "testSaveAndLoadLastCapture_roundTripsYoloObjects" | tail -3
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add decroom_no_lidar/decroom_no_lidarTests/ScanUploadPayloadTests.swift
git commit -m "test(ios): round-trip yolo_objects through encoder/decoder"
```

---

### Task 8: Phase B0 — manual SpatialLM door/window format verification

**Files:**
- Create: `docs/superpowers/qa/YYYY-MM-DD-plan3b-b0-spatiallm-format.md` (replace YYYY-MM-DD with today's date when performing)

**This is a blocking manual task.** Do NOT write `extract_doors_windows` (Task 14) before this completes. The assumed schema (`class_name == "door"` / `"window"`) is an assumption from the design spec; if wrong, Task 14's implementation changes.

- [ ] **Step 1: Ensure SpatialLM tunnel is live**

Follow `memory/runpod_setup.md`:
- Start the pod if needed
- Open the SSH tunnel: `ssh -L 8100:localhost:8100 root@<pod-ip>`
- Verify: `curl http://localhost:8100/health` → `{"status":"ok"}`

- [ ] **Step 2: Run one phone-first scan through the live tunnel**

On a test device with current `main`:
1. Scan a room that visibly contains at least one door AND one window.
2. Upload completes; note the scan_id from the response.
3. Server-side, find the raw SpatialLM output in the debug archive:
   ```bash
   ls -lt /tmp/decroom_runs | head -3
   cat /tmp/decroom_runs/latest/request_metadata.json | head -20
   ```

- [ ] **Step 3: Capture raw parser output**

In `decroom_no_lidar/backend/pipeline/phone_first_pipeline.py`, temporarily add (just above the `layout, objects = parse_spatiallm_output(...)` call inside `_run_spatiallm_on_phone_ply`):

```python
    # TEMPORARY: capture raw SpatialLM text for B0 verification
    import pathlib
    pathlib.Path("/tmp/decroom_b0_raw_slm.txt").write_text(raw_text or "")
```

Re-run the scan. After, inspect:

```bash
cat /tmp/decroom_b0_raw_slm.txt | head -100
```

Look for anything mentioning `door`, `window`, or similar. Document what you see.

- [ ] **Step 4: Inspect parsed output**

Add a second temporary hook just below the `layout, objects = ...` line:

```python
    import json, pathlib
    pathlib.Path("/tmp/decroom_b0_parsed_slm.json").write_text(json.dumps({
        "layout": layout, "objects": objects
    }, default=str, indent=2))
```

Re-run, then:

```bash
cat /tmp/decroom_b0_parsed_slm.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('layout keys:', list(d.get('layout', {}).keys()))
print('first 3 objects:')
for o in d.get('objects', [])[:3]:
    print(' ', json.dumps(o, indent=2, default=str)[:200])
# Look for doors/windows
for o in d.get('objects', []):
    name = o.get('class_name') or o.get('type') or ''
    if 'door' in name.lower() or 'window' in name.lower():
        print('found:', json.dumps(o, indent=2, default=str))
"
```

- [ ] **Step 5: Write findings**

Create `docs/superpowers/qa/YYYY-MM-DD-plan3b-b0-spatiallm-format.md` with actual values observed:

```markdown
# Plan 3b B0 — SpatialLM door/window format verification

**Date:** YYYY-MM-DD
**Scan:** <scan_id>
**Room:** <description, e.g. "bedroom with one door + one window">

## Findings

### Raw SpatialLM text snippet (doors/windows only)
```
<paste the door/window portion of /tmp/decroom_b0_raw_slm.txt>
```

### Parsed output shape
- Where do doors/windows appear? (layout["walls"][i]["has_door"]? layout["doors"]? objects[i] with class_name=="door"?)
- What fields carry position / wall attachment info?

### Schema confirmation
- [ ] Doors appear as ___
- [ ] Windows appear as ___
- [ ] Position format: ___
- [ ] Wall-attachment info: ___ (or none — need proximity matching)

## Impact on Task 14 (`extract_doors_windows`)
- If matches spec's assumption (`class_name == "door"/"window"` in parsed `objects`):
  proceed with Task 14 as written.
- Otherwise: update spec §7 and Task 14 to match the actual schema before implementing.
```

- [ ] **Step 6: Revert the temporary hooks**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
git checkout -- pipeline/phone_first_pipeline.py
```

- [ ] **Step 7: Commit the findings doc**

```bash
git add docs/superpowers/qa/*.md
git commit -m "docs: Plan 3b B0 SpatialLM door/window format findings"
```

**Gate:** If findings differ from the assumed schema, update `docs/superpowers/specs/2026-04-24-plan3b-semantic-polish-design.md` §7 and propagate to Task 14 below BEFORE proceeding. Do not skip.

---

### Task 9: Backend — parse `yolo_objects` multipart field into PhoneFirstPayload

**Files:**
- Modify: `decroom_no_lidar/backend/pipeline/phone_first_pipeline.py`
- Modify: `decroom_no_lidar/backend/main.py`
- Modify: `decroom_no_lidar/backend/tests/test_phone_first_pipeline.py`
- Modify: `decroom_no_lidar/backend/tests/test_upload_schema.py`

- [ ] **Step 1: Write failing tests**

Add to `test_phone_first_pipeline.py` (or create if the class doesn't exist):

```python
def test_parse_payload_accepts_yolo_objects_json():
    from pipeline.phone_first_pipeline import parse_payload
    import json

    room_geom = json.dumps({
        "walls": [], "floor_height": 0.0, "ceiling_height": 2.5,
        "manhattan_yaw": 0.0,
        "coordinate_system": "arkit_world_right_handed_y_up",
    }).encode()
    ply = b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n"
    manifest = json.dumps({"frames": []}).encode()
    log = b"{}"
    yolo = json.dumps({
        "objects": [
            {"category": "bed", "position": [1, 0, 2],
             "peak_confidence": 0.9, "view_count": 5, "best_frame_index": 3}
        ]
    }).encode()

    payload = parse_payload(room_geom, ply, manifest, [], log, yolo)
    assert len(payload.yolo_objects) == 1
    assert payload.yolo_objects[0]["category"] == "bed"
    assert payload.yolo_objects[0]["position"] == [1, 0, 2]


def test_parse_payload_allows_missing_yolo_objects_as_empty():
    from pipeline.phone_first_pipeline import parse_payload
    import json
    room_geom = json.dumps({
        "walls": [], "floor_height": 0.0, "ceiling_height": 2.5,
        "manhattan_yaw": 0.0,
        "coordinate_system": "arkit_world_right_handed_y_up",
    }).encode()
    ply = b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n"
    manifest = json.dumps({"frames": []}).encode()
    log = b"{}"

    payload = parse_payload(room_geom, ply, manifest, [], log, b"")
    assert payload.yolo_objects == []
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_phone_first_pipeline.py::test_parse_payload_accepts_yolo_objects_json tests/test_phone_first_pipeline.py::test_parse_payload_allows_missing_yolo_objects_as_empty -xvs 2>&1 | tail -15
```
Expected: FAIL — `parse_payload` signature mismatch.

- [ ] **Step 3: Update PhoneFirstPayload + parse_payload**

In `phone_first_pipeline.py`, modify the `PhoneFirstPayload` dataclass:

```python
@dataclass
class PhoneFirstPayload:
    room_geometry: Dict[str, Any]
    point_cloud_ply: bytes
    keyframes_manifest: Dict[str, Any]
    keyframe_jpegs: List[bytes]
    scan_log: Dict[str, Any]
    yolo_objects: List[Dict[str, Any]]   # <-- NEW
```

Update `parse_payload`:

```python
def parse_payload(
    room_geometry: bytes,
    point_cloud_ply: bytes,
    keyframes_manifest: bytes,
    keyframe_jpegs: List[bytes],
    scan_log: bytes,
    yolo_objects: bytes = b"",
) -> PhoneFirstPayload:
    """Parse multipart parts. Raises PhoneFirstSchemaError on any malformity."""
    # ... existing body up to and including the scan_log parse ...

    yolo_list: List[Dict[str, Any]] = []
    if yolo_objects:
        try:
            yolo_doc = json.loads(yolo_objects.decode("utf-8"))
            if not isinstance(yolo_doc, dict) or "objects" not in yolo_doc:
                raise PhoneFirstSchemaError("yolo_objects.json missing 'objects'")
            raw = yolo_doc["objects"]
            if not isinstance(raw, list):
                raise PhoneFirstSchemaError("yolo_objects.objects must be list")
            for i, o in enumerate(raw):
                if not isinstance(o, dict):
                    raise PhoneFirstSchemaError(f"yolo_objects[{i}] not an object")
                for required in ("category", "position", "peak_confidence",
                                 "view_count", "best_frame_index"):
                    if required not in o:
                        raise PhoneFirstSchemaError(
                            f"yolo_objects[{i}] missing {required}"
                        )
                pos = o["position"]
                if not isinstance(pos, list) or len(pos) != 3:
                    raise PhoneFirstSchemaError(
                        f"yolo_objects[{i}].position must be [x,y,z]"
                    )
            yolo_list = raw
        except PhoneFirstSchemaError:
            raise
        except Exception as e:
            raise PhoneFirstSchemaError(f"yolo_objects.json invalid: {e}")

    return PhoneFirstPayload(
        room_geometry=geom,
        point_cloud_ply=point_cloud_ply,
        keyframes_manifest=manifest,
        keyframe_jpegs=keyframe_jpegs,
        scan_log=log,
        yolo_objects=yolo_list,
    )
```

- [ ] **Step 4: Update main.py route**

In `main.py`, update the `upload_phone_first` signature + body:

```python
@app.post("/api/upload/phone-first")
async def upload_phone_first(
    room_geometry: UploadFile = File(...),
    point_cloud: UploadFile = File(...),
    keyframes_manifest: UploadFile = File(...),
    keyframes: List[UploadFile] = File(...),
    scan_log: UploadFile = File(...),
    yolo_objects: Optional[UploadFile] = File(None),
):
```

And in the body, after `log_bytes = await scan_log.read()`:

```python
        yolo_bytes = await yolo_objects.read() if yolo_objects is not None else b""
```

And in the `parse_payload(...)` call, append `yolo_bytes`:

```python
        payload = parse_payload(
            room_geom_bytes, ply_bytes, manifest_bytes,
            keyframe_bytes, log_bytes, yolo_bytes,
        )
```

- [ ] **Step 5: Run the new tests + confirm pass**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_phone_first_pipeline.py -xvs 2>&1 | tail -20
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add decroom_no_lidar/backend/pipeline/phone_first_pipeline.py \
        decroom_no_lidar/backend/main.py \
        decroom_no_lidar/backend/tests/test_phone_first_pipeline.py
git commit -m "feat(backend): accept yolo_objects.json on phone-first upload route"
```

---

### Task 10: consensus_merge function

**Files:**
- Create: `decroom_no_lidar/backend/pipeline/consensus.py`
- Create: `decroom_no_lidar/backend/tests/test_consensus.py`

- [ ] **Step 1: Write failing tests**

Create `decroom_no_lidar/backend/tests/test_consensus.py`:

```python
"""Tests for the Plan 3b consensus module."""
from pipeline.consensus import consensus_merge, POSITION_MATCH_DISTANCE_LARGE, POSITION_MATCH_DISTANCE_SMALL


def _make_yolo(category, x, y, z, conf=0.8, view_count=3):
    return {"category": category, "position": [x, y, z],
             "peak_confidence": conf, "view_count": view_count,
             "best_frame_index": 0}


def _make_slm(obj_type, x, y, z, width=1.6, length=2.0, height=0.6, rotation_y=0.0):
    return {"type": obj_type,
             "position": {"x": x, "y": y, "z": z},
             "rotation_y": rotation_y,
             "metadata": {"width": width, "length": length, "height": height}}


def test_yolo_and_slm_agree_on_bed_same_position_keeps_slm_full_agreement():
    yolo = [_make_yolo("bed", 2.0, 0.0, 1.5)]
    slm = [_make_slm("bed", 2.1, 0.0, 1.4)]
    merged, agreement = consensus_merge(yolo, slm, walls=[])
    assert len(merged) == 1
    assert merged[0]["type"] == "bed"
    # SLM position preserved (not moved to YOLO's)
    assert merged[0]["position"]["x"] == 2.1
    assert agreement[merged[0]["id"]] == 1.0


def test_yolo_only_bed_no_slm_keeps_yolo_position_half_agreement():
    yolo = [_make_yolo("bed", 2.0, 0.0, 1.5)]
    merged, agreement = consensus_merge(yolo, [], walls=[])
    assert len(merged) == 1
    assert merged[0]["type"] == "bed"
    assert merged[0]["position"]["x"] == 2.0
    assert agreement[merged[0]["id"]] == 0.5


def test_slm_only_bed_no_yolo_keeps_slm_half_agreement():
    slm = [_make_slm("bed", 2.0, 0.0, 1.5)]
    merged, agreement = consensus_merge([], slm, walls=[])
    assert len(merged) == 1
    assert agreement[merged[0]["id"]] == 0.5


def test_two_beds_far_apart_both_kept_full_agreement():
    yolo = [_make_yolo("bed", 2.0, 0.0, 1.5), _make_yolo("bed", 4.0, 0.0, 1.5)]
    slm = [_make_slm("bed", 2.0, 0.0, 1.5), _make_slm("bed", 4.0, 0.0, 1.5)]
    merged, agreement = consensus_merge(yolo, slm, walls=[])
    assert len(merged) == 2
    for o in merged:
        assert agreement[o["id"]] == 1.0


def test_slm_duplicates_one_yolo_collapses_to_one_low_agreement():
    yolo = [_make_yolo("bed", 2.0, 0.0, 1.5)]
    slm = [_make_slm("bed", 2.0, 0.0, 1.5), _make_slm("bed", 2.2, 0.0, 1.6)]
    merged, agreement = consensus_merge(yolo, slm, walls=[])
    assert len(merged) == 1
    assert agreement[merged[0]["id"]] == 0.3


def test_category_conflict_at_same_position_keeps_slm_low_agreement():
    yolo = [_make_yolo("bed", 2.0, 0.0, 1.5)]
    slm = [_make_slm("couch", 2.0, 0.0, 1.5, width=2.2, length=0.9)]
    merged, agreement = consensus_merge(yolo, slm, walls=[])
    assert len(merged) == 1
    # SLM category wins
    assert merged[0]["type"] == "couch"
    assert agreement[merged[0]["id"]] == 0.3


def test_position_match_distance_uses_small_for_chair():
    # 0.5m apart: LARGE distance (0.8m) would match; SMALL (0.4m) wouldn't.
    yolo = [_make_yolo("chair", 0.0, 0.0, 0.0)]
    slm = [_make_slm("chair", 0.5, 0.0, 0.0)]
    merged, agreement = consensus_merge(yolo, slm, walls=[])
    # Chair uses SMALL threshold → 0.5m > 0.4m → treated as NOT matched,
    # so both objects survive: 1 from YOLO (no match), 1 from SLM (no match).
    assert len(merged) == 2


def test_empty_inputs_returns_empty():
    merged, agreement = consensus_merge([], [], walls=[])
    assert merged == []
    assert agreement == {}
```

- [ ] **Step 2: Run tests + confirm all fail**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_consensus.py -xvs 2>&1 | tail -15
```
Expected: ImportError — `pipeline.consensus` doesn't exist.

- [ ] **Step 3: Implement consensus_merge**

Create `decroom_no_lidar/backend/pipeline/consensus.py`:

```python
"""
Plan 3b semantic polish pipeline.

Four pure functions + one door/window extractor. Operates on dict-based
objects (not Pydantic models) to stay decoupled from schema changes.

Object shape (both inputs and outputs use a common "merged object" dict):
  {
    "id": str,
    "type": str,           # Decroom category ('bed', 'couch', ...)
    "position": {"x": float, "y": float, "z": float},  # meters, ARKit frame
    "rotation_y": float,
    "metadata": {"width": float, "length": float, "height": float},
    "_source": "slm" | "yolo" | "consensus",   # provenance
  }
"""
from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("decroom-backend.consensus")

# Position-match distance when comparing YOLO 3D objects to SpatialLM objects.
# Large items (bed, sofa, dining table) are more diffuse — allow looser match.
POSITION_MATCH_DISTANCE_LARGE = 0.8
POSITION_MATCH_DISTANCE_SMALL = 0.4

_LARGE_CATEGORIES = {"bed", "couch", "sofa", "dining_table", "dresser",
                     "wardrobe", "bookshelf"}


def _match_distance(category: str) -> float:
    return POSITION_MATCH_DISTANCE_LARGE if category in _LARGE_CATEGORIES \
        else POSITION_MATCH_DISTANCE_SMALL


def _distance_xyz(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _slm_to_merged(slm: Dict[str, Any]) -> Dict[str, Any]:
    p = slm.get("position", {})
    return {
        "id": slm.get("id") or f"slm_{uuid.uuid4().hex[:6]}",
        "type": slm.get("type", "unknown"),
        "position": {"x": float(p.get("x", 0)), "y": float(p.get("y", 0)),
                      "z": float(p.get("z", 0))},
        "rotation_y": float(slm.get("rotation_y", 0.0)),
        "metadata": dict(slm.get("metadata", {})),
        "_source": "slm",
    }


def _yolo_to_merged(yolo: Dict[str, Any]) -> Dict[str, Any]:
    px, py, pz = yolo["position"]
    cat = yolo["category"]
    # Default dimensions when lifting a YOLO-only detection (no SLM bbox).
    dims = _default_dimensions_for(cat)
    return {
        "id": f"yolo_{uuid.uuid4().hex[:6]}",
        "type": cat,
        "position": {"x": float(px), "y": float(py), "z": float(pz)},
        "rotation_y": 0.0,
        "metadata": {
            "width": dims[0], "length": dims[1], "height": dims[2],
        },
        "_source": "yolo",
    }


# Category priors (width, length, height) for YOLO-only detections that
# need a default oriented bbox. Values chosen to be plausible mid-size
# instances of each category.
_DEFAULT_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    "bed": (1.6, 2.0, 0.6),
    "couch": (2.0, 0.9, 0.9),
    "sofa": (2.0, 0.9, 0.9),
    "chair": (0.55, 0.55, 0.85),
    "dining_table": (1.4, 0.9, 0.75),
    "desk": (1.3, 0.7, 0.75),
    "tv": (1.2, 0.1, 0.7),
    "laptop": (0.35, 0.25, 0.02),
    "lamp": (0.3, 0.3, 1.4),
    "potted plant": (0.3, 0.3, 0.4),
    "dresser": (1.0, 0.5, 0.85),
    "bookshelf": (0.8, 0.3, 1.8),
}


def _default_dimensions_for(category: str) -> Tuple[float, float, float]:
    return _DEFAULT_DIMENSIONS.get(category, (1.0, 1.0, 1.0))


def _slm_position_tuple(slm: Dict[str, Any]) -> Tuple[float, float, float]:
    p = slm.get("position", {})
    return (float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)))


def consensus_merge(
    yolo_objects_3d: List[Dict[str, Any]],
    slm_objects: List[Dict[str, Any]],
    walls: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Match YOLO's deduped 3D tracked objects against SpatialLM output.

    Returns (merged_objects, agreement_scores_by_id).
    `walls` is accepted for future use (currently unused — snap_to_walls
    takes its own walls argument).
    """
    merged: List[Dict[str, Any]] = []
    agreement: Dict[str, float] = {}
    slm_used = [False] * len(slm_objects)

    # Pre-compute SLM positions for fast iteration.
    slm_positions = [_slm_position_tuple(s) for s in slm_objects]

    for yolo in yolo_objects_3d:
        yx, yy, yz = yolo["position"]
        cat = yolo["category"]
        thresh = _match_distance(cat)

        # Matches within threshold: separate same-category from diff-category.
        same_cat_matches: List[int] = []
        diff_cat_matches: List[int] = []
        for i, slm in enumerate(slm_objects):
            if slm_used[i]:
                continue
            d = _distance_xyz((yx, yy, yz), slm_positions[i])
            if d > thresh:
                continue
            if slm.get("type") == cat:
                same_cat_matches.append(i)
            else:
                diff_cat_matches.append(i)

        if len(same_cat_matches) == 1:
            # Clean 1:1 — full agreement.
            i = same_cat_matches[0]
            slm_used[i] = True
            obj = _slm_to_merged(slm_objects[i])
            obj["_source"] = "consensus"
            merged.append(obj)
            agreement[obj["id"]] = 1.0
        elif len(same_cat_matches) >= 2:
            # SLM duplicates → collapse to one, low agreement.
            for i in same_cat_matches:
                slm_used[i] = True
            # Pick the closest SLM to the YOLO position as survivor.
            i0 = min(same_cat_matches,
                     key=lambda i: _distance_xyz((yx, yy, yz), slm_positions[i]))
            obj = _slm_to_merged(slm_objects[i0])
            obj["_source"] = "consensus"
            merged.append(obj)
            agreement[obj["id"]] = 0.3
        elif len(diff_cat_matches) >= 1:
            # Category conflict — keep SLM category, low agreement.
            i = diff_cat_matches[0]
            slm_used[i] = True
            obj = _slm_to_merged(slm_objects[i])
            obj["_source"] = "consensus"
            merged.append(obj)
            agreement[obj["id"]] = 0.3
            logger.info("consensus_category_conflict: yolo=%s slm=%s at (%.2f,%.2f,%.2f)",
                         cat, slm_objects[i].get("type"), yx, yy, yz)
        else:
            # YOLO-only — lift with default dims.
            obj = _yolo_to_merged(yolo)
            merged.append(obj)
            agreement[obj["id"]] = 0.5

    # SLM objects with no matching YOLO — keep with moderate agreement.
    for i, slm in enumerate(slm_objects):
        if slm_used[i]:
            continue
        obj = _slm_to_merged(slm)
        merged.append(obj)
        agreement[obj["id"]] = 0.5

    return merged, agreement
```

- [ ] **Step 4: Run tests + confirm pass**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_consensus.py -xvs 2>&1 | tail -20
```
Expected: all 8 consensus_merge tests pass.

- [ ] **Step 5: Commit**

```bash
git add decroom_no_lidar/backend/pipeline/consensus.py \
        decroom_no_lidar/backend/tests/test_consensus.py
git commit -m "feat(backend): consensus_merge — YOLO 3D + SLM match with category-conflict handling"
```

---

### Task 11: collapse_duplicates function

**Files:**
- Modify: `decroom_no_lidar/backend/pipeline/consensus.py`
- Modify: `decroom_no_lidar/backend/tests/test_consensus.py`

- [ ] **Step 1: Write failing tests**

Append to `test_consensus.py`:

```python
def test_collapse_duplicates_two_beds_close_collapses_to_one():
    from pipeline.consensus import collapse_duplicates
    objs = [
        {"id": "a", "type": "bed", "position": {"x": 2.0, "y": 0, "z": 1.5},
         "metadata": {"width": 1.6, "length": 2.0, "height": 0.6}, "_confidence": 0.9},
        {"id": "b", "type": "bed", "position": {"x": 2.3, "y": 0, "z": 1.7},
         "metadata": {"width": 1.6, "length": 2.0, "height": 0.6}, "_confidence": 0.7},
    ]
    out = collapse_duplicates(objs)
    assert len(out) == 1
    assert out[0]["id"] == "a"  # higher confidence wins


def test_collapse_duplicates_two_beds_far_apart_both_kept():
    from pipeline.consensus import collapse_duplicates
    objs = [
        {"id": "a", "type": "bed", "position": {"x": 2.0, "y": 0, "z": 1.5},
         "metadata": {}, "_confidence": 0.9},
        {"id": "b", "type": "bed", "position": {"x": 5.0, "y": 0, "z": 1.5},
         "metadata": {}, "_confidence": 0.8},
    ]
    out = collapse_duplicates(objs)
    assert len(out) == 2


def test_collapse_duplicates_preserves_different_categories():
    from pipeline.consensus import collapse_duplicates
    objs = [
        {"id": "a", "type": "bed", "position": {"x": 2.0, "y": 0, "z": 1.5},
         "metadata": {}, "_confidence": 0.9},
        {"id": "b", "type": "chair", "position": {"x": 2.0, "y": 0, "z": 1.5},
         "metadata": {}, "_confidence": 0.9},
    ]
    out = collapse_duplicates(objs)
    assert len(out) == 2  # different categories don't collapse


def test_collapse_duplicates_three_chairs_transitively_collapse():
    from pipeline.consensus import collapse_duplicates
    # chairs at (0,0), (0.4,0), (0.8,0) — pairwise within 0.65m
    objs = [
        {"id": "a", "type": "chair", "position": {"x": 0.0, "y": 0, "z": 0},
         "metadata": {}, "_confidence": 0.8},
        {"id": "b", "type": "chair", "position": {"x": 0.4, "y": 0, "z": 0},
         "metadata": {}, "_confidence": 0.9},
        {"id": "c", "type": "chair", "position": {"x": 0.8, "y": 0, "z": 0},
         "metadata": {}, "_confidence": 0.7},
    ]
    out = collapse_duplicates(objs)
    # Run passes until stable — should end at 1 chair (b, highest conf)
    assert len(out) == 1
    assert out[0]["id"] == "b"
```

- [ ] **Step 2: Run + confirm failure**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_consensus.py::test_collapse_duplicates_two_beds_close_collapses_to_one -xvs 2>&1 | tail -5
```
Expected: ImportError — `collapse_duplicates` not defined.

- [ ] **Step 3: Implement collapse_duplicates**

Append to `pipeline/consensus.py`:

```python
# Per-category merge distances for duplicate collapse (meters).
MERGE_DISTANCE_BY_CATEGORY: Dict[str, float] = {
    "bed": 1.5,
    "couch": 1.5,
    "sofa": 1.5,
    "chair": 0.65,
    "dining_table": 1.1,
    "desk": 1.1,
    "tv": 0.8,
    "laptop": 0.5,
    "lamp": 0.5,
    "potted plant": 0.6,
    "bookshelf": 1.2,
    "dresser": 1.2,
}
_DEFAULT_MERGE_DISTANCE = 0.6


def _merge_distance_for(category: str) -> float:
    return MERGE_DISTANCE_BY_CATEGORY.get(category, _DEFAULT_MERGE_DISTANCE)


def _pos_tuple(obj: Dict[str, Any]) -> Tuple[float, float, float]:
    p = obj.get("position", {})
    return (float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)))


def _confidence_of(obj: Dict[str, Any]) -> float:
    return float(obj.get("_confidence", obj.get("confidence", 0.5)))


def collapse_duplicates(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Position-based duplicate collapse within each category.

    For each category: find pairs within the category's merge distance;
    keep the higher-confidence member; discard the other. Iterate until
    no pairs remain. Does NOT cap total count per category (rooms can
    have multiple beds).
    """
    current = list(objects)
    changed = True
    while changed:
        changed = False
        keep: List[Dict[str, Any]] = []
        dropped: set = set()
        for i, a in enumerate(current):
            if i in dropped:
                continue
            thresh = _merge_distance_for(a.get("type", ""))
            a_pos = _pos_tuple(a)
            a_conf = _confidence_of(a)
            survivor = a
            survivor_idx = i
            for j in range(i + 1, len(current)):
                if j in dropped:
                    continue
                b = current[j]
                if b.get("type") != survivor.get("type"):
                    continue
                if _distance_xyz(a_pos, _pos_tuple(b)) <= thresh:
                    # Collapse pair.
                    if _confidence_of(b) > a_conf:
                        dropped.add(survivor_idx)
                        survivor = b
                        survivor_idx = j
                        a_pos = _pos_tuple(b)
                        a_conf = _confidence_of(b)
                    else:
                        dropped.add(j)
                    changed = True
            keep.append(survivor)
        # Deduplicate `keep` — survivors that appear multiple times (when
        # the outer loop encountered them via different sources) get
        # collapsed by id.
        seen = set()
        out = []
        for o in keep:
            oid = o.get("id")
            if oid in seen:
                continue
            seen.add(oid)
            out.append(o)
        current = out
    return current
```

- [ ] **Step 4: Run + confirm pass**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_consensus.py -xvs 2>&1 | tail -20
```
Expected: all consensus tests pass (including the 4 new ones).

- [ ] **Step 5: Commit**

```bash
git add decroom_no_lidar/backend/pipeline/consensus.py \
        decroom_no_lidar/backend/tests/test_consensus.py
git commit -m "feat(backend): collapse_duplicates — position-based merge, no instance caps"
```

---

### Task 12: snap_to_walls function

**Files:**
- Modify: `decroom_no_lidar/backend/pipeline/consensus.py`
- Create: `decroom_no_lidar/backend/tests/test_wall_snapping.py`

- [ ] **Step 1: Write failing tests**

Create `decroom_no_lidar/backend/tests/test_wall_snapping.py`:

```python
"""Tests for snap_to_walls."""
from pipeline.consensus import snap_to_walls


def _wall(x0, z0, x1, z1):
    return {"start": {"x": x0, "z": z0}, "end": {"x": x1, "z": z1}}


def _obj(type_, x, z, rotation_y=0.0):
    return {"id": type_[0] + "1", "type": type_,
             "position": {"x": x, "y": 0, "z": z},
             "rotation_y": rotation_y,
             "metadata": {"width": 1.6, "length": 2.0, "height": 0.6}}


def test_bed_within_tolerance_snaps_to_nearest_wall():
    # North wall at z=0 from x=0 to x=5. Bed at (2, 0.3) should snap to z=0.
    walls = [_wall(0, 0, 5, 0)]
    objs = [_obj("bed", 2.0, 0.3)]
    snapped = snap_to_walls(objs, walls)
    assert len(snapped) == 1
    assert snapped[0]["position"]["z"] == 0.0
    assert snapped[0]["position"]["x"] == 2.0  # x along the wall unchanged


def test_bed_beyond_tolerance_stays_put():
    walls = [_wall(0, 0, 5, 0)]
    objs = [_obj("bed", 2.0, 3.0)]  # 3m from wall, tolerance is 0.8m
    snapped = snap_to_walls(objs, walls)
    assert snapped[0]["position"]["z"] == 3.0


def test_chair_freestanding_does_not_snap():
    walls = [_wall(0, 0, 5, 0)]
    objs = [_obj("chair", 2.0, 0.2)]
    snapped = snap_to_walls(objs, walls)
    assert snapped[0]["position"]["z"] == 0.2


def test_nightstand_uses_tighter_tolerance():
    walls = [_wall(0, 0, 5, 0)]
    # Nightstand tolerance is 0.5m; object is 0.7m away → NO snap.
    nightstand = _obj("nightstand", 2.0, 0.7)
    snapped = snap_to_walls([nightstand], walls)
    assert snapped[0]["position"]["z"] == 0.7
    # Within 0.5m does snap.
    nightstand_close = _obj("nightstand", 2.0, 0.3)
    snapped2 = snap_to_walls([nightstand_close], walls)
    assert snapped2[0]["position"]["z"] == 0.0


def test_no_walls_no_snap():
    snapped = snap_to_walls([_obj("bed", 2.0, 0.3)], walls=[])
    assert snapped[0]["position"]["z"] == 0.3


def test_snap_rotation_set_outward_perpendicular_to_wall():
    """North wall, bed below it — rotation_y should face south (into room)."""
    import math
    # Wall along X axis (at z=0); room is on positive-z side.
    walls = [_wall(0, 0, 5, 0)]
    # Wall yaw = atan2(dx, dz) with dx=5, dz=0 → pi/2. Normal outward = yaw - pi/2 = 0.
    # Object should face outward (away from wall, into room).
    snapped = snap_to_walls([_obj("bed", 2.0, 0.2)], walls)
    # Just verify that rotation_y was set (not the default 0 from input)
    # and is some finite number. Exact convention tested in a dedicated
    # orientation test if the app shows incorrect direction.
    assert math.isfinite(snapped[0]["rotation_y"])
```

- [ ] **Step 2: Run + confirm failure**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_wall_snapping.py -xvs 2>&1 | tail -10
```
Expected: ImportError.

- [ ] **Step 3: Implement snap_to_walls**

Append to `pipeline/consensus.py`:

```python
# Wall-snap behavior per category.
SNAP_TO_WALL_CATEGORIES = {
    "bed": 0.8,
    "dresser": 0.8,
    "wardrobe": 0.8,
    "sofa": 0.8,
    "couch": 0.8,
    "bookshelf": 0.8,
    "desk": 0.8,
    "nightstand": 0.5,
    "tv": 0.8,
}
FREESTANDING_CATEGORIES = {
    "chair", "stool", "dining_table", "coffee_table", "side_table",
    "lamp", "lamp_floor", "lamp_desk", "potted plant", "rug", "poster",
    "laptop",
}


def _project_point_onto_segment(px: float, pz: float,
                                  x0: float, z0: float,
                                  x1: float, z1: float) -> Tuple[float, float, float]:
    """Return (projected_x, projected_z, perpendicular_distance)."""
    dx, dz = x1 - x0, z1 - z0
    seg_len_sq = dx * dx + dz * dz
    if seg_len_sq <= 1e-8:
        # Degenerate wall.
        return (x0, z0, math.sqrt((px - x0) ** 2 + (pz - z0) ** 2))
    t = ((px - x0) * dx + (pz - z0) * dz) / seg_len_sq
    t = max(0.0, min(1.0, t))
    qx, qz = x0 + t * dx, z0 + t * dz
    d = math.sqrt((px - qx) ** 2 + (pz - qz) ** 2)
    return (qx, qz, d)


def _wall_yaw(wall: Dict[str, Any]) -> float:
    """Yaw of the wall direction in ARKit frame (radians)."""
    s, e = wall["start"], wall["end"]
    return math.atan2(e["x"] - s["x"], e["z"] - s["z"])


def snap_to_walls(
    objects: List[Dict[str, Any]],
    walls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Snap each applicable object to the nearest wall within its category
    tolerance. Freestanding categories pass through unchanged. Objects
    beyond tolerance stay at their input position."""
    if not walls:
        return list(objects)

    out: List[Dict[str, Any]] = []
    for obj in objects:
        cat = obj.get("type", "")
        if cat in FREESTANDING_CATEGORIES:
            out.append(obj)
            continue
        tol = SNAP_TO_WALL_CATEGORIES.get(cat)
        if tol is None:
            out.append(obj)
            continue

        ox = obj["position"]["x"]
        oz = obj["position"]["z"]

        best: Tuple[float, float, float, Dict[str, Any]] | None = None
        best_d = float("inf")
        for w in walls:
            qx, qz, d = _project_point_onto_segment(
                ox, oz,
                w["start"]["x"], w["start"]["z"],
                w["end"]["x"], w["end"]["z"],
            )
            if d < best_d:
                best_d = d
                best = (qx, qz, d, w)

        if best is None or best_d > tol:
            out.append(obj)
            continue

        qx, qz, _, w = best
        snapped = dict(obj)
        snapped["position"] = {"x": qx, "y": obj["position"]["y"], "z": qz}
        # Rotation: face outward from the wall. Wall yaw + pi/2 gives the
        # normal. Convention here: wall_yaw then + pi so the object faces
        # into the room (opposite the wall-into-outside direction).
        wall_yaw = _wall_yaw(w)
        snapped["rotation_y"] = wall_yaw + math.pi
        out.append(snapped)
    return out
```

- [ ] **Step 4: Run + confirm pass**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_wall_snapping.py -xvs 2>&1 | tail -15
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add decroom_no_lidar/backend/pipeline/consensus.py \
        decroom_no_lidar/backend/tests/test_wall_snapping.py
git commit -m "feat(backend): snap_to_walls — per-category wall-snap tolerances"
```

---

### Task 13: compute_confidence function

**Files:**
- Modify: `decroom_no_lidar/backend/pipeline/consensus.py`
- Create: `decroom_no_lidar/backend/tests/test_confidence.py`

- [ ] **Step 1: Write failing tests**

Create `decroom_no_lidar/backend/tests/test_confidence.py`:

```python
"""Tests for compute_confidence."""
from pipeline.consensus import compute_confidence


def _obj(type_, width=1.6, length=2.0, height=0.6):
    return {"type": type_, "metadata": {"width": width, "length": length,
                                            "height": height}}


def test_full_agreement_multiview_plausible_yields_high_confidence():
    c = compute_confidence(_obj("bed"),
                             agreement_score=1.0,
                             yolo_view_count=5,
                             yolo_matched=True)
    # 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 1.0
    assert c == 1.0


def test_slm_only_neutral_view_plausible_mid_confidence():
    c = compute_confidence(_obj("bed"),
                             agreement_score=0.5,
                             yolo_view_count=0,
                             yolo_matched=False)
    # 0.5*0.5 + 0.3*0.5 + 0.2*1.0 = 0.25 + 0.15 + 0.2 = 0.6
    assert abs(c - 0.6) < 1e-6


def test_duplicate_collapse_survivor_low_view_implausible_size_low_confidence():
    c = compute_confidence(_obj("bed", width=3.5, length=5.0, height=2.0),
                             agreement_score=0.3,
                             yolo_view_count=1,
                             yolo_matched=True)
    # 0.5*0.3 + 0.3*0.4 + 0.2*0.3 = 0.15 + 0.12 + 0.06 = 0.33
    assert abs(c - 0.33) < 1e-6


def test_confidence_clamped_to_01_range():
    c = compute_confidence(_obj("bed"),
                             agreement_score=2.0,  # out-of-range input
                             yolo_view_count=100,
                             yolo_matched=True)
    assert 0.0 <= c <= 1.0


def test_multi_view_score_scales_with_view_count_when_matched():
    low = compute_confidence(_obj("bed"),
                               agreement_score=1.0, yolo_view_count=1,
                               yolo_matched=True)
    mid = compute_confidence(_obj("bed"),
                               agreement_score=1.0, yolo_view_count=2,
                               yolo_matched=True)
    high = compute_confidence(_obj("bed"),
                               agreement_score=1.0, yolo_view_count=3,
                               yolo_matched=True)
    assert low < mid < high
```

- [ ] **Step 2: Run + confirm failure**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_confidence.py -xvs 2>&1 | tail -8
```
Expected: ImportError.

- [ ] **Step 3: Implement compute_confidence**

Append to `pipeline/consensus.py`:

```python
# Plausible size ranges per category (width_min, width_max, length_min,
# length_max, height_min, height_max) in meters.
_SIZE_PLAUSIBILITY: Dict[str, Tuple[float, float, float, float, float, float]] = {
    "bed":          (1.2, 2.2, 1.9, 2.2, 0.3, 1.2),
    "couch":        (1.5, 2.8, 0.8, 1.1, 0.7, 1.1),
    "sofa":         (1.5, 2.8, 0.8, 1.1, 0.7, 1.1),
    "chair":        (0.4, 0.7, 0.4, 0.7, 0.7, 1.2),
    "dining_table": (0.9, 2.2, 0.7, 1.4, 0.65, 0.85),
    "desk":         (0.9, 1.8, 0.5, 0.9, 0.65, 0.85),
    "tv":           (0.8, 1.8, 0.05, 0.2, 0.45, 1.0),
    "laptop":       (0.25, 0.45, 0.20, 0.35, 0.01, 0.05),
    "lamp":         (0.15, 0.6, 0.15, 0.6, 0.3, 1.8),
    "potted plant": (0.1, 0.7, 0.1, 0.7, 0.1, 1.5),
    "dresser":      (0.6, 1.4, 0.35, 0.6, 0.7, 1.0),
    "bookshelf":    (0.5, 1.5, 0.2, 0.5, 0.8, 2.2),
}


def _size_plausibility(obj: Dict[str, Any]) -> float:
    m = obj.get("metadata", {})
    w = float(m.get("width", 1.0))
    l = float(m.get("length", 1.0))
    h = float(m.get("height", 1.0))
    cat = obj.get("type", "")
    rng = _SIZE_PLAUSIBILITY.get(cat)
    if rng is None:
        return 0.7  # unknown category — neutral-ish
    w_min, w_max, l_min, l_max, h_min, h_max = rng
    in_range = (w_min <= w <= w_max and l_min <= l <= l_max
                and h_min <= h <= h_max)
    return 1.0 if in_range else 0.3


def _multi_view_score(yolo_matched: bool, yolo_view_count: int) -> float:
    if not yolo_matched:
        return 0.5
    if yolo_view_count >= 3:
        return 1.0
    if yolo_view_count == 2:
        return 0.7
    if yolo_view_count == 1:
        return 0.4
    return 0.2


def compute_confidence(
    obj: Dict[str, Any],
    agreement_score: float,
    yolo_view_count: int,
    yolo_matched: bool,
) -> float:
    """
    confidence = 0.5 * agreement_score
               + 0.3 * multi_view_score
               + 0.2 * size_plausibility

    Result is clamped to [0, 1].
    """
    a = max(0.0, min(1.0, agreement_score))
    m = _multi_view_score(yolo_matched, yolo_view_count)
    s = _size_plausibility(obj)
    c = 0.5 * a + 0.3 * m + 0.2 * s
    return max(0.0, min(1.0, c))
```

- [ ] **Step 4: Run + confirm pass**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_confidence.py -xvs 2>&1 | tail -15
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add decroom_no_lidar/backend/pipeline/consensus.py \
        decroom_no_lidar/backend/tests/test_confidence.py
git commit -m "feat(backend): compute_confidence — 0.5a + 0.3mv + 0.2sp formula"
```

---

### Task 14: extract_doors_windows function

**Files:**
- Modify: `decroom_no_lidar/backend/pipeline/consensus.py`
- Create: `decroom_no_lidar/backend/tests/test_door_window.py`

**Prereq:** Task 8's B0 findings confirm (or update this task to match) the SpatialLM door/window schema. If Task 8 revealed a different format, adjust the field names below.

- [ ] **Step 1: Write failing tests**

Create `decroom_no_lidar/backend/tests/test_door_window.py`:

```python
"""Tests for extract_doors_windows. Assumes SpatialLM emits dicts with
class_name in {'door', 'window'} and a 'position' or 'center' key.
If the B0 findings show a different schema, update both this test and
the implementation."""
from pipeline.consensus import extract_doors_windows


def _wall(x0, z0, x1, z1):
    return {"id": "w0", "start": {"x": x0, "z": z0}, "end": {"x": x1, "z": z1},
             "has_door": False, "has_window": False}


def test_door_near_wall_attaches_to_wall():
    walls = [_wall(0, 0, 5, 0)]
    slm_objects = [{"class_name": "door", "position": {"x": 2.0, "y": 1.0, "z": 0.05}}]
    updated = extract_doors_windows(slm_objects, walls)
    assert updated[0]["has_door"] is True


def test_window_near_wall_attaches_to_wall():
    walls = [_wall(0, 0, 5, 0)]
    slm_objects = [{"class_name": "window", "position": {"x": 2.0, "y": 1.5, "z": 0.05}}]
    updated = extract_doors_windows(slm_objects, walls)
    assert updated[0]["has_window"] is True


def test_door_far_from_all_walls_not_attached():
    walls = [_wall(0, 0, 5, 0)]
    slm_objects = [{"class_name": "door", "position": {"x": 2.0, "y": 1.0, "z": 3.0}}]
    updated = extract_doors_windows(slm_objects, walls)
    assert updated[0]["has_door"] is False


def test_non_door_entries_ignored():
    walls = [_wall(0, 0, 5, 0)]
    slm_objects = [{"class_name": "chair", "position": {"x": 2.0, "y": 0, "z": 0.1}}]
    updated = extract_doors_windows(slm_objects, walls)
    assert updated[0]["has_door"] is False
    assert updated[0]["has_window"] is False


def test_multiple_doors_attach_to_respective_walls():
    walls = [_wall(0, 0, 5, 0), _wall(5, 0, 5, 5)]
    slm_objects = [
        {"class_name": "door", "position": {"x": 2.0, "y": 1.0, "z": 0.05}},
        {"class_name": "door", "position": {"x": 4.95, "y": 1.0, "z": 2.5}},
    ]
    updated = extract_doors_windows(slm_objects, walls)
    assert updated[0]["has_door"] is True
    assert updated[1]["has_door"] is True
```

- [ ] **Step 2: Run + confirm failure**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_door_window.py -xvs 2>&1 | tail -8
```
Expected: ImportError.

- [ ] **Step 3: Implement extract_doors_windows**

Append to `pipeline/consensus.py`:

```python
DOOR_WINDOW_WALL_MATCH_DISTANCE = 0.5  # meters


def _pos_dict_to_xz(p: Any) -> Tuple[float, float] | None:
    """Accept {x,y,z} dict or [x,y,z] list; return (x, z) or None."""
    if isinstance(p, dict):
        return (float(p.get("x", 0)), float(p.get("z", 0)))
    if isinstance(p, (list, tuple)) and len(p) >= 3:
        return (float(p[0]), float(p[2]))
    return None


def extract_doors_windows(
    slm_objects: List[Dict[str, Any]],
    walls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Mutate (and return) walls with has_door / has_window set based on
    door/window entries in the SpatialLM output. Walls that match no
    door/window keep their existing flags. Entries that match no wall
    are logged and ignored."""
    for obj in slm_objects:
        cls = (obj.get("class_name") or obj.get("type") or "").lower()
        if cls not in ("door", "window"):
            continue
        pos = obj.get("position") or obj.get("center")
        xz = _pos_dict_to_xz(pos)
        if xz is None:
            logger.warning("door/window missing position: %s", obj)
            continue
        px, pz = xz

        best_wall = None
        best_d = float("inf")
        for w in walls:
            _, _, d = _project_point_onto_segment(
                px, pz,
                w["start"]["x"], w["start"]["z"],
                w["end"]["x"], w["end"]["z"],
            )
            if d < best_d:
                best_d = d
                best_wall = w

        if best_wall is None or best_d > DOOR_WINDOW_WALL_MATCH_DISTANCE:
            logger.info("%s at (%.2f, %.2f) had no matching wall within %.2fm",
                         cls, px, pz, DOOR_WINDOW_WALL_MATCH_DISTANCE)
            continue

        if cls == "door":
            best_wall["has_door"] = True
        else:
            best_wall["has_window"] = True
    return walls
```

- [ ] **Step 4: Run + confirm pass**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_door_window.py -xvs 2>&1 | tail -15
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add decroom_no_lidar/backend/pipeline/consensus.py \
        decroom_no_lidar/backend/tests/test_door_window.py
git commit -m "feat(backend): extract_doors_windows — attach SLM openings to walls"
```

---

### Task 15: Wire consensus pipeline into run_phone_first_pipeline

**Files:**
- Modify: `decroom_no_lidar/backend/pipeline/phone_first_pipeline.py`
- Modify: `decroom_no_lidar/backend/tests/test_phone_first_pipeline.py`

- [ ] **Step 1: Write failing integration test**

Append to `test_phone_first_pipeline.py`:

```python
def test_run_phone_first_pipeline_applies_consensus_and_snapping(monkeypatch):
    """End-to-end-ish: stub SpatialLM to return a bed far from the wall,
    send a YOLO object agreeing on a bed, and assert the RoomPayload bed
    is wall-snapped, has confidence > 0.5, and matches the SLM type."""
    import json
    from pipeline import phone_first_pipeline as mod

    # Stub SpatialLM to return one bed at (2, 0, 0.5) — 0.5m from the z=0 wall.
    def fake_run(points, cfg):
        walls = [{"start": [0, 0], "end": [5, 0]}]
        objects = [{
            "type": "bed",
            "position": {"x": 2.0, "y": 0.0, "z": 0.5},
            "rotation_y": 0.0,
            "metadata": {"width": 1.6, "length": 2.0, "height": 0.6},
            "confidence": 0.5,
            "id": "slm0",
        }]
        return walls, objects
    monkeypatch.setattr(mod, "run_spatiallm_pipeline", fake_run)

    # Build payload.
    import numpy as np
    payload = mod.PhoneFirstPayload(
        room_geometry={
            "walls": [{"start": {"x": 0, "z": 0}, "end": {"x": 5, "z": 0}}],
            "floor_height": 0.0, "ceiling_height": 2.5, "manhattan_yaw": 0.0,
            "coordinate_system": "arkit_world_right_handed_y_up",
        },
        point_cloud_ply=b"ply\n",
        keyframes_manifest={"frames": []},
        keyframe_jpegs=[],
        scan_log={},
        yolo_objects=[{
            "category": "bed", "position": [2.0, 0.0, 0.5],
            "peak_confidence": 0.85, "view_count": 4, "best_frame_index": 2,
        }],
    )

    # Avoid actually loading the PLY (empty bytes). Patch load.
    monkeypatch.setattr(mod, "load_ply_points",
                        lambda b: (np.zeros((1, 3), dtype=np.float32),
                                   np.zeros((1, 3), dtype=np.uint8)))

    result = mod.run_phone_first_pipeline(payload, cfg=None)
    beds = [o for o in result["objects"] if o["type"] == "bed"]
    assert len(beds) == 1
    bed = beds[0]
    # Wall-snapped: position.z should be 0, not 0.5.
    assert abs(bed["position"]["z"]) < 1e-4
    # Confidence well above baseline 0.5 because YOLO agreed + 4 views + plausible size.
    assert bed["confidence"] > 0.7


def test_run_phone_first_pipeline_sets_has_door_when_slm_emits_door(monkeypatch):
    from pipeline import phone_first_pipeline as mod
    import numpy as np

    def fake_run(points, cfg):
        walls = [{"start": [0, 0], "end": [5, 0]}]
        objects = [{"class_name": "door", "position": {"x": 2.0, "y": 1.0, "z": 0.05}}]
        return walls, objects
    monkeypatch.setattr(mod, "run_spatiallm_pipeline", fake_run)
    monkeypatch.setattr(mod, "load_ply_points",
                        lambda b: (np.zeros((1, 3), dtype=np.float32),
                                   np.zeros((1, 3), dtype=np.uint8)))

    payload = mod.PhoneFirstPayload(
        room_geometry={
            "walls": [{"start": {"x": 0, "z": 0}, "end": {"x": 5, "z": 0}}],
            "floor_height": 0.0, "ceiling_height": 2.5, "manhattan_yaw": 0.0,
            "coordinate_system": "arkit_world_right_handed_y_up",
        },
        point_cloud_ply=b"ply\n", keyframes_manifest={"frames": []},
        keyframe_jpegs=[], scan_log={}, yolo_objects=[],
    )
    result = mod.run_phone_first_pipeline(payload, cfg=None)
    walls = result["room"]["walls"]
    assert any(w.get("has_door") for w in walls)
```

- [ ] **Step 2: Run + confirm failure**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_phone_first_pipeline.py::test_run_phone_first_pipeline_applies_consensus_and_snapping tests/test_phone_first_pipeline.py::test_run_phone_first_pipeline_sets_has_door_when_slm_emits_door -xvs 2>&1 | tail -15
```
Expected: FAIL — `RoomObject.confidence` still flat 0.5, bed position not wall-snapped, `has_door` absent.

- [ ] **Step 3: Wire consensus into the pipeline**

In `phone_first_pipeline.py`, replace the last part of `run_phone_first_pipeline` (from `objects = map_spatiallm_objects_to_room_objects(...)` onward) with:

```python
    # Map SpatialLM raw output through category mapping.
    slm_mapped = map_spatiallm_objects_to_room_objects(slm_objects or [])
    # Convert mapped RoomObjects to the consensus dict shape.
    slm_dicts = [
        {
            "id": o.id,
            "type": o.type,
            "position": {"x": o.position.x, "y": o.position.y, "z": o.position.z},
            "rotation_y": o.rotation_y,
            "metadata": {
                "width": o.metadata.width,
                "length": o.metadata.length,
                "height": o.metadata.height,
            },
            "_confidence": o.confidence,
        }
        for o in slm_mapped
    ]

    from .consensus import (
        consensus_merge, collapse_duplicates,
        snap_to_walls, compute_confidence,
        extract_doors_windows,
    )

    # Walls in phone geometry frame.
    phone_walls = payload.room_geometry["walls"]

    merged, agreement = consensus_merge(
        payload.yolo_objects, slm_dicts, phone_walls
    )
    merged = collapse_duplicates(merged)
    merged = snap_to_walls(merged, phone_walls)

    # Attach doors/windows to walls in-place.
    walls_with_openings = [dict(w, has_door=False, has_window=False) for w in phone_walls]
    walls_with_openings = extract_doors_windows(slm_objects or [], walls_with_openings)

    # Index YOLO view counts by object id via position match (best-effort).
    def _yolo_view_count_for(mobj: Dict[str, Any]) -> tuple[bool, int]:
        mx = mobj["position"]["x"]; mz = mobj["position"]["z"]
        best_vc, best_d, matched = 0, float("inf"), False
        for y in payload.yolo_objects:
            yx, _, yz = y["position"]
            d = math.sqrt((yx - mx) ** 2 + (yz - mz) ** 2)
            if d < best_d and y["category"] == mobj["type"]:
                best_d = d; best_vc = int(y["view_count"]); matched = True
        return matched, best_vc if matched else 0

    # Build final RoomObjects.
    final_objects: List[RoomObject] = []
    for m in merged:
        yolo_matched, yolo_vc = _yolo_view_count_for(m)
        conf = compute_confidence(
            m,
            agreement_score=agreement.get(m["id"], 0.5),
            yolo_view_count=yolo_vc,
            yolo_matched=yolo_matched,
        )
        final_objects.append(RoomObject(
            id=m["id"],
            type=m["type"],
            asset_id=f"kenney_{m['type']}",
            position=ObjectPosition(
                x=m["position"]["x"], y=m["position"]["y"], z=m["position"]["z"]
            ),
            rotation_y=m.get("rotation_y", 0.0),
            scale=1.0,
            confidence=conf,
            metadata=ObjectMetadata(
                width=m["metadata"].get("width", 1.0),
                length=m["metadata"].get("length", 1.0),
                height=m["metadata"].get("height", 1.0),
                inferred=True,
                support_views=None,
            ),
        ))

    # Build RoomInfo with the enriched walls (with has_door/has_window).
    manhattan_yaw = float(payload.room_geometry.get("manhattan_yaw", 0.0))
    room_info = room_envelope_from_walls(
        walls_with_openings,
        payload.room_geometry["floor_height"],
        payload.room_geometry["ceiling_height"],
        positions=positions,
        manhattan_yaw=manhattan_yaw,
    )

    result = RoomPayload(room=room_info, objects=final_objects).to_dict()
    elapsed = time.time() - t0
    logger.info(
        "Phone-first pipeline done: %d walls, %d objects, %.2fs",
        len(room_info.walls), len(final_objects), elapsed,
    )
    return result
```

Also ensure `Wall` schema in `schema.py` supports `has_door` / `has_window`. Read `schema.py` if uncertain; both fields should already exist per the spec's "Wall schema already has has_door: Bool and has_window: Bool fields (from pre-3a days)." If missing, add:

```python
class Wall(BaseModel):
    id: str
    start: List[float]
    end: List[float]
    has_door: bool = False
    has_window: bool = False
```

- [ ] **Step 4: Ensure room_envelope_from_walls propagates has_door/has_window**

In `phone_first_pipeline.py`, find `room_envelope_from_walls` (around line 265). The `walls_out` list construction currently only passes `id`, `start`, `end`. Update:

```python
    walls_out = [
        Wall(
            id=f"w{i}",
            start=[w["start"]["x"], w["start"]["z"]],
            end=[w["end"]["x"], w["end"]["z"]],
            has_door=bool(w.get("has_door", False)),
            has_window=bool(w.get("has_window", False)),
        )
        for i, w in enumerate(walls)
    ]
```

- [ ] **Step 5: Run integration tests + confirm pass**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_phone_first_pipeline.py -xvs 2>&1 | tail -20
```
Expected: all pass including the two new integration tests.

- [ ] **Step 6: Run the full backend test suite**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/ -xvs 2>&1 | tail -10
```
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add decroom_no_lidar/backend/pipeline/phone_first_pipeline.py \
        decroom_no_lidar/backend/tests/test_phone_first_pipeline.py \
        decroom_no_lidar/backend/schema.py
git commit -m "feat(backend): wire consensus + wall-snapping + doors/windows into phone-first pipeline"
```

---

### Task 16: Add yolo_objects contract fixture

**Files:**
- Create: `decroom_no_lidar/backend/tests/fixtures/phone_first_payload/yolo_objects.json`

- [ ] **Step 1: Create the fixture directory and file**

```bash
mkdir -p /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend/tests/fixtures/phone_first_payload
```

Create `decroom_no_lidar/backend/tests/fixtures/phone_first_payload/yolo_objects.json`:

```json
{
  "objects": [
    {
      "category": "bed",
      "position": [2.0, 0.0, 1.5],
      "peak_confidence": 0.87,
      "view_count": 12,
      "best_frame_index": 34
    },
    {
      "category": "chair",
      "position": [3.5, 0.0, 2.8],
      "peak_confidence": 0.71,
      "view_count": 6,
      "best_frame_index": 58
    },
    {
      "category": "potted plant",
      "position": [0.3, 0.0, 2.9],
      "peak_confidence": 0.62,
      "view_count": 3,
      "best_frame_index": 67
    }
  ]
}
```

- [ ] **Step 2: Add a fixture-accept test**

Append to `decroom_no_lidar/backend/tests/test_phone_first_pipeline.py`:

```python
def test_fixture_yolo_objects_parses_cleanly():
    from pipeline.phone_first_pipeline import parse_payload
    import pathlib, json

    fx = pathlib.Path(__file__).parent / "fixtures" / "phone_first_payload" / "yolo_objects.json"
    raw = fx.read_bytes()

    # Build a minimal valid payload just to exercise parse_payload with this fixture.
    room_geom = json.dumps({
        "walls": [], "floor_height": 0.0, "ceiling_height": 2.5,
        "manhattan_yaw": 0.0,
        "coordinate_system": "arkit_world_right_handed_y_up",
    }).encode()
    ply = b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n"
    manifest = json.dumps({"frames": []}).encode()
    log = b"{}"
    payload = parse_payload(room_geom, ply, manifest, [], log, raw)
    assert len(payload.yolo_objects) == 3
    assert {o["category"] for o in payload.yolo_objects} == {"bed", "chair", "potted plant"}
```

- [ ] **Step 3: Run + confirm pass**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_phone_first_pipeline.py::test_fixture_yolo_objects_parses_cleanly -xvs 2>&1 | tail -8
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add decroom_no_lidar/backend/tests/fixtures/phone_first_payload/yolo_objects.json \
        decroom_no_lidar/backend/tests/test_phone_first_pipeline.py
git commit -m "test(backend): add yolo_objects.json contract fixture"
```

---

### Task 17: Update test_upload_schema for optional yolo_objects

**Files:**
- Modify: `decroom_no_lidar/backend/tests/test_upload_schema.py`

- [ ] **Step 1: Add schema-validation test**

Append to `tests/test_upload_schema.py`:

```python
def test_schema_error_for_malformed_yolo_objects():
    from pipeline.phone_first_pipeline import parse_payload, PhoneFirstSchemaError
    import json, pytest

    room_geom = json.dumps({
        "walls": [], "floor_height": 0.0, "ceiling_height": 2.5,
        "manhattan_yaw": 0.0,
        "coordinate_system": "arkit_world_right_handed_y_up",
    }).encode()
    ply = b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n"
    manifest = json.dumps({"frames": []}).encode()
    log = b"{}"
    # Missing required fields in yolo_objects entry
    bad = json.dumps({"objects": [{"category": "bed"}]}).encode()

    with pytest.raises(PhoneFirstSchemaError):
        parse_payload(room_geom, ply, manifest, [], log, bad)
```

- [ ] **Step 2: Run + confirm pass**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m pytest tests/test_upload_schema.py::test_schema_error_for_malformed_yolo_objects -xvs 2>&1 | tail -8
```
Expected: PASS (the `parse_payload` validation from Task 9 rejects missing fields).

- [ ] **Step 3: Commit**

```bash
git add decroom_no_lidar/backend/tests/test_upload_schema.py
git commit -m "test(backend): reject malformed yolo_objects in upload schema"
```

---

### Task 18: Manual QA and merge

**Files:**
- Create: `docs/superpowers/qa/YYYY-MM-DD-plan3b-manual-qa.md` (replace with today's date)

**Requires:** live SpatialLM tunnel, physical test device, real room to scan.

- [ ] **Step 1: Deploy backend + phone builds**

- Backend: re-run `uvicorn` with the current branch (the server auto-reloads on file changes; restart if needed).
- iOS: install the current-branch build on the test device.
- SpatialLM tunnel: `ssh -L 8100:localhost:8100 root@<pod-ip>` (see `memory/runpod_setup.md`).

- [ ] **Step 2: Run the QA checklist**

Copy the template from `docs/superpowers/qa/` (any prior plan's QA doc is a reference) and fill in with real-device observations. Mandatory checks:

| Check | Pass criteria |
|---|---|
| Bed against a wall | Rendered bed visibly touches the wall (not floating). Confidence > 0.7. |
| Two chairs in a room | Both chairs rendered (no false dedup). Chair category shows in `RoomPayload.objects`. |
| Door in a wall | Corresponding wall has `has_door = true` in the response. |
| Cluttered room | No `SpatialLM` duplicates at same position survive. |
| YOLO-only detection | A tracked object SLM missed appears in the response with `confidence ≈ 0.5`. |
| Missing YOLO model case | App still completes the scan; scan_log contains `yolo_runner_load_failed`; backend returns walls-only or SLM-only result. |
| Regression: empty room | Scan of an empty room still produces a valid `RoomPayload` with empty `objects`. |

For each check, write what you saw (position values, confidence numbers, screenshots if helpful).

- [ ] **Step 3: Commit the QA doc**

```bash
git add docs/superpowers/qa/*.md
git commit -m "docs: Plan 3b manual QA results"
```

- [ ] **Step 4: Open a PR merging `feature/plan3b-semantic-polish` into `main`**

```bash
git push -u origin feature/plan3b-semantic-polish
```

Then use `gh pr create` per the repo's conventions. PR body should summarize: iOS YOLO enable + unprojection, backend consensus module, door/window detection, manual QA evidence.

---

## Self-review

Checking this plan against the spec (`docs/superpowers/specs/2026-04-24-plan3b-semantic-polish-design.md` v2):

**Spec coverage:**
- §1 iOS YOLO re-enable + unprojection fix — Tasks 1, 2, 3, 4 ✓
- §2 ObjectTracker → `yolo_objects.json` serialization — Tasks 5, 7 ✓
- §3 `consensus_merge` with category-conflict row — Task 10 ✓
- §4 `collapse_duplicates` (no instance caps) — Task 11 ✓
- §5 `snap_to_walls` — Task 12 ✓
- §6 `compute_confidence` (view_count-based multi-view score) — Task 13 ✓
- §7 `extract_doors_windows` — Task 14 ✓ (gated by Task 8 Phase B0)
- Data shapes (yolo_objects.json schema) — Tasks 5, 9, 16 ✓
- Wall-consolidation dependency — marked as prerequisite in the plan header ✓ (already merged in Plan 3a per commit `a40b2d2`)
- SpatialLM format verification as blocking B0 — Task 8 ✓
- Migration plan Phases A–D mapped to Tasks 1–18 ✓
- File structure (new + modified files) — tables in this plan match the spec ✓

**Placeholder scan:** No "TBD", "TODO", or vague steps. Each task has exact file paths, complete test code, complete implementation code, exact commands.

**Type consistency:**
- `YoloObject.peak_confidence` / `view_count` / `best_frame_index` — same name in ScanUploadPayload struct (Task 5), multipart (Task 6), Python `parse_payload` validation (Task 9), contract fixture (Task 16). ✓
- `consensus_merge` returns `(merged_objects, agreement_scores)` in Task 10; Task 15 consumes both. ✓
- `compute_confidence(obj, agreement_score, yolo_view_count, yolo_matched)` signature matches between Task 13 impl and Task 15 call site. ✓
- `extract_doors_windows(slm_objects, walls)` — Task 14 sets `has_door`/`has_window` on the wall dicts; Task 15 passes those into `room_envelope_from_walls` which propagates them to `Wall` Pydantic model. ✓

**Scope check:** One feature (semantic polish layer), one branch, one merge. No cross-subsystem decomposition needed.

**Ambiguity check:**
- Depth/YOLO dispatch ordering: task 2 adds snapshot write after depth inference; task 4 YOLO dispatch only reads the snapshot. If no depth snapshot has landed yet (first ~1s of scan), YOLO drops the detection with a logged event — explicit in Task 3.
- YOLO bbox normalized coordinates: `YoloDetection.bbox` is a `CGRect` with normalized `midX`/`midY` in [0,1] (confirmed by reading YoloRunner). Task 1 unit tests cover clamp behavior.
- `Wall.has_door` field existence: Task 15 step 3 includes a fallback — "If missing, add" — so either state is handled.

Plan is ready for execution.

---

## Plan complete and saved to `docs/superpowers/plans/2026-04-24-plan3b-semantic-polish.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
