# Foundation + On-Device Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the on-device foundation for phone-first AR scanning — device tier detection, per-scan logging, Core ML depth inference, room geometry from plane anchors, voxelized point cloud accumulation, and smart keyframe selection.

**Architecture:** Seven focused Swift files in the iOS app, each with a single responsibility. Tested via Swift Testing (`@Test` / `#expect`) with protocol abstractions for ARKit/Core ML dependencies so unit tests can run without simulator hardware. No UI changes — this layer produces testable data that Plan 2 (visual capture) will consume.

**Tech Stack:** Swift 5.9+, SwiftUI, ARKit, Core ML, Swift Testing, XCTest runner.

---

## File Structure

**New source files** (all in `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/`):
- `DeviceCapabilities.swift` — chip/ARKit tier detection
- `ScanLogger.swift` — per-scan JSON log writer
- `DepthRunner.swift` — Core ML metric depth inference wrapper
- `PlaneAnchorDescriptor.swift` — protocol abstraction over `ARPlaneAnchor` for testing
- `RoomGeometryBuilder.swift` — plane anchors → wall segments
- `PointCloudAccumulator.swift` — depth + pose + RGB → voxelized point cloud + PLY export
- `KeyframeSelector.swift` — farthest-point + semantic keyframe selection

**New test files** (all in `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/`):
- `DeviceCapabilitiesTests.swift`
- `ScanLoggerTests.swift`
- `DepthRunnerTests.swift`
- `RoomGeometryBuilderTests.swift`
- `PointCloudAccumulatorTests.swift`
- `KeyframeSelectorTests.swift`

**Bundle resources:**
- `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/DepthAnythingV2Small.mlpackage` — Core ML depth model (~20MB)

**Modified:**
- `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj/project.pbxproj` — only for the Core ML model (bundle resource). Source files auto-discovered via synchronized file groups; no pbxproj edit needed for `.swift` files.

## IMPORTANT: Xcode Synchronized File Groups

This project uses `PBXFileSystemSynchronizedRootGroup` in `project.pbxproj` (see lines 134-153). This means **Xcode auto-discovers all `.swift` files** in `decroom_no_lidar/`, `decroom_no_lidarTests/`, and `decroom_no_lidarUITests/` automatically.

**Do NOT manually edit `project.pbxproj` for new `.swift` files.** Just create the file at the right path and it's immediately part of the build. The plan's "Step 4: Register files in Xcode project" instructions below are OBSOLETE for `.swift` files — skip them. Only the Core ML model (`.mlpackage` bundle resource in Task 3) still requires Xcode's UI to add.

(The legacy `PBXBuildFile` / `PBXFileReference` entries you see in pbxproj are stale, in a "Recovered References" group not wired into any target.)

---

## Pre-Flight Check

Before starting, verify the environment is ready.

- [ ] **Confirm Xcode simulators exist**

Run:
```bash
xcrun simctl list devices available | grep "iPhone 17"
```

Expected: at least one `iPhone 17 Pro` line with `(Booted)` or `(Shutdown)`. If unavailable, substitute any available iPhone simulator in subsequent `xcodebuild` commands (e.g., `iPhone 16 Pro`, `iPhone 15 Pro`). The test code itself is simulator-agnostic.

- [ ] **Verify test target runs baseline test**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/decroom_no_lidarTests/example 2>&1 | tail -20
```

Expected: `** TEST SUCCEEDED **`. If it fails, fix before continuing.

---

## Task 1: DeviceCapabilities

**Purpose:** Determine which capture path to use (`.full` / `.reduced` / `.fallback`) based on device chip generation and ARKit availability.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/DeviceCapabilities.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/DeviceCapabilitiesTests.swift`
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj/project.pbxproj`

- [ ] **Step 1: Write the failing test**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/DeviceCapabilitiesTests.swift`:

```swift
import Testing
@testable import decroom_no_lidar

struct DeviceCapabilitiesTests {

    @Test func tierForA17ProIsFull() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .a17pro)
        #expect(tier == .full)
    }

    @Test func tierForA15IsFull() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .a15)
        #expect(tier == .full)
    }

    @Test func tierForA14IsFull() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .a14)
        #expect(tier == .full)
    }

    @Test func tierForA12IsReduced() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .a12)
        #expect(tier == .reduced)
    }

    @Test func tierForA13IsReduced() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .a13)
        #expect(tier == .reduced)
    }

    @Test func tierForA11IsFallback() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .a11)
        #expect(tier == .fallback)
    }

    @Test func tierForUnknownIsFallback() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .unknown)
        #expect(tier == .fallback)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/DeviceCapabilitiesTests 2>&1 | tail -30
```

Expected: compile failure with "cannot find 'DeviceCapabilities' in scope".

- [ ] **Step 3: Create DeviceCapabilities.swift**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/DeviceCapabilities.swift`:

```swift
//
//  DeviceCapabilities.swift
//  decroom_no_lidar
//
//  Detects device tier (.full / .reduced / .fallback) based on chip generation
//  and ARKit availability. Used by CaptureFlowView to choose between
//  phone-first AR capture and the legacy backend-heavy pipeline.
//

import Foundation
import ARKit

enum DeviceTier: String, Codable {
    case full       // A14+ (iPhone 12 and later): full on-device depth + AR capture
    case reduced    // A12-A13 (iPhone XR/XS/11): on-device depth at lower fps
    case fallback   // Older devices: use existing backend-heavy pipeline
}

enum ChipGeneration: String {
    case a11, a12, a13, a14, a15, a16, a17pro, a18, a18pro
    case unknown
}

enum DeviceCapabilities {

    /// Map chip generation to device tier. Pure function, testable.
    static func tier(forChipGeneration chip: ChipGeneration) -> DeviceTier {
        switch chip {
        case .a14, .a15, .a16, .a17pro, .a18, .a18pro:
            return .full
        case .a12, .a13:
            return .reduced
        case .a11, .unknown:
            return .fallback
        }
    }

    /// Detect current device chip generation from machine identifier.
    static func currentChipGeneration() -> ChipGeneration {
        var systemInfo = utsname()
        uname(&systemInfo)
        let identifier = withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0) ?? ""
            }
        }
        return chipGeneration(forMachineIdentifier: identifier)
    }

    /// Map machine identifier (e.g., "iPhone16,1") to chip generation.
    /// Exposed for testing.
    static func chipGeneration(forMachineIdentifier id: String) -> ChipGeneration {
        // iPhone identifiers: https://gist.github.com/adamawolf/3048717
        switch id {
        case "iPhone10,1", "iPhone10,2", "iPhone10,3", "iPhone10,4", "iPhone10,5", "iPhone10,6":
            return .a11  // iPhone 8, 8 Plus, X
        case "iPhone11,2", "iPhone11,4", "iPhone11,6", "iPhone11,8":
            return .a12  // iPhone XS, XS Max, XR
        case "iPhone12,1", "iPhone12,3", "iPhone12,5", "iPhone12,8":
            return .a13  // iPhone 11, 11 Pro, 11 Pro Max, SE 2nd gen
        case "iPhone13,1", "iPhone13,2", "iPhone13,3", "iPhone13,4":
            return .a14  // iPhone 12 mini, 12, 12 Pro, 12 Pro Max
        case "iPhone14,2", "iPhone14,3", "iPhone14,4", "iPhone14,5", "iPhone14,6", "iPhone14,7", "iPhone14,8":
            return .a15  // iPhone 13 series, SE 3rd gen, 14, 14 Plus
        case "iPhone15,2", "iPhone15,3", "iPhone15,4", "iPhone15,5":
            return .a16  // iPhone 14 Pro, 14 Pro Max, 15, 15 Plus
        case "iPhone16,1", "iPhone16,2":
            return .a17pro  // iPhone 15 Pro, 15 Pro Max
        case "iPhone17,1", "iPhone17,2":
            return .a18pro  // iPhone 16 Pro, 16 Pro Max
        case "iPhone17,3", "iPhone17,4":
            return .a18  // iPhone 16, 16 Plus
        default:
            return .unknown
        }
    }

    /// Current device's tier.
    static var currentTier: DeviceTier {
        // ARKit world tracking must be supported for any AR path.
        guard ARWorldTrackingConfiguration.isSupported else {
            return .fallback
        }
        return tier(forChipGeneration: currentChipGeneration())
    }
}
```

- [ ] **Step 4: Register files in Xcode project**

Open `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj/project.pbxproj` in a text editor.

Find the existing line containing `FurnitureScanView.swift in Sources` (there are several). Use the pattern from that file as a template. Add four entries:

1. In the `PBXBuildFile section` (near line ~25), after the last `C201B113` entry, add:
```
		C201B114EAF54A0E000024 /* decroom_no_lidar/DeviceCapabilities.swift in Sources */ = {isa = PBXBuildFile; fileRef = C201A114EAF54A0E000014 /* decroom_no_lidar/DeviceCapabilities.swift */; };
		C201B115EAF54A0E000025 /* decroom_no_lidar/DeviceCapabilitiesTests.swift in Sources */ = {isa = PBXBuildFile; fileRef = C201A115EAF54A0E000015 /* decroom_no_lidar/DeviceCapabilitiesTests.swift */; };
```

2. In the `PBXFileReference section` (near line ~80), after the last `C201A113` entry, add:
```
		C201A114EAF54A0E000014 /* decroom_no_lidar/DeviceCapabilities.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = decroom_no_lidar/DeviceCapabilities.swift; sourceTree = "<group>"; };
		C201A115EAF54A0E000015 /* decroom_no_lidar/DeviceCapabilitiesTests.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = decroom_no_lidarTests/DeviceCapabilitiesTests.swift; sourceTree = "<group>"; };
```

3. In the group children list (near line ~128), add `DeviceCapabilities.swift,` alphabetically between `CoverageEngine.swift,` and `FurnitureScanView.swift,`.

4. In the second file list (near line ~220), add after `C201A113EAF54A0E000013 /* decroom_no_lidar/FurnitureScanView.swift */,`:
```
				C201A114EAF54A0E000014 /* decroom_no_lidar/DeviceCapabilities.swift */,
```

5. In the main target's Sources build phase (near line ~412), add after `C201B113EAF54A0E000023 /* decroom_no_lidar/FurnitureScanView.swift in Sources */,`:
```
				C201B114EAF54A0E000024 /* decroom_no_lidar/DeviceCapabilities.swift in Sources */,
```

6. Find the test target's Sources build phase — search for `821EA13A` area (tests target). Add the test file to that target's Sources list:
```
				C201B115EAF54A0E000025 /* decroom_no_lidar/DeviceCapabilitiesTests.swift in Sources */,
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/DeviceCapabilitiesTests 2>&1 | tail -20
```

Expected: `** TEST SUCCEEDED **` with 7 tests passing.

- [ ] **Step 6: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/DeviceCapabilities.swift decroom_no_lidar/decroom_no_lidarTests/DeviceCapabilitiesTests.swift decroom_no_lidar/decroom_no_lidar.xcodeproj/project.pbxproj
git commit -m "feat: add DeviceCapabilities tier detection

Detects device tier (.full/.reduced/.fallback) based on chip generation
and ARKit availability. Used to choose between phone-first capture path
and the existing backend-heavy fallback pipeline."
```

---

## Task 2: ScanLogger

**Purpose:** Write per-scan JSON log with events, timing, device info, and diagnostics. Retain last 50 logs.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ScanLogger.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/ScanLoggerTests.swift`
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj/project.pbxproj`

- [ ] **Step 1: Write the failing test**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/ScanLoggerTests.swift`:

```swift
import Testing
import Foundation
@testable import decroom_no_lidar

struct ScanLoggerTests {

    @Test func newLoggerHasScanId() {
        let logger = ScanLogger(scanId: "test-123", directory: tempDir())
        #expect(logger.scanId == "test-123")
    }

    @Test func logEventAppendsToEvents() {
        let logger = ScanLogger(scanId: "test-123", directory: tempDir())
        logger.logEvent("scan_started", at: 0.0)
        logger.logEvent("first_plane_detected", at: 2.1, data: ["type": "wall"])
        #expect(logger.events.count == 2)
        #expect(logger.events[0].event == "scan_started")
        #expect(logger.events[1].data?["type"] == "wall")
    }

    @Test func finalizeWritesJSONFile() throws {
        let dir = tempDir()
        let logger = ScanLogger(scanId: "test-123", directory: dir)
        logger.recordDevice(DeviceInfo(model: "iPhone 15 Pro", ios: "17.2", chip: "A17 Pro", tier: .full))
        logger.logEvent("scan_started", at: 0.0)
        try logger.finalize()

        let files = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        #expect(files.contains { $0.contains("test-123") && $0.hasSuffix(".json") })
    }

    @Test func retentionDropsOldFilesAboveLimit() throws {
        let dir = tempDir()
        // Create 52 fake log files
        for i in 0..<52 {
            let timestamp = Int(Date().timeIntervalSince1970) - (52 - i) * 100
            let url = dir.appendingPathComponent("\(timestamp)_log\(i).json")
            try "{}".data(using: .utf8)!.write(to: url)
        }
        try ScanLogger.enforceRetention(directory: dir, keepLast: 50)
        let remaining = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        #expect(remaining.count == 50)
    }

    // Helper: create a unique temp directory per test
    private func tempDir() -> URL {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan_logger_test_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/ScanLoggerTests 2>&1 | tail -30
```

Expected: compile failure with "cannot find 'ScanLogger' in scope".

- [ ] **Step 3: Create ScanLogger.swift**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ScanLogger.swift`:

```swift
//
//  ScanLogger.swift
//  decroom_no_lidar
//
//  Per-scan JSON log writer. Captures events, timing, device info, and
//  diagnostics for post-mortem analysis. Retains last 50 logs in
//  Documents/scan_logs/.
//

import Foundation

struct DeviceInfo: Codable {
    let model: String
    let ios: String
    let chip: String
    let tier: DeviceTier
}

struct ScanEvent: Codable {
    let t: Double           // Seconds since scan start
    let event: String
    let data: [String: String]?
}

final class ScanLogger {
    let scanId: String
    let startedAt: Date
    private let directory: URL

    private(set) var events: [ScanEvent] = []
    private(set) var device: DeviceInfo?

    init(scanId: String = UUID().uuidString.prefix(8).lowercased(),
         directory: URL = ScanLogger.defaultDirectory()) {
        self.scanId = String(scanId)
        self.startedAt = Date()
        self.directory = directory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    func recordDevice(_ info: DeviceInfo) {
        self.device = info
    }

    func logEvent(_ event: String, at t: Double, data: [String: String]? = nil) {
        events.append(ScanEvent(t: t, event: event, data: data))
    }

    /// Write log to disk. Call once when scan + upload are done.
    func finalize() throws {
        let timestamp = Int(startedAt.timeIntervalSince1970)
        let filename = "\(timestamp)_\(scanId).json"
        let url = directory.appendingPathComponent(filename)

        let payload: [String: Any] = [
            "scan_id": scanId,
            "started_at": ISO8601DateFormatter().string(from: startedAt),
            "completed_at": ISO8601DateFormatter().string(from: Date()),
            "device": device.map { [
                "model": $0.model,
                "ios": $0.ios,
                "chip": $0.chip,
                "tier": $0.tier.rawValue
            ] } as Any? ?? NSNull(),
            "events": events.map { [
                "t": $0.t,
                "event": $0.event,
                "data": $0.data as Any? ?? NSNull()
            ] }
        ]

        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    /// Keep only the `keepLast` most recent log files in the directory.
    static func enforceRetention(directory: URL, keepLast: Int) throws {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(at: directory,
                                                   includingPropertiesForKeys: [.creationDateKey],
                                                   options: [])
            .filter { $0.pathExtension == "json" }

        guard contents.count > keepLast else { return }

        let sortedByDate = contents.sorted { a, b in
            let aDate = (try? a.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            let bDate = (try? b.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            return aDate > bDate  // Newest first
        }

        for url in sortedByDate.dropFirst(keepLast) {
            try? fm.removeItem(at: url)
        }
    }

    static func defaultDirectory() -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("scan_logs")
    }
}
```

- [ ] **Step 4: Register files in Xcode project**

Follow the same pattern as Task 1 Step 4, but for:
- `ScanLogger.swift` (main target)
- `ScanLoggerTests.swift` (test target)

Use IDs `C201B116/C201A116` and `C201B117/C201A117` to avoid collisions.

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/ScanLoggerTests 2>&1 | tail -20
```

Expected: `** TEST SUCCEEDED **` with 4 tests passing.

- [ ] **Step 6: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/ScanLogger.swift decroom_no_lidar/decroom_no_lidarTests/ScanLoggerTests.swift decroom_no_lidar/decroom_no_lidar.xcodeproj/project.pbxproj
git commit -m "feat: add ScanLogger for per-scan JSON diagnostics

Writes per-scan logs to Documents/scan_logs/ with device info, events,
and timing. Retention keeps last 50 logs, older ones auto-deleted."
```

---

## Task 3: Integrate Core ML Depth Model

**Purpose:** Add a Core ML depth model to the app bundle. Model choice: **Depth Anything V2 Metric Indoor Small** (~55MB), which provides metric depth without the 400MB cost of Depth Pro.

**Files:**
- Obtain: `DepthAnythingV2MetricIndoorSmall.mlpackage`
- Add to Xcode project as bundle resource
- Modify: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj/project.pbxproj`

- [ ] **Step 1: Download the Core ML model**

The community has Core ML exports on Hugging Face. Run:

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar
curl -L -o DepthModel.mlpackage.zip \
  "https://huggingface.co/apple/coreml-depth-anything-v2-small/resolve/main/DepthAnythingV2SmallF16.mlpackage.zip"
unzip DepthModel.mlpackage.zip
# The zip may contain a differently-named directory — rename if needed:
if [ ! -d DepthModel.mlpackage ]; then
    mv DepthAnythingV2SmallF16.mlpackage DepthModel.mlpackage 2>/dev/null || true
fi
rm DepthModel.mlpackage.zip
ls -la DepthModel.mlpackage
```

Expected: a directory `DepthModel.mlpackage/` containing `Manifest.json` and `Data/` subdirectory.

If that URL is unavailable, convert from PyTorch using Apple's `coremltools`:
```bash
pip install coremltools torch
# See https://github.com/apple/ml-depth-pro for conversion instructions
```
Then rename the output to `DepthModel.mlpackage`.

**Important**: the file MUST be named `DepthModel.mlpackage` (case-sensitive). The `DepthRunner.swift` in Task 4 looks up `DepthModel` specifically. If you need a different name, update the string in `DepthRunner.init()` to match.

- [ ] **Step 2: Add model to Xcode project**

Open `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj` in Xcode.

Drag `DepthModel.mlpackage` into the project navigator, drop into the `decroom_no_lidar` folder group. In the dialog:
- Copy items if needed: unchecked (file is already in place)
- Add to targets: check `decroom_no_lidar` only (not the test target)

Xcode compiles `.mlpackage` → `.mlmodelc` at build time; the app bundle contains the `.mlmodelc` version.

- [ ] **Step 3: Verify with a build**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' 2>&1 | tail -10
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 4: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/DepthModel.mlpackage decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj/project.pbxproj
git commit -m "feat: add Depth Anything V2 Metric Core ML model as DepthModel

Bundles the ~55MB metric depth model (renamed to DepthModel.mlpackage
for consistent loading) for on-device inference. More practical than
Depth Pro (~400MB) while still producing metric depth."
```

---

## Task 4: DepthRunner

**Purpose:** Wrap Core ML depth inference with a clean API: `CVPixelBuffer` in, metric depth map out. Handle thermal throttling.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/DepthRunner.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/DepthRunnerTests.swift`
- Modify: project.pbxproj

- [ ] **Step 1: Write the failing test**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/DepthRunnerTests.swift`:

```swift
import Testing
import CoreVideo
@testable import decroom_no_lidar

struct DepthRunnerTests {

    @Test func inferReturnsDepthMapWithConsistentDimensions() throws {
        let runner = try DepthRunner()
        let buffer = makeTestPixelBuffer(width: 518, height: 518)
        let result = try runner.infer(pixelBuffer: buffer)
        // Dimensions must be positive and match the buffer count exactly.
        #expect(result.width > 0)
        #expect(result.height > 0)
        #expect(result.depths.count == result.width * result.height)
    }

    @Test func depthsAreFiniteAndInReasonableRange() throws {
        let runner = try DepthRunner()
        let buffer = makeTestPixelBuffer(width: 518, height: 518)
        let result = try runner.infer(pixelBuffer: buffer)
        // All depths should be finite and bounded.
        let allFinite = result.depths.allSatisfy { $0.isFinite && $0 >= 0 && $0 < 100 }
        #expect(allFinite)
    }

    @Test func thermalThrottleDropsFrameRate() {
        let runner = try? DepthRunner()
        guard let runner else { return }
        // Simulate thermal state transitions.
        runner.updateThermalState(.serious)
        #expect(runner.targetFrameInterval >= 0.5)
        runner.updateThermalState(.nominal)
        #expect(runner.targetFrameInterval <= 0.3)
    }

    // Helper: create a solid-color CVPixelBuffer for testing
    private func makeTestPixelBuffer(width: Int, height: Int) -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                            kCVPixelFormatType_32BGRA, attrs, &buffer)
        return buffer!
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/DepthRunnerTests 2>&1 | tail -20
```

Expected: compile failure with "cannot find 'DepthRunner' in scope".

- [ ] **Step 3: Create DepthRunner.swift**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/DepthRunner.swift`:

```swift
//
//  DepthRunner.swift
//  decroom_no_lidar
//
//  Wraps the Depth Anything V2 Metric Core ML model for on-device depth inference.
//  Accepts a CVPixelBuffer, returns a metric depth map (meters).
//  Adjusts inference rate based on thermal state to avoid throttling.
//

import Foundation
import CoreML
import CoreVideo
import Vision

struct DepthResult {
    let width: Int
    let height: Int
    let depths: [Float]  // Row-major, width*height floats in meters
}

enum DepthRunnerError: Error {
    case modelLoadFailed
    case inferenceFailed
    case unexpectedOutput
}

final class DepthRunner {
    private let model: MLModel

    /// Target time between inference calls (seconds). Adjusted by thermal state.
    private(set) var targetFrameInterval: TimeInterval = 0.2  // 5fps default

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Neural Engine if available
        // NOTE: Xcode compiles .mlpackage → .mlmodelc in the app bundle.
        // Resource name must match what you named the file in Task 3.
        guard let url = Bundle.main.url(forResource: "DepthModel", withExtension: "mlmodelc"),
              let loaded = try? MLModel(contentsOf: url, configuration: config) else {
            throw DepthRunnerError.modelLoadFailed
        }
        self.model = loaded
    }

    /// Run inference on a pixel buffer. Returns metric depth map at the model's native resolution.
    func infer(pixelBuffer: CVPixelBuffer) throws -> DepthResult {
        // Most Core ML depth models name their input "image" or "pixel_values".
        // If your model uses a different name, update this dictionary.
        let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        let input = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
        let output = try model.prediction(from: input)

        // Find the depth output — typically the only MultiArray output.
        guard let depthName = output.featureNames.first(where: {
                  output.featureValue(for: $0)?.multiArrayValue != nil
              }),
              let multiArray = output.featureValue(for: depthName)?.multiArrayValue else {
            throw DepthRunnerError.unexpectedOutput
        }

        // MultiArray shape is typically [1, 1, H, W] or [1, H, W]
        let shape = multiArray.shape.map { $0.intValue }
        let height: Int
        let width: Int
        switch shape.count {
        case 4: height = shape[2]; width = shape[3]
        case 3: height = shape[1]; width = shape[2]
        case 2: height = shape[0]; width = shape[1]
        default: throw DepthRunnerError.unexpectedOutput
        }

        let count = multiArray.count
        var depths = [Float](repeating: 0, count: count)
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            depths[i] = ptr[i]
        }

        return DepthResult(width: width, height: height, depths: depths)
    }

    /// Adjust inference rate based on thermal state. Call when thermal state changes.
    func updateThermalState(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal:
            targetFrameInterval = 0.2   // 5fps
        case .fair:
            targetFrameInterval = 0.3   // ~3fps
        case .serious:
            targetFrameInterval = 0.5   // 2fps
        case .critical:
            targetFrameInterval = 1.0   // 1fps
        @unknown default:
            targetFrameInterval = 0.3
        }
    }
}
```

- [ ] **Step 4: Register files in Xcode project**

Follow Task 1 Step 4 pattern. Use IDs `C201B118/C201A118` for `DepthRunner.swift` and `C201B119/C201A119` for `DepthRunnerTests.swift`.

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/DepthRunnerTests 2>&1 | tail -20
```

Expected: `** TEST SUCCEEDED **` with 3 tests passing. If inference fails because the simulator doesn't have a GPU, the test for `inferReturnsDepthMapOfExpectedShape` may fail — in that case, run on a physical device:

```bash
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS,name=<Your iPhone Name>' -only-testing:decroom_no_lidarTests/DepthRunnerTests
```

- [ ] **Step 6: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/DepthRunner.swift decroom_no_lidar/decroom_no_lidarTests/DepthRunnerTests.swift decroom_no_lidar/decroom_no_lidar.xcodeproj/project.pbxproj
git commit -m "feat: add DepthRunner for on-device metric depth

Wraps Depth Anything V2 Metric Core ML inference. Accepts CVPixelBuffer,
returns metric depth map. Adjusts target frame rate based on thermal state."
```

---

## Task 5: PlaneAnchorDescriptor + RoomGeometryBuilder

**Purpose:** Abstract `ARPlaneAnchor` behind a protocol so tests can run without a real ARKit session. Build wall segments from plane anchors with Manhattan frame snapping.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/PlaneAnchorDescriptor.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/RoomGeometryBuilder.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/RoomGeometryBuilderTests.swift`
- Modify: project.pbxproj

- [ ] **Step 1: Create PlaneAnchorDescriptor protocol**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/PlaneAnchorDescriptor.swift`:

```swift
//
//  PlaneAnchorDescriptor.swift
//  decroom_no_lidar
//
//  Protocol abstraction over ARPlaneAnchor so RoomGeometryBuilder can be
//  unit-tested without a live ARKit session.
//

import Foundation
import ARKit
import simd

enum PlaneAlignment {
    case horizontal  // floor or ceiling
    case vertical    // wall
}

protocol PlaneAnchorDescriptor {
    var identifier: UUID { get }
    var alignment: PlaneAlignment { get }
    var transform: simd_float4x4 { get }  // anchor-to-world
    var center: SIMD3<Float> { get }       // local to anchor
    var extent: SIMD3<Float> { get }       // local extents
}

/// Bridge real ARPlaneAnchor to the protocol.
extension ARPlaneAnchor: PlaneAnchorDescriptor {
    var alignment: PlaneAlignment {
        switch self.alignment {
        case .horizontal: return .horizontal
        case .vertical: return .vertical
        @unknown default: return .horizontal
        }
    }
}
```

- [ ] **Step 2: Write failing test for RoomGeometryBuilder**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/RoomGeometryBuilderTests.swift`:

```swift
import Testing
import simd
@testable import decroom_no_lidar

struct RoomGeometryBuilderTests {

    @Test func emptyInputProducesEmptyGeometry() {
        let builder = RoomGeometryBuilder()
        let geometry = builder.build(from: [])
        #expect(geometry.walls.isEmpty)
    }

    @Test func singleHorizontalPlaneSetsFloorHeight() {
        let floor = MockPlane(id: UUID(), alignment: .horizontal,
                              worldY: 0.0, extents: SIMD3(3, 0, 3))
        let builder = RoomGeometryBuilder()
        let geometry = builder.build(from: [floor])
        #expect(abs(geometry.floorHeight) < 0.01)
        #expect(geometry.walls.isEmpty)  // One horizontal doesn't give walls
    }

    @Test func fourVerticalPlanesYieldFourWalls() {
        // A 3m x 3m room: 4 walls at x=-1.5, x=1.5, z=-1.5, z=1.5, floor at y=0
        let floor = MockPlane(id: UUID(), alignment: .horizontal,
                              worldY: 0.0, extents: SIMD3(3, 0, 3))
        let wallNorth = MockWall(center: SIMD3(0, 1.25, -1.5),
                                  normal: SIMD3(0, 0, 1),
                                  width: 3.0, height: 2.5)
        let wallSouth = MockWall(center: SIMD3(0, 1.25, 1.5),
                                  normal: SIMD3(0, 0, -1),
                                  width: 3.0, height: 2.5)
        let wallEast = MockWall(center: SIMD3(1.5, 1.25, 0),
                                 normal: SIMD3(-1, 0, 0),
                                 width: 3.0, height: 2.5)
        let wallWest = MockWall(center: SIMD3(-1.5, 1.25, 0),
                                 normal: SIMD3(1, 0, 0),
                                 width: 3.0, height: 2.5)

        let builder = RoomGeometryBuilder()
        let geometry = builder.build(from: [floor, wallNorth, wallSouth, wallEast, wallWest])

        #expect(geometry.walls.count == 4)
        #expect(abs(geometry.floorHeight) < 0.01)
    }

    @Test func manhattanYawSnapsToNearestAxis() {
        // Wall with normal slightly off-axis (3° rotation) should snap to clean axis
        let rotated = simd_quatf(angle: .pi / 60, axis: SIMD3(0, 1, 0))  // 3°
        let normal = rotated.act(SIMD3<Float>(0, 0, 1))
        let wall = MockWall(center: SIMD3(0, 1.25, -1.5),
                             normal: normal,
                             width: 3.0, height: 2.5)
        let floor = MockPlane(id: UUID(), alignment: .horizontal,
                              worldY: 0.0, extents: SIMD3(3, 0, 3))

        let builder = RoomGeometryBuilder()
        let geometry = builder.build(from: [floor, wall])

        // Should snap to Manhattan frame — yaw near 0 or pi/2
        let yaw = geometry.manhattanYaw
        let snapTolerance: Float = 0.05
        let snappedToZero = abs(yaw) < snapTolerance
        let snappedToPiOver2 = abs(abs(yaw) - .pi / 2) < snapTolerance
        #expect(snappedToZero || snappedToPiOver2)
    }
}

// MARK: - Test Doubles

struct MockPlane: PlaneAnchorDescriptor {
    let identifier: UUID
    let alignment: PlaneAlignment
    let worldY: Float
    let extents: SIMD3<Float>

    var transform: simd_float4x4 {
        var t = matrix_identity_float4x4
        t.columns.3.y = worldY
        return t
    }
    var center: SIMD3<Float> { .zero }
    var extent: SIMD3<Float> { extents }
}

struct MockWall: PlaneAnchorDescriptor {
    let identifier = UUID()
    let alignment: PlaneAlignment = .vertical
    let center: SIMD3<Float>
    let normal: SIMD3<Float>
    let width: Float
    let height: Float

    var transform: simd_float4x4 {
        // Build a transform where +Z is the wall normal direction
        let up = SIMD3<Float>(0, 1, 0)
        let right = simd_normalize(simd_cross(up, normal))
        let trueUp = simd_cross(normal, right)
        var t = matrix_identity_float4x4
        t.columns.0 = SIMD4(right, 0)
        t.columns.1 = SIMD4(trueUp, 0)
        t.columns.2 = SIMD4(normal, 0)
        t.columns.3 = SIMD4(center, 1)
        return t
    }
    var extent: SIMD3<Float> { SIMD3(width, height, 0) }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/RoomGeometryBuilderTests 2>&1 | tail -20
```

Expected: compile failure with "cannot find 'RoomGeometryBuilder' in scope".

- [ ] **Step 4: Create RoomGeometryBuilder.swift**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/RoomGeometryBuilder.swift`:

```swift
//
//  RoomGeometryBuilder.swift
//  decroom_no_lidar
//
//  Converts ARKit plane anchors into room geometry: walls + floor height
//  + Manhattan yaw. Snaps plane normals to the nearest orthogonal axis
//  (Manhattan world assumption) for clean wall alignment.
//

import Foundation
import simd

struct WallSegment {
    let start: SIMD2<Float>  // x, z in world space
    let end: SIMD2<Float>
    let height: Float
}

struct RoomGeometry {
    let walls: [WallSegment]
    let floorHeight: Float    // y coordinate of floor plane
    let ceilingHeight: Float  // y coordinate of ceiling (0 if unknown)
    let manhattanYaw: Float   // room rotation around Y axis (radians)
}

final class RoomGeometryBuilder {

    /// Build room geometry from a set of plane anchors.
    func build(from planes: [PlaneAnchorDescriptor]) -> RoomGeometry {
        let horizontals = planes.filter { $0.alignment == .horizontal }
        let verticals = planes.filter { $0.alignment == .vertical }

        let floorHeight = floorY(from: horizontals)
        let ceilingHeight = ceilingY(from: horizontals)
        let yaw = manhattanYaw(from: verticals)
        let walls = wallSegments(from: verticals, yaw: yaw, floorY: floorHeight)

        return RoomGeometry(walls: walls,
                            floorHeight: floorHeight,
                            ceilingHeight: ceilingHeight,
                            manhattanYaw: yaw)
    }

    // MARK: - Floor / Ceiling

    private func floorY(from horizontals: [PlaneAnchorDescriptor]) -> Float {
        // Floor = lowest horizontal plane
        let ys = horizontals.map { $0.transform.columns.3.y }
        return ys.min() ?? 0
    }

    private func ceilingY(from horizontals: [PlaneAnchorDescriptor]) -> Float {
        let ys = horizontals.map { $0.transform.columns.3.y }
        guard let max = ys.max(), let min = ys.min(), max - min > 1.5 else {
            return 0  // Not enough vertical span to identify ceiling
        }
        return max
    }

    // MARK: - Manhattan Yaw

    /// Cluster wall normals and return the dominant yaw (0 or pi/2 equivalent).
    private func manhattanYaw(from verticals: [PlaneAnchorDescriptor]) -> Float {
        guard !verticals.isEmpty else { return 0 }

        var sumSin: Float = 0
        var sumCos: Float = 0
        var totalWeight: Float = 0

        for plane in verticals {
            let normalWorld = wallNormal(from: plane)
            // Project onto XZ plane
            let nxz = SIMD2<Float>(normalWorld.x, normalWorld.z)
            let length = simd_length(nxz)
            guard length > 0.1 else { continue }
            let n = nxz / length

            // Wall yaw is atan2(nx, nz). Use 2*yaw to fold opposite normals together.
            let yaw = atan2(n.x, n.y)
            let weight = max(plane.extent.x, plane.extent.y)  // larger walls vote more
            sumSin += weight * sin(2 * yaw)
            sumCos += weight * cos(2 * yaw)
            totalWeight += weight
        }

        guard totalWeight > 0 else { return 0 }
        let dominantYaw = 0.5 * atan2(sumSin, sumCos)

        // Snap to nearest 90° increment if close
        let snapOptions: [Float] = [-Float.pi / 2, 0, Float.pi / 2]
        for option in snapOptions {
            if abs(dominantYaw - option) < 0.1 { return option }
        }
        return dominantYaw
    }

    private func wallNormal(from plane: PlaneAnchorDescriptor) -> SIMD3<Float> {
        // Wall plane's +Z in local frame is the normal
        let col2 = plane.transform.columns.2
        return SIMD3(col2.x, col2.y, col2.z)
    }

    // MARK: - Walls

    private func wallSegments(from verticals: [PlaneAnchorDescriptor],
                               yaw: Float,
                               floorY: Float) -> [WallSegment] {
        verticals.map { plane in
            wallSegment(from: plane, floorY: floorY)
        }
    }

    private func wallSegment(from plane: PlaneAnchorDescriptor,
                              floorY: Float) -> WallSegment {
        let transform = plane.transform
        let center = SIMD3<Float>(transform.columns.3.x,
                                    transform.columns.3.y,
                                    transform.columns.3.z)

        // Wall extends along its local +X direction
        let rightWorld = SIMD3<Float>(transform.columns.0.x,
                                        transform.columns.0.y,
                                        transform.columns.0.z)
        let halfWidth = plane.extent.x / 2
        let start3D = center - rightWorld * halfWidth
        let end3D = center + rightWorld * halfWidth

        return WallSegment(start: SIMD2(start3D.x, start3D.z),
                           end: SIMD2(end3D.x, end3D.z),
                           height: plane.extent.y)
    }
}
```

- [ ] **Step 5: Register files in Xcode project**

Follow Task 1 Step 4 pattern. IDs:
- `PlaneAnchorDescriptor.swift` → `C201B11A/C201A11A`
- `RoomGeometryBuilder.swift` → `C201B11B/C201A11B`
- `RoomGeometryBuilderTests.swift` → `C201B11C/C201A11C`

- [ ] **Step 6: Run test to verify it passes**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/RoomGeometryBuilderTests 2>&1 | tail -20
```

Expected: `** TEST SUCCEEDED **` with 4 tests passing.

- [ ] **Step 7: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/PlaneAnchorDescriptor.swift decroom_no_lidar/decroom_no_lidar/RoomGeometryBuilder.swift decroom_no_lidar/decroom_no_lidarTests/RoomGeometryBuilderTests.swift decroom_no_lidar/decroom_no_lidar.xcodeproj/project.pbxproj
git commit -m "feat: add RoomGeometryBuilder with Manhattan yaw snapping

Builds room geometry (walls + floor/ceiling heights + Manhattan yaw)
from ARKit plane anchors. Uses weighted circular mean for yaw with
snap to nearest 90 degree increment. Protocol abstraction over
ARPlaneAnchor enables unit testing without a live AR session."
```

---

## Task 6: PointCloudAccumulator

**Purpose:** Unproject depth pixels into world space, accumulate into a voxelized colored point cloud, export as binary PLY.

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/PointCloudAccumulator.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/PointCloudAccumulatorTests.swift`
- Modify: project.pbxproj

- [ ] **Step 1: Write the failing test**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/PointCloudAccumulatorTests.swift`:

```swift
import Testing
import simd
import Foundation
@testable import decroom_no_lidar

struct PointCloudAccumulatorTests {

    @Test func unprojectAtImageCenterReturnsPointOnCameraAxis() {
        // Intrinsics: fx=500, fy=500, cx=256, cy=256 (fake 512x512 cam)
        // Identity pose: camera at origin looking down -Z
        let intrinsics = Intrinsics(fx: 500, fy: 500, cx: 256, cy: 256, width: 512, height: 512)
        let pose = matrix_identity_float4x4
        let accumulator = PointCloudAccumulator(voxelSize: 0.02, maxPoints: 10_000)
        let pt = accumulator.unproject(u: 256, v: 256, depth: 2.0, intrinsics: intrinsics, worldFromCamera: pose)
        // Camera looks down -Z, so depth 2.0 at center should give (0, 0, -2)
        #expect(abs(pt.x) < 1e-4)
        #expect(abs(pt.y) < 1e-4)
        #expect(abs(pt.z + 2.0) < 1e-4)
    }

    @Test func addPointsVoxelizesDuplicates() {
        let accumulator = PointCloudAccumulator(voxelSize: 0.02, maxPoints: 10_000)
        // Add 100 points all within one voxel
        for _ in 0..<100 {
            accumulator.addPoint(SIMD3(0.001, 0.001, 0.001),
                                  color: SIMD3(128, 128, 128))
        }
        #expect(accumulator.count == 1)
    }

    @Test func addPointsRespectsMaxLimit() {
        let accumulator = PointCloudAccumulator(voxelSize: 0.001, maxPoints: 100)
        // Add 200 distinct points (each in its own voxel due to small voxel size)
        for i in 0..<200 {
            accumulator.addPoint(SIMD3(Float(i) * 0.01, 0, 0),
                                  color: SIMD3(128, 128, 128))
        }
        #expect(accumulator.count <= 100)
    }

    @Test func exportPLYProducesValidBinaryHeader() throws {
        let accumulator = PointCloudAccumulator(voxelSize: 0.02, maxPoints: 10_000)
        accumulator.addPoint(SIMD3(1, 2, 3), color: SIMD3(255, 0, 0))
        let data = accumulator.exportPLY()

        // Read header
        let headerRange = data.range(of: "end_header\n".data(using: .ascii)!)
        #expect(headerRange != nil)
        let headerStr = String(data: data[..<headerRange!.upperBound], encoding: .ascii)!
        #expect(headerStr.contains("ply"))
        #expect(headerStr.contains("format binary_little_endian 1.0"))
        #expect(headerStr.contains("element vertex 1"))
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/PointCloudAccumulatorTests 2>&1 | tail -20
```

Expected: compile failure with "cannot find 'PointCloudAccumulator' in scope" and "cannot find 'Intrinsics' in scope".

- [ ] **Step 3: Create PointCloudAccumulator.swift**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/PointCloudAccumulator.swift`:

```swift
//
//  PointCloudAccumulator.swift
//  decroom_no_lidar
//
//  Accumulates world-space 3D points from unprojected depth maps.
//  Voxel-deduplicates at configurable grid resolution. Caps total
//  points to bound memory. Exports as binary PLY for backend upload.
//

import Foundation
import simd

struct Intrinsics {
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float
    let width: Int
    let height: Int
}

final class PointCloudAccumulator {
    private let voxelSize: Float
    private let maxPoints: Int

    /// Voxel key → (summed position, summed color, count)
    private var voxels: [Int64: (position: SIMD3<Double>, color: SIMD3<Double>, count: Int)] = [:]

    var count: Int { voxels.count }

    init(voxelSize: Float = 0.02, maxPoints: Int = 500_000) {
        self.voxelSize = voxelSize
        self.maxPoints = maxPoints
    }

    /// Unproject a depth pixel to world space.
    func unproject(u: Int, v: Int, depth: Float,
                    intrinsics i: Intrinsics,
                    worldFromCamera pose: simd_float4x4) -> SIMD3<Float> {
        let camX = (Float(u) - i.cx) * depth / i.fx
        let camY = (Float(v) - i.cy) * depth / i.fy
        // Camera-space Z is -depth (camera looks down -Z in ARKit convention)
        let camPoint = SIMD4<Float>(camX, camY, -depth, 1)
        let worldPoint = pose * camPoint
        return SIMD3(worldPoint.x, worldPoint.y, worldPoint.z)
    }

    /// Add a world-space point with its color.
    func addPoint(_ world: SIMD3<Float>, color: SIMD3<UInt8>) {
        let key = voxelKey(for: world)
        let pos = SIMD3<Double>(Double(world.x), Double(world.y), Double(world.z))
        let col = SIMD3<Double>(Double(color.x), Double(color.y), Double(color.z))
        if let existing = voxels[key] {
            voxels[key] = (existing.position + pos, existing.color + col, existing.count + 1)
        } else {
            if voxels.count >= maxPoints {
                // Drop a random voxel to make room. Simple eviction — could be improved.
                if let evictKey = voxels.keys.randomElement() {
                    voxels.removeValue(forKey: evictKey)
                }
            }
            voxels[key] = (pos, col, 1)
        }
    }

    /// Add a batch of points from a depth map.
    func addDepthMap(_ depths: [Float],
                     width: Int, height: Int,
                     intrinsics: Intrinsics,
                     worldFromCamera pose: simd_float4x4,
                     colorAt: (Int, Int) -> SIMD3<UInt8>,
                     pixelStride: Int = 4,
                     maxDepth: Float = 5.0) {
        var v = 0
        while v < height {
            var u = 0
            while u < width {
                let d = depths[v * width + u]
                if d.isFinite, d > 0.1, d < maxDepth {
                    let world = unproject(u: u, v: v, depth: d, intrinsics: intrinsics, worldFromCamera: pose)
                    addPoint(world, color: colorAt(u, v))
                }
                u += pixelStride
            }
            v += pixelStride
        }
    }

    /// Export as binary little-endian PLY with x,y,z,r,g,b per vertex.
    func exportPLY() -> Data {
        var output = Data()

        // Compute averaged points first
        let averaged: [(pos: SIMD3<Float>, col: SIMD3<UInt8>)] = voxels.values.map { entry in
            let n = Double(entry.count)
            let pos = SIMD3<Float>(Float(entry.position.x / n),
                                     Float(entry.position.y / n),
                                     Float(entry.position.z / n))
            let col = SIMD3<UInt8>(UInt8(min(255, Int(entry.color.x / n))),
                                     UInt8(min(255, Int(entry.color.y / n))),
                                     UInt8(min(255, Int(entry.color.z / n))))
            return (pos, col)
        }

        // ASCII header
        let header = """
        ply
        format binary_little_endian 1.0
        element vertex \(averaged.count)
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header

        """
        output.append(header.data(using: .ascii)!)

        // Binary payload
        for (pos, col) in averaged {
            var x = pos.x, y = pos.y, z = pos.z
            var r = col.x, g = col.y, b = col.z
            withUnsafeBytes(of: &x) { output.append(contentsOf: $0) }
            withUnsafeBytes(of: &y) { output.append(contentsOf: $0) }
            withUnsafeBytes(of: &z) { output.append(contentsOf: $0) }
            output.append(r)
            output.append(g)
            output.append(b)
        }

        return output
    }

    private func voxelKey(for p: SIMD3<Float>) -> Int64 {
        let vx = Int32(floor(p.x / voxelSize))
        let vy = Int32(floor(p.y / voxelSize))
        let vz = Int32(floor(p.z / voxelSize))
        // Pack three int32s into int64 (truncates high bits, acceptable for ±32K voxels)
        return (Int64(vx) & 0x1FFFFF) << 42 | (Int64(vy) & 0x1FFFFF) << 21 | (Int64(vz) & 0x1FFFFF)
    }
}
```

- [ ] **Step 4: Register files in Xcode project**

Follow Task 1 Step 4 pattern. IDs:
- `PointCloudAccumulator.swift` → `C201B11D/C201A11D`
- `PointCloudAccumulatorTests.swift` → `C201B11E/C201A11E`

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/PointCloudAccumulatorTests 2>&1 | tail -20
```

Expected: `** TEST SUCCEEDED **` with 4 tests passing.

- [ ] **Step 6: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/PointCloudAccumulator.swift decroom_no_lidar/decroom_no_lidarTests/PointCloudAccumulatorTests.swift decroom_no_lidar/decroom_no_lidar.xcodeproj/project.pbxproj
git commit -m "feat: add PointCloudAccumulator with voxel dedup + PLY export

Unprojects depth map pixels into world space, voxel-deduplicates at
2cm resolution, caps at 500K points for bounded memory. Exports as
binary PLY for upload to SpatialLM backend."
```

---

## Task 7: KeyframeSelector

**Purpose:** Select a small diverse subset of captured frames for backend upload. Prioritize spatial coverage (farthest-point sampling in pose space) and semantic coverage (best view of each detected object — semantic info will be filled in by Plan 2).

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/KeyframeSelector.swift`
- Create: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/KeyframeSelectorTests.swift`
- Modify: project.pbxproj

- [ ] **Step 1: Write the failing test**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidarTests/KeyframeSelectorTests.swift`:

```swift
import Testing
import simd
@testable import decroom_no_lidar

struct KeyframeSelectorTests {

    @Test func selectorReturnsRequestedCount() {
        var candidates: [KeyframeCandidate] = []
        for i in 0..<50 {
            let pose = poseAt(x: Float(i) * 0.1, yaw: Float(i) * 0.01)
            candidates.append(KeyframeCandidate(frameIndex: i,
                                                 pose: pose,
                                                 blurScore: 150,
                                                 featurePointCount: 20,
                                                 detectedObjectIds: []))
        }
        let selected = KeyframeSelector.select(candidates, maxCount: 10)
        #expect(selected.count == 10)
    }

    @Test func selectorDropsBlurryFrames() {
        var candidates: [KeyframeCandidate] = []
        for i in 0..<20 {
            let blur: Float = (i % 2 == 0) ? 50 : 200   // Half are too blurry
            let pose = poseAt(x: Float(i) * 0.1, yaw: 0)
            candidates.append(KeyframeCandidate(frameIndex: i,
                                                 pose: pose,
                                                 blurScore: blur,
                                                 featurePointCount: 20,
                                                 detectedObjectIds: []))
        }
        let selected = KeyframeSelector.select(candidates, maxCount: 10)
        for kf in selected {
            #expect(kf.blurScore >= 120)
        }
    }

    @Test func selectorFarthestPointSamplingPrefersDiversePoses() {
        var candidates: [KeyframeCandidate] = []
        // Create 10 frames clustered at origin, 2 frames far away
        for i in 0..<10 {
            let pose = poseAt(x: 0.01 * Float(i), yaw: 0)
            candidates.append(KeyframeCandidate(frameIndex: i, pose: pose,
                                                 blurScore: 150, featurePointCount: 20,
                                                 detectedObjectIds: []))
        }
        candidates.append(KeyframeCandidate(frameIndex: 10,
                                              pose: poseAt(x: 5.0, yaw: 0),
                                              blurScore: 150, featurePointCount: 20,
                                              detectedObjectIds: []))
        candidates.append(KeyframeCandidate(frameIndex: 11,
                                              pose: poseAt(x: -5.0, yaw: 0),
                                              blurScore: 150, featurePointCount: 20,
                                              detectedObjectIds: []))

        let selected = KeyframeSelector.select(candidates, maxCount: 3)
        let selectedIndices = selected.map { $0.frameIndex }
        #expect(selectedIndices.contains(10))
        #expect(selectedIndices.contains(11))
    }

    @Test func selectorIncludesBestFramePerObject() {
        let candidates: [KeyframeCandidate] = [
            KeyframeCandidate(frameIndex: 0, pose: poseAt(x: 0, yaw: 0),
                              blurScore: 150, featurePointCount: 20,
                              detectedObjectIds: ["bed_1": 0.5]),
            KeyframeCandidate(frameIndex: 1, pose: poseAt(x: 0.1, yaw: 0),
                              blurScore: 150, featurePointCount: 20,
                              detectedObjectIds: ["bed_1": 0.9]),  // Best view of bed_1
            KeyframeCandidate(frameIndex: 2, pose: poseAt(x: 0.2, yaw: 0),
                              blurScore: 150, featurePointCount: 20,
                              detectedObjectIds: ["bed_1": 0.7]),
        ]
        let selected = KeyframeSelector.select(candidates, maxCount: 2)
        let indices = selected.map { $0.frameIndex }
        #expect(indices.contains(1))  // Must include the best view of bed_1
    }

    private func poseAt(x: Float, yaw: Float) -> simd_float4x4 {
        let rot = simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0))
        var t = simd_float4x4(rot)
        t.columns.3 = SIMD4(x, 0, 0, 1)
        return t
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/KeyframeSelectorTests 2>&1 | tail -20
```

Expected: compile failure with "cannot find 'KeyframeSelector' in scope".

- [ ] **Step 3: Create KeyframeSelector.swift**

Create `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/KeyframeSelector.swift`:

```swift
//
//  KeyframeSelector.swift
//  decroom_no_lidar
//
//  Selects a diverse subset of captured frames for backend upload.
//  Combines farthest-point sampling (spatial diversity) with
//  per-object best-view selection (semantic richness) and filters
//  low-quality frames (blur, sparse features).
//

import Foundation
import simd

struct KeyframeCandidate {
    let frameIndex: Int
    let pose: simd_float4x4
    let blurScore: Float
    let featurePointCount: Int
    /// Map of object instance id → detection confidence in this frame.
    let detectedObjectIds: [String: Float]
}

enum KeyframeSelector {

    /// Select up to `maxCount` keyframes prioritizing:
    /// 1. Quality — drop blur < 120 or feature points < 5
    /// 2. Best view per detected object
    /// 3. Spatial diversity via farthest-point sampling
    static func select(_ candidates: [KeyframeCandidate], maxCount: Int) -> [KeyframeCandidate] {
        let qualityFiltered = candidates.filter { $0.blurScore >= 120 && $0.featurePointCount >= 5 }
        guard !qualityFiltered.isEmpty else { return [] }

        // Step 1: best frame per detected object
        var objectBestFrame: [String: KeyframeCandidate] = [:]
        for candidate in qualityFiltered {
            for (objectId, confidence) in candidate.detectedObjectIds {
                if let existing = objectBestFrame[objectId] {
                    if confidence > (existing.detectedObjectIds[objectId] ?? 0) {
                        objectBestFrame[objectId] = candidate
                    }
                } else {
                    objectBestFrame[objectId] = candidate
                }
            }
        }
        var selected = Array(Set(objectBestFrame.values)).sorted { $0.frameIndex < $1.frameIndex }

        // Step 2: farthest-point sampling to fill remaining slots
        var remaining = qualityFiltered.filter { cand in
            !selected.contains(where: { $0.frameIndex == cand.frameIndex })
        }

        while selected.count < maxCount && !remaining.isEmpty {
            // Find the candidate farthest from all currently selected
            var bestIdx = 0
            var bestDist: Float = 0
            for (i, cand) in remaining.enumerated() {
                let candPos = positionOf(cand.pose)
                let minDistToSelected = selected.map {
                    simd_distance(candPos, positionOf($0.pose))
                }.min() ?? Float.greatestFiniteMagnitude
                if minDistToSelected > bestDist {
                    bestDist = minDistToSelected
                    bestIdx = i
                }
            }
            selected.append(remaining.remove(at: bestIdx))
        }

        return selected.sorted { $0.frameIndex < $1.frameIndex }
    }

    private static func positionOf(_ pose: simd_float4x4) -> SIMD3<Float> {
        SIMD3(pose.columns.3.x, pose.columns.3.y, pose.columns.3.z)
    }
}

// MARK: - Hashable conformance for Set<KeyframeCandidate>

extension KeyframeCandidate: Hashable {
    func hash(into hasher: inout Hasher) { hasher.combine(frameIndex) }
    static func == (lhs: KeyframeCandidate, rhs: KeyframeCandidate) -> Bool {
        lhs.frameIndex == rhs.frameIndex
    }
}
```

- [ ] **Step 4: Register files in Xcode project**

Follow Task 1 Step 4 pattern. IDs:
- `KeyframeSelector.swift` → `C201B11F/C201A11F`
- `KeyframeSelectorTests.swift` → `C201B120/C201A120`

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/KeyframeSelectorTests 2>&1 | tail -20
```

Expected: `** TEST SUCCEEDED **` with 4 tests passing.

- [ ] **Step 6: Commit**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git add decroom_no_lidar/decroom_no_lidar/KeyframeSelector.swift decroom_no_lidar/decroom_no_lidarTests/KeyframeSelectorTests.swift decroom_no_lidar/decroom_no_lidar.xcodeproj/project.pbxproj
git commit -m "feat: add KeyframeSelector with FPS + best-view-per-object

Selects up to N keyframes prioritizing quality (blur/feature filter),
best view per detected object (semantic), and farthest-point sampling
in pose space (spatial diversity)."
```

---

## Task 8: Full Test Suite Run

**Purpose:** Verify all new components pass tests together and no regressions in existing code.

- [ ] **Step 1: Run the entire test suite**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild test -scheme decroom_no_lidar -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -40
```

Expected: `** TEST SUCCEEDED **` with all tests from Tasks 1-7 passing plus the existing `decroom_no_lidarTests.example` test.

If any failures, diagnose and fix before continuing.

- [ ] **Step 2: Build for device to verify release path works**

Run:
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar
xcodebuild build -scheme decroom_no_lidar -destination 'generic/platform=iOS' 2>&1 | tail -10
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 3: Final commit (if any fixes were needed)**

If any issues were fixed above, commit with:
```bash
git commit -am "fix: address integration issues found in full test suite"
```

Otherwise, skip.

---

## What's next

With this plan complete, the foundation is in place. Plan 2 will add the visual capture layer (`ARRoomPreviewView` grid shell + reticle + brightness gradient + furniture emergence) and the on-device YOLO (`YoloRunner`, `ObjectTracker`) that feeds into `KeyframeSelector.detectedObjectIds`.

Until then, these components are individually testable via unit tests and the app still runs the existing capture flow (no user-visible change yet).
