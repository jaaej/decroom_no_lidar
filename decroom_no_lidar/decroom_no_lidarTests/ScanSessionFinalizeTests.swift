import Testing
import Foundation
import simd
@testable import decroom_no_lidar

@MainActor
struct ScanSessionFinalizeTests {

    /// ScanSession loads DepthRunner asynchronously via Task.detached; in
    /// production the user scans for tens of seconds before tapping Done,
    /// so the model is always ready at finalize time. In tests we call
    /// finalize() immediately — wait for modelsReady first so the
    /// depthRunner nil-check isn't tripped.
    private func awaitModelsLoad(_ session: ScanSession, timeoutSec: Double = 5.0) async {
        let start = Date()
        while !session.modelsReady && Date().timeIntervalSince(start) < timeoutSec {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    @Test func finalizeWithNoWallsThrowsNoWallsDetected() async throws {
        let session = ScanSession()
        await awaitModelsLoad(session)
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
        await awaitModelsLoad(session)
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
        await awaitModelsLoad(session)
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
