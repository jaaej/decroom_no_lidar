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
