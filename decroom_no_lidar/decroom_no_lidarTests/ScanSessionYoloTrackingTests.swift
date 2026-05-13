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
            depthMap: depths,
            depthWidth: 10,
            depthHeight: 10,
            depthIntrinsics: intr,
            worldFromCamera: pose
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

        let frames: [Int] = [1, 10, 20]
        let centers: [Double] = [0.50, 0.52, 0.48]
        for (f, cx) in zip(frames, centers) {
            let det = YoloDetection(
                category: "bed",
                confidence: Float(0.7 + Double(f) * 0.01),
                bbox: CGRect(x: cx - 0.1, y: 0.4, width: 0.2, height: 0.2),
                mask: nil,
                frameIndex: f
            )
            let w = YoloUnprojector.worldPosition(
                yoloBboxCenterNormalized: SIMD2<Float>(Float(det.bbox.midX),
                                                         Float(det.bbox.midY)),
                depthMap: depths,
                depthWidth: 10, depthHeight: 10,
                depthIntrinsics: intr,
                worldFromCamera: pose
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
