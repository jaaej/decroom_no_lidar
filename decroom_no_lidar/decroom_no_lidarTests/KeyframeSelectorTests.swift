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
                                                 detectedObjectIds: [:]))
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
                                                 detectedObjectIds: [:]))
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
                                                 detectedObjectIds: [:]))
        }
        candidates.append(KeyframeCandidate(frameIndex: 10,
                                              pose: poseAt(x: 5.0, yaw: 0),
                                              blurScore: 150, featurePointCount: 20,
                                              detectedObjectIds: [:]))
        candidates.append(KeyframeCandidate(frameIndex: 11,
                                              pose: poseAt(x: -5.0, yaw: 0),
                                              blurScore: 150, featurePointCount: 20,
                                              detectedObjectIds: [:]))

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
