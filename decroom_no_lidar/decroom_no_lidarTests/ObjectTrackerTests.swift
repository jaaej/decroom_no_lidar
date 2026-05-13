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
