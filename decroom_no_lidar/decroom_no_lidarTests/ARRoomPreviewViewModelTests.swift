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
