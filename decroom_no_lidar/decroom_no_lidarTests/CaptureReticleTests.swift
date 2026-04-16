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
