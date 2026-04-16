import Testing
import Foundation
import simd
@testable import decroom_no_lidar

struct RoomGeometryBuilderTests {

    @Test func emptyInputProducesEmptyGeometry() {
        let builder = RoomGeometryBuilder()
        let geometry = builder.build(from: [])
        #expect(geometry.walls.isEmpty)
    }

    @Test func singleHorizontalPlaneSetsFloorHeight() {
        let floor = MockPlane(identifier: UUID(), planeAlignment: .horizontal,
                              worldY: 0.0, extents: SIMD3(3, 0, 3))
        let builder = RoomGeometryBuilder()
        let geometry = builder.build(from: [floor])
        #expect(abs(geometry.floorHeight) < 0.01)
        #expect(geometry.walls.isEmpty)  // One horizontal doesn't give walls
    }

    @Test func fourVerticalPlanesYieldFourWalls() {
        // A 3m x 3m room: 4 walls at x=-1.5, x=1.5, z=-1.5, z=1.5, floor at y=0
        let floor = MockPlane(identifier: UUID(), planeAlignment: .horizontal,
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
        let floor = MockPlane(identifier: UUID(), planeAlignment: .horizontal,
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

    @Test func nonSnappingYawPointsInCorrectDirection() {
        // 15° rotation is outside the snap zone (0.1 rad tolerance) —
        // builder must return a yaw close to the actual rotation, not snapped.
        // This guards against sign/axis swap regressions.
        let angle: Float = .pi / 12  // 15°
        let rotated = simd_quatf(angle: angle, axis: SIMD3(0, 1, 0))
        let normal = rotated.act(SIMD3<Float>(0, 0, 1))
        let wall = MockWall(center: SIMD3(0, 1.25, -1.5),
                             normal: normal,
                             width: 3.0, height: 2.5)
        let floor = MockPlane(identifier: UUID(), planeAlignment: .horizontal,
                              worldY: 0.0, extents: SIMD3(3, 0, 3))

        let builder = RoomGeometryBuilder()
        let geometry = builder.build(from: [floor, wall])

        // manhattanYaw is folded via 2*yaw, so the returned yaw should be
        // within ±0.02 rad of ±angle (mod pi/2). Not strictly 0 or pi/2.
        let yaw = geometry.manhattanYaw
        let modulated = fmod(abs(yaw), .pi / 2)
        let wrappedDistance = min(modulated, .pi / 2 - modulated)
        let expectedDistance = min(angle, .pi / 2 - angle)
        #expect(abs(wrappedDistance - expectedDistance) < 0.05)
    }
}

// MARK: - Test Doubles

struct MockPlane: PlaneAnchorDescriptor {
    let identifier: UUID
    let planeAlignment: PlaneAlignment
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
    let planeAlignment: PlaneAlignment = .vertical
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
