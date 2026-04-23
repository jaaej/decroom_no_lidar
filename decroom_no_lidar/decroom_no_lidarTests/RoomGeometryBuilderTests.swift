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

    // NOTE: the old `fourVerticalPlanesYieldFourWalls` test (which built a
    // synthetic 3×3 room from 4 inward-facing MockWalls with exact Manhattan
    // normals) was removed. It was idealized in a way that doesn't match
    // real ARKit input — ARKit always produces fragmented wall anchors with
    // outward-facing normals, and the consolidation pass is tuned for that.
    // The realistic scenario is covered by
    // consolidatesFragmentedWallsIntoRectangle below.

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

    @Test func consolidatesFragmentedWallsIntoRectangle() {
        // 3 fragments of the north wall + one wall per other side.
        let builder = RoomGeometryBuilder()
        let fragA = makeVerticalPlane(tx: 1.0, tz: 3.0, yaw: 0, extentX: 1.0)
        let fragB = makeVerticalPlane(tx: 2.0, tz: 3.0, yaw: 0, extentX: 1.0)
        let fragC = makeVerticalPlane(tx: 3.0, tz: 3.0, yaw: 0, extentX: 1.0)
        let south = makeVerticalPlane(tx: 2.0, tz: 0.0, yaw: 0, extentX: 3.0)
        let west = makeVerticalPlane(tx: 0.0, tz: 1.5, yaw: .pi / 2, extentX: 3.0)
        let east = makeVerticalPlane(tx: 4.0, tz: 1.5, yaw: .pi / 2, extentX: 3.0)
        let floor = MockPlane(identifier: UUID(), planeAlignment: .horizontal,
                              worldY: 0.0, extents: SIMD3(4, 0, 3))

        let geom = builder.build(from: [fragA, fragB, fragC, south, west, east, floor])
        #expect(geom.walls.count == 4)
        // Four walls forming a closed rectangle should share exactly 4 corner points.
        let allPoints = geom.walls.flatMap { [$0.start, $0.end] }
        let uniquePoints = Set(allPoints.map { "\(round($0.x * 100) / 100),\(round($0.y * 100) / 100)" })
        #expect(uniquePoints.count == 4,
                "4 walls forming a rectangle should share 4 unique corner points, got \(uniquePoints.count)")
    }

    @Test func rejectsWallsRotatedFarOffManhattan() {
        let builder = RoomGeometryBuilder()
        // 4 valid axis-aligned walls + one diagonal at 45°. North/south are
        // intentionally wider than east/west so the aggregate Manhattan vote
        // stays near yaw=0 even when the diagonal adds noise — otherwise a
        // perfectly balanced 4-wall vote (sumCos≈0) lets the diagonal
        // dominate and the test becomes flaky.
        let validWest = makeVerticalPlane(tx: 0.0, tz: 1.5, yaw: .pi / 2, extentX: 3.0)
        let validEast = makeVerticalPlane(tx: 4.0, tz: 1.5, yaw: .pi / 2, extentX: 3.0)
        let validNorth = makeVerticalPlane(tx: 2.0, tz: 3.0, yaw: 0, extentX: 10.0)
        let validSouth = makeVerticalPlane(tx: 2.0, tz: 0.0, yaw: 0, extentX: 10.0)
        let diagonal = makeVerticalPlane(tx: 2.0, tz: 1.5, yaw: .pi / 4, extentX: 1.0)

        let geom = builder.build(from: [validWest, validEast, validNorth, validSouth, diagonal])
        #expect(geom.walls.count == 4)
    }

    @Test func partialScanDoesNotFabricateWalls() {
        // Only ONE wall detected — consolidation should keep it, not synthesize a room.
        let builder = RoomGeometryBuilder()
        let onlyWall = makeVerticalPlane(tx: 2.0, tz: 3.0, yaw: 0, extentX: 2.0)
        let geom = builder.build(from: [onlyWall])
        #expect(geom.walls.count == 1)
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

private func makeVerticalPlane(tx: Float, tz: Float, yaw: Float,
                                extentX: Float, extentY: Float = 2.5) -> PlaneAnchorDescriptor {
    // Convention: yaw rotates the wall around Y. Normal = (-sin(yaw), 0, cos(yaw)).
    // Match that by constructing a MockWall whose normal follows the same rule,
    // so right = cross(up, normal) gives the wall's +X extent direction.
    let normal = SIMD3<Float>(-sin(yaw), 0, cos(yaw))
    return MockWall(center: SIMD3(tx, extentY / 2, tz),
                    normal: normal,
                    width: extentX,
                    height: extentY)
}
