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
        let horizontals = planes.filter { $0.planeAlignment == .horizontal }
        let verticals = planes.filter { $0.planeAlignment == .vertical }

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
            // Project onto XZ floor plane: n.x holds world X, n.y holds world Z.
            // (SIMD2 has no .z; we pack XZ into .x/.y for 2D angle math.)
            let nxz = SIMD2<Float>(normalWorld.x, normalWorld.z)
            let length = simd_length(nxz)
            guard length > 0.1 else { continue }
            let n = nxz / length

            // Wall yaw = atan2(worldX, worldZ). n.y represents world Z here.
            // Use 2*yaw to fold opposite normals together (walls have two faces).
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
