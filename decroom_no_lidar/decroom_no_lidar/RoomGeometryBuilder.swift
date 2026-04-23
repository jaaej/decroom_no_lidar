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
        let rawWalls = wallSegments(from: verticals, yaw: yaw, floorY: floorHeight)
        let walls = consolidate(rawWalls, yaw: yaw)

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

    // MARK: - Consolidation

    /// Turn raw ARKit plane-anchor segments into a clean room rectangle.
    /// Steps:
    ///   1. Rotate segments into the Manhattan frame, classify by dominant
    ///      axis, and drop anything >~20° off both axes.
    ///   2. Merge colinear fragments (group by perpendicular distance within
    ///      `mergeTolerance` and span min–max).
    ///   3. If we have ≥2 horizontal or ≥2 vertical walls (with at least one
    ///      of each), keep the outer-most on each side and extend endpoints
    ///      to close a rectangle. Otherwise return the merged set as-is
    ///      (partial scan — don't fabricate walls).
    ///   4. Rotate back into world space and emit `WallSegment`s.
    private func consolidate(
        _ raw: [WallSegment],
        yaw: Float,
        mergeTolerance: Float = 0.3,      // meters
        manhattanTolerance: Float = 0.35  // |sin(angle-off-axis)|
    ) -> [WallSegment] {
        guard !raw.isEmpty else { return [] }

        let cosY = cos(-yaw)
        let sinY = sin(-yaw)
        func rotateInto(_ p: SIMD2<Float>) -> SIMD2<Float> {
            SIMD2(p.x * cosY - p.y * sinY,
                  p.x * sinY + p.y * cosY)
        }
        let cosYInv = cos(yaw)
        let sinYInv = sin(yaw)
        func rotateBack(_ p: SIMD2<Float>) -> SIMD2<Float> {
            SIMD2(p.x * cosYInv - p.y * sinYInv,
                  p.x * sinYInv + p.y * cosYInv)
        }

        struct Segment {
            var startLocal: SIMD2<Float>
            var endLocal: SIMD2<Float>
            var height: Float
            var isHorizontal: Bool
            var perpPos: Float    // Z for horizontal, X for vertical
            var spanMin: Float    // min X for horizontal, min Z for vertical
            var spanMax: Float    // max X for horizontal, max Z for vertical
        }

        // `finalize` declared before any early-return so every branch sees it.
        func finalize(_ s: Segment) -> WallSegment {
            let sw = rotateBack(s.startLocal)
            let ew = rotateBack(s.endLocal)
            return WallSegment(start: sw, end: ew,
                               height: s.height > 0 ? s.height : 2.5)
        }

        // 1. Rotate + classify + filter out mis-oriented segments.
        var segs: [Segment] = []
        for w in raw {
            let s = rotateInto(w.start)
            let e = rotateInto(w.end)
            let d = e - s
            let length = simd_length(d)
            guard length > 0.001 else { continue }
            let dir = d / length
            // |sin(angleOffX)| = |dir.y| ; |sin(angleOffZ)| = |dir.x|.
            // Keep only if close to at least one Manhattan axis.
            let minOff = min(abs(dir.y), abs(dir.x))
            guard minOff < manhattanTolerance else { continue }

            let isHorizontal = abs(dir.x) >= abs(dir.y)
            if isHorizontal {
                segs.append(Segment(
                    startLocal: s, endLocal: e, height: w.height,
                    isHorizontal: true,
                    perpPos: (s.y + e.y) / 2,
                    spanMin: min(s.x, e.x),
                    spanMax: max(s.x, e.x)
                ))
            } else {
                segs.append(Segment(
                    startLocal: s, endLocal: e, height: w.height,
                    isHorizontal: false,
                    perpPos: (s.x + e.x) / 2,
                    spanMin: min(s.y, e.y),
                    spanMax: max(s.y, e.y)
                ))
            }
        }

        guard !segs.isEmpty else { return [] }

        // 2. Merge colinear segments by perpPos proximity.
        func mergeColinear(_ group: [Segment]) -> [Segment] {
            var sorted = group.sorted { $0.perpPos < $1.perpPos }
            var merged: [Segment] = []
            while !sorted.isEmpty {
                var cluster: [Segment] = [sorted.removeFirst()]
                while let next = sorted.first,
                      abs(next.perpPos - cluster[0].perpPos) < mergeTolerance {
                    cluster.append(sorted.removeFirst())
                }
                let avgPos = cluster.map { $0.perpPos }.reduce(0, +) / Float(cluster.count)
                let spanMin = cluster.map { $0.spanMin }.min()!
                let spanMax = cluster.map { $0.spanMax }.max()!
                let avgHeight = cluster.map { $0.height }.reduce(0, +) / Float(cluster.count)
                let proto = cluster[0]
                merged.append(Segment(
                    startLocal: proto.isHorizontal
                        ? SIMD2(spanMin, avgPos) : SIMD2(avgPos, spanMin),
                    endLocal: proto.isHorizontal
                        ? SIMD2(spanMax, avgPos) : SIMD2(avgPos, spanMax),
                    height: avgHeight,
                    isHorizontal: proto.isHorizontal,
                    perpPos: avgPos,
                    spanMin: spanMin,
                    spanMax: spanMax
                ))
            }
            return merged
        }

        let horizontals = mergeColinear(segs.filter { $0.isHorizontal })
        let verticals = mergeColinear(segs.filter { !$0.isHorizontal })

        // 3. Need both axes represented AND at least one axis with ≥2 walls
        //    to form a rectangle. Otherwise return merged walls as-is.
        guard !horizontals.isEmpty, !verticals.isEmpty,
              horizontals.count >= 2 || verticals.count >= 2 else {
            return (horizontals + verticals).map { finalize($0) }
        }

        let minZWall = horizontals.min(by: { $0.perpPos < $1.perpPos })!
        let maxZWall = horizontals.max(by: { $0.perpPos < $1.perpPos })!
        let minXWall = verticals.min(by: { $0.perpPos < $1.perpPos })!
        let maxXWall = verticals.max(by: { $0.perpPos < $1.perpPos })!

        let minX = minXWall.perpPos
        let maxX = maxXWall.perpPos
        let minZ = minZWall.perpPos
        let maxZ = maxZWall.perpPos
        let avgHeight = (minZWall.height + maxZWall.height + minXWall.height + maxXWall.height) / 4
        let height = avgHeight > 0 ? avgHeight : 2.5

        // 4. Build 4 walls sharing the rectangle corners.
        let rect: [Segment] = [
            Segment(startLocal: SIMD2(minX, minZ), endLocal: SIMD2(maxX, minZ),
                    height: height, isHorizontal: true, perpPos: minZ,
                    spanMin: minX, spanMax: maxX),
            Segment(startLocal: SIMD2(minX, maxZ), endLocal: SIMD2(maxX, maxZ),
                    height: height, isHorizontal: true, perpPos: maxZ,
                    spanMin: minX, spanMax: maxX),
            Segment(startLocal: SIMD2(minX, minZ), endLocal: SIMD2(minX, maxZ),
                    height: height, isHorizontal: false, perpPos: minX,
                    spanMin: minZ, spanMax: maxZ),
            Segment(startLocal: SIMD2(maxX, minZ), endLocal: SIMD2(maxX, maxZ),
                    height: height, isHorizontal: false, perpPos: maxX,
                    spanMin: minZ, spanMax: maxZ),
        ]

        return rect.map { finalize($0) }
    }
}
