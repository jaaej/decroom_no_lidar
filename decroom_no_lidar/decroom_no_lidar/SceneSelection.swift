//
//  SceneSelection.swift
//  decroom_no_lidar
//
//  Plan 4a — SceneKit-specific helpers for the correction UI:
//   hitTest    : tap location -> RoomObject.id (with 22pt retry radius)
//   floorPoint : tap location -> world-space XZ via ray-plane intersection
//   wallSnap   : wall-snap-category positions snap to nearest wall
//   clampToFootprint : clamp position to the rectangular room bbox
//   setSelected: apply/remove the selection highlight to a SCNNode
//

import Foundation
import SceneKit
import UIKit
import simd

enum SceneSelection {

    // MARK: - Snap rules (mirror Plan 3b consensus.snap_to_walls categories)

    private static let wallSnapCategories: Set<String> = [
        "bed", "dresser", "wardrobe", "sofa", "couch", "bookshelf", "nightstand",
    ]

    // MARK: - Wall snap

    static func wallSnap(
        position: SIMD2<Float>,
        category: String,
        walls: [Wall]
    ) -> SIMD2<Float> {
        guard wallSnapCategories.contains(category) else { return position }
        guard !walls.isEmpty else { return position }

        var bestDist: Float = .greatestFiniteMagnitude
        var snapped = position
        for wall in walls {
            guard wall.start.count >= 2, wall.end.count >= 2 else { continue }
            let projected = projectOntoSegment(
                p: position,
                a: SIMD2<Float>(Float(wall.start[0]), Float(wall.start[1])),
                b: SIMD2<Float>(Float(wall.end[0]),   Float(wall.end[1]))
            )
            let d = simd_distance(position, projected)
            if d < bestDist {
                bestDist = d
                snapped = projected
            }
        }
        return snapped
    }

    private static func projectOntoSegment(
        p: SIMD2<Float>, a: SIMD2<Float>, b: SIMD2<Float>
    ) -> SIMD2<Float> {
        let ab = b - a
        let len2 = simd_dot(ab, ab)
        guard len2 > 1e-6 else { return a }
        let t = simd_dot(p - a, ab) / len2
        let tClamped = max(0, min(1, t))
        return a + ab * tClamped
    }

    // MARK: - Footprint clamp

    static func clampToFootprint(
        _ p: SIMD2<Float>, walls: [Wall]
    ) -> SIMD2<Float> {
        guard !walls.isEmpty else { return p }
        var xMin: Float = .greatestFiniteMagnitude
        var xMax: Float = -.greatestFiniteMagnitude
        var zMin: Float = .greatestFiniteMagnitude
        var zMax: Float = -.greatestFiniteMagnitude
        for w in walls {
            for endpoint in [w.start, w.end] {
                guard endpoint.count >= 2 else { continue }
                let x = Float(endpoint[0])
                let z = Float(endpoint[1])
                xMin = min(xMin, x); xMax = max(xMax, x)
                zMin = min(zMin, z); zMax = max(zMax, z)
            }
        }
        return SIMD2<Float>(
            max(xMin, min(xMax, p.x)),
            max(zMin, min(zMax, p.y))
        )
    }

    // MARK: - Hit test (Task 7)

    static func hitTest(_ scnView: SCNView, at point: CGPoint) -> String? {
        return nil  // Task 7
    }

    // MARK: - Floor raycast (Task 8)

    static func floorPoint(forScreen point: CGPoint, in scnView: SCNView) -> SIMD2<Float>? {
        return nil  // Task 8
    }

    // MARK: - Selection highlight (Task 11)

    static func setSelected(_ id: String?, in rootNode: SCNNode) {
        // Task 11
    }
}
