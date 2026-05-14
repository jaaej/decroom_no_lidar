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

    // MARK: - Snap rules (mirror backend consensus.SNAP_TO_WALL_CATEGORIES)

    private static let wallSnapCategories: Set<String> = [
        "bed", "bookshelf", "couch", "desk", "dresser", "nightstand", "sofa", "tv", "wardrobe",
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
        let options: [SCNHitTestOption: Any] = [
            SCNHitTestOption.searchMode: SCNHitTestSearchMode.closest.rawValue,
            SCNHitTestOption.ignoreHiddenNodes: true,
        ]
        // Center hit first.
        if let hit = scnView.hitTest(point, options: options).first,
           let id = roomObjectID(forNode: hit.node) {
            return id
        }
        // Retry within a 22pt radius (8 surrounding points).
        let offsets: [(Int, Int)] = [
            (11, 0), (-11, 0), (0, 11), (0, -11),
            (8, 8), (-8, 8), (8, -8), (-8, -8),
        ]
        for (dx, dy) in offsets {
            let p = CGPoint(x: point.x + CGFloat(dx), y: point.y + CGFloat(dy))
            if let hit = scnView.hitTest(p, options: options).first,
               let id = roomObjectID(forNode: hit.node) {
                return id
            }
        }
        return nil
    }

    /// Walk up the node tree until we find a node whose name starts with
    /// "obj_" (the convention used by RoomSceneBuilder).
    /// Returns the bare RoomObject.id (the "obj_" prefix is stripped).
    private static func roomObjectID(forNode node: SCNNode) -> String? {
        var current: SCNNode? = node
        while let n = current {
            if let name = n.name, name.hasPrefix("obj_") {
                return String(name.dropFirst(4))
            }
            current = n.parent
        }
        return nil
    }

    // MARK: - Floor raycast

    /// Project a screen point onto the y=0 floor plane in world space.
    /// Returns nil if the ray is parallel to the floor or points away.
    static func floorPoint(forScreen point: CGPoint, in scnView: SCNView) -> SIMD2<Float>? {
        let near = scnView.unprojectPoint(SCNVector3(Float(point.x), Float(point.y), 0))
        let far  = scnView.unprojectPoint(SCNVector3(Float(point.x), Float(point.y), 1))
        return floorPointFromRay(
            near: SIMD3<Float>(near.x, near.y, near.z),
            far:  SIMD3<Float>(far.x,  far.y,  far.z)
        )
    }

    /// Pure-math ray/plane intersection — unit-testable.
    static func floorPointFromRay(
        near: SIMD3<Float>, far: SIMD3<Float>
    ) -> SIMD2<Float>? {
        let dir = far - near
        guard abs(dir.y) > 1e-4 else { return nil }
        let t = -near.y / dir.y
        guard t > 0 else { return nil }
        let world = near + dir * t
        return SIMD2<Float>(world.x, world.z)
    }

    // MARK: - Selection highlight (Task 11)

    static func setSelected(_ id: String?, in rootNode: SCNNode) {
        // Task 11
    }
}
