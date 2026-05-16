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

    /// Project `position` onto the nearest wall segment for wall-snap
    /// categories, then offset inward by `inwardOffset` so the object's
    /// geometry doesn't extend past the wall. Pass the object's metadata
    /// so the offset matches its actual depth (half of the shorter XZ
    /// dimension). Non-wall-snap categories return position unchanged.
    static func wallSnap(
        position: SIMD2<Float>,
        category: String,
        walls: [Wall],
        objectMetadata: ObjectMetadata? = nil
    ) -> SIMD2<Float> {
        guard wallSnapCategories.contains(category) else { return position }
        guard !walls.isEmpty else { return position }

        let inset = inwardOffset(for: objectMetadata)
        let center = roomCenter(walls: walls)

        var bestDist: Float = .greatestFiniteMagnitude
        var snapped = position
        for wall in walls {
            guard wall.start.count >= 2, wall.end.count >= 2 else { continue }
            let a = SIMD2<Float>(Float(wall.start[0]), Float(wall.start[1]))
            let b = SIMD2<Float>(Float(wall.end[0]),   Float(wall.end[1]))
            let projected = projectOntoSegment(p: position, a: a, b: b)
            let d = simd_distance(position, projected)
            if d < bestDist {
                bestDist = d
                // Compute inward normal of this wall (toward room center).
                let dir = b - a
                let len = simd_length(dir)
                if len > 1e-6 {
                    let n = simd_normalize(SIMD2<Float>(-dir.y, dir.x))
                    let toCenter = center - projected
                    let inward = simd_dot(n, toCenter) >= 0 ? n : -n
                    snapped = projected + inward * inset
                } else {
                    snapped = projected
                }
            }
        }
        return snapped
    }

    /// Half of the object's shortest XZ dimension — the side that faces
    /// the wall when the object is in standard "against the wall" pose.
    /// Falls back to 15 cm for missing metadata.
    private static func inwardOffset(for meta: ObjectMetadata?) -> Float {
        guard let meta = meta else { return 0.15 }
        let depth = Float(min(meta.width, meta.length))
        return max(0.05, depth / 2)
    }

    private static func roomCenter(walls: [Wall]) -> SIMD2<Float> {
        var xs: [Float] = []
        var zs: [Float] = []
        for w in walls {
            for ep in [w.start, w.end] {
                guard ep.count >= 2 else { continue }
                xs.append(Float(ep[0]))
                zs.append(Float(ep[1]))
            }
        }
        guard !xs.isEmpty else { return .zero }
        let xMin = xs.min()!, xMax = xs.max()!
        let zMin = zs.min()!, zMax = zs.max()!
        return SIMD2<Float>((xMin + xMax) / 2, (zMin + zMax) / 2)
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

    /// Clamp `p` to the rectangular bbox of all wall endpoints, optionally
    /// inset by the object's half-depth so the rendered geometry stays
    /// inside the walls even at the bbox boundary.
    static func clampToFootprint(
        _ p: SIMD2<Float>, walls: [Wall],
        objectMetadata: ObjectMetadata? = nil
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
        let inset = inwardOffset(for: objectMetadata)
        let xLo = xMin + inset, xHi = xMax - inset
        let zLo = zMin + inset, zHi = zMax - inset
        // Guard against tiny rooms where inset would invert the bounds.
        let finalXLo = xLo <= xHi ? xLo : (xMin + xMax) / 2
        let finalXHi = xLo <= xHi ? xHi : (xMin + xMax) / 2
        let finalZLo = zLo <= zHi ? zLo : (zMin + zMax) / 2
        let finalZHi = zLo <= zHi ? zHi : (zMin + zMax) / 2
        return SIMD2<Float>(
            max(finalXLo, min(finalXHi, p.x)),
            max(finalZLo, min(finalZHi, p.y))
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

    // MARK: - Selection highlight

    private static let highlightNodeName = "_selection_highlight"

    /// Apply a thin emissive cyan outline to the SCNNode owning the given
    /// RoomObject.id. Pass nil to clear all highlights. Uses a child node
    /// scaled slightly larger than the parent so the outline shows through.
    static func setSelected(_ id: String?, in rootNode: SCNNode) {
        // Remove existing highlight wherever it lives.
        rootNode.enumerateChildNodes { node, _ in
            if node.name == highlightNodeName {
                node.removeFromParentNode()
            }
        }
        guard let id = id else { return }
        guard let target = findNode(named: "obj_\(id)", under: rootNode) else { return }

        let bounds = localBounds(for: target)
        let minVec = bounds.min
        let maxVec = bounds.max
        let size = SCNVector3(
            max(0.12, maxVec.x - minVec.x),
            max(0.12, maxVec.y - minVec.y),
            max(0.12, maxVec.z - minVec.z)
        )
        let outline = SCNNode(geometry: SCNBox(
            width: CGFloat(size.x), height: CGFloat(size.y),
            length: CGFloat(size.z), chamferRadius: 0
        ))
        outline.name = highlightNodeName
        outline.position = SCNVector3(
            (minVec.x + maxVec.x) / 2,
            (minVec.y + maxVec.y) / 2,
            (minVec.z + maxVec.z) / 2
        )
        outline.scale = SCNVector3(1.08, 1.08, 1.08)
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.systemCyan
        material.emission.contents = UIColor.systemCyan
        material.lightingModel = .constant
        material.isDoubleSided = true
        material.fillMode = .lines
        material.readsFromDepthBuffer = false
        material.writesToDepthBuffer = false
        outline.geometry?.firstMaterial = material
        outline.renderingOrder = 1000
        target.addChildNode(outline)
    }

    private static func findNode(named name: String, under root: SCNNode) -> SCNNode? {
        if root.name == name { return root }
        for child in root.childNodes {
            if let found = findNode(named: name, under: child) { return found }
        }
        return nil
    }

    private static func localBounds(for node: SCNNode) -> (min: SCNVector3, max: SCNVector3) {
        var hasPoint = false
        var minVec = SCNVector3(
            Float.greatestFiniteMagnitude,
            Float.greatestFiniteMagnitude,
            Float.greatestFiniteMagnitude
        )
        var maxVec = SCNVector3(
            -Float.greatestFiniteMagnitude,
            -Float.greatestFiniteMagnitude,
            -Float.greatestFiniteMagnitude
        )

        node.enumerateChildNodes { child, _ in
            guard child.name != highlightNodeName else { return }
            let (childMin, childMax) = child.boundingBox
            let corners = [
                SCNVector3(childMin.x, childMin.y, childMin.z),
                SCNVector3(childMin.x, childMin.y, childMax.z),
                SCNVector3(childMin.x, childMax.y, childMin.z),
                SCNVector3(childMin.x, childMax.y, childMax.z),
                SCNVector3(childMax.x, childMin.y, childMin.z),
                SCNVector3(childMax.x, childMin.y, childMax.z),
                SCNVector3(childMax.x, childMax.y, childMin.z),
                SCNVector3(childMax.x, childMax.y, childMax.z),
            ]
            for corner in corners {
                let p = node.convertPosition(corner, from: child)
                minVec = SCNVector3(min(minVec.x, p.x), min(minVec.y, p.y), min(minVec.z, p.z))
                maxVec = SCNVector3(max(maxVec.x, p.x), max(maxVec.y, p.y), max(maxVec.z, p.z))
                hasPoint = true
            }
        }

        if hasPoint {
            return (minVec, maxVec)
        }
        let (fallbackMin, fallbackMax) = node.boundingBox
        return (fallbackMin, fallbackMax)
    }
}
