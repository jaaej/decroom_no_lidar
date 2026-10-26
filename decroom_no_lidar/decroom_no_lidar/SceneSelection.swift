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
    /// categories, then offset inward by the object's half-extent along
    /// the wall's inward normal so the rendered geometry sits flush with
    /// the inside face of the wall. Non-wall-snap categories return
    /// position unchanged.
    static func wallSnap(
        position: SIMD2<Float>,
        category: String,
        walls: [Wall],
        objectMetadata: ObjectMetadata? = nil,
        rotationY: Double = 0
    ) -> SIMD2<Float> {
        guard wallSnapCategories.contains(category) else { return position }
        guard !walls.isEmpty else { return position }

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
                let dir = b - a
                let len = simd_length(dir)
                if len > 1e-6 {
                    let n = simd_normalize(SIMD2<Float>(-dir.y, dir.x))
                    let toCenter = center - projected
                    let inward = simd_dot(n, toCenter) >= 0 ? n : -n
                    let inset = halfExtentAlong(
                        normal: inward, meta: objectMetadata, rotationY: rotationY
                    )
                    snapped = projected + inward * inset
                } else {
                    snapped = projected
                }
            }
        }
        return snapped
    }

    /// World-space AABB half-extents (halfX, halfZ) of an object's
    /// footprint after rotating by `rotationY` about the Y axis.
    /// width is along local X, length along local Z. Falls back to a
    /// uniform 15 cm half-extent when metadata is missing.
    static func rotatedHalfExtents(
        meta: ObjectMetadata?, rotationY: Double
    ) -> SIMD2<Float> {
        guard let meta = meta else { return SIMD2<Float>(0.15, 0.15) }
        let hx = Float(meta.width) / 2
        let hz = Float(meta.length) / 2
        let c = abs(Float(cos(rotationY)))
        let s = abs(Float(sin(rotationY)))
        return SIMD2<Float>(
            max(0.05, hx * c + hz * s),
            max(0.05, hx * s + hz * c)
        )
    }

    /// Half-extent of the rotated OBB along an arbitrary unit direction.
    /// Used to inset wall-snap positions by exactly enough that the
    /// object's far edge sits on the wall plane.
    private static func halfExtentAlong(
        normal n: SIMD2<Float>, meta: ObjectMetadata?, rotationY: Double
    ) -> Float {
        guard let meta = meta else { return 0.15 }
        let hx = Float(meta.width) / 2
        let hz = Float(meta.length) / 2
        // Local x-axis = (cos θ, sin θ); local z-axis = (-sin θ, cos θ).
        let cosT = Float(cos(rotationY))
        let sinT = Float(sin(rotationY))
        let projX = abs(n.x * cosT + n.y * sinT)
        let projZ = abs(-n.x * sinT + n.y * cosT)
        return max(0.05, hx * projX + hz * projZ)
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

    /// Clamp `p` so the object's rotated footprint stays inside every
    /// wall's inward half-plane. Walls are NOT assumed axis-aligned —
    /// `RoomGeometryBuilder` consolidates in a Manhattan frame and rotates
    /// back by the room's yaw, so the world-space room is generally a
    /// rotated rectangle (or arbitrary polygon). A simple axis-aligned
    /// bbox clamp lets objects sit in bbox-but-outside-room corners; this
    /// per-wall half-plane clamp does not.
    ///
    /// Algorithm: for each wall, compute the inward unit normal (toward
    /// the centroid of wall endpoints) and the half-plane offset so the
    /// far edge of the object's rotated AABB sits exactly on the inner
    /// face of the wall. Iterate a few times so corner constraints from
    /// adjacent walls settle.
    static func clampToFootprint(
        _ p: SIMD2<Float>, walls: [Wall],
        objectMetadata: ObjectMetadata? = nil,
        rotationY: Double = 0
    ) -> SIMD2<Float> {
        guard !walls.isEmpty else { return p }
        let center = roomCenter(walls: walls)

        struct Plane {
            let point: SIMD2<Float>
            let inward: SIMD2<Float>     // unit normal toward room interior
            let minDistance: Float       // required signed distance from `point` along `inward`
        }
        var planes: [Plane] = []
        planes.reserveCapacity(walls.count)
        for w in walls {
            guard w.start.count >= 2, w.end.count >= 2 else { continue }
            let a = SIMD2<Float>(Float(w.start[0]), Float(w.start[1]))
            let b = SIMD2<Float>(Float(w.end[0]),   Float(w.end[1]))
            let dir = b - a
            let len = simd_length(dir)
            guard len > 1e-4 else { continue }
            let nRaw = simd_normalize(SIMD2<Float>(-dir.y, dir.x))
            let toCenter = center - a
            let inward = simd_dot(nRaw, toCenter) >= 0 ? nRaw : -nRaw
            let halfExt = halfExtentAlong(normal: inward, meta: objectMetadata, rotationY: rotationY)
            // Half the visual wall thickness, so the object sits flush
            // with the inner face rather than the wall centerline.
            let wallInset: Float = 0.04
            planes.append(Plane(point: a, inward: inward, minDistance: halfExt + wallInset))
        }
        guard !planes.isEmpty else { return p }

        var result = p
        // 4 iterations is plenty for a rectangle (1-2 in practice) and
        // gives quasi-stable behavior for non-orthogonal walls.
        for _ in 0..<4 {
            var moved = false
            for plane in planes {
                let d = simd_dot(result - plane.point, plane.inward)
                let deficit = plane.minDistance - d
                if deficit > 0 {
                    result += plane.inward * deficit
                    moved = true
                }
            }
            if !moved { break }
        }
        return result
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
