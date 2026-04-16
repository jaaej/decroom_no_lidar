//
//  GridShellEntity.swift
//  decroom_no_lidar
//
//  Wireframe cube around the user — only the 12 edges are rendered (thin
//  cyan cylinders), so the user sees through to the live AR camera. As
//  matching plane anchors are detected, the corresponding face shows
//  4 brighter "highlighted" edges to indicate that direction is captured.
//

import Foundation
import RealityKit
import UIKit
import simd

final class GridShellEntity {
    let entity: Entity
    private var edgesByFace: [String: [ModelEntity]] = [:]

    /// Six faces, each with the 4 edges that compose it.
    /// "+y" / "-y" share edges with "+x" / "-x" / "+z" / "-z" — for highlighting we
    /// just bump the brightness of the edges belonging to a detected face.
    init(sideLength: Float = 3.0) {
        entity = Entity()
        let half = sideLength / 2
        let edgeRadius: Float = 0.005  // 5mm

        // 12 edges of the cube. Each is identified by its 3D start/end and tagged
        // with the (up to) 2 faces it belongs to.
        let edges: [(SIMD3<Float>, SIMD3<Float>, [String])] = [
            // 4 edges along Z (top + bottom rows)
            (SIMD3(-half, -half, -half), SIMD3(-half, -half, half), ["-x", "-y"]),
            (SIMD3( half, -half, -half), SIMD3( half, -half, half), ["+x", "-y"]),
            (SIMD3(-half,  half, -half), SIMD3(-half,  half, half), ["-x", "+y"]),
            (SIMD3( half,  half, -half), SIMD3( half,  half, half), ["+x", "+y"]),
            // 4 edges along X
            (SIMD3(-half, -half, -half), SIMD3( half, -half, -half), ["-y", "-z"]),
            (SIMD3(-half, -half,  half), SIMD3( half, -half,  half), ["-y", "+z"]),
            (SIMD3(-half,  half, -half), SIMD3( half,  half, -half), ["+y", "-z"]),
            (SIMD3(-half,  half,  half), SIMD3( half,  half,  half), ["+y", "+z"]),
            // 4 edges along Y (vertical pillars)
            (SIMD3(-half, -half, -half), SIMD3(-half,  half, -half), ["-x", "-z"]),
            (SIMD3( half, -half, -half), SIMD3( half,  half, -half), ["+x", "-z"]),
            (SIMD3(-half, -half,  half), SIMD3(-half,  half,  half), ["-x", "+z"]),
            (SIMD3( half, -half,  half), SIMD3( half,  half,  half), ["+x", "+z"]),
        ]

        for (start, end, faces) in edges {
            let edgeEntity = makeEdge(from: start, to: end, radius: edgeRadius)
            entity.addChild(edgeEntity)
            for face in faces {
                edgesByFace[face, default: []].append(edgeEntity)
            }
        }
    }

    /// Highlight a face's edges (brighter cyan) when the matching plane anchor is detected.
    func solidify(face: String) {
        guard let edges = edgesByFace[face] else { return }
        var bright = UnlitMaterial()
        bright.color = .init(tint: UIColor.systemCyan)
        for edge in edges {
            edge.model?.materials = [bright]
        }
    }

    /// Build one edge as a thin cylinder oriented from start to end.
    private func makeEdge(from start: SIMD3<Float>, to end: SIMD3<Float>, radius: Float) -> ModelEntity {
        let length = simd_distance(start, end)
        let mesh = MeshResource.generateCylinder(height: length, radius: radius)

        var material = UnlitMaterial()
        material.color = .init(tint: UIColor.cyan.withAlphaComponent(0.6))

        let edge = ModelEntity(mesh: mesh, materials: [material])

        // Cylinders are generated along Y axis. Rotate to point from start→end.
        let direction = simd_normalize(end - start)
        let yAxis = SIMD3<Float>(0, 1, 0)
        let dot = simd_dot(yAxis, direction)
        if dot < 0.9999 && dot > -0.9999 {
            let axis = simd_normalize(simd_cross(yAxis, direction))
            let angle = acos(dot)
            edge.orientation = simd_quatf(angle: angle, axis: axis)
        } else if dot < 0 {
            // Pointing -Y, flip 180° around any perpendicular axis.
            edge.orientation = simd_quatf(angle: .pi, axis: SIMD3(1, 0, 0))
        }

        edge.position = (start + end) / 2
        return edge
    }
}
