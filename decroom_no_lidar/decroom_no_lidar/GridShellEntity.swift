//
//  GridShellEntity.swift
//  decroom_no_lidar
//
//  Wireframe cube around the user. Each of the 6 faces is a separate
//  entity; as matching plane anchors are detected, the face is swapped
//  from wireframe to a solid semi-transparent material.
//

import Foundation
import RealityKit
import simd

final class GridShellEntity {
    let entity: Entity
    private var faces: [String: ModelEntity] = [:]  // key: "+x", "-x", "+y", "-y", "+z", "-z"

    init(sideLength: Float = 3.0) {
        entity = Entity()
        let halfSide = sideLength / 2

        for face in ["+x", "-x", "+y", "-y", "+z", "-z"] {
            let plane = MeshResource.generatePlane(width: sideLength, height: sideLength)
            var material = UnlitMaterial()
            material.color = .init(tint: .cyan.withAlphaComponent(0.3))
            let faceEntity = ModelEntity(mesh: plane, materials: [material])
            switch face {
            case "+x": faceEntity.position = SIMD3(halfSide, 0, 0)
                faceEntity.orientation = simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 1, 0))
            case "-x": faceEntity.position = SIMD3(-halfSide, 0, 0)
                faceEntity.orientation = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0))
            case "+y": faceEntity.position = SIMD3(0, halfSide, 0)
                faceEntity.orientation = simd_quatf(angle: .pi / 2, axis: SIMD3(1, 0, 0))
            case "-y": faceEntity.position = SIMD3(0, -halfSide, 0)
                faceEntity.orientation = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
            case "+z": faceEntity.position = SIMD3(0, 0, halfSide)
                faceEntity.orientation = simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
            case "-z": faceEntity.position = SIMD3(0, 0, -halfSide)
                faceEntity.orientation = simd_quatf()
            default: break
            }
            faces[face] = faceEntity
            entity.addChild(faceEntity)
        }
    }

    /// Mark a face as solidified (detected by ARKit plane anchor).
    /// The face swaps to an opaque material.
    func solidify(face: String) {
        guard let faceEntity = faces[face] else { return }
        var material = UnlitMaterial()
        material.color = .init(tint: .white.withAlphaComponent(0.5))
        faceEntity.model?.materials = [material]
    }
}
