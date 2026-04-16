//
//  CaptureReticle.swift
//  decroom_no_lidar
//
//  World-anchored reticle entity. Uses a thin cyan torus to indicate
//  the next direction the user should aim the camera. Position is
//  recomputed each frame from ARRoomPreviewViewModel.reticleDirection.
//

import Foundation
import RealityKit
import simd

final class CaptureReticle {
    /// The RealityKit entity — attach to an AnchorEntity(.world(transform:)).
    let entity: ModelEntity

    init() {
        let mesh = MeshResource.generateCylinder(height: 0.015, radius: 0.1)
        var material = PhysicallyBasedMaterial()
        material.baseColor = .init(tint: .cyan.withAlphaComponent(0.9))
        material.emissiveColor = .init(color: .cyan)
        material.emissiveIntensity = 2.0
        self.entity = ModelEntity(mesh: mesh, materials: [material])
    }

    /// Move reticle to where the ARRoomPreviewViewModel says next.
    func update(direction: SIMD3<Float>, cameraOrigin: SIMD3<Float>, distance: Float = 1.5) {
        entity.position = Self.worldPosition(
            forDirection: direction, atDistance: distance, origin: cameraOrigin
        )
    }

    /// Pure computation for testing.
    static func worldPosition(
        forDirection direction: SIMD3<Float>,
        atDistance distance: Float,
        origin: SIMD3<Float>
    ) -> SIMD3<Float> {
        let len = simd_length(direction)
        guard len > 0.01 else { return origin }
        return origin + (direction / len) * distance
    }
}
