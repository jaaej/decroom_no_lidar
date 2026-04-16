//
//  PlaneAnchorDescriptor.swift
//  decroom_no_lidar
//
//  Protocol abstraction over ARPlaneAnchor so RoomGeometryBuilder can be
//  unit-tested without a live ARKit session.
//

import Foundation
import ARKit
import simd

enum PlaneAlignment {
    case horizontal  // floor or ceiling
    case vertical    // wall
}

protocol PlaneAnchorDescriptor {
    var identifier: UUID { get }
    var planeAlignment: PlaneAlignment { get }
    var transform: simd_float4x4 { get }  // anchor-to-world
    var center: SIMD3<Float> { get }       // local to anchor
    var extent: SIMD3<Float> { get }       // local extents
}

/// Bridge real ARPlaneAnchor to the protocol.
extension ARPlaneAnchor: PlaneAnchorDescriptor {
    var planeAlignment: PlaneAlignment {
        switch self.alignment {
        case .horizontal: return .horizontal
        case .vertical: return .vertical
        @unknown default: return .horizontal
        }
    }
}
