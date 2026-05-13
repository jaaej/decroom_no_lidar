//
//  RoomModels.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import Foundation
import SwiftUI

// MARK: - Room Payload (matches backend JSON)

struct RoomPayload: Codable, Identifiable {
    var id: String { room.id }
    let room: RoomInfo
    let objects: [RoomObject]

    var envelope: RoomInfo { room }

    enum CodingKeys: String, CodingKey {
        case room
        case objects
    }
}

struct RoomInfo: Codable, Identifiable {
    let id: String
    let width_m: Double
    let length_m: Double
    let height_m: Double
    let walls: [Wall]

    var width: Double { width_m }
    var length: Double { length_m }
    var height: Double { height_m }
}

struct Wall: Codable, Identifiable {
    let id: String
    let start: [Double]
    let end: [Double]
    let has_window: Bool
    let has_door: Bool
}

struct RoomObject: Codable, Identifiable {
    let id: String
    let type: String
    let asset_id: String
    let position: ObjectPosition
    let rotation_y: Double
    let scale: Double
    let confidence: Double
    let metadata: ObjectMetadata
}

struct ObjectPosition: Codable {
    let x: Double
    let y: Double
    let z: Double
}

struct ObjectMetadata: Codable {
    let width: Double
    let length: Double
    let height: Double
}

// MARK: - Plan 4a helpers

extension RoomObject {
    /// Returns a copy of this RoomObject with optionally-changed position
    /// or rotation_y. All other fields preserved.
    func with(
        position: ObjectPosition? = nil,
        rotationY: Double? = nil
    ) -> RoomObject {
        RoomObject(
            id: id,
            type: type,
            asset_id: asset_id,
            position: position ?? self.position,
            rotation_y: rotationY ?? self.rotation_y,
            scale: scale,
            confidence: confidence,
            metadata: metadata
        )
    }
}

extension RoomPayload {
    /// Returns a copy of this RoomPayload with a new objects list.
    func with(objects: [RoomObject]) -> RoomPayload {
        RoomPayload(room: room, objects: objects)
    }
}
