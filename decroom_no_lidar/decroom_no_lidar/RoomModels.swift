//
//  RoomModels.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import Foundation
import SwiftUI
import simd

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

    // MARK: - Geometric normalization (Y-snap, stack, dedup)

    /// Lowercased object types that live on the floor. Their Y is forced
    /// to 0 at load time — backend-supplied Y is treated as noise.
    private static let floorFurnitureCategories: Set<String> = [
        "bed", "sofa", "couch", "chair", "armchair", "desk", "table",
        "dining table", "coffee table", "dresser", "wardrobe",
        "bookshelf", "nightstand", "fridge", "refrigerator",
        "toilet", "sink", "bathtub",
    ]

    /// Floor-furniture types that things can rest on (drives Y of stackables).
    private static let surfaceCategories: Set<String> = [
        "desk", "table", "dining table", "coffee table", "dresser",
        "nightstand", "bookshelf",
    ]

    /// Small items that sit on top of a surface when their footprint
    /// center overlaps one; otherwise they sit on the floor.
    private static let stackableCategories: Set<String> = [
        "lamp", "table lamp", "monitor", "tv", "vase", "potted plant",
        "plant", "picture", "book", "books", "computer", "laptop",
        "plate", "cup", "bowl", "clock",
    ]

    /// Apply geometric normalization to backend-supplied geometry:
    ///   1. Drop floor-furniture pairs that overlap by >50% (one is a
    ///      detection error — keep higher confidence).
    ///   2. Force Y=0 for floor furniture.
    ///   3. Raise stackable items to a surface's height when their XZ
    ///      center sits over that surface's footprint; else Y=0.
    func normalized() -> RoomPayload {
        let deduped = dedupOverlappingFloorFurniture()
        return deduped.groundAndStack()
    }

    private func dedupOverlappingFloorFurniture() -> RoomPayload {
        var keep = Array(repeating: true, count: objects.count)
        for i in 0..<objects.count {
            guard keep[i] else { continue }
            let a = objects[i]
            guard Self.floorFurnitureCategories.contains(a.type.lowercased()) else { continue }
            for j in (i + 1)..<objects.count {
                guard keep[j] else { continue }
                let b = objects[j]
                guard Self.floorFurnitureCategories.contains(b.type.lowercased()) else { continue }
                let overlap = Self.aabbOverlapRatio(a, b)
                guard overlap > 0.5 else { continue }
                let dropIdx = a.confidence >= b.confidence ? j : i
                let keepIdx = dropIdx == i ? j : i
                let dropped = objects[dropIdx]
                let kept = objects[keepIdx]
                keep[dropIdx] = false
                print(String(format: "[Decroom] dedup: dropping %@ id=%@ conf=%.2f — overlaps %@ id=%@ conf=%.2f by %.0f%%",
                             dropped.type, dropped.id, dropped.confidence,
                             kept.type, kept.id, kept.confidence,
                             overlap * 100))
                if dropIdx == i { break }
            }
        }
        let survivors = objects.enumerated().compactMap { keep[$0.offset] ? $0.element : nil }
        return with(objects: survivors)
    }

    private func groundAndStack() -> RoomPayload {
        let surfaces = objects.filter {
            Self.surfaceCategories.contains($0.type.lowercased())
        }
        let newObjects: [RoomObject] = objects.map { obj in
            let lc = obj.type.lowercased()
            let newY: Double
            if Self.stackableCategories.contains(lc),
               let surface = Self.firstSurfaceContaining(
                xz: SIMD2<Float>(Float(obj.position.x), Float(obj.position.z)),
                surfaces: surfaces)
            {
                newY = surface.metadata.height
            } else {
                newY = 0
            }
            let newPos = ObjectPosition(x: obj.position.x, y: newY, z: obj.position.z)
            return obj.with(position: newPos)
        }
        return with(objects: newObjects)
    }

    // MARK: - Geometric helpers

    private static func aabbOverlapRatio(_ a: RoomObject, _ b: RoomObject) -> Float {
        let aHalf = SceneSelection.rotatedHalfExtents(meta: a.metadata, rotationY: a.rotation_y)
        let bHalf = SceneSelection.rotatedHalfExtents(meta: b.metadata, rotationY: b.rotation_y)
        let aCx = Float(a.position.x), aCz = Float(a.position.z)
        let bCx = Float(b.position.x), bCz = Float(b.position.z)
        let dx = max(0, min(aCx + aHalf.x, bCx + bHalf.x) - max(aCx - aHalf.x, bCx - bHalf.x))
        let dz = max(0, min(aCz + aHalf.y, bCz + bHalf.y) - max(aCz - aHalf.y, bCz - bHalf.y))
        let intersection = dx * dz
        let aArea = (aHalf.x * 2) * (aHalf.y * 2)
        let bArea = (bHalf.x * 2) * (bHalf.y * 2)
        let smaller = min(aArea, bArea)
        return smaller > 1e-6 ? intersection / smaller : 0
    }

    private static func firstSurfaceContaining(
        xz: SIMD2<Float>, surfaces: [RoomObject]
    ) -> RoomObject? {
        surfaces.first { s in
            let half = SceneSelection.rotatedHalfExtents(meta: s.metadata, rotationY: s.rotation_y)
            let cx = Float(s.position.x), cz = Float(s.position.z)
            return xz.x >= cx - half.x && xz.x <= cx + half.x
                && xz.y >= cz - half.y && xz.y <= cz + half.y
        }
    }
}
