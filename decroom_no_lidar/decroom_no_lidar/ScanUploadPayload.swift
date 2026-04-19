//
//  ScanUploadPayload.swift
//  decroom_no_lidar
//
//  Container for everything the phone hands to the backend in a
//  phone-first upload: walls/floor/ceiling JSON, binary PLY point
//  cloud, 10 keyframe JPEGs + per-frame metadata, and the scan log.
//

import Foundation
import simd

struct ScanUploadPayload {
    struct KeyframeMetadata {
        let filename: String
        let timestamp: Double
        let poseWorldFromCamera: simd_float4x4
        let intrinsics: Intrinsics
        // detections intentionally empty in 3a (YOLO output goes to scan_log only)
    }

    let scanID: String
    let roomGeometryJSON: Data
    let pointCloudPLY: Data
    let keyframes: [(jpeg: Data, meta: KeyframeMetadata)]
    let scanLogJSON: Data
    let deviceInfo: DeviceInfo

    /// Matches room_geometry.json schema from the design spec.
    struct RoomGeometryJSON: Codable {
        struct Point: Codable { let x: Float; let z: Float }
        struct WallEntry: Codable {
            let start: Point
            let end: Point
        }
        let walls: [WallEntry]
        let floor_height: Float
        let ceiling_height: Float
        let manhattan_yaw: Float
        let coordinate_system: String
    }

    /// Matches keyframes.json manifest schema.
    struct KeyframesManifest: Codable {
        struct Intrinsics: Codable {
            let fx: Float
            let fy: Float
            let cx: Float
            let cy: Float
            let width: Int
            let height: Int
        }
        struct FrameEntry: Codable {
            let filename: String
            let timestamp: Double
            /// 16 floats, row-major 4x4 pose matrix (world_from_camera).
            let pose_world_from_camera: [Float]
            let intrinsics: Intrinsics
            let detections: [String]  // always [] in 3a
        }
        let frames: [FrameEntry]
    }

    // MARK: - JSON serialization helpers

    static func encodeRoomGeometry(_ geometry: RoomGeometry) throws -> Data {
        let json = RoomGeometryJSON(
            walls: geometry.walls.map {
                RoomGeometryJSON.WallEntry(
                    start: .init(x: $0.start.x, z: $0.start.y),
                    end: .init(x: $0.end.x, z: $0.end.y)
                )
            },
            floor_height: geometry.floorHeight,
            ceiling_height: geometry.ceilingHeight,
            manhattan_yaw: geometry.manhattanYaw,
            coordinate_system: "arkit_world_right_handed_y_up"
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        return try encoder.encode(json)
    }

    static func encodeKeyframesManifest(_ frames: [KeyframeMetadata]) throws -> Data {
        let entries = frames.map { kf -> KeyframesManifest.FrameEntry in
            let p = kf.poseWorldFromCamera
            // Row-major flatten of 4x4 (columns are stored contiguous in simd, but
            // JSON should be row-major for server-side convenience).
            let flat: [Float] = [
                p.columns.0.x, p.columns.1.x, p.columns.2.x, p.columns.3.x,
                p.columns.0.y, p.columns.1.y, p.columns.2.y, p.columns.3.y,
                p.columns.0.z, p.columns.1.z, p.columns.2.z, p.columns.3.z,
                p.columns.0.w, p.columns.1.w, p.columns.2.w, p.columns.3.w,
            ]
            return KeyframesManifest.FrameEntry(
                filename: kf.filename,
                timestamp: kf.timestamp,
                pose_world_from_camera: flat,
                intrinsics: .init(
                    fx: kf.intrinsics.fx, fy: kf.intrinsics.fy,
                    cx: kf.intrinsics.cx, cy: kf.intrinsics.cy,
                    width: kf.intrinsics.width, height: kf.intrinsics.height
                ),
                detections: []
            )
        }
        let manifest = KeyframesManifest(frames: entries)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        return try encoder.encode(manifest)
    }
}
