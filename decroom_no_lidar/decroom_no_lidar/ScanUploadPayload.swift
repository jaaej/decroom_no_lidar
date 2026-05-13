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
    let yoloObjectsJSON: Data
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

    /// Matches yolo_objects.json schema from the design spec.
    /// One entry per deduped 3D-tracked object from on-device ObjectTracker.
    struct YoloObject: Codable {
        let category: String
        let position: [Float]       // [x, y, z] ARKit world frame, meters
        let peak_confidence: Float
        let view_count: Int
        let best_frame_index: Int
    }
    struct YoloObjectsManifest: Codable {
        let objects: [YoloObject]
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

    static func encodeYoloObjects(_ tracks: [ObjectTrack]) throws -> Data {
        let entries = tracks.map { t in
            YoloObject(
                category: t.category,
                position: [t.latestPosition.x, t.latestPosition.y, t.latestPosition.z],
                peak_confidence: t.bestConfidence,
                view_count: t.observationCount,
                best_frame_index: t.bestFrameIndex
            )
        }
        let manifest = YoloObjectsManifest(objects: entries)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        return try encoder.encode(manifest)
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

    // MARK: - Persist to disk (QA convenience)

    /// Directory used to persist a single most-recent scan payload.
    /// Present only when a prior scan succeeded.
    static var lastCaptureDirectory: URL {
        let docs = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask
        ).first!
        return docs.appendingPathComponent("last_payload")
    }

    /// True if a saved payload exists on disk (i.e., a previous scan
    /// completed successfully and was persisted).
    static var hasPersistedCapture: Bool {
        FileManager.default.fileExists(
            atPath: lastCaptureDirectory.path
        )
    }

    /// Persist this payload to disk. Overwrites any prior saved payload.
    /// Fires and forgets — errors are logged but not surfaced, because
    /// this is a QA convenience, not a user-facing feature.
    func saveAsLastCapture() {
        let dir = Self.lastCaptureDirectory
        do {
            let fm = FileManager.default
            if fm.fileExists(atPath: dir.path) {
                try fm.removeItem(at: dir)
            }
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            try roomGeometryJSON.write(
                to: dir.appendingPathComponent("room_geometry.json"))
            try pointCloudPLY.write(
                to: dir.appendingPathComponent("point_cloud.ply"))
            try scanLogJSON.write(
                to: dir.appendingPathComponent("scan_log.json"))
            try yoloObjectsJSON.write(
                to: dir.appendingPathComponent("yolo_objects.json"))
            // keyframes.json manifest rebuilt from the metas + ID
            let manifestData = try Self.encodeKeyframesManifest(
                keyframes.map { $0.meta }
            )
            try manifestData.write(
                to: dir.appendingPathComponent("keyframes.json"))
            for (i, kf) in keyframes.enumerated() {
                let filename = kf.meta.filename.isEmpty
                    ? "frame_\(String(format: "%02d", i)).jpg"
                    : kf.meta.filename
                try kf.jpeg.write(to: dir.appendingPathComponent(filename))
            }
            // scanID + deviceInfo blob
            let sidebar: [String: String] = [
                "scan_id": scanID,
                "device_model": deviceInfo.model,
                "device_ios": deviceInfo.ios,
                "device_chip": deviceInfo.chip,
                "device_tier": deviceInfo.tier.rawValue,
            ]
            let sidebarData = try JSONEncoder().encode(sidebar)
            try sidebarData.write(
                to: dir.appendingPathComponent("sidebar.json"))
            print("[ScanUploadPayload] persisted last capture → \(dir.path)")
        } catch {
            print("[ScanUploadPayload] save failed: \(error.localizedDescription)")
        }
    }

    /// Load the most recent persisted payload from disk.
    /// Returns nil if the directory doesn't exist or is malformed.
    static func loadLastCapture() -> ScanUploadPayload? {
        let dir = lastCaptureDirectory
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.path) else { return nil }

        do {
            let roomGeometryJSON = try Data(
                contentsOf: dir.appendingPathComponent("room_geometry.json"))
            let pointCloudPLY = try Data(
                contentsOf: dir.appendingPathComponent("point_cloud.ply"))
            let scanLogJSON = try Data(
                contentsOf: dir.appendingPathComponent("scan_log.json"))
            let manifestData = try Data(
                contentsOf: dir.appendingPathComponent("keyframes.json"))
            let yoloData = (try? Data(
                contentsOf: dir.appendingPathComponent("yolo_objects.json")))
                ?? Data("{\"objects\":[]}".utf8)
            let sidebarData = try Data(
                contentsOf: dir.appendingPathComponent("sidebar.json"))

            let manifest = try JSONDecoder().decode(
                KeyframesManifest.self, from: manifestData)
            let sidebar = try JSONDecoder().decode(
                [String: String].self, from: sidebarData)

            var keyframes: [(jpeg: Data, meta: KeyframeMetadata)] = []
            for frame in manifest.frames {
                let jpeg = try Data(
                    contentsOf: dir.appendingPathComponent(frame.filename))
                let pose = Self.matrixFromRowMajor(frame.pose_world_from_camera)
                let intr = Intrinsics(
                    fx: frame.intrinsics.fx, fy: frame.intrinsics.fy,
                    cx: frame.intrinsics.cx, cy: frame.intrinsics.cy,
                    width: frame.intrinsics.width, height: frame.intrinsics.height
                )
                let meta = KeyframeMetadata(
                    filename: frame.filename,
                    timestamp: frame.timestamp,
                    poseWorldFromCamera: pose,
                    intrinsics: intr
                )
                keyframes.append((jpeg: jpeg, meta: meta))
            }

            let scanID = sidebar["scan_id"] ?? "unknown"
            let tier = DeviceTier(rawValue: sidebar["device_tier"] ?? "fallback")
                ?? .fallback
            let deviceInfo = DeviceInfo(
                model: sidebar["device_model"] ?? "unknown",
                ios: sidebar["device_ios"] ?? "unknown",
                chip: sidebar["device_chip"] ?? "unknown",
                tier: tier
            )

            return ScanUploadPayload(
                scanID: scanID,
                roomGeometryJSON: roomGeometryJSON,
                pointCloudPLY: pointCloudPLY,
                keyframes: keyframes,
                scanLogJSON: scanLogJSON,
                yoloObjectsJSON: yoloData,
                deviceInfo: deviceInfo
            )
        } catch {
            print("[ScanUploadPayload] load failed: \(error.localizedDescription)")
            return nil
        }
    }

    private static func matrixFromRowMajor(_ m: [Float]) -> simd_float4x4 {
        guard m.count == 16 else { return matrix_identity_float4x4 }
        return simd_float4x4(
            SIMD4<Float>(m[0],  m[4],  m[8],  m[12]),
            SIMD4<Float>(m[1],  m[5],  m[9],  m[13]),
            SIMD4<Float>(m[2],  m[6],  m[10], m[14]),
            SIMD4<Float>(m[3],  m[7],  m[11], m[15])
        )
    }
}
