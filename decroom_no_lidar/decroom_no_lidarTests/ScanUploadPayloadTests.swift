import Testing
import simd
import Foundation
@testable import decroom_no_lidar

struct ScanUploadPayloadTests {

    @Test func encodeRoomGeometryProducesParseableJSON() throws {
        let g = RoomGeometry(
            walls: [WallSegment(start: SIMD2(0, 0), end: SIMD2(3, 0), height: 2.5)],
            floorHeight: 0,
            ceilingHeight: 2.4,
            manhattanYaw: 0
        )
        let data = try ScanUploadPayload.encodeRoomGeometry(g)
        let decoded = try JSONDecoder().decode(
            ScanUploadPayload.RoomGeometryJSON.self, from: data
        )
        #expect(decoded.walls.count == 1)
        #expect(decoded.walls[0].start.x == 0)
        #expect(decoded.walls[0].end.x == 3)
        #expect(decoded.floor_height == 0)
        #expect(decoded.ceiling_height == 2.4)
        #expect(decoded.coordinate_system == "arkit_world_right_handed_y_up")
    }

    @Test func encodeKeyframesManifestFlattensPose() throws {
        let intr = Intrinsics(fx: 1500, fy: 1500, cx: 720, cy: 960, width: 1440, height: 1920)
        let pose = matrix_identity_float4x4
        let meta = ScanUploadPayload.KeyframeMetadata(
            filename: "frame_00.jpg",
            timestamp: 1.23,
            poseWorldFromCamera: pose,
            intrinsics: intr
        )
        let data = try ScanUploadPayload.encodeKeyframesManifest([meta])
        let decoded = try JSONDecoder().decode(
            ScanUploadPayload.KeyframesManifest.self, from: data
        )
        #expect(decoded.frames.count == 1)
        #expect(decoded.frames[0].filename == "frame_00.jpg")
        #expect(decoded.frames[0].pose_world_from_camera.count == 16)
        // Identity matrix row-major
        #expect(decoded.frames[0].pose_world_from_camera[0] == 1)
        #expect(decoded.frames[0].pose_world_from_camera[5] == 1)
        #expect(decoded.frames[0].pose_world_from_camera[10] == 1)
        #expect(decoded.frames[0].pose_world_from_camera[15] == 1)
        #expect(decoded.frames[0].intrinsics.fx == 1500)
    }

    @Test func manifestAlwaysHasEmptyDetectionsIn3a() throws {
        let intr = Intrinsics(fx: 1500, fy: 1500, cx: 720, cy: 960, width: 1440, height: 1920)
        let meta = ScanUploadPayload.KeyframeMetadata(
            filename: "frame_00.jpg",
            timestamp: 0,
            poseWorldFromCamera: matrix_identity_float4x4,
            intrinsics: intr
        )
        let data = try ScanUploadPayload.encodeKeyframesManifest([meta])
        let decoded = try JSONDecoder().decode(
            ScanUploadPayload.KeyframesManifest.self, from: data
        )
        #expect(decoded.frames[0].detections.isEmpty)
    }

    @Test func encodeYoloObjectsPopulatedListProducesParseableJSON() throws {
        let tracks = [
            ObjectTrack(id: UUID(), category: "bed",
                         latestPosition: SIMD3<Float>(2.0, 0.0, 1.5),
                         bestConfidence: 0.87, bestFrameIndex: 34,
                         observationCount: 12),
            ObjectTrack(id: UUID(), category: "potted plant",
                         latestPosition: SIMD3<Float>(0.3, 0.8, 2.9),
                         bestConfidence: 0.62, bestFrameIndex: 67,
                         observationCount: 5),
        ]
        let data = try ScanUploadPayload.encodeYoloObjects(tracks)
        let decoded = try JSONDecoder().decode(
            ScanUploadPayload.YoloObjectsManifest.self, from: data
        )
        #expect(decoded.objects.count == 2)
        let bed = decoded.objects[0]
        #expect(bed.category == "bed")
        #expect(bed.position == [2.0, 0.0, 1.5])
        #expect(bed.peak_confidence == 0.87)
        #expect(bed.view_count == 12)
        #expect(bed.best_frame_index == 34)
    }

    @Test func encodeYoloObjectsEmptyListSerializesAsEmptyArray() throws {
        let data = try ScanUploadPayload.encodeYoloObjects([])
        let decoded = try JSONDecoder().decode(
            ScanUploadPayload.YoloObjectsManifest.self, from: data
        )
        #expect(decoded.objects.isEmpty)
    }
}
