import Testing
import Foundation
import simd
@testable import decroom_no_lidar

struct UploadServicePhoneFirstTests {

    private func makePayload() -> ScanUploadPayload {
        let intr = Intrinsics(fx: 1500, fy: 1500, cx: 720, cy: 960, width: 1440, height: 1920)
        let meta = ScanUploadPayload.KeyframeMetadata(
            filename: "frame_00.jpg",
            timestamp: 1.23,
            poseWorldFromCamera: matrix_identity_float4x4,
            intrinsics: intr
        )
        return ScanUploadPayload(
            scanID: "testscan",
            roomGeometryJSON: #"{"walls":[]}"#.data(using: .utf8)!,
            pointCloudPLY: Data("ply".utf8),
            keyframes: [(Data("jpeg-bytes".utf8), meta)],
            scanLogJSON: Data("{}".utf8),
            deviceInfo: DeviceInfo(model: "test", ios: "17.0",
                                    chip: "test", tier: .full)
        )
    }

    @Test func multipartBodyContainsAllRequiredParts() throws {
        let payload = makePayload()
        let body = try UploadService._buildPhoneFirstMultipartBody(payload, boundary: "XXX")
        let bodyStr = String(data: body, encoding: .utf8) ?? ""
        #expect(bodyStr.contains(#"name="room_geometry""#))
        #expect(bodyStr.contains(#"name="point_cloud""#))
        #expect(bodyStr.contains(#"name="keyframes_manifest""#))
        #expect(bodyStr.contains(#"name="keyframes""#))
        #expect(bodyStr.contains(#"name="scan_log""#))
        #expect(bodyStr.hasSuffix("--XXX--\r\n"))
    }

    @Test func multipartBodyUsesCorrectContentTypes() throws {
        let payload = makePayload()
        let body = try UploadService._buildPhoneFirstMultipartBody(payload, boundary: "YYY")
        let bodyStr = String(data: body, encoding: .utf8) ?? ""
        #expect(bodyStr.contains("Content-Type: application/json"))
        #expect(bodyStr.contains("Content-Type: application/octet-stream"))
        #expect(bodyStr.contains("Content-Type: image/jpeg"))
    }

    @Test func multipartBodyIncludesJpegBytes() throws {
        let payload = makePayload()
        let body = try UploadService._buildPhoneFirstMultipartBody(payload, boundary: "ZZZ")
        // The JPEG payload string should appear in the body as-is (small test JPEG is "jpeg-bytes").
        let bodyStr = String(data: body, encoding: .utf8) ?? ""
        #expect(bodyStr.contains("jpeg-bytes"))
    }
}
