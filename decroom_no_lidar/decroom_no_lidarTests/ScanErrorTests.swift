import Testing
@testable import decroom_no_lidar

struct ScanErrorTests {

    @Test func everyCaseHasNonEmptyMessage() {
        let cases: [ScanError] = [
            .unsupportedDevice,
            .depthModelLoadFailed(detail: "x"),
            .arKitTrackingLimited,
            .arKitSessionFailed(detail: "x"),
            .noWallsDetected,
            .insufficientCoverage(binsFilled: 0, binsRequired: 12),
            .payloadBuildFailed(detail: "x"),
            .uploadNoServer,
            .uploadTimeout,
            .backendSchemaError(scanID: "abc", detail: "x"),
            .backendServerError(status: 500, detail: "x"),
            .noFurnitureDetected,
        ]
        for c in cases {
            #expect(!c.message.isEmpty, "\(c) has empty message")
        }
    }

    @Test func unsupportedDeviceMessageMentionsMinimumDevice() {
        #expect(ScanError.unsupportedDevice.message.contains("iPhone XS"))
    }

    @Test func backendSchemaErrorShowsScanID() {
        let e = ScanError.backendSchemaError(scanID: "deadbeef", detail: "bad json")
        #expect(e.message.contains("deadbeef"))
    }

    @Test func retryableCasesAreCorrect() {
        #expect(ScanError.uploadNoServer.isRetryable)
        #expect(ScanError.uploadTimeout.isRetryable)
        #expect(ScanError.backendServerError(status: 500, detail: "").isRetryable)
        #expect(ScanError.noWallsDetected.isRetryable)
        #expect(ScanError.insufficientCoverage(binsFilled: 0, binsRequired: 12).isRetryable)
        #expect(!ScanError.unsupportedDevice.isRetryable)
        #expect(!ScanError.depthModelLoadFailed(detail: "").isRetryable)
        #expect(!ScanError.backendSchemaError(scanID: "x", detail: "y").isRetryable)
    }
}
