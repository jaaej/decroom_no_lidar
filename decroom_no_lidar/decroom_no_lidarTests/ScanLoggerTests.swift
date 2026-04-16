import Testing
import Foundation
@testable import decroom_no_lidar

struct ScanLoggerTests {

    @Test func newLoggerHasScanId() {
        let logger = ScanLogger(scanId: "test-123", directory: tempDir())
        #expect(logger.scanId == "test-123")
    }

    @Test func logEventAppendsToEvents() {
        let logger = ScanLogger(scanId: "test-123", directory: tempDir())
        logger.logEvent("scan_started", at: 0.0)
        logger.logEvent("first_plane_detected", at: 2.1, data: ["type": "wall"])
        #expect(logger.events.count == 2)
        #expect(logger.events[0].event == "scan_started")
        #expect(logger.events[1].data?["type"] == "wall")
    }

    @Test func finalizeWritesJSONFile() throws {
        let dir = tempDir()
        let logger = ScanLogger(scanId: "test-123", directory: dir)
        logger.recordDevice(DeviceInfo(model: "iPhone 15 Pro", ios: "17.2", chip: "A17 Pro", tier: .full))
        logger.logEvent("scan_started", at: 0.0)
        try logger.finalize()

        let files = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        #expect(files.contains { $0.contains("test-123") && $0.hasSuffix(".json") })
    }

    @Test func retentionDropsOldFilesAboveLimit() throws {
        let dir = tempDir()
        // Create 52 fake log files
        for i in 0..<52 {
            let timestamp = Int(Date().timeIntervalSince1970) - (52 - i) * 100
            let url = dir.appendingPathComponent("\(timestamp)_log\(i).json")
            try "{}".data(using: .utf8)!.write(to: url)
        }
        try ScanLogger.enforceRetention(directory: dir, keepLast: 50)
        let remaining = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        #expect(remaining.count == 50)
    }

    // Helper: create a unique temp directory per test
    private func tempDir() -> URL {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan_logger_test_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }
}
