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
        // Create 52 fake log files with increasing filename timestamps.
        // File i=0 has oldest timestamp, i=51 has newest.
        let baseTimestamp = Int(Date().timeIntervalSince1970)
        for i in 0..<52 {
            let timestamp = baseTimestamp - (52 - i) * 100
            let url = dir.appendingPathComponent("\(timestamp)_log\(i).json")
            try "{}".data(using: .utf8)!.write(to: url)
        }
        try ScanLogger.enforceRetention(directory: dir, keepLast: 50)
        let remaining = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        #expect(remaining.count == 50)
        // The two oldest (i=0, i=1) should have been deleted.
        #expect(!remaining.contains { $0.contains("log0.json") })
        #expect(!remaining.contains { $0.contains("log1.json") })
        #expect(remaining.contains { $0.contains("log51.json") })
    }

    // Helper: create a unique temp directory per test
    private func tempDir() -> URL {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan_logger_test_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }
}
