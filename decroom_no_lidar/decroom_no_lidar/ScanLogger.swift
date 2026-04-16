//
//  ScanLogger.swift
//  decroom_no_lidar
//
//  Per-scan JSON log writer. Captures events, timing, device info, and
//  diagnostics for post-mortem analysis. Retains last 50 logs in
//  Documents/scan_logs/.
//

import Foundation

struct DeviceInfo: Codable {
    let model: String
    let ios: String
    let chip: String
    let tier: DeviceTier
}

struct ScanEvent: Codable {
    let t: Double           // Seconds since scan start
    let event: String
    let data: [String: String]?

    enum CodingKeys: String, CodingKey {
        case t = "t_sec"
        case event
        case data
    }
}

final class ScanLogger {
    let scanId: String
    let startedAt: Date
    private let directory: URL
    private let queue = DispatchQueue(label: "decroom.scan-logger", qos: .utility)

    private var _events: [ScanEvent] = []
    private var _device: DeviceInfo?

    /// Snapshot of events (thread-safe copy).
    var events: [ScanEvent] { queue.sync { _events } }
    /// Snapshot of device info (thread-safe).
    var device: DeviceInfo? { queue.sync { _device } }

    init(scanId: String = String(UUID().uuidString.prefix(8)).lowercased(),
         directory: URL = ScanLogger.defaultDirectory()) {
        self.scanId = scanId
        self.startedAt = Date()
        self.directory = directory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    func recordDevice(_ info: DeviceInfo) {
        queue.async { self._device = info }
    }

    func logEvent(_ event: String, at t: Double, data: [String: String]? = nil) {
        let ev = ScanEvent(t: t, event: event, data: data)
        queue.async { self._events.append(ev) }
    }

    /// Write log to disk. Call once when scan + upload are done.
    func finalize() throws {
        // Snapshot state synchronously under the queue.
        let (eventsSnapshot, deviceSnapshot) = queue.sync {
            (_events, _device)
        }

        let timestamp = Int(startedAt.timeIntervalSince1970)
        let filename = "\(timestamp)_\(scanId).json"
        let url = directory.appendingPathComponent(filename)

        let log = ScanLog(
            scanId: scanId,
            startedAt: startedAt,
            completedAt: Date(),
            device: deviceSnapshot,
            events: eventsSnapshot
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(log)
        try data.write(to: url, options: .atomic)
    }

    /// Keep only the `keepLast` most recent log files in the directory.
    /// Parses the unix-timestamp prefix in filenames so behavior survives
    /// filesystem metadata changes (iCloud restore, copy, etc.).
    static func enforceRetention(directory: URL, keepLast: Int) throws {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(at: directory,
                                                   includingPropertiesForKeys: nil,
                                                   options: [])
            .filter { $0.pathExtension == "json" }

        guard contents.count > keepLast else { return }

        // Parse the leading "{unix_timestamp}_" from each filename.
        // Files without a parseable prefix sort to the oldest end.
        let sortedByTimestamp = contents.sorted { a, b in
            timestampPrefix(of: a) > timestampPrefix(of: b)  // Newest first
        }

        for url in sortedByTimestamp.dropFirst(keepLast) {
            try? fm.removeItem(at: url)
        }
    }

    /// Parse the leading integer before the first `_` in the filename.
    /// Returns Int.min if unparseable so files without prefix sort oldest.
    private static func timestampPrefix(of url: URL) -> Int {
        let name = url.deletingPathExtension().lastPathComponent
        guard let underscore = name.firstIndex(of: "_") else { return Int.min }
        return Int(name[..<underscore]) ?? Int.min
    }

    static func defaultDirectory() -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("scan_logs")
    }
}

/// Serialized log structure — what gets written to disk.
private struct ScanLog: Codable {
    let scanId: String
    let startedAt: Date
    let completedAt: Date
    let device: DeviceInfo?
    let events: [ScanEvent]
}
