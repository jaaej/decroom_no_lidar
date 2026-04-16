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
}

final class ScanLogger {
    let scanId: String
    let startedAt: Date
    private let directory: URL

    private(set) var events: [ScanEvent] = []
    private(set) var device: DeviceInfo?

    init(scanId: String = UUID().uuidString.prefix(8).lowercased(),
         directory: URL = ScanLogger.defaultDirectory()) {
        self.scanId = String(scanId)
        self.startedAt = Date()
        self.directory = directory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    func recordDevice(_ info: DeviceInfo) {
        self.device = info
    }

    func logEvent(_ event: String, at t: Double, data: [String: String]? = nil) {
        events.append(ScanEvent(t: t, event: event, data: data))
    }

    /// Write log to disk. Call once when scan + upload are done.
    func finalize() throws {
        let timestamp = Int(startedAt.timeIntervalSince1970)
        let filename = "\(timestamp)_\(scanId).json"
        let url = directory.appendingPathComponent(filename)

        let payload: [String: Any] = [
            "scan_id": scanId,
            "started_at": ISO8601DateFormatter().string(from: startedAt),
            "completed_at": ISO8601DateFormatter().string(from: Date()),
            "device": device.map { [
                "model": $0.model,
                "ios": $0.ios,
                "chip": $0.chip,
                "tier": $0.tier.rawValue
            ] } as Any? ?? NSNull(),
            "events": events.map { [
                "t": $0.t,
                "event": $0.event,
                "data": $0.data as Any? ?? NSNull()
            ] }
        ]

        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    /// Keep only the `keepLast` most recent log files in the directory.
    static func enforceRetention(directory: URL, keepLast: Int) throws {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(at: directory,
                                                   includingPropertiesForKeys: [.creationDateKey],
                                                   options: [])
            .filter { $0.pathExtension == "json" }

        guard contents.count > keepLast else { return }

        let sortedByDate = contents.sorted { a, b in
            let aDate = (try? a.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            let bDate = (try? b.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            return aDate > bDate  // Newest first
        }

        for url in sortedByDate.dropFirst(keepLast) {
            try? fm.removeItem(at: url)
        }
    }

    static func defaultDirectory() -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("scan_logs")
    }
}
