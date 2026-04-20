//
//  UploadService.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import Foundation

enum UploadError: Error, LocalizedError {
    case imageConversionFailed
    case invalidURL
    case serverError(String)
    case decodingFailed(String)
    case noServerFound

    var errorDescription: String? {
        switch self {
        case .imageConversionFailed: return "Failed to convert image to JPEG."
        case .invalidURL: return "Invalid server URL."
        case .serverError(let msg): return "Server error: \(msg)"
        case .decodingFailed(let msg): return "Decoding failed: \(msg)"
        case .noServerFound: return "No Decroom server found on the network. Make sure the backend is running."
        }
    }
}

struct UploadService {
    /// Fallback URL if Bonjour discovery hasn't found the server yet.
    static let fallbackURL = "http://192.168.0.125:8000/api/upload"
    private static let uploadTimeoutSeconds: TimeInterval = 1800
    private static let serverProbeTimeoutSeconds: TimeInterval = 8

    /// Resolves the best upload URL: Bonjour-discovered first, fallback second.
    @MainActor
    static var resolvedURL: String {
        ServerDiscovery.shared.uploadURL ?? fallbackURL
    }

    @MainActor
    private static var candidateUploadURLs: [String] {
        var urls: [String] = []
        if let discovered = ServerDiscovery.shared.uploadURL {
            urls.append(discovered)
        }
        if !urls.contains(fallbackURL) {
            urls.append(fallbackURL)
        }
        return urls
    }

    private static func healthURL(for uploadURL: URL) -> URL? {
        guard var components = URLComponents(url: uploadURL, resolvingAgainstBaseURL: false) else {
            return nil
        }
        components.path = "/health"
        components.query = nil
        components.fragment = nil
        return components.url
    }

    private static func isServerReachable(uploadURL: URL) async -> Bool {
        guard let healthURL = healthURL(for: uploadURL) else {
            print("[UploadService] Health probe skipped: invalid health URL for \(uploadURL.absoluteString)")
            return false
        }
        var request = URLRequest(url: healthURL)
        request.httpMethod = "GET"
        request.timeoutInterval = serverProbeTimeoutSeconds

        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = serverProbeTimeoutSeconds
        configuration.timeoutIntervalForResource = serverProbeTimeoutSeconds
        let session = URLSession(configuration: configuration)
        defer {
            session.finishTasksAndInvalidate()
        }

        do {
            let (_, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                print("[UploadService] Health probe failed: non-HTTP response for \(healthURL.absoluteString)")
                return false
            }
            print("[UploadService] Health probe \(healthURL.absoluteString) -> HTTP \(http.statusCode)")
            return (200...299).contains(http.statusCode)
        } catch {
            print("[UploadService] Health probe error for \(healthURL.absoluteString): \(error.localizedDescription)")
            return false
        }
    }

    private static func resolveUploadURL() async throws -> URL {
        let candidates = await MainActor.run { candidateUploadURLs }
        print("[UploadService] Candidate upload URLs: \(candidates)")
        for candidate in candidates {
            guard let parsed = URL(string: candidate) else {
                print("[UploadService] Skipping invalid candidate URL: \(candidate)")
                continue
            }
            if await isServerReachable(uploadURL: parsed) {
                print("[UploadService] Selected upload URL: \(parsed.absoluteString)")
                return parsed
            }
        }
        print("[UploadService] No reachable backend found among candidates")
        throw UploadError.noServerFound
    }

    // MARK: - Phone-first upload (Plan 3a)

    /// Upload a phone-first payload: geometry JSON, PLY, keyframes + manifest,
    /// scan log. Maps backend 4xx/5xx/network errors into `ScanError` cases.
    static func uploadPhoneFirst(_ payload: ScanUploadPayload) async throws -> RoomPayload {
        let url: URL
        do {
            url = try await resolvePhoneFirstUploadURL()
        } catch let e as ScanError {
            throw e
        } catch {
            throw ScanError.uploadNoServer
        }
        print("[UploadService] Phone-first upload → \(url.absoluteString) (scan=\(payload.scanID))")

        let boundary = UUID().uuidString

        // Build multipart body off-thread (JPEG concat is cheap but keep main responsive).
        let body: Data
        do {
            body = try await Task.detached(priority: .userInitiated) {
                try UploadService._buildPhoneFirstMultipartBody(payload, boundary: boundary)
            }.value
        } catch {
            throw ScanError.payloadBuildFailed(
                detail: "multipart build: \(error.localizedDescription)"
            )
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 60  // was 1800 for old SfM path; 60s is plenty for SpatialLM
        request.httpBody = body

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 60
        let session = URLSession(configuration: config)
        defer { session.finishTasksAndInvalidate() }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch let err as URLError where err.code == .timedOut {
            throw ScanError.uploadTimeout
        } catch {
            throw ScanError.uploadNoServer
        }

        guard let http = response as? HTTPURLResponse else {
            throw ScanError.backendServerError(status: -1, detail: "non-HTTP response")
        }

        switch http.statusCode {
        case 200:
            do {
                return try JSONDecoder().decode(RoomPayload.self, from: data)
            } catch {
                throw ScanError.backendSchemaError(
                    scanID: payload.scanID,
                    detail: "response decode: \(error.localizedDescription)"
                )
            }
        case 400..<500:
            let body = String(data: data, encoding: .utf8) ?? "(unparseable body)"
            throw ScanError.backendSchemaError(scanID: payload.scanID, detail: body)
        default:
            let body = String(data: data, encoding: .utf8) ?? "(unparseable body)"
            throw ScanError.backendServerError(status: http.statusCode, detail: body)
        }
    }

    /// Resolves the phone-first upload URL (Bonjour first, fallback second).
    /// Swaps the path component to `/api/upload/phone-first`.
    private static func resolvePhoneFirstUploadURL() async throws -> URL {
        let base = try await resolveUploadURL()
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false) else {
            throw ScanError.uploadNoServer
        }
        components.path = "/api/upload/phone-first"
        components.query = nil
        guard let url = components.url else {
            throw ScanError.uploadNoServer
        }
        return url
    }

    /// Exposed for tests — builds the multipart body without a network call.
    static func _buildPhoneFirstMultipartBody(
        _ payload: ScanUploadPayload,
        boundary: String
    ) throws -> Data {
        var data = Data()
        func appendPart(name: String, filename: String,
                         contentType: String, bytes: Data) {
            data.append("--\(boundary)\r\n")
            data.append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n")
            data.append("Content-Type: \(contentType)\r\n\r\n")
            data.append(bytes)
            data.append("\r\n")
        }

        let manifestJSON = try ScanUploadPayload.encodeKeyframesManifest(
            payload.keyframes.map { $0.meta }
        )

        appendPart(name: "room_geometry", filename: "room_geometry.json",
                    contentType: "application/json",
                    bytes: payload.roomGeometryJSON)
        appendPart(name: "point_cloud", filename: "point_cloud.ply",
                    contentType: "application/octet-stream",
                    bytes: payload.pointCloudPLY)
        appendPart(name: "keyframes_manifest", filename: "keyframes.json",
                    contentType: "application/json",
                    bytes: manifestJSON)
        for (i, kf) in payload.keyframes.enumerated() {
            appendPart(name: "keyframes", filename: kf.meta.filename.isEmpty
                                                     ? "frame_\(i).jpg"
                                                     : kf.meta.filename,
                        contentType: "image/jpeg",
                        bytes: kf.jpeg)
        }
        appendPart(name: "scan_log", filename: "scan_log.json",
                    contentType: "application/json",
                    bytes: payload.scanLogJSON)

        data.append("--\(boundary)--\r\n")
        return data
    }
}

// MARK: - Data Extension for Multipart

private extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }
}
