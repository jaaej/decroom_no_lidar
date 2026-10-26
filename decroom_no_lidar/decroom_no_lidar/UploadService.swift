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
    /// Set this to your backend host's LAN IP (the primary path is Bonjour discovery).
    static let fallbackURL = "http://YOUR_SERVER_IP:8000/api/upload"
    private static let uploadTimeoutSeconds: TimeInterval = 1800
    private static let serverProbeTimeoutSeconds: TimeInterval = 8

    /// Resolves the best upload URL: Bonjour-discovered first, fallback second.
    @MainActor
    static var resolvedURL: String {
        ServerDiscovery.shared.uploadURL ?? fallbackURL
    }

    @MainActor
    private static var candidateUploadURLs: [String] {
        var urls = ServerDiscovery.shared.uploadURLs
        if !urls.contains(fallbackURL) {
            urls.append(fallbackURL)
        }
        return urls
    }

    private static func healthURL(for uploadURL: URL, nonce: String) -> URL? {
        guard var components = URLComponents(url: uploadURL, resolvingAgainstBaseURL: false) else {
            return nil
        }
        components.path = "/health"
        components.queryItems = [URLQueryItem(name: "nonce", value: nonce)]
        components.fragment = nil
        return components.url
    }

    private enum ProbeResult {
        case unreachable
        case wrongKey
        case verified
    }

    /// GET /health?nonce=N and check the server answers with
    /// HMAC(key, "decroom-health:N"), i.e. it holds our server key.
    /// Any device on the network can advertise _decroom._tcp; this keeps
    /// us from uploading room photos to one that isn't our backend.
    private static func probeServer(uploadURL: URL, key: String) async -> ProbeResult {
        let nonce = ServerKey.randomNonce()
        guard let healthURL = healthURL(for: uploadURL, nonce: nonce) else {
            print("[UploadService] Health probe skipped: invalid health URL for \(uploadURL.absoluteString)")
            return .unreachable
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
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                print("[UploadService] Health probe failed: non-HTTP response for \(uploadURL.absoluteString)")
                return .unreachable
            }
            print("[UploadService] Health probe \(uploadURL.absoluteString) -> HTTP \(http.statusCode)")
            guard (200...299).contains(http.statusCode) else { return .unreachable }
            let proof = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["proof"] as? String
            guard proof == ServerKey.expectedHealthProof(nonce: nonce, key: key) else {
                print("[UploadService] Health probe \(uploadURL.absoluteString): key proof missing or wrong")
                return .wrongKey
            }
            return .verified
        } catch {
            print("[UploadService] Health probe error for \(uploadURL.absoluteString): \(error.localizedDescription)")
            return .unreachable
        }
    }

    /// First candidate that proves it holds our server key. Throws
    /// `serverKeyMismatch` if servers answered but none had the key.
    private static func resolveUploadURL(key: String) async throws -> URL {
        let candidates = await MainActor.run { candidateUploadURLs }
        print("[UploadService] Candidate upload URLs: \(candidates)")
        var sawWrongKey = false
        for candidate in candidates {
            guard let parsed = URL(string: candidate) else {
                print("[UploadService] Skipping invalid candidate URL: \(candidate)")
                continue
            }
            switch await probeServer(uploadURL: parsed, key: key) {
            case .verified:
                print("[UploadService] Selected upload URL: \(parsed.absoluteString)")
                return parsed
            case .wrongKey:
                sawWrongKey = true
            case .unreachable:
                break
            }
        }
        print("[UploadService] No verified backend found among candidates")
        if sawWrongKey { throw ScanError.serverKeyMismatch }
        throw UploadError.noServerFound
    }

    // MARK: - Phone-first upload (Plan 3a)

    /// Upload a phone-first payload: geometry JSON, PLY, keyframes + manifest,
    /// scan log. Maps backend 4xx/5xx/network errors into `ScanError` cases.
    static func uploadPhoneFirst(_ payload: ScanUploadPayload) async throws -> RoomPayload {
        guard let key = ServerKey.load() else {
            throw ScanError.serverKeyMissing
        }
        let url: URL
        do {
            url = try await resolvePhoneFirstUploadURL(key: key)
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
        request.setValue(ServerKey.uploadAuthValue(key: key), forHTTPHeaderField: ServerKey.authHeader)
        // Plan 3c: backend runs SpatialLM (~7s) + Grounding DINO on all
        // ~66 keyframes (~33s on RTX 4090) + raycast/clustering/consensus.
        // Real scans land around 40-50s; 180s gives ample headroom for
        // network variability + cold-start cases. Was 60s pre-Plan-3c.
        request.timeoutInterval = 180
        request.httpBody = body

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 180
        config.timeoutIntervalForResource = 180
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
        case 401:
            throw ScanError.serverKeyMismatch
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
    private static func resolvePhoneFirstUploadURL(key: String) async throws -> URL {
        let base = try await resolveUploadURL(key: key)
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
        appendPart(name: "yolo_objects", filename: "yolo_objects.json",
                    contentType: "application/json",
                    bytes: payload.yoloObjectsJSON)

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
