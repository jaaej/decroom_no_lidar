//
//  UploadService.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import Foundation
import UIKit

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

    /// Upload from disk-based JPEG files (video mode) — avoids holding CGImages in memory.
    static func uploadFromDisk(
        jpegPaths: [URL],
        metadataJSON: String?
    ) async throws -> RoomPayload {
        let url = try await resolveUploadURL()
        print("[UploadService] Uploading \(jpegPaths.count) keyframes from disk to \(url.absoluteString)")

        let boundary = UUID().uuidString

        // Build multipart body OFF the main thread — reads JPEGs from disk one at a time
        let body: Data = try await Task.detached(priority: .userInitiated) {
            var bodyData = Data()

            for (index, jpegURL) in jpegPaths.enumerated() {
                let jpegData = try Data(contentsOf: jpegURL)

                bodyData.append("--\(boundary)\r\n")
                bodyData.append("Content-Disposition: form-data; name=\"frames\"; filename=\"frame_\(index).jpg\"\r\n")
                bodyData.append("Content-Type: image/jpeg\r\n\r\n")
                bodyData.append(jpegData)
                bodyData.append("\r\n")
            }

            if let metadata = metadataJSON, let metaData = metadata.data(using: .utf8) {
                bodyData.append("--\(boundary)\r\n")
                bodyData.append("Content-Disposition: form-data; name=\"metadata\"; filename=\"metadata.json\"\r\n")
                bodyData.append("Content-Type: application/json\r\n\r\n")
                bodyData.append(metaData)
                bodyData.append("\r\n")
            }

            bodyData.append("--\(boundary)--\r\n")
            return bodyData
        }.value

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = uploadTimeoutSeconds
        request.httpBody = body

        let configuration = URLSessionConfiguration.default
        configuration.waitsForConnectivity = false
        configuration.timeoutIntervalForRequest = uploadTimeoutSeconds
        configuration.timeoutIntervalForResource = uploadTimeoutSeconds
        configuration.allowsExpensiveNetworkAccess = true
        configuration.allowsConstrainedNetworkAccess = true
        let session = URLSession(configuration: configuration)
        defer {
            session.finishTasksAndInvalidate()
        }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch let error as URLError where error.code == .timedOut {
            throw UploadError.serverError(
                "Upload timed out after \(Int(uploadTimeoutSeconds))s. " +
                "If SfM/BA is enabled, disable it for quick tests or keep this request running longer."
            )
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw UploadError.serverError("No HTTP response")
        }

        guard httpResponse.statusCode == 200 else {
            let errorBody = String(data: data, encoding: .utf8) ?? "Unknown"
            throw UploadError.serverError("HTTP \(httpResponse.statusCode): \(errorBody)")
        }

        if let jsonStr = String(data: data, encoding: .utf8) {
            try? jsonStr.write(
                to: URL(fileURLWithPath: "/tmp/decroom_payload.json"),
                atomically: true,
                encoding: .utf8
            )
        }

        do {
            let payload = try JSONDecoder().decode(RoomPayload.self, from: data)
            return payload
        } catch {
            throw UploadError.decodingFailed(error.localizedDescription)
        }
    }

    static func upload(
        images: [CGImage],
        metadataJSON: String?
    ) async throws -> RoomPayload {
        let url = try await resolveUploadURL()
        print("[UploadService] Uploading to \(url.absoluteString)")

        let boundary = UUID().uuidString

        // Build multipart body OFF the main thread — JPEG encoding is expensive
        let body: Data = try await Task.detached(priority: .userInitiated) {
            var bodyData = Data()

            for (index, cgImage) in images.enumerated() {
                let uiImage = UIImage(cgImage: cgImage)
                guard let jpegData = uiImage.jpegData(compressionQuality: 0.75) else {
                    throw UploadError.imageConversionFailed
                }

                bodyData.append("--\(boundary)\r\n")
                bodyData.append("Content-Disposition: form-data; name=\"frames\"; filename=\"frame_\(index).jpg\"\r\n")
                bodyData.append("Content-Type: image/jpeg\r\n\r\n")
                bodyData.append(jpegData)
                bodyData.append("\r\n")
            }

            if let metadata = metadataJSON, let metaData = metadata.data(using: .utf8) {
                bodyData.append("--\(boundary)\r\n")
                bodyData.append("Content-Disposition: form-data; name=\"metadata\"; filename=\"metadata.json\"\r\n")
                bodyData.append("Content-Type: application/json\r\n\r\n")
                bodyData.append(metaData)
                bodyData.append("\r\n")
            }

            bodyData.append("--\(boundary)--\r\n")
            return bodyData
        }.value

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = uploadTimeoutSeconds
        request.httpBody = body

        let configuration = URLSessionConfiguration.default
        configuration.waitsForConnectivity = false
        configuration.timeoutIntervalForRequest = uploadTimeoutSeconds
        configuration.timeoutIntervalForResource = uploadTimeoutSeconds
        configuration.allowsExpensiveNetworkAccess = true
        configuration.allowsConstrainedNetworkAccess = true
        let session = URLSession(configuration: configuration)
        defer {
            session.finishTasksAndInvalidate()
        }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch let error as URLError where error.code == .timedOut {
            throw UploadError.serverError(
                "Upload timed out after \(Int(uploadTimeoutSeconds))s. " +
                "If SfM/BA is enabled, disable it for quick tests or keep this request running longer."
            )
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw UploadError.serverError("No HTTP response")
        }

        guard httpResponse.statusCode == 200 else {
            let errorBody = String(data: data, encoding: .utf8) ?? "Unknown"
            throw UploadError.serverError("HTTP \(httpResponse.statusCode): \(errorBody)")
        }

        // Write raw response for debugging
        if let jsonStr = String(data: data, encoding: .utf8) {
            try? jsonStr.write(
                to: URL(fileURLWithPath: "/tmp/decroom_payload.json"),
                atomically: true,
                encoding: .utf8
            )
        }

        do {
            let payload = try JSONDecoder().decode(RoomPayload.self, from: data)
            return payload
        } catch {
            throw UploadError.decodingFailed(error.localizedDescription)
        }
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
