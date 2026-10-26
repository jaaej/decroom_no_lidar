//
//  ScanError.swift
//  decroom_no_lidar
//
//  Named failure cases for the phone-first scan + upload flow. Every
//  case carries a user-facing message (the exact string shown in the
//  alert) and an optional technical detail (logged, not displayed).
//

import Foundation

enum ScanError: Error, Equatable {
    case unsupportedDevice
    case depthModelLoadFailed(detail: String)
    case arKitTrackingLimited
    case arKitSessionFailed(detail: String)
    case noWallsDetected
    case insufficientCoverage(binsFilled: Int, binsRequired: Int)
    case payloadBuildFailed(detail: String)
    case uploadNoServer
    case serverKeyMissing
    case serverKeyMismatch
    case uploadTimeout
    case backendSchemaError(scanID: String, detail: String)
    case backendServerError(status: Int, detail: String)
    case noFurnitureDetected   // informational, not an error — surfaced as info toast

    /// Exact string shown in the user-facing alert.
    var message: String {
        switch self {
        case .unsupportedDevice:
            return "This iPhone doesn't support Decroom's scanning. Minimum: iPhone XS."
        case .depthModelLoadFailed:
            return "Can't start scan: the depth engine isn't available. Please reinstall the app."
        case .arKitTrackingLimited:
            return "Lost tracking — slow down, more light."
        case .arKitSessionFailed:
            return "ARKit stopped working. Restart the app."
        case .noWallsDetected:
            return "No walls found. Point at the walls of your room and keep scanning."
        case .insufficientCoverage:
            return "Only part of the room was captured. Rotate 360° and try again."
        case .payloadBuildFailed:
            return "Couldn't save scan data. Check phone storage and try again."
        case .uploadNoServer:
            return "Can't reach the Decroom server. Check WiFi and tap Retry."
        case .serverKeyMissing:
            return "Add the server key in Settings → Server key. The backend prints it the first time it starts (or run cat ~/.decroom/api_token on the server)."
        case .serverKeyMismatch:
            return "Found a Decroom server, but it doesn't match your server key. Check Settings → Server key."
        case .uploadTimeout:
            return "Server took too long. Try again in a moment."
        case .backendSchemaError(let id, _):
            return "Scan data was rejected. Please report this. Scan ID: \(id)"
        case .backendServerError:
            return "Server error. Try again in a moment."
        case .noFurnitureDetected:
            return "Scan complete — no furniture found."
        }
    }

    /// Is this error recoverable by tapping Retry (same button, re-attempt)?
    var isRetryable: Bool {
        switch self {
        case .payloadBuildFailed,
             .uploadNoServer,
             .uploadTimeout,
             .backendServerError:
            return true
        case .noWallsDetected, .insufficientCoverage:
            return true  // "Resume scan" rather than "Retry upload"
        case .unsupportedDevice,
             .serverKeyMissing,
             .serverKeyMismatch,
             .depthModelLoadFailed,
             .arKitSessionFailed,
             .backendSchemaError,
             .arKitTrackingLimited,
             .noFurnitureDetected:
            return false
        }
    }

    /// Technical detail for scan log / debugging. Never shown to the user.
    var detail: String {
        switch self {
        case .unsupportedDevice:
            return "device tier .fallback"
        case .depthModelLoadFailed(let d), .arKitSessionFailed(let d),
             .payloadBuildFailed(let d):
            return d
        case .arKitTrackingLimited:
            return "ARKit trackingState reported limited"
        case .noWallsDetected:
            return "RoomGeometry.walls.isEmpty at finalize"
        case .insufficientCoverage(let filled, let req):
            return "yaw bins filled=\(filled), required=\(req)"
        case .uploadNoServer:
            return "No reachable backend among candidate URLs"
        case .serverKeyMissing:
            return "No server key in Keychain"
        case .serverKeyMismatch:
            return "Server failed the /health key proof or returned 401"
        case .uploadTimeout:
            return "URLSession timed out"
        case .backendSchemaError(_, let d):
            return "HTTP 4xx: \(d)"
        case .backendServerError(let status, let d):
            return "HTTP \(status): \(d)"
        case .noFurnitureDetected:
            return "SpatialLM returned 0 objects"
        }
    }
}
