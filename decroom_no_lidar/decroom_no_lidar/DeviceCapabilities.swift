//
//  DeviceCapabilities.swift
//  decroom_no_lidar
//
//  Detects device tier (.full / .reduced / .fallback) based on chip generation
//  and ARKit availability. Used by CaptureFlowView to choose between
//  phone-first AR capture and the legacy backend-heavy pipeline.
//

import Foundation
import ARKit

enum DeviceTier: String, Codable {
    case full       // A14+ (iPhone 12 and later): full on-device depth + AR capture
    case reduced    // A12-A13 (iPhone XR/XS/11): on-device depth at lower fps
    case fallback   // Older devices: use existing backend-heavy pipeline
}

enum ChipGeneration: String {
    case a11, a12, a13, a14, a15, a16, a17pro, a18, a18pro
    case unknown
}

enum DeviceCapabilities {

    /// Map chip generation to device tier. Pure function, testable.
    static func tier(forChipGeneration chip: ChipGeneration) -> DeviceTier {
        switch chip {
        case .a14, .a15, .a16, .a17pro, .a18, .a18pro:
            return .full
        case .a12, .a13:
            return .reduced
        case .a11, .unknown:
            return .fallback
        }
    }

    /// Detect current device chip generation from machine identifier.
    static func currentChipGeneration() -> ChipGeneration {
        var systemInfo = utsname()
        uname(&systemInfo)
        let identifier = withUnsafeBytes(of: &systemInfo.machine) { raw -> String in
            let cString = raw.bindMemory(to: CChar.self)
            return String(cString: cString.baseAddress!)
        }
        return chipGeneration(forMachineIdentifier: identifier)
    }

    /// Map machine identifier (e.g., "iPhone16,1") to chip generation.
    /// Exposed for testing.
    static func chipGeneration(forMachineIdentifier id: String) -> ChipGeneration {
        // iPhone identifiers: https://gist.github.com/adamawolf/3048717
        switch id {
        case "iPhone10,1", "iPhone10,2", "iPhone10,3", "iPhone10,4", "iPhone10,5", "iPhone10,6":
            return .a11  // iPhone 8, 8 Plus, X
        case "iPhone11,2", "iPhone11,4", "iPhone11,6", "iPhone11,8":
            return .a12  // iPhone XS, XS Max, XR
        case "iPhone12,1", "iPhone12,3", "iPhone12,5", "iPhone12,8":
            return .a13  // iPhone 11, 11 Pro, 11 Pro Max, SE 2nd gen
        case "iPhone13,1", "iPhone13,2", "iPhone13,3", "iPhone13,4":
            return .a14  // iPhone 12 mini, 12, 12 Pro, 12 Pro Max
        case "iPhone14,2", "iPhone14,3", "iPhone14,4", "iPhone14,5", "iPhone14,6", "iPhone14,7", "iPhone14,8":
            return .a15  // iPhone 13 series, SE 3rd gen, 14, 14 Plus
        case "iPhone15,2", "iPhone15,3", "iPhone15,4", "iPhone15,5":
            return .a16  // iPhone 14 Pro, 14 Pro Max, 15, 15 Plus
        case "iPhone16,1", "iPhone16,2":
            return .a17pro  // iPhone 15 Pro, 15 Pro Max
        // NOTE: iPhone 17,x identifiers are community-sourced and should be
        // verified before release. iPhone 16e and other 2024/2025 variants
        // may have additional identifiers that will fall through to .unknown.
        case "iPhone17,1", "iPhone17,2":
            return .a18pro  // iPhone 16 Pro, 16 Pro Max
        case "iPhone17,3", "iPhone17,4":
            return .a18  // iPhone 16, 16 Plus
        default:
            return .unknown
        }
    }

    /// Current device's tier.
    static var currentTier: DeviceTier {
        // ARKit world tracking must be supported for any AR path.
        guard ARWorldTrackingConfiguration.isSupported else {
            return .fallback
        }
        return tier(forChipGeneration: currentChipGeneration())
    }
}
