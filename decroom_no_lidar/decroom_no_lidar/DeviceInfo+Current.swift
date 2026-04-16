//
//  DeviceInfo+Current.swift
//  decroom_no_lidar
//
//  Factory for DeviceInfo.current() — populates from UIDevice + DeviceCapabilities.
//

import Foundation
import UIKit

extension DeviceInfo {
    /// Build a DeviceInfo describing the current device.
    static func current() -> DeviceInfo {
        let modelId = currentModelIdentifier()
        let chip = DeviceCapabilities.currentChipGeneration().displayName
        return DeviceInfo(
            model: modelId,
            ios: UIDevice.current.systemVersion,
            chip: chip,
            tier: DeviceCapabilities.currentTier
        )
    }

    /// Machine identifier (e.g., "iPhone16,1"). On Simulator returns "Simulator".
    private static func currentModelIdentifier() -> String {
        #if targetEnvironment(simulator)
        return ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] ?? "Simulator"
        #else
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafeBytes(of: &systemInfo.machine) { raw -> String in
            let cString = raw.bindMemory(to: CChar.self)
            return String(cString: cString.baseAddress!)
        }
        #endif
    }
}

extension ChipGeneration {
    /// Human-readable chip name for display in logs.
    var displayName: String {
        switch self {
        case .a11: return "A11 Bionic"
        case .a12: return "A12 Bionic"
        case .a13: return "A13 Bionic"
        case .a14: return "A14 Bionic"
        case .a15: return "A15 Bionic"
        case .a16: return "A16 Bionic"
        case .a17pro: return "A17 Pro"
        case .a18: return "A18"
        case .a18pro: return "A18 Pro"
        case .unknown: return "Unknown"
        }
    }
}
