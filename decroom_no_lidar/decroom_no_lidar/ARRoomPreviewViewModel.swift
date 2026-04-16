//
//  ARRoomPreviewViewModel.swift
//  decroom_no_lidar
//
//  Pure logic for ARRoomPreviewView state. Derives reticle direction,
//  coverage percent, and "scan complete" status from the underlying
//  ScanSession state. Extracted from the view so tests cover behavior.
//

import Foundation
import simd

struct ARRoomPreviewViewModel {
    /// Current camera yaw in world frame (radians, 0 = +Z).
    let cameraYaw: Float
    /// Number of wall plane anchors detected (cap at 4 for "complete" check).
    let wallsDetected: Int
    /// Which yaw bins have received enough captured frames.
    let coverageBinsFilled: Set<Int>
    /// Total yaw bins (typically 24 = 15° each).
    let totalCoverageBins: Int

    /// Coverage as 0..1 fraction.
    var coveragePercent: Double {
        Double(coverageBinsFilled.count) / Double(totalCoverageBins)
    }

    /// True when scan is "done enough": 4 walls + full yaw coverage.
    var scanComplete: Bool {
        wallsDetected >= 4 && coverageBinsFilled.count >= totalCoverageBins
    }

    /// Unit direction vector (world-space) the reticle should point toward.
    /// Heuristic: find the yaw bin with lowest coverage "weight" around the user.
    var reticleDirection: SIMD3<Float> {
        guard totalCoverageBins > 0 else {
            return SIMD3<Float>(0, 0, 1)
        }
        // Find first unfilled bin. If all filled, point forward (cameraYaw).
        let firstUnfilled = (0..<totalCoverageBins).first(where: {
            !coverageBinsFilled.contains($0)
        })
        let targetBin = firstUnfilled ?? Int(cameraYaw / (2 * .pi) * Float(totalCoverageBins))
        let binAngle = (Float(targetBin) + 0.5) / Float(totalCoverageBins) * 2 * .pi
        // Convert binAngle (0 = camera forward = -Z in world) to a direction vector.
        return SIMD3<Float>(-sin(binAngle), 0, -cos(binAngle))
    }
}
