//
//  KeyframeSelector.swift
//  decroom_no_lidar
//
//  Selects a diverse subset of captured frames for backend upload.
//  Combines farthest-point sampling (spatial diversity) with
//  per-object best-view selection (semantic richness) and filters
//  low-quality frames (blur, sparse features).
//

import Foundation
import simd

struct KeyframeCandidate {
    let frameIndex: Int
    let pose: simd_float4x4
    let blurScore: Float
    let featurePointCount: Int
    /// Map of object instance id → detection confidence in this frame.
    let detectedObjectIds: [String: Float]
}

enum KeyframeSelector {

    /// Select up to `maxCount` keyframes prioritizing:
    /// 1. Quality — drop blur < 120 or feature points < 5
    /// 2. Best view per detected object
    /// 3. Spatial diversity via farthest-point sampling
    static func select(_ candidates: [KeyframeCandidate], maxCount: Int) -> [KeyframeCandidate] {
        let qualityFiltered = candidates.filter { $0.blurScore >= 120 && $0.featurePointCount >= 5 }
        guard !qualityFiltered.isEmpty else { return [] }

        // Step 1: best frame per detected object
        var objectBestFrame: [String: KeyframeCandidate] = [:]
        for candidate in qualityFiltered {
            for (objectId, confidence) in candidate.detectedObjectIds {
                if let existing = objectBestFrame[objectId] {
                    if confidence > (existing.detectedObjectIds[objectId] ?? 0) {
                        objectBestFrame[objectId] = candidate
                    }
                } else {
                    objectBestFrame[objectId] = candidate
                }
            }
        }
        var selected = Array(Set(objectBestFrame.values)).sorted { $0.frameIndex < $1.frameIndex }

        // Step 2: farthest-point sampling to fill remaining slots
        var remaining = qualityFiltered.filter { cand in
            !selected.contains(where: { $0.frameIndex == cand.frameIndex })
        }

        while selected.count < maxCount && !remaining.isEmpty {
            // Find the candidate farthest from all currently selected
            var bestIdx = 0
            var bestDist: Float = 0
            for (i, cand) in remaining.enumerated() {
                let candPos = positionOf(cand.pose)
                let minDistToSelected = selected.map {
                    simd_distance(candPos, positionOf($0.pose))
                }.min() ?? Float.greatestFiniteMagnitude
                if minDistToSelected > bestDist {
                    bestDist = minDistToSelected
                    bestIdx = i
                }
            }
            selected.append(remaining.remove(at: bestIdx))
        }

        return selected.sorted { $0.frameIndex < $1.frameIndex }
    }

    private static func positionOf(_ pose: simd_float4x4) -> SIMD3<Float> {
        SIMD3(pose.columns.3.x, pose.columns.3.y, pose.columns.3.z)
    }
}

// MARK: - Hashable conformance for Set<KeyframeCandidate>

extension KeyframeCandidate: Hashable {
    func hash(into hasher: inout Hasher) { hasher.combine(frameIndex) }
    static func == (lhs: KeyframeCandidate, rhs: KeyframeCandidate) -> Bool {
        lhs.frameIndex == rhs.frameIndex
    }
}
