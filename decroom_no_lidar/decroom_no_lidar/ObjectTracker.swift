//
//  ObjectTracker.swift
//  decroom_no_lidar
//
//  Nearest-neighbor 3D instance tracking. Given detections with world
//  positions, assigns each to a unique track ID. Keeps the highest-
//  confidence observation per track for later best-frame keyframe
//  selection.
//
//  Pure logic, no ARKit dependency — unit-testable.
//

import Foundation
import simd

struct ObjectTrack {
    let id: UUID
    let category: String
    var latestPosition: SIMD3<Float>
    var bestConfidence: Float
    var bestFrameIndex: Int
    var observationCount: Int
}

struct ObjectTracker {
    private(set) var tracks: [ObjectTrack] = []
    private let mergeDistanceByCategory: [String: Float]
    private let defaultMergeDistance: Float

    init(mergeDistanceByCategory: [String: Float],
         defaultMergeDistance: Float = 0.6) {
        self.mergeDistanceByCategory = mergeDistanceByCategory
        self.defaultMergeDistance = defaultMergeDistance
    }

    /// Record an observation. Returns the assigned track ID (new or existing).
    @discardableResult
    mutating func observe(category: String,
                           worldPosition: SIMD3<Float>,
                           confidence: Float,
                           frameIndex: Int) -> UUID {
        let mergeDist = mergeDistanceByCategory[category] ?? defaultMergeDistance

        // Find nearest existing track of the same category within merge distance.
        if let idx = nearestTrackIndex(category: category,
                                         position: worldPosition,
                                         within: mergeDist) {
            tracks[idx].latestPosition = worldPosition
            tracks[idx].observationCount += 1
            if confidence > tracks[idx].bestConfidence {
                tracks[idx].bestConfidence = confidence
                tracks[idx].bestFrameIndex = frameIndex
            }
            return tracks[idx].id
        }

        // New track.
        let track = ObjectTrack(
            id: UUID(),
            category: category,
            latestPosition: worldPosition,
            bestConfidence: confidence,
            bestFrameIndex: frameIndex,
            observationCount: 1
        )
        tracks.append(track)
        return track.id
    }

    private func nearestTrackIndex(category: String,
                                     position: SIMD3<Float>,
                                     within maxDist: Float) -> Int? {
        var best: (idx: Int, dist: Float)?
        for (i, track) in tracks.enumerated() where track.category == category {
            let d = simd_distance(track.latestPosition, position)
            if d <= maxDist {
                if best == nil || d < best!.dist {
                    best = (i, d)
                }
            }
        }
        return best?.idx
    }
}
