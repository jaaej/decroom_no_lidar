//
//  CoverageEngine.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import Foundation
import SwiftUI
import simd

/// Manages coverage-driven capture guidance.
/// Tracks three dimensions of coverage:
///   1. Angular: 24 yaw bins x 3 pitch tiers
///   2. Spatial: 0.5m XZ grid cells + cluster counting + radius from start
///   3. Feature-point voxels: 0.08m voxel coverage (excluding floor/ceiling)
enum CaptureMode {
    case photo
    case video
}

@MainActor
final class CoverageEngine: ObservableObject {

    // MARK: - Constants

    // Angular coverage
    let yawBins: Int = 24   // 15 degrees each
    let pitchBins: Int = 3

    // Capture mode (photo vs video)
    var captureMode: CaptureMode = .photo

    // Mode-dependent thresholds
    var minimumShots: Int { captureMode == .video ? 40 : 10 }
    var maximumShots: Int { captureMode == .video ? 200 : 30 }
    var minDuplicateDistance: Float { captureMode == .video ? 0.02 : 0.06 }
    var minBaselineDistance: Float { captureMode == .video ? 0.04 : 0.12 }

    // Constraints
    let maxAllowedYawGap: Float = Float.pi / 2  // 90 degrees

    // Spatial coverage
    let spatialCellSize: Float = 0.5             // 0.5m grid
    let minSpatialClusters: Int = 2              // require at least two scan regions
    let minSpatialRadius: Float = 0.8            // require meaningful travel
    let requiredSpatialClustersForUpload: Int = 3
    let requiredTravelDistanceForUpload: Float = 1.0
    let clusterMergeDistance: Float = 1.0         // merge captures within 1m into one cluster

    // Feature voxel coverage
    let featureVoxelSize: Float = 0.08           // 8cm voxel resolution
    let minFeatureVoxels: Int = 100              // min for upload readiness (relaxed for bare rooms)
    let goodFeatureVoxels: Int = 800             // "good quality" threshold
    let minFeaturePointsPerCapture: Int = 8      // only block when FP is extremely sparse
    let minFeatureSpreadCellsPerCapture: Int = 5 // 2D image grid cells hit by FPs
    let minFeatureSpreadShots: Int = 2            // number of captures with good FP spread
    let minFeatureSpreadCoverage: Float = 0.20    // fraction of image grid covered by FPs

    // Motion / quality
    let maxCameraVelocity: Float = 1.5           // m/s, reject only very shaky captures
    let minPitchRange: Float = Float.pi / 6      // 30 degrees min pitch variation
    let sequenceTargetShots: Int = 12            // ordered 360 sweep target
    let sequenceStepYaw: Float = (2 * .pi) / 12  // 30 degrees
    let sequenceYawTolerance: Float = Float.pi / 8 // 22.5 degrees
    let minOverlapPolicyScore: Float = 0.22
    let recommendedMaxCaptureStepM: Float = 1.35
    let hardMaxCaptureStepM: Float = 1.85
    let minTextureFeaturePointsForPolicy: Int = 35

    // Pitch tier boundaries (radians, negative = looking down)
    let pitchTierEdges: [Float] = [-Float.pi / 8, Float.pi / 12]

    // MARK: - Published State

    // Angular
    @Published private(set) var gridFilled: Set<Int> = []
    @Published private(set) var capturedCount: Int = 0
    @Published private(set) var coverageScore: Float = 0.0
    @Published private(set) var largestYawGap: Float = 2 * .pi
    @Published private(set) var largestGapCenterYaw: Float? = nil  // center of biggest uncovered arc
    @Published private(set) var pitchTiersUsed: Int = 0
    @Published var filledCellCount: Int = 0

    // Spatial
    @Published private(set) var spatialCellsFilled: Set<Int64> = []
    @Published private(set) var spatialClusterCount: Int = 0
    @Published private(set) var spatialBounds: (minX: Float, maxX: Float, minZ: Float, maxZ: Float) = (0, 0, 0, 0)
    @Published private(set) var liveGridPosition: (x: Int, z: Int) = (0, 0)
    @Published private(set) var maxDistanceFromStart: Float = 0

    // Feature voxels
    @Published private(set) var featureVoxelCount: Int = 0
    @Published private(set) var featureVoxelProgress: Float = 0  // 0..1 vs goodFeatureVoxels

    // Motion
    @Published private(set) var currentCameraVelocity: Float = 0

    // Pitch diversity
    @Published private(set) var pitchRange: Float = 0

    // Checklist metrics
    @Published private(set) var uniqueYawBinCount: Int = 0  // unique yaw directions captured
    @Published private(set) var goodFeatureShots: Int = 0   // shots with ≥80 feature points
    @Published private(set) var goodFeatureSpreadShots: Int = 0
    @Published private(set) var averageFeatureSpreadCoverage: Float = 0

    // Per-yaw-bin coverage (for visual arc display)
    var filledYawBins: Set<Int> {
        var yawSet: Set<Int> = []
        for flat in gridFilled { yawSet.insert(flat % yawBins) }
        return yawSet
    }

    // Upload readiness
    @Published private(set) var isReadyToUpload: Bool = false
    @Published private(set) var nextSequenceYaw: Float? = nil
    @Published private(set) var lastCaptureUsableScore: Float = 0
    @Published private(set) var lastCaptureOverlapScore: Float = 0

    // MARK: - Private State

    private var capturedYaws: [Float] = []
    private var capturedPitches: [Float] = []
    private var pitchTierSet: Set<Int> = []
    private var capturedAnglesAndPositions: [(yaw: Float, pitch: Float, position: SIMD3<Float>)] = []
    private var capturePositions: [SIMD3<Float>] = []

    private var featureVoxelSet: Set<Int64> = []
    private var featureSpreadCoverageSamples: [Float] = []
    private var estimatedFloorY: Float? = nil

    private var previousFramePosition: SIMD3<Float>? = nil
    private var previousFrameTime: TimeInterval = 0

    private var minCapturedPitch: Float = .greatestFiniteMagnitude
    private var maxCapturedPitch: Float = -.greatestFiniteMagnitude
    private var firstCaptureYaw: Float? = nil
    private var sequenceDirection: Float? = nil  // +1 clockwise in yaw-space, -1 opposite

    // MARK: - Capture Rejection

    enum CaptureRejectionReason {
        case tooCloseAngular
        case tooCloseSpatial
        case insufficientFeatures
        case movingTooFast
        case outOfSequence
    }

    struct CapturePolicyFeedback {
        let usableScore: Float
        let overlapScore: Float
        let textureScore: Float
        let continuityScore: Float
        let stepM: Float?
        let yawDeltaDeg: Float?
        let requiresStabilizingShot: Bool
        let isSequenceBreak: Bool
        let warnings: [String]
    }

    // MARK: - Public API

    func reset() {
        gridFilled = []
        capturedCount = 0
        coverageScore = 0
        largestYawGap = 2 * .pi
        largestGapCenterYaw = nil
        pitchTiersUsed = 0
        filledCellCount = 0
        capturedYaws = []
        capturedPitches = []
        pitchTierSet = []
        capturedAnglesAndPositions = []
        capturePositions = []
        spatialCellsFilled = []
        spatialClusterCount = 0
        spatialBounds = (0, 0, 0, 0)
        liveGridPosition = (0, 0)
        maxDistanceFromStart = 0
        featureVoxelCount = 0
        featureVoxelProgress = 0
        featureVoxelSet = []
        estimatedFloorY = nil
        currentCameraVelocity = 0
        previousFramePosition = nil
        previousFrameTime = 0
        minCapturedPitch = .greatestFiniteMagnitude
        maxCapturedPitch = -.greatestFiniteMagnitude
        pitchRange = 0
        uniqueYawBinCount = 0
        goodFeatureShots = 0
        goodFeatureSpreadShots = 0
        averageFeatureSpreadCoverage = 0
        featureSpreadCoverageSamples = []
        isReadyToUpload = false
        nextSequenceYaw = nil
        firstCaptureYaw = nil
        sequenceDirection = nil
        lastCaptureUsableScore = 0
        lastCaptureOverlapScore = 0
    }

    func canCapture() -> Bool {
        capturedCount < maximumShots
    }

    /// Check if a capture at the given position/angle is allowed.
    /// Returns nil if OK, or a rejection reason.
    func canCaptureHere(
        yaw: Float,
        pitch: Float,
        position: SIMD3<Float>,
        featurePointCount: Int
    ) -> CaptureRejectionReason? {
        // Motion stability check
        if currentCameraVelocity > maxCameraVelocity && capturedCount >= 1 {
            return .movingTooFast
        }

        // Ordered 360 sweep gate:
        // enforce capture progression around the room for the first sequenceTargetShots.
        if capturedCount >= 1 && capturedCount < sequenceTargetShots {
            if let firstYaw = firstCaptureYaw {
                if let dir = sequenceDirection {
                    let expected = normalizeTo0TwoPi(firstYaw + Float(capturedCount) * sequenceStepYaw * dir)
                    let err = abs(normalizeAngle(yaw - expected))
                    _ = err > sequenceYawTolerance  // advisory only (no hard block)
                } else {
                    // Direction not locked yet (typically second shot): allow either side.
                    let expectedCW = normalizeTo0TwoPi(firstYaw + sequenceStepYaw)
                    let expectedCCW = normalizeTo0TwoPi(firstYaw - sequenceStepYaw)
                    let errCW = abs(normalizeAngle(yaw - expectedCW))
                    let errCCW = abs(normalizeAngle(yaw - expectedCCW))
                    _ = min(errCW, errCCW) > sequenceYawTolerance  // advisory only
                }
            }
        }

        // Feature-point density is now advisory only (shown in guidance),
        // not a hard blocker for capture.

        // Spatial duplicate check:
        // Allow center-rotation captures. Only reject when position is very close
        // AND heading is also nearly identical (true duplicate tap).
        let posXZ = SIMD2<Float>(position.x, position.z)
        if capturedCount >= 3 {
            for entry in capturedAnglesAndPositions {
                let prevXZ = SIMD2<Float>(entry.position.x, entry.position.z)
                let dist = simd_distance(posXZ, prevXZ)
                let yawDiff = abs(normalizeAngle(yaw - entry.yaw))
                if dist < minDuplicateDistance && yawDiff < Float.pi / 36 {
                    return .tooCloseSpatial
                }
            }
        }

        // Angular duplicate check: same yaw bin AND close position
        let yi = yawBinIndex(yaw)
        if capturedCount >= 4 {
            for entry in capturedAnglesAndPositions {
                let entryYI = yawBinIndex(entry.yaw)
                if entryYI == yi {
                    let entryXZ = SIMD2<Float>(entry.position.x, entry.position.z)
                    let dist = simd_distance(posXZ, entryXZ)
                    let yawDiff = abs(normalizeAngle(yaw - entry.yaw))
                    // Same bin, close position, very similar angle → duplicate
                    if dist < minBaselineDistance && yawDiff < Float.pi / 30 {
                        return .tooCloseAngular
                    }
                }
            }
        }

        return nil
    }

    /// Live pre-capture policy feedback for overlap/continuity quality.
    func previewCapturePolicy(
        yaw: Float,
        pitch: Float,
        position: SIMD3<Float>,
        featurePointCount: Int,
        trackingQuality: String
    ) -> CapturePolicyFeedback {
        _ = pitch
        let textureScore = clamp01(Float(featurePointCount) / 120.0)
        let trackingScore = trackingQualityWeight(trackingQuality)

        var stepM: Float? = nil
        var yawDeltaDeg: Float? = nil
        var stepScore: Float = capturedCount == 0 ? 1.0 : 0.5
        var yawScore: Float = capturedCount == 0 ? 1.0 : 0.5

        if let last = capturedAnglesAndPositions.last {
            let prevXZ = SIMD2<Float>(last.position.x, last.position.z)
            let posXZ = SIMD2<Float>(position.x, position.z)
            let step = simd_distance(prevXZ, posXZ)
            stepM = step

            let yawDelta = abs(normalizeAngle(yaw - last.yaw)) * (180.0 / Float.pi)
            yawDeltaDeg = yawDelta

            if step < 0.05 {
                stepScore = 0.1
            } else if step < 0.18 {
                stepScore = 0.45
            } else if step <= recommendedMaxCaptureStepM {
                stepScore = 1.0
            } else if step <= hardMaxCaptureStepM {
                let t = (step - recommendedMaxCaptureStepM)
                    / max(0.05, hardMaxCaptureStepM - recommendedMaxCaptureStepM)
                stepScore = max(0.2, 1.0 - t)
            } else {
                stepScore = 0.05
            }

            if yawDelta < 6 {
                yawScore = 0.15
            } else if yawDelta < 12 {
                yawScore = 0.45
            } else if yawDelta <= 45 {
                yawScore = 1.0
            } else if yawDelta <= 70 {
                yawScore = 0.55
            } else {
                yawScore = 0.25
            }
        }

        let overlapScore = clamp01(0.15 * textureScore + 0.45 * yawScore + 0.40 * stepScore)
        let continuityScore = clamp01(0.55 * stepScore + 0.45 * yawScore)
        let usableScore = clamp01(0.45 * textureScore + 0.25 * trackingScore + 0.30 * continuityScore)

        var warnings: [String] = []
        let isSequenceBreak = capturedCount >= 2 && (
            ((stepM ?? 0.0) > 0.40 && (yawDeltaDeg ?? 0.0) > 80.0) ||
            (yawDeltaDeg ?? 0.0) > 100.0 ||
            (stepM ?? 0.0) > 1.5
        )
        let requiresStabilizingShot = (stepM ?? 0) > recommendedMaxCaptureStepM && capturedCount >= 2
        if (yawDeltaDeg ?? 0.0) > 55.0 && !isSequenceBreak {
            warnings.append("Large rotation. Capture soon for good overlap.")
        }
        if featurePointCount < minTextureFeaturePointsForPolicy {
            warnings.append("Need textured overlap in view (furniture, edges, corners).")
        }
        if capturedCount >= 2
            && (stepM ?? 0.0) > 0.30
            && (yawDeltaDeg ?? 0.0) > 80.0
            && !isSequenceBreak {
            warnings.append("Large turn + movement. Take one intermediate overlap shot.")
        }
        if requiresStabilizingShot {
            warnings.append("Large movement since last shot. Take an intermediate overlap shot.")
        }
        if capturedCount >= 1 && overlapScore < minOverlapPolicyScore {
            warnings.append("Low overlap with the previous shot. Rotate less and capture sooner.")
        }
        if trackingScore < 0.6 {
            warnings.append("Tracking is limited. Hold steady before capturing.")
        }

        return CapturePolicyFeedback(
            usableScore: usableScore,
            overlapScore: overlapScore,
            textureScore: textureScore,
            continuityScore: continuityScore,
            stepM: stepM,
            yawDeltaDeg: yawDeltaDeg,
            requiresStabilizingShot: requiresStabilizingShot,
            isSequenceBreak: isSequenceBreak,
            warnings: warnings
        )
    }

    /// Record a capture at given yaw/pitch/position with feature points.
    /// Returns true if a new angular grid cell was filled.
    @discardableResult
    func addCapture(
        yaw: Float,
        pitch: Float,
        position: SIMD3<Float>,
        featurePoints: [SIMD3<Float>],
        featureImageCoverage: Float = 0,
        featureGridCellCount: Int = 0,
        trackingQuality: String = "normal"
    ) -> Bool {
        let policy = previewCapturePolicy(
            yaw: yaw,
            pitch: pitch,
            position: position,
            featurePointCount: featurePoints.count,
            trackingQuality: trackingQuality
        )
        lastCaptureUsableScore = policy.usableScore
        lastCaptureOverlapScore = policy.overlapScore

        // Angular grid update
        capturedYaws.append(yaw)
        capturedPitches.append(pitch)
        capturedCount += 1

        if firstCaptureYaw == nil {
            firstCaptureYaw = yaw
        }
        if capturedCount == 2, let first = firstCaptureYaw {
            let delta = normalizeAngle(yaw - first)
            if abs(delta) > 0.05 {
                sequenceDirection = delta >= 0 ? 1.0 : -1.0
            } else {
                sequenceDirection = 1.0
            }
        }

        let yi = yawBinIndex(yaw)
        let pi = pitchBinIndex(pitch)
        let flat = pi * yawBins + yi
        let isNew = !gridFilled.contains(flat)
        gridFilled.insert(flat)
        filledCellCount = gridFilled.count

        // Pitch tiers
        pitchTierSet.insert(pi)
        pitchTiersUsed = pitchTierSet.count

        // Pitch diversity
        if pitch < minCapturedPitch { minCapturedPitch = pitch }
        if pitch > maxCapturedPitch { maxCapturedPitch = pitch }
        if minCapturedPitch < .greatestFiniteMagnitude {
            pitchRange = maxCapturedPitch - minCapturedPitch
        }

        // Coverage score (proportion of max shots)
        coverageScore = Float(capturedCount) / Float(maximumShots)

        // Store for baseline enforcement
        capturedAnglesAndPositions.append((yaw: yaw, pitch: pitch, position: position))
        capturePositions.append(position)

        // Spatial grid
        updateSpatialGrid(position: position)

        // Spatial radius from start
        if capturePositions.count >= 2 {
            let startXZ = SIMD2<Float>(capturePositions[0].x, capturePositions[0].z)
            let currentXZ = SIMD2<Float>(position.x, position.z)
            let dist = simd_distance(startXZ, currentXZ)
            if dist > maxDistanceFromStart {
                maxDistanceFromStart = dist
            }
        }

        // Feature voxels (with floor/ceiling filtering)
        updateFeatureVoxels(points: featurePoints, cameraY: position.y)

        // Spatial clusters
        updateSpatialClusters()

        // Yaw gap
        updateYawGap()

        // Update unique yaw bin count
        var yawSet: Set<Int> = []
        for flat in gridFilled { yawSet.insert(flat % yawBins) }
        uniqueYawBinCount = yawSet.count

        // Track shots with good feature density (≥80 FP = meaningful detail)
        if featurePoints.count >= 80 {
            goodFeatureShots += 1
        }
        if featureGridCellCount >= minFeatureSpreadCellsPerCapture
            || featureImageCoverage >= minFeatureSpreadCoverage {
            goodFeatureSpreadShots += 1
        }
        featureSpreadCoverageSamples.append(max(0.0, min(1.0, featureImageCoverage)))
        if !featureSpreadCoverageSamples.isEmpty {
            averageFeatureSpreadCoverage = featureSpreadCoverageSamples.reduce(0, +)
                / Float(featureSpreadCoverageSamples.count)
        }

        // Upload readiness
        updateUploadReadiness()
        updateNextSequenceYaw()

        return isNew
    }

    /// Called per-frame to update live state (minimap position, camera velocity).
    func updateLiveState(position: SIMD3<Float>, timestamp: TimeInterval) {
        // Minimap live dot
        liveGridPosition = (
            x: Int(floor(position.x / spatialCellSize)),
            z: Int(floor(position.z / spatialCellSize))
        )

        // Camera velocity
        if let prevPos = previousFramePosition, previousFrameTime > 0 {
            let dt = Float(timestamp - previousFrameTime)
            if dt > 0.001 {
                let dist = simd_distance(position, prevPos)
                currentCameraVelocity = dist / dt
            }
        }
        previousFramePosition = position
        previousFrameTime = timestamp
    }

    func flatIndex(yaw: Float, pitch: Float) -> Int {
        let yi = yawBinIndex(yaw)
        let pi = pitchBinIndex(pitch)
        return pi * yawBins + yi
    }

    // MARK: - Private Helpers

    private func yawBinIndex(_ yaw: Float) -> Int {
        let normalized = yaw / (2 * .pi)
        let bin = Int(normalized * Float(yawBins)) % yawBins
        return bin < 0 ? bin + yawBins : bin
    }

    private func pitchBinIndex(_ pitch: Float) -> Int {
        if pitch < pitchTierEdges[0] { return 0 }      // looking down
        if pitch < pitchTierEdges[1] { return 1 }      // level
        return 2                                         // looking up
    }

    private func updateYawGap() {
        guard capturedYaws.count >= 2 else {
            largestYawGap = 2 * .pi
            largestGapCenterYaw = nil
            return
        }

        var sorted = capturedYaws.map { $0.truncatingRemainder(dividingBy: 2 * .pi) }
        sorted = sorted.map { $0 < 0 ? $0 + 2 * .pi : $0 }
        sorted.sort()

        var maxGap: Float = 0
        var gapStart: Float = 0
        for i in 1..<sorted.count {
            let gap = sorted[i] - sorted[i - 1]
            if gap > maxGap {
                maxGap = gap
                gapStart = sorted[i - 1]
            }
        }
        // Wrap-around gap
        let wrapGap = (2 * .pi) - sorted.last! + sorted.first!
        if wrapGap > maxGap {
            maxGap = wrapGap
            gapStart = sorted.last!
        }

        largestYawGap = maxGap

        // Center of the gap (direction user should face)
        var center = gapStart + maxGap / 2
        if center >= 2 * .pi { center -= 2 * .pi }
        largestGapCenterYaw = center
    }

    // MARK: - Spatial Grid

    private func updateSpatialGrid(position: SIMD3<Float>) {
        let gx = Int(floor(position.x / spatialCellSize))
        let gz = Int(floor(position.z / spatialCellSize))
        let key = spatialKey(gx: gx, gz: gz)
        spatialCellsFilled.insert(key)

        // Update bounds
        if capturePositions.count == 1 {
            spatialBounds = (minX: position.x, maxX: position.x,
                             minZ: position.z, maxZ: position.z)
        } else {
            spatialBounds = (
                minX: min(spatialBounds.minX, position.x),
                maxX: max(spatialBounds.maxX, position.x),
                minZ: min(spatialBounds.minZ, position.z),
                maxZ: max(spatialBounds.maxZ, position.z)
            )
        }
    }

    private func spatialKey(gx: Int, gz: Int) -> Int64 {
        return (Int64(gx) & 0xFFFFFFFF) << 32 | (Int64(gz) & 0xFFFFFFFF)
    }

    // MARK: - Spatial Clustering

    private func updateSpatialClusters() {
        guard !capturePositions.isEmpty else {
            spatialClusterCount = 0
            return
        }

        // Simple greedy clustering: sort by X, merge within clusterMergeDistance
        let sortedPositions = capturePositions.sorted { $0.x < $1.x }

        var clusters: [SIMD2<Float>] = []  // cluster centroids in XZ
        for pos in sortedPositions {
            let posXZ = SIMD2<Float>(pos.x, pos.z)
            var merged = false
            for i in 0..<clusters.count {
                if simd_distance(posXZ, clusters[i]) < clusterMergeDistance {
                    // Update centroid with running average
                    clusters[i] = (clusters[i] + posXZ) / 2
                    merged = true
                    break
                }
            }
            if !merged {
                clusters.append(posXZ)
            }
        }

        spatialClusterCount = clusters.count
    }

    // MARK: - Feature Voxels

    private func updateFeatureVoxels(points: [SIMD3<Float>], cameraY: Float) {
        // Estimate floor height from lowest camera position
        if estimatedFloorY == nil {
            estimatedFloorY = cameraY - 1.2  // assume standing height ~1.2m
        } else {
            estimatedFloorY = min(estimatedFloorY!, cameraY - 1.2)
        }
        let floorY = estimatedFloorY!

        for p in points {
            // Skip floor points (within 10cm of estimated floor)
            if abs(p.y - floorY) < 0.1 { continue }
            // Skip ceiling points (above camera + 0.5m)
            if p.y > cameraY + 0.5 { continue }

            let key = voxelKey(x: p.x, y: p.y, z: p.z)
            featureVoxelSet.insert(key)
        }

        featureVoxelCount = featureVoxelSet.count
        featureVoxelProgress = min(1.0, Float(featureVoxelCount) / Float(goodFeatureVoxels))
    }

    private func voxelKey(x: Float, y: Float, z: Float) -> Int64 {
        let vx = Int64(floor(x / featureVoxelSize))
        let vy = Int64(floor(y / featureVoxelSize))
        let vz = Int64(floor(z / featureVoxelSize))
        return (vx & 0xFFFFF) << 40 | (vy & 0xFFFFF) << 20 | (vz & 0xFFFFF)
    }

    // MARK: - Upload Readiness

    private func updateUploadReadiness() {
        let enoughPhotos = capturedCount >= minimumShots
        let walkedAround = spatialClusterCount >= requiredSpatialClustersForUpload
                        || maxDistanceFromStart >= requiredTravelDistanceForUpload
        let multipleAngles = uniqueYawBinCount >= 6
                           && largestYawGap <= Float.pi  // 180° max gap ensures spread
        let capturedDetail = goodFeatureShots >= 3
                           || featureVoxelCount >= 300
        let hasFeatureSpread = goodFeatureSpreadShots >= minFeatureSpreadShots
                            || averageFeatureSpreadCoverage >= 0.22
                            || capturedCount >= 16

        if captureMode == .video {
            // Video mode: dense angular coverage is sufficient.
            // Walking around helps but shouldn't block upload.
            isReadyToUpload = enoughPhotos && multipleAngles
        } else {
            isReadyToUpload = enoughPhotos
                && multipleAngles
                && capturedDetail
                && hasFeatureSpread
                && walkedAround
        }
    }

    private func updateNextSequenceYaw() {
        guard capturedCount < sequenceTargetShots else {
            nextSequenceYaw = nil
            return
        }
        guard let first = firstCaptureYaw else {
            nextSequenceYaw = nil
            return
        }
        let dir = sequenceDirection ?? 1.0
        nextSequenceYaw = normalizeTo0TwoPi(first + Float(capturedCount) * sequenceStepYaw * dir)
    }

    private func trackingQualityWeight(_ quality: String) -> Float {
        switch quality {
        case "normal":
            return 1.0
        case "limited_initializing":
            return 0.55
        case "limited_excessive_motion":
            return 0.45
        case "limited_insufficient_features":
            return 0.4
        case "limited_relocalizing":
            return 0.5
        default:
            return 0.5
        }
    }

    private func clamp01(_ v: Float) -> Float {
        return min(1.0, max(0.0, v))
    }
}

// MARK: - Angle Utilities

func normalizeAngle(_ angle: Float) -> Float {
    var a = angle
    while a > .pi { a -= 2 * .pi }
    while a < -.pi { a += 2 * .pi }
    return a
}

func normalizeTo0TwoPi(_ angle: Float) -> Float {
    var a = angle.truncatingRemainder(dividingBy: 2 * .pi)
    if a < 0 { a += 2 * .pi }
    return a
}
