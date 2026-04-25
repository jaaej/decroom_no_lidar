//
//  ScanSession.swift
//  decroom_no_lidar
//
//  Coordinates Plan 1 + Plan 2 components during a scan.
//
//  ML inference (depth + YOLO) runs on a dedicated background queue to
//  avoid blocking the main thread / AR camera. Only results are published
//  back to MainActor for SwiftUI consumption.
//

import Foundation
import ARKit
import SwiftUI
import CoreImage
import simd

@MainActor
final class ScanSession: ObservableObject {
    // Plan 1 components
    let logger: ScanLogger
    private(set) var depthRunner: DepthRunner?
    let geometryBuilder = RoomGeometryBuilder()
    /// pointCloud is accessed from the ML background queue (add/voxelize
    /// during inference) AND the main actor (exportPLY at finalize). mlQueue
    /// is serial and finalize uses mlQueue.sync for a drain, so concurrent
    /// access is prevented structurally — hence nonisolated(unsafe).
    nonisolated(unsafe) let pointCloud = PointCloudAccumulator(
        voxelSize: 0.02, maxPoints: 500_000
    )

    // Plan 2 components
    private(set) var yoloRunner: YoloRunner?
    var tracker = ObjectTracker(mergeDistanceByCategory: [
        "bed": 1.5, "couch": 1.5, "chair": 0.65, "dining table": 1.1,
        "tv": 0.8, "laptop": 0.5, "potted plant": 0.6
    ])

    // Published for SwiftUI
    @Published var geometry: RoomGeometry = RoomGeometry(
        walls: [], floorHeight: 0, ceilingHeight: 0, manhattanYaw: 0
    )
    @Published var recentDetections: [YoloDetection] = []
    @Published var coverageProgress: Double = 0.0
    @Published var modelsReady: Bool = false
    @Published var visitedYawBins: Set<Int> = []
    @Published var currentCameraYaw: Float = 0

    /// Short scan identifier (hex, 8 chars) — useful for error messages.
    var scanID: String { logger.scanId }

    // Internal state
    private let mlQueue = DispatchQueue(label: "decroom.scan-session.ml", qos: .userInitiated)
    private let ciContext = CIContext()
    private let startTime = Date()
    private var frameIndex = 0
    private var lastDepthFrame = -1
    private let depthFrameInterval = 15   // ~2fps at 30fps (was 5fps — too heavy)
    private let totalYawBins = 24

    // Drop-on-busy guard: if a prior depth inference is still running, skip
    // this dispatch rather than queue. Without this, enqueues at a faster
    // rate than inference completes retain ARKit capturedImage buffers until
    // the pool is exhausted and ARKit pauses frame delivery (~10s freezes).
    private var depthInFlight = false

    /// Cached snapshot of the last depth inference. Written by the ML queue
    /// after each depth inference; read by the ML queue before each YOLO
    /// inference to unproject bbox centroids to world space.
    /// Nil before the first depth result lands — YOLO drops detections
    /// that arrive before depth has a chance.
    private struct DepthSnapshot {
        let depths: [Float]
        let width: Int
        let height: Int
        let intrinsics: Intrinsics
        let worldFromCamera: simd_float4x4
    }
    nonisolated(unsafe) private let snapshotQueue = DispatchQueue(label: "decroom.scan-session.depth-snapshot")
    nonisolated(unsafe) private var _latestDepthSnapshot: DepthSnapshot?

    nonisolated private func storeDepthSnapshot(_ snap: DepthSnapshot) {
        snapshotQueue.sync { _latestDepthSnapshot = snap }
    }
    nonisolated private func latestDepthSnapshot() -> DepthSnapshot? {
        snapshotQueue.sync { _latestDepthSnapshot }
    }

    init() {
        self.logger = ScanLogger()
        logger.recordDevice(DeviceInfo.current())
        logger.logEvent("scan_started", at: 0)

        Task.detached(priority: .userInitiated) { [weak self] in
            let depth = try? DepthRunner()
            let yolo = try? YoloRunner()
            await MainActor.run {
                guard let self else { return }
                self.depthRunner = depth
                self.yoloRunner = yolo
                self.modelsReady = (depth != nil) || (yolo != nil)
                if depth == nil {
                    self.logger.logEvent("depth_runner_load_failed", at: self.elapsed())
                }
                if yolo == nil {
                    self.logger.logEvent("yolo_runner_load_failed", at: self.elapsed())
                }
                self.logger.logEvent("models_ready", at: self.elapsed())
            }
        }
    }

    /// Called from ARSessionDelegate on each new frame. Must be FAST —
    /// only updates lightweight state. ML inference dispatched to mlQueue.
    func update(frame: ARFrame) {
        frameIndex += 1

        // Track yaw — used for panorama scrolling + coverage tracking
        let yaw = cameraYaw(from: frame.camera.transform)
        currentCameraYaw = yaw
        var normalizedYaw = yaw
        if normalizedYaw < 0 { normalizedYaw += 2 * .pi }
        let bin = Int(normalizedYaw / (2 * .pi) * Float(totalYawBins)) % totalYawBins
        visitedYawBins.insert(bin)

        guard modelsReady else { return }

        // Dispatch depth inference to background if not already busy. Dropping
        // frames when in-flight prevents ARKit buffer-pool exhaustion.
        if !depthInFlight,
           let depth = depthRunner,
           frameIndex - lastDepthFrame >= depthFrameInterval {
            lastDepthFrame = frameIndex
            depthInFlight = true
            let capturedImage = frame.capturedImage
            let intrinsics = Intrinsics.make(from: frame.camera, portrait: true)
            let pose = frame.camera.transform
            let fi = frameIndex

            mlQueue.async { [weak self] in
                self?.runDepthBackground(
                    capturedImage: capturedImage,
                    intrinsics: intrinsics,
                    pose: pose,
                    depth: depth,
                    frameIndex: fi
                )
                Task { @MainActor [weak self] in self?.depthInFlight = false }
            }
        }

        // YOLO inference is disabled in Plan 3a — its outputs
        // (recentDetections, tracker observations) are not consumed by any
        // live code path. Keeping it disabled avoids the extra ML pressure.
        // Reenable in 3b when consensus logic lands. (runYoloBackground
        // remains in the file and can be re-dispatched there.)
    }

    /// Update plane anchors — called from didAdd/didUpdate anchor delegate.
    func updatePlaneAnchors(_ anchors: [ARPlaneAnchor]) {
        let descriptors: [PlaneAnchorDescriptor] = anchors
        self.geometry = geometryBuilder.build(from: descriptors)
        if !geometry.walls.isEmpty {
            logger.logEvent("geometry_updated", at: elapsed(),
                             data: ["wall_count": "\(geometry.walls.count)"])
        }
    }

    /// Produce the phone-first upload payload. Runs phone-side validation;
    /// throws a named `ScanError` if the scan is not uploadable.
    /// - Parameter keyframes: keyframe JPEGs + metadata assembled by the
    ///   caller (GuidedARFrameRecorder owns the JPEG files and pose stream).
    /// - Parameter totalYawBins: total bins the coverage engine split the
    ///   360° rotation into. Currently 24 (15° each).
    /// - Parameter requiredBins: minimum filled bins for a valid scan.
    ///   Currently 12 (50%).
    func finalize(
        keyframes: [ScanUploadPayload.KeyframeMetadata],
        keyframeJPEGs: [Data],
        totalYawBins: Int = 24,
        requiredBins: Int = 12
    ) throws -> ScanUploadPayload {
        logger.logEvent("scan_finalize_started", at: elapsed())

        // NOTE: depth-model-load failures are surfaced separately at scan
        // start (not here). finalize only validates that the scan produced
        // enough data to upload.

        // Phone-side validation.
        guard !geometry.walls.isEmpty else {
            logger.logEvent("scan_finalize_rejected_no_walls", at: elapsed())
            throw ScanError.noWallsDetected
        }
        guard visitedYawBins.count >= requiredBins else {
            logger.logEvent("scan_finalize_rejected_coverage", at: elapsed(),
                            data: ["filled": "\(visitedYawBins.count)",
                                   "required": "\(requiredBins)"])
            throw ScanError.insufficientCoverage(
                binsFilled: visitedYawBins.count,
                binsRequired: requiredBins
            )
        }
        guard keyframes.count == keyframeJPEGs.count else {
            throw ScanError.payloadBuildFailed(detail: "keyframe metadata/jpeg count mismatch")
        }

        // Serialize components. Errors here → payloadBuildFailed.
        let roomGeometryJSON: Data
        do {
            roomGeometryJSON = try ScanUploadPayload.encodeRoomGeometry(geometry)
        } catch {
            throw ScanError.payloadBuildFailed(detail: error.localizedDescription)
        }

        // Drain any in-flight depth inference + accumulation on mlQueue
        // before exporting the PLY, so the exported bytes reflect the
        // complete scan. mlQueue is serial; sync{} blocks until the last
        // enqueued task (if any) finishes.
        let plyData: Data = mlQueue.sync { pointCloud.exportPLY() }

        // Write log to disk (also keeps Plan 1's retention behavior).
        let logData: Data
        do {
            try logger.finalize()
            // Read back what we wrote for inclusion in the payload.
            let dir = ScanLogger.defaultDirectory()
            let file = "\(Int(logger.startedAt.timeIntervalSince1970))_\(logger.scanId).json"
            logData = try Data(contentsOf: dir.appendingPathComponent(file))
        } catch {
            throw ScanError.payloadBuildFailed(detail: "scan log write: \(error.localizedDescription)")
        }

        logger.logEvent("scan_finalize_succeeded", at: elapsed(),
                        data: ["walls": "\(geometry.walls.count)",
                               "points": "\(pointCloud.count)",
                               "keyframes": "\(keyframes.count)"])

        return ScanUploadPayload(
            scanID: logger.scanId,
            roomGeometryJSON: roomGeometryJSON,
            pointCloudPLY: plyData,
            keyframes: zip(keyframeJPEGs, keyframes).map { ($0.0, $0.1) },
            scanLogJSON: logData,
            deviceInfo: DeviceInfo.current()
        )
    }

    // MARK: - Private

    private func elapsed() -> Double { Date().timeIntervalSince(startTime) }

    private func cameraYaw(from transform: simd_float4x4) -> Float {
        let fx = -transform.columns.2.x
        let fz = -transform.columns.2.z
        return atan2(fx, fz)
    }

    /// Runs on mlQueue (background serial). Accumulates the point cloud on
    /// the same queue (off MainActor) and only hops to MainActor for the
    /// small coverageProgress update. Heavy per-inference voxel work used to
    /// run on MainActor and choke the UI; this keeps the main thread free.
    private nonisolated func runDepthBackground(
        capturedImage: CVPixelBuffer,
        intrinsics: Intrinsics,
        pose: simd_float4x4,
        depth: DepthRunner,
        frameIndex: Int
    ) {
        let modelSize = CGSize(width: 518, height: 392)
        let ciCtx = CIContext()  // per-call context for thread safety
        guard let bgra = PixelBufferRGB.convert(capturedImage,
                                                  toSize: modelSize,
                                                  ciContext: ciCtx,
                                                  rotateToPortrait: false) else {
            return
        }
        guard let result = try? depth.infer(pixelBuffer: bgra) else { return }

        let scaledIntrinsics = Intrinsics(
            fx: intrinsics.fx * Float(result.width) / Float(intrinsics.width),
            fy: intrinsics.fy * Float(result.height) / Float(intrinsics.height),
            cx: Float(result.width) / 2,
            cy: Float(result.height) / 2,
            width: result.width,
            height: result.height
        )

        // Accumulate on mlQueue (this thread) — pointCloud is nonisolated(unsafe)
        // so Swift concurrency allows it. Serial access is guaranteed because
        // mlQueue is serial and finalize() uses mlQueue.sync to drain.
        // pixelStride 8 (was 4) = 4× less work per inference.
        pointCloud.addDepthMap(
            result.depths,
            width: result.width,
            height: result.height,
            intrinsics: scaledIntrinsics,
            worldFromCamera: pose,
            colorAt: { _, _ in SIMD3<UInt8>(128, 128, 128) },
            pixelStride: 8
        )
        let count = pointCloud.count

        // Store a snapshot for YOLO unprojection on subsequent frames.
        self.storeDepthSnapshot(DepthSnapshot(
            depths: result.depths,
            width: result.width,
            height: result.height,
            intrinsics: scaledIntrinsics,
            worldFromCamera: pose
        ))

        // Only the small coverage update hops to MainActor.
        Task { @MainActor [weak self] in
            self?.coverageProgress = min(1.0, Double(count) / 50_000)
        }
    }

    /// Runs on mlQueue (background). Publishes results to main.
    private nonisolated func runYoloBackground(
        capturedImage: CVPixelBuffer,
        cameraTransform: simd_float4x4,
        yolo: YoloRunner,
        frameIndex: Int
    ) {
        let modelSize = CGSize(width: 640, height: 640)
        let ciCtx = CIContext()
        guard let bgra = PixelBufferRGB.convert(capturedImage,
                                                  toSize: modelSize,
                                                  ciContext: ciCtx,
                                                  rotateToPortrait: true) else {
            return
        }
        guard let dets = try? yolo.infer(pixelBuffer: bgra, frameIndex: frameIndex) else {
            return
        }

        // Need a recent depth snapshot to unproject. If we haven't produced
        // one yet (depth hasn't run, or lost), drop this YOLO result — the
        // placeholder "2m in front" was worse than dropping.
        guard let snap = self.latestDepthSnapshot() else {
            Task { @MainActor [weak self] in
                self?.logger.logEvent("yolo_detection_no_depth", at: self?.elapsed() ?? 0,
                                        data: ["frame": "\(frameIndex)",
                                               "detections": "\(dets.count)"])
            }
            return
        }

        // Observe each detection at its real world position.
        var observations: [(YoloDetection, SIMD3<Float>)] = []
        for d in dets where d.confidence > 0.5 {
            let center = SIMD2<Float>(Float(d.bbox.midX), Float(d.bbox.midY))
            guard let worldPos = YoloUnprojector.worldPosition(
                yoloBboxCenterNormalized: center,
                depthMap: snap.depths,
                depthWidth: snap.width, depthHeight: snap.height,
                depthIntrinsics: snap.intrinsics,
                worldFromCamera: snap.worldFromCamera
            ) else { continue }
            observations.append((d, worldPos))
        }

        Task { @MainActor [weak self] in
            guard let self else { return }
            self.recentDetections = dets
            for (d, pos) in observations {
                self.tracker.observe(category: d.category,
                                      worldPosition: pos,
                                      confidence: d.confidence,
                                      frameIndex: d.frameIndex)
            }
        }
    }
}
