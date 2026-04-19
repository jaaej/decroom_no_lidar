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
    let pointCloud = PointCloudAccumulator(voxelSize: 0.02, maxPoints: 500_000)

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

    // Panorama — array of (thumbnail, yaw) drawn directly on Canvas.
    // Simpler and more reliable than equirectangular projection.
    @Published var panoramaThumbnails: [(image: CGImage, yaw: Float)] = []

    // Internal state
    private let mlQueue = DispatchQueue(label: "decroom.scan-session.ml", qos: .userInitiated)
    private let ciContext = CIContext()
    private let startTime = Date()
    private var frameIndex = 0
    private var lastDepthFrame = -1
    private var lastYoloFrame = -1
    private let depthFrameInterval = 6    // ~5fps at 30fps
    private let yoloFrameInterval = 15    // ~2fps at 30fps
    private let panoramaFrameInterval = 10 // ~3fps for panorama capture
    private var lastPanoramaFrame = -1
    private let totalYawBins = 24
    private let panoContext = CIContext()

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

        // Capture panorama keyframe (lightweight — just a CGImage projection)
        if frameIndex - lastPanoramaFrame >= panoramaFrameInterval {
            lastPanoramaFrame = frameIndex
            let pitch = cameraPitch(from: frame.camera.transform)
            capturePanoramaFrame(frame: frame, yaw: yaw, pitch: pitch)
        }

        guard modelsReady else { return }

        // Dispatch depth inference to background (non-blocking)
        if let depth = depthRunner, frameIndex - lastDepthFrame >= depthFrameInterval {
            lastDepthFrame = frameIndex
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
            }
        }

        // Dispatch YOLO inference to background (non-blocking)
        if let yolo = yoloRunner, frameIndex - lastYoloFrame >= yoloFrameInterval {
            lastYoloFrame = frameIndex
            let capturedImage = frame.capturedImage
            let cameraTransform = frame.camera.transform
            let fi = frameIndex

            mlQueue.async { [weak self] in
                self?.runYoloBackground(
                    capturedImage: capturedImage,
                    cameraTransform: cameraTransform,
                    yolo: yolo,
                    frameIndex: fi
                )
            }
        }
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

    func finalize() -> (point_cloud_ply: Data, log_written: Bool) {
        logger.logEvent("scan_finalized", at: elapsed())
        let ply = pointCloud.exportPLY()
        let logOK = (try? logger.finalize()) != nil
        return (ply, logOK)
    }

    // MARK: - Private

    private func elapsed() -> Double { Date().timeIntervalSince(startTime) }

    private func cameraYaw(from transform: simd_float4x4) -> Float {
        let fx = -transform.columns.2.x
        let fz = -transform.columns.2.z
        return atan2(fx, fz)
    }

    private func cameraPitch(from transform: simd_float4x4) -> Float {
        let fy = -transform.columns.2.y
        let fxz = sqrt(transform.columns.2.x * transform.columns.2.x +
                        transform.columns.2.z * transform.columns.2.z)
        return atan2(fy, fxz)
    }

    private func capturePanoramaFrame(frame: ARFrame, yaw: Float, pitch: Float) {
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)

        // Create a thumbnail (480×640) for the panorama strip.
        // Full-res would use too much memory at 100+ frames.
        let thumbW = 480
        let thumbH = 640
        let scale = CGFloat(thumbW) / ciImage.extent.width
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let cgImage = panoContext.createCGImage(scaled, from: CGRect(x: 0, y: 0, width: thumbW, height: thumbH)) else {
            return
        }
        panoramaThumbnails.append((image: cgImage, yaw: yaw))
    }

    /// Runs on mlQueue (background). Publishes results to main.
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

        Task { @MainActor [weak self] in
            guard let self else { return }
            self.pointCloud.addDepthMap(
                result.depths,
                width: result.width,
                height: result.height,
                intrinsics: scaledIntrinsics,
                worldFromCamera: pose,
                colorAt: { _, _ in SIMD3<UInt8>(128, 128, 128) },
                pixelStride: 4
            )
            self.coverageProgress = min(1.0, Double(self.pointCloud.count) / 50_000)
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

        let cameraPos = SIMD3<Float>(
            cameraTransform.columns.3.x,
            cameraTransform.columns.3.y,
            cameraTransform.columns.3.z
        )
        let fwd = -SIMD3<Float>(
            cameraTransform.columns.2.x,
            cameraTransform.columns.2.y,
            cameraTransform.columns.2.z
        )

        Task { @MainActor [weak self] in
            guard let self else { return }
            self.recentDetections = dets
            for d in dets where d.confidence > 0.5 {
                let approxPos = cameraPos + fwd * 2
                self.tracker.observe(category: d.category,
                                      worldPosition: approxPos,
                                      confidence: d.confidence,
                                      frameIndex: frameIndex)
            }
        }
    }
}
