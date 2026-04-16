//
//  ScanSession.swift
//  decroom_no_lidar
//
//  Coordinates Plan 1 + Plan 2 components during a scan. Owned by the
//  ARRoomPreviewView controller, fed by ARFrame updates from the session
//  delegate. Publishes @Published state for the view to render.
//
//  Core ML models load on a background queue to avoid blocking main during
//  view appearance — iOS watchdog kills apps that block main for too long
//  on cold launch, and the depth + YOLO models together are ~70 MB.
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
    @Published var coverageProgress: Double = 0.0   // 0..1
    @Published var modelsReady: Bool = false

    // Internal state
    private let ciContext = CIContext()
    private let startTime = Date()
    private var frameIndex = 0
    private var lastYoloFrame = -1
    private let yoloFrameInterval = 15  // run YOLO every 15th ARFrame (~2fps at 30fps)

    init() {
        self.logger = ScanLogger()
        logger.recordDevice(DeviceInfo.current())
        logger.logEvent("scan_started", at: 0)

        // Load Core ML models on background queue so the view can appear
        // immediately. Inference is gated on `modelsReady` and the optionals.
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

    /// Called from ARSessionDelegate on each new frame.
    func update(frame: ARFrame) {
        frameIndex += 1

        // Skip ML inference until models are loaded — but pose tracking and
        // plane anchors continue to work via the ARSession itself.
        guard modelsReady else { return }

        let intrinsics = Intrinsics.make(from: frame.camera, portrait: true)
        let pose = frame.camera.transform

        if let depth = depthRunner, frameIndex % 6 == 0 {
            runDepth(frame: frame, intrinsics: intrinsics, pose: pose, depth: depth)
        }

        if let yolo = yoloRunner, frameIndex - lastYoloFrame >= yoloFrameInterval {
            lastYoloFrame = frameIndex
            runYolo(frame: frame, yolo: yolo)
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

    private func runDepth(frame: ARFrame,
                           intrinsics: Intrinsics,
                           pose: simd_float4x4,
                           depth: DepthRunner) {
        let modelSize = CGSize(width: 518, height: 392)
        guard let bgra = PixelBufferRGB.convert(frame.capturedImage,
                                                  toSize: modelSize,
                                                  ciContext: ciContext,
                                                  rotateToPortrait: false) else {
            return
        }
        guard let result = try? depth.infer(pixelBuffer: bgra) else { return }

        pointCloud.addDepthMap(
            result.depths,
            width: result.width,
            height: result.height,
            intrinsics: Intrinsics(
                fx: intrinsics.fx * Float(result.width) / Float(intrinsics.width),
                fy: intrinsics.fy * Float(result.height) / Float(intrinsics.height),
                cx: Float(result.width) / 2,
                cy: Float(result.height) / 2,
                width: result.width,
                height: result.height
            ),
            worldFromCamera: pose,
            colorAt: { _, _ in SIMD3<UInt8>(128, 128, 128) },
            pixelStride: 4
        )

        coverageProgress = min(1.0, Double(pointCloud.count) / 50_000)
    }

    private func runYolo(frame: ARFrame, yolo: YoloRunner) {
        let modelSize = CGSize(width: 640, height: 640)
        guard let bgra = PixelBufferRGB.convert(frame.capturedImage,
                                                  toSize: modelSize,
                                                  ciContext: ciContext,
                                                  rotateToPortrait: true) else {
            return
        }
        guard let dets = try? yolo.infer(pixelBuffer: bgra, frameIndex: frameIndex) else {
            return
        }
        self.recentDetections = dets

        let cameraPos = SIMD3<Float>(
            frame.camera.transform.columns.3.x,
            frame.camera.transform.columns.3.y,
            frame.camera.transform.columns.3.z
        )
        for d in dets where d.confidence > 0.5 {
            let fwd = -SIMD3<Float>(
                frame.camera.transform.columns.2.x,
                frame.camera.transform.columns.2.y,
                frame.camera.transform.columns.2.z
            )
            let approxPos = cameraPos + fwd * 2
            tracker.observe(category: d.category,
                             worldPosition: approxPos,
                             confidence: d.confidence,
                             frameIndex: frameIndex)
        }
    }
}
