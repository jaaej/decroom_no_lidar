//
//  ContentView.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import SwiftUI
import ARKit
import RealityKit
import CoreImage
import AVFoundation
import Accelerate

struct ContentView: View {
    @ObservedObject var processor: RecordingProcessor

    var body: some View {
        GuidedCaptureView(processor: processor)
    }
}

// MARK: - Guided Capture (Coverage-Driven)

struct GuidedCaptureView: View {
    @StateObject private var recorder = GuidedARFrameRecorder()
    @StateObject private var coverage = CoverageEngine()
    @ObservedObject var processor: RecordingProcessor
    @State private var lastError: String?
    @State private var showPreview = false
    @State private var justCaptured = false
    @State private var isCapturing = false

    private struct ReadinessCheck {
        let label: String
        let passed: Bool
        let detail: String
    }

    var body: some View {
        ZStack {
            // LAYER 1: Full-screen AR camera
            GuidedARViewContainer(recorder: recorder)
                .ignoresSafeArea()

            // LAYER 2: Coverage checklist overlay + directional chevron
            ScanGuidanceOverlayView(
                capturedCount: coverage.capturedCount,
                minimumShots: coverage.minimumShots,
                isWalkedAround: coverage.spatialClusterCount >= coverage.requiredSpatialClustersForUpload
                             || coverage.maxDistanceFromStart >= coverage.requiredTravelDistanceForUpload,
                isMultipleAngles: coverage.uniqueYawBinCount >= 6
                                && coverage.largestYawGap <= .pi,
                isCapturedDetail: coverage.goodFeatureShots >= 3
                                || coverage.featureVoxelCount >= 300,
                isReadyToUpload: coverage.isReadyToUpload,
                turnAngle: (coverage.nextSequenceYaw ?? coverage.largestGapCenterYaw).map {
                    normalizeAngle($0 - recorder.smoothedYaw)
                },
                largestGap: coverage.largestYawGap,
                dynamicPrompt: livePolicyHint
            )

            // LAYER 3: Bottom controls panel
            VStack(spacing: 0) {
                Spacer()

                // BOTTOM: Compact controls panel
                VStack(spacing: 6) {
                    // Single-line hint or error
                    if let lastError {
                        Label(lastError, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red)
                            .font(.caption)
                            .lineLimit(1)
                    } else {
                        compactHint
                    }

                    // Buttons row
                    HStack(spacing: 12) {
                        Button {
                            captureShot()
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "camera.circle.fill")
                                    .font(.title3)
                                Text("Capture")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(captureButtonColor)
                        .disabled(!coverage.canCapture() || isCapturing)

                        Button("Upload Now") {
                            // TODO(Plan 3b): photo-mode upload not wired for phone-first; remove the tab or implement.
                            _ = recorder.stop()
                        }
                        .buttonStyle(.bordered)
                        .disabled(!coverage.isReadyToUpload)
                    }

                    readinessChecksView
                    statusSection
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial)
            }

            // Flash overlay on capture
            if justCaptured {
                Color.white
                    .opacity(0.3)
                    .allowsHitTesting(false)
                    .transition(.opacity)
                    .ignoresSafeArea()
            }
        }
        .onAppear {
            recorder.start()
            coverage.reset()
        }
        .onDisappear {
            recorder.stop()
        }
        .onChange(of: recorder.currentYaw) { _, _ in
            let pos = SIMD3<Float>(
                recorder.latestCameraTransform.columns.3.x,
                recorder.latestCameraTransform.columns.3.y,
                recorder.latestCameraTransform.columns.3.z
            )
            DispatchQueue.main.async {
                coverage.updateLiveState(position: pos, timestamp: CACurrentMediaTime())
            }
        }
        .navigationDestination(isPresented: $showPreview) {
            if case .finished(let payload) = processor.state {
                RoomPreviewPage(payload: payload, processor: processor)
            }
        }
    }

    // MARK: - Capture Button Color

    private var captureButtonColor: Color {
        if !coverage.canCapture() || isCapturing { return .gray }
        if recorder.currentTrackingQuality != "normal" { return .orange }
        if let policy = liveCapturePolicy, policy.overlapScore < coverage.minOverlapPolicyScore {
            return .yellow
        }
        return .blue
    }

    // MARK: - Compact Hint (single line)

    @ViewBuilder
    private var compactHint: some View {
        if coverage.capturedCount == 0 {
            Text("Point at the room and tap Capture")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else if isGoodAngle {
            Label("Good angle", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
        } else if recorder.currentTrackingQuality != "normal" {
            Label("Tracking is limited. You can still capture; hold steady.", systemImage: "hand.raised")
                .font(.caption)
                .foregroundStyle(.orange)
        } else if let liveHint = livePolicyHint {
            Label(liveHint, systemImage: "scope")
                .font(.caption)
                .foregroundStyle(.yellow)
        } else if coverage.isReadyToUpload {
            Label("Good coverage! Tap Upload Now", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
        } else {
            Text(missingCoverageHint)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    /// Single-line hint matching the first unchecked checklist item
    private var missingCoverageHint: String {
        if coverage.capturedCount < coverage.minimumShots {
            return "Take \(coverage.minimumShots - coverage.capturedCount) more photos"
        }
        if coverage.spatialClusterCount < coverage.requiredSpatialClustersForUpload {
            return "Walk to another spot in the room"
        }
        if coverage.maxDistanceFromStart < coverage.requiredTravelDistanceForUpload {
            return "Move further across the room"
        }
        if coverage.uniqueYawBinCount < 6 || coverage.largestYawGap > .pi {
            return "Turn to face other walls"
        }
        if coverage.goodFeatureShots < 3 && coverage.featureVoxelCount < 300 {
            return "Point at furniture or textured surfaces"
        }
        if coverage.goodFeatureSpreadShots < coverage.minFeatureSpreadShots
            && coverage.averageFeatureSpreadCoverage < 0.22
            && coverage.capturedCount < 16 {
            return "Pan across uncovered image regions before upload"
        }
        return "Keep capturing from new angles"
    }

    private var isGoodAngle: Bool {
        guard let target = coverage.nextSequenceYaw ?? coverage.largestGapCenterYaw else {
            return false
        }
        let delta = abs(normalizeAngle(target - recorder.smoothedYaw))
        return delta <= (.pi / 12) // ±15°
    }

    private var liveCapturePolicy: CoverageEngine.CapturePolicyFeedback? {
        guard coverage.capturedCount > 0 else { return nil }
        let position = SIMD3<Float>(
            recorder.latestCameraTransform.columns.3.x,
            recorder.latestCameraTransform.columns.3.y,
            recorder.latestCameraTransform.columns.3.z
        )
        return coverage.previewCapturePolicy(
            yaw: recorder.currentYaw,
            pitch: recorder.currentPitch,
            position: position,
            featurePointCount: recorder.currentFeaturePointCount,
            trackingQuality: recorder.currentTrackingQuality
        )
    }

    private var livePolicyHint: String? {
        guard let policy = liveCapturePolicy else { return nil }
        if policy.requiresStabilizingShot, let step = policy.stepM {
            return String(format: "Large move (%.2fm). Take one intermediate overlap shot.", step)
        }
        if policy.overlapScore < coverage.minOverlapPolicyScore {
            return "Rotate less and capture sooner for stronger overlap."
        }
        if policy.textureScore < 0.25 {
            return "Aim at textured/edge-rich areas before capture."
        }
        return nil
    }

    private var readinessChecks: [ReadinessCheck] {
        [
            ReadinessCheck(
                label: "Shots",
                passed: coverage.capturedCount >= coverage.minimumShots,
                detail: "\(coverage.capturedCount)/\(coverage.minimumShots)"
            ),
            ReadinessCheck(
                label: "Yaw bins",
                passed: coverage.uniqueYawBinCount >= 10,
                detail: "\(coverage.uniqueYawBinCount)/10"
            ),
            ReadinessCheck(
                label: "Movement",
                passed: coverage.spatialClusterCount >= coverage.requiredSpatialClustersForUpload,
                detail: "\(coverage.spatialClusterCount)/\(coverage.requiredSpatialClustersForUpload)"
            ),
            ReadinessCheck(
                label: "Travel",
                passed: coverage.maxDistanceFromStart >= coverage.requiredTravelDistanceForUpload,
                detail: String(format: "%.2fm", coverage.maxDistanceFromStart)
            ),
            ReadinessCheck(
                label: "Detail",
                passed: coverage.goodFeatureShots >= 4 || coverage.featureVoxelCount >= 450,
                detail: "\(coverage.goodFeatureShots) shots / \(coverage.featureVoxelCount) voxels"
            ),
            ReadinessCheck(
                label: "Spread",
                passed: coverage.goodFeatureSpreadShots >= coverage.minFeatureSpreadShots
                     || coverage.averageFeatureSpreadCoverage >= 0.22
                     || coverage.capturedCount >= 16,
                detail: "\(coverage.goodFeatureSpreadShots)/\(coverage.minFeatureSpreadShots) shots / \(Int((coverage.averageFeatureSpreadCoverage * 100).rounded()))%"
            )
        ]
    }

    @ViewBuilder
    private var readinessChecksView: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Upload Readiness")
                .font(.caption2)
                .foregroundStyle(.secondary)
            ForEach(Array(readinessChecks.enumerated()), id: \.offset) { _, check in
                HStack(spacing: 6) {
                    Image(systemName: check.passed ? "checkmark.circle.fill" : "xmark.circle")
                        .foregroundStyle(check.passed ? .green : .orange)
                        .font(.caption2)
                    Text("\(check.label): \(check.detail)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
    }

    // MARK: - Capture Action

    private func captureShot() {
        if isCapturing { return }
        let position = SIMD3<Float>(
            recorder.latestCameraTransform.columns.3.x,
            recorder.latestCameraTransform.columns.3.y,
            recorder.latestCameraTransform.columns.3.z
        )

        let policy = coverage.previewCapturePolicy(
            yaw: recorder.currentYaw,
            pitch: recorder.currentPitch,
            position: position,
            featurePointCount: recorder.currentFeaturePointCount,
            trackingQuality: recorder.currentTrackingQuality
        )
        if policy.isSequenceBreak {
            if let step = policy.stepM, let yawDelta = policy.yawDeltaDeg {
                lastError = String(
                    format: "Continuity break (%.2fm, %.0f°). Take one intermediate overlap shot.",
                    step,
                    yawDelta
                )
            } else {
                lastError = "Continuity break. Take one intermediate overlap shot."
            }
            return
        }
        if policy.requiresStabilizingShot {
            if let step = policy.stepM {
                lastError = String(format: "Large camera jump (%.2fm). Take one intermediate overlap shot first.", step)
            } else {
                lastError = "Large camera jump. Take one intermediate overlap shot first."
            }
            return
        }
        if coverage.capturedCount >= 2 && policy.overlapScore < 0.16 {
            lastError = "Very low overlap predicted. Rotate 10–30° from the last shot and capture again."
            return
        }
        if policy.textureScore < 0.18 {
            lastError = "Low-detail view. Capture accepted, but include textured overlap next."
        }

        // Check capture gating
        if let reason = coverage.canCaptureHere(
            yaw: recorder.currentYaw,
            pitch: recorder.currentPitch,
            position: position,
            featurePointCount: recorder.currentFeaturePointCount
        ) {
            var shouldBlockCapture = true
            switch reason {
            case .tooCloseAngular:
                lastError = "Too similar to a previous shot. Turn about 20–30° and capture."
            case .tooCloseSpatial:
                lastError = "Too close to an existing capture. Shift a little or rotate further."
            case .insufficientFeatures:
                lastError = "Low detail in view. Capture accepted, but include textured surfaces next."
                shouldBlockCapture = false
            case .movingTooFast:
                lastError = "Hold still \u{2014} moving too fast."
            case .outOfSequence:
                lastError = "Not ideal order. Capture accepted; rotate toward the guidance arrow next."
                shouldBlockCapture = false
            }
            if shouldBlockCapture {
                return
            }
        }

        isCapturing = true

        // Capture the frame (image processing happens on background thread)
        recorder.captureAsync(
            motionOverlapScore: policy.overlapScore,
            sequenceBreakFlag: policy.isSequenceBreak
        ) { result in
            switch result {
            case .success(let capture):
                if policy.textureScore >= 0.18 && policy.overlapScore >= coverage.minOverlapPolicyScore {
                    lastError = nil
                }
                coverage.addCapture(
                    yaw: capture.yaw,
                    pitch: capture.pitch,
                    position: capture.position,
                    featurePoints: capture.featurePoints,
                    featureImageCoverage: capture.featureImageCoverage,
                    featureGridCellCount: capture.featureGridCellCount,
                    trackingQuality: recorder.currentTrackingQuality
                )

                // LiDAR mode: snapshot mesh only at capture time.
                // Non-LiDAR fallback: place capture marker only.
                if recorder.supportsSceneMesh {
                    recorder.snapshotNearbyMesh(at: capture.cameraTransform)
                } else {
                    recorder.addCaptureMarker(at: capture.cameraTransform)
                }

                // Brief flash effect
                withAnimation(.easeOut(duration: 0.15)) {
                    justCaptured = true
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                    withAnimation(.easeIn(duration: 0.1)) {
                        justCaptured = false
                    }
                }

            case .failure(let error):
                lastError = errorMessage(error)
            }
            isCapturing = false
        }
    }

    // MARK: - Status Section

    @ViewBuilder
    private var statusSection: some View {
        switch processor.state {
        case .idle:
            EmptyView()
        case .extracting:
            HStack(spacing: 8) {
                ProgressView()
                Text("Extracting frames...")
            }
        case .uploading:
            HStack(spacing: 8) {
                ProgressView()
                Text("Uploading to backend...")
            }
        case .finished:
            Button {
                showPreview = true
            } label: {
                Label("View 3D Room", systemImage: "cube.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(.green)
        case .failed(let message):
            VStack(alignment: .leading, spacing: 8) {
                Label("Processing failed", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                Text(message)
                    .font(.subheadline)
                Button("Try again") {
                    processor.reset()
                }
            }
        }
    }

    private func errorMessage(_ error: GuidedARFrameRecorder.CaptureError) -> String {
        switch error {
        case .noFrame:
            return "No AR frame yet. Hold steady and try again."
        case .blurry:
            return "Shot is too blurry. Hold steady and retry."
        case .dark:
            return "Shot is too dark. Add light or tilt toward brighter area."
        }
    }
}

// MARK: - Room Preview Page (Separate Navigation Destination)

struct RoomPreviewPage: View {
    let payload: RoomPayload
    @ObservedObject var processor: RecordingProcessor

    var body: some View {
        VStack(spacing: 16) {
            RoomSceneView(payload: payload)
                .frame(maxHeight: .infinity)

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Label("\(payload.objects.count) objects", systemImage: "chair.lounge")
                    Spacer()
                    Label(
                        "\(String(format: "%.1f", payload.envelope.width))m x \(String(format: "%.1f", payload.envelope.length))m x \(String(format: "%.1f", payload.envelope.height))m",
                        systemImage: "ruler"
                    )
                }
                .font(.subheadline)
                .foregroundStyle(.secondary)
            }
            .padding(.horizontal)
        }
        .padding()
        .navigationTitle("Room Preview")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("Reset") {
                    processor.reset()
                }
            }
        }
    }
}

// MARK: - AR View Container

struct GuidedARViewContainer: UIViewRepresentable {
    @ObservedObject var recorder: GuidedARFrameRecorder

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        let config = ARWorldTrackingConfiguration()
        config.isAutoFocusEnabled = true
        config.planeDetection = [.horizontal, .vertical]
        let supportsMesh = ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
        if supportsMesh {
            config.sceneReconstruction = .mesh
        }
        arView.session.run(config)
        arView.session.delegate = context.coordinator
        // Important: keep live mesh overlay OFF. We only persist snapshots at capture.
        arView.debugOptions.remove(.showSceneUnderstanding)
        recorder.setSceneMeshSupport(enabled: supportsMesh)
        recorder.arView = arView  // let recorder place capture markers
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        // Nothing to update dynamically
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(recorder: recorder)
    }

    final class Coordinator: NSObject, ARSessionDelegate {
        private let recorder: GuidedARFrameRecorder

        init(recorder: GuidedARFrameRecorder) {
            self.recorder = recorder
        }

        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            Task { @MainActor in
                recorder.handle(frame: frame)
            }
        }

        func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
            Task { @MainActor in
                recorder.upsertPlaneAnchors(anchors)
                recorder.upsertMeshAnchors(anchors)
            }
        }

        func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
            Task { @MainActor in
                recorder.upsertPlaneAnchors(anchors)
                recorder.upsertMeshAnchors(anchors)
            }
        }

        func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
            Task { @MainActor in
                recorder.removePlaneAnchors(anchors)
                recorder.removeMeshAnchors(anchors)
            }
        }
    }
}

// MARK: - Guided AR Frame Recorder

@MainActor
final class GuidedARFrameRecorder: ObservableObject {
    struct CaptureResult {
        let frames: [CGImage]
        let metadataJSON: String?
    }

    /// Data returned on successful capture
    struct CaptureData {
        let yaw: Float
        let pitch: Float
        let position: SIMD3<Float>
        let featurePoints: [SIMD3<Float>]
        let cameraTransform: simd_float4x4
        let featureImageCoverage: Float
        let featureGridCellCount: Int
    }

    enum CaptureError: Error {
        case noFrame
        case blurry
        case dark
    }

    @Published private(set) var isRecording = false
    @Published private(set) var capturedCount: Int = 0
    @Published private(set) var currentYaw: Float = 0       // raw yaw in radians [0, 2pi)
    @Published private(set) var currentPitch: Float = 0     // radians, negative = looking down
    @Published private(set) var smoothedYaw: Float = 0      // EMA-smoothed for UI display
    @Published private(set) var smoothedPitch: Float = 0    // EMA-smoothed for UI display
    @Published private(set) var latestCameraTransform: simd_float4x4 = matrix_identity_float4x4
    @Published private(set) var currentTrackingQuality: String = "not_available"
    @Published private(set) var currentFeaturePointCount: Int = 0
    @Published private(set) var supportsSceneMesh: Bool = false

    // Video mode properties
    var isVideoMode: Bool = false
    private var lastKeyframeFeatureCount: Int = 0
    private var lastKeyframeTransform: simd_float4x4?
    private var lastKeyframeTime: TimeInterval = 0
    private var videoFrameCount: Int = 0
    private(set) var keyframeJPEGPaths: [URL] = []
    private var scanStartTime: TimeInterval = 0
    private let candidateInterval: Int = 10  // check every 10th frame ≈ 3fps candidates

    /// Callback invoked on main thread when a keyframe is accepted in video mode
    var onKeyframeCaptured: ((CaptureData) -> Void)?

    // AR scene feedback
    weak var arView: ARView?
    private var markerRootAnchor: AnchorEntity?
    private var markerEntities: [ModelEntity] = []
    private let markerCap = 80
    private let markerMesh = MeshResource.generateSphere(radius: 0.03)
    private let markerMaterial = UnlitMaterial(color: UIColor.systemBlue.withAlphaComponent(0.5))

    private var livePlaneAnchors: [UUID: ARPlaneAnchor] = [:]
    private let planeCaptureRadius: Float = 5.0
    private let maxPlaneAnchorsPerShot = 24

    private var liveMeshAnchors: [UUID: ARMeshAnchor] = [:]
    private var persistedMeshRootAnchor: AnchorEntity?
    private var persistedMeshEntities: [UUID: ModelEntity] = [:]
    private var persistedMeshOrder: [UUID] = []
    private let persistedMeshCap = 96
    private let meshCaptureRadius: Float = 2.6
    private let meshCaptureConeCos: Float = 0.2
    private let maxMeshAnchorsPerCapture = 12
    private let persistedMeshMaterial = UnlitMaterial(color: UIColor.systemCyan.withAlphaComponent(0.03))

    private var frames: [CGImage] = []
    private var shots: [GuidedShotPayload] = []
    private var latestFrame: ARFrame?
    /// On-device phone-first session. Accumulates geometry, point cloud,
    /// and keyframes in parallel with the existing video capture. Created
    /// lazily on start() so the heavy model loads don't happen at app launch.
    private(set) var scanSession: ScanSession?
    private let ciContext = CIContext()
    private var shotCounter: Int = 0
    private var lastCapturePosition: SIMD3<Float>? = nil
    private var lastCaptureYaw: Float? = nil

    private let blurVarianceThreshold: Float = 45.0
    private let minBrightness: Float = 0.15
    private let smoothingAlpha: Float = 0.15

    func setSceneMeshSupport(enabled: Bool) {
        supportsSceneMesh = enabled
    }

    func start() {
        scanSession = ScanSession()
        clearScanFeedback()
        frames = []
        shots = []
        latestFrame = nil
        capturedCount = 0
        currentYaw = 0
        currentPitch = 0
        smoothedYaw = 0
        smoothedPitch = 0
        latestCameraTransform = matrix_identity_float4x4
        currentTrackingQuality = "not_available"
        currentFeaturePointCount = 0
        shotCounter = 0
        lastCapturePosition = nil
        lastCaptureYaw = nil

        // Video mode reset — clean up previous keyframe files from disk
        cleanupKeyframeFiles()
        lastKeyframeFeatureCount = 0
        lastKeyframeTransform = nil
        lastKeyframeTime = 0
        videoFrameCount = 0
        keyframeJPEGPaths = []
        scanStartTime = 0

        if isVideoMode {
            lockExposureAndFocus()
        }

        isRecording = true
    }

    @discardableResult
    func stop() -> CaptureResult {
        isRecording = false
        // Keep scanSession alive — caller will invoke finalize() before releasing.
        if isVideoMode {
            unlockExposureAndFocus()
        }
        let ordered = zip(frames, shots)
            .sorted { $0.1.timestamp < $1.1.timestamp }
        let orderedFrames = ordered.map { $0.0 }
        let orderedShots = ordered.map { $0.1 }
        let metadataJSON = buildMetadataJSON(shots: orderedShots)
        clearScanFeedback()
        return CaptureResult(frames: orderedFrames, metadataJSON: metadataJSON)
    }

    /// Returns a CaptureResult using disk-based keyframe JPEGs (video mode)
    func stopVideoMode() -> (jpegPaths: [URL], metadataJSON: String?) {
        isRecording = false
        // Keep scanSession alive — caller will invoke finalize() before releasing.
        unlockExposureAndFocus()
        let ordered = shots.sorted { $0.timestamp < $1.timestamp }
        let metadataJSON = buildMetadataJSON(shots: ordered)
        clearScanFeedback()
        return (jpegPaths: keyframeJPEGPaths, metadataJSON: metadataJSON)
    }

    /// Build and return the phone-first upload payload. Must be called after
    /// stopVideoMode() (so keyframeJPEGPaths are finalized). On any phone-side
    /// validation failure, throws a ScanError.
    @MainActor
    func finalizePhoneFirstPayload() async throws -> ScanUploadPayload {
        guard let session = scanSession else {
            throw ScanError.payloadBuildFailed(
                detail: "ScanSession missing — start() was never called?"
            )
        }

        // keyframeJPEGPaths and shots are appended pairwise in video-mode
        // capture; they are index-aligned. Read JPEG data from disk and build
        // KeyframeMetadata from the matching shot.
        var jpegs: [Data] = []
        var metas: [ScanUploadPayload.KeyframeMetadata] = []

        for (i, url) in keyframeJPEGPaths.enumerated() {
            let data: Data
            do {
                data = try Data(contentsOf: url)
            } catch {
                throw ScanError.payloadBuildFailed(
                    detail: "read keyframe \(url.lastPathComponent): \(error.localizedDescription)"
                )
            }

            // Prefer the matching shot; fall back to iPhone 15 Pro portrait defaults
            // if indexing doesn't line up (can happen in edge cases where some shots
            // are photo-mode-only with no JPEG).
            let shot: GuidedShotPayload? = (i < shots.count) ? shots[i] : nil
            let intr: Intrinsics
            let pose: simd_float4x4
            let timestamp: Double

            if let shot = shot {
                intr = Intrinsics(
                    fx: shot.intrinsics.fx,
                    fy: shot.intrinsics.fy,
                    cx: shot.intrinsics.cx,
                    cy: shot.intrinsics.cy,
                    width: shot.intrinsics.width,
                    height: shot.intrinsics.height
                )
                pose = matrixFromRowMajor(shot.pose.matrix)
                timestamp = shot.timestamp
            } else {
                intr = Intrinsics(fx: 1500, fy: 1500, cx: 720, cy: 960,
                                   width: 1440, height: 1920)
                pose = matrix_identity_float4x4
                timestamp = 0
            }

            jpegs.append(data)
            metas.append(ScanUploadPayload.KeyframeMetadata(
                filename: "frame_\(String(format: "%02d", i)).jpg",
                timestamp: timestamp,
                poseWorldFromCamera: pose,
                intrinsics: intr
            ))
        }

        return try session.finalize(keyframes: metas, keyframeJPEGs: jpegs)
    }

    /// Reconstruct a simd_float4x4 from a row-major 16-float array.
    /// Matches GuidedShotPayload.pose.matrix format produced by toRowMajorArray().
    private func matrixFromRowMajor(_ m: [Float]) -> simd_float4x4 {
        guard m.count == 16 else { return matrix_identity_float4x4 }
        return simd_float4x4(
            SIMD4<Float>(m[0],  m[4],  m[8],  m[12]),
            SIMD4<Float>(m[1],  m[5],  m[9],  m[13]),
            SIMD4<Float>(m[2],  m[6],  m[10], m[14]),
            SIMD4<Float>(m[3],  m[7],  m[11], m[15])
        )
    }

    /// Remove keyframe JPEG files from temp directory to prevent disk/privacy leaks.
    func cleanupKeyframeFiles() {
        for url in keyframeJPEGPaths {
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// Adds a small semi-transparent marker at capture location.
    /// Marker is offset forward so it remains visible and avoids camera clipping.
    func addCaptureMarker(at cameraTransform: simd_float4x4) {
        guard let arView else { return }

        if markerRootAnchor == nil {
            let root = AnchorEntity(world: .zero)
            arView.scene.addAnchor(root)
            markerRootAnchor = root
        }
        guard let markerRootAnchor else { return }

        let cameraPos = SIMD3<Float>(
            cameraTransform.columns.3.x,
            cameraTransform.columns.3.y,
            cameraTransform.columns.3.z
        )
        let forward = normalize(SIMD3<Float>(
            -cameraTransform.columns.2.x,
            -cameraTransform.columns.2.y,
            -cameraTransform.columns.2.z
        ))
        let markerPos = cameraPos + forward * 0.45

        let marker = ModelEntity(mesh: markerMesh, materials: [markerMaterial])
        marker.position = markerPos
        markerRootAnchor.addChild(marker)
        markerEntities.append(marker)

        if markerEntities.count > markerCap {
            let oldest = markerEntities.removeFirst()
            oldest.removeFromParent()
        }
    }

    /// Clears all in-view scan feedback for a fresh session.
    func clearScanFeedback() {
        markerEntities.forEach { $0.removeFromParent() }
        markerEntities.removeAll()
        if let markerRootAnchor {
            markerRootAnchor.removeFromParent()
            self.markerRootAnchor = nil
        }

        persistedMeshEntities.values.forEach { $0.removeFromParent() }
        persistedMeshEntities.removeAll()
        persistedMeshOrder.removeAll()
        if let persistedMeshRootAnchor {
            persistedMeshRootAnchor.removeFromParent()
            self.persistedMeshRootAnchor = nil
        }
        livePlaneAnchors.removeAll()
        liveMeshAnchors.removeAll()
    }

    func upsertPlaneAnchors(_ anchors: [ARAnchor]) {
        for anchor in anchors {
            guard let planeAnchor = anchor as? ARPlaneAnchor else { continue }
            livePlaneAnchors[planeAnchor.identifier] = planeAnchor
        }
        // Also feed the phone-first room geometry builder
        let planes = anchors.compactMap { $0 as? ARPlaneAnchor }
        if !planes.isEmpty {
            scanSession?.updatePlaneAnchors(planes)
        }
    }

    func removePlaneAnchors(_ anchors: [ARAnchor]) {
        for anchor in anchors {
            guard let planeAnchor = anchor as? ARPlaneAnchor else { continue }
            livePlaneAnchors.removeValue(forKey: planeAnchor.identifier)
        }
        // Re-send the surviving set so the geometry builder re-derives walls.
        let surviving = Array(livePlaneAnchors.values)
        scanSession?.updatePlaneAnchors(surviving)
    }

    func upsertMeshAnchors(_ anchors: [ARAnchor]) {
        guard supportsSceneMesh else { return }
        for anchor in anchors {
            guard let meshAnchor = anchor as? ARMeshAnchor else { continue }
            liveMeshAnchors[meshAnchor.identifier] = meshAnchor
        }
    }

    func removeMeshAnchors(_ anchors: [ARAnchor]) {
        guard supportsSceneMesh else { return }
        for anchor in anchors {
            guard let meshAnchor = anchor as? ARMeshAnchor else { continue }
            liveMeshAnchors.removeValue(forKey: meshAnchor.identifier)
        }
    }

    func snapshotNearbyMesh(at cameraTransform: simd_float4x4) {
        guard supportsSceneMesh else { return }
        guard let arView else { return }
        guard !liveMeshAnchors.isEmpty else { return }

        if persistedMeshRootAnchor == nil {
            let root = AnchorEntity(world: .zero)
            arView.scene.addAnchor(root)
            persistedMeshRootAnchor = root
        }
        guard let persistedMeshRootAnchor else { return }

        let cameraPos = SIMD3<Float>(
            cameraTransform.columns.3.x,
            cameraTransform.columns.3.y,
            cameraTransform.columns.3.z
        )
        let forward = normalize(SIMD3<Float>(
            -cameraTransform.columns.2.x,
            -cameraTransform.columns.2.y,
            -cameraTransform.columns.2.z
        ))

        let nearbyAnchors = liveMeshAnchors.values
            .filter { anchor in
                let anchorPos = SIMD3<Float>(
                    anchor.transform.columns.3.x,
                    anchor.transform.columns.3.y,
                    anchor.transform.columns.3.z
                )
                let delta = anchorPos - cameraPos
                let distance = simd_length(delta)
                if distance > meshCaptureRadius || distance < 0.05 { return false }
                let direction = delta / distance
                let facingDot = simd_dot(direction, forward)
                return facingDot >= meshCaptureConeCos
            }
            .sorted { a, b in
                let pa = SIMD3<Float>(a.transform.columns.3.x, a.transform.columns.3.y, a.transform.columns.3.z)
                let pb = SIMD3<Float>(b.transform.columns.3.x, b.transform.columns.3.y, b.transform.columns.3.z)
                return simd_distance(pa, cameraPos) < simd_distance(pb, cameraPos)
            }
            .prefix(maxMeshAnchorsPerCapture)

        for anchor in nearbyAnchors {
            guard let mesh = Self.meshResource(from: anchor.geometry) else { continue }

            if let existing = persistedMeshEntities[anchor.identifier] {
                existing.removeFromParent()
                persistedMeshEntities.removeValue(forKey: anchor.identifier)
                if let existingIndex = persistedMeshOrder.firstIndex(of: anchor.identifier) {
                    persistedMeshOrder.remove(at: existingIndex)
                }
            }

            let entity = ModelEntity(mesh: mesh, materials: [persistedMeshMaterial])
            entity.transform = Transform(matrix: anchor.transform)
            persistedMeshRootAnchor.addChild(entity)

            persistedMeshEntities[anchor.identifier] = entity
            persistedMeshOrder.append(anchor.identifier)

            if persistedMeshOrder.count > persistedMeshCap {
                let oldest = persistedMeshOrder.removeFirst()
                if let oldEntity = persistedMeshEntities.removeValue(forKey: oldest) {
                    oldEntity.removeFromParent()
                }
            }
        }
    }

    func handle(frame: ARFrame) {
        guard isRecording else { return }
        latestFrame = frame
        latestCameraTransform = frame.camera.transform

        // Tracking quality and feature point count (per-frame)
        currentTrackingQuality = trackingStateString(frame.camera.trackingState)
        currentFeaturePointCount = frame.rawFeaturePoints?.points.count ?? 0

        // Extract camera forward direction from transform
        let transform = frame.camera.transform
        let fwdX = -transform.columns.2.x
        let fwdZ = -transform.columns.2.z
        let fwdY = -transform.columns.2.y

        // Raw yaw: angle in XZ plane, normalized to [0, 2pi)
        var yaw = atan2(fwdX, fwdZ)
        if yaw < 0 { yaw += 2 * .pi }
        currentYaw = yaw

        // Pitch: angle from horizontal (negative = looking down)
        let horizontal = sqrt(fwdX * fwdX + fwdZ * fwdZ)
        currentPitch = atan2(fwdY, horizontal)

        // Smoothed yaw for UI (EMA with shortest-arc delta)
        let delta = normalizeAngle(yaw - smoothedYaw)
        smoothedYaw = smoothedYaw + smoothingAlpha * delta
        // Re-normalize smoothedYaw to [0, 2pi)
        if smoothedYaw < 0 { smoothedYaw += 2 * .pi }
        if smoothedYaw >= 2 * .pi { smoothedYaw -= 2 * .pi }

        // Smoothed pitch for UI (simple EMA, no wrapping needed)
        let pitchDelta = currentPitch - smoothedPitch
        smoothedPitch = smoothedPitch + smoothingAlpha * pitchDelta

        // Video mode: evaluate keyframe candidates at reduced rate
        if isVideoMode {
            if scanStartTime == 0 {
                scanStartTime = frame.timestamp
            }
            videoFrameCount += 1
            if videoFrameCount % candidateInterval == 0 {
                evaluateKeyframeCandidate(frame: frame)
            }
        }

        // Forward to phone-first pipeline (depth + geometry + keyframes)
        scanSession?.update(frame: frame)
    }

    /// Capture the current frame asynchronously — image processing happens off the main thread
    /// to prevent camera feed stutter.
    func captureAsync(
        motionOverlapScore: Float? = nil,
        sequenceBreakFlag: Bool = false,
        completion: @escaping (Result<CaptureData, CaptureError>) -> Void
    ) {
        guard let frame = latestFrame else {
            completion(.failure(.noFrame))
            return
        }

        let captureYaw = currentYaw
        let capturePitch = currentPitch
        let captureTransform = frame.camera.transform
        let cameraPosition = SIMD3<Float>(
            captureTransform.columns.3.x,
            captureTransform.columns.3.y,
            captureTransform.columns.3.z
        )
        let captureStepM: Float? = {
            guard let prev = self.lastCapturePosition else { return nil }
            return simd_distance(cameraPosition, prev)
        }()
        let captureYawDeltaDeg: Float? = {
            guard let prevYaw = self.lastCaptureYaw else { return nil }
            let delta = abs(normalizeAngle(captureYaw - prevYaw))
            return delta * (180.0 / Float.pi)
        }()
        let rawFeaturePoints = frame.rawFeaturePoints?.points ?? []
        let pixelBuffer = frame.capturedImage
        let intrinsics = frame.camera.intrinsics
        let res = frame.camera.imageResolution
        let pose = frame.camera.transform.toRowMajorArray()
        let trackingQuality = trackingStateString(frame.camera.trackingState)
        let metadataViewport = CGSize(width: Int(res.height), height: Int(res.width))
        let (featurePointsForMetadata, featurePointsUVForMetadata) = sampleFeaturePointsWithProjection(
            rawFeaturePoints,
            camera: frame.camera,
            viewportSize: metadataViewport,
            maxCount: 500
        )
        let featureSpread = computeFeatureImageSpread(
            uvPoints: featurePointsUVForMetadata,
            imageSize: metadataViewport
        )
        let planeAnchorsForMetadata = samplePlaneAnchorsForMetadata(
            around: captureTransform,
            maxCount: maxPlaneAnchorsPerShot
        )
        let shotId = "shot_\(shotCounter)"
        shotCounter += 1

        // Do heavy image processing on a background thread
        let ciCtx = self.ciContext
        let blurThreshold = self.blurVarianceThreshold
        let brightnessThreshold = self.minBrightness

        // Convert feature points to SIMD3<Float> for coverage engine
        let featurePointsSIMD = rawFeaturePoints.map { SIMD3<Float>($0.x, $0.y, $0.z) }

        Task.detached(priority: .userInitiated) {
            // Rotate image (ARKit sensor -> portrait)
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let rotated = ciImage.oriented(.right)
            guard let cgImage = ciCtx.createCGImage(rotated, from: rotated.extent) else {
                await MainActor.run { completion(.failure(.noFrame)) }
                return
            }

            // Quality checks (on background thread)
            let variance = Self.laplacianVarianceStatic(cgImage, context: ciCtx)
            if variance < blurThreshold {
                await MainActor.run { completion(.failure(.blurry)) }
                return
            }

            let brightness = Self.averageLuminanceStatic(cgImage)
            if brightness < brightnessThreshold {
                await MainActor.run { completion(.failure(.dark)) }
                return
            }
            let spatialQuality = Self.spatialQualityStatic(cgImage)

            // Build metadata on background thread
            let oldFx = intrinsics.columns.0.x
            let oldFy = intrinsics.columns.1.y
            let oldCx = intrinsics.columns.2.x
            let oldCy = intrinsics.columns.2.y
            let oldW = res.width
            let oldH = res.height

            let shot = GuidedShotPayload(
                shot_id: shotId,
                timestamp: frame.timestamp,
                intrinsics: CameraIntrinsicsPayload(
                    fx: oldFy,
                    fy: oldFx,
                    cx: oldCy,
                    cy: Float(oldW) - oldCx,
                    width: Int(oldH),
                    height: Int(oldW)
                ),
                pose: CameraPosePayload(matrix: pose),
                gravity_w: [0.0, -1.0, 0.0],
                tracking_quality: trackingQuality,
                quality: ShotQualityPayload(
                    texture_score: spatialQuality.textureScore,
                    blur_score: variance,
                    edge_density: spatialQuality.edgeDensity,
                    matchability_seed: spatialQuality.matchabilitySeed,
                    overexposure_ratio: spatialQuality.overexposureRatio,
                    motion_overlap_score: motionOverlapScore
                ),
                capture_step_m: captureStepM,
                capture_yaw_delta_deg: captureYawDeltaDeg,
                sequence_break_flag: sequenceBreakFlag,
                feature_points_w: featurePointsForMetadata,
                feature_points_uv: featurePointsUVForMetadata,
                plane_anchors: planeAnchorsForMetadata.isEmpty ? nil : planeAnchorsForMetadata,
                image_transform: ImageTransformPayload(
                    rotate_deg: 90,
                    flip_x: false,
                    flip_y: false,
                    crop: [0, 0, Int(oldH), Int(oldW)],
                    resize: [Int(oldH), Int(oldW)],
                    letterbox_pad: [0, 0]
                )
            )

            // Update state back on main thread
            await MainActor.run { [cgImage, shot] in
                self.frames.append(cgImage)
                self.shots.append(shot)
                self.capturedCount = self.frames.count
                self.lastCapturePosition = cameraPosition
                self.lastCaptureYaw = captureYaw
                completion(.success(CaptureData(
                    yaw: captureYaw,
                    pitch: capturePitch,
                    position: cameraPosition,
                    featurePoints: featurePointsSIMD,
                    cameraTransform: captureTransform,
                    featureImageCoverage: featureSpread.coverage,
                    featureGridCellCount: featureSpread.cells
                )))
            }
        }
    }

    // MARK: - Video Mode: Keyframe Extraction

    private func evaluateKeyframeCandidate(frame: ARFrame) {
        // Gate 1: Tracking quality
        switch frame.camera.trackingState {
        case .normal: break
        case .limited(.insufficientFeatures): break  // accept but mark
        default: return  // reject relocalizing, excessive motion, etc.
        }

        // Gate 2: Blur rejection (Laplacian variance on CVPixelBuffer)
        let blur = computeLaplacianVariance(frame.capturedImage, downscaleTo: 480)

        // Gate 3: Visual change (feature point count as proxy for scene change)
        let currentFPCount = frame.rawFeaturePoints?.points.count ?? 0

        // Gate 4: Geometric bounds
        let trans = translationDelta(frame.camera.transform, lastKeyframeTransform)
        let rot = rotationDelta(frame.camera.transform, lastKeyframeTransform)

        // Max keyframes: min(200, 3 × scan_duration)
        let scanDuration = frame.timestamp - scanStartTime
        let maxKeyframes = min(200, Int(3.0 * scanDuration))
        if capturedCount >= maxKeyframes { return }

        // Force capture if gap is dangerously large (prevents coverage holes)
        let forceCapture = (trans > 0.30 || rot > 20.0)

        // Reject if too blurry (unless forced — but even forced needs a hard floor
        // to avoid feeding extremely blurry frames to SfM)
        let blurHardFloor: Float = 30.0
        if blur < blurHardFloor { return }  // unconditional: never accept extremely blurry
        if blur < 120.0 && !forceCapture { return }

        // Primary: visual change — significant feature point change indicates new content
        // Guard: when feature count is very low (dark room / textureless area),
        // don't use FP count as signal — it's pure noise. Require geometric change instead.
        let fpChangeRatio: Float
        if lastKeyframeFeatureCount > 10 && currentFPCount > 10 {
            fpChangeRatio = abs(Float(currentFPCount - lastKeyframeFeatureCount))
                            / Float(lastKeyframeFeatureCount)
        } else if capturedCount == 0 {
            fpChangeRatio = 1.0  // first keyframe → always accept
        } else {
            fpChangeRatio = 0.0  // too few features to judge → require geometric change
        }

        // Secondary: geometric minimum
        let hasVisualChange = fpChangeRatio > 0.25
        let hasGeometricChange = trans > 0.08 || rot > 5.0

        if !hasVisualChange && !hasGeometricChange && !forceCapture {
            return  // not enough change
        }

        // Accept keyframe
        lastKeyframeTransform = frame.camera.transform
        lastKeyframeTime = frame.timestamp
        lastKeyframeFeatureCount = currentFPCount

        // Write JPEG to disk immediately (memory-safe)
        captureAndSaveToDisk(frame: frame, blurScore: blur) { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success(let captureData):
                self.onKeyframeCaptured?(captureData)
            case .failure:
                break  // silently skip failed keyframes
            }
        }
    }

    private func translationDelta(_ current: simd_float4x4, _ reference: simd_float4x4?) -> Float {
        guard let ref = reference else { return Float.greatestFiniteMagnitude }
        let dx = current.columns.3.x - ref.columns.3.x
        let dy = current.columns.3.y - ref.columns.3.y
        let dz = current.columns.3.z - ref.columns.3.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    private func rotationDelta(_ current: simd_float4x4, _ reference: simd_float4x4?) -> Float {
        guard let ref = reference else { return Float.greatestFiniteMagnitude }
        // Extract forward vectors and compute angle between them
        let fwd1 = SIMD3<Float>(-current.columns.2.x, -current.columns.2.y, -current.columns.2.z)
        let fwd2 = SIMD3<Float>(-ref.columns.2.x, -ref.columns.2.y, -ref.columns.2.z)
        let dot = simd_dot(simd_normalize(fwd1), simd_normalize(fwd2))
        let clamped = min(1.0, max(-1.0, dot))
        return acos(clamped) * (180.0 / Float.pi)  // degrees
    }

    /// Compute Laplacian variance on CVPixelBuffer using Accelerate framework.
    /// Returns a sharpness score — higher = sharper, lower = blurrier.
    nonisolated func computeLaplacianVariance(_ pixelBuffer: CVPixelBuffer, downscaleTo targetHeight: Int) -> Float {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        // Extract luminance plane (Y from YCbCr)
        let srcWidth: Int
        let srcHeight: Int
        let srcBytesPerRow: Int
        let srcBaseAddress: UnsafeMutableRawPointer

        if CVPixelBufferGetPlaneCount(pixelBuffer) > 0 {
            // Bi-planar YCbCr — plane 0 is luminance
            srcWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
            srcHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
            srcBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
            guard let addr = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return 0 }
            srcBaseAddress = addr
        } else {
            srcWidth = CVPixelBufferGetWidth(pixelBuffer)
            srcHeight = CVPixelBufferGetHeight(pixelBuffer)
            srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            guard let addr = CVPixelBufferGetBaseAddress(pixelBuffer) else { return 0 }
            srcBaseAddress = addr
        }

        var srcBuffer = vImage_Buffer(
            data: srcBaseAddress,
            height: vImagePixelCount(srcHeight),
            width: vImagePixelCount(srcWidth),
            rowBytes: srcBytesPerRow
        )

        // Downscale to target height (maintain aspect ratio)
        let scale = Float(targetHeight) / Float(srcHeight)
        let dstWidth = max(1, Int(Float(srcWidth) * scale))
        let dstHeight = targetHeight

        var dstData = [UInt8](repeating: 0, count: dstWidth * dstHeight)
        var dstBuffer = vImage_Buffer(
            data: &dstData,
            height: vImagePixelCount(dstHeight),
            width: vImagePixelCount(dstWidth),
            rowBytes: dstWidth
        )

        let scaleErr = vImageScale_Planar8(&srcBuffer, &dstBuffer, nil, vImage_Flags(kvImageNoFlags))
        guard scaleErr == kvImageNoError else { return 0 }

        // Convert to Float for convolution
        let pixelCount = dstWidth * dstHeight
        var floatData = [Float](repeating: 0, count: pixelCount)
        vDSP_vfltu8(dstData, 1, &floatData, 1, vDSP_Length(pixelCount))

        // Convolve with Laplacian kernel: [0,1,0; 1,-4,1; 0,1,0]
        var floatBuffer = vImage_Buffer(
            data: &floatData,
            height: vImagePixelCount(dstHeight),
            width: vImagePixelCount(dstWidth),
            rowBytes: dstWidth * MemoryLayout<Float>.size
        )

        var resultData = [Float](repeating: 0, count: pixelCount)
        var resultBuffer = vImage_Buffer(
            data: &resultData,
            height: vImagePixelCount(dstHeight),
            width: vImagePixelCount(dstWidth),
            rowBytes: dstWidth * MemoryLayout<Float>.size
        )

        var kernel: [Float] = [0, 1, 0, 1, -4, 1, 0, 1, 0]
        let convErr = vImageConvolve_PlanarF(
            &floatBuffer,
            &resultBuffer,
            nil,
            0, 0,
            &kernel,
            3, 3,
            0,
            vImage_Flags(kvImageEdgeExtend)
        )
        guard convErr == kvImageNoError else { return 0 }

        // Compute variance: mean(x²) - mean(x)²
        var meanSq: Float = 0
        vDSP_measqv(resultData, 1, &meanSq, vDSP_Length(pixelCount))
        var mean: Float = 0
        vDSP_meanv(resultData, 1, &mean, vDSP_Length(pixelCount))

        return meanSq - mean * mean
    }

    // MARK: - Exposure/Focus Lock

    /// Attempt to lock exposure/focus for consistent video frames.
    /// NOTE: When ARKit owns the camera session, AVCaptureDevice configuration
    /// may be silently ignored or conflict. This is best-effort — ARKit's built-in
    /// autoexposure is already fairly smooth. If this causes issues, disable by
    /// removing the lockExposureAndFocus() call in start().
    nonisolated func lockExposureAndFocus() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        do {
            try device.lockForConfiguration()
            // Use custom auto-exposure with bias instead of full lock,
            // so the camera can still adapt to lighting changes (e.g., walking
            // from dark corner to bright window). Full .locked would freeze
            // exposure at the initial value.
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
                // Slight bias toward current value for stability
                device.setExposureTargetBias(0.0, completionHandler: nil)
            }
            if device.isFocusModeSupported(.autoFocus) {
                device.focusMode = .autoFocus  // one-shot autofocus, then hold
            }
            device.unlockForConfiguration()
        } catch {
            // Expected to fail when ARKit owns the camera — not critical
        }
    }

    nonisolated func unlockExposureAndFocus() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        do {
            try device.lockForConfiguration()
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
            device.unlockForConfiguration()
        } catch {
            // Expected to fail when ARKit owns the camera — not critical
        }
    }

    // MARK: - Disk-Based Keyframe Capture

    func captureAndSaveToDisk(frame: ARFrame, blurScore: Float = 0, completion: @escaping (Result<CaptureData, CaptureError>) -> Void) {
        let captureYaw = currentYaw
        let capturePitch = currentPitch
        let captureTransform = frame.camera.transform
        let cameraPosition = SIMD3<Float>(
            captureTransform.columns.3.x,
            captureTransform.columns.3.y,
            captureTransform.columns.3.z
        )
        // Compute capture_step_m and capture_yaw_delta_deg from last capture
        let captureStepM: Float? = {
            guard let prev = self.lastCapturePosition else { return nil }
            return simd_distance(cameraPosition, prev)
        }()
        let captureYawDeltaDeg: Float? = {
            guard let prevYaw = self.lastCaptureYaw else { return nil }
            let delta = abs(normalizeAngle(captureYaw - prevYaw))
            return delta * (180.0 / Float.pi)
        }()
        let rawFeaturePoints = frame.rawFeaturePoints?.points ?? []
        let pixelBuffer = frame.capturedImage
        let intrinsics = frame.camera.intrinsics
        let res = frame.camera.imageResolution
        let pose = frame.camera.transform.toRowMajorArray()
        let trackingQuality = trackingStateString(frame.camera.trackingState)
        let metadataViewport = CGSize(width: Int(res.height), height: Int(res.width))
        let (featurePointsForMetadata, featurePointsUVForMetadata) = sampleFeaturePointsWithProjection(
            rawFeaturePoints,
            camera: frame.camera,
            viewportSize: metadataViewport,
            maxCount: 500
        )
        let featureSpread = computeFeatureImageSpread(
            uvPoints: featurePointsUVForMetadata,
            imageSize: metadataViewport
        )
        let planeAnchorsForMetadata = samplePlaneAnchorsForMetadata(
            around: captureTransform,
            maxCount: maxPlaneAnchorsPerShot
        )
        let shotId = "shot_\(shotCounter)"
        shotCounter += 1
        let keyframeIndex = keyframeJPEGPaths.count

        let featurePointsSIMD = rawFeaturePoints.map { SIMD3<Float>($0.x, $0.y, $0.z) }
        let ciCtx = self.ciContext
        let isVideoModeCapture = self.isVideoMode

        Task.detached(priority: .userInitiated) {
            // Rotate image (ARKit sensor -> portrait)
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let rotated = ciImage.oriented(.right)
            guard let cgImage = ciCtx.createCGImage(rotated, from: rotated.extent) else {
                await MainActor.run { completion(.failure(.noFrame)) }
                return
            }

            // Compress to JPEG and write to disk
            let uiImage = UIImage(cgImage: cgImage)
            guard let jpegData = uiImage.jpegData(compressionQuality: 0.65) else {
                await MainActor.run { completion(.failure(.noFrame)) }
                return
            }

            let tempDir = NSTemporaryDirectory()
            let fileName = "decroom_kf_\(keyframeIndex).jpg"
            let fileURL = URL(fileURLWithPath: tempDir).appendingPathComponent(fileName)
            do {
                try jpegData.write(to: fileURL)
            } catch {
                await MainActor.run { completion(.failure(.noFrame)) }
                return
            }

            // Build metadata
            let oldFx = intrinsics.columns.0.x
            let oldFy = intrinsics.columns.1.y
            let oldCx = intrinsics.columns.2.x
            let oldCy = intrinsics.columns.2.y
            let oldW = res.width
            let oldH = res.height

            let spatialQuality = Self.spatialQualityStatic(cgImage)

            let shot = GuidedShotPayload(
                shot_id: shotId,
                timestamp: frame.timestamp,
                intrinsics: CameraIntrinsicsPayload(
                    fx: oldFy,
                    fy: oldFx,
                    cx: oldCy,
                    cy: Float(oldW) - oldCx,
                    width: Int(oldH),
                    height: Int(oldW)
                ),
                pose: CameraPosePayload(matrix: pose),
                gravity_w: [0.0, -1.0, 0.0],
                tracking_quality: trackingQuality,
                quality: ShotQualityPayload(
                    texture_score: spatialQuality.textureScore,
                    blur_score: blurScore,
                    edge_density: spatialQuality.edgeDensity,
                    matchability_seed: spatialQuality.matchabilitySeed,
                    overexposure_ratio: spatialQuality.overexposureRatio,
                    motion_overlap_score: nil
                ),
                capture_step_m: captureStepM,
                capture_yaw_delta_deg: captureYawDeltaDeg,
                sequence_break_flag: false,
                feature_points_w: featurePointsForMetadata,
                feature_points_uv: featurePointsUVForMetadata,
                plane_anchors: planeAnchorsForMetadata.isEmpty ? nil : planeAnchorsForMetadata,
                image_transform: ImageTransformPayload(
                    rotate_deg: 90,
                    flip_x: false,
                    flip_y: false,
                    crop: [0, 0, Int(oldH), Int(oldW)],
                    resize: [Int(oldH), Int(oldW)],
                    letterbox_pad: [0, 0]
                )
            )

            await MainActor.run { [cgImage, shot] in
                // Store JPEG path for disk-based upload
                self.keyframeJPEGPaths.append(fileURL)
                // Also keep shot metadata for upload
                self.shots.append(shot)
                // Only keep CGImage in memory for photo mode; video mode reads from disk
                if !self.isVideoMode {
                    self.frames.append(cgImage)
                }
                self.capturedCount = self.isVideoMode ? self.keyframeJPEGPaths.count : self.frames.count
                self.lastCapturePosition = cameraPosition
                self.lastCaptureYaw = captureYaw

                completion(.success(CaptureData(
                    yaw: captureYaw,
                    pitch: capturePitch,
                    position: cameraPosition,
                    featurePoints: featurePointsSIMD,
                    cameraTransform: captureTransform,
                    featureImageCoverage: featureSpread.coverage,
                    featureGridCellCount: featureSpread.cells
                )))
            }
        }
    }

    // MARK: - Private Helpers

    private func computeFeatureImageSpread(
        uvPoints: [[Float]],
        imageSize: CGSize,
        gridCols: Int = 6,
        gridRows: Int = 4
    ) -> (coverage: Float, cells: Int) {
        guard !uvPoints.isEmpty else { return (0, 0) }
        guard imageSize.width > 1, imageSize.height > 1 else { return (0, 0) }
        guard gridCols > 0, gridRows > 0 else { return (0, 0) }

        let w = Float(imageSize.width)
        let h = Float(imageSize.height)
        var visited: Set<Int> = []
        visited.reserveCapacity(gridCols * gridRows)

        for uv in uvPoints {
            if uv.count < 2 { continue }
            let u = uv[0]
            let v = uv[1]
            if !u.isFinite || !v.isFinite { continue }
            if u < 0 || v < 0 || u >= w || v >= h { continue }

            let col = min(gridCols - 1, max(0, Int((u / w) * Float(gridCols))))
            let row = min(gridRows - 1, max(0, Int((v / h) * Float(gridRows))))
            visited.insert(row * gridCols + col)
        }

        let totalCells = gridCols * gridRows
        if totalCells <= 0 {
            return (0, visited.count)
        }
        let coverage = Float(visited.count) / Float(totalCells)
        return (coverage, visited.count)
    }

    private func buildMetadataJSON(shots: [GuidedShotPayload]) -> String? {
        let payload = GuidedCaptureMetadata(mode: "guided", shots: shots)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(payload) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func trackingStateString(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal:
            return "normal"
        case .notAvailable:
            return "not_available"
        case .limited(let reason):
            switch reason {
            case .excessiveMotion: return "limited_excessive_motion"
            case .insufficientFeatures: return "limited_insufficient_features"
            case .initializing: return "limited_initializing"
            case .relocalizing: return "limited_relocalizing"
            @unknown default: return "limited_unknown"
            }
        }
    }

    private func sampleFeaturePointsWithProjection(
        _ points: [simd_float3],
        camera: ARCamera,
        viewportSize: CGSize,
        maxCount: Int
    ) -> (world: [[Float]], uv: [[Float]]) {
        guard !points.isEmpty else { return ([], []) }
        let step = max(1, points.count / maxCount)
        var sampledWorld: [[Float]] = []
        var sampledUV: [[Float]] = []
        sampledWorld.reserveCapacity(min(maxCount, points.count))
        sampledUV.reserveCapacity(min(maxCount, points.count))

        for idx in stride(from: 0, to: points.count, by: step) {
            let p = points[idx]
            let projected = camera.projectPoint(
                p,
                orientation: .portrait,
                viewportSize: viewportSize
            )
            let u = Float(projected.x)
            let v = Float(projected.y)
            if !u.isFinite || !v.isFinite {
                continue
            }
            if u < 0 || v < 0 || u >= Float(viewportSize.width) || v >= Float(viewportSize.height) {
                continue
            }
            sampledWorld.append([p.x, p.y, p.z])
            sampledUV.append([u, v])
            if sampledWorld.count >= maxCount {
                break
            }
        }
        return (sampledWorld, sampledUV)
    }

    private func samplePlaneAnchorsForMetadata(
        around cameraTransform: simd_float4x4,
        maxCount: Int
    ) -> [PlaneAnchorPayload] {
        guard !livePlaneAnchors.isEmpty else { return [] }
        let cameraPos = SIMD3<Float>(
            cameraTransform.columns.3.x,
            cameraTransform.columns.3.y,
            cameraTransform.columns.3.z
        )
        let forward = normalize(SIMD3<Float>(
            -cameraTransform.columns.2.x,
            -cameraTransform.columns.2.y,
            -cameraTransform.columns.2.z
        ))

        let anchors = livePlaneAnchors.values
            .filter { anchor in
                let p = SIMD3<Float>(
                    anchor.transform.columns.3.x,
                    anchor.transform.columns.3.y,
                    anchor.transform.columns.3.z
                )
                let delta = p - cameraPos
                let dist = simd_length(delta)
                if !dist.isFinite || dist > planeCaptureRadius || dist < 0.05 {
                    return false
                }
                let dir = delta / dist
                return simd_dot(dir, forward) >= -0.2
            }
            .sorted { a, b in
                let pa = SIMD3<Float>(a.transform.columns.3.x, a.transform.columns.3.y, a.transform.columns.3.z)
                let pb = SIMD3<Float>(b.transform.columns.3.x, b.transform.columns.3.y, b.transform.columns.3.z)
                return simd_distance(pa, cameraPos) < simd_distance(pb, cameraPos)
            }
            .prefix(maxCount)

        return anchors.map { anchor in
            PlaneAnchorPayload(
                anchor_id: anchor.identifier.uuidString,
                alignment: planeAlignmentString(anchor.alignment),
                transform: anchor.transform.toRowMajorArray(),
                center: [anchor.center.x, anchor.center.y, anchor.center.z],
                extent: [anchor.extent.x, anchor.extent.y, anchor.extent.z]
            )
        }
    }

    private func planeAlignmentString(_ alignment: ARPlaneAnchor.Alignment) -> String {
        switch alignment {
        case .horizontal:
            return "horizontal"
        case .vertical:
            return "vertical"
        @unknown default:
            return "unknown"
        }
    }

    // MARK: - Static image quality checks (for background thread)
    // These are pure computation — nonisolated so Task.detached can call them.

    nonisolated static func laplacianVarianceStatic(_ image: CGImage, context: CIContext) -> Float {
        let targetSize = CGSize(width: 96, height: 96)
        guard let resized = resizeStatic(image: image, to: targetSize) else { return 0 }
        let ciImage = CIImage(cgImage: resized)
        let kernel: [CGFloat] = [0, 1, 0, 1, -4, 1, 0, 1, 0]
        let filter = CIFilter(
            name: "CIConvolution3X3",
            parameters: [
                kCIInputImageKey: ciImage,
                "inputWeights": CIVector(values: kernel, count: 9),
                "inputBias": 0.0
            ]
        )
        guard let output = filter?.outputImage else { return 0 }
        guard let outCG = context.createCGImage(output, from: output.extent) else { return 0 }
        guard let data = pixelDataStatic(from: outCG) else { return 0 }

        let count = data.count
        if count == 0 { return 0 }
        var sum: Float = 0
        for v in data { sum += Float(v) }
        let mean = sum / Float(count)
        var variance: Float = 0
        for v in data {
            let diff = Float(v) - mean
            variance += diff * diff
        }
        return variance / Float(count)
    }

    nonisolated static func averageLuminanceStatic(_ image: CGImage) -> Float {
        let targetSize = CGSize(width: 64, height: 64)
        guard let resized = resizeStatic(image: image, to: targetSize) else { return 0 }
        guard let data = pixelDataStatic(from: resized) else { return 0 }
        let count = data.count
        if count == 0 { return 0 }
        var sum: Float = 0
        for v in data { sum += Float(v) }
        return sum / Float(count) / 255.0
    }

    nonisolated static func spatialQualityStatic(
        _ image: CGImage
    ) -> (
        textureScore: Float,
        edgeDensity: Float,
        matchabilitySeed: Int,
        overexposureRatio: Float
    ) {
        let targetSize = CGSize(width: 128, height: 96)
        guard let resized = resizeStatic(image: image, to: targetSize) else {
            return (0, 0, 0, 0)
        }
        guard let data = pixelDataStatic(from: resized) else {
            return (0, 0, 0, 0)
        }

        let width = resized.width
        let height = resized.height
        if width < 4 || height < 4 || data.count < width * height {
            return (0, 0, 0, 0)
        }

        let n = width * height
        var sum: Float = 0
        var sumSq: Float = 0
        for p in data {
            let v = Float(p)
            sum += v
            sumSq += v * v
        }
        let mean = sum / Float(max(1, n))
        let variance = max(0.0, sumSq / Float(max(1, n)) - mean * mean)
        let std = sqrt(variance)
        let textureScore = min(1.0, max(0.0, std / 64.0))

        var edgeCount = 0
        let edgeThreshold: Float = 22.0
        let cellCols = 8
        let cellRows = 6
        var cellHit = [Bool](repeating: false, count: cellCols * cellRows)

        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let idx = y * width + x
                let gx = Float(data[idx + 1]) - Float(data[idx - 1])
                let gy = Float(data[idx + width]) - Float(data[idx - width])
                let g = abs(gx) + abs(gy)
                if g > edgeThreshold {
                    edgeCount += 1
                    let cx = min(cellCols - 1, max(0, (x * cellCols) / width))
                    let cy = min(cellRows - 1, max(0, (y * cellRows) / height))
                    cellHit[cy * cellCols + cx] = true
                }
            }
        }

        let denom = Float(max(1, (width - 2) * (height - 2)))
        let edgeDensity = min(1.0, max(0.0, Float(edgeCount) / denom))
        let activeCells = cellHit.reduce(0) { $0 + ($1 ? 1 : 0) }
        let matchabilitySeed = min(
            4096,
            activeCells * 24 + Int((0.6 * textureScore + 0.4 * edgeDensity) * 1200.0)
        )
        let overexposed = data.reduce(into: 0) { acc, p in
            if p >= 245 { acc += 1 }
        }
        let overexposureRatio = min(1.0, max(0.0, Float(overexposed) / Float(max(1, n))))

        return (textureScore, edgeDensity, matchabilitySeed, overexposureRatio)
    }

    nonisolated static func resizeStatic(image: CGImage, to size: CGSize) -> CGImage? {
        guard let colorSpace = image.colorSpace else { return nil }
        guard let context = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        context.interpolationQuality = .medium
        context.draw(image, in: CGRect(origin: .zero, size: size))
        return context.makeImage()
    }

    nonisolated static func pixelDataStatic(from image: CGImage) -> [UInt8]? {
        let width = image.width
        let height = image.height
        let bytesPerPixel = 1
        let bytesPerRow = bytesPerPixel * width
        var data = [UInt8](repeating: 0, count: height * bytesPerRow)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.genericGrayGamma2_2) else { return nil }
        guard let context = CGContext(
            data: &data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return data
    }

    static func meshResource(from geometry: ARMeshGeometry) -> MeshResource? {
        let vertices = readVertices(geometry.vertices)
        let triangleIndices = readTriangleIndices(geometry.faces)
        guard !vertices.isEmpty, !triangleIndices.isEmpty else { return nil }

        var descriptor = MeshDescriptor()
        descriptor.positions = MeshBuffers.Positions(vertices)
        descriptor.primitives = .triangles(triangleIndices)

        return try? MeshResource.generate(from: [descriptor])
    }

    nonisolated static func readVertices(_ source: ARGeometrySource) -> [SIMD3<Float>] {
        let count = source.count
        if count == 0 { return [] }

        let pointer = source.buffer.contents()
        var vertices: [SIMD3<Float>] = []
        vertices.reserveCapacity(count)

        for i in 0..<count {
            let start = source.offset + i * source.stride
            let p = pointer.advanced(by: start).assumingMemoryBound(to: Float.self)
            vertices.append(SIMD3<Float>(p[0], p[1], p[2]))
        }
        return vertices
    }

    nonisolated static func readTriangleIndices(_ element: ARGeometryElement) -> [UInt32] {
        guard element.indexCountPerPrimitive == 3 else { return [] }

        let primitiveCount = element.count
        if primitiveCount == 0 { return [] }

        // Keep only a sparse subset of triangles to make capture snapshots read as
        // lightweight wireframe hints instead of a filled surface overlay.
        let primitiveStride = 48

        let bytesPerIndex = element.bytesPerIndex
        let basePointer = element.buffer.contents()
        var triangles: [UInt32] = []
        triangles.reserveCapacity((primitiveCount / primitiveStride) * 3)

        for primitive in stride(from: 0, to: primitiveCount, by: primitiveStride) {
            let baseIndex = primitive * element.indexCountPerPrimitive
            var tri: [UInt32] = []
            tri.reserveCapacity(3)
            for offset in 0..<element.indexCountPerPrimitive {
                let idx = baseIndex + offset
                let address = basePointer.advanced(by: idx * bytesPerIndex)
                if bytesPerIndex == MemoryLayout<UInt16>.size {
                    let value = address.assumingMemoryBound(to: UInt16.self).pointee
                    tri.append(UInt32(value))
                } else if bytesPerIndex == MemoryLayout<UInt32>.size {
                    let value = address.assumingMemoryBound(to: UInt32.self).pointee
                    tri.append(value)
                } else {
                    return []
                }
            }
            if tri.count != 3 { continue }
            triangles.append(tri[0])
            triangles.append(tri[1])
            triangles.append(tri[2])
        }

        return triangles
    }
}

// MARK: - Metadata payloads (shared by guided capture)

private struct GuidedCaptureMetadata: Codable {
    let mode: String
    let shots: [GuidedShotPayload]
}

private struct GuidedShotPayload: Codable {
    let shot_id: String
    let timestamp: TimeInterval
    let intrinsics: CameraIntrinsicsPayload
    let pose: CameraPosePayload
    let gravity_w: [Float]
    let tracking_quality: String
    let quality: ShotQualityPayload?
    let capture_step_m: Float?
    let capture_yaw_delta_deg: Float?
    let sequence_break_flag: Bool?
    let feature_points_w: [[Float]]
    let feature_points_uv: [[Float]]?
    let plane_anchors: [PlaneAnchorPayload]?
    let image_transform: ImageTransformPayload?
}

private struct ShotQualityPayload: Codable {
    let texture_score: Float
    let blur_score: Float
    let edge_density: Float
    let matchability_seed: Int
    let overexposure_ratio: Float?
    let motion_overlap_score: Float?
}

private struct PlaneAnchorPayload: Codable {
    let anchor_id: String
    let alignment: String
    let transform: [Float]
    let center: [Float]
    let extent: [Float]
}

private struct ImageTransformPayload: Codable {
    let rotate_deg: Int
    let flip_x: Bool
    let flip_y: Bool
    let crop: [Int]
    let resize: [Int]
    let letterbox_pad: [Int]
}

private struct CameraIntrinsicsPayload: Codable {
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float
    let width: Int
    let height: Int
}

private struct CameraPosePayload: Codable {
    let matrix: [Float]
}

private extension simd_float4x4 {
    func toRowMajorArray() -> [Float] {
        let m = self
        return [
            m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x,
            m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y,
            m.columns.0.z, m.columns.1.z, m.columns.2.z, m.columns.3.z,
            m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w,
        ]
    }
}

#Preview {
    ContentView(processor: RecordingProcessor())
}
