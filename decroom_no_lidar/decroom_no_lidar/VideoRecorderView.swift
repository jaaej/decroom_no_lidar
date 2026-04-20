//
//  VideoRecorderView.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//
//  Video-based auto-capture with panoramic mosaic display.
//  As you rotate, captured frames fill in a cylindrical panorama strip.
//  Uncaptured areas remain dark — you can see exactly what's missing.
//

import SwiftUI
import ARKit

// MARK: - Panoramic Mosaic View

/// Renders keyframe thumbnails arranged by yaw angle as a cylindrical panorama.
/// The live camera feed shows through a transparent center window.
/// Dark areas = uncaptured directions.
struct PanoramaMosaicView: View {
    let thumbnails: [(image: CGImage, yaw: Float)]
    let currentYaw: Float
    let isScanning: Bool

    // Camera horizontal FOV in radians (~60° for iPhone wide lens)
    private let cameraFOV: Float = 1.05  // ~60°

    var body: some View {
        Canvas { context, size in
            let w = size.width
            let h = size.height

            // 1. Fill entire screen black (uncaptured areas)
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black.opacity(0.85)))

            // 2. Pixels per radian — how wide one radian of yaw maps to on screen
            let pixelsPerRadian = w / CGFloat(cameraFOV)

            // 3. Draw keyframe thumbnails at their angular positions
            let thumbWidth = w * 0.72   // slightly wider than center window for overlap
            let thumbHeight = h * 0.78
            let thumbY = h * 0.08

            for thumb in thumbnails {
                let relAngle = Self.normalizeAngle(thumb.yaw - currentYaw)

                // Skip thumbnails far behind the camera
                if abs(relAngle) > 2.8 { continue }

                let centerX = w / 2 + CGFloat(relAngle) * pixelsPerRadian

                let drawRect = CGRect(
                    x: centerX - thumbWidth / 2,
                    y: thumbY,
                    width: thumbWidth,
                    height: thumbHeight
                )

                // Only draw if at least partially on screen
                guard drawRect.maxX > 0 && drawRect.minX < w else { continue }

                // Draw the thumbnail with slight opacity fade at edges
                let edgeFade = min(1.0, Double(2.0 - abs(relAngle)))
                context.opacity = edgeFade
                context.draw(
                    Image(decorative: thumb.image, scale: 1),
                    in: drawRect
                )
                context.opacity = 1.0
            }

            // 4. Cut out center window for live camera feed
            // Rounded rectangle in the center — the ARView shows through here
            let centerW = w * 0.68
            let centerH = h * 0.72
            let centerRect = CGRect(
                x: (w - centerW) / 2,
                y: (h - centerH) / 2,
                width: centerW,
                height: centerH
            )

            // Clear the center to show AR camera behind
            context.blendMode = .clear
            let cornerRadius: CGFloat = 16
            context.fill(
                Path(roundedRect: centerRect, cornerRadius: cornerRadius),
                with: .color(.white)
            )
            context.blendMode = .normal

            // 5. Draw thin border around center window
            context.stroke(
                Path(roundedRect: centerRect, cornerRadius: cornerRadius),
                with: .color(.white.opacity(0.4)),
                lineWidth: 1.5
            )

            // 6. Draw subtle grid lines on dark areas to show "empty space"
            let gridSpacing: CGFloat = 40
            context.blendMode = .normal
            for x in stride(from: CGFloat(0), to: w, by: gridSpacing) {
                var line = Path()
                line.move(to: CGPoint(x: x, y: 0))
                line.addLine(to: CGPoint(x: x, y: h))
                context.stroke(line, with: .color(.white.opacity(0.03)), lineWidth: 0.5)
            }
            for y in stride(from: CGFloat(0), to: h, by: gridSpacing) {
                var line = Path()
                line.move(to: CGPoint(x: 0, y: y))
                line.addLine(to: CGPoint(x: w, y: y))
                context.stroke(line, with: .color(.white.opacity(0.03)), lineWidth: 0.5)
            }
        }
        .allowsHitTesting(false)
        .compositingGroup()
    }

    static func normalizeAngle(_ a: Float) -> Float {
        var v = a
        while v > .pi { v -= 2 * .pi }
        while v < -.pi { v += 2 * .pi }
        return v
    }
}

// MARK: - Coverage Arc

/// Blue arc at the bottom showing which yaw directions have been captured.
struct CoverageArcView: View {
    let filledBins: Set<Int>
    let totalBins: Int

    var body: some View {
        Canvas { context, size in
            let centerX = size.width / 2
            let y = size.height / 2
            let arcWidth = size.width * 0.7
            let segmentWidth = arcWidth / CGFloat(totalBins)
            let startX = centerX - arcWidth / 2

            for bin in 0..<totalBins {
                let x = startX + CGFloat(bin) * segmentWidth
                let rect = CGRect(x: x + 1, y: y - 3, width: segmentWidth - 2, height: 6)
                let filled = filledBins.contains(bin)
                let path = Path(roundedRect: rect, cornerRadius: 2)
                context.fill(path, with: .color(filled ? .cyan : .white.opacity(0.15)))
            }
        }
    }
}

// MARK: - Video Recorder View

struct VideoRecorderView: View {
    @StateObject private var recorder = GuidedARFrameRecorder()
    @StateObject private var coverage = CoverageEngine()
    @ObservedObject var processor: RecordingProcessor
    @State private var isScanning = false
    @State private var showSpeedWarning = false
    @State private var activeError: ScanError? = nil

    private var coveragePercent: Int {
        let filled = coverage.filledYawBins.count
        return coverage.yawBins > 0 ? Int(Float(filled) / Float(coverage.yawBins) * 100) : 0
    }

    var body: some View {
        ZStack {
            // LAYER 1: Live AR camera (shows through the center window)
            GuidedARViewContainer(recorder: recorder)
                .ignoresSafeArea()

            // LAYER 2: Panoramic mosaic overlay
            if isScanning || recorder.capturedCount > 0 {
                PanoramaMosaicView(
                    thumbnails: recorder.keyframeThumbnails,
                    currentYaw: recorder.currentYaw,
                    isScanning: isScanning
                )
                .ignoresSafeArea()
            }

            // LAYER 3: UI controls
            VStack(spacing: 0) {
                // Speed warning
                if showSpeedWarning {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                        Text("Slow down")
                    }
                    .font(.subheadline.bold())
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(.orange)
                    .clipShape(Capsule())
                    .padding(.top, 60)
                    .transition(.move(edge: .top).combined(with: .opacity))
                }

                Spacer()

                // Bottom controls
                VStack(spacing: 8) {
                    // Coverage arc
                    if isScanning || recorder.capturedCount > 0 {
                        CoverageArcView(
                            filledBins: coverage.filledYawBins,
                            totalBins: coverage.yawBins
                        )
                        .frame(height: 20)
                        .padding(.horizontal, 30)
                    }

                    // Stats line
                    if recorder.capturedCount > 0 {
                        HStack(spacing: 16) {
                            Label("\(recorder.capturedCount)", systemImage: "camera.fill")
                            Label("\(coveragePercent)%", systemImage: "circle.dotted")
                        }
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.7))
                    }

                    // Guidance
                    if isScanning {
                        Text(guidanceText)
                            .font(.callout)
                            .foregroundStyle(.white)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 24)
                    }

                    // Buttons
                    HStack(spacing: 16) {
                        if !isScanning && recorder.capturedCount == 0 {
                            Button {
                                startScan()
                            } label: {
                                HStack(spacing: 8) {
                                    Circle().fill(.red).frame(width: 14, height: 14)
                                    Text("Start Scan").fontWeight(.semibold)
                                }
                                .padding(.horizontal, 24)
                                .padding(.vertical, 14)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.white.opacity(0.15))

                        } else if isScanning {
                            Button {
                                stopScan()
                            } label: {
                                HStack(spacing: 6) {
                                    Image(systemName: "stop.fill").font(.caption)
                                    Text(coverage.isReadyToUpload ? "Done" : "Stop")
                                        .fontWeight(.semibold)
                                }
                                .padding(.horizontal, 24)
                                .padding(.vertical, 14)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(coverage.isReadyToUpload ? .green : .orange)

                        } else if recorder.capturedCount > 0 {
                            Button { upload() } label: {
                                HStack(spacing: 6) {
                                    Image(systemName: "arrow.up.circle.fill")
                                    Text("Upload").fontWeight(.semibold)
                                }
                                .padding(.horizontal, 24)
                                .padding(.vertical, 14)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.blue)

                            Button { startScan() } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "arrow.counterclockwise").font(.caption)
                                    Text("More")
                                }
                            }
                            .buttonStyle(.bordered)
                            .tint(.white)
                        }
                    }

                    // Status
                    if isScanning {
                        Text(statusText)
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.4))
                    } else if recorder.capturedCount > 0 && !coverage.isReadyToUpload {
                        Text(readinessHint)
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 20)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)
            }

            // Uploading overlay
            if case .uploading = processor.state {
                Color.black.opacity(0.6).ignoresSafeArea()
                VStack(spacing: 16) {
                    ProgressView().scaleEffect(1.5).tint(.white)
                    Text("Uploading \(recorder.capturedCount) frames...")
                        .font(.headline).foregroundStyle(.white)
                    Text("This may take a minute")
                        .font(.caption).foregroundStyle(.white.opacity(0.6))
                }
            }

            // Error overlay
            if case .failed(let message) = processor.state {
                Color.black.opacity(0.6).ignoresSafeArea()
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.largeTitle).foregroundStyle(.orange)
                    Text("Upload Failed").font(.headline).foregroundStyle(.white)
                    Text(message).font(.caption).foregroundStyle(.white.opacity(0.7))
                        .multilineTextAlignment(.center).padding(.horizontal, 40)
                    Button("Retry") { upload() }.buttonStyle(.borderedProminent)
                    Button("Back") { processor.reset() }.foregroundStyle(.white)
                }
            }
        }
        .onChange(of: processor.state) { _, newState in
            if case .failed(let msg) = newState {
                activeError = Self.errorForMessage(msg)
            }
        }
        .alert(
            isPresented: Binding(
                get: { activeError != nil },
                set: { if !$0 { activeError = nil } }
            )
        ) {
            guard let err = activeError else {
                return Alert(title: Text("Error"))
            }
            if err.isRetryable {
                let retryLabel: String
                switch err {
                case .noWallsDetected, .insufficientCoverage:
                    retryLabel = "Resume Scan"
                default:
                    retryLabel = "Retry"
                }
                return Alert(
                    title: Text("Scan Error"),
                    message: Text(err.message),
                    primaryButton: .default(Text(retryLabel)) {
                        activeError = nil
                        handleRetry(for: err)
                    },
                    secondaryButton: .cancel()
                )
            } else {
                return Alert(
                    title: Text("Scan Error"),
                    message: Text(err.message),
                    dismissButton: .default(Text("OK")) {
                        activeError = nil
                    }
                )
            }
        }
        .onAppear {
            coverage.captureMode = .video
            coverage.reset()
            recorder.isVideoMode = true
            recorder.onKeyframeCaptured = { [coverage] captureData in
                coverage.addCapture(
                    yaw: captureData.yaw,
                    pitch: captureData.pitch,
                    position: captureData.position,
                    featurePoints: captureData.featurePoints,
                    featureImageCoverage: captureData.featureImageCoverage,
                    featureGridCellCount: captureData.featureGridCellCount
                )
            }
        }
        .onDisappear {
            if isScanning { recorder.stop() }
        }
        .onChange(of: recorder.currentYaw) { _, _ in
            let pos = SIMD3<Float>(
                recorder.latestCameraTransform.columns.3.x,
                recorder.latestCameraTransform.columns.3.y,
                recorder.latestCameraTransform.columns.3.z
            )
            coverage.updateLiveState(position: pos, timestamp: CACurrentMediaTime())

            let isLimited = recorder.currentTrackingQuality.hasPrefix("limited")
            if isLimited != showSpeedWarning {
                withAnimation(.easeInOut(duration: 0.3)) {
                    showSpeedWarning = isLimited
                }
            }
        }
    }

    // MARK: - Text

    private var guidanceText: String {
        if recorder.capturedCount == 0 {
            return "Slowly rotate to scan the room"
        } else if coveragePercent < 40 {
            return "Keep rotating — fill in the dark areas"
        } else if coveragePercent < 70 {
            return "Good — keep filling the gaps"
        } else if coveragePercent < 90 {
            return "Almost done — turn toward the dark areas"
        } else {
            return "Room captured! Tap Done"
        }
    }

    private var statusText: String {
        recorder.currentTrackingQuality == "normal"
            ? "Scanning..."
            : "Tracking: \(recorder.currentTrackingQuality.replacingOccurrences(of: "_", with: " "))"
    }

    private var readinessHint: String {
        if coverage.capturedCount < coverage.minimumShots {
            return "Need \(coverage.minimumShots - coverage.capturedCount) more frames"
        }
        if coverage.uniqueYawBinCount < 6 {
            return "Rotate more to cover different directions"
        }
        return "Tap More to improve, or Upload to process"
    }

    // MARK: - Actions

    private func startScan() {
        coverage.captureMode = .video
        if recorder.capturedCount == 0 {
            coverage.reset()
        }
        recorder.start()
        isScanning = true
    }

    private func stopScan() {
        isScanning = false
    }

    /// Reverse-lookup a ScanError from its user-facing message.
    /// Used when processor.state = .failed(msg) needs an alert that matches
    /// the retryable-vs-not treatment. Fallback: backendServerError.
    static func errorForMessage(_ msg: String) -> ScanError {
        let table: [ScanError] = [
            .unsupportedDevice,
            .depthModelLoadFailed(detail: ""),
            .arKitTrackingLimited,
            .arKitSessionFailed(detail: ""),
            .noWallsDetected,
            .insufficientCoverage(binsFilled: 0, binsRequired: 12),
            .payloadBuildFailed(detail: ""),
            .uploadNoServer,
            .uploadTimeout,
            .backendServerError(status: 0, detail: ""),
            .noFurnitureDetected,
        ]
        return table.first(where: { $0.message == msg })
            ?? .backendServerError(status: 0, detail: msg)
    }

    private func handleRetry(for error: ScanError) {
        switch error {
        case .noWallsDetected, .insufficientCoverage:
            // Resume scan — keep recorder state, user keeps rotating.
            isScanning = true
            recorder.start()
        case .uploadNoServer, .uploadTimeout, .payloadBuildFailed, .backendServerError:
            upload()
        default:
            break  // non-retryable cases dismiss with OK
        }
    }

    private func upload() {
        guard recorder.capturedCount > 0 else { return }
        // Preserve disk-based keyframe paths (used inside finalizePhoneFirstPayload).
        _ = recorder.stopVideoMode()
        Task {
            do {
                let payload = try await recorder.finalizePhoneFirstPayload()
                print("[VideoRecorderView] Uploading phone-first payload scan=\(payload.scanID) keyframes=\(payload.keyframes.count)")
                processor.processPhoneFirstPayload(payload)
            } catch let e as ScanError {
                activeError = e
                processor.state = .failed(e.message)
            } catch {
                processor.state = .failed("Scan finalization failed: \(error.localizedDescription)")
            }
        }
    }
}

#Preview {
    VideoRecorderView(processor: RecordingProcessor())
}
