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
    @State private var scanStartTime: Date? = nil
    @State private var showNoWallsWarning = false
    @State private var lastUploadedPayload: ScanUploadPayload? = nil

    // Phase-0 debug recorder (spike-quality video + pose JSONL capture,
    // independent of the normal upload flow). Off by default; toggled via
    // a small, unobtrusive icon so it doesn't interfere with the main flow.
    @State private var debugRecorder: DebugScanRecorder? = nil
    @State private var isDebugRecording = false
    @State private var debugShareURL: URL? = nil
    @State private var showDebugShare = false

    /// Number of frames being uploaded right now. Prefers the in-flight
    /// payload's keyframe count (works for both fresh scans and
    /// "Use Last Capture"); falls back to the recorder's live count for
    /// the rare case the payload isn't set yet.
    private var uploadFrameCount: Int {
        lastUploadedPayload?.keyframes.count ?? recorder.capturedCount
    }

    private var coveragePercent: Int {
        let filled = coverage.filledYawBins.count
        return coverage.yawBins > 0 ? Int(Float(filled) / Float(coverage.yawBins) * 100) : 0
    }

    var body: some View {
        ZStack {
            // LAYER 1: Live AR camera — full-screen, no overlay
            GuidedARViewContainer(recorder: recorder)
                .ignoresSafeArea()

            // LAYER 1.5: Debug recorder toggle — small, unobtrusive, top-trailing.
            // Debug builds only: it writes raw video + poses to Documents.
            #if DEBUG
            VStack {
                HStack {
                    Spacer()
                    Button(action: toggleDebugRecording) {
                        Image(systemName: isDebugRecording ? "record.circle.fill" : "ladybug")
                            .font(.callout)
                            .foregroundStyle(isDebugRecording ? .red : .white.opacity(0.25))
                            .padding(10)
                    }
                }
                Spacer()
            }
            .padding(.top, 54)
            .padding(.trailing, 12)
            #endif

            // LAYER 2: UI controls
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

                // No-walls warning
                if showNoWallsWarning {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                        Text("No walls detected — look at wall edges, slow down")
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

                            if ScanUploadPayload.hasPersistedCapture {
                                Button {
                                    useLastCapture()
                                } label: {
                                    HStack(spacing: 6) {
                                        Image(systemName: "arrow.up.doc")
                                            .font(.caption)
                                        Text("Use Last Capture").fontWeight(.semibold)
                                    }
                                    .padding(.horizontal, 20)
                                    .padding(.vertical, 14)
                                }
                                .buttonStyle(.bordered)
                                .tint(.white)
                            }

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
                    Text("Uploading \(uploadFrameCount) frames...")
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
            // Persist the last successful upload for "Use Last Capture"
            // (QA convenience). Only saves on .finished — errors and idle
            // states don't overwrite.
            if case .finished = newState,
               let payload = lastUploadedPayload {
                payload.saveAsLastCapture()
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
            // Forwarding closure itself is (re)assigned in toggleDebugRecording,
            // weak-capturing the specific DebugScanRecorder instance so a
            // stop→start re-tap can't forward frames into the wrong recorder.
        }
        .onDisappear {
            if isScanning { recorder.stop() }
            if isDebugRecording {
                isDebugRecording = false
                recorder.onRawFrame = nil
                let rec = debugRecorder
                rec?.stop { _ in
                    DispatchQueue.main.async {
                        if debugRecorder === rec { debugRecorder = nil }
                    }
                }
            }
        }
        .sheet(isPresented: $showDebugShare) {
            if let url = debugShareURL {
                DebugScanShareSheet(activityItems: [url])
            }
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

            // Show "no walls" warning if ARKit hasn't committed any vertical
            // plane anchors after 15s of active scanning. Helps the user
            // catch a failing scan before tapping Done.
            if isScanning, let scanStart = scanStartTime {
                let elapsed = Date().timeIntervalSince(scanStart)
                let walls = recorder.scanSession?.geometry.walls.count ?? 0
                let shouldShow = elapsed > 15 && walls == 0
                if shouldShow != showNoWallsWarning {
                    withAnimation(.easeInOut(duration: 0.3)) {
                        showNoWallsWarning = shouldShow
                    }
                }
            } else if showNoWallsWarning {
                withAnimation(.easeInOut(duration: 0.3)) {
                    showNoWallsWarning = false
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
            recorder.start()
        } else {
            // User tapped "More" to add more frames to an existing scan —
            // preserve what's already been captured.
            recorder.resume()
        }
        isScanning = true
        scanStartTime = Date()
    }

    private func stopScan() {
        isScanning = false
        scanStartTime = nil
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
            .serverKeyMissing,
            .serverKeyMismatch,
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
            // Resume scan — keep previously captured frames, just continue.
            isScanning = true
            recorder.resume()
            scanStartTime = Date()
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
                lastUploadedPayload = payload
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

    private func useLastCapture() {
        guard let payload = ScanUploadPayload.loadLastCapture() else {
            processor.state = .failed("Couldn't load the previous capture.")
            return
        }
        lastUploadedPayload = payload
        print("[VideoRecorderView] Re-uploading last capture scan=\(payload.scanID) keyframes=\(payload.keyframes.count)")
        processor.processPhoneFirstPayload(payload)
    }

    // MARK: - Debug recorder (Phase-0 spike)

    /// Starts or stops the debug recorder. On stop, presents a share sheet
    /// for the recorded `scan_<timestamp>/` folder (video.mov + frames.jsonl
    /// + points.jsonl + stats.json) so it can be AirDropped off-device.
    private func toggleDebugRecording() {
        if isDebugRecording {
            isDebugRecording = false
            // Stop forwarding immediately so no more frames reach a recorder
            // that's about to finish writing.
            recorder.onRawFrame = nil
            let rec = debugRecorder
            rec?.stop { url in
                DispatchQueue.main.async {
                    // Identity-guard: a fast stop→start re-tap may already
                    // have replaced debugRecorder with a new, actively-
                    // recording instance by the time this completion runs.
                    if debugRecorder === rec {
                        debugRecorder = nil
                    }
                    if let url = url {
                        debugShareURL = url
                        showDebugShare = true
                    }
                }
            }
        } else {
            let rec = DebugScanRecorder()
            do {
                try rec.start()
                debugRecorder = rec
                isDebugRecording = true
                // Weak-capture this specific instance rather than reading
                // the mutable debugRecorder property, so the closure can't
                // be redirected to a later recorder instance.
                recorder.onRawFrame = { [weak rec] frame in
                    rec?.append(frame)
                }
            } catch {
                print("[VideoRecorderView] DebugScanRecorder start failed: \(error)")
            }
        }
    }
}

/// Thin wrapper presenting a UIActivityViewController (AirDrop/Files/etc.)
/// for the debug recorder's output folder.
private struct DebugScanShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

#Preview {
    VideoRecorderView(processor: RecordingProcessor())
}
