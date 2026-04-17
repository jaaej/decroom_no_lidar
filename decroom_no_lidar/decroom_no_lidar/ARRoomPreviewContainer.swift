//
//  ARRoomPreviewContainer.swift
//  decroom_no_lidar
//
//  UIViewRepresentable bridging SwiftUI to an ARView. Renders the
//  guidance reticle in AR space and feeds ARFrame updates into ScanSession.
//
//  Re-entrancy guard: drops frames if previous update is still in-flight
//  to prevent unbounded Task queue growth at 30fps.
//

import SwiftUI
import ARKit
import RealityKit

struct ARRoomPreviewContainer: UIViewRepresentable {
    @ObservedObject var session: ScanSession

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        config.isAutoFocusEnabled = true
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }
        arView.session.run(config)
        arView.session.delegate = context.coordinator

        // Reticle (world-anchored).
        let reticleAnchor = AnchorEntity(world: .zero)
        reticleAnchor.addChild(context.coordinator.reticle.entity)
        arView.scene.addAnchor(reticleAnchor)

        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(session: session)
    }

    final class Coordinator: NSObject, ARSessionDelegate {
        let session: ScanSession
        let reticle = CaptureReticle()
        private var isProcessing = false  // re-entrancy guard

        init(session: ScanSession) {
            self.session = session
            super.init()
        }

        func session(_ arSession: ARSession, didUpdate frame: ARFrame) {
            // Drop frame if previous update hasn't finished — prevents
            // unbounded Task queue growth when ML inference is slow.
            guard !isProcessing else { return }
            isProcessing = true

            Task { @MainActor [weak self] in
                guard let self else { return }
                defer { self.isProcessing = false }

                self.session.update(frame: frame)

                // Move reticle toward the nearest unvisited yaw bin.
                let vm = ARRoomPreviewViewModel(
                    cameraYaw: self.cameraYaw(from: frame.camera.transform),
                    wallsDetected: self.session.geometry.walls.count,
                    coverageBinsFilled: self.session.visitedYawBins,
                    totalCoverageBins: 24
                )
                let camPos = SIMD3<Float>(
                    frame.camera.transform.columns.3.x,
                    frame.camera.transform.columns.3.y,
                    frame.camera.transform.columns.3.z
                )
                self.reticle.update(
                    direction: vm.reticleDirection,
                    cameraOrigin: camPos,
                    distance: 1.5
                )
            }
        }

        func session(_ arSession: ARSession, didAdd anchors: [ARAnchor]) {
            let planes = anchors.compactMap { $0 as? ARPlaneAnchor }
            guard !planes.isEmpty else { return }
            Task { @MainActor in
                self.session.updatePlaneAnchors(planes)
            }
        }

        func session(_ arSession: ARSession, didUpdate anchors: [ARAnchor]) {
            let planes = anchors.compactMap { $0 as? ARPlaneAnchor }
            guard !planes.isEmpty else { return }
            Task { @MainActor in
                self.session.updatePlaneAnchors(planes)
            }
        }

        private func cameraYaw(from transform: simd_float4x4) -> Float {
            let fx = -transform.columns.2.x
            let fz = -transform.columns.2.z
            return atan2(fx, fz)
        }
    }
}
