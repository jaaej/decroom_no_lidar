//
//  ARRoomPreviewContainer.swift
//  decroom_no_lidar
//
//  UIViewRepresentable bridging SwiftUI to an ARView. Renders only the
//  guidance reticle in AR space — the wireframe grid shell was removed
//  in favor of letting ARKit's plane visualization show captured walls
//  directly. The reticle alone provides "look here next" guidance.
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
        context.coordinator.reticleAnchor = reticleAnchor

        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        // State changes pushed via Coordinator.
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(session: session)
    }

    final class Coordinator: NSObject, ARSessionDelegate {
        let session: ScanSession
        let reticle = CaptureReticle()
        var reticleAnchor: AnchorEntity?

        init(session: ScanSession) {
            self.session = session
            super.init()
        }

        func session(_ arSession: ARSession, didUpdate frame: ARFrame) {
            Task { @MainActor in
                self.session.update(frame: frame)
                // Move reticle per latest viewmodel direction
                // Coverage bins left empty in Plan 2; a future Plan 3 task
                // can wire the existing CoverageEngine's filledYawBins here.
                // With empty bins, the reticle defaults to the first unfilled
                // bin (bin 0 = world +Z), which biases toward "look forward".
                let vm = ARRoomPreviewViewModel(
                    cameraYaw: self.cameraYaw(from: frame.camera.transform),
                    wallsDetected: self.session.geometry.walls.count,
                    coverageBinsFilled: [],
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
            Task { @MainActor in
                self.session.updatePlaneAnchors(planes)
            }
        }

        func session(_ arSession: ARSession, didUpdate anchors: [ARAnchor]) {
            let planes = anchors.compactMap { $0 as? ARPlaneAnchor }
            Task { @MainActor in
                self.session.updatePlaneAnchors(planes)
            }
        }

        private func cameraYaw(from transform: simd_float4x4) -> Float {
            // forward = -Z column
            let fx = -transform.columns.2.x
            let fz = -transform.columns.2.z
            return atan2(fx, fz)
        }
    }
}
