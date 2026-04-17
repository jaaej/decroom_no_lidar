//
//  ARRoomPreviewContainer.swift
//  decroom_no_lidar
//
//  UIViewRepresentable bridging SwiftUI to an ARView for the live camera
//  feed. Feeds ARFrame updates into ScanSession. No AR entities rendered —
//  the panorama overlay handles all visual feedback in 2D.
//
//  Re-entrancy guard drops frames if previous update is still in-flight.
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
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(session: session)
    }

    final class Coordinator: NSObject, ARSessionDelegate {
        let session: ScanSession
        private var isProcessing = false

        init(session: ScanSession) {
            self.session = session
            super.init()
        }

        func session(_ arSession: ARSession, didUpdate frame: ARFrame) {
            guard !isProcessing else { return }
            isProcessing = true

            Task { @MainActor [weak self] in
                guard let self else { return }
                defer { self.isProcessing = false }
                self.session.update(frame: frame)
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
    }
}
