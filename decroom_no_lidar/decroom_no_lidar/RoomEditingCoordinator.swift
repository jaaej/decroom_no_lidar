//
//  RoomEditingCoordinator.swift
//  decroom_no_lidar
//
//  Plan 4a — owns gesture recognizers for the editing UI and bridges
//  them to RoomEditModel.
//

import Foundation
import SceneKit
import UIKit

@MainActor
final class RoomEditingCoordinator: NSObject, UIGestureRecognizerDelegate {
    weak var scnView: SCNView?
    var editModel: RoomEditModel?

    private var tapRecognizer: UITapGestureRecognizer?
    private var panRecognizer: UIPanGestureRecognizer?
    private var rotateRecognizer: UIRotationGestureRecognizer?
    private var installedAction: EditAction?

    func attach(to scnView: SCNView, editModel: RoomEditModel) {
        self.scnView = scnView
        self.editModel = editModel

        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        tap.numberOfTapsRequired = 1
        tap.delegate = self
        tap.cancelsTouchesInView = false
        scnView.addGestureRecognizer(tap)
        tapRecognizer = tap

        // SceneKit's allowsCameraControl installs default gesture recognizers
        // lazily — including a double-tap-zoom that swallows our single-tap
        // selection and produces the "always zooms to the same object" bug.
        // Disable specifically the double-tap zoom; leave pan/pinch so the
        // user can still orbit and zoom the room.
        defeatSceneKitDoubleTapZoom(in: scnView)
    }

    private func defeatSceneKitDoubleTapZoom(in scnView: SCNView) {
        for gr in scnView.gestureRecognizers ?? [] {
            if let tap = gr as? UITapGestureRecognizer, tap.numberOfTapsRequired == 2 {
                tap.isEnabled = false
            }
        }
    }

    // MARK: - UIGestureRecognizerDelegate

    /// Allow our tap to coexist with SCNView's pan/pinch (so orbit + zoom
    /// still work) without making our tap wait on a phantom double-tap.
    nonisolated func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer
    ) -> Bool {
        return true
    }

    @objc func handleTap(_ gr: UITapGestureRecognizer) {
        guard let scnView = scnView, let model = editModel else { return }
        let point = gr.location(in: scnView)
        let id = SceneSelection.hitTest(scnView, at: point)
        model.select(id)
    }

    @objc func handlePan(_ gr: UIPanGestureRecognizer) {
        guard let scnView = scnView, let model = editModel else { return }
        guard model.activeAction == .move, let id = model.selectedObjectID else { return }
        let point = gr.location(in: scnView)
        guard let floor = SceneSelection.floorPoint(forScreen: point, in: scnView) else { return }

        let category = model.roomPayload.objects.first(where: { $0.id == id })?.type ?? ""
        let snapped = SceneSelection.wallSnap(
            position: floor, category: category,
            walls: model.roomPayload.room.walls
        )
        let clamped = SceneSelection.clampToFootprint(
            snapped, walls: model.roomPayload.room.walls
        )

        let began = (gr.state == .began)
        model.move(id, to: clamped, gestureBegan: began)
    }

    @objc func handleRotate(_ gr: UIRotationGestureRecognizer) {
        guard let model = editModel else { return }
        guard model.activeAction == .rotate, let id = model.selectedObjectID else { return }

        let snapped = round(gr.rotation / (.pi / 2)) * (.pi / 2)
        let current = model.roomPayload.objects.first(where: { $0.id == id })?.rotation_y ?? 0
        let newRotation = current - Double(snapped)   // negate: SceneKit CW vs backend CCW

        let began = (gr.state == .began)
        model.rotate(id, toRadians: Float(newRotation), gestureBegan: began)

        if gr.state == .ended || gr.state == .cancelled {
            gr.rotation = 0
        }
    }

    func setActiveAction(_ action: EditAction?) {
        guard let scnView = scnView else { return }

        scnView.allowsCameraControl = (action == nil)

        // SceneKit re-installs its default gestures whenever
        // allowsCameraControl flips, so re-defeat the double-tap zoom
        // every time we return to selection mode.
        if action == nil {
            defeatSceneKitDoubleTapZoom(in: scnView)
        }

        guard installedAction != action else { return }

        if let pr = panRecognizer {
            scnView.removeGestureRecognizer(pr); panRecognizer = nil
        }
        if let rr = rotateRecognizer {
            scnView.removeGestureRecognizer(rr); rotateRecognizer = nil
        }
        installedAction = action

        switch action {
        case .move:
            let pr = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            scnView.addGestureRecognizer(pr)
            panRecognizer = pr
        case .rotate:
            let rr = UIRotationGestureRecognizer(target: self, action: #selector(handleRotate(_:)))
            scnView.addGestureRecognizer(rr)
            rotateRecognizer = rr
        case .none:
            break
        }
    }
}
