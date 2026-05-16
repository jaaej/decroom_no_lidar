//
//  RoomEditingCoordinator.swift
//  decroom_no_lidar
//
//  Plan 4a (v2 — direct manipulation): owns gesture recognizers for the
//  editing UI and bridges them to RoomEditModel.
//
//  Interaction model:
//   • Touch + drag on furniture → move that furniture along the floor
//   • Touch + drag on empty space → orbit camera (SceneKit default)
//   • Pinch → zoom (SceneKit default)
//   • Tap furniture → selection highlight + Delete pill appears
//   • Tap empty space → deselect
//   • Two-finger twist (when something is selected) → rotate by 90° snap
//

import Foundation
import SceneKit
import UIKit

/// Pan recognizer that hit-tests on the first touch and yields (state =
/// .failed) when the finger isn't on a furniture node — letting SCNView's
/// default camera-control pan take over.
final class FurnitureDragRecognizer: UIPanGestureRecognizer {
    var hitTester: ((CGPoint) -> String?)?
    /// The RoomObject.id this drag is claiming, set on touchesBegan if hit.
    var draggingObjectID: String?
    /// Snapshot of the object's Y at drag start — used in vertical mode
    /// so the height change is computed from a stable reference.
    var beginObjectY: Double = 0
    /// Snapshot of the finger's Y in view coordinates at drag start.
    var beginFingerY: CGFloat = 0

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent) {
        super.touchesBegan(touches, with: event)
        guard let touch = touches.first, let v = view else { return }
        let point = touch.location(in: v)
        if let id = hitTester?(point) {
            draggingObjectID = id
            beginFingerY = point.y
        } else {
            // No furniture under finger — fail so SCNView's pan handles it.
            draggingObjectID = nil
            state = .failed
        }
    }

    override func reset() {
        super.reset()
        draggingObjectID = nil
    }
}

@MainActor
final class RoomEditingCoordinator: NSObject, UIGestureRecognizerDelegate {
    weak var scnView: SCNView?
    var editModel: RoomEditModel?

    private var tapRecognizer: UITapGestureRecognizer?
    private var dragRecognizer: FurnitureDragRecognizer?
    private var rotateRecognizer: UIRotationGestureRecognizer?

    func attach(to scnView: SCNView, editModel: RoomEditModel) {
        self.scnView = scnView
        self.editModel = editModel

        // Tap: select / deselect.
        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        tap.numberOfTapsRequired = 1
        tap.delegate = self
        tap.cancelsTouchesInView = false
        scnView.addGestureRecognizer(tap)
        tapRecognizer = tap

        // Drag: move furniture if the finger landed on one; otherwise yield
        // to SceneKit's default camera-control pan.
        let drag = FurnitureDragRecognizer(target: self, action: #selector(handleDrag(_:)))
        drag.delegate = self
        drag.maximumNumberOfTouches = 1
        drag.cancelsTouchesInView = false
        drag.hitTester = { [weak self] point in
            guard let scnView = self?.scnView else { return nil }
            return SceneSelection.hitTest(scnView, at: point)
        }
        scnView.addGestureRecognizer(drag)
        dragRecognizer = drag

        // Rotation: only acts when something is selected.
        let rot = UIRotationGestureRecognizer(target: self, action: #selector(handleRotate(_:)))
        rot.delegate = self
        rot.cancelsTouchesInView = false
        scnView.addGestureRecognizer(rot)
        rotateRecognizer = rot

        // SceneKit's allowsCameraControl installs a default double-tap-zoom
        // recognizer that swallows our single-tap selection. Disable it
        // specifically while leaving pan/pinch (orbit + zoom) intact.
        defeatSceneKitDoubleTapZoom(in: scnView)
    }

    private func defeatSceneKitDoubleTapZoom(in scnView: SCNView) {
        for gr in scnView.gestureRecognizers ?? [] {
            // Disable every tap recognizer that isn't ours (SceneKit
            // installs a double-tap-zoom; some iOS versions also install
            // a single-tap "focus" that swallows our selection tap).
            if let t = gr as? UITapGestureRecognizer, t !== tapRecognizer {
                t.isEnabled = false
            }
        }
    }

    /// Called from RoomSceneView.updateUIView every time the model
    /// publishes — re-runs the defeat in case SceneKit lazily installed
    /// new recognizers after our initial attach.
    func ensureSceneKitTapsDefeated() {
        guard let scnView = scnView else { return }
        defeatSceneKitDoubleTapZoom(in: scnView)
    }

    // MARK: - UIGestureRecognizerDelegate

    nonisolated func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer
    ) -> Bool {
        // Our furniture drag must NOT run simultaneously with SceneKit's
        // camera pan — otherwise the camera orbits while the user is
        // moving an object. Same for our rotation vs SceneKit's pan:
        // a two-finger twist must not also strafe the camera.
        if gestureRecognizer is FurnitureDragRecognizer
            && other is UIPanGestureRecognizer {
            return false
        }
        if gestureRecognizer is UIRotationGestureRecognizer
            && other is UIPanGestureRecognizer {
            return false
        }
        // Tap can coexist with everything (so it doesn't wait for
        // camera-pan / pinch to fail).
        return true
    }

    /// Make SceneKit's camera-pan wait for both our FurnitureDragRecognizer
    /// AND our UIRotationGestureRecognizer to fail first.
    ///   - Drag fails immediately on empty-space touches (touchesBegan
    ///     override) → camera-pan fires for orbit.
    ///   - Drag succeeds on furniture touches → camera-pan never fires.
    ///   - Rotation succeeds on two-finger twist with a selected object
    ///     → camera-pan never fires.
    ///   - Rotation fails (no selection / no twist) → camera-pan fires.
    nonisolated func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldBeRequiredToFailBy other: UIGestureRecognizer
    ) -> Bool {
        if gestureRecognizer is UIPanGestureRecognizer {
            if other is FurnitureDragRecognizer { return true }
            if other is UIRotationGestureRecognizer { return true }
        }
        return false
    }

    // MARK: - Tap

    @objc func handleTap(_ gr: UITapGestureRecognizer) {
        guard let scnView = scnView, let model = editModel else { return }
        let point = gr.location(in: scnView)
        let id = SceneSelection.hitTest(scnView, at: point)
        print("[Decroom] tap at \(point) -> hit \(id ?? "nil")")
        model.select(id)   // id == nil → deselect
    }

    // MARK: - Drag (direct-manipulation move)

    @objc func handleDrag(_ gr: UIPanGestureRecognizer) {
        guard let scnView = scnView, let model = editModel else { return }
        guard let drag = gr as? FurnitureDragRecognizer,
              let id = drag.draggingObjectID else { return }

        // Mark the dragged object as selected (so the highlight follows the move).
        if model.selectedObjectID != id {
            model.select(id)
        }

        let began = (gr.state == .began)

        if model.verticalDragMode {
            // Vertical (Y-axis) drag — finger up = object up, down = down.
            // 1 screen point ≈ 5 mm of world-space height. Scale is empirical
            // — feels natural at typical viewing distance.
            let point = gr.location(in: scnView)
            if began,
               let obj = model.roomPayload.objects.first(where: { $0.id == id }) {
                drag.beginObjectY = obj.position.y
            }
            let deltaPixels = drag.beginFingerY - point.y   // up = positive
            let deltaY = Double(deltaPixels) * 0.005
            let newY = drag.beginObjectY + deltaY
            model.moveY(id, to: Float(newY), gestureBegan: began)
        } else {
            // Floor (XZ) drag — raycast finger to floor plane, snap, clamp.
            let point = gr.location(in: scnView)
            guard let floor = SceneSelection.floorPoint(forScreen: point, in: scnView) else { return }
            let obj = model.roomPayload.objects.first(where: { $0.id == id })
            let category = obj?.type ?? ""
            let metadata = obj?.metadata
            let snapped = SceneSelection.wallSnap(
                position: floor, category: category,
                walls: model.roomPayload.room.walls,
                objectMetadata: metadata
            )
            let clamped = SceneSelection.clampToFootprint(
                snapped, walls: model.roomPayload.room.walls,
                objectMetadata: metadata
            )
            model.move(id, to: clamped, gestureBegan: began)
        }
    }

    // MARK: - Rotate (two-finger twist on a selected object)

    @objc func handleRotate(_ gr: UIRotationGestureRecognizer) {
        guard let model = editModel, let id = model.selectedObjectID else { return }

        let snapped = round(gr.rotation / (.pi / 2)) * (.pi / 2)
        let current = model.roomPayload.objects.first(where: { $0.id == id })?.rotation_y ?? 0
        let newRotation = current - Double(snapped)   // SceneKit CW vs backend CCW

        let began = (gr.state == .began)
        model.rotate(id, toRadians: Float(newRotation), gestureBegan: began)

        if gr.state == .ended || gr.state == .cancelled {
            gr.rotation = 0
        }
    }
}
