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
final class RoomEditingCoordinator: NSObject {
    weak var scnView: SCNView?
    var editModel: RoomEditModel?

    private var panRecognizer: UIPanGestureRecognizer?
    private var rotateRecognizer: UIRotationGestureRecognizer?

    func attach(to scnView: SCNView, editModel: RoomEditModel) {
        self.scnView = scnView
        self.editModel = editModel
        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        scnView.addGestureRecognizer(tap)
    }

    @objc func handleTap(_ gr: UITapGestureRecognizer) {
        guard let scnView = scnView, let model = editModel else { return }
        let point = gr.location(in: scnView)
        let id = SceneSelection.hitTest(scnView, at: point)
        model.select(id)
    }

    @objc func handlePan(_ gr: UIPanGestureRecognizer) {
        // Task 12
    }

    @objc func handleRotate(_ gr: UIRotationGestureRecognizer) {
        // Task 12
    }

    func setActiveAction(_ action: EditAction?) {
        // Task 12
    }
}
