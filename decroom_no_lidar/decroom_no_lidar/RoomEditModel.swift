//
//  RoomEditModel.swift
//  decroom_no_lidar
//
//  Plan 4a — selection state, mutations, single-step undo for the
//  correction UI. Pure Swift, no SceneKit imports.
//

import Foundation
import simd

enum EditAction: Equatable {
    case move
    case rotate
}

@MainActor
final class RoomEditModel: ObservableObject {
    @Published private(set) var roomPayload: RoomPayload
    @Published var activeAction: EditAction?
    @Published private(set) var canUndo: Bool = false

    @Published var selectedObjectID: String? {
        didSet {
            // Validate against current objects; setting an invalid ID clears.
            if let id = selectedObjectID,
               !roomPayload.objects.contains(where: { $0.id == id }) {
                selectedObjectID = nil
                return
            }
            // Changing selection cancels any in-progress action.
            if oldValue != selectedObjectID {
                activeAction = nil
            }
        }
    }

    private var previousObjects: [RoomObject]?

    init(initial: RoomPayload) {
        self.roomPayload = initial
    }

    // MARK: - Selection

    func select(_ id: String?) {
        selectedObjectID = id
    }

    // MARK: - Mutations (stubs — implemented in later tasks)

    func delete(_ id: String) {
        // Task 3
    }

    func move(_ id: String, to xz: SIMD2<Float>, gestureBegan: Bool) {
        // Task 4
    }

    func rotate(_ id: String, toRadians y: Float, gestureBegan: Bool) {
        // Task 4
    }

    func undo() {
        // Task 5
    }
}
