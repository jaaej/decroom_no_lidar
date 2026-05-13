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
        guard roomPayload.objects.contains(where: { $0.id == id }) else { return }
        previousObjects = roomPayload.objects
        canUndo = true
        let newObjects = roomPayload.objects.filter { $0.id != id }
        roomPayload = roomPayload.with(objects: newObjects)
        if selectedObjectID == id {
            selectedObjectID = nil
        }
    }

    func move(_ id: String, to xz: SIMD2<Float>, gestureBegan: Bool) {
        guard let idx = roomPayload.objects.firstIndex(where: { $0.id == id }) else { return }
        if gestureBegan {
            previousObjects = roomPayload.objects
            canUndo = true
        }
        let old = roomPayload.objects[idx]
        let newPos = ObjectPosition(x: Double(xz.x), y: old.position.y, z: Double(xz.y))
        var newObjects = roomPayload.objects
        newObjects[idx] = old.with(position: newPos)
        roomPayload = roomPayload.with(objects: newObjects)
    }

    func rotate(_ id: String, toRadians y: Float, gestureBegan: Bool) {
        guard let idx = roomPayload.objects.firstIndex(where: { $0.id == id }) else { return }
        if gestureBegan {
            previousObjects = roomPayload.objects
            canUndo = true
        }
        let old = roomPayload.objects[idx]
        var newObjects = roomPayload.objects
        newObjects[idx] = old.with(rotationY: Double(y))
        roomPayload = roomPayload.with(objects: newObjects)
    }

    func undo() {
        guard let prev = previousObjects else { return }
        roomPayload = roomPayload.with(objects: prev)
        previousObjects = nil
        canUndo = false
        selectedObjectID = nil
    }
}
