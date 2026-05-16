//
//  RoomEditMenuView.swift
//  decroom_no_lidar
//
//  Plan 4a v2 (direct manipulation) — overlay shows:
//   • Top-trailing Undo button when editModel.canUndo
//   • Top-leading Delete pill when editModel.selectedObjectID != nil
//
//  Move and Rotate happen via direct gestures (drag furniture, two-finger
//  twist a selected one); no on-screen mode toggle is needed.
//

import SwiftUI

struct RoomEditMenuView: View {
    @ObservedObject var editModel: RoomEditModel

    var body: some View {
        HStack {
            if editModel.selectedObjectID != nil {
                deleteButton
            }
            Spacer()
            if editModel.canUndo {
                undoButton
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 12)
        .animation(.easeInOut(duration: 0.15), value: editModel.selectedObjectID)
        .animation(.easeInOut(duration: 0.15), value: editModel.canUndo)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .allowsHitTesting(true)
    }

    private var deleteButton: some View {
        Button {
            if let id = editModel.selectedObjectID {
                editModel.delete(id)
            }
        } label: {
            Label("Delete", systemImage: "trash")
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())
        }
        .accessibilityIdentifier("editAction.Delete")
        .tint(.red)
    }

    private var undoButton: some View {
        Button {
            editModel.undo()
        } label: {
            Label("Undo", systemImage: "arrow.uturn.backward")
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())
        }
        .accessibilityIdentifier("editAction.Undo")
    }
}
