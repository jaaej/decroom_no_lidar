//
//  RoomEditMenuView.swift
//  decroom_no_lidar
//
//  Plan 4a v3 — bottom edit sheet that shows:
//   • [Delete] (visible when something is selected)
//   • [Undo] (visible when canUndo)
//
//  Sits above CaptureFlowView's existing "Save Room / Rescan" bottom area
//  so it doesn't get hidden. Move + Rotate are direct-manipulation
//  gestures (drag furniture, two-finger twist) — no buttons needed.
//

import SwiftUI

struct RoomEditMenuView: View {
    @ObservedObject var editModel: RoomEditModel

    var body: some View {
        VStack {
            Spacer()
            if shouldShowSheet {
                actionSheet
                    .padding(.horizontal, 16)
                    .padding(.bottom, 180)   // clears the Save Room overlay
            }
        }
        .animation(.easeInOut(duration: 0.15), value: editModel.selectedObjectID)
        .animation(.easeInOut(duration: 0.15), value: editModel.canUndo)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        .allowsHitTesting(true)
    }

    private var shouldShowSheet: Bool {
        editModel.selectedObjectID != nil || editModel.canUndo
    }

    private var actionSheet: some View {
        HStack(spacing: 16) {
            if editModel.selectedObjectID != nil {
                deleteButton
            }
            if editModel.selectedObjectID != nil && editModel.canUndo {
                Spacer()
            }
            if editModel.canUndo {
                undoButton
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var deleteButton: some View {
        Button {
            if let id = editModel.selectedObjectID {
                editModel.delete(id)
            }
        } label: {
            Label("Delete", systemImage: "trash")
                .font(.body.weight(.medium))
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .frame(minHeight: 44)
        }
        .accessibilityIdentifier("editAction.Delete")
        .tint(.red)
    }

    private var undoButton: some View {
        Button {
            editModel.undo()
        } label: {
            Label("Undo", systemImage: "arrow.uturn.backward")
                .font(.body.weight(.medium))
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .frame(minHeight: 44)
        }
        .accessibilityIdentifier("editAction.Undo")
    }
}
