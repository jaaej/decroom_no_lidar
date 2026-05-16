//
//  RoomEditMenuView.swift
//  decroom_no_lidar
//
//  Plan 4a v5 — vertical side bar, ALWAYS visible while in edit mode.
//  Diagnostic version: a bright bar so we can verify the menu is being
//  added to the view hierarchy at all. Buttons disable when their action
//  isn't available, but the bar itself stays on-screen.
//

import SwiftUI

struct RoomEditMenuView: View {
    @ObservedObject var editModel: RoomEditModel

    var body: some View {
        HStack(spacing: 0) {
            Spacer()
            VStack(spacing: 14) {
                deleteButton
                liftButton
                undoButton
            }
            .padding(.vertical, 14)
            .padding(.horizontal, 8)
            .background(
                Capsule().fill(Color.black.opacity(0.55))
            )
            .overlay(
                Capsule().stroke(Color.white.opacity(0.4), lineWidth: 1)
            )
            .padding(.trailing, 14)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .trailing)
        .animation(.easeInOut(duration: 0.15), value: editModel.selectedObjectID)
        .animation(.easeInOut(duration: 0.15), value: editModel.canUndo)
        .animation(.easeInOut(duration: 0.15), value: editModel.verticalDragMode)
        .allowsHitTesting(true)
    }

    private var deleteButton: some View {
        Button {
            if let id = editModel.selectedObjectID {
                editModel.delete(id)
            }
        } label: {
            iconLabel("trash", text: "Delete", tint: .red,
                      enabled: editModel.selectedObjectID != nil)
        }
        .disabled(editModel.selectedObjectID == nil)
        .accessibilityIdentifier("editAction.Delete")
    }

    private var liftButton: some View {
        Button {
            editModel.verticalDragMode.toggle()
        } label: {
            iconLabel("arrow.up.and.down", text: "Lift", tint: .cyan,
                      enabled: editModel.selectedObjectID != nil,
                      filled: editModel.verticalDragMode)
        }
        .disabled(editModel.selectedObjectID == nil)
        .accessibilityIdentifier("editAction.Lift")
    }

    private var undoButton: some View {
        Button {
            editModel.undo()
        } label: {
            iconLabel("arrow.uturn.backward", text: "Undo", tint: .white,
                      enabled: editModel.canUndo)
        }
        .disabled(!editModel.canUndo)
        .accessibilityIdentifier("editAction.Undo")
    }

    private func iconLabel(
        _ systemImage: String,
        text: String,
        tint: Color,
        enabled: Bool,
        filled: Bool = false
    ) -> some View {
        let resolved = enabled ? tint : Color.gray
        return VStack(spacing: 2) {
            Image(systemName: systemImage)
                .font(.title3)
            Text(text)
                .font(.caption2)
        }
        .foregroundStyle(filled ? .black : resolved)
        .frame(width: 52, height: 52)
        .background(
            Circle().fill(
                filled
                    ? AnyShapeStyle(resolved)
                    : AnyShapeStyle(Color.white.opacity(0.001))
            )
        )
    }
}
