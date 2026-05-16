//
//  RoomEditMenuView.swift
//  decroom_no_lidar
//
//  Plan 4a v4 — vertical side bar on the trailing edge of the screen
//  with the editing actions. Easier to spot than a bottom sheet that
//  competes with the Save Room overlay; matches the user-requested
//  layout (vertical side bar).
//
//   • Delete (when selectedObjectID != nil)
//   • Lift toggle (when selectedObjectID != nil)
//   • Undo (when canUndo)
//

import SwiftUI

struct RoomEditMenuView: View {
    @ObservedObject var editModel: RoomEditModel

    var body: some View {
        HStack {
            Spacer()
            VStack(spacing: 14) {
                Spacer()
                if editModel.selectedObjectID != nil {
                    deleteButton
                    liftButton
                }
                if editModel.canUndo {
                    undoButton
                }
                Spacer()
            }
            .padding(.vertical, 8)
            .padding(.horizontal, 8)
            .background(
                Capsule().fill(.ultraThinMaterial)
                    .opacity(shouldShowAny ? 1 : 0)
            )
            .padding(.trailing, 12)
            .padding(.vertical, 80)   // pull away from notch + bottom Save Room
        }
        .animation(.easeInOut(duration: 0.15), value: editModel.selectedObjectID)
        .animation(.easeInOut(duration: 0.15), value: editModel.canUndo)
        .animation(.easeInOut(duration: 0.15), value: editModel.verticalDragMode)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .allowsHitTesting(shouldShowAny)
    }

    private var shouldShowAny: Bool {
        editModel.selectedObjectID != nil || editModel.canUndo
    }

    private var deleteButton: some View {
        Button {
            if let id = editModel.selectedObjectID {
                editModel.delete(id)
            }
        } label: {
            iconLabel("trash", text: "Delete", tint: .red, filled: false)
        }
        .accessibilityIdentifier("editAction.Delete")
    }

    private var liftButton: some View {
        Button {
            editModel.verticalDragMode.toggle()
        } label: {
            iconLabel(
                "arrow.up.and.down",
                text: "Lift",
                tint: .accentColor,
                filled: editModel.verticalDragMode
            )
        }
        .accessibilityIdentifier("editAction.Lift")
    }

    private var undoButton: some View {
        Button {
            editModel.undo()
        } label: {
            iconLabel("arrow.uturn.backward", text: "Undo", tint: .primary, filled: false)
        }
        .accessibilityIdentifier("editAction.Undo")
    }

    private func iconLabel(
        _ systemImage: String,
        text: String,
        tint: Color,
        filled: Bool
    ) -> some View {
        VStack(spacing: 2) {
            Image(systemName: systemImage)
                .font(.title3)
            Text(text)
                .font(.caption2)
        }
        .foregroundStyle(filled ? .white : tint)
        .frame(width: 52, height: 52)
        .background(
            Circle().fill(
                filled
                    ? AnyShapeStyle(tint)
                    : AnyShapeStyle(Color.white.opacity(0.001))   // hit area
            )
        )
    }
}
