//
//  RoomEditMenuView.swift
//  decroom_no_lidar
//
//  Plan 4a — bottom action sheet + top-trailing Undo button.
//  Bound to RoomEditModel; renders nothing when selectedObjectID is nil
//  and canUndo is false.
//

import SwiftUI

struct RoomEditMenuView: View {
    @ObservedObject var editModel: RoomEditModel

    var body: some View {
        ZStack(alignment: .topTrailing) {
            if editModel.canUndo {
                Button {
                    editModel.undo()
                } label: {
                    Label("Undo", systemImage: "arrow.uturn.backward")
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                }
                .padding(12)
            }

            if editModel.selectedObjectID != nil {
                VStack(spacing: 0) {
                    Spacer()
                    actionSheet
                }
                .frame(maxWidth: .infinity)
            }
        }
        .animation(.easeInOut(duration: 0.15), value: editModel.selectedObjectID)
        .animation(.easeInOut(duration: 0.15), value: editModel.activeAction)
        .animation(.easeInOut(duration: 0.15), value: editModel.canUndo)
    }

    @ViewBuilder
    private var actionSheet: some View {
        VStack(spacing: 8) {
            if let action = editModel.activeAction {
                Text(hintText(for: action))
                    .font(.subheadline)
                    .padding(.vertical, 6)
                HStack {
                    Spacer()
                    Button("Done") {
                        editModel.activeAction = nil
                    }
                    .buttonStyle(.borderedProminent)
                    Spacer()
                }
            } else {
                HStack(spacing: 16) {
                    actionButton("Delete", systemImage: "trash") {
                        if let id = editModel.selectedObjectID {
                            editModel.delete(id)
                        }
                    }
                    actionButton("Move", systemImage: "arrow.up.and.down.and.arrow.left.and.right") {
                        editModel.activeAction = .move
                    }
                    actionButton("Rotate", systemImage: "rotate.right") {
                        editModel.activeAction = .rotate
                    }
                    actionButton("Done", systemImage: "checkmark") {
                        editModel.select(nil)
                    }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal, 12)
        .padding(.bottom, 16)
    }

    private func actionButton(
        _ label: String, systemImage: String, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: systemImage).font(.title3)
                Text(label).font(.caption)
            }
            .frame(minWidth: 56, minHeight: 44)
        }
        .accessibilityIdentifier("editAction.\(label)")
    }

    private func hintText(for action: EditAction) -> String {
        switch action {
        case .move:   return "Drag to reposition. Tap Done to finish."
        case .rotate: return "Twist with two fingers. Tap Done to finish."
        }
    }
}
