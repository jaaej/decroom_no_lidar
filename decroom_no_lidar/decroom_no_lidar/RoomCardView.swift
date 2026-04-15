//
//  RoomCardView.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import SwiftUI

/// A large vertical card showing room name above a big isometric 3D preview.
struct RoomCardView: View {
    let room: SavedRoom
    @Binding var navigationPath: NavigationPath

    @ObservedObject private var snapshotCache = RoomSnapshotCache.shared

    private let cardColor = Color(red: 0.91, green: 0.88, blue: 0.82) // warm cream
    private let previewHeight: CGFloat = 260

    var body: some View {
        Button {
            navigationPath.append(room.id)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                // Room name label above the card
                Text(room.name)
                    .font(.headline)
                    .foregroundStyle(.white)

                // Large 3D preview card
                snapshotPreview
                    .frame(height: previewHeight)
                    .frame(maxWidth: .infinity)
                    .background(cardColor)
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                    .shadow(color: .black.opacity(0.15), radius: 10, x: 0, y: 5)
            }
        }
        .buttonStyle(.plain)
        .onAppear {
            snapshotCache.ensureSnapshot(
                for: room,
                size: CGSize(width: UIScreen.main.bounds.width - 48, height: previewHeight)
            )
        }
    }

    @ViewBuilder
    private var snapshotPreview: some View {
        if let image = snapshotCache.image(for: room, size: CGSize(width: UIScreen.main.bounds.width - 48, height: previewHeight)) {
            Image(uiImage: image)
                .resizable()
                .aspectRatio(contentMode: .fill)
        } else {
            ZStack {
                Color(.systemGray6)
                VStack(spacing: 8) {
                    ProgressView()
                    Text("\(room.payload.objects.count) objects")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}

/// Full-screen room detail view with interactive 3D scene.
struct RoomDetailView: View {
    let room: SavedRoom

    var body: some View {
        VStack(spacing: 16) {
            RoomSceneView(payload: room.payload)
                .frame(maxHeight: .infinity)

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Label("\(room.payload.objects.count) objects", systemImage: "chair.lounge")
                    Spacer()
                    Label(
                        "\(String(format: "%.1f", room.payload.envelope.width))m × \(String(format: "%.1f", room.payload.envelope.length))m × \(String(format: "%.1f", room.payload.envelope.height))m",
                        systemImage: "ruler"
                    )
                }
                .font(.subheadline)
                .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)
        }
        .padding()
        .navigationTitle(room.name)
        .navigationBarTitleDisplayMode(.inline)
    }
}
