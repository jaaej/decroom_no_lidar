//
//  HomeView.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import SwiftUI

struct HomeView: View {
    @EnvironmentObject var roomStore: RoomStore
    @Binding var showTabBar: Bool
    var onScanFurniture: (() -> Void)? = nil

    @State private var navigationPath = NavigationPath()

    // Blue/purple matching the Figma design
    private let backgroundColor = Color(red: 0.38, green: 0.45, blue: 0.95) // ~#6173F2

    var body: some View {
        NavigationStack(path: $navigationPath) {
            ZStack {
                backgroundColor.ignoresSafeArea()

                if roomStore.isLoading {
                    ProgressView()
                        .tint(.white)
                        .scaleEffect(1.5)
                } else if roomStore.rooms.isEmpty {
                    emptyState
                } else {
                    roomList
                }
            }
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbarBackground(.hidden, for: .navigationBar)
            .navigationDestination(for: SavedRoom.ID.self) { roomId in
                if let room = roomStore.rooms.first(where: { $0.id == roomId }) {
                    RoomDetailView(room: room)
                }
            }
        }
        .onChange(of: navigationPath.count) { _, newCount in
            showTabBar = newCount == 0
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "house.fill")
                .font(.system(size: 72))
                .foregroundStyle(.white.opacity(0.4))

            VStack(spacing: 8) {
                Text("No rooms yet")
                    .font(.title2)
                    .bold()
                    .foregroundStyle(.white)
                Text("Tap the record button below to scan your first room.")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.6))
                    .multilineTextAlignment(.center)
            }

            scanFurnitureButton

            Spacer()
        }
        .padding()
    }

    // MARK: - Room List

    private var roomList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(roomStore.rooms) { room in
                    RoomCardView(room: room, navigationPath: $navigationPath)
                }

                scanFurnitureButton
                    .padding(.top, 8)
            }
            .padding(.horizontal, 24)
            .padding(.top, 8)
            .padding(.bottom, 24)
        }
    }

    private var scanFurnitureButton: some View {
        Button {
            onScanFurniture?()
        } label: {
            HStack(spacing: 10) {
                Image(systemName: "cube.transparent")
                    .font(.system(size: 20))
                Text("Scan Furniture")
                    .font(.system(size: 16, weight: .semibold))
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .frame(height: 50)
            .background(.white.opacity(0.15))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(.white.opacity(0.3), lineWidth: 1)
            )
        }
    }
}

#Preview {
    HomeView(showTabBar: .constant(true), onScanFurniture: {})
        .environmentObject(RoomStore())
}
