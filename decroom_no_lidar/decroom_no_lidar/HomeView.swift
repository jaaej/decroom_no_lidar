//
//  HomeView.swift
//  decroom_no_lidar
//
//  Cream paper home screen — direct port of screens-home.jsx structure:
//  greeting, stat strip, filter chips, featured card, your-rooms grid,
//  inspiration row, plus an interactive edit mode.
//

import SwiftUI

struct HomeView: View {
    @EnvironmentObject var roomStore: RoomStore
    @Binding var showTabBar: Bool
    var onScanFurniture: (() -> Void)? = nil

    @State private var navigationPath = NavigationPath()
    @State private var editing: Bool = false
    @State private var filter: String = "all"
    @State private var showSettings: Bool = false
    @AppStorage("home.blocks.json") private var blocksData: Data = Data()
    @AppStorage("home.showCardLikes") private var showCardLikes: Bool = true

    @State private var blocks: [HomeBlock] = DR_DEFAULT_HOME_BLOCKS

    var body: some View {
        NavigationStack(path: $navigationPath) {
            ZStack {
                (editing ? DR.cream2 : DR.cream).ignoresSafeArea()

                if roomStore.isLoading {
                    loading
                } else {
                    content
                }

                if editing { editTopBar }
            }
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(for: SavedRoom.ID.self) { roomId in
                if let room = roomStore.rooms.first(where: { $0.id == roomId }) {
                    RoomDetailView(room: room)
                }
            }
            .navigationDestination(isPresented: $showSettings) {
                SettingsView()
            }
        }
        .onAppear {
            if let stored = [HomeBlock].fromAppStorageData(blocksData) {
                self.blocks = stored
            }
        }
        .onChange(of: blocks) { _, new in
            blocksData = new.asAppStorageData
        }
        .onChange(of: navigationPath.count) { _, count in
            showTabBar = count == 0 && !editing
        }
        .onChange(of: editing) { _, isEditing in
            showTabBar = !isEditing && navigationPath.count == 0
        }
        .animation(.easeOut(duration: 0.22), value: editing)
    }

    // MARK: - Header (view mode)

    private var headerRow: some View {
        HStack(spacing: 8) {
            Spacer()
            iconBtn("pencil") { editing = true }
            iconBtn("gearshape") { showSettings = true }
            Button { /* open profile via tab — no-op for now */ } label: {
                DRAvatar(name: "Jae", size: 40, hue: 285)
            }
        }
    }

    private func iconBtn(_ system: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: system)
                .font(.system(size: 17, weight: .regular))
                .foregroundStyle(DR.ink2)
                .frame(width: 40, height: 40)
                .background(DR.paper, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(DR.line, lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Content

    private var content: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                if !editing {
                    headerRow
                        .padding(.horizontal, 20)
                        .padding(.top, 54)
                        .padding(.bottom, 4)
                }

                HomeBlockListView(blocks: $blocks, editing: editing) { block in
                    blockContent(block)
                }
                .padding(.horizontal, editing ? 12 : 20)
                .padding(.top, editing ? 130 : 0)
                .padding(.bottom, 120)
            }
        }
        .scrollIndicators(.hidden)
    }

    @ViewBuilder
    private func blockContent(_ block: HomeBlock) -> some View {
        switch block.id {
        case "greeting":
            VStack(alignment: .leading, spacing: 4) {
                (Text("Hi, ").foregroundStyle(DR.ink) +
                 Text("Jae").foregroundStyle(DR.ink).italic())
                    .font(DR.serif(30))
                    .tracking(-0.3)
                Text("Welcome back")
                    .font(DR.sans(12))
                    .foregroundStyle(DR.ink3)
            }
            .padding(.vertical, 4)

        case "stat_rooms":
            DRStatCard(label: "Rooms", value: "\(roomStore.rooms.count)",
                       sub: roomStore.rooms.count > 0 ? "+\(min(2, roomStore.rooms.count)) this week" : "scan to start",
                       dot: DR.indigo)

        case "stat_likes":
            let totalLikes = roomStore.rooms.reduce(0) { $0 + mockLikes(forRoomId: $1.id) }
            DRStatCard(label: "Likes", value: "\(totalLikes)",
                       sub: "this month", dot: DR.sageText)

        case "stat_rank":
            DRStatCard(label: "Rank", value: "#\(MockData.profile.rank)", sub: "rising", dot: DR.coral)

        case "filters":
            HStack(spacing: 8) {
                ForEach(["all", "recent", "favorites", "drafts"], id: \.self) { f in
                    DRChip(label: f.capitalized, active: filter == f) { filter = f }
                }
                Spacer(minLength: 0)
            }

        case "featured":
            if let first = roomStore.rooms.first {
                FeaturedRoomCard(room: first) {
                    navigationPath.append(first.id)
                }
                .padding(.top, 4)
            } else {
                emptyFeatured
            }

        case "grid":
            VStack(alignment: .leading, spacing: 10) {
                DRSectionHeader(title: "Your rooms", action: roomStore.rooms.count > 1 ? "See all" : nil)
                let cols = [GridItem(.flexible(), spacing: 12), GridItem(.flexible(), spacing: 12)]
                LazyVGrid(columns: cols, spacing: 12) {
                    let rest = Array(roomStore.rooms.dropFirst()) // featured used .first
                    ForEach(rest) { room in
                        GridRoomCard(room: room,
                                     onTap: { navigationPath.append(room.id) },
                                     showLikes: showCardLikes)
                    }
                    NewRoomCard(onTap: { onScanFurniture?() })
                }
            }

        case "inspiration":
            VStack(alignment: .leading, spacing: 10) {
                DRSectionHeader(title: "Inspiration", action: "Explore")
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(alignment: .top, spacing: 10) {
                        ForEach(Array(MockData.inspirationRows.enumerated()), id: \.offset) { i, item in
                            InspirationCard(title: item.title, author: item.author,
                                            seed: 20 + i, vibe: item.vibe)
                        }
                    }
                }
            }

        default:
            EmptyView()
        }
    }

    // MARK: - Empty featured

    private var emptyFeatured: some View {
        VStack(spacing: 10) {
            Image(systemName: "sparkles")
                .font(.system(size: 22, weight: .regular))
                .foregroundStyle(DR.indigo)
            Text("No rooms yet")
                .font(DR.serif(18))
                .foregroundStyle(DR.ink)
            Text("Tap the scan button below to begin.")
                .font(DR.sans(12))
                .foregroundStyle(DR.ink3)
        }
        .padding(.vertical, 36)
        .frame(maxWidth: .infinity)
        .background(DR.paper, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(DR.line2, lineWidth: 1)
        )
        .padding(.top, 4)
    }

    // MARK: - Edit mode top bar

    private var editTopBar: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .center, spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        DREyebrow(text: "EDIT MODE", color: .white.opacity(0.6))
                        Text("Arrange your home")
                            .font(DR.serif(20))
                            .foregroundStyle(DR.cream)
                            .tracking(-0.2)
                    }
                    Spacer()
                    Button {
                        withAnimation { blocks = DR_DEFAULT_HOME_BLOCKS }
                    } label: {
                        Text("Reset")
                            .font(DR.sans(12, weight: .medium))
                            .foregroundStyle(DR.cream)
                            .padding(.horizontal, 12).padding(.vertical, 8)
                            .overlay(Capsule().stroke(.white.opacity(0.2), lineWidth: 1))
                    }
                    Button {
                        withAnimation { editing = false }
                    } label: {
                        Text("Done")
                            .font(DR.sans(13, weight: .semibold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 16).padding(.vertical, 8)
                            .background(DR.coral, in: Capsule())
                    }
                }
                Text("Drag ⋮⋮ to reorder. Tap the eye to show/hide.")
                    .font(DR.sans(11))
                    .foregroundStyle(.white.opacity(0.6))
            }
            .padding(.horizontal, 16)
            .padding(.top, 54)
            .padding(.bottom, 14)
            .background(DR.ink.ignoresSafeArea(edges: .top))
            .shadow(color: DR.ink.opacity(0.18), radius: 8, y: 4)
            Spacer()
        }
    }

    // MARK: - Loading

    private var loading: some View {
        VStack(spacing: 14) {
            Spacer()
            ProgressView().tint(DR.indigo).scaleEffect(1.1)
            Text("LOADING ARCHIVE")
                .font(DR.mono(10, weight: .medium))
                .tracking(2.5)
                .foregroundStyle(DR.ink3)
            Spacer()
        }
    }
}

#Preview {
    HomeView(showTabBar: .constant(true), onScanFurniture: {})
        .environmentObject(RoomStore())
}
