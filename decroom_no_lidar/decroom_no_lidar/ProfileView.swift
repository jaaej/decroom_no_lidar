//
//  ProfileView.swift
//  decroom_no_lidar
//
//  Profile screen ported from screens-social.jsx ProfileScreen.
//  Real saved rooms appear in the Rooms tab; the rest is mock.
//

import SwiftUI

struct ProfileView: View {
    @EnvironmentObject var roomStore: RoomStore
    @State private var tab: ProfileTab = .rooms
    enum ProfileTab { case rooms, liked, activity }

    var body: some View {
        ZStack(alignment: .top) {
            DR.cream.ignoresSafeArea()

            // Gradient header (250pt)
            ZStack {
                LinearGradient(
                    colors: [DR.profTop, DR.profBot],
                    startPoint: .topLeading,
                    endPoint: UnitPoint(x: 0.17, y: 1)
                )
                // Decorative circles
                Circle()
                    .fill(.white.opacity(0.18))
                    .frame(width: 140, height: 140)
                    .offset(x: 159, y: -65)
                Circle()
                    .fill(DR.coral.opacity(0.30))
                    .frame(width: 180, height: 180)
                    .offset(x: -171, y: 75)
            }
            .frame(height: 250)
            .clipped()
            .ignoresSafeArea(edges: .top)

            // Top chrome
            VStack {
                HStack {
                    Spacer()
                    headerButton("square.and.arrow.up")
                    headerButton("gearshape")
                }
                .padding(.horizontal, 16)
                .padding(.top, 50)
                Spacer()
            }

            // Body content
            ScrollView(showsIndicators: false) {
                VStack(spacing: 0) {
                    Spacer().frame(height: 140)
                    profileCard
                        .padding(.horizontal, 16)
                    profileTabs
                        .padding(.horizontal, 16)
                        .padding(.top, 22)
                    tabContent
                        .padding(.top, 12)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 120)
                }
            }
        }
    }

    private func headerButton(_ system: String) -> some View {
        Image(systemName: system)
            .font(.system(size: 17, weight: .regular))
            .foregroundStyle(.white)
            .frame(width: 38, height: 38)
            .background(.white.opacity(0.2), in: RoundedRectangle(cornerRadius: 12))
    }

    private var profileCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .center, spacing: 14) {
                DRAvatar(name: MockData.profile.name, size: 64, hue: 285)
                VStack(alignment: .leading, spacing: 2) {
                    Text(MockData.profile.name)
                        .font(DR.serif(22))
                        .foregroundStyle(DR.ink)
                        .tracking(-0.2)
                    Text("\(MockData.profile.handle) · \(MockData.profile.joined)")
                        .font(DR.sans(12.5))
                        .foregroundStyle(DR.ink3)
                    HStack(spacing: 6) {
                        if MockData.profile.isPro {
                            Text("PRO")
                                .font(DR.mono(9.5, weight: .medium))
                                .tracking(1.4)
                                .foregroundStyle(DR.sageText)
                                .padding(.horizontal, 8).padding(.vertical, 3)
                                .background(DR.sageBg, in: Capsule())
                        }
                        Text("RANK #\(MockData.profile.rank)")
                            .font(DR.mono(9.5, weight: .medium))
                            .tracking(1.4)
                            .foregroundStyle(DR.indigo)
                            .padding(.horizontal, 8).padding(.vertical, 3)
                            .background(DR.indigoTint, in: Capsule())
                    }
                    .padding(.top, 6)
                }
                Spacer()
            }

            Text(MockData.profile.bio)
                .font(DR.sans(13))
                .foregroundStyle(DR.ink2)
                .lineSpacing(3)
                .padding(.top, 14)

            Rectangle().fill(DR.line2).frame(height: 1)
                .padding(.top, 14)

            HStack(spacing: 0) {
                profileStat(label: "ROOMS", value: "\(roomStore.rooms.count)")
                Rectangle().fill(DR.line2).frame(width: 1)
                profileStat(label: "LIKES", value: "\(MockData.profile.likesCount)")
                Rectangle().fill(DR.line2).frame(width: 1)
                profileStat(label: "FRIENDS", value: "\(MockData.profile.friendsCount)")
            }
            .padding(.top, 14)
        }
        .padding(18)
        .background(DR.paper, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .shadow(color: DR.ink.opacity(0.10), radius: 16, y: 12)
    }

    private func profileStat(label: String, value: String) -> some View {
        VStack(spacing: 3) {
            Text(value)
                .font(DR.serif(22))
                .foregroundStyle(DR.ink)
            Text(label)
                .font(DR.mono(10, weight: .medium))
                .tracking(1.6)
                .foregroundStyle(DR.ink3)
        }
        .frame(maxWidth: .infinity)
    }

    private var profileTabs: some View {
        HStack(spacing: 20) {
            tabButton("Rooms",    target: .rooms)
            tabButton("Liked",    target: .liked)
            tabButton("Activity", target: .activity)
            Spacer()
        }
        .overlay(alignment: .bottom) {
            Rectangle().fill(DR.line).frame(height: 1)
        }
    }

    private func tabButton(_ title: String, target: ProfileTab) -> some View {
        let active = tab == target
        return Button {
            withAnimation(.easeOut(duration: 0.18)) { tab = target }
        } label: {
            Text(title)
                .font(DR.sans(13, weight: active ? .semibold : .medium))
                .foregroundStyle(active ? DR.ink : DR.ink3)
                .padding(.vertical, 12)
                .overlay(alignment: .bottom) {
                    Rectangle()
                        .fill(active ? DR.ink : .clear)
                        .frame(height: 2)
                        .offset(y: 1)
                }
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var tabContent: some View {
        switch tab {
        case .rooms:    roomsTab
        case .liked:    likedTab
        case .activity: activityTab
        }
    }

    private var roomsTab: some View {
        let cols = [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)]
        return Group {
            if roomStore.rooms.isEmpty {
                emptyState(icon: "house", title: "No rooms yet", subtitle: "Tap scan to start your archive.")
            } else {
                LazyVGrid(columns: cols, spacing: 10) {
                    ForEach(roomStore.rooms) { room in
                        VStack(alignment: .leading, spacing: 6) {
                            RoomPlaceholderView(seed: abs(room.id.drStableHash) % 999, label: nil, vibe: RoomVibe.from(id: room.id))
                                .frame(height: 130)
                                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                            Text(room.name)
                                .font(DR.sans(12.5, weight: .semibold))
                                .foregroundStyle(DR.ink)
                                .lineLimit(1)
                            HStack {
                                Text("\(room.payload.objects.count) items")
                                    .font(DR.sans(10.5))
                                    .foregroundStyle(DR.ink3)
                                Spacer()
                                HStack(spacing: 3) {
                                    Image(systemName: "heart").font(.system(size: 10))
                                    Text("\(mockLikes(forRoomId: room.id))").font(DR.sans(10.5))
                                }
                                .foregroundStyle(DR.ink3)
                            }
                        }
                    }
                }
            }
        }
    }

    private var likedTab: some View {
        emptyState(icon: "heart", title: "No likes yet", subtitle: "Rooms you like will appear here.")
    }

    private var activityTab: some View {
        VStack(spacing: 0) {
            ForEach(MockData.activity) { item in
                HStack(spacing: 12) {
                    DRAvatar(name: item.who, size: 36, hue: item.hue)
                    HStack(spacing: 0) {
                        Text(item.who).font(DR.sans(13, weight: .semibold)).foregroundStyle(DR.ink)
                        Text(" \(item.verb) ").font(DR.sans(13)).foregroundStyle(DR.ink2)
                        if let target = item.target {
                            Text("\u{201c}\(target)\u{201d}").font(DR.sans(13, weight: .semibold)).foregroundStyle(DR.indigo)
                        }
                    }
                    .lineLimit(2)
                    Spacer()
                    Text(item.time)
                        .font(DR.sans(11))
                        .foregroundStyle(DR.ink3)
                }
                .padding(.vertical, 10)
                Rectangle().fill(DR.line2).frame(height: 1)
            }
        }
    }

    private func emptyState(icon: String, title: String, subtitle: String) -> some View {
        VStack(spacing: 14) {
            ZStack {
                Circle().fill(DR.cream2).frame(width: 60, height: 60)
                Image(systemName: icon).font(.system(size: 22, weight: .regular)).foregroundStyle(DR.ink3)
            }
            Text(title)
                .font(DR.sans(14, weight: .semibold))
                .foregroundStyle(DR.ink)
            Text(subtitle)
                .font(DR.sans(12.5))
                .foregroundStyle(DR.ink3)
        }
        .padding(.vertical, 60)
        .frame(maxWidth: .infinity)
    }
}

#Preview {
    ProfileView()
        .environmentObject(RoomStore())
}
