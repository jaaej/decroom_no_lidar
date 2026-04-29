//
//  MainTabView.swift
//  decroom_no_lidar
//
//  5-tab cream tab bar. Center scan button routes to the existing
//  CaptureFlowView; top/leaderboard remains a placeholder; settings
//  is reached via the gear icon on Home, not the bar.
//

import SwiftUI

struct MainTabView: View {
    @State private var selectedTab: Int = 0
    @State private var showTabBar: Bool = true
    @State private var initialScanMode: ScanMode = .video

    var body: some View {
        ZStack(alignment: .bottom) {
            Group {
                switch selectedTab {
                case 0:
                    HomeView(showTabBar: $showTabBar, onScanFurniture: {
                        initialScanMode = .furniture
                        selectedTab = 2
                    })
                case 1:
                    ExploreView()
                case 2:
                    CaptureFlowView(goHome: { selectedTab = 0 }, initialMode: initialScanMode)
                case 3:
                    TopPlaceholderView()
                case 4:
                    ProfileView()
                default:
                    HomeView(showTabBar: $showTabBar)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .transition(.opacity.combined(with: .move(edge: .top)))
            .animation(.easeOut(duration: 0.26), value: selectedTab)

            if showTabBar && selectedTab != 2 {
                customTabBar
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .ignoresSafeArea(.keyboard)
        .background(DR.cream)
        .animation(.spring(response: 0.45, dampingFraction: 0.85), value: showTabBar)
    }

    // MARK: - Tab bar

    private var customTabBar: some View {
        ZStack(alignment: .bottom) {
            // Cream paper background with soft fade-in from above
            LinearGradient(
                colors: [DR.cream.opacity(0), DR.cream],
                startPoint: .top,
                endPoint: .center
            )
            .frame(height: 48)
            .frame(maxWidth: .infinity, alignment: .top)
            .allowsHitTesting(false)

            HStack(spacing: 0) {
                tabGlyph(icon: "house",       label: "Home",    index: 0)
                tabGlyph(icon: "magnifyingglass", label: "Explore", index: 1)
                scanButton
                tabGlyph(icon: "trophy",      label: "Top",     index: 3)
                tabGlyph(icon: "person.crop.circle", label: "You", index: 4)
            }
            .padding(.horizontal, 10)
            .padding(.bottom, 20)
            .frame(height: 88, alignment: .bottom)
            .background(
                DR.paper.ignoresSafeArea(edges: .bottom)
                    .opacity(0.0001) // hit area, transparent
            )
        }
        .frame(height: 88)
        .background(
            DR.paper.opacity(0.98).ignoresSafeArea(edges: .bottom)
        )
    }

    private func tabGlyph(icon: String, label: String, index: Int) -> some View {
        let active = selectedTab == index
        return Button {
            withAnimation(.easeOut(duration: 0.2)) { selectedTab = index }
        } label: {
            VStack(spacing: 3) {
                Image(systemName: icon)
                    .font(.system(size: 20, weight: active ? .semibold : .regular))
                    .foregroundStyle(active ? DR.indigo : DR.ink3)
                Text(label)
                    .font(DR.sans(10.5, weight: active ? .semibold : .medium))
                    .tracking(0.1)
                    .foregroundStyle(active ? DR.indigo : DR.ink3)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
        }
        .buttonStyle(.plain)
    }

    private var scanButton: some View {
        Button {
            initialScanMode = .video
            withAnimation(.easeOut(duration: 0.2)) { selectedTab = 2 }
        } label: {
            ZStack {
                Circle()
                    .fill(DR.ink)
                    .frame(width: 62, height: 62)
                    .overlay(
                        // 4px cream paper ring (matches border: 4px solid var(--paper))
                        Circle()
                            .stroke(DR.paper, lineWidth: 4)
                    )
                    .shadow(color: DR.ink.opacity(0.25), radius: 12, y: 6)

                Circle()
                    .stroke(DR.cream.opacity(0.5), lineWidth: 1.5)
                    .frame(width: 50, height: 50)

                Image(systemName: "qrcode.viewfinder")
                    .font(.system(size: 22, weight: .regular))
                    .foregroundStyle(DR.cream)
            }
            .offset(y: -14)
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity)
    }
}

// MARK: - "Top" leaderboard placeholder

struct TopPlaceholderView: View {
    var body: some View {
        ZStack {
            DR.cream.ignoresSafeArea()

            VStack(spacing: 18) {
                Spacer()
                ZStack {
                    Circle().stroke(DR.indigo.opacity(0.4), lineWidth: 1).frame(width: 96, height: 96)
                    Image(systemName: "trophy")
                        .font(.system(size: 28, weight: .regular))
                        .foregroundStyle(DR.indigo)
                }
                VStack(spacing: 6) {
                    DREyebrow(text: "TOP / SOON")
                    Text("Leaderboard")
                        .font(DR.serif(28))
                        .foregroundStyle(DR.ink)
                    Text("Ranked rooms and creators land here.")
                        .font(DR.sans(13))
                        .foregroundStyle(DR.ink3)
                }
                Spacer()
            }
            .frame(maxWidth: .infinity)
        }
    }
}

// Backwards-compat for any older code that still spins up `PlaceholderView`.
struct PlaceholderView: View {
    let title: String
    let icon: String
    let goHome: () -> Void

    var body: some View {
        ZStack {
            DR.cream.ignoresSafeArea()
            VStack(spacing: 18) {
                Spacer()
                Image(systemName: icon)
                    .font(.system(size: 56, weight: .regular))
                    .foregroundStyle(DR.indigo.opacity(0.5))
                Text(title)
                    .font(DR.serif(28))
                    .foregroundStyle(DR.ink)
                Text("Coming soon")
                    .font(DR.sans(13))
                    .foregroundStyle(DR.ink3)
                Spacer()
            }
            .overlay(alignment: .topLeading) {
                Button(action: goHome) {
                    HStack(spacing: 6) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 13, weight: .semibold))
                        Text("Home")
                            .font(DR.sans(13, weight: .medium))
                    }
                    .foregroundStyle(DR.ink2)
                    .padding(.horizontal, 14).padding(.vertical, 10)
                    .background(DR.cream2, in: Capsule())
                }
                .padding(.leading, 16).padding(.top, 56)
            }
        }
    }
}

#Preview {
    MainTabView()
        .environmentObject(RoomStore())
}
