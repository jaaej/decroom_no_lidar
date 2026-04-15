//
//  MainTabView.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import SwiftUI

struct MainTabView: View {
    @State private var selectedTab = 0
    @State private var showTabBar = true
    @State private var initialScanMode: ScanMode = .video

    var body: some View {
        ZStack(alignment: .bottom) {
            // Content area
            Group {
                switch selectedTab {
                case 0:
                    HomeView(showTabBar: $showTabBar, onScanFurniture: {
                        initialScanMode = .furniture
                        selectedTab = 2
                    })
                case 1:
                    PlaceholderView(title: "Achievements", icon: "trophy.fill") { selectedTab = 0 }
                case 2:
                    CaptureFlowView(goHome: { selectedTab = 0 }, initialMode: initialScanMode)
                case 3:
                    PlaceholderView(title: "Settings", icon: "gearshape.fill") { selectedTab = 0 }
                case 4:
                    PlaceholderView(title: "Profile", icon: "person.fill") { selectedTab = 0 }
                default:
                    HomeView(showTabBar: $showTabBar)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Custom tab bar — only visible on the home screen (not in detail views)
            if selectedTab == 0 && showTabBar {
                customTabBar
            }
        }
        .ignoresSafeArea(.keyboard)
    }

    // MARK: - Custom Tab Bar

    private var customTabBar: some View {
        HStack(spacing: 0) {
            // Tab 0: Home (smiley)
            tabButton(icon: "face.smiling", index: 0)

            // Tab 1: Achievements (trophy)
            tabButton(icon: "trophy.fill", index: 1)

            // Tab 2: Record (center red button)
            Button {
                initialScanMode = .video
                selectedTab = 2
            } label: {
                ZStack {
                    Circle()
                        .fill(selectedTab == 2 ? Color.red : Color(red: 0.85, green: 0.3, blue: 0.3))
                        .frame(width: 56, height: 56)
                        .shadow(color: .red.opacity(0.3), radius: 6, y: 3)

                    Circle()
                        .fill(Color.red.opacity(0.8))
                        .frame(width: 44, height: 44)
                }
                .offset(y: -8)
            }
            .frame(maxWidth: .infinity)

            // Tab 3: Settings (gear)
            tabButton(icon: "gearshape.fill", index: 3)

            // Tab 4: Profile (person)
            tabButton(icon: "person.circle", index: 4)
        }
        .padding(.horizontal, 16)
        .padding(.top, 8)
        .padding(.bottom, 20)
        .background(
            Color.white
                .shadow(color: .black.opacity(0.08), radius: 8, y: -2)
                .ignoresSafeArea(edges: .bottom)
        )
    }

    private func tabButton(icon: String, index: Int) -> some View {
        Button {
            selectedTab = index
        } label: {
            Image(systemName: icon)
                .font(.system(size: 22))
                .foregroundStyle(selectedTab == index ? .black : Color(.darkGray).opacity(0.5))
                .frame(maxWidth: .infinity)
                .frame(height: 36)
        }
    }
}

// MARK: - Placeholder View

struct PlaceholderView: View {
    let title: String
    let icon: String
    let goHome: () -> Void

    var body: some View {
        ZStack(alignment: .topLeading) {
            Color(red: 0.38, green: 0.45, blue: 0.95)
                .ignoresSafeArea()

            VStack(spacing: 16) {
                Spacer()

                Image(systemName: icon)
                    .font(.system(size: 56))
                    .foregroundStyle(.white.opacity(0.4))

                Text(title)
                    .font(.title2)
                    .bold()
                    .foregroundStyle(.white)

                Text("Coming soon")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.5))

                Spacer()
            }
            .frame(maxWidth: .infinity)

            // Back button
            Button(action: goHome) {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 17, weight: .semibold))
                    Text("Home")
                        .font(.body)
                }
                .foregroundStyle(.white)
            }
            .padding(.leading, 16)
            .padding(.top, 56)
        }
    }
}

#Preview {
    MainTabView()
        .environmentObject(RoomStore())
}
