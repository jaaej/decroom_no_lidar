//
//  decroom_no_lidarApp.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import SwiftUI

@main
struct decroom_no_lidarApp: App {
    @StateObject private var roomStore = RoomStore()
    @State private var didFinishSplash = false

    var body: some Scene {
        WindowGroup {
            ZStack {
                MainTabView()
                    .environmentObject(roomStore)
                    .opacity(didFinishSplash ? 1 : 0)

                if !didFinishSplash {
                    SplashView {
                        withAnimation(.easeInOut(duration: 0.35)) {
                            didFinishSplash = true
                        }
                    }
                    .transition(.opacity)
                }
            }
        }
    }
}
