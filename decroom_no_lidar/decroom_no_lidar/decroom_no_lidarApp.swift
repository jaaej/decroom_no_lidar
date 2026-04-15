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

    var body: some Scene {
        WindowGroup {
            MainTabView()
                .environmentObject(roomStore)
        }
    }
}
