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
            if ProcessInfo.processInfo.arguments.contains("--ui-test-room-edit-stub") {
                UITestStubView()
            } else {
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
}

// MARK: - UI Test Stub (active only when --ui-test-room-edit-stub launch arg is set)

private struct UITestStubView: View {
    @StateObject private var editModel: RoomEditModel
    init() {
        let chair = RoomObject(
            id: "chair_test", type: "chair", asset_id: "kenney_chair",
            position: ObjectPosition(x: 0, y: 0, z: 0), rotation_y: 0, scale: 1,
            confidence: 0.9, metadata: ObjectMetadata(width: 0.5, length: 0.5, height: 0.85)
        )
        let walls = [
            Wall(id: "w0", start: [-2, -2], end: [2, -2], has_window: false, has_door: false),
            Wall(id: "w1", start: [2, -2],  end: [2, 2],  has_window: false, has_door: false),
            Wall(id: "w2", start: [2, 2],   end: [-2, 2], has_window: false, has_door: false),
            Wall(id: "w3", start: [-2, 2],  end: [-2, -2], has_window: false, has_door: false),
        ]
        let room = RoomInfo(id: "r", width_m: 4, length_m: 4, height_m: 2.5, walls: walls)
        let payload = RoomPayload(room: room, objects: [chair])
        _editModel = StateObject(wrappedValue: RoomEditModel(initial: payload))
    }
    var body: some View {
        RoomSceneView(payload: editModel.roomPayload, editModel: editModel)
            .accessibilityIdentifier("roomScene")
    }
}
