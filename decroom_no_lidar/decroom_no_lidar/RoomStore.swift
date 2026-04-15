//
//  RoomStore.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import Foundation
import SwiftUI

struct SavedRoom: Identifiable, Codable {
    let id: String
    let name: String
    let payload: RoomPayload
    let createdAt: Date

    init(name: String, payload: RoomPayload) {
        self.id = UUID().uuidString
        self.name = name
        self.payload = payload
        self.createdAt = Date()
    }
}

@MainActor
final class RoomStore: ObservableObject {
    @Published private(set) var rooms: [SavedRoom] = []
    @Published private(set) var isLoading: Bool = true

    private let saveKey = "saved_rooms"

    init() {
        // Start async load — UI shows immediately, rooms appear when ready
        Task { await loadAsync() }
    }

    func add(payload: RoomPayload, name: String) {
        let room = SavedRoom(name: name, payload: payload)
        rooms.insert(room, at: 0)
        save()
    }

    func delete(at offsets: IndexSet) {
        rooms.remove(atOffsets: offsets)
        save()
    }

    private func save() {
        let snapshot = rooms
        let url = Self.fileURL
        Task.detached(priority: .utility) {
            do {
                let data = try JSONEncoder().encode(snapshot)
                try data.write(to: url, options: .atomic)
            } catch {
                print("[RoomStore] Save failed: \(error)")
            }
        }
    }

    private func loadAsync() async {
        isLoading = true
        let url = Self.fileURL
        let key = saveKey

        let loaded: [SavedRoom]? = await Task.detached(priority: .userInitiated) {
            // Try file first (new storage), fall back to UserDefaults (legacy)
            if let data = try? Data(contentsOf: url) {
                return try? JSONDecoder().decode([SavedRoom].self, from: data)
            }
            // Legacy: migrate from UserDefaults
            if let data = UserDefaults.standard.data(forKey: key) {
                let rooms = try? JSONDecoder().decode([SavedRoom].self, from: data)
                // Migrate to file storage
                if let rooms = rooms, let encoded = try? JSONEncoder().encode(rooms) {
                    try? encoded.write(to: url, options: .atomic)
                    UserDefaults.standard.removeObject(forKey: key)
                }
                return rooms
            }
            return nil
        }.value

        rooms = loaded ?? []
        isLoading = false
    }

    // MARK: - File Storage

    private static var fileURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("decroom_saved_rooms.json")
    }
}
