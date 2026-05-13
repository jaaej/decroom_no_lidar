//
//  RecordingProcessor.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import Foundation
import SwiftUI

@MainActor
final class RecordingProcessor: ObservableObject {
    enum ProcessingState: Equatable {
        case idle
        case extracting
        case uploading
        case finished(RoomPayload)
        case failed(String)

        static func == (lhs: ProcessingState, rhs: ProcessingState) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.extracting, .extracting), (.uploading, .uploading):
                return true
            case (.finished(let a), .finished(let b)):
                return a.id == b.id
            case (.failed(let a), .failed(let b)):
                return a == b
            default:
                return false
            }
        }

        var isFinished: Bool {
            if case .finished = self { return true }
            return false
        }
    }

    @Published var state: ProcessingState = .idle

    /// Upload a phone-first payload. Publishes `.uploading` then either
    /// `.finished(RoomPayload)` or `.failed(String)` with the specific
    /// ScanError message (preserves "no fallback, errors explain what
    /// happened").
    func processPhoneFirstPayload(_ payload: ScanUploadPayload) {
        state = .uploading
        Task {
            do {
                let roomPayload = try await UploadService.uploadPhoneFirst(payload)
                self.state = .finished(roomPayload)
            } catch let e as ScanError {
                self.state = .failed(e.message)
            } catch {
                self.state = .failed("Unexpected upload error: \(error.localizedDescription)")
            }
        }
    }

    func reset() {
        state = .idle
    }
}
