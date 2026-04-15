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

    /// Upload from disk-based JPEG paths (video mode) — no CGImages held in memory.
    func processKeyframesFromDisk(_ jpegPaths: [URL], metadataJSON: String?) {
        guard !jpegPaths.isEmpty else {
            state = .failed("No keyframes captured.")
            return
        }
        state = .uploading

        Task {
            do {
                let payload = try await UploadService.uploadFromDisk(
                    jpegPaths: jpegPaths,
                    metadataJSON: metadataJSON
                )
                self.state = .finished(payload)
            } catch {
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    func processCapturedFrames(_ frames: [CGImage], metadataJSON: String?) {
        guard !frames.isEmpty else {
            state = .failed("No frames captured.")
            return
        }
        state = .uploading

        // Use Task {} (not Task.detached) so state updates stay on @MainActor.
        // The heavy work (JPEG encoding, network I/O) happens inside
        // UploadService.upload() via its own Task.detached.
        Task {
            do {
                let payload = try await UploadService.upload(
                    images: frames,
                    metadataJSON: metadataJSON
                )
                self.state = .finished(payload)
            } catch {
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    func reset() {
        state = .idle
    }
}
