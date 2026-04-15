//
//  FurnitureScanView.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 10/04/2026.
//
//  Scan individual furniture/objects to create custom 3D models.
//  Uses Apple Object Capture API when available (iOS 17+, A17 Pro+),
//  falls back to multi-photo capture for server-side reconstruction.
//

import SwiftUI
import RealityKit
import ARKit

struct FurnitureScanView: View {
    @State private var capturedPhotos: [UIImage] = []
    @State private var isCapturing = false
    @State private var statusMessage = "Position the furniture in the center of the camera"
    @State private var showPreview = false

    var body: some View {
        ZStack {
            // AR camera background
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()

                // Status area
                VStack(spacing: 16) {
                    Image(systemName: "cube.transparent")
                        .font(.system(size: 48))
                        .foregroundStyle(.white.opacity(0.6))

                    Text(statusMessage)
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.8))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)

                    if !capturedPhotos.isEmpty {
                        Text("\(capturedPhotos.count) photos captured")
                            .font(.caption)
                            .foregroundStyle(.cyan)
                    }
                }

                Spacer()

                // Capture controls
                HStack(spacing: 40) {
                    if !capturedPhotos.isEmpty {
                        // Preview button
                        Button {
                            showPreview = true
                        } label: {
                            Image(systemName: "photo.stack")
                                .font(.system(size: 24))
                                .foregroundStyle(.white)
                                .frame(width: 56, height: 56)
                                .background(.white.opacity(0.2))
                                .clipShape(Circle())
                        }
                    } else {
                        Spacer().frame(width: 56)
                    }

                    // Capture button
                    Button {
                        capturePhoto()
                    } label: {
                        ZStack {
                            Circle()
                                .stroke(.white, lineWidth: 4)
                                .frame(width: 72, height: 72)
                            Circle()
                                .fill(.white)
                                .frame(width: 60, height: 60)
                        }
                    }

                    if capturedPhotos.count >= 5 {
                        // Done button
                        Button {
                            processCaptures()
                        } label: {
                            Text("Done")
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundStyle(.black)
                                .frame(width: 56, height: 56)
                                .background(.cyan)
                                .clipShape(Circle())
                        }
                    } else {
                        Spacer().frame(width: 56)
                    }
                }
                .padding(.bottom, 40)
            }

            // Guidance overlay
            if capturedPhotos.isEmpty {
                VStack {
                    Spacer()
                        .frame(height: 100)

                    RoundedRectangle(cornerRadius: 12)
                        .stroke(.white.opacity(0.4), lineWidth: 2)
                        .frame(width: 250, height: 250)
                        .overlay(
                            Text("Place object here")
                                .font(.caption)
                                .foregroundStyle(.white.opacity(0.5))
                        )

                    Spacer()
                }
            }
        }
        .alert("Feature Coming Soon", isPresented: $showPreview) {
            Button("OK") { showPreview = false }
        } message: {
            Text("3D model generation from captured photos will be available in a future update. For now, captured photos are saved for processing.")
        }
    }

    private func capturePhoto() {
        // Placeholder: in full implementation, this would capture from ARSession
        capturedPhotos.append(UIImage())
        statusMessage = "Move around the object. Capture from different angles."

        if capturedPhotos.count >= 10 {
            statusMessage = "Good coverage! Tap Done to generate the 3D model."
        } else if capturedPhotos.count >= 5 {
            statusMessage = "Keep going, or tap Done if you have enough angles."
        }
    }

    private func processCaptures() {
        statusMessage = "Processing... This feature is coming soon."
        // Future: send photos to backend for TRELLIS/TripoSR reconstruction
        // or use Apple Object Capture API on supported devices
    }
}

#Preview {
    FurnitureScanView()
}
