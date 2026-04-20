//
//  CaptureFlowView.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 29/11/2025.
//

import SwiftUI

enum ScanMode: String, CaseIterable {
    case video = "Video Scan"
    case photo = "Photo Capture"
    case furniture = "Scan Furniture"
}

struct CaptureFlowView: View {
    @EnvironmentObject var roomStore: RoomStore
    @StateObject private var processor = RecordingProcessor()
    @State private var roomName: String = ""
    @State private var showNamingSheet = false
    @State private var scanMode: ScanMode

    var goHome: (() -> Void)? = nil
    var initialMode: ScanMode = .video

    init(goHome: (() -> Void)? = nil, initialMode: ScanMode = .video) {
        self.goHome = goHome
        self.initialMode = initialMode
        _scanMode = State(initialValue: initialMode)
    }

    var body: some View {
        NavigationStack {
            if DeviceCapabilities.currentTier == .fallback {
                unsupportedDeviceView
            } else {
                mainCaptureBody
            }
        }
    }

    private var unsupportedDeviceView: some View {
        VStack(spacing: 24) {
            Image(systemName: "iphone.slash")
                .font(.system(size: 72))
                .foregroundStyle(.secondary)
            Text(ScanError.unsupportedDevice.message)
                .font(.title3)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            if let goHome = goHome {
                Button("Back to Home") {
                    goHome()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var mainCaptureBody: some View {
        VStack {
            if case .idle = processor.state {
                    // Mode picker shown before capture starts
                    Picker("Scan Mode", selection: $scanMode) {
                        ForEach(ScanMode.allCases, id: \.self) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)
                    .padding(.top, 8)
                }

                switch scanMode {
                case .video:
                    VideoRecorderView(processor: processor)
                case .photo:
                    ContentView(processor: processor)
                case .furniture:
                    FurnitureScanView()
                }

                if case .finished(let payload) = processor.state {
                    // Auto-open 3D room preview with save overlay
                    ZStack {
                        RoomSceneView(payload: payload)
                            .ignoresSafeArea()

                        // Bottom overlay with stats + save button
                        VStack {
                            Spacer()

                            VStack(spacing: 12) {
                                HStack {
                                    Label("\(payload.objects.count) objects", systemImage: "chair.lounge")
                                    Spacer()
                                    Label(
                                        "\(String(format: "%.1f", payload.envelope.width))m x \(String(format: "%.1f", payload.envelope.length))m",
                                        systemImage: "ruler"
                                    )
                                }
                                .font(.subheadline)
                                .foregroundStyle(.white.opacity(0.8))

                                HStack(spacing: 12) {
                                    Button {
                                        processor.reset()
                                    } label: {
                                        Label("Rescan", systemImage: "arrow.counterclockwise")
                                            .frame(maxWidth: .infinity)
                                    }
                                    .buttonStyle(.bordered)
                                    .tint(.white)

                                    Button {
                                        showNamingSheet = true
                                    } label: {
                                        Label("Save Room", systemImage: "square.and.arrow.down")
                                            .frame(maxWidth: .infinity)
                                    }
                                    .buttonStyle(.borderedProminent)
                                }
                            }
                            .padding()
                            .background(.ultraThinMaterial)
                            .cornerRadius(16)
                            .padding()
                        }
                    }
                    .sheet(isPresented: $showNamingSheet) {
                        NavigationStack {
                            Form {
                                TextField("Room Name", text: $roomName)
                            }
                            .navigationTitle("Name Your Room")
                            .toolbar {
                                ToolbarItem(placement: .confirmationAction) {
                                    Button("Save") {
                                        let name = roomName.isEmpty ? "Room \(roomStore.rooms.count + 1)" : roomName
                                        roomStore.add(payload: payload, name: name)
                                        showNamingSheet = false
                                        roomName = ""
                                        processor.reset()
                                        goHome?()
                                    }
                                }
                                ToolbarItem(placement: .cancellationAction) {
                                    Button("Cancel") {
                                        showNamingSheet = false
                                    }
                                }
                            }
                        }
                        .presentationDetents([.medium])
                    }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar(.visible, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    if let goHome = goHome {
                        Button(action: {
                            processor.reset()
                            goHome()
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: "chevron.left")
                                    .font(.system(size: 17, weight: .semibold))
                                Text("Home")
                                    .font(.body)
                            }
                            .foregroundStyle(.white)
                        }
                    }
                }
                if case .finished = processor.state {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Reset") {
                            processor.reset()
                        }
                    }
                }
            }
    }
}

#Preview {
    CaptureFlowView()
        .environmentObject(RoomStore())
}
