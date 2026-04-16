//
//  ARRoomPreviewView.swift
//  decroom_no_lidar
//
//  The new visual-first scanning experience. Grid shell + reticle +
//  solidifying walls + coverage progress. Replaces VideoRecorderView
//  for .full and .reduced tier devices.
//

import SwiftUI

struct ARRoomPreviewView: View {
    @StateObject private var session = ScanSession()

    var body: some View {
        ZStack {
            ARRoomPreviewContainer(session: session)
                .edgesIgnoringSafeArea(.all)

            VStack {
                Spacer()
                coverageCard
                    .padding()
            }
        }
    }

    private var coverageCard: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("\(session.geometry.walls.count) wall\(session.geometry.walls.count == 1 ? "" : "s") detected")
                    .font(.subheadline)
                    .foregroundStyle(.white)
                Text("\(Int(session.coverageProgress * 100))% coverage")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
            }
            Spacer()
        }
        .padding(12)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }
}
