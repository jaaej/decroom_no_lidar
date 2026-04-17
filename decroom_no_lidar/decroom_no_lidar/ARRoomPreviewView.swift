//
//  ARRoomPreviewView.swift
//  decroom_no_lidar
//
//  Scanning experience: equirectangular panorama builds up seamlessly
//  as the user rotates. Live camera feed shows through a center window.
//  Uncaptured areas show a dark grid pattern. Coverage card at bottom.
//

import SwiftUI

struct ARRoomPreviewView: View {
    @StateObject private var session = ScanSession()

    var body: some View {
        ZStack {
            // Layer 0: Dark background (uncaptured areas)
            Color.black
                .ignoresSafeArea()

            // Layer 1: Subtle grid dots on uncaptured areas
            gridPatternOverlay
                .ignoresSafeArea()
                .allowsHitTesting(false)

            // Layer 2: Panorama image (captured areas, scrolling with camera)
            panoramaLayer
                .ignoresSafeArea()

            // Layer 3: Live AR camera feed (center window)
            ARRoomPreviewContainer(session: session)
                .mask(
                    RoundedRectangle(cornerRadius: 16)
                        .frame(width: UIScreen.main.bounds.width * 0.6,
                               height: UIScreen.main.bounds.height * 0.55)
                )

            // Layer 4: Thin border around center window
            RoundedRectangle(cornerRadius: 16)
                .stroke(.white.opacity(0.4), lineWidth: 1.5)
                .frame(width: UIScreen.main.bounds.width * 0.6,
                       height: UIScreen.main.bounds.height * 0.55)

            // Layer 5: Coverage card at bottom
            VStack {
                Spacer()
                coverageCard
                    .padding()
            }
        }
    }

    @ViewBuilder
    private var panoramaLayer: some View {
        let _ = session.panoramaVersion  // trigger redraw on new frames

        Canvas { context, size in
            let currentYaw = estimateCurrentYaw()

            guard let sliceImage = session.panorama.slice(
                centeredAtYaw: currentYaw,
                visibleWidthDeg: 220,
                outputSize: size
            ) else { return }

            context.draw(
                Image(decorative: sliceImage, scale: 1),
                in: CGRect(origin: .zero, size: size)
            )

            // Clear center window so AR camera shows through
            let centerW = size.width * 0.6
            let centerH = size.height * 0.55
            let centerRect = CGRect(
                x: (size.width - centerW) / 2,
                y: (size.height - centerH) / 2,
                width: centerW,
                height: centerH
            )
            context.blendMode = .clear
            context.fill(
                Path(roundedRect: centerRect, cornerRadius: 16),
                with: .color(.white)
            )
        }
    }

    private var gridPatternOverlay: some View {
        Canvas { context, size in
            let spacing: CGFloat = 30
            for x in stride(from: CGFloat(0), to: size.width, by: spacing) {
                for y in stride(from: CGFloat(0), to: size.height, by: spacing) {
                    context.fill(
                        Path(ellipseIn: CGRect(x: x - 1, y: y - 1, width: 2, height: 2)),
                        with: .color(.white.opacity(0.08))
                    )
                }
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
            if session.panorama.frameCount > 0 {
                Text("\(session.panorama.frameCount) frames")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
        .padding(12)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private func estimateCurrentYaw() -> Float {
        guard let lastBin = session.visitedYawBins.max() else { return 0 }
        return (Float(lastBin) + 0.5) / 24.0 * 2 * .pi - .pi
    }
}
