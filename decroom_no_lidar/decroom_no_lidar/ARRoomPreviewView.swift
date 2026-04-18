//
//  ARRoomPreviewView.swift
//  decroom_no_lidar
//
//  Scanning experience: live AR camera fills the entire screen.
//  A Canvas overlay draws the equirectangular panorama around a
//  transparent center window, so captured room images surround
//  the live camera feed. Uncaptured areas are dark.
//
//  Layer order (bottom to top):
//    1. ARView (full screen live camera)
//    2. Canvas overlay (panorama + dark uncaptured + transparent center hole)
//    3. Center window border
//    4. Coverage card
//

import SwiftUI

struct ARRoomPreviewView: View {
    @StateObject private var session = ScanSession()

    var body: some View {
        GeometryReader { geo in
            let screenW = geo.size.width
            let screenH = geo.size.height
            let centerW = screenW * 0.55
            let centerH = screenH * 0.50
            let centerRect = CGRect(
                x: (screenW - centerW) / 2,
                y: (screenH - centerH) / 2,
                width: centerW,
                height: centerH
            )

            ZStack {
                // Layer 1: Live AR camera (FULL SCREEN, at the bottom)
                ARRoomPreviewContainer(session: session)
                    .ignoresSafeArea()

                // Layer 2: Panorama overlay with transparent center hole
                panoramaOverlay(size: geo.size, centerRect: centerRect)
                    .ignoresSafeArea()
                    .allowsHitTesting(false)

                // Layer 3: Center window border
                RoundedRectangle(cornerRadius: 14)
                    .stroke(.white.opacity(0.5), lineWidth: 1.5)
                    .frame(width: centerW, height: centerH)

                // Layer 4: Coverage card at bottom
                VStack {
                    Spacer()
                    coverageCard
                        .padding(.horizontal)
                        .padding(.bottom, 8)
                }
            }
        }
    }

    private func panoramaOverlay(size: CGSize, centerRect: CGRect) -> some View {
        let _ = session.panoramaVersion  // trigger redraw

        return Canvas { context, canvasSize in
            // Start with fully dark background
            context.fill(
                Path(CGRect(origin: .zero, size: canvasSize)),
                with: .color(.black.opacity(0.92))
            )

            // Draw subtle grid dots on dark background
            let spacing: CGFloat = 28
            for x in stride(from: CGFloat(0), to: canvasSize.width, by: spacing) {
                for y in stride(from: CGFloat(0), to: canvasSize.height, by: spacing) {
                    context.fill(
                        Path(ellipseIn: CGRect(x: x - 0.8, y: y - 0.8, width: 1.6, height: 1.6)),
                        with: .color(.white.opacity(0.06))
                    )
                }
            }

            // Draw panorama slice (if we have captured frames)
            if let sliceImage = session.panorama.slice(
                centeredAtYaw: session.currentCameraYaw,
                visibleWidthDeg: 220,
                outputSize: canvasSize
            ) {
                context.draw(
                    Image(decorative: sliceImage, scale: 1),
                    in: CGRect(origin: .zero, size: canvasSize)
                )
            }

            // Cut out center window — makes it TRANSPARENT so the ARView behind shows through
            context.blendMode = .clear
            context.fill(
                Path(roundedRect: centerRect, cornerRadius: 14),
                with: .color(.white)
            )
        }
    }

    private var coverageCard: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text("\(session.geometry.walls.count) wall\(session.geometry.walls.count == 1 ? "" : "s") detected")
                    .font(.subheadline.weight(.medium))
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
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
