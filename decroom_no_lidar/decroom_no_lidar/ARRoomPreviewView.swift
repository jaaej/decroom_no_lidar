//
//  ARRoomPreviewView.swift
//  decroom_no_lidar
//
//  Scanning experience: live camera fills center window. Captured frame
//  thumbnails fill in around it at their angular positions, creating a
//  panoramic mosaic. Uncaptured areas are dark. As you rotate, the strip
//  scrolls — previously captured areas visible, uncaptured areas are dark.
//
//  No equirectangular projection — frames drawn flat at their yaw position
//  with edge opacity fading for seamless overlap. Same approach as
//  Martin Maly's app.
//

import SwiftUI

struct ARRoomPreviewView: View {
    @StateObject private var session = ScanSession()

    // Camera horizontal FOV in radians (~45° for portrait iPhone)
    private let cameraFOV: CGFloat = 0.82  // ~47°

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
                // Bottom: live AR camera (full screen)
                ARRoomPreviewContainer(session: session)
                    .ignoresSafeArea()

                // Overlay: panorama mosaic + dark uncaptured + transparent center
                panoramaMosaicOverlay(size: geo.size, centerRect: centerRect)
                    .ignoresSafeArea()
                    .allowsHitTesting(false)

                // Center window border
                RoundedRectangle(cornerRadius: 14)
                    .stroke(.white.opacity(0.4), lineWidth: 1.5)
                    .frame(width: centerW, height: centerH)

                // Coverage card
                VStack {
                    Spacer()
                    coverageCard
                        .padding(.horizontal)
                        .padding(.bottom, 8)
                }
            }
        }
    }

    private func panoramaMosaicOverlay(size: CGSize, centerRect: CGRect) -> some View {
        let _ = session.panoramaThumbnails.count  // trigger redraw

        return Canvas { context, canvasSize in
            let w = canvasSize.width
            let h = canvasSize.height

            // Dark background for uncaptured areas
            context.fill(
                Path(CGRect(origin: .zero, size: canvasSize)),
                with: .color(.black.opacity(0.92))
            )

            // Subtle grid dots
            let spacing: CGFloat = 28
            for x in stride(from: CGFloat(0), to: w, by: spacing) {
                for y in stride(from: CGFloat(0), to: h, by: spacing) {
                    context.fill(
                        Path(ellipseIn: CGRect(x: x - 0.7, y: y - 0.7, width: 1.4, height: 1.4)),
                        with: .color(.white.opacity(0.05))
                    )
                }
            }

            // Draw captured thumbnails at their angular positions
            let pixelsPerRadian = w / cameraFOV
            let currentYaw = CGFloat(session.currentCameraYaw)
            let thumbWidth = w * 0.65   // each thumb covers roughly one camera FOV width
            let thumbHeight = h * 0.85

            for thumb in session.panoramaThumbnails {
                let relAngle = normalizeAngle(CGFloat(thumb.yaw) - currentYaw)

                // Skip thumbnails far behind camera (> ±160°)
                if abs(relAngle) > 2.8 { continue }

                let centerX = w / 2 + relAngle * pixelsPerRadian

                let drawRect = CGRect(
                    x: centerX - thumbWidth / 2,
                    y: (h - thumbHeight) / 2,
                    width: thumbWidth,
                    height: thumbHeight
                )

                // Only draw if at least partially on screen
                guard drawRect.maxX > 0 && drawRect.minX < w else { continue }

                // Edge fade: opacity decreases at the edges for seamless blending
                let edgeFade = min(1.0, max(0.0, (2.2 - abs(relAngle)) / 1.0))
                context.opacity = edgeFade
                context.draw(
                    Image(decorative: thumb.image, scale: 1),
                    in: drawRect
                )
                context.opacity = 1.0
            }

            // Clear center window — transparent so ARView shows through
            context.blendMode = .clear
            context.fill(
                Path(roundedRect: centerRect, cornerRadius: 14),
                with: .color(.white)
            )
        }
    }

    /// Normalize angle difference to [-π, +π]
    private func normalizeAngle(_ angle: CGFloat) -> CGFloat {
        var a = angle
        while a > .pi { a -= 2 * .pi }
        while a < -.pi { a += 2 * .pi }
        return a
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
            if !session.panoramaThumbnails.isEmpty {
                Text("\(session.panoramaThumbnails.count) frames")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
        .padding(12)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
