//
//  ScanGuidanceOverlayView.swift
//  decroom_no_lidar
//
//  Created by Jae Jeon on 20/02/2026.
//

import SwiftUI

/// Screen-space overlay showing:
/// 1. A prominent step-by-step instruction banner at the top
/// 2. A compact progress checklist (top-right)
/// 3. A directional chevron pointing toward uncovered areas
struct ScanGuidanceOverlayView: View {
    let capturedCount: Int
    let minimumShots: Int
    let isWalkedAround: Bool
    let isMultipleAngles: Bool
    let isCapturedDetail: Bool
    let isReadyToUpload: Bool

    /// Angle difference (radians) from current camera yaw to the center of the
    /// largest uncovered arc.  Negative = turn left, positive = turn right.
    /// Nil when no captures yet or coverage is complete.
    let turnAngle: Float?

    /// Size of the largest yaw gap in radians.
    let largestGap: Float
    let dynamicPrompt: String?
    private let angleDeadZone: Float = 0.25  // ~14°

    // MARK: - Scanning Phase

    private enum ScanPhase {
        case start          // 0 captures
        case firstSpot      // 1-2 captures, haven't moved yet
        case keepTurning    // at first spot, need more angles
        case moveSpots      // enough angles at first spot, need to walk
        case moreAngles     // at new spot, need more directions
        case getDetail      // good spread, need close-ups of furniture
        case ready          // all checks passed
    }

    private var phase: ScanPhase {
        if isReadyToUpload { return .ready }
        if capturedCount == 0 { return .start }

        let hasEnoughPhotos = capturedCount >= minimumShots
        let hasMoved = isWalkedAround
        let hasAngles = isMultipleAngles
        let hasDetail = isCapturedDetail

        if !hasMoved && capturedCount <= 2 {
            return .firstSpot
        }
        if !hasAngles {
            return hasMoved ? .moreAngles : .keepTurning
        }
        if !hasMoved {
            return .moveSpots
        }
        if !hasDetail {
            return .getDetail
        }
        if !hasEnoughPhotos {
            return .moreAngles
        }
        return .ready
    }

    // MARK: - Body

    var body: some View {
        GeometryReader { geo in
            // TOP: Step-by-step instruction banner
            instructionBanner
                .frame(maxWidth: geo.size.width - 32)
                .position(x: geo.size.width / 2, y: 60)

            // TOP-RIGHT: Compact progress
            progressBadge
                .position(x: geo.size.width - 50, y: 130)

            // Directional chevron — only show when there's a genuinely large
            // uncovered gap (>90°). Disappears once coverage is reasonably spread.
            if let angle = turnAngle,
               capturedCount > 0,
               largestGap > Float.pi / 2  // only for gaps > 90°
            {
                directionChevron(angle: angle, geo: geo)

                if abs(angle) <= angleDeadZone {
                    goodAngleBadge(geo: geo)
                }
            }
        }
        .allowsHitTesting(false)
    }

    // MARK: - Instruction Banner

    private var instructionBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: instructionIcon)
                .font(.system(size: 22))
                .foregroundStyle(instructionIconColor)

            VStack(alignment: .leading, spacing: 2) {
                Text(instructionTitle)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.white)
                Text(instructionDetail)
                    .font(.system(size: 11))
                    .foregroundStyle(.white.opacity(0.7))
                    .lineLimit(2)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 12))
    }

    private var instructionIcon: String {
        switch phase {
        case .start:       return "camera.viewfinder"
        case .firstSpot:   return "arrow.turn.right.up"
        case .keepTurning: return "arrow.triangle.2.circlepath"
        case .moveSpots:   return "figure.walk"
        case .moreAngles:  return "arrow.triangle.2.circlepath"
        case .getDetail:   return "plus.magnifyingglass"
        case .ready:       return "checkmark.circle.fill"
        }
    }

    private var instructionIconColor: Color {
        switch phase {
        case .ready: return .green
        default:     return .cyan
        }
    }

    private var instructionTitle: String {
        switch phase {
        case .start:
            return "Step 1: Stand near center"
        case .firstSpot:
            return "Step 1: Rotate in place"
        case .keepTurning:
            return "Step 1: Continue 360° sweep"
        case .moveSpots:
            return "Step 2 (Recommended): Move 1–2m"
        case .moreAngles:
            return "Step 2: Capture extra angles"
        case .getDetail:
            return "Step 3: Get closer to furniture"
        case .ready:
            return "Ready to upload!"
        }
    }

    private var instructionDetail: String {
        if let dynamicPrompt, !dynamicPrompt.isEmpty, phase != .ready {
            return dynamicPrompt
        }
        switch phase {
        case .start:
            return "Take 10–14 photos while rotating clockwise. Capture about every 25–35°."
        case .firstSpot:
            return "Stay in place. Turn ~30° each shot until you've covered a full 360°."
        case .keepTurning:
            return "Follow the side arrow for the biggest missing direction. Keep angle gaps under 90°."
        case .moveSpots:
            return "Optional but improves 3D accuracy: move 1–2m and take 2–4 additional photos."
        case .moreAngles:
            return "From this offset position, rotate and capture missing directions."
        case .getDetail:
            return "Point at furniture, shelves, or textured surfaces from 1–2m away and capture."
        case .ready:
            return "Good coverage from multiple positions! Tap Upload Now to process your room."
        }
    }

    // MARK: - Progress Badge

    private var progressBadge: some View {
        HStack(spacing: 4) {
            Image(systemName: "camera.fill")
                .font(.system(size: 10))
            Text("\(capturedCount)")
                .font(.system(size: 13, weight: .bold, design: .rounded))
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.black.opacity(0.5), in: Capsule())
    }

    // MARK: - Directional Chevron

    @ViewBuilder
    private func directionChevron(angle: Float, geo: GeometryProxy) -> some View {
        if abs(angle) > angleDeadZone {
            let turnRight = angle > 0
            let chevronName = turnRight ? "chevron.right" : "chevron.left"
            let xPos: CGFloat = turnRight ? geo.size.width - 24 : 24
            let yPos: CGFloat = geo.size.height / 2

            Image(systemName: chevronName)
                .font(.system(size: 30, weight: .bold))
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.6), radius: 3, x: 0, y: 0)
                .position(x: xPos, y: yPos)
                .opacity(Double(min(1.0, abs(angle) / 0.5)))
        }
    }

    @ViewBuilder
    private func goodAngleBadge(geo: GeometryProxy) -> some View {
        Label("Good angle", systemImage: "checkmark.circle.fill")
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(.green.opacity(0.85), in: Capsule())
            .position(x: geo.size.width / 2, y: 155)
    }
}
