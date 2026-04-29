//
//  RoomPlaceholderView.swift
//  decroom_no_lidar
//
//  SwiftUI port of the JSX `RoomPlaceholder` from
//  /tmp/decroom_design/decroom/project/decroom/shared.jsx.
//  Generates a seed-stable gradient + isometric room hint for cards
//  that don't have a real 3D snapshot (Explore feed, Inspiration row,
//  Profile-of-others). Real saved rooms still use RoomSnapshotCache.
//

import SwiftUI

struct RoomPlaceholderView: View {
    let seed: Int
    var label: String? = nil
    var vibe: RoomVibe = .warm

    var body: some View {
        GeometryReader { proxy in
            let palette = vibePalette(vibe)
            let angle = Angle.degrees(Double(Int(prng(1) * 180)))
            let horizon = 0.55 + prng(2) * 0.15

            ZStack {
                // Vibe gradient
                LinearGradient(
                    stops: [
                        .init(color: palette[0], location: 0.0),
                        .init(color: palette[1], location: horizon),
                        .init(color: palette[2], location: 1.0)
                    ],
                    startPoint: gradientStart(for: angle),
                    endPoint:   gradientEnd(for: angle)
                )

                // Isometric room hint
                IsoRoomHint(seed: seed)
                    .blendMode(.multiply)
                    .opacity(0.35)

                // 135° hairline stripes
                Canvas { ctx, size in
                    var path = Path()
                    let step: CGFloat = 12
                    let line: CGFloat = 2
                    var d: CGFloat = -size.height
                    while d < size.width + size.height {
                        path.move(to: CGPoint(x: d, y: 0))
                        path.addLine(to: CGPoint(x: d + size.height, y: size.height))
                        d += step
                    }
                    ctx.stroke(path, with: .color(.white.opacity(0.03)), lineWidth: line)
                }
                .allowsHitTesting(false)

                // Mono label bottom-left
                if let label, !label.isEmpty {
                    VStack {
                        Spacer()
                        HStack {
                            Text(label.uppercased())
                                .font(DR.mono(10, weight: .medium))
                                .tracking(1.6)
                                .foregroundStyle(Color.white.opacity(0.9))
                                .shadow(color: .black.opacity(0.25), radius: 1, y: 1)
                            Spacer()
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.bottom, 10)
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
            .clipped()
        }
    }

    // PRNG matching JSX: x = sin(seed * 9301 + n * 49297) * 233280; return x - floor(x)
    private func prng(_ n: Int) -> Double {
        let x = sin(Double(seed) * 9301.0 + Double(n) * 49297.0) * 233280.0
        return x - floor(x)
    }

    private func gradientStart(for angle: Angle) -> UnitPoint {
        let r = angle.radians
        return UnitPoint(x: 0.5 - 0.5 * sin(r), y: 0.5 + 0.5 * cos(r))
    }
    private func gradientEnd(for angle: Angle) -> UnitPoint {
        let r = angle.radians
        return UnitPoint(x: 0.5 + 0.5 * sin(r), y: 0.5 - 0.5 * cos(r))
    }
}

// MARK: - Vibe palettes (precomputed sRGB from oklch)

private func vibePalette(_ vibe: RoomVibe) -> [Color] {
    switch vibe {
    case .warm: return [
        Color(red: 0.973, green: 0.685, blue: 0.466),  // oklch(0.78 0.09 55)
        Color(red: 0.957, green: 0.776, blue: 0.643),  // oklch(0.85 0.06 35)
        Color(red: 0.870, green: 0.555, blue: 0.443),  // oklch(0.70 0.08 20)
    ]
    case .cool: return [
        Color(red: 0.540, green: 0.695, blue: 0.886),  // oklch(0.75 0.08 240)
        Color(red: 0.692, green: 0.769, blue: 0.916),  // oklch(0.82 0.06 260)
        Color(red: 0.620, green: 0.601, blue: 0.829),  // oklch(0.70 0.07 290)
    ]
    case .sage: return [
        Color(red: 0.679, green: 0.840, blue: 0.726),  // oklch(0.80 0.06 150)
        Color(red: 0.795, green: 0.886, blue: 0.819),  // oklch(0.86 0.04 140)
        Color(red: 0.546, green: 0.766, blue: 0.622),  // oklch(0.72 0.06 165)
    ]
    case .lilac: return [
        Color(red: 0.703, green: 0.687, blue: 0.927),  // oklch(0.78 0.08 290)
        Color(red: 0.808, green: 0.788, blue: 0.927),  // oklch(0.85 0.06 280)
        Color(red: 0.626, green: 0.586, blue: 0.880),  // oklch(0.72 0.08 275)
    ]
    case .clay: return [
        Color(red: 0.913, green: 0.652, blue: 0.461),  // oklch(0.75 0.08 45)
        Color(red: 0.910, green: 0.752, blue: 0.602),  // oklch(0.82 0.05 60)
        Color(red: 0.851, green: 0.486, blue: 0.346),  // oklch(0.68 0.07 30)
    ]
    case .mono: return [
        Color(red: 0.792, green: 0.797, blue: 0.825),  // oklch(0.85 0.01 275)
        Color(red: 0.683, green: 0.689, blue: 0.722),  // oklch(0.78 0.01 275)
        Color(red: 0.876, green: 0.880, blue: 0.903),  // oklch(0.92 0.01 275)
    ]
    }
}

// MARK: - Iso room hint

private struct IsoRoomHint: View {
    let seed: Int

    var body: some View {
        Canvas { ctx, size in
            // Map the JSX 300x200 viewBox into the actual rect (slice)
            let scaleX = size.width / 300.0
            let scaleY = size.height / 200.0
            let scale = max(scaleX, scaleY)
            let dx = (size.width - 300 * scale) / 2
            let dy = (size.height - 200 * scale) / 2

            func pt(_ x: CGFloat, _ y: CGFloat) -> CGPoint {
                CGPoint(x: x * scale + dx, y: y * scale + dy)
            }

            // Floor diamond gradient (top transparent → bottom black 0.25)
            var floor = Path()
            floor.move(to: pt(40, 180))
            floor.addLine(to: pt(150, 130))
            floor.addLine(to: pt(260, 180))
            floor.addLine(to: pt(150, 230))
            floor.closeSubpath()
            let g = Gradient(stops: [
                .init(color: .black.opacity(0.0), location: 0.0),
                .init(color: .black.opacity(0.25), location: 1.0)
            ])
            ctx.fill(floor, with: .linearGradient(g, startPoint: pt(150, 130), endPoint: pt(150, 230)))

            // Back wall
            var back = Path()
            back.move(to: pt(40, 180))
            back.addLine(to: pt(40, 60))
            back.addLine(to: pt(150, 10))
            back.addLine(to: pt(150, 130))
            back.closeSubpath()
            ctx.fill(back, with: .color(.black.opacity(0.08)))

            // Side wall
            var side = Path()
            side.move(to: pt(260, 180))
            side.addLine(to: pt(260, 60))
            side.addLine(to: pt(150, 10))
            side.addLine(to: pt(150, 130))
            side.closeSubpath()
            ctx.fill(side, with: .color(.black.opacity(0.04)))

            // Window
            var win = Path()
            win.addRect(CGRect(x: 70 * scale + dx, y: 80 * scale + dy, width: 40 * scale, height: 50 * scale))
            ctx.fill(win, with: .color(.white.opacity(0.30)))

            // Sofa hint (seed-positioned)
            let r3 = randAt(3), r4 = randAt(4)
            let sofaX = 100 + r3 * 40
            let sofaY = 140 + r4 * 10
            let sofaRect = CGRect(x: sofaX * scale + dx, y: sofaY * scale + dy, width: 60 * scale, height: 20 * scale)
            ctx.fill(Path(roundedRect: sofaRect, cornerRadius: 4 * scale), with: .color(.black.opacity(0.12)))
        }
        .allowsHitTesting(false)
    }

    private func randAt(_ n: Int) -> CGFloat {
        let x = sin(Double(seed) * 9301.0 + Double(n) * 49297.0) * 233280.0
        return CGFloat(x - floor(x))
    }
}

#Preview {
    VStack(spacing: 12) {
        RoomPlaceholderView(seed: 1, label: "MODEL #001", vibe: .warm)
            .frame(height: 220)
            .clipShape(RoundedRectangle(cornerRadius: 16))
        HStack(spacing: 8) {
            RoomPlaceholderView(seed: 2, label: "PUBLISHED", vibe: .sage)
                .frame(height: 110)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            RoomPlaceholderView(seed: 3, label: "DRAFT", vibe: .clay)
                .frame(height: 110)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }
    .padding()
    .background(DR.cream)
}
