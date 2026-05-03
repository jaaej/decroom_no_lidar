//
//  SplashView.swift
//  decroom_no_lidar
//
//  Reskinned launch animation matching the Decroom.html brand-mark:
//  cream paper bg, Instrument Serif "Decroom" wordmark with a coral dot.
//
//  Choreography:
//    0.00s  Cream bg + faint indigo wash
//    0.15s  Indigo hairline draws out under the wordmark
//    0.30s  Letters of "Decroom" rise with stagger (60ms per glyph)
//    0.85s  Tagline fades in
//    1.05s  Coral dot pulse
//    1.70s  Crossfade to MainTabView
//

import SwiftUI

struct SplashView: View {
    var onFinished: () -> Void

    @State private var dotScale: CGFloat = 0
    @State private var lineProgress: CGFloat = 0
    @State private var letterProgress: [CGFloat] = Array(repeating: 0, count: 7)
    @State private var taglineOpacity: Double = 0
    @State private var dotPulse: CGFloat = 1.0
    @State private var rootOpacity: Double = 1.0

    private let glyphs: [String] = ["D", "e", "c", "r", "o", "o", "m"]

    var body: some View {
        ZStack {
            // Cream paper background with faint indigo wash from below
            DR.cream.ignoresSafeArea()
            RadialGradient(
                colors: [DR.indigoSoft.opacity(0.35), .clear],
                center: .init(x: 0.5, y: 1.05),
                startRadius: 60,
                endRadius: 520
            )
            .ignoresSafeArea()
            .blendMode(.multiply)

            VStack(spacing: 18) {
                // Wordmark row
                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    Circle()
                        .fill(DR.coral)
                        .frame(width: 12, height: 12)
                        .scaleEffect(dotScale * dotPulse)
                        .alignmentGuide(.firstTextBaseline) { d in d[VerticalAlignment.center] + 8 }

                    HStack(spacing: 0) {
                        ForEach(Array(glyphs.enumerated()), id: \.offset) { i, g in
                            Text(g)
                                .font(DR.serif(56))
                                .foregroundStyle(DR.ink)
                                .opacity(Double(letterProgress[i]))
                                .offset(y: (1 - letterProgress[i]) * 14)
                        }
                    }
                }

                // Hairline + tagline
                VStack(spacing: 14) {
                    GeometryReader { proxy in
                        let w = proxy.size.width
                        Rectangle()
                            .fill(DR.indigo.opacity(0.65))
                            .frame(width: w * lineProgress, height: 1)
                            .frame(width: w, alignment: .center)
                    }
                    .frame(height: 1)
                    .padding(.horizontal, 80)

                    HStack(spacing: 8) {
                        Text("INTERIOR ARCHIVE")
                            .font(DR.mono(10, weight: .medium))
                            .tracking(2.6)
                            .foregroundStyle(DR.ink3)
                        Circle().fill(DR.ink3).frame(width: 3, height: 3)
                        Text("EST. 2026")
                            .font(DR.mono(10, weight: .medium))
                            .tracking(2.6)
                            .foregroundStyle(DR.ink3)
                    }
                    .opacity(taglineOpacity)
                }
            }
        }
        .opacity(rootOpacity)
        .task {
            #if DEBUG
            DRFontDebug.logRegisteredFonts()
            #endif
            await runChoreography()
        }
    }

    private func runChoreography() async {
        // Coral dot fades in alongside the line draw
        try? await Task.sleep(nanoseconds: 100_000_000)
        withAnimation(.spring(response: 0.5, dampingFraction: 0.7)) {
            dotScale = 1
        }

        try? await Task.sleep(nanoseconds: 50_000_000)
        withAnimation(.easeOut(duration: 0.55)) {
            lineProgress = 1
        }

        // Stagger letters
        try? await Task.sleep(nanoseconds: 150_000_000)
        for i in 0..<glyphs.count {
            withAnimation(.spring(response: 0.55, dampingFraction: 0.8)) {
                letterProgress[i] = 1
            }
            try? await Task.sleep(nanoseconds: 60_000_000)
        }

        // Tagline fade
        try? await Task.sleep(nanoseconds: 120_000_000)
        withAnimation(.easeOut(duration: 0.45)) {
            taglineOpacity = 1
        }

        // Coral dot pulse
        try? await Task.sleep(nanoseconds: 200_000_000)
        withAnimation(.easeInOut(duration: 0.35)) { dotPulse = 1.45 }
        try? await Task.sleep(nanoseconds: 350_000_000)
        withAnimation(.easeInOut(duration: 0.35)) { dotPulse = 1.0 }

        // Hold + fade out
        try? await Task.sleep(nanoseconds: 450_000_000)
        withAnimation(.easeInOut(duration: 0.4)) { rootOpacity = 0 }
        try? await Task.sleep(nanoseconds: 420_000_000)
        onFinished()
    }
}

#Preview {
    SplashView(onFinished: {})
}
