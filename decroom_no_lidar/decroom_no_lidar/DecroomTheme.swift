//
//  DecroomTheme.swift
//  decroom_no_lidar
//
//  Design system tokens for the cream-paper / indigo / coral aesthetic
//  ported from /tmp/decroom_design/decroom/project/Decroom.html.
//  All colors are precomputed sRGB equivalents of CSS oklch tokens
//  (CSS Color Level 4 / Björn Ottosson oklab matrices).
//

import SwiftUI

enum DR {

    // MARK: - Palette

    // Neutrals
    static let ink         = Color(red: 0.073, green: 0.083, blue: 0.131)  // oklch(0.20 0.025 275)
    static let ink2        = Color(red: 0.227, green: 0.237, blue: 0.282)  // oklch(0.36 0.02  275)
    static let ink3        = Color(red: 0.434, green: 0.443, blue: 0.481)  // oklch(0.55 0.015 275)
    static let line        = Color(red: 0.858, green: 0.868, blue: 0.911)  // oklch(0.90 0.015 275)
    static let line2       = Color(red: 0.913, green: 0.921, blue: 0.949)  // oklch(0.94 0.01  275)
    static let cream       = Color(red: 0.996, green: 0.971, blue: 0.931)  // oklch(0.98 0.015  80)
    static let cream2      = Color(red: 0.976, green: 0.944, blue: 0.891)  // oklch(0.96 0.02   80)
    static let paper       = Color.white                                   // oklch(1 0 0)

    // Accents
    static let indigo      = Color(red: 0.206, green: 0.260, blue: 0.596)  // oklch(0.42 0.14 272)
    static let indigo2     = Color(red: 0.293, green: 0.365, blue: 0.766)  // oklch(0.52 0.16 272)
    static let indigoSoft  = Color(red: 0.798, green: 0.836, blue: 1.000)  // oklch(0.88 0.06 275)
    static let indigoTint  = Color(red: 0.910, green: 0.931, blue: 1.000)  // oklch(0.95 0.03 275)
    static let lilac       = Color(red: 0.714, green: 0.682, blue: 0.931)  // oklch(0.78 0.09 290)
    static let coral       = Color(red: 0.965, green: 0.426, blue: 0.402)  // oklch(0.70 0.17  25)
    static let sage        = Color(red: 0.602, green: 0.766, blue: 0.654)  // oklch(0.78 0.06 155)
    static let butter      = Color(red: 0.979, green: 0.855, blue: 0.598)  // oklch(0.90 0.09  85)

    // Semantic helpers
    static let redLine     = Color(red: 0.992, green: 0.788, blue: 0.767)  // delete-btn border
    static let redBg       = Color(red: 1.000, green: 0.956, blue: 0.948)  // delete-btn bg
    static let redText     = Color(red: 0.688, green: 0.165, blue: 0.175)  // delete-btn text
    static let sageBg      = Color(red: 0.857, green: 0.969, blue: 0.890)  // pro-badge bg
    static let sageText    = Color(red: 0.170, green: 0.385, blue: 0.256)  // pro-badge text
    static let coralBg     = Color(red: 1.000, green: 0.897, blue: 0.883)  // coral chip bg

    // Profile gradient stops
    static let profTop     = Color(red: 0.764, green: 0.737, blue: 0.961)  // oklch(0.82 0.08 290)
    static let profBot     = Color(red: 0.445, green: 0.498, blue: 0.807)  // oklch(0.62 0.12 275)

    // MARK: - Fonts (bundled — see decroom-no-lidar-Info.plist UIAppFonts)

    /// Display serif. Italic via `serifItalic`.
    static func serif(_ size: CGFloat) -> Font {
        Font.custom("InstrumentSerif-Regular", size: size)
    }

    static func serifItalic(_ size: CGFloat) -> Font {
        Font.custom("InstrumentSerif-Italic", size: size)
    }

    /// Sans UI. Variable font: weight axis applied via `.weight(_:)`.
    static func sans(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        Font.custom("InstrumentSans-Regular", size: size).weight(weight)
    }

    /// Monospaced metadata.
    static func mono(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        if weight == .medium {
            return Font.custom("JetBrainsMono-Medium", size: size)
        }
        return Font.custom("JetBrainsMono-Regular", size: size)
    }

    // MARK: - Shadows (CSS px → SwiftUI radius ≈ blur/2)

    /// `--shadow-sm`: subtle paper lift.
    static func shadowSmall<V: View>(on v: V) -> some View {
        v
            .shadow(color: ink.opacity(0.06), radius: 1, x: 0, y: 1)
            .shadow(color: ink.opacity(0.04), radius: 3, x: 0, y: 2)
    }

    /// `--shadow-md`: dropdown / featured card.
    static func shadowMedium<V: View>(on v: V) -> some View {
        v
            .shadow(color: ink.opacity(0.08), radius: 4, x: 0, y: 2)
            .shadow(color: ink.opacity(0.10), radius: 16, x: 0, y: 12)
    }

    /// `--shadow-lg`: elevated overlays / drag-lifted card.
    static func shadowLarge<V: View>(on v: V) -> some View {
        v
            .shadow(color: ink.opacity(0.12), radius: 12, x: 0, y: 8)
            .shadow(color: ink.opacity(0.14), radius: 30, x: 0, y: 30)
    }
}

// MARK: - Press feedback

extension View {
    /// Spring-animated 0.97 scale on press. Used on tappable cards.
    func drPressable() -> some View {
        self.buttonStyle(DRPressableButtonStyle())
    }
}

struct DRPressableButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: configuration.isPressed)
    }
}

// MARK: - Debug font registration check

#if DEBUG
@MainActor
enum DRFontDebug {
    private static var didRun = false
    static func logRegisteredFonts() {
        guard !didRun else { return }
        didRun = true
        let needles = ["Instrument", "JetBrains"]
        let names = UIFont.familyNames
            .flatMap { UIFont.fontNames(forFamilyName: $0) }
            .filter { name in needles.contains(where: { name.contains($0) }) }
            .sorted()
        print("[DR] Registered custom fonts: \(names)")
    }
}
#endif
