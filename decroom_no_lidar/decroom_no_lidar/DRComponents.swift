//
//  DRComponents.swift
//  decroom_no_lidar
//
//  Shared SwiftUI primitives ported from
//  /tmp/decroom_design/decroom/project/decroom/shared.jsx.
//

import SwiftUI

// MARK: - Avatar

/// Initial-based gradient circle. Hue param picks the gradient sweep.
struct DRAvatar: View {
    let name: String
    var size: CGFloat = 40
    var hue: Double = 275           // 0..360
    var ring: Bool = false

    var body: some View {
        let initials = name
            .split(separator: " ")
            .compactMap { $0.first }
            .prefix(2)
            .map(String.init)
            .joined()
            .uppercased()

        ZStack {
            LinearGradient(
                colors: [
                    Color(hue: hue / 360, saturation: 0.45, brightness: 0.85),
                    Color(hue: ((hue + 30).truncatingRemainder(dividingBy: 360)) / 360, saturation: 0.65, brightness: 0.55)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .clipShape(Circle())

            Text(initials)
                .font(DR.sans(size * 0.38, weight: .semibold))
                .foregroundStyle(.white)
                .tracking(0.4)
        }
        .frame(width: size, height: size)
        .overlay(
            ring ? Circle()
                .stroke(DR.indigo, lineWidth: 2)
                .padding(-3)
                : nil
        )
        .background(
            ring ? Circle()
                .fill(DR.paper)
                .padding(-5)
                : nil
        )
    }
}

// MARK: - Top bar

struct DRTopBar<Right: View>: View {
    let title: String
    var onBack: (() -> Void)? = nil
    @ViewBuilder var right: Right

    init(title: String, onBack: (() -> Void)? = nil, @ViewBuilder right: () -> Right = { EmptyView() }) {
        self.title = title
        self.onBack = onBack
        self.right = right()
    }

    var body: some View {
        ZStack {
            Text(title)
                .font(DR.sans(15, weight: .semibold))
                .foregroundStyle(DR.ink)

            HStack {
                if let onBack {
                    Button(action: onBack) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundStyle(DR.ink2)
                            .frame(width: 36, height: 36)
                            .background(DR.cream2, in: RoundedRectangle(cornerRadius: 12))
                    }
                } else {
                    Color.clear.frame(width: 36, height: 36)
                }
                Spacer()
                right
            }
        }
        .frame(height: 52)
        .padding(.horizontal, 18)
    }
}

// MARK: - Chip

struct DRChip: View {
    let label: String
    var active: Bool = false
    var icon: String? = nil
    var onTap: () -> Void = {}

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 6) {
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 12, weight: .medium))
                }
                Text(label)
                    .font(DR.sans(12.5, weight: .medium))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 7)
            .background(active ? DR.ink : Color.clear)
            .foregroundStyle(active ? DR.cream : DR.ink2)
            .clipShape(Capsule())
            .overlay(
                Capsule().stroke(active ? DR.ink : DR.line, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Section header

struct DRSectionHeader: View {
    let title: String
    var action: String? = nil
    var onAction: (() -> Void)? = nil

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
                .font(DR.serif(20))
                .foregroundStyle(DR.ink)
                .tracking(-0.2)
            Spacer()
            if let action {
                Button(action: { onAction?() }) {
                    Text("\(action) →")
                        .font(DR.sans(12, weight: .medium))
                        .foregroundStyle(DR.indigo)
                }
            }
        }
        .padding(.top, 20)
        .padding(.bottom, 10)
    }
}

// MARK: - Pill stat

struct DRPillStat: View {
    enum Tone { case neutral, indigo, coral, sage }
    let icon: String?
    let label: String
    let value: String
    var tone: Tone = .neutral

    private var bg: Color {
        switch tone {
        case .neutral: return DR.cream2
        case .indigo:  return DR.indigoTint
        case .coral:   return DR.coralBg
        case .sage:    return DR.sageBg
        }
    }
    private var fg: Color {
        switch tone {
        case .neutral: return DR.ink2
        case .indigo:  return DR.indigo
        case .coral:   return DR.coral
        case .sage:    return DR.sageText
        }
    }

    var body: some View {
        HStack(spacing: 8) {
            if let icon {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .medium))
            }
            Text(label.uppercased())
                .font(DR.mono(11, weight: .medium))
                .tracking(0.8)
                .opacity(0.7)
            Spacer()
            Text(value)
                .font(DR.sans(14, weight: .semibold))
        }
        .foregroundStyle(fg)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(bg, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

// MARK: - Stat card

struct DRStatCard: View {
    let label: String
    let value: String
    let sub: String
    var dot: Color = DR.indigo

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 5) {
                Circle().fill(dot).frame(width: 5, height: 5)
                Text(label.uppercased())
                    .font(DR.mono(10, weight: .medium))
                    .tracking(1.0)
                    .foregroundStyle(DR.ink3)
            }
            Text(value)
                .font(DR.serif(24))
                .foregroundStyle(DR.ink)
                .padding(.top, 1)
            Text(sub)
                .font(DR.sans(10.5))
                .foregroundStyle(DR.ink3)
                .padding(.top, 1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 11)
        .background(DR.paper, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(DR.line2, lineWidth: 1)
        )
    }
}

// MARK: - Paper card container

struct DRPaperCard<Content: View>: View {
    var corner: CGFloat = 14
    @ViewBuilder var content: Content

    var body: some View {
        content
            .background(DR.paper, in: RoundedRectangle(cornerRadius: corner, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: corner, style: .continuous)
                    .stroke(DR.line2, lineWidth: 1)
            )
    }
}

// MARK: - Eyebrow text

struct DREyebrow: View {
    let text: String
    var color: Color = DR.ink3
    var body: some View {
        Text(text.uppercased())
            .font(DR.mono(10.5, weight: .medium))
            .tracking(2.4)
            .foregroundStyle(color)
    }
}

// MARK: - Toggle switch (cream paper, ink active)

struct DRToggle: View {
    @Binding var isOn: Bool
    var body: some View {
        Button { isOn.toggle() } label: {
            ZStack(alignment: isOn ? .trailing : .leading) {
                Capsule()
                    .fill(isOn ? DR.ink : DR.line)
                    .frame(width: 42, height: 24)
                Circle()
                    .fill(.white)
                    .frame(width: 20, height: 20)
                    .padding(.horizontal, 2)
                    .shadow(color: .black.opacity(0.18), radius: 1.5, y: 1)
            }
            .animation(.spring(response: 0.28, dampingFraction: 0.85), value: isOn)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Backdrop-blur tinted pill

/// Tinted ultraThinMaterial pill matching CSS `backdropFilter: blur(8px)`
/// over a translucent white/colored fill.
struct DRBlurPill<Content: View>: View {
    let tint: Color
    let opacity: Double
    @ViewBuilder var content: Content

    init(tint: Color = .white, opacity: Double = 0.85, @ViewBuilder content: () -> Content) {
        self.tint = tint
        self.opacity = opacity
        self.content = content()
    }

    var body: some View {
        content
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                ZStack {
                    Capsule().fill(tint.opacity(opacity))
                    Capsule().fill(.ultraThinMaterial)
                }
            )
    }
}
