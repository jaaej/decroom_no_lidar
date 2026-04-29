//
//  SettingsView.swift
//  decroom_no_lidar
//
//  Settings screen ported from screens-settings.jsx. Toggle/select state
//  persists via @AppStorage; nothing wires to backend.
//

import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss

    @AppStorage("home.showCardLikes")  private var showCardLikes: Bool   = true
    @AppStorage("notif.likes")          private var notifLikes: Bool      = true
    @AppStorage("notif.follows")        private var notifFollows: Bool    = true
    @AppStorage("notif.weekly")         private var notifWeekly: Bool     = false
    @AppStorage("notif.marketing")      private var notifMarketing: Bool  = false
    @AppStorage("priv.private")         private var privateProfile: Bool  = false
    @AppStorage("priv.leaderboard")     private var showOnLeaderboard: Bool = true
    @AppStorage("scan.units")           private var unitsRaw: String      = "metric"
    @AppStorage("scan.quality")         private var qualityRaw: String    = "high"

    var body: some View {
        ZStack(alignment: .top) {
            DR.cream.ignoresSafeArea()

            VStack(spacing: 0) {
                topBar
                ScrollView(showsIndicators: false) {
                    VStack(spacing: 0) {
                        profileCard
                            .padding(.bottom, 20)

                        homeGroup
                            .padding(.bottom, 20)
                        notifGroup
                            .padding(.bottom, 20)
                        privacyGroup
                            .padding(.bottom, 20)
                        captureGroup
                            .padding(.bottom, 20)
                        supportGroup
                            .padding(.bottom, 20)
                        actionRow
                            .padding(.bottom, 22)
                        footer
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 14)
                    .padding(.bottom, 40)
                }
            }
        }
        .toolbar(.hidden, for: .navigationBar)
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(alignment: .center, spacing: 14) {
            Button { dismiss() } label: {
                Image(systemName: "chevron.left")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(DR.ink2)
                    .frame(width: 36, height: 36)
                    .background(DR.paper, in: RoundedRectangle(cornerRadius: 10))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10).stroke(DR.line, lineWidth: 1)
                    )
            }
            VStack(alignment: .leading, spacing: 2) {
                DREyebrow(text: "ACCOUNT")
                Text("Settings")
                    .font(DR.serif(24))
                    .foregroundStyle(DR.ink)
                    .tracking(-0.2)
            }
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.top, 54)
        .padding(.bottom, 14)
        .background(
            VStack(spacing: 0) {
                DR.cream
                Rectangle().fill(DR.line2).frame(height: 1)
            }
            .ignoresSafeArea(edges: .top)
        )
    }

    // MARK: - Profile card

    private var profileCard: some View {
        HStack(spacing: 12) {
            DRAvatar(name: MockData.profile.name, size: 48, hue: 285)
            VStack(alignment: .leading, spacing: 2) {
                Text(MockData.profile.name)
                    .font(DR.sans(14, weight: .semibold))
                    .foregroundStyle(DR.ink)
                Text("\(MockData.profile.handle) · jae@decroom.app")
                    .font(DR.sans(11.5))
                    .foregroundStyle(DR.ink3)
            }
            Spacer()
            Button {} label: {
                Text("Edit")
                    .font(DR.sans(12, weight: .medium))
                    .foregroundStyle(DR.ink2)
                    .padding(.horizontal, 12).padding(.vertical, 7)
                    .overlay(Capsule().stroke(DR.line, lineWidth: 1))
            }
            .buttonStyle(.plain)
        }
        .padding(14)
        .background(DR.paper, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(DR.line2, lineWidth: 1)
        )
    }

    // MARK: - Groups

    private var homeGroup: some View {
        SettingsGroup(label: "HOME SCREEN",
                      hint: "Arrange blocks and hide sections directly on the home screen — tap the pencil icon there.") {
            ToggleRow(title: "Like counts on room cards",
                      desc: "Show ♥ under each room thumbnail",
                      value: $showCardLikes)
        }
    }

    private var notifGroup: some View {
        SettingsGroup(label: "NOTIFICATIONS") {
            ToggleRow(title: "Likes", desc: "When someone likes your room", value: $notifLikes)
            Divider().background(DR.line2)
            ToggleRow(title: "New followers", desc: "When someone follows you", value: $notifFollows)
            Divider().background(DR.line2)
            ToggleRow(title: "Weekly digest", desc: "Top rooms in your feed, Sundays", value: $notifWeekly)
            Divider().background(DR.line2)
            ToggleRow(title: "Product updates", desc: "New features & tips", value: $notifMarketing)
        }
    }

    private var privacyGroup: some View {
        SettingsGroup(label: "PRIVACY") {
            ToggleRow(title: "Private profile", desc: "Only followers can see your rooms", value: $privateProfile)
            Divider().background(DR.line2)
            ToggleRow(title: "Show on leaderboard", desc: "Appear in the public Top rankings", value: $showOnLeaderboard)
        }
    }

    private var captureGroup: some View {
        SettingsGroup(label: "CAPTURE & UNITS") {
            SelectRow(title: "Units",
                      options: [("metric", "Metric (m²)"), ("imperial", "Imperial (ft²)")],
                      value: $unitsRaw)
            Divider().background(DR.line2)
            SelectRow(title: "Scan quality",
                      options: [("eco", "Eco · faster, smaller files"),
                                ("high", "High · best detail"),
                                ("pro", "Pro · maximum precision")],
                      value: $qualityRaw)
        }
    }

    private var supportGroup: some View {
        SettingsGroup(label: "SUPPORT") {
            LinkRow(title: "Help center")
            Divider().background(DR.line2)
            LinkRow(title: "Send feedback")
            Divider().background(DR.line2)
            LinkRow(title: "Terms of service")
            Divider().background(DR.line2)
            LinkRow(title: "Privacy policy")
        }
    }

    // MARK: - Action row

    private var actionRow: some View {
        HStack(spacing: 8) {
            Button {} label: {
                Text("Log out")
                    .font(DR.sans(13, weight: .medium))
                    .foregroundStyle(DR.ink2)
                    .frame(maxWidth: .infinity, minHeight: 44)
                    .background(DR.paper, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(DR.line, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)

            Button {} label: {
                Text("Delete account")
                    .font(DR.sans(13, weight: .medium))
                    .foregroundStyle(DR.redText)
                    .frame(maxWidth: .infinity, minHeight: 44)
                    .background(DR.redBg, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(DR.redLine, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Footer

    private var footer: some View {
        Text("DECROOM · v0.2.4")
            .font(DR.mono(10.5, weight: .medium))
            .tracking(1.4)
            .foregroundStyle(DR.ink3)
            .frame(maxWidth: .infinity)
    }
}

// MARK: - Subviews

private struct SettingsGroup<Content: View>: View {
    let label: String
    var hint: String? = nil
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            DREyebrow(text: label)
                .padding(.leading, 4)
            if let hint {
                Text(hint)
                    .font(DR.sans(11))
                    .foregroundStyle(DR.ink3)
                    .lineSpacing(2)
                    .padding(.leading, 4)
                    .padding(.bottom, 2)
            }
            VStack(spacing: 0) {
                content
            }
            .background(DR.paper, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(DR.line2, lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
    }
}

private struct ToggleRow: View {
    let title: String
    var desc: String? = nil
    @Binding var value: Bool

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(DR.sans(13, weight: .medium))
                    .foregroundStyle(DR.ink)
                if let desc {
                    Text(desc)
                        .font(DR.sans(11))
                        .foregroundStyle(DR.ink3)
                        .lineSpacing(1.5)
                }
            }
            Spacer()
            DRToggle(isOn: $value)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }
}

private struct SelectRow: View {
    let title: String
    let options: [(id: String, label: String)]
    @Binding var value: String
    @State private var open: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            Button {
                withAnimation(.easeOut(duration: 0.18)) { open.toggle() }
            } label: {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(title)
                            .font(DR.sans(13, weight: .medium))
                            .foregroundStyle(DR.ink)
                        Text(options.first(where: { $0.id == value })?.label ?? "—")
                            .font(DR.sans(11))
                            .foregroundStyle(DR.ink3)
                    }
                    Spacer()
                    Image(systemName: "chevron.down")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(DR.ink3)
                        .rotationEffect(.degrees(open ? 180 : 0))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if open {
                VStack(spacing: 0) {
                    ForEach(options, id: \.id) { opt in
                        Button {
                            value = opt.id
                            withAnimation(.easeOut(duration: 0.15)) { open = false }
                        } label: {
                            HStack(spacing: 10) {
                                ZStack {
                                    Circle()
                                        .stroke(opt.id == value ? DR.ink : DR.line, lineWidth: 1.5)
                                        .frame(width: 16, height: 16)
                                    if opt.id == value {
                                        Circle().fill(DR.ink).frame(width: 8, height: 8)
                                    }
                                }
                                Text(opt.label)
                                    .font(DR.sans(12.5, weight: opt.id == value ? .semibold : .medium))
                                    .foregroundStyle(opt.id == value ? DR.ink : DR.ink2)
                                Spacer()
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
                .background(DR.cream2)
                .overlay(alignment: .top) {
                    Rectangle().fill(DR.line2).frame(height: 1)
                }
            }
        }
    }
}

private struct LinkRow: View {
    let title: String
    var body: some View {
        HStack {
            Text(title)
                .font(DR.sans(13, weight: .medium))
                .foregroundStyle(DR.ink)
            Spacer()
            Image(systemName: "chevron.right")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(DR.ink3)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 13)
        .contentShape(Rectangle())
    }
}

#Preview {
    NavigationStack { SettingsView() }
}
