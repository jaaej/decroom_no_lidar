//
//  RoomCardView.swift
//  decroom_no_lidar
//
//  Two card sizes used by Home:
//   - GridRoomCard: 110pt-thumb tile in the 2-col grid (matches `RoomCard` JSX)
//   - FeaturedRoomCard: 220pt full-bleed hero (matches `FeaturedCard` JSX)
//   - NewRoomCard: dashed "Scan new room" tile
//
//  RoomDetailView keeps the dark ink palette but swaps amber→indigo.
//

import SwiftUI

// MARK: - Grid card (110pt thumb)

struct GridRoomCard: View {
    let room: SavedRoom
    let onTap: () -> Void
    var showLikes: Bool = true

    @ObservedObject private var snapshotCache = RoomSnapshotCache.shared

    private let previewHeight: CGFloat = 110

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 0) {
                snapshot
                    .frame(height: previewHeight)
                    .clipped()
                VStack(alignment: .leading, spacing: 2) {
                    Text(room.name)
                        .font(DR.sans(13.5, weight: .semibold))
                        .foregroundStyle(DR.ink)
                        .lineLimit(1)
                    HStack {
                        Text("\(room.payload.objects.count) items")
                            .font(DR.sans(11))
                            .foregroundStyle(DR.ink3)
                        Spacer()
                        if showLikes {
                            HStack(spacing: 3) {
                                Image(systemName: "heart")
                                    .font(.system(size: 10, weight: .regular))
                                Text("\(mockLikes(forRoomId: room.id))")
                                    .font(DR.sans(11))
                            }
                            .foregroundStyle(DR.ink3)
                        }
                    }
                    .padding(.top, 1)
                }
                .padding(.horizontal, 12)
                .padding(.top, 9)
                .padding(.bottom, 11)
            }
        }
        .background(DR.paper, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(DR.line2, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .drPressable()
        .onAppear {
            let w = (UIScreen.main.bounds.width - 44 - 12) / 2
            snapshotCache.ensureSnapshot(for: room, size: CGSize(width: w, height: previewHeight))
        }
    }

    @ViewBuilder
    private var snapshot: some View {
        if let img = snapshotCache.image(for: room, size: CGSize(width: 200, height: previewHeight)) {
            Image(uiImage: img)
                .resizable()
                .aspectRatio(contentMode: .fill)
        } else {
            // Fallback: vibe-tinted placeholder
            RoomPlaceholderView(
                seed: abs(room.id.drStableHash) % 999,
                label: nil,
                vibe: RoomVibe.from(id: room.id)
            )
        }
    }
}

// MARK: - Featured card (220pt hero)

struct FeaturedRoomCard: View {
    let room: SavedRoom
    let onTap: () -> Void

    @ObservedObject private var snapshotCache = RoomSnapshotCache.shared

    private let height: CGFloat = 220

    var body: some View {
        Button(action: onTap) {
            ZStack(alignment: .bottomLeading) {
                snapshot

                LinearGradient(
                    colors: [.clear, DR.ink.opacity(0.65)],
                    startPoint: .center,
                    endPoint: .bottom
                )

                // "Featured" pill (top-left)
                VStack {
                    HStack {
                        DRBlurPill(tint: .white, opacity: 0.85) {
                            HStack(spacing: 6) {
                                Image(systemName: "sparkles")
                                    .font(.system(size: 10, weight: .semibold))
                                Text("FEATURED")
                                    .font(DR.mono(10, weight: .medium))
                                    .tracking(2.0)
                            }
                            .foregroundStyle(DR.ink)
                        }
                        Spacer()
                    }
                    .padding(12)
                    Spacer()
                }

                // Bottom meta
                VStack(alignment: .leading, spacing: 4) {
                    Text(room.name)
                        .font(DR.serif(24))
                        .foregroundStyle(.white)
                        .tracking(-0.2)
                        .lineLimit(1)
                    HStack(spacing: 10) {
                        Text("\(room.payload.objects.count) items")
                            .font(DR.sans(12))
                        Circle().fill(.white.opacity(0.5)).frame(width: 3, height: 3)
                        Text(String(format: "%.1fm × %.1fm", room.payload.envelope.width, room.payload.envelope.length))
                            .font(DR.sans(12))
                        Spacer()
                        HStack(spacing: 4) {
                            Image(systemName: "heart")
                                .font(.system(size: 12))
                            Text("\(mockLikes(forRoomId: room.id))")
                                .font(DR.sans(12))
                        }
                    }
                    .foregroundStyle(.white.opacity(0.9))
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 14)
            }
        }
        .frame(height: height)
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .drPressable()
        .onAppear {
            let w = UIScreen.main.bounds.width - 44
            snapshotCache.ensureSnapshot(for: room, size: CGSize(width: w, height: height))
        }
    }

    @ViewBuilder
    private var snapshot: some View {
        if let img = snapshotCache.image(for: room, size: CGSize(width: 400, height: height)) {
            Image(uiImage: img)
                .resizable()
                .aspectRatio(contentMode: .fill)
        } else {
            RoomPlaceholderView(
                seed: abs(room.id.drStableHash) % 999,
                label: "MODEL #\(String(format: "%03d", abs(room.id.drStableHash) % 1000))",
                vibe: RoomVibe.from(id: room.id)
            )
        }
    }
}

// MARK: - "Scan new room" dashed tile

struct NewRoomCard: View {
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            VStack(spacing: 8) {
                ZStack {
                    Circle().fill(DR.cream2).frame(width: 44, height: 44)
                    Image(systemName: "plus")
                        .font(.system(size: 18, weight: .regular))
                        .foregroundStyle(DR.ink2)
                }
                Text("Scan new room")
                    .font(DR.sans(12, weight: .medium))
                    .foregroundStyle(DR.ink3)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 164)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(DR.line, style: StrokeStyle(lineWidth: 1.5, dash: [6, 4]))
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Inspiration row card (160 wide)

struct InspirationCard: View {
    let title: String
    let author: String
    let seed: Int
    let vibe: RoomVibe

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            RoomPlaceholderView(seed: seed, label: nil, vibe: vibe)
                .frame(width: 160, height: 160)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            Text(title)
                .font(DR.sans(12.5, weight: .semibold))
                .foregroundStyle(DR.ink)
                .lineLimit(2)
            Text(author)
                .font(DR.sans(11))
                .foregroundStyle(DR.ink3)
        }
        .frame(width: 160, alignment: .leading)
    }
}

// MARK: - RoomDetailView (3D scene; ink bg with indigo accents)

struct RoomDetailView: View {
    let room: SavedRoom

    @Environment(\.dismiss) private var dismiss
    @StateObject private var editModel: RoomEditModel

    init(room: SavedRoom) {
        self.room = room
        _editModel = StateObject(wrappedValue: RoomEditModel(initial: room.payload))
    }

    var body: some View {
        ZStack {
            DR.ink.ignoresSafeArea()

            RoomSceneView(payload: room.payload, editModel: editModel)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                LinearGradient(
                    colors: [DR.ink.opacity(0.85), .clear],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .frame(height: 160)
                .allowsHitTesting(false)
                Spacer()
            }
            .ignoresSafeArea()

            VStack(spacing: 0) {
                topBar
                Spacer()
                statsSheet
            }

            // Plan 4a edit menu (Delete / Lift / Undo side bar).
            RoomEditMenuView(editModel: editModel)
                .allowsHitTesting(true)
        }
        .toolbar(.hidden, for: .navigationBar)
    }

    private var topBar: some View {
        HStack {
            Button { dismiss() } label: {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 13, weight: .semibold))
                    Text("BACK")
                        .font(DR.mono(10, weight: .medium))
                        .tracking(2.2)
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 14).padding(.vertical, 10)
                .background(.ultraThinMaterial, in: Capsule())
            }
            Spacer()
            Text(formattedDate)
                .font(DR.mono(10, weight: .medium))
                .tracking(2.2)
                .foregroundStyle(.white.opacity(0.7))
        }
        .padding(.horizontal, 18)
        .padding(.top, 10)
    }

    private var statsSheet: some View {
        let payload = editModel.roomPayload

        return VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                DREyebrow(text: "ROOM / DETAIL", color: DR.indigoSoft)
                Text(room.name)
                    .font(DR.serif(20))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    .minimumScaleFactor(0.72)
            }

            Rectangle().fill(.white.opacity(0.10)).frame(height: 1)

            HStack(alignment: .top, spacing: 12) {
                statColumn(label: "FOOTPRINT",
                           value: String(format: "%.1f × %.1f", payload.envelope.width, payload.envelope.length),
                           unit: "m")
                statColumn(label: "HEIGHT",
                           value: String(format: "%.2f", payload.envelope.height),
                           unit: "m")
                statColumn(label: "OBJECTS",
                           value: "\(payload.objects.count)",
                           unit: "indexed")
            }
        }
        .padding(14)
        .frame(maxWidth: 300, alignment: .leading)
        .background(
            ZStack {
                DR.ink.opacity(0.92)
                Rectangle().fill(.ultraThinMaterial).opacity(0.35)
            }
        )
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(.white.opacity(0.10), lineWidth: 1)
        )
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.leading, 12)
        .padding(.trailing, 104)
        .padding(.bottom, 12)
    }

    private func statColumn(label: String, value: String, unit: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(DR.mono(7.5, weight: .medium))
                .tracking(1.3)
                .foregroundStyle(.white.opacity(0.55))

            HStack(alignment: .lastTextBaseline, spacing: 3) {
                Text(value)
                    .font(DR.serif(15))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                Text(unit)
                    .font(DR.mono(7.5, weight: .medium))
                    .foregroundStyle(DR.indigoSoft)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var formattedDate: String {
        let f = DateFormatter()
        f.dateFormat = "dd MMM yyyy"
        return f.string(from: room.createdAt).uppercased()
    }
}
