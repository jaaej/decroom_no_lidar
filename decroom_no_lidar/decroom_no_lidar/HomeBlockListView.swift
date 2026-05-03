//
//  HomeBlockListView.swift
//  decroom_no_lidar
//
//  Drag-to-reorder list used by Home in edit mode. Mirrors the JSX
//  `BlockList`/`BlockWrapper` from screens-home.jsx:
//   - 36pt grab handle column on the left
//   - tap-to-toggle eye on the right hides/shows blocks
//   - dashed border + small block label above each row
//   - long-pressing the handle lifts the card; drag swaps rows live
//   - in view mode, adjacent stat blocks render in a shared horizontal row
//

import SwiftUI

// MARK: - Public block model

struct HomeBlock: Codable, Equatable, Identifiable {
    var id: String
    var visible: Bool
}

enum HomeBlockMeta {
    static let label: [String: String] = [
        "greeting":    "Greeting",
        "stat_rooms":  "Rooms stat",
        "stat_rank":   "Rank stat",
        "stat_likes":  "Likes stat",
        "filters":     "Filter chips",
        "featured":    "Featured room",
        "grid":        "Your rooms grid",
        "inspiration": "Inspiration row",
    ]
    static let fixed: Set<String> = ["greeting"]
    static let stat: Set<String> = ["stat_rooms", "stat_rank", "stat_likes"]
}

let DR_DEFAULT_HOME_BLOCKS: [HomeBlock] = [
    .init(id: "greeting",    visible: true),
    .init(id: "stat_rooms",  visible: true),
    .init(id: "stat_likes",  visible: true),
    .init(id: "stat_rank",   visible: false),
    .init(id: "filters",     visible: true),
    .init(id: "featured",    visible: true),
    .init(id: "grid",        visible: true),
    .init(id: "inspiration", visible: true),
]

// MARK: - View

struct HomeBlockListView<Content: View>: View {
    @Binding var blocks: [HomeBlock]
    let editing: Bool
    @ViewBuilder var renderBlock: (HomeBlock) -> Content

    @State private var dragId: String? = nil
    @State private var dragY: CGFloat = 0
    @State private var rowFrames: [String: CGRect] = [:]

    private let coord = "homeBlockList"

    /// Group of stats rendered in a row (view mode only).
    private struct StatGroup: Identifiable {
        let id: String          // join of contained ids
        let blocks: [HomeBlock]
    }

    /// Composite row entry (either single block or grouped stat row).
    private enum RenderEntry: Identifiable {
        case single(HomeBlock)
        case group(StatGroup)

        var id: String {
            switch self {
            case .single(let b): return b.id
            case .group(let g):  return "stats:\(g.id)"
            }
        }
    }

    private var renderList: [RenderEntry] {
        if editing {
            return blocks.map { .single($0) }
        }
        // View mode: collapse adjacent visible stat blocks
        var out: [RenderEntry] = []
        var i = 0
        while i < blocks.count {
            let b = blocks[i]
            if HomeBlockMeta.stat.contains(b.id) {
                var grp: [HomeBlock] = []
                while i < blocks.count, HomeBlockMeta.stat.contains(blocks[i].id) {
                    if blocks[i].visible { grp.append(blocks[i]) }
                    i += 1
                }
                if !grp.isEmpty {
                    out.append(.group(StatGroup(id: grp.map(\.id).joined(separator: "+"), blocks: grp)))
                }
            } else {
                if b.visible { out.append(.single(b)) }
                i += 1
            }
        }
        return out
    }

    var body: some View {
        VStack(spacing: editing ? 8 : 0) {
            ForEach(renderList) { entry in
                switch entry {
                case .group(let g):
                    statRow(blocks: g.blocks)
                case .single(let b):
                    blockRow(block: b)
                }
            }
        }
        .coordinateSpace(name: coord)
    }

    // MARK: View-mode stat row

    private func statRow(blocks: [HomeBlock]) -> some View {
        HStack(spacing: 8) {
            ForEach(blocks) { b in
                renderBlock(b)
            }
        }
        .padding(.bottom, 16)
    }

    // MARK: Single block row

    @ViewBuilder
    private func blockRow(block: HomeBlock) -> some View {
        let isFixed = HomeBlockMeta.fixed.contains(block.id)
        let isDragging = dragId == block.id

        if !editing {
            renderBlock(block)
                .padding(.bottom, 4)
                .background(
                    GeometryReader { p in
                        Color.clear.preference(
                            key: BlockFramePreferenceKey.self,
                            value: [block.id: p.frame(in: .named(coord))]
                        )
                    }
                )
        } else {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text((HomeBlockMeta.label[block.id] ?? block.id).uppercased())
                        .font(DR.mono(9.5, weight: .medium))
                        .tracking(1.0)
                        .foregroundStyle(DR.ink3)
                    if !block.visible {
                        Text("· HIDDEN")
                            .font(DR.mono(9, weight: .medium))
                            .tracking(0.6)
                            .foregroundStyle(DR.ink2)
                    }
                }
                .padding(.leading, 36)

                ZStack(alignment: .topLeading) {
                    HStack(alignment: .top, spacing: 0) {
                        // Grab handle column
                        if !isFixed {
                            grabHandle(for: block.id)
                                .frame(width: 36)
                        } else {
                            Color.clear.frame(width: 36)
                        }
                        renderBlock(block)
                            .padding(.trailing, 44)   // room for eye toggle
                            .allowsHitTesting(false)  // disable inner taps in edit mode
                            .padding(.vertical, 6)
                    }

                    // Eye toggle (right edge)
                    if !isFixed {
                        eyeToggle(blockId: block.id)
                            .frame(maxWidth: .infinity, alignment: .trailing)
                            .padding(.trailing, 8)
                            .padding(.vertical, 0)
                            .frame(maxHeight: .infinity)
                    }
                }
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(block.visible ? DR.paper : DR.line2.opacity(0.7))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .stroke(DR.line, style: StrokeStyle(lineWidth: 1.2, dash: [5, 4]))
                )
                .opacity(block.visible ? 1.0 : 0.6)
                .opacity(isDragging ? 0.0 : 1.0)
                .background(
                    GeometryReader { p in
                        Color.clear.preference(
                            key: BlockFramePreferenceKey.self,
                            value: [block.id: p.frame(in: .named(coord))]
                        )
                    }
                )
            }
            .overlay(alignment: .topLeading) {
                if isDragging, let frame = rowFrames[block.id] {
                    draggedFloater(block: block, baseFrame: frame)
                }
            }
            .onPreferenceChange(BlockFramePreferenceKey.self) { merged in
                for (k, v) in merged { rowFrames[k] = v }
            }
        }
    }

    // MARK: Floating dragged card

    @ViewBuilder
    private func draggedFloater(block: HomeBlock, baseFrame: CGRect) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text((HomeBlockMeta.label[block.id] ?? block.id).uppercased())
                .font(DR.mono(9.5, weight: .medium))
                .tracking(1.0)
                .foregroundStyle(DR.ink3)
                .padding(.leading, 36)

            HStack(alignment: .top, spacing: 0) {
                Image(systemName: "line.3.horizontal")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(DR.ink3)
                    .frame(width: 36, height: 36)
                renderBlock(block)
                    .padding(.trailing, 44)
                    .padding(.vertical, 6)
                    .allowsHitTesting(false)
            }
            .background(DR.paper, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(DR.indigo.opacity(0.6), lineWidth: 1.5)
            )
            .shadow(color: DR.ink.opacity(0.18), radius: 16, y: 12)
        }
        .frame(width: baseFrame.width)
        .scaleEffect(1.02)
        .offset(y: dragY - baseFrame.midY)
    }

    // MARK: Grab handle gesture

    private func grabHandle(for id: String) -> some View {
        Image(systemName: "line.3.horizontal")
            .font(.system(size: 14, weight: .medium))
            .foregroundStyle(DR.ink3)
            .frame(width: 36, height: 44)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0, coordinateSpace: .named(coord))
                    .onChanged { v in
                        if dragId != id {
                            withAnimation(.easeOut(duration: 0.12)) {
                                dragId = id
                            }
                        }
                        dragY = v.location.y
                        // Find which row contains the finger
                        for (otherId, frame) in rowFrames where otherId != id {
                            if frame.minY < v.location.y && v.location.y < frame.maxY,
                               !HomeBlockMeta.fixed.contains(otherId) {
                                let from = blocks.firstIndex { $0.id == id } ?? -1
                                let to   = blocks.firstIndex { $0.id == otherId } ?? -1
                                if from != -1 && to != -1 && from != to {
                                    var copy = blocks
                                    let moved = copy.remove(at: from)
                                    copy.insert(moved, at: to)
                                    blocks = copy
                                }
                                break
                            }
                        }
                    }
                    .onEnded { _ in
                        withAnimation(.spring(response: 0.32, dampingFraction: 0.85)) {
                            dragId = nil
                        }
                    }
            )
    }

    // MARK: Eye toggle

    private func eyeToggle(blockId: String) -> some View {
        Button {
            if let i = blocks.firstIndex(where: { $0.id == blockId }) {
                blocks[i].visible.toggle()
            }
        } label: {
            Image(systemName: blocks.first { $0.id == blockId }?.visible == true ? "eye" : "eye.slash")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(DR.ink2)
                .frame(width: 30, height: 30)
                .background(DR.paper, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(DR.line, lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Frame collection

private struct BlockFramePreferenceKey: PreferenceKey {
    static var defaultValue: [String: CGRect] = [:]
    static func reduce(value: inout [String: CGRect], nextValue: () -> [String: CGRect]) {
        value.merge(nextValue(), uniquingKeysWith: { _, new in new })
    }
}

// MARK: - @AppStorage helpers

extension Array where Element == HomeBlock {
    var asAppStorageData: Data {
        (try? JSONEncoder().encode(self)) ?? Data()
    }
    static func fromAppStorageData(_ data: Data) -> [HomeBlock]? {
        guard !data.isEmpty else { return nil }
        return try? JSONDecoder().decode([HomeBlock].self, from: data)
    }
}
