//
//  ExploreView.swift
//  decroom_no_lidar
//
//  Two-tab Explore: Rooms feed and Marketplace. All data is mocked
//  from MockData; cart state is local @State.
//

import SwiftUI

struct ExploreView: View {
    @State private var query: String = ""
    @State private var tab: ExploreTab = .rooms
    @State private var section: String = "trending"
    @State private var sectionOpen: Bool = false
    @State private var category: String = "all"
    @State private var cart: [Int: Int] = [:]   // marketItem.id → qty

    enum ExploreTab { case rooms, shop }

    var body: some View {
        ZStack(alignment: .bottom) {
            DR.cream.ignoresSafeArea()

            VStack(spacing: 0) {
                header
                    .padding(.horizontal, 16)
                    .padding(.top, 54)
                    .padding(.bottom, 10)

                searchBar
                    .padding(.horizontal, 16)
                    .padding(.bottom, 10)

                tabs
                    .padding(.horizontal, 16)
                    .padding(.bottom, 12)

                ScrollView(.vertical, showsIndicators: false) {
                    if tab == .rooms { roomsTab } else { shopTab }
                }
            }

            if tab == .shop, !cart.isEmpty {
                cartPill
                    .padding(.horizontal, 16)
                    .padding(.bottom, 100)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.22), value: cart.isEmpty)
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 3) {
                DREyebrow(text: "DISCOVER")
                HStack(spacing: 0) {
                    Text("Explore")
                        .font(DR.serif(28))
                        .foregroundStyle(DR.ink)
                        .tracking(-0.2)
                    Text(".")
                        .font(DR.serif(28))
                        .foregroundStyle(DR.coral)
                }
            }
            Spacer()
            // Bell with notification dot
            ZStack(alignment: .topTrailing) {
                Image(systemName: "bell")
                    .font(.system(size: 17, weight: .regular))
                    .foregroundStyle(DR.ink2)
                    .frame(width: 40, height: 40)
                    .background(DR.paper, in: RoundedRectangle(cornerRadius: 12))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(DR.line, lineWidth: 1)
                    )
                Circle().fill(DR.coral).frame(width: 7, height: 7)
                    .offset(x: -7, y: 7)
            }
        }
    }

    // MARK: - Search

    private var searchBar: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(DR.ink3)
            TextField(
                tab == .rooms ? "Search rooms, @users, styles" : "Search the marketplace",
                text: $query
            )
            .font(DR.sans(13.5))
            .foregroundStyle(DR.ink)
            .submitLabel(.search)

            if !query.isEmpty {
                Button { query = "" } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(DR.ink3)
                        .frame(width: 22, height: 22)
                        .background(DR.cream2, in: Circle())
                }
            }
        }
        .padding(.horizontal, 14)
        .frame(height: 46)
        .background(DR.paper, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(DR.line2, lineWidth: 1)
        )
    }

    // MARK: - Segmented control

    private var tabs: some View {
        HStack(spacing: 0) {
            tabButton(label: "Rooms", icon: "house", count: "2.4k", target: .rooms)
            tabButton(label: "Marketplace", icon: "tag", count: "8.1k", target: .shop)
        }
        .padding(3)
        .background(DR.cream2, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func tabButton(label: String, icon: String, count: String, target: ExploreTab) -> some View {
        let active = tab == target
        return Button {
            withAnimation(.easeOut(duration: 0.18)) { tab = target }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .medium))
                Text(label)
                    .font(DR.sans(13, weight: .semibold))
                Text(count)
                    .font(DR.mono(10, weight: .medium))
                    .opacity(0.6)
            }
            .foregroundStyle(active ? DR.ink : DR.ink3)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(
                active
                    ? RoundedRectangle(cornerRadius: 10, style: .continuous).fill(DR.paper)
                        .shadow(color: DR.ink.opacity(0.06), radius: 1, y: 1)
                    : nil
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Rooms tab

    private var roomsTab: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Dropdown
            sectionDropdown
                .padding(.horizontal, 16)
                .padding(.bottom, 14)

            // Hero
            if let hero = MockData.exploreRooms.first(where: { $0.isHero }) {
                heroCard(for: hero)
                    .padding(.horizontal, 16)
                    .padding(.bottom, 12)
                    .onTapGesture {} // visual-only
            }

            // Grid
            let cols = [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)]
            LazyVGrid(columns: cols, spacing: 10) {
                ForEach(MockData.exploreRooms.filter { !$0.isHero }) { room in
                    exploreRoomTile(room)
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 120)
        }
    }

    private var sectionDropdown: some View {
        let s = MockData.exploreSections.first(where: { $0.id == section }) ?? MockData.exploreSections[0]
        return VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeOut(duration: 0.18)) { sectionOpen.toggle() }
            } label: {
                HStack(spacing: 10) {
                    VStack(alignment: .leading, spacing: 1) {
                        Text(s.title)
                            .font(DR.sans(13, weight: .semibold))
                            .foregroundStyle(DR.ink)
                        Text(s.subtitle)
                            .font(DR.sans(10.5))
                            .foregroundStyle(DR.ink3)
                    }
                    Spacer()
                    Image(systemName: "chevron.down")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(DR.ink3)
                        .rotationEffect(.degrees(sectionOpen ? 180 : 0))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(DR.paper, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(DR.line, lineWidth: 1)
                )
            }
            .buttonStyle(.plain)

            if sectionOpen {
                VStack(spacing: 0) {
                    ForEach(MockData.exploreSections) { item in
                        Button {
                            section = item.id
                            withAnimation(.easeOut(duration: 0.15)) { sectionOpen = false }
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(item.title)
                                        .font(DR.sans(13, weight: item.id == section ? .semibold : .medium))
                                        .foregroundStyle(item.id == section ? DR.ink : DR.ink2)
                                    Text(item.subtitle)
                                        .font(DR.sans(10.5))
                                        .foregroundStyle(DR.ink3)
                                }
                                Spacer()
                                if item.id == section {
                                    Image(systemName: "checkmark")
                                        .font(.system(size: 12, weight: .semibold))
                                        .foregroundStyle(DR.indigo)
                                }
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(item.id == section ? DR.cream2 : DR.paper)
                        }
                        .buttonStyle(.plain)
                        if item.id != MockData.exploreSections.last?.id {
                            Rectangle().fill(DR.line2).frame(height: 1)
                        }
                    }
                }
                .background(DR.paper, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(DR.line, lineWidth: 1)
                )
                .padding(.top, 4)
                .shadow(color: DR.ink.opacity(0.10), radius: 12, y: 6)
            }
        }
    }

    private func heroCard(for room: ExploreRoom) -> some View {
        ZStack(alignment: .bottomLeading) {
            RoomPlaceholderView(seed: room.id, label: nil, vibe: room.vibe)
                .frame(height: 240)

            LinearGradient(
                colors: [.clear, .black.opacity(0.55)],
                startPoint: .center, endPoint: .bottom
            )

            // Top-left "Room of the day" pill
            VStack {
                HStack {
                    DRBlurPill(tint: .white, opacity: 0.85) {
                        HStack(spacing: 6) {
                            Image(systemName: "sparkles")
                                .font(.system(size: 10, weight: .semibold))
                            Text("ROOM OF THE DAY")
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

            VStack(alignment: .leading, spacing: 4) {
                Text(room.name)
                    .font(DR.serif(26))
                    .foregroundStyle(.white)
                    .tracking(-0.2)
                HStack(spacing: 8) {
                    DRAvatar(name: room.author.replacingOccurrences(of: "@", with: ""), size: 22, hue: 290)
                    Text(room.author)
                        .font(DR.sans(12))
                        .foregroundStyle(.white.opacity(0.95))
                    Circle().fill(.white.opacity(0.5)).frame(width: 3, height: 3)
                    Text(room.size)
                        .font(DR.sans(12))
                        .foregroundStyle(.white.opacity(0.85))
                    Spacer()
                    HStack(spacing: 4) {
                        Image(systemName: "heart").font(.system(size: 12))
                        Text("\(room.likes)").font(DR.sans(12))
                    }
                    .foregroundStyle(.white.opacity(0.95))
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 14)
        }
        .frame(height: 240)
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func exploreRoomTile(_ room: ExploreRoom) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            RoomPlaceholderView(seed: room.id, label: nil, vibe: room.vibe)
                .frame(height: 130)

            VStack(alignment: .leading, spacing: 5) {
                Text(room.name)
                    .font(DR.sans(12.5, weight: .semibold))
                    .foregroundStyle(DR.ink)
                    .lineLimit(2)
                HStack {
                    Text(room.author)
                        .font(DR.sans(10.5))
                        .foregroundStyle(DR.ink3)
                    Spacer()
                    HStack(spacing: 3) {
                        Image(systemName: "heart").font(.system(size: 10))
                        Text("\(room.likes)").font(DR.sans(10.5))
                    }
                    .foregroundStyle(DR.ink3)
                }
                if !room.tags.isEmpty {
                    HStack(spacing: 4) {
                        ForEach(room.tags.prefix(2), id: \.self) { t in
                            Text(t)
                                .font(DR.sans(9.5, weight: .medium))
                                .foregroundStyle(DR.ink2)
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(DR.cream2, in: RoundedRectangle(cornerRadius: 4))
                        }
                    }
                    .padding(.top, 2)
                }
            }
            .padding(.horizontal, 10)
            .padding(.top, 9)
            .padding(.bottom, 11)
        }
        .background(DR.paper, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(DR.line2, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    // MARK: - Shop tab

    private var shopTab: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Categories
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(MockData.categories) { c in
                        DRChip(
                            label: c.label,
                            active: category == c.id,
                            icon: c.icon.sfSymbol,
                            onTap: { category = c.id }
                        )
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 14)
            }

            let visible = MockData.marketItems.filter { category == "all" || $0.category == category }
            let cols = [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)]
            LazyVGrid(columns: cols, spacing: 10) {
                ForEach(visible) { item in
                    productCard(item)
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 160)
        }
    }

    private func productCard(_ item: MarketItem) -> some View {
        let inCart = (cart[item.id] ?? 0) > 0
        return VStack(alignment: .leading, spacing: 0) {
            ZStack {
                LinearGradient(
                    colors: [
                        item.color,
                        item.color.opacity(0.78)
                    ],
                    startPoint: .topLeading, endPoint: .bottomTrailing
                )
                Image(systemName: item.icon.sfSymbol)
                    .font(.system(size: 44, weight: .light))
                    .foregroundStyle(DR.ink.opacity(0.6))

                if let tag = item.tag {
                    VStack {
                        HStack {
                            DRBlurPill(tint: .white, opacity: 0.9) {
                                Text(tag.uppercased())
                                    .font(DR.mono(9.5, weight: .medium))
                                    .tracking(1.4)
                                    .foregroundStyle(DR.ink)
                            }
                            Spacer()
                        }
                        Spacer()
                    }
                    .padding(8)
                }

                VStack {
                    HStack {
                        Spacer()
                        Button {
                            withAnimation(.spring(response: 0.28, dampingFraction: 0.8)) {
                                cart[item.id] = inCart ? nil : 1
                            }
                        } label: {
                            Image(systemName: inCart ? "checkmark" : "plus")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(inCart ? DR.cream : DR.ink)
                                .frame(width: 30, height: 30)
                                .background(
                                    Circle().fill(inCart ? DR.ink : .white.opacity(0.92))
                                )
                        }
                    }
                    Spacer()
                }
                .padding(8)
            }
            .frame(height: 130)

            VStack(alignment: .leading, spacing: 2) {
                Text(item.brand.uppercased())
                    .font(DR.mono(9.5, weight: .medium))
                    .tracking(1.4)
                    .foregroundStyle(DR.ink3)
                Text(item.name)
                    .font(DR.sans(12.5, weight: .semibold))
                    .foregroundStyle(DR.ink)
                    .lineLimit(2)
                HStack(alignment: .lastTextBaseline) {
                    Text(item.price)
                        .font(DR.serif(18))
                        .foregroundStyle(DR.ink)
                    Spacer()
                    Text("★ \(item.rating) · \(item.orders)")
                        .font(DR.sans(10))
                        .foregroundStyle(DR.ink3)
                }
                .padding(.top, 4)
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .padding(.bottom, 12)
        }
        .background(DR.paper, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(DR.line2, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    // MARK: - Cart pill

    private var cartPill: some View {
        let count = cart.values.reduce(0, +)
        let total = cart.reduce(0.0) { acc, kv in
            let item = MockData.marketItems.first { $0.id == kv.key }
            let raw = (item?.price ?? "$0").replacingOccurrences(of: "$", with: "")
            return acc + (Double(raw) ?? 0) * Double(kv.value)
        }
        return HStack(spacing: 12) {
            Text("\(count)")
                .font(DR.sans(12, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 30, height: 30)
                .background(DR.coral, in: RoundedRectangle(cornerRadius: 10))

            VStack(alignment: .leading, spacing: 1) {
                Text("View cart")
                    .font(DR.sans(13, weight: .semibold))
                    .foregroundStyle(DR.cream)
                Text("Ships to 10001 · Free over $100")
                    .font(DR.sans(11))
                    .foregroundStyle(DR.cream.opacity(0.7))
            }
            Spacer()
            Text(String(format: "$%.0f", total))
                .font(DR.serif(20))
                .foregroundStyle(DR.cream)
            Image(systemName: "chevron.right")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(DR.cream)
        }
        .padding(.horizontal, 18)
        .frame(height: 54)
        .background(DR.ink, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .shadow(color: DR.ink.opacity(0.28), radius: 14, y: 10)
    }
}

#Preview {
    ExploreView()
}
