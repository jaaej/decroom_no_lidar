//
//  MockData.swift
//  decroom_no_lidar
//
//  Hardcoded sample content used by Explore, Profile, and the Inspiration row.
//  No backend wiring — all literals.
//

import Foundation
import SwiftUI

// MARK: - Stable hash (for deriving display likes)

extension String {
    /// Deterministic 32-bit-ish hash used for display-only fields like
    /// fake like counts. NOT cryptographic.
    var drStableHash: Int {
        var h: UInt64 = 0xcbf29ce484222325 // FNV offset basis
        for byte in self.utf8 {
            h ^= UInt64(byte)
            h &*= 0x100000001b3
        }
        return Int(truncatingIfNeeded: h & 0x7fffffff)
    }
}

// MARK: - Vibe palette key (matches RoomPlaceholderView)

enum RoomVibe: String, CaseIterable {
    case warm, cool, sage, lilac, clay, mono
}

// MARK: - Explore: rooms feed

struct ExploreRoom: Identifiable {
    let id: Int
    let name: String
    let author: String
    let size: String       // e.g. "24 m²"
    let items: Int
    let likes: Int
    let vibe: RoomVibe
    let tags: [String]
    let isHero: Bool
}

// MARK: - Explore: marketplace

enum MarketIcon: String {
    case chair, lamp, plant, cube, image, rug
    var sfSymbol: String {
        switch self {
        case .chair: return "chair.lounge"
        case .lamp:  return "lamp.desk"
        case .plant: return "leaf"
        case .cube:  return "cube"
        case .image: return "photo"
        case .rug:   return "oval"
        }
    }
}

struct MarketItem: Identifiable {
    let id: Int
    let name: String
    let brand: String
    let price: String
    let rating: String
    let orders: Int
    let tag: String?
    let color: Color
    let icon: MarketIcon
    let category: String
}

struct ShopCategory: Identifiable {
    let id: String
    let label: String
    let icon: MarketIcon
}

// MARK: - Profile

struct ActivityItem: Identifiable {
    let id = UUID()
    let who: String
    let hue: Double         // avatar hue 0..360
    let verb: String        // "liked", "commented on", "followed you", "remixed"
    let target: String?     // optional room name
    let time: String        // "2h ago"
}

struct ProfileBio {
    let name: String
    let handle: String
    let joined: String
    let bio: String
    let roomsCount: Int
    let likesCount: Int
    let friendsCount: Int
    let rank: Int
    let isPro: Bool
}

// MARK: - Explore: dropdown sections

struct ExploreSection: Identifiable {
    let id: String
    let title: String
    let subtitle: String
}

// MARK: - Container

enum MockData {

    static let exploreSections: [ExploreSection] = [
        .init(id: "trending",  title: "Trending",  subtitle: "Rooms everyone's looking at"),
        .init(id: "following", title: "Following", subtitle: "From people you follow"),
        .init(id: "nearby",    title: "Nearby",    subtitle: "Within your city"),
        .init(id: "style",     title: "By style",  subtitle: "Browse by aesthetic"),
    ]

    static let exploreRooms: [ExploreRoom] = [
        .init(id: 101, name: "Sunlit Studio, 14m²", author: "@sabrina", size: "14 m²", items: 22, likes: 1840, vibe: .warm,  tags: ["minimal", "bright"], isHero: true),
        .init(id: 102, name: "Maximalist Reading Nook", author: "@codyjohnson1", size: "9 m²", items: 18, likes: 612,  vibe: .clay,  tags: ["maximalist"], isHero: false),
        .init(id: 103, name: "Soft Industrial Loft", author: "@_maxjordy",  size: "32 m²", items: 26, likes: 884,  vibe: .mono,  tags: ["industrial"], isHero: false),
        .init(id: 104, name: "Plant Parent's Corner", author: "@kateysmith", size: "6 m²",  items: 14, likes: 421,  vibe: .sage,  tags: ["green", "cozy"], isHero: false),
        .init(id: 105, name: "Mid-Century Den", author: "@robinleigh",  size: "20 m²", items: 19, likes: 318,  vibe: .clay,  tags: ["midcentury"], isHero: false),
        .init(id: 106, name: "Quiet Bedroom", author: "@nori", size: "12 m²", items: 11, likes: 256,  vibe: .lilac, tags: ["calm"], isHero: false),
        .init(id: 107, name: "Linen & Light", author: "@hemse",       size: "18 m²", items: 16, likes: 1042, vibe: .warm,  tags: ["linen"], isHero: false),
        .init(id: 108, name: "Concrete & Oak Office", author: "@studio.b", size: "11 m²", items: 13, likes: 199, vibe: .cool, tags: ["work"], isHero: false),
    ]

    static let categories: [ShopCategory] = [
        .init(id: "all",      label: "All",      icon: .cube),
        .init(id: "seating",  label: "Seating",  icon: .chair),
        .init(id: "lighting", label: "Lighting", icon: .lamp),
        .init(id: "plants",   label: "Plants",   icon: .plant),
        .init(id: "rugs",     label: "Rugs",     icon: .rug),
        .init(id: "art",      label: "Art",      icon: .image),
    ]

    static let marketItems: [MarketItem] = [
        .init(id: 1, name: "Linen Lounge Chair",   brand: "Hemse & Co",  price: "$480", rating: "4.8", orders: 1240, tag: "Best seller", color: Color(red: 0.890, green: 0.804, blue: 0.682), icon: .chair, category: "seating"),
        .init(id: 2, name: "Ceramic Table Lamp",   brand: "Luma",        price: "$129", rating: "4.6", orders: 860,  tag: "New",         color: Color(red: 0.969, green: 0.910, blue: 0.764), icon: .lamp,  category: "lighting"),
        .init(id: 3, name: "Monstera, 1.2m",       brand: "Verdé",       price: "$65",  rating: "4.9", orders: 2110, tag: "Live plant",  color: Color(red: 0.745, green: 0.875, blue: 0.769), icon: .plant, category: "plants"),
        .init(id: 4, name: "Hand-tufted Rug",      brand: "Loom Studio", price: "$320", rating: "4.7", orders: 412,  tag: nil,           color: Color(red: 0.871, green: 0.694, blue: 0.553), icon: .rug,   category: "rugs"),
        .init(id: 5, name: "Walnut Side Table",    brand: "Kin",         price: "$210", rating: "4.5", orders: 305,  tag: nil,           color: Color(red: 0.541, green: 0.404, blue: 0.302), icon: .cube,  category: "seating"),
        .init(id: 6, name: "Framed Print — Dune",  brand: "Papier",      price: "$48",  rating: "4.8", orders: 980,  tag: "Ships free",  color: Color(red: 0.949, green: 0.882, blue: 0.733), icon: .image, category: "art"),
        .init(id: 7, name: "Bouclé Floor Cushion", brand: "Hemse & Co",  price: "$95",  rating: "4.6", orders: 620,  tag: nil,           color: Color(red: 0.937, green: 0.851, blue: 0.737), icon: .cube,  category: "seating"),
        .init(id: 8, name: "Arc Floor Lamp",       brand: "Nord",        price: "$389", rating: "4.9", orders: 240,  tag: "Designer",    color: Color(red: 0.290, green: 0.290, blue: 0.380), icon: .lamp,  category: "lighting"),
    ]

    static let inspirationRows: [(title: String, author: String, vibe: RoomVibe)] = [
        ("Maximalist Reading Nook", "@codyjohnson1", .clay),
        ("Sunlit Studio, 14m²",     "@sabrina",      .warm),
        ("Soft Industrial",         "@_maxjordy",    .mono),
        ("Plant Parent's Corner",   "@kateysmith",   .sage),
    ]

    static let activity: [ActivityItem] = [
        .init(who: "Sabrina",   hue:  35, verb: "liked your room",       target: "Jae's Living Room", time: "2h ago"),
        .init(who: "Cody",      hue: 220, verb: "commented on",          target: "Reading Nook",      time: "5h ago"),
        .init(who: "Max",       hue: 150, verb: "followed you",          target: nil,                  time: "1d ago"),
        .init(who: "Katey",     hue: 290, verb: "remixed your room",     target: "Plant Corner",       time: "2d ago"),
    ]

    static let profile = ProfileBio(
        name: "Jae Jeon",
        handle: "@jaejeon",
        joined: "Joined Mar 2025",
        bio: "Designing small rooms with big light. Plants, paper, and quiet edges.",
        roomsCount: 12,
        likesCount: 284,
        friendsCount: 142,
        rank: 34,
        isPro: true
    )
}

// MARK: - Display likes for own (real) saved rooms

extension RoomVibe {
    /// Pick a stable vibe based on room id so its inspiration card is consistent.
    static func from(id: String) -> RoomVibe {
        let cases = RoomVibe.allCases
        return cases[abs(id.drStableHash) % cases.count]
    }
}

func mockLikes(forRoomId id: String) -> Int {
    return (id.drStableHash % 200) + 12
}
