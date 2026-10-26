//
//  ServerDiscovery.swift
//  decroom_no_lidar
//
//  Discovers the Decroom backend on the local network via Bonjour.
//  The backend advertises "_decroom._tcp." — this class watches for it
//  and exposes the resolved URL so UploadService can use it automatically.
//

import Foundation
import Network

@MainActor
final class ServerDiscovery: ObservableObject {
    static let shared = ServerDiscovery()

    /// Upload URLs of every advertised server, e.g.
    /// "http://192.168.1.42:8000/api/upload". Anything on the network can
    /// advertise _decroom._tcp, so UploadService verifies each one holds
    /// the server key before using it.
    @Published private(set) var uploadURLs: [String] = []

    /// First discovered upload URL. Nil until a server is found.
    var uploadURL: String? { uploadURLs.first }

    /// Whether actively searching.
    @Published private(set) var isSearching = false

    private var browser: NWBrowser?
    private var connections: [NWEndpoint: NWConnection] = [:]
    private var resolved: [NWEndpoint: String] = [:]

    private init() {
        startBrowsing()
    }

    func startBrowsing() {
        // Cancel any previous browser
        browser?.cancel()

        let params = NWParameters()
        // The backend is a Mac on the same Wi-Fi; peer-to-peer (AWDL) would
        // only widen who can answer as "the server".
        params.includePeerToPeer = false
        let descriptor = NWBrowser.Descriptor.bonjour(type: "_decroom._tcp", domain: nil)
        let newBrowser = NWBrowser(for: descriptor, using: params)

        newBrowser.stateUpdateHandler = { [weak self] state in
            Task { @MainActor in
                switch state {
                case .ready:
                    self?.isSearching = true
                case .failed, .cancelled:
                    self?.isSearching = false
                default:
                    break
                }
            }
        }

        newBrowser.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @MainActor in
                self?.handleResults(results)
            }
        }

        newBrowser.start(queue: .main)
        browser = newBrowser
        isSearching = true
    }

    private func handleResults(_ results: Set<NWBrowser.Result>) {
        let endpoints = Set(results.map(\.endpoint))
        for (endpoint, conn) in connections where !endpoints.contains(endpoint) {
            conn.cancel()
            connections[endpoint] = nil
        }
        resolved = resolved.filter { endpoints.contains($0.key) }
        publish()
        for endpoint in endpoints where resolved[endpoint] == nil && connections[endpoint] == nil {
            resolve(endpoint)
        }
    }

    /// Resolve a Bonjour endpoint to IP + port via a throwaway TCP connection.
    private func resolve(_ endpoint: NWEndpoint) {
        let conn = NWConnection(to: endpoint, using: .tcp)
        conn.stateUpdateHandler = { [weak self] state in
            Task { @MainActor in
                guard let self else { return }
                switch state {
                case .ready:
                    if let innerEndpoint = conn.currentPath?.remoteEndpoint,
                       case let .hostPort(host, port) = innerEndpoint {
                        let ip: String
                        switch host {
                        case .ipv4(let addr):
                            ip = "\(addr)"
                        case .ipv6(let addr):
                            ip = "\(addr)"
                        case .name(let name, _):
                            ip = name
                        @unknown default:
                            ip = "\(host)"
                        }
                        let url = "http://\(ip):\(port)/api/upload"
                        self.resolved[endpoint] = url
                        self.publish()
                        print("[ServerDiscovery] Found backend at \(url)")
                    }
                    conn.cancel()
                    self.connections[endpoint] = nil
                case .failed, .cancelled:
                    self.connections[endpoint] = nil
                default:
                    break
                }
            }
        }
        connections[endpoint] = conn
        conn.start(queue: .main)
    }

    private func publish() {
        let urls = resolved.values.sorted()
        if urls != uploadURLs {
            uploadURLs = urls
        }
    }

    deinit {
        browser?.cancel()
        for conn in connections.values {
            conn.cancel()
        }
    }
}
