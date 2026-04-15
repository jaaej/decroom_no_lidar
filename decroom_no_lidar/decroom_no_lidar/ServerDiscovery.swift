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

    /// The discovered upload URL, e.g. "http://192.168.1.42:8000/api/upload".
    /// Nil until a server is found on the network.
    @Published private(set) var uploadURL: String?

    /// Whether actively searching.
    @Published private(set) var isSearching = false

    private var browser: NWBrowser?
    private var connection: NWConnection?

    private init() {
        startBrowsing()
    }

    func startBrowsing() {
        // Cancel any previous browser
        browser?.cancel()

        let params = NWParameters()
        params.includePeerToPeer = true
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
        // Pick the first result
        guard let result = results.first else {
            uploadURL = nil
            return
        }

        // Resolve the endpoint to get IP + port
        let conn = NWConnection(to: result.endpoint, using: .tcp)
        conn.stateUpdateHandler = { [weak self] state in
            Task { @MainActor in
                if case .ready = state {
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
                        self?.uploadURL = "http://\(ip):\(port)/api/upload"
                        print("[ServerDiscovery] Found backend at \(self?.uploadURL ?? "?")")
                    }
                    conn.cancel()
                }
            }
        }
        conn.start(queue: .main)
        connection = conn
    }

    deinit {
        browser?.cancel()
        connection?.cancel()
    }
}
