//
//  ServerKey.swift
//  decroom_no_lidar
//
//  The shared "server key" between the app and the Decroom backend.
//  Stored in the Keychain and never sent over the network:
//    - uploads carry X-Decroom-Auth "<ts>:<nonce>:<hmac>", a single-use
//      HMAC-SHA256 signature of "decroom-upload:<ts>:<nonce>";
//    - a discovered server must answer GET /health?nonce=N with
//      HMAC-SHA256(key, "decroom-health:N") before we upload to it.
//  Must stay in sync with backend/main.py.
//

import CryptoKit
import Foundation
import Security

enum ServerKey {
    static let authHeader = "X-Decroom-Auth"

    private static let service = "com.jaejeon.decroom-no-lidar.server-key"
    private static let account = "default"

    // MARK: - Keychain

    static func load() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let key = String(data: data, encoding: .utf8),
              !key.isEmpty else {
            return nil
        }
        return key
    }

    /// Saves a trimmed key; an empty string removes it.
    @discardableResult
    static func save(_ rawKey: String) -> Bool {
        delete()
        let key = rawKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { return true }
        let attributes: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
            kSecValueData as String: Data(key.utf8),
        ]
        return SecItemAdd(attributes as CFDictionary, nil) == errSecSuccess
    }

    static func delete() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }

    // MARK: - Crypto

    /// First 8 hex chars of SHA-256(key). The backend prints the same value
    /// at startup, so the user can check the two match without revealing it.
    static func fingerprint(_ key: String) -> String {
        String(hex(SHA256.hash(data: Data(key.utf8))).prefix(8))
    }

    static func hmacHex(_ message: String, key: String) -> String {
        let mac = HMAC<SHA256>.authenticationCode(
            for: Data(message.utf8),
            using: SymmetricKey(data: Data(key.utf8))
        )
        return hex(mac)
    }

    static func uploadAuthValue(key: String, now: Date = Date()) -> String {
        let ts = String(Int(now.timeIntervalSince1970))
        let nonce = randomNonce()
        let sig = hmacHex("decroom-upload:\(ts):\(nonce)", key: key)
        return "\(ts):\(nonce):\(sig)"
    }

    static func expectedHealthProof(nonce: String, key: String) -> String {
        hmacHex("decroom-health:\(nonce)", key: key)
    }

    static func randomNonce() -> String {
        hex(SymmetricKey(size: .bits128).withUnsafeBytes { Data($0) })
    }

    private static func hex<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
        bytes.map { String(format: "%02x", $0) }.joined()
    }
}
