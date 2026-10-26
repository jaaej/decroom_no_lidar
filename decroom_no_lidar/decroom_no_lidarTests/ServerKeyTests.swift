import Testing
import Foundation
@testable import decroom_no_lidar

/// Expected values computed with Python's hmac/hashlib, i.e. what
/// backend/main.py computes for the same key and message.
struct ServerKeyTests {

    let key = "test-server-key"

    @Test func healthProofMatchesBackend() {
        #expect(ServerKey.expectedHealthProof(nonce: "abc123", key: key)
                == "e9c75312932a26ad9263c1b2d9a6b82646884841eff442cd76f40b50271b71f6")
    }

    @Test func uploadSignatureMatchesBackend() {
        #expect(ServerKey.hmacHex("decroom-upload:1700000000:00112233445566778899aabbccddeeff", key: key)
                == "872ae4ce464e6e1f1295728f0ab2c2031e558bfe07e0ee0f714e2317b8cee4e1")
    }

    @Test func fingerprintMatchesBackend() {
        #expect(ServerKey.fingerprint(key) == "1b7aa20a")
    }

    @Test func uploadAuthValueHasTimestampNonceAndSignature() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let parts = ServerKey.uploadAuthValue(key: key, now: now).split(separator: ":").map(String.init)
        #expect(parts.count == 3)
        #expect(parts[0] == "1700000000")
        #expect(parts[1].count == 32)
        #expect(parts[2] == ServerKey.hmacHex("decroom-upload:\(parts[0]):\(parts[1])", key: key))
    }

    @Test func noncesAreUnique() {
        #expect(ServerKey.randomNonce() != ServerKey.randomNonce())
    }
}
