import Testing
@testable import decroom_no_lidar

struct DeviceCapabilitiesTests {

    @Test func tierForA17ProIsFull() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .a17pro)
        #expect(tier == .full)
    }

    @Test func tierForA15IsFull() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .a15)
        #expect(tier == .full)
    }

    @Test func tierForA14IsFull() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .a14)
        #expect(tier == .full)
    }

    @Test func tierForA12IsReduced() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .a12)
        #expect(tier == .reduced)
    }

    @Test func tierForA13IsReduced() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .a13)
        #expect(tier == .reduced)
    }

    @Test func tierForA11IsFallback() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .a11)
        #expect(tier == .fallback)
    }

    @Test func tierForUnknownIsFallback() {
        let tier = DeviceCapabilities.tier(forChipGeneration: .unknown)
        #expect(tier == .fallback)
    }
}
