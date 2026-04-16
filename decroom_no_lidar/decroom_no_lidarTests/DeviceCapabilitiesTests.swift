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

    // MARK: - Machine identifier mapping tests

    @Test func a17ProIdentifier() {
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone16,1") == .a17pro)
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone16,2") == .a17pro)
    }

    @Test func a16Identifier() {
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone15,2") == .a16)
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone15,5") == .a16)
    }

    @Test func a15Identifier() {
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone14,2") == .a15)
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone14,8") == .a15)
    }

    @Test func a14Identifier() {
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone13,1") == .a14)
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone13,4") == .a14)
    }

    @Test func a13Identifier() {
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone12,1") == .a13)
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone12,8") == .a13)
    }

    @Test func a12Identifier() {
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone11,2") == .a12)
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone11,8") == .a12)
    }

    @Test func a11Identifier() {
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone10,1") == .a11)
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone10,6") == .a11)
    }

    @Test func unknownIdentifierFallsBack() {
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPhone99,1") == .unknown)
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "") == .unknown)
        #expect(DeviceCapabilities.chipGeneration(forMachineIdentifier: "iPad13,1") == .unknown)
    }
}
