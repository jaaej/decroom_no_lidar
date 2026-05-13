import Testing
@testable import decroom_no_lidar

struct DeviceInfoCurrentTests {

    @Test func currentReturnsNonEmptyFields() {
        let info = DeviceInfo.current()
        #expect(!info.model.isEmpty)
        #expect(!info.ios.isEmpty)
        #expect(!info.chip.isEmpty)
    }

    @Test func currentTierMatchesDeviceCapabilities() {
        let info = DeviceInfo.current()
        #expect(info.tier == DeviceCapabilities.currentTier)
    }

    @Test func iosStringLooksLikeVersion() {
        let info = DeviceInfo.current()
        // Examples: "17.2", "18.1.1". Should start with a digit.
        let firstChar = info.ios.first
        #expect(firstChar?.isNumber == true)
    }
}
