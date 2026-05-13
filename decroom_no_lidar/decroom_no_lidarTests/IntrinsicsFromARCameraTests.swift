import Testing
import simd
import Foundation
import CoreGraphics
@testable import decroom_no_lidar

struct IntrinsicsFromARCameraTests {

    @Test func landscapeIntrinsicsPassThrough() {
        // Fake intrinsics: fx=1200, fy=1200, cx=960, cy=720 (1920x1440 landscape)
        let raw = simd_float3x3(rows: [
            SIMD3<Float>(1200, 0, 960),
            SIMD3<Float>(0, 1200, 720),
            SIMD3<Float>(0, 0, 1)
        ])
        let result = Intrinsics.make(
            fromRawIntrinsics: raw,
            imageResolution: CGSize(width: 1920, height: 1440),
            portrait: false
        )
        #expect(result.fx == 1200)
        #expect(result.fy == 1200)
        #expect(result.cx == 960)
        #expect(result.cy == 720)
        #expect(result.width == 1920)
        #expect(result.height == 1440)
    }

    @Test func portraitIntrinsicsSwapFxFyAndRotateCxCy() {
        // Same raw sensor intrinsics, but phone is portrait.
        // Conventions (from CLAUDE.md):
        //   new_fx = old_fy
        //   new_fy = old_fx
        //   new_cx = old_cy
        //   new_cy = old_width - old_cx
        let raw = simd_float3x3(rows: [
            SIMD3<Float>(1200, 0, 960),
            SIMD3<Float>(0, 1200, 720),
            SIMD3<Float>(0, 0, 1)
        ])
        let result = Intrinsics.make(
            fromRawIntrinsics: raw,
            imageResolution: CGSize(width: 1920, height: 1440),
            portrait: true
        )
        #expect(result.fx == 1200)  // old_fy
        #expect(result.fy == 1200)  // old_fx
        #expect(result.cx == 720)   // old_cy
        #expect(result.cy == 1920 - 960)  // = 960
        #expect(result.width == 1440)  // swapped
        #expect(result.height == 1920)
    }

    @Test func differentFxFyPortraitSwapsCorrectly() {
        let raw = simd_float3x3(rows: [
            SIMD3<Float>(1500, 0, 800),
            SIMD3<Float>(0, 1000, 400),
            SIMD3<Float>(0, 0, 1)
        ])
        let result = Intrinsics.make(
            fromRawIntrinsics: raw,
            imageResolution: CGSize(width: 1600, height: 1200),
            portrait: true
        )
        #expect(result.fx == 1000)  // old_fy
        #expect(result.fy == 1500)  // old_fx
        #expect(result.cx == 400)   // old_cy
        #expect(result.cy == 1600 - 800)  // = 800
    }
}
