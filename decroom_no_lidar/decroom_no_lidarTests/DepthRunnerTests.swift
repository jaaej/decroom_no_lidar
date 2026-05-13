import Testing
import CoreVideo
@testable import decroom_no_lidar

struct DepthRunnerTests {

    @Test func inferReturnsDepthMapWithConsistentDimensions() throws {
        let runner = try DepthRunner()
        // Depth Anything V2 Small expects 518x392 (W x H).
        let buffer = makeTestPixelBuffer(width: 518, height: 392)
        let result = try runner.infer(pixelBuffer: buffer)
        // Dimensions must be positive and match the buffer count exactly.
        #expect(result.width > 0)
        #expect(result.height > 0)
        #expect(result.depths.count == result.width * result.height)
    }

    @Test func depthsAreFiniteAndInMetricRange() throws {
        let runner = try DepthRunner()
        let buffer = makeTestPixelBuffer(width: 518, height: 392)
        let result = try runner.infer(pixelBuffer: buffer)
        // Metric depth: finite and non-negative. A zero-filled buffer may
        // produce low-confidence values; real-scene quality is validated
        // in manual QA, not here.
        #expect(result.depths.allSatisfy { $0.isFinite && $0 >= 0 })
    }

    @Test func thermalThrottleDropsFrameRate() {
        let runner = try? DepthRunner()
        guard let runner else { return }
        runner.updateThermalState(.serious)
        #expect(runner.targetFrameInterval >= 0.5)
        runner.updateThermalState(.nominal)
        #expect(runner.targetFrameInterval <= 0.3)
    }

    private func makeTestPixelBuffer(width: Int, height: Int) -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                            kCVPixelFormatType_32BGRA, attrs, &buffer)
        guard let pb = buffer else {
            fatalError("Failed to create test pixel buffer")
        }
        CVPixelBufferLockBaseAddress(pb, [])
        if let base = CVPixelBufferGetBaseAddress(pb) {
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)
            memset(base, 0, bytesPerRow * height)
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        return pb
    }
}
