//
//  PixelBufferRGB.swift
//  decroom_no_lidar
//
//  Converts ARKit YUV capturedImage to a BGRA CVPixelBuffer suitable
//  for Core ML models. Also handles portrait rotation via CIImage.oriented(.right).
//

import Foundation
import CoreVideo
import CoreImage
import VideoToolbox

enum PixelBufferRGB {

    /// Convert a YUV capturedImage to a BGRA CVPixelBuffer of the given size.
    /// Rotates via `.oriented(.right)` so the output is portrait when the
    /// input was landscape-right (ARKit's default sensor orientation).
    ///
    /// - Parameters:
    ///   - source: ARKit's `ARFrame.capturedImage`
    ///   - size: Target size (e.g., 518x392 for DepthAnythingV2Small)
    ///   - ciContext: shared CIContext for reuse
    ///   - rotateToPortrait: apply `.oriented(.right)` before resize
    static func convert(
        _ source: CVPixelBuffer,
        toSize size: CGSize,
        ciContext: CIContext,
        rotateToPortrait: Bool
    ) -> CVPixelBuffer? {
        var ciImage = CIImage(cvPixelBuffer: source)
        if rotateToPortrait {
            ciImage = ciImage.oriented(.right)
        }
        // Scale to target size
        let sx = size.width / ciImage.extent.width
        let sy = size.height / ciImage.extent.height
        ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        // Create output buffer (BGRA)
        var output: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &output
        )
        guard status == kCVReturnSuccess, let output else { return nil }

        ciContext.render(ciImage, to: output)
        return output
    }
}
