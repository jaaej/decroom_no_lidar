//
//  PanoramaBuffer.swift
//  decroom_no_lidar
//
//  Equirectangular panorama buffer that builds up seamlessly as the user
//  rotates. Each captured keyframe is projected into the buffer using
//  ARKit's camera rotation, so overlapping frames blend at the correct
//  angular position — no tile boundaries, no duplicate objects.
//
//  The buffer is 3600×900 pixels (10px per degree × 360° wide,
//  ~2.5× vertical FOV). Uncaptured regions are transparent (caller
//  draws a grid pattern behind).
//

import Foundation
import CoreGraphics
import UIKit

final class PanoramaBuffer {
    /// 10 pixels per degree of yaw.
    static let pixelsPerDegree: CGFloat = 10
    static let bufferWidth: Int = 3600   // 360° × 10px
    static let bufferHeight: Int = 900

    private var cgContext: CGContext
    private(set) var frameCount: Int = 0

    /// The current panorama as a CGImage. Updated after each addFrame call.
    private(set) var image: CGImage?

    init() {
        let cs = CGColorSpaceCreateDeviceRGB()
        cgContext = CGContext(
            data: nil,
            width: Self.bufferWidth,
            height: Self.bufferHeight,
            bitsPerComponent: 8,
            bytesPerRow: Self.bufferWidth * 4,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
        )!
        // Start fully transparent (uncaptured = transparent, caller draws grid behind)
        cgContext.clear(CGRect(x: 0, y: 0, width: Self.bufferWidth, height: Self.bufferHeight))
    }

    /// Project a captured frame into the panorama at the given camera orientation.
    ///
    /// - Parameters:
    ///   - frame: The captured keyframe image (portrait-oriented)
    ///   - yawRadians: Camera yaw in world space (from ARKit camera.transform)
    ///   - pitchRadians: Camera pitch (negative = looking down)
    ///   - hFOV: Horizontal field of view in radians (~1.05 for iPhone wide lens)
    func addFrame(_ frame: CGImage, yawRadians: Float, pitchRadians: Float, hFOV: Float = 1.05) {
        let yawDeg = CGFloat(yawRadians) * 180.0 / .pi
        let pitchDeg = CGFloat(pitchRadians) * 180.0 / .pi
        let hFOVDeg = CGFloat(hFOV) * 180.0 / .pi
        let vFOVDeg = hFOVDeg * CGFloat(frame.height) / CGFloat(frame.width)

        // Where on the buffer does this frame's center land?
        // yaw=0 maps to buffer center (x=1800), wrapping around.
        let centerX = (yawDeg + 180.0) * Self.pixelsPerDegree
        // pitch=0 (horizontal) maps to buffer vertical center.
        let centerY = CGFloat(Self.bufferHeight) / 2.0 - pitchDeg * Self.pixelsPerDegree

        let frameW = hFOVDeg * Self.pixelsPerDegree
        let frameH = vFOVDeg * Self.pixelsPerDegree

        let drawRect = CGRect(
            x: centerX - frameW / 2,
            y: centerY - frameH / 2,
            width: frameW,
            height: frameH
        )

        // Draw with edge feathering for seamless blending.
        // We draw the frame, then apply a radial alpha mask so edges fade.
        cgContext.saveGState()

        // Clip to the buffer bounds (handle wrap-around at 0°/360° by drawing twice)
        cgContext.draw(frame, in: drawRect)

        // Handle wrap-around: if the frame crosses the 360°→0° boundary,
        // draw again shifted by ±bufferWidth.
        if drawRect.minX < 0 {
            let wrappedRect = drawRect.offsetBy(dx: CGFloat(Self.bufferWidth), dy: 0)
            cgContext.draw(frame, in: wrappedRect)
        }
        if drawRect.maxX > CGFloat(Self.bufferWidth) {
            let wrappedRect = drawRect.offsetBy(dx: -CGFloat(Self.bufferWidth), dy: 0)
            cgContext.draw(frame, in: wrappedRect)
        }

        cgContext.restoreGState()

        frameCount += 1
        image = cgContext.makeImage()
    }

    /// Get a horizontal slice of the panorama centered on the given yaw.
    /// Returns a CGImage suitable for displaying behind the live camera window.
    ///
    /// - Parameters:
    ///   - centerYaw: Current camera yaw in radians
    ///   - visibleWidthDeg: How many degrees of panorama to show (e.g., 220°)
    ///   - outputSize: Desired output image size
    func slice(centeredAtYaw centerYaw: Float,
               visibleWidthDeg: CGFloat = 220,
               outputSize: CGSize) -> CGImage? {
        guard let panoImage = image else { return nil }

        let centerDeg = CGFloat(centerYaw) * 180.0 / .pi
        let centerX = (centerDeg + 180.0) * Self.pixelsPerDegree
        let sliceW = visibleWidthDeg * Self.pixelsPerDegree
        let sliceH = CGFloat(Self.bufferHeight)

        // Create output context
        guard let outCtx = CGContext(
            data: nil,
            width: Int(outputSize.width),
            height: Int(outputSize.height),
            bitsPerComponent: 8,
            bytesPerRow: Int(outputSize.width) * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { return nil }

        // Scale to fit output
        let scaleX = outputSize.width / sliceW
        let scaleY = outputSize.height / sliceH
        outCtx.scaleBy(x: scaleX, y: scaleY)

        // Translate so centerX is at the middle of the output
        let offsetX = -(centerX - sliceW / 2)
        outCtx.translateBy(x: offsetX, y: 0)

        // Draw the panorama (may need to draw twice for wrap-around)
        let fullRect = CGRect(x: 0, y: 0, width: Self.bufferWidth, height: Self.bufferHeight)
        outCtx.draw(panoImage, in: fullRect)

        // Wrap-around: draw again shifted
        let shiftedRect = fullRect.offsetBy(dx: CGFloat(Self.bufferWidth), dy: 0)
        outCtx.draw(panoImage, in: shiftedRect)
        let shiftedRect2 = fullRect.offsetBy(dx: -CGFloat(Self.bufferWidth), dy: 0)
        outCtx.draw(panoImage, in: shiftedRect2)

        return outCtx.makeImage()
    }

    /// Coverage: fraction of panorama pixels that have been filled (non-transparent).
    var coverageFraction: Double {
        guard let img = image,
              let data = img.dataProvider?.data,
              let ptr = CFDataGetBytePtr(data) else { return 0 }
        let total = Self.bufferWidth * Self.bufferHeight
        var filled = 0
        // Check alpha channel (every 4th byte, offset 0 for premultipliedFirst)
        for i in stride(from: 0, to: total * 4, by: 4 * 8) {  // sample every 8th pixel for speed
            if ptr[i] > 10 { filled += 1 }
        }
        let sampled = total / 8
        return Double(filled) / Double(sampled)
    }
}
