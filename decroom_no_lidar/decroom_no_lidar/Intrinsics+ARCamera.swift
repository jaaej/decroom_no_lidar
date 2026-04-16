//
//  Intrinsics+ARCamera.swift
//  decroom_no_lidar
//
//  Build a PointCloudAccumulator.Intrinsics from ARCamera, handling the
//  landscape-to-portrait rotation ARKit requires for vertical-held phones.
//
//  Per CLAUDE.md:
//    new_fx = old_fy
//    new_fy = old_fx
//    new_cx = old_cy
//    new_cy = old_width - old_cx
//

import Foundation
import ARKit
import CoreGraphics
import simd

extension Intrinsics {
    /// Build from an ARCamera. Pass `portrait: true` when the phone is held vertically
    /// (the capturedImage has been `.oriented(.right)` rotated to portrait before passing
    /// to depth/detection models).
    static func make(from camera: ARCamera, portrait: Bool) -> Intrinsics {
        return make(
            fromRawIntrinsics: camera.intrinsics,
            imageResolution: camera.imageResolution,
            portrait: portrait
        )
    }

    /// Exposed for testing — takes raw intrinsics + resolution directly.
    static func make(
        fromRawIntrinsics raw: simd_float3x3,
        imageResolution res: CGSize,
        portrait: Bool
    ) -> Intrinsics {
        let fx = raw.columns.0.x
        let fy = raw.columns.1.y
        let cx = raw.columns.2.x
        let cy = raw.columns.2.y
        let oldW = Int(res.width)
        let oldH = Int(res.height)

        if portrait {
            return Intrinsics(
                fx: fy,
                fy: fx,
                cx: cy,
                cy: Float(oldW) - cx,
                width: oldH,
                height: oldW
            )
        } else {
            return Intrinsics(
                fx: fx,
                fy: fy,
                cx: cx,
                cy: cy,
                width: oldW,
                height: oldH
            )
        }
    }
}
