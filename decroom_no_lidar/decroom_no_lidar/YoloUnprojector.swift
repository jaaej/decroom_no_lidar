//
//  YoloUnprojector.swift
//  decroom_no_lidar
//
//  Pure function: convert a YOLO bbox center + a depth map into a 3D
//  world position. No ARKit dependency — testable without a device.
//

import Foundation
import simd

enum YoloUnprojector {
    /// Unproject a YOLO bbox center to a world-space position.
    ///
    /// - Parameters:
    ///   - yoloBboxCenterNormalized: YOLO bbox center in [0, 1] × [0, 1]
    ///     (normalized image coordinates in the YOLO input frame).
    ///   - depthMap: row-major metric depth map (meters).
    ///   - depthWidth / depthHeight: depth map size.
    ///   - depthIntrinsics: camera intrinsics aligned to the depth map
    ///     resolution (cx/cy/fx/fy in depth-map pixel units).
    ///   - worldFromCamera: 4×4 ARKit pose matrix (right-handed, Y up).
    /// - Returns: 3D world position, or nil if depth at the centroid is
    ///   invalid (0, NaN, infinite).
    static func worldPosition(
        yoloBboxCenterNormalized: SIMD2<Float>,
        depthMap: [Float],
        depthWidth: Int,
        depthHeight: Int,
        depthIntrinsics: Intrinsics,
        worldFromCamera: simd_float4x4
    ) -> SIMD3<Float>? {
        guard depthWidth > 0, depthHeight > 0,
              depthMap.count == depthWidth * depthHeight else { return nil }

        // Clamp normalized coords into [0, 1] defensively (YOLO bboxes should
        // be in-range, but NMS rarely emits off-by-epsilon values).
        let nx = max(0.0, min(1.0, yoloBboxCenterNormalized.x))
        let ny = max(0.0, min(1.0, yoloBboxCenterNormalized.y))

        let px = min(depthWidth - 1, Int(nx * Float(depthWidth)))
        let py = min(depthHeight - 1, Int(ny * Float(depthHeight)))

        let depth = depthMap[py * depthWidth + px]
        guard depth.isFinite, depth > 0 else { return nil }

        // Pinhole unprojection in camera space (ARKit: camera looks down -Z).
        let xc = (Float(px) - depthIntrinsics.cx) * depth / depthIntrinsics.fx
        let yc = (Float(py) - depthIntrinsics.cy) * depth / depthIntrinsics.fy
        let zc = -depth
        let camPoint = SIMD4<Float>(xc, yc, zc, 1.0)

        let world4 = worldFromCamera * camPoint
        return SIMD3<Float>(world4.x, world4.y, world4.z)
    }
}
