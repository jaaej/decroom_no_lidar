//
//  PointCloudAccumulator.swift
//  decroom_no_lidar
//
//  Accumulates world-space 3D points from unprojected metric depth maps
//  (meters, from DepthRunner). Voxel-deduplicates at configurable grid
//  resolution. Caps total points to bound memory. Exports as binary PLY
//  for backend upload.
//

import Foundation
import simd

struct Intrinsics {
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float
    let width: Int
    let height: Int
}

final class PointCloudAccumulator {
    private let voxelSize: Float
    private let maxPoints: Int

    /// Voxel key → (summed position, summed color, count)
    private var voxels: [Int64: (position: SIMD3<Double>, color: SIMD3<Double>, count: Int)] = [:]

    var count: Int { voxels.count }

    init(voxelSize: Float = 0.02, maxPoints: Int = 500_000) {
        self.voxelSize = voxelSize
        self.maxPoints = maxPoints
    }

    /// Unproject a depth pixel to world space.
    func unproject(u: Int, v: Int, depth: Float,
                    intrinsics i: Intrinsics,
                    worldFromCamera pose: simd_float4x4) -> SIMD3<Float> {
        let camX = (Float(u) - i.cx) * depth / i.fx
        let camY = (Float(v) - i.cy) * depth / i.fy
        // Camera-space Z is -depth (camera looks down -Z in ARKit convention)
        let camPoint = SIMD4<Float>(camX, camY, -depth, 1)
        let worldPoint = pose * camPoint
        return SIMD3(worldPoint.x, worldPoint.y, worldPoint.z)
    }

    /// Add a world-space point with its color.
    func addPoint(_ world: SIMD3<Float>, color: SIMD3<UInt8>) {
        let key = voxelKey(for: world)
        let pos = SIMD3<Double>(Double(world.x), Double(world.y), Double(world.z))
        let col = SIMD3<Double>(Double(color.x), Double(color.y), Double(color.z))
        if let existing = voxels[key] {
            voxels[key] = (existing.position + pos, existing.color + col, existing.count + 1)
        } else {
            if voxels.count >= maxPoints {
                // Drop a random voxel to make room. Simple eviction — could be improved.
                if let evictKey = voxels.keys.randomElement() {
                    voxels.removeValue(forKey: evictKey)
                }
            }
            voxels[key] = (pos, col, 1)
        }
    }

    /// Add a batch of points from a depth map.
    func addDepthMap(_ depths: [Float],
                     width: Int, height: Int,
                     intrinsics: Intrinsics,
                     worldFromCamera pose: simd_float4x4,
                     colorAt: (Int, Int) -> SIMD3<UInt8>,
                     pixelStride: Int = 4,
                     maxDepth: Float = 5.0) {
        var v = 0
        while v < height {
            var u = 0
            while u < width {
                let d = depths[v * width + u]
                if d.isFinite, d > 0.1, d < maxDepth {
                    let world = unproject(u: u, v: v, depth: d, intrinsics: intrinsics, worldFromCamera: pose)
                    addPoint(world, color: colorAt(u, v))
                }
                u += pixelStride
            }
            v += pixelStride
        }
    }

    /// Export as binary little-endian PLY with x,y,z,r,g,b per vertex.
    func exportPLY() -> Data {
        var output = Data()

        // Compute averaged points first
        let averaged: [(pos: SIMD3<Float>, col: SIMD3<UInt8>)] = voxels.values.map { entry in
            let n = Double(entry.count)
            let pos = SIMD3<Float>(Float(entry.position.x / n),
                                     Float(entry.position.y / n),
                                     Float(entry.position.z / n))
            let col = SIMD3<UInt8>(UInt8(min(255, Int(entry.color.x / n))),
                                     UInt8(min(255, Int(entry.color.y / n))),
                                     UInt8(min(255, Int(entry.color.z / n))))
            return (pos, col)
        }

        // ASCII header
        let header = """
        ply
        format binary_little_endian 1.0
        element vertex \(averaged.count)
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header

        """
        output.append(header.data(using: .ascii)!)

        // Binary payload
        for (pos, col) in averaged {
            var x = pos.x, y = pos.y, z = pos.z
            var r = col.x, g = col.y, b = col.z
            withUnsafeBytes(of: &x) { output.append(contentsOf: $0) }
            withUnsafeBytes(of: &y) { output.append(contentsOf: $0) }
            withUnsafeBytes(of: &z) { output.append(contentsOf: $0) }
            output.append(r)
            output.append(g)
            output.append(b)
        }

        return output
    }

    private func voxelKey(for p: SIMD3<Float>) -> Int64 {
        let vx = Int32(floor(p.x / voxelSize))
        let vy = Int32(floor(p.y / voxelSize))
        let vz = Int32(floor(p.z / voxelSize))
        // Pack three int32s into int64 (truncates high bits, acceptable for ±32K voxels)
        return (Int64(vx) & 0x1FFFFF) << 42 | (Int64(vy) & 0x1FFFFF) << 21 | (Int64(vz) & 0x1FFFFF)
    }
}
