import Testing
import simd
import Foundation
@testable import decroom_no_lidar

struct PointCloudAccumulatorTests {

    @Test func unprojectAtImageCenterReturnsPointOnCameraAxis() {
        // Intrinsics: fx=500, fy=500, cx=256, cy=256 (fake 512x512 cam)
        // Identity pose: camera at origin looking down -Z
        let intrinsics = Intrinsics(fx: 500, fy: 500, cx: 256, cy: 256, width: 512, height: 512)
        let pose = matrix_identity_float4x4
        let accumulator = PointCloudAccumulator(voxelSize: 0.02, maxPoints: 10_000)
        let pt = accumulator.unproject(u: 256, v: 256, depth: 2.0, intrinsics: intrinsics, worldFromCamera: pose)
        // Camera looks down -Z, so depth 2.0 at center should give (0, 0, -2)
        #expect(abs(pt.x) < 1e-4)
        #expect(abs(pt.y) < 1e-4)
        #expect(abs(pt.z + 2.0) < 1e-4)
    }

    @Test func addPointsVoxelizesDuplicates() {
        let accumulator = PointCloudAccumulator(voxelSize: 0.02, maxPoints: 10_000)
        // Add 100 points all within one voxel
        for _ in 0..<100 {
            accumulator.addPoint(SIMD3(0.001, 0.001, 0.001),
                                  color: SIMD3(128, 128, 128))
        }
        #expect(accumulator.count == 1)
    }

    @Test func addPointsRespectsMaxLimit() {
        let accumulator = PointCloudAccumulator(voxelSize: 0.001, maxPoints: 100)
        // Add 200 distinct points (each in its own voxel due to small voxel size)
        for i in 0..<200 {
            accumulator.addPoint(SIMD3(Float(i) * 0.01, 0, 0),
                                  color: SIMD3(128, 128, 128))
        }
        #expect(accumulator.count <= 100)
    }

    @Test func exportPLYProducesValidBinaryHeader() throws {
        let accumulator = PointCloudAccumulator(voxelSize: 0.02, maxPoints: 10_000)
        accumulator.addPoint(SIMD3(1, 2, 3), color: SIMD3(255, 0, 0))
        let data = accumulator.exportPLY()

        // Read header
        let headerRange = data.range(of: "end_header\n".data(using: .ascii)!)
        #expect(headerRange != nil)
        let headerStr = String(data: data[..<headerRange!.upperBound], encoding: .ascii)!
        #expect(headerStr.contains("ply"))
        #expect(headerStr.contains("format binary_little_endian 1.0"))
        #expect(headerStr.contains("element vertex 1"))
    }
}
