//
//  DebugScanRecorder.swift
//  decroom_no_lidar
//
//  Phase-0 spike: minimal ARKit debug recorder.
//

import ARKit
import AVFoundation

/// Phase-0 debug recorder: HEVC video + per-frame pose/intrinsics JSONL.
/// Spike-quality; records every 2nd ARFrame, copies pixel buffers via a
/// pool so ARKit's camera pool is never retained (spec §2).
final class DebugScanRecorder {
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var jsonl: FileHandle?
    private var ptsJsonl: FileHandle?
    private(set) var dir: URL?
    private var frameCounter = 0
    private var appended = 0, dropped = 0
    private var startTime: TimeInterval?
    private var thermalEvents: [[String: Any]] = []
    private var lastThermal: String = "nominal"
    // Cross-thread stop guard: ARKit delivers frames on a background queue,
    // while stop() runs on the main queue. This is a plain (unsynchronized)
    // Bool — acceptable for spike code since the worst case is a single
    // stray append racing the flag flip, not a torn read.
    private var isStopped = false

    func start() throws {
        // Millisecond resolution: two starts within the same second would
        // otherwise collide on the same "scan_<seconds>" folder and throw
        // when video.mov already exists.
        let stamp = Int(Date().timeIntervalSince1970 * 1000)
        let d = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("scan_\(stamp)", isDirectory: true)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        dir = d
        let w = try AVAssetWriter(outputURL: d.appendingPathComponent("video.mov"), fileType: .mov)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: 1920, AVVideoHeightKey: 1440,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2],
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 8_000_000]]
        let inp = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        inp.expectsMediaDataInRealTime = true
        let ad = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: inp,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String:
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                kCVPixelBufferWidthKey as String: 1920,
                kCVPixelBufferHeightKey as String: 1440])
        w.add(inp)
        w.startWriting()
        writer = w; input = inp; adaptor = ad
        FileManager.default.createFile(atPath: d.appendingPathComponent("frames.jsonl").path, contents: nil)
        jsonl = try FileHandle(forWritingTo: d.appendingPathComponent("frames.jsonl"))
        FileManager.default.createFile(atPath: d.appendingPathComponent("points.jsonl").path, contents: nil)
        ptsJsonl = try FileHandle(forWritingTo: d.appendingPathComponent("points.jsonl"))
    }

    func append(_ frame: ARFrame) {
        guard !isStopped else { return }
        frameCounter += 1
        guard frameCounter % 2 == 0, let input = input, let adaptor = adaptor else { return }
        if startTime == nil {
            startTime = frame.timestamp
            writer?.startSession(atSourceTime: CMTime(seconds: 0, preferredTimescale: 600))
        }
        guard input.isReadyForMoreMediaData else { dropped += 1; return }
        guard let pool = adaptor.pixelBufferPool else { dropped += 1; return }
        var copy: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &copy)
        guard let dst = copy else { dropped += 1; return }
        // Copy both planes so ARKit's buffer is released immediately.
        // Clamp to min(src, dst) height/row-bytes: the pool is fixed at
        // 1920x1440, but capturedImage resolution varies by device/format,
        // so the source can be smaller OR larger than the pool buffer.
        let src = frame.capturedImage
        CVPixelBufferLockBaseAddress(src, .readOnly); CVPixelBufferLockBaseAddress(dst, [])
        for plane in 0..<2 {
            let h = min(CVPixelBufferGetHeightOfPlane(src, plane), CVPixelBufferGetHeightOfPlane(dst, plane))
            let bpr = min(CVPixelBufferGetBytesPerRowOfPlane(src, plane),
                          CVPixelBufferGetBytesPerRowOfPlane(dst, plane))
            guard let srcBase = CVPixelBufferGetBaseAddressOfPlane(src, plane),
                  let dstBase = CVPixelBufferGetBaseAddressOfPlane(dst, plane) else { continue }
            let srcRowBytes = CVPixelBufferGetBytesPerRowOfPlane(src, plane)
            let dstRowBytes = CVPixelBufferGetBytesPerRowOfPlane(dst, plane)
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstRowBytes),
                       srcBase.advanced(by: row * srcRowBytes), bpr)
            }
        }
        CVPixelBufferUnlockBaseAddress(src, .readOnly); CVPixelBufferUnlockBaseAddress(dst, [])
        let t = frame.timestamp - startTime!
        if adaptor.append(dst, withPresentationTime: CMTime(seconds: t, preferredTimescale: 600)) {
            appended += 1
            let p = frame.camera.transform, K = frame.camera.intrinsics
            let pose = (0..<4).flatMap { c in (0..<4).map { r in p[c][r] } }
            let tracking: String
            switch frame.camera.trackingState {
            case .normal: tracking = "normal"
            default: tracking = "limited"
            }
            let thermal = Self.thermalString(ProcessInfo.processInfo.thermalState)
            if thermal != lastThermal {
                thermalEvents.append(["t": frame.timestamp, "thermal": thermal])
                lastThermal = thermal
            }
            let line: [String: Any] = ["t": frame.timestamp, "pose": pose,
                "K": [K[0][0], K[1][1], K[2][0], K[2][1]], "tracking": tracking,
                "thermal": thermal]
            if let data = try? JSONSerialization.data(withJSONObject: line) {
                jsonl?.write(data); jsonl?.write("\n".data(using: .utf8)!)
            }
            if frameCounter % 10 == 0, let pts = frame.rawFeaturePoints?.points.prefix(200) {
                let obj: [String: Any] = ["t": frame.timestamp,
                    "pts": pts.map { [$0.x, $0.y, $0.z] }]
                if let data = try? JSONSerialization.data(withJSONObject: obj) {
                    ptsJsonl?.write(data); ptsJsonl?.write("\n".data(using: .utf8)!)
                }
            }
        } else { dropped += 1 }
    }

    func stop(completion: @escaping (URL?) -> Void) {
        isStopped = true
        guard let w = writer, let d = dir else { completion(nil); return }
        input?.markAsFinished()
        w.finishWriting {
            let stats: [String: Any] = ["frames_appended": self.appended,
                                        "frames_dropped_by_encoder": self.dropped,
                                        "thermal_events": self.thermalEvents]
            try? JSONSerialization.data(withJSONObject: stats)
                .write(to: d.appendingPathComponent("stats.json"))
            try? self.jsonl?.close(); try? self.ptsJsonl?.close()
            completion(d)
        }
    }

    private static func thermalString(_ t: ProcessInfo.ThermalState) -> String {
        switch t {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "nominal"
        }
    }
}
