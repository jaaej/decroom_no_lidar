//
//  DepthRunner.swift
//  decroom_no_lidar
//
//  Wraps Depth Anything V2 Small Core ML for on-device relative depth inference.
//  Output is relative (not metric) — scale calibration happens downstream using
//  ARKit feature point depths. Adjusts inference rate based on thermal state.
//

import Foundation
import CoreML
import CoreVideo
import Vision

struct DepthResult {
    let width: Int
    let height: Int
    let depths: [Float]  // Row-major width*height relative depth values
}

enum DepthRunnerError: Error {
    case modelLoadFailed
    case inferenceFailed
    case unexpectedOutput
}

final class DepthRunner {
    private let model: MLModel

    /// Target time between inference calls (seconds). Adjusted by thermal state.
    private(set) var targetFrameInterval: TimeInterval = 0.2  // 5fps default

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Neural Engine if available
        guard let url = Bundle.main.url(forResource: "DepthModel", withExtension: "mlmodelc"),
              let loaded = try? MLModel(contentsOf: url, configuration: config) else {
            throw DepthRunnerError.modelLoadFailed
        }
        self.model = loaded
    }

    /// Run inference on a pixel buffer. Returns relative depth map at the model's native resolution.
    func infer(pixelBuffer: CVPixelBuffer) throws -> DepthResult {
        let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        let input = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
        let output = try model.prediction(from: input)

        // Depth output may be either a grayscale image (CVPixelBuffer) or a MLMultiArray,
        // depending on how the Core ML model was exported. Handle both.
        for name in output.featureNames {
            guard let feature = output.featureValue(for: name) else { continue }
            if let multiArray = feature.multiArrayValue {
                return try decodeMultiArray(multiArray)
            }
            if let imageBuffer = feature.imageBufferValue {
                return try decodePixelBuffer(imageBuffer)
            }
        }
        throw DepthRunnerError.unexpectedOutput
    }

    private func decodeMultiArray(_ multiArray: MLMultiArray) throws -> DepthResult {
        let shape = multiArray.shape.map { $0.intValue }
        let height: Int
        let width: Int
        switch shape.count {
        case 4: height = shape[2]; width = shape[3]
        case 3: height = shape[1]; width = shape[2]
        case 2: height = shape[0]; width = shape[1]
        default: throw DepthRunnerError.unexpectedOutput
        }

        let count = multiArray.count
        var depths = [Float](repeating: 0, count: count)
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            depths[i] = ptr[i]
        }

        return DepthResult(width: width, height: height, depths: depths)
    }

    private func decodePixelBuffer(_ pixelBuffer: CVPixelBuffer) throws -> DepthResult {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DepthRunnerError.unexpectedOutput
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        var depths = [Float](repeating: 0, count: width * height)

        switch format {
        case kCVPixelFormatType_OneComponent16Half:
            // Float16 grayscale — upcast to Float32.
            for y in 0..<height {
                let row = base.advanced(by: y * bytesPerRow)
                    .bindMemory(to: UInt16.self, capacity: width)
                for x in 0..<width {
                    depths[y * width + x] = Float(Float16(bitPattern: row[x]))
                }
            }
        case kCVPixelFormatType_OneComponent32Float, kCVPixelFormatType_DepthFloat32:
            for y in 0..<height {
                let row = base.advanced(by: y * bytesPerRow)
                    .bindMemory(to: Float.self, capacity: width)
                for x in 0..<width {
                    depths[y * width + x] = row[x]
                }
            }
        case kCVPixelFormatType_OneComponent8:
            for y in 0..<height {
                let row = base.advanced(by: y * bytesPerRow)
                    .bindMemory(to: UInt8.self, capacity: width)
                for x in 0..<width {
                    depths[y * width + x] = Float(row[x]) / 255.0
                }
            }
        default:
            throw DepthRunnerError.unexpectedOutput
        }

        return DepthResult(width: width, height: height, depths: depths)
    }

    /// Adjust inference rate based on thermal state. Call when thermal state changes.
    func updateThermalState(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal:
            targetFrameInterval = 0.2   // 5fps
        case .fair:
            targetFrameInterval = 0.3   // ~3fps
        case .serious:
            targetFrameInterval = 0.5   // 2fps
        case .critical:
            targetFrameInterval = 1.0   // 1fps
        @unknown default:
            targetFrameInterval = 0.3
        }
    }
}
