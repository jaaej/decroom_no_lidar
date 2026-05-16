//
//  YoloRunner.swift
//  decroom_no_lidar
//
//  Core ML YOLOv8-seg inference wrapper. Accepts BGRA 640x640 pixel buffers,
//  returns [YoloDetection] filtered by confidence threshold.
//

import Foundation
import CoreML
import CoreVideo
import CoreImage

enum YoloRunnerError: Error {
    case modelLoadFailed
    case inferenceFailed
    case unexpectedOutput
}

final class YoloRunner: @unchecked Sendable {
    private let model: MLModel
    let inputSize = CGSize(width: 640, height: 640)
    let confidenceThreshold: Float

    init(confidenceThreshold: Float = 0.35) throws {
        self.confidenceThreshold = confidenceThreshold
        let config = MLModelConfiguration()
        config.computeUnits = .all
        guard let url = Bundle.main.url(forResource: "YOLOv8s-seg", withExtension: "mlmodelc") else {
            throw YoloRunnerError.modelLoadFailed
        }
        self.model = try MLModel(contentsOf: url, configuration: config)
    }

    /// Run inference on a BGRA 640x640 pixel buffer. Returns detections with
    /// confidence >= threshold.
    func infer(pixelBuffer: CVPixelBuffer, frameIndex: Int) throws -> [YoloDetection] {
        let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        let input = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
        let output = try model.prediction(from: input)

        // Ultralytics Core ML export with NMS has two outputs:
        //   "confidence": MLMultiArray [1, numBoxes, 80]
        //   "coordinates": MLMultiArray [1, numBoxes, 4] (x, y, w, h normalized)
        // (If export was without NMS, there is a single raw output needing NMS here —
        //  Ultralytics `nms=True` during export is required for this path.)
        guard let confidenceFeature = output.featureValue(for: "confidence")?.multiArrayValue,
              let coordinatesFeature = output.featureValue(for: "coordinates")?.multiArrayValue else {
            throw YoloRunnerError.unexpectedOutput
        }

        return decodeDetections(
            confidence: confidenceFeature,
            coordinates: coordinatesFeature,
            frameIndex: frameIndex
        )
    }

    private func decodeDetections(
        confidence: MLMultiArray,
        coordinates: MLMultiArray,
        frameIndex: Int
    ) -> [YoloDetection] {
        // Ultralytics CoreML NMS export shape varies by version. Handle both
        // 2D [numBoxes, numClasses] and 3D [batch, numBoxes, numClasses].
        let confShape = confidence.shape.map { $0.intValue }
        let numBoxes: Int
        let numClasses: Int
        switch confShape.count {
        case 2:
            numBoxes = confShape[0]
            numClasses = confShape[1]
        case 3:
            numBoxes = confShape[1]
            numClasses = confShape[2]
        default:
            return []  // unexpected shape, skip this frame
        }

        // Sanity bounds — bail if either dim is zero or implausible.
        guard numBoxes > 0, numBoxes < 10_000,
              numClasses > 0, numClasses <= 1000 else {
            return []
        }

        let confPtr = confidence.dataPointer.assumingMemoryBound(to: Float.self)
        let coordPtr = coordinates.dataPointer.assumingMemoryBound(to: Float.self)
        let confTotal = confidence.count
        let coordTotal = coordinates.count

        var detections: [YoloDetection] = []
        for b in 0..<numBoxes {
            let confBase = b * numClasses
            // Defensive bounds guard against malformed multi-arrays.
            guard confBase + numClasses <= confTotal else { break }

            var maxConf: Float = 0
            var maxClass = 0
            for c in 0..<numClasses {
                let conf = confPtr[confBase + c]
                if conf > maxConf {
                    maxConf = conf
                    maxClass = c
                }
            }
            guard maxConf >= confidenceThreshold else { continue }
            let category = maxClass < COCO80.classes.count ? COCO80.classes[maxClass] : "unknown"

            let coordBase = b * 4
            guard coordBase + 4 <= coordTotal else { break }

            let cx = coordPtr[coordBase + 0]
            let cy = coordPtr[coordBase + 1]
            let w = coordPtr[coordBase + 2]
            let h = coordPtr[coordBase + 3]
            let rect = CGRect(
                x: CGFloat(cx - w / 2),
                y: CGFloat(cy - h / 2),
                width: CGFloat(w),
                height: CGFloat(h)
            )

            detections.append(YoloDetection(
                category: category,
                confidence: maxConf,
                bbox: rect,
                mask: nil,
                frameIndex: frameIndex
            ))
        }
        return detections
    }
}
