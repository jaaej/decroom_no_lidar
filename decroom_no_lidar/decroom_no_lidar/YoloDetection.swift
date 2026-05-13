//
//  YoloDetection.swift
//  decroom_no_lidar
//
//  Plain struct for a single YOLO detection: category, confidence, bbox, mask.
//

import Foundation
import CoreGraphics

struct YoloDetection {
    /// COCO-80 category name (e.g., "bed", "chair", "couch").
    let category: String
    /// 0..1 confidence.
    let confidence: Float
    /// Normalized bbox in [0,1] image coordinates (x, y, width, height).
    let bbox: CGRect
    /// Optional raw mask data (model dependent — nil if NMS-only export).
    let mask: [Float]?
    /// Source frame index.
    let frameIndex: Int
}

/// COCO-80 class index → name. Index matches Ultralytics ordering.
enum COCO80 {
    static let classes: [String] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
}
