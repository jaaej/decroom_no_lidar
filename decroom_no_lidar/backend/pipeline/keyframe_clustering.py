"""Plan 3c: cross-keyframe NN-merge of lifted detections.

A real bed seen in 5 keyframes produces 5 LiftedDetection entries with
similar 3D positions. This module collapses them into one tracked
object before they enter Plan 3b's consensus pipeline.

Output shape mirrors the on-device ObjectTracker output (yolo_objects.json),
so the merged list can be appended to payload.yolo_objects unchanged.

Phase A0 also surfaced "visual confusion between storage categories"
(wardrobe / dresser / bookshelf often label the same physical object).
A `_RESOLUTION_FAMILIES` map collapses near-position detections within
a family even when category labels differ; the highest-confidence label
becomes the canonical category.
"""
from __future__ import annotations

import logging
import math
import uuid
from typing import Dict, List, Optional

from .raycast_lift import LiftedDetection

logger = logging.getLogger("decroom-backend.keyframe_clustering")

# Per-category 3D NN-merge distance for cross-keyframe clustering.
# Plan 3d retune: real scans upload ~66 keyframes (vs spec's projected 10),
# so per-object detection density is ~6× higher than originally tuned.
# Raycast through the bbox center hits different surfaces of the same
# physical object across keyframes (top, side, front), producing positions
# 0.5–1.5m apart for tall furniture. Thresholds bumped accordingly.
BACKEND_TRACKER_DISTANCES: Dict[str, float] = {
    "bed":          1.5,
    "sofa":         1.5,
    "couch":        1.5,
    "chair":        0.8,
    "stool":        0.8,
    "dining_table": 1.1,
    "desk":         1.1,
    "monitor":      0.8,
    "tv":           0.8,
    "laptop":       0.5,
    "lamp":         0.6,
    "lamp_floor":   0.6,
    "lamp_desk":    0.6,
    "potted plant": 0.6,
    "potted_plant": 0.6,
    "plant":        0.6,
    "bookshelf":    1.2,
    "dresser":      1.2,
    "wardrobe":     1.5,
    "nightstand":   0.8,
    "rug":          0.8,
}
DEFAULT_MERGE_DISTANCE = 0.8

# Plan 3d: detector emits both "tv" and "monitor" labels for the same
# physical screen. Canonicalize before clustering so they merge.
_CATEGORY_ALIASES: Dict[str, str] = {
    "tv": "monitor",
    "couch": "sofa",
    "potted_plant": "potted plant",
    "lamp_floor": "lamp",
    "lamp_desk": "lamp",
    "plant": "potted plant",
}


def _canonicalize(category: str) -> str:
    return _CATEGORY_ALIASES.get(category, category)

# Categories that the open-vocab detector visually confuses with each
# other on the same physical object. Keys are the family names; values
# are the member categories. When a new detection's category is in a
# family AND it's near (within the family's merge distance) an existing
# cluster of any category in the same family, they merge — and the
# canonical category becomes whichever of them has higher confidence.
_RESOLUTION_FAMILIES: Dict[str, List[str]] = {
    "storage": ["wardrobe", "dresser", "bookshelf", "nightstand"],
}

_CATEGORY_TO_FAMILY: Dict[str, str] = {
    cat: family
    for family, cats in _RESOLUTION_FAMILIES.items()
    for cat in cats
}


def _distance(a, b) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def _category_match(
    new_cat: str, cluster_cat: str,
) -> Optional[str]:
    """Return the family name if new_cat and cluster_cat match (same
    category OR same resolution family), else None."""
    if new_cat == cluster_cat:
        return new_cat  # not a family — exact category match
    family = _CATEGORY_TO_FAMILY.get(new_cat)
    if family and _CATEGORY_TO_FAMILY.get(cluster_cat) == family:
        return family
    return None


def cluster_keyframe_detections(
    detections: List[LiftedDetection],
    merge_distance_by_category: Dict[str, float] = None,
) -> List[Dict]:
    """NN-merge same-category (or same-family) detections within per-category
    3D distance.

    Returns a list of dicts in the on-device yolo_objects.json shape:
      {category, position [x,y,z], peak_confidence, view_count, best_frame_index}

    Detections with position_3d=None are dropped.
    """
    thresholds = merge_distance_by_category or BACKEND_TRACKER_DISTANCES

    valid = [d for d in detections if d.position_3d is not None]
    valid.sort(key=lambda d: d.confidence, reverse=True)

    clusters: List[Dict] = []
    for det in valid:
        cat = _canonicalize(det.category)  # Plan 3d: tv→monitor etc.
        threshold = thresholds.get(cat, DEFAULT_MERGE_DISTANCE)
        merged = False
        for c in clusters:
            if _category_match(cat, c["category"]) is None:
                continue
            if _distance(c["position"], det.position_3d) <= threshold:
                c["view_count"] += 1
                if det.confidence > c["peak_confidence"]:
                    c["peak_confidence"] = det.confidence
                    c["best_frame_index"] = det.frame_index
                    c["position"] = list(det.position_3d)
                    c["category"] = cat  # higher-confidence label wins
                merged = True
                break
        if not merged:
            clusters.append({
                "id": f"rgb_{uuid.uuid4().hex[:6]}",
                "category": cat,
                "position": list(det.position_3d),
                "peak_confidence": det.confidence,
                "view_count": 1,
                "best_frame_index": det.frame_index,
                "_source": "rgb",
            })

    logger.info(
        "keyframe_clustering: %d detections -> %d clusters",
        len(valid), len(clusters),
    )
    return clusters
