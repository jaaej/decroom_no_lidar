"""
Plan 3b semantic polish pipeline.

Pure functions for: YOLO/SpatialLM consensus merge, position-based
duplicate collapse, wall-snapping, per-object confidence, door/window
extraction.

Operates on dict-based objects (not Pydantic models) to stay decoupled
from schema.py changes.

Common merged-object shape (input + output):
  {
    "id": str,
    "type": str,           # Decroom category ('bed', 'couch', ...)
    "position": {"x": float, "y": float, "z": float},  # meters, ARKit
    "rotation_y": float,
    "metadata": {"width": float, "length": float, "height": float},
    "_source": "slm" | "yolo" | "consensus",   # provenance
  }
"""
from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("decroom-backend.consensus")

# Position-match distance when comparing YOLO 3D objects to SpatialLM
# objects. Large items (bed, sofa, dining table) are more diffuse — allow
# looser match. Small items (chair, lamp) need tighter match to avoid
# false consensus on neighbouring objects.
POSITION_MATCH_DISTANCE_LARGE = 0.8
POSITION_MATCH_DISTANCE_SMALL = 0.4

_LARGE_CATEGORIES = {"bed", "couch", "sofa", "dining_table", "dresser",
                      "wardrobe", "bookshelf"}


def _match_distance(category: str) -> float:
    return POSITION_MATCH_DISTANCE_LARGE if category in _LARGE_CATEGORIES \
        else POSITION_MATCH_DISTANCE_SMALL


def _distance_xyz(a: Tuple[float, float, float],
                   b: Tuple[float, float, float]) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


# Category priors (width, length, height) for YOLO-only detections that
# need a default oriented bbox when SLM didn't see the object.
_DEFAULT_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    "bed":          (1.6, 2.0, 0.6),
    "couch":        (2.0, 0.9, 0.9),
    "sofa":         (2.0, 0.9, 0.9),
    "chair":        (0.55, 0.55, 0.85),
    "dining_table": (1.4, 0.9, 0.75),
    "desk":         (1.3, 0.7, 0.75),
    "tv":           (1.2, 0.1, 0.7),
    "laptop":       (0.35, 0.25, 0.02),
    "lamp":         (0.3, 0.3, 1.4),
    "potted plant": (0.3, 0.3, 0.4),
    "dresser":      (1.0, 0.5, 0.85),
    "bookshelf":    (0.8, 0.3, 1.8),
}


def _default_dimensions_for(category: str) -> Tuple[float, float, float]:
    return _DEFAULT_DIMENSIONS.get(category, (1.0, 1.0, 1.0))


def _slm_position_tuple(slm: Dict[str, Any]) -> Tuple[float, float, float]:
    p = slm.get("position", {})
    return (float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)))


def _slm_to_merged(slm: Dict[str, Any]) -> Dict[str, Any]:
    p = slm.get("position", {})
    return {
        "id": slm.get("id") or f"slm_{uuid.uuid4().hex[:6]}",
        "type": slm.get("type", "unknown"),
        "position": {"x": float(p.get("x", 0)), "y": float(p.get("y", 0)),
                      "z": float(p.get("z", 0))},
        "rotation_y": float(slm.get("rotation_y", 0.0)),
        "metadata": dict(slm.get("metadata", {})),
        "_source": "slm",
    }


def _yolo_to_merged(yolo: Dict[str, Any]) -> Dict[str, Any]:
    px, py, pz = yolo["position"]
    cat = yolo["category"]
    dims = _default_dimensions_for(cat)
    return {
        "id": f"yolo_{uuid.uuid4().hex[:6]}",
        "type": cat,
        "position": {"x": float(px), "y": float(py), "z": float(pz)},
        "rotation_y": 0.0,
        "metadata": {"width": dims[0], "length": dims[1], "height": dims[2]},
        "_source": "yolo",
    }


def consensus_merge(
    yolo_objects_3d: List[Dict[str, Any]],
    slm_objects: List[Dict[str, Any]],
    walls: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Match YOLO's deduped 3D-tracked objects against SpatialLM output.

    Returns (merged_objects, agreement_scores_by_id).
    The `walls` argument is accepted for future use; consensus_merge
    itself doesn't snap to walls (that's snap_to_walls).

    Outcomes (one row per scenario from the design spec):
      - YOLO + SLM same category, within match distance →
        keep SLM (oriented bbox), agreement = 1.0
      - YOLO sees object SLM missed →
        keep at YOLO position with default dims, agreement = 0.5
      - SLM sees object YOLO missed →
        keep SLM as-is, agreement = 0.5
      - YOLO + SLM both have N objects of same category at distinct
        positions → keep all N pairs at agreement = 1.0
      - One YOLO bed matches multiple SLM beds (SLM duplicate at same
        position) → collapse to closest SLM, agreement = 0.3
      - YOLO + SLM disagree on category at the same position →
        keep SLM (oriented bbox), agreement = 0.3, log diagnostic
    """
    merged: List[Dict[str, Any]] = []
    agreement: Dict[str, float] = {}
    slm_used = [False] * len(slm_objects)
    slm_positions = [_slm_position_tuple(s) for s in slm_objects]

    for yolo in yolo_objects_3d:
        yx, yy, yz = yolo["position"]
        cat = yolo["category"]
        thresh = _match_distance(cat)

        same_cat: List[int] = []
        diff_cat: List[int] = []
        for i, slm in enumerate(slm_objects):
            if slm_used[i]:
                continue
            d = _distance_xyz((yx, yy, yz), slm_positions[i])
            if d > thresh:
                continue
            if slm.get("type") == cat:
                same_cat.append(i)
            else:
                diff_cat.append(i)

        if len(same_cat) == 1:
            i = same_cat[0]
            slm_used[i] = True
            obj = _slm_to_merged(slm_objects[i])
            obj["_source"] = "consensus"
            merged.append(obj)
            agreement[obj["id"]] = 1.0
        elif len(same_cat) >= 2:
            # SLM duplicates → collapse to closest, low agreement.
            for i in same_cat:
                slm_used[i] = True
            i0 = min(same_cat,
                     key=lambda i: _distance_xyz((yx, yy, yz),
                                                   slm_positions[i]))
            obj = _slm_to_merged(slm_objects[i0])
            obj["_source"] = "consensus"
            merged.append(obj)
            agreement[obj["id"]] = 0.3
        elif len(diff_cat) >= 1:
            # Category conflict at same position — keep SLM, low agreement.
            i = diff_cat[0]
            slm_used[i] = True
            obj = _slm_to_merged(slm_objects[i])
            obj["_source"] = "consensus"
            merged.append(obj)
            agreement[obj["id"]] = 0.3
            logger.info(
                "consensus_category_conflict: yolo=%s slm=%s at (%.2f,%.2f,%.2f)",
                cat, slm_objects[i].get("type"), yx, yy, yz,
            )
        else:
            # YOLO-only — lift with default dims, moderate agreement.
            obj = _yolo_to_merged(yolo)
            merged.append(obj)
            agreement[obj["id"]] = 0.5

    # SLM objects with no matching YOLO — keep with moderate agreement.
    for i, slm in enumerate(slm_objects):
        if slm_used[i]:
            continue
        obj = _slm_to_merged(slm)
        merged.append(obj)
        agreement[obj["id"]] = 0.5

    return merged, agreement
