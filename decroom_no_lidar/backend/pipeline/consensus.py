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

# Map (input category) -> (kenney asset name without extension). Only
# categories that resolve to an actually-bundled USDZ make it through
# this map. Categories outside the keys get filtered out — including
# 'laptop' (no asset) and 'tv' falls back to the closest substitute,
# 'monitor'.
#
# 13 bundled Kenney models (per ls of decroom_no_lidar/*.usdz):
#   bed, bookshelf, chair, desk, dresser, lamp_desk, lamp_floor,
#   monitor, nightstand, plant, rug, sofa, table.
CATEGORY_TO_ASSET: Dict[str, str] = {
    # COCO-80 names + Decroom internal names mapped to Kenney files.
    "bed":              "bed",
    "couch":            "sofa",
    "sofa":             "sofa",
    "chair":            "chair",
    "stool":            "chair",
    "dining_table":     "table",
    "dining table":     "table",
    "coffee_table":     "table",
    "side_table":       "table",
    "desk":             "desk",
    "dresser":          "dresser",
    "wardrobe":         "dresser",
    "bookshelf":        "bookshelf",
    "nightstand":       "nightstand",
    # Electronics — use the monitor model as a substitute for tv/monitor.
    # Laptops are NOT mapped (no good asset; drop from rendering).
    "tv":               "monitor",
    "monitor":          "monitor",
    # Decor
    "lamp":             "lamp_floor",
    "lamp_floor":       "lamp_floor",
    "lamp_desk":        "lamp_desk",
    "potted plant":     "plant",
    "potted_plant":     "plant",
    "plant":            "plant",
    "rug":              "rug",
}

# Categories accepted into consensus = anything we have an asset for.
INDOOR_CATEGORIES_WHITELIST = set(CATEGORY_TO_ASSET.keys())


def asset_id_for(category: str) -> str:
    """Map a category name to the bundled Kenney asset id used by iOS."""
    base = CATEGORY_TO_ASSET.get(category)
    return f"kenney_{base}" if base else f"kenney_{category}"


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


def _room_xz_bounds(walls: List[Dict[str, Any]],
                       margin: float = 0.3) -> Tuple[float, float, float, float] | None:
    """Return (min_x, max_x, min_z, max_z) of the wall polygon plus a
    small margin (so objects right against a wall aren't filtered out).
    Returns None if walls are empty."""
    if not walls:
        return None
    xs: List[float] = []
    zs: List[float] = []
    for w in walls:
        for endpoint_key in ("start", "end"):
            ep = w.get(endpoint_key)
            if isinstance(ep, dict):
                xs.append(float(ep.get("x", 0)))
                zs.append(float(ep.get("z", 0)))
            elif isinstance(ep, (list, tuple)) and len(ep) >= 2:
                xs.append(float(ep[0]))
                zs.append(float(ep[1]))
    if not xs:
        return None
    return (min(xs) - margin, max(xs) + margin,
            min(zs) - margin, max(zs) + margin)


def _yolo_inside_room(yolo: Dict[str, Any],
                        bounds: Tuple[float, float, float, float] | None) -> bool:
    if bounds is None:
        return True
    x_min, x_max, z_min, z_max = bounds
    px, _, pz = yolo["position"]
    return x_min <= px <= x_max and z_min <= pz <= z_max


def _filter_yolo_input(
    yolo_objects_3d: List[Dict[str, Any]],
    walls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop YOLO objects with non-indoor categories or positions outside
    the room footprint. These produce noise in the consensus output."""
    bounds = _room_xz_bounds(walls)
    out: List[Dict[str, Any]] = []
    for y in yolo_objects_3d:
        cat = y.get("category", "")
        if cat not in INDOOR_CATEGORIES_WHITELIST:
            logger.info("YOLO filter: dropping non-indoor category %r", cat)
            continue
        if not _yolo_inside_room(y, bounds):
            px, _, pz = y["position"]
            logger.info(
                "YOLO filter: dropping %r at (%.2f, %.2f) outside room bounds %s",
                cat, px, pz, bounds,
            )
            continue
        out.append(y)
    return out


def consensus_merge(
    yolo_objects_3d: List[Dict[str, Any]],
    slm_objects: List[Dict[str, Any]],
    walls: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Match YOLO's deduped 3D-tracked objects against SpatialLM output.

    Returns (merged_objects, agreement_scores_by_id).
    The `walls` argument scopes the YOLO input filter (drops objects
    whose 3D position is outside the room polygon). Wall-snapping
    happens later in snap_to_walls.

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
    # Pre-filter YOLO: drop non-indoor categories and out-of-room positions.
    yolo_objects_3d = _filter_yolo_input(yolo_objects_3d, walls)

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


# ── Duplicate collapse ───────────────────────────────────────────────────────

# Per-category merge distances for duplicate collapse (meters). Tuned so a
# real room rarely has two of the same category within these distances.
# No hardcoded instance caps — rooms can legitimately have N beds (hostel),
# N chairs (kitchen), etc.
MERGE_DISTANCE_BY_CATEGORY: Dict[str, float] = {
    "bed": 1.5,
    "couch": 1.5,
    "sofa": 1.5,
    "chair": 0.65,
    "dining_table": 1.1,
    "desk": 1.1,
    "tv": 0.8,
    "laptop": 0.5,
    "lamp": 0.5,
    "potted plant": 0.6,
    "bookshelf": 1.2,
    "dresser": 1.2,
}
_DEFAULT_MERGE_DISTANCE = 0.6


def _merge_distance_for(category: str) -> float:
    return MERGE_DISTANCE_BY_CATEGORY.get(category, _DEFAULT_MERGE_DISTANCE)


def _pos_tuple(obj: Dict[str, Any]) -> Tuple[float, float, float]:
    p = obj.get("position", {})
    return (float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)))


def _confidence_of(obj: Dict[str, Any]) -> float:
    return float(obj.get("_confidence", obj.get("confidence", 0.5)))


def collapse_duplicates(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Position-based duplicate collapse within each category.

    Iterates: find any pair of same-category objects within the category's
    merge distance; drop the lower-confidence one. Continues until no
    such pair remains. No instance caps — all surviving objects are kept.
    """
    current = list(objects)
    while True:
        dropped: int | None = None
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                a, b = current[i], current[j]
                if a.get("type") != b.get("type"):
                    continue
                thresh = _merge_distance_for(a.get("type", ""))
                if _distance_xyz(_pos_tuple(a), _pos_tuple(b)) > thresh:
                    continue
                # Pair within tolerance — drop lower-confidence member.
                if _confidence_of(b) > _confidence_of(a):
                    dropped = i
                else:
                    dropped = j
                break
            if dropped is not None:
                break
        if dropped is None:
            break
        del current[dropped]
    return current


# ── Wall snapping ────────────────────────────────────────────────────────────

# Per-category snap-to-wall tolerances (meters). Categories not in either
# table get no snap.
SNAP_TO_WALL_CATEGORIES: Dict[str, float] = {
    "bed": 0.8,
    "dresser": 0.8,
    "wardrobe": 0.8,
    "sofa": 0.8,
    "couch": 0.8,
    "bookshelf": 0.8,
    "desk": 0.8,
    "nightstand": 0.5,
    "tv": 0.8,
}
FREESTANDING_CATEGORIES = {
    "chair", "stool", "dining_table", "coffee_table", "side_table",
    "lamp", "lamp_floor", "lamp_desk", "potted plant", "rug", "poster",
    "laptop",
}


def _project_point_onto_segment(
    px: float, pz: float,
    x0: float, z0: float,
    x1: float, z1: float,
) -> Tuple[float, float, float]:
    """Project (px, pz) onto segment (x0,z0)→(x1,z1). Returns
    (projected_x, projected_z, perpendicular_distance)."""
    dx, dz = x1 - x0, z1 - z0
    seg_len_sq = dx * dx + dz * dz
    if seg_len_sq <= 1e-8:
        return (x0, z0, math.sqrt((px - x0) ** 2 + (pz - z0) ** 2))
    t = ((px - x0) * dx + (pz - z0) * dz) / seg_len_sq
    t = max(0.0, min(1.0, t))
    qx, qz = x0 + t * dx, z0 + t * dz
    d = math.sqrt((px - qx) ** 2 + (pz - qz) ** 2)
    return (qx, qz, d)


def _wall_yaw(wall: Dict[str, Any]) -> float:
    s, e = wall["start"], wall["end"]
    return math.atan2(e["x"] - s["x"], e["z"] - s["z"])


def snap_to_walls(
    objects: List[Dict[str, Any]],
    walls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Snap each applicable object to the nearest wall within its category
    tolerance. Freestanding categories pass through unchanged. Objects
    beyond tolerance stay at their input position."""
    if not walls:
        return list(objects)

    out: List[Dict[str, Any]] = []
    for obj in objects:
        cat = obj.get("type", "")
        if cat in FREESTANDING_CATEGORIES:
            out.append(obj)
            continue
        tol = SNAP_TO_WALL_CATEGORIES.get(cat)
        if tol is None:
            out.append(obj)
            continue

        ox = obj["position"]["x"]
        oz = obj["position"]["z"]

        best_qx = best_qz = 0.0
        best_d = float("inf")
        best_wall: Dict[str, Any] | None = None
        for w in walls:
            qx, qz, d = _project_point_onto_segment(
                ox, oz,
                w["start"]["x"], w["start"]["z"],
                w["end"]["x"], w["end"]["z"],
            )
            if d < best_d:
                best_d = d
                best_qx, best_qz = qx, qz
                best_wall = w

        if best_wall is None or best_d > tol:
            out.append(obj)
            continue

        snapped = dict(obj)
        snapped["position"] = {"x": best_qx, "y": obj["position"]["y"],
                                 "z": best_qz}
        # Face into the room: wall_yaw + pi.
        snapped["rotation_y"] = _wall_yaw(best_wall) + math.pi
        out.append(snapped)
    return out


# ── Per-object confidence ────────────────────────────────────────────────────

# Plausible size ranges per category (width_min, width_max,
# length_min, length_max, height_min, height_max) in meters.
_SIZE_PLAUSIBILITY: Dict[str, Tuple[float, float, float, float, float, float]] = {
    "bed":          (1.2, 2.2, 1.9, 2.2, 0.3, 1.2),
    "couch":        (1.5, 2.8, 0.8, 1.1, 0.7, 1.1),
    "sofa":         (1.5, 2.8, 0.8, 1.1, 0.7, 1.1),
    "chair":        (0.4, 0.7, 0.4, 0.7, 0.7, 1.2),
    "dining_table": (0.9, 2.2, 0.7, 1.4, 0.65, 0.85),
    "desk":         (0.9, 1.8, 0.5, 0.9, 0.65, 0.85),
    "tv":           (0.8, 1.8, 0.05, 0.2, 0.45, 1.0),
    "laptop":       (0.25, 0.45, 0.20, 0.35, 0.01, 0.05),
    "lamp":         (0.15, 0.6, 0.15, 0.6, 0.3, 1.8),
    "potted plant": (0.1, 0.7, 0.1, 0.7, 0.1, 1.5),
    "dresser":      (0.6, 1.4, 0.35, 0.6, 0.7, 1.0),
    "bookshelf":    (0.5, 1.5, 0.2, 0.5, 0.8, 2.2),
}


def _size_plausibility(obj: Dict[str, Any]) -> float:
    m = obj.get("metadata", {})
    w = float(m.get("width", 1.0))
    l = float(m.get("length", 1.0))
    h = float(m.get("height", 1.0))
    rng = _SIZE_PLAUSIBILITY.get(obj.get("type", ""))
    if rng is None:
        # Unknown category — neutral-ish score (not full credit, not penalty).
        return 0.7
    w_min, w_max, l_min, l_max, h_min, h_max = rng
    in_range = (w_min <= w <= w_max and l_min <= l <= l_max
                and h_min <= h <= h_max)
    return 1.0 if in_range else 0.3


def _multi_view_score(yolo_matched: bool, yolo_view_count: int) -> float:
    if not yolo_matched:
        # No YOLO match → no multi-view evidence; use neutral default.
        return 0.5
    if yolo_view_count >= 3:
        return 1.0
    if yolo_view_count == 2:
        return 0.7
    if yolo_view_count == 1:
        return 0.4
    return 0.2


def compute_confidence(
    obj: Dict[str, Any],
    agreement_score: float,
    yolo_view_count: int,
    yolo_matched: bool,
) -> float:
    """
    confidence = 0.5 * agreement_score
               + 0.3 * multi_view_score
               + 0.2 * size_plausibility

    Result clamped to [0, 1]. agreement_score clamped to [0, 1] before use.
    """
    a = max(0.0, min(1.0, agreement_score))
    m = _multi_view_score(yolo_matched, yolo_view_count)
    s = _size_plausibility(obj)
    c = 0.5 * a + 0.3 * m + 0.2 * s
    return max(0.0, min(1.0, c))
