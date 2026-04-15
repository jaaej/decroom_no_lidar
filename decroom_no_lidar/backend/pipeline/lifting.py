"""
3D object lifting module for the Decroom pipeline.
Converts 2D detections + depth maps into 3D positioned objects.
Handles: position estimation, orientation estimation, dimension estimation,
wall snapping, clustering, and final RoomObject assembly.
"""
from __future__ import annotations

import logging
from math import atan2, cos, pi, sin, sqrt
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .config import PipelineConfig
from .detection import Detection
from .geometry import associate_fp_to_detection, robust_lift_detection
from .metadata import CameraIntrinsics, ShotMetadata
from .tracking import Track

logger = logging.getLogger("decroom-backend.lifting")

# Categories to skip size similarity check during clustering (Session 6, 8)
# Depth noise makes dimensions too unreliable — skip for ALL categories
# since guided capture (wide viewpoint changes) amplifies depth error significantly
SKIP_SIZE_CHECK = {"bed", "sofa", "dresser", "bookshelf", "monitor", "desk",
                   "table", "chair", "nightstand", "lamp_floor", "lamp_desk",
                   "plant", "poster", "rug"}

# Default physical dimensions (meters) for each furniture type
DEFAULT_DIMS = {
    "bed": (2.0, 1.5, 0.6),       # length, width, height
    "sofa": (2.0, 0.9, 0.85),
    "desk": (1.2, 0.6, 0.75),
    "chair": (0.5, 0.5, 0.9),
    "table": (1.0, 1.0, 0.75),
    "dresser": (1.0, 0.5, 0.9),
    "nightstand": (0.5, 0.4, 0.55),
    "lamp_floor": (0.3, 0.3, 1.5),
    "lamp_desk": (0.25, 0.25, 0.45),
    "plant": (0.4, 0.4, 0.8),
    "bookshelf": (0.8, 0.35, 1.8),
    "monitor": (0.6, 0.2, 0.4),
    "rug": (2.0, 1.5, 0.02),
    "poster": (0.6, 0.02, 0.8),
}

# ── Wall proximity threshold ────────────────────────────────────────────────
# Unified threshold for BOTH wall snapping and wall-based orientation logic.
# Tightened to reduce over-snapping and wrong wall associations in cluttered rooms.
WALL_PROXIMITY_THRESHOLD = 0.85  # meters

# Categories where the object's front face points away from the nearest wall
WALL_FACING_TYPES = {"dresser", "bookshelf", "poster", "monitor", "nightstand",
                     "bed", "sofa", "desk", "table"}

# Rotationally symmetric or flat — no meaningful facing direction
SYMMETRIC_TYPES = {"lamp_floor", "lamp_desk", "plant", "rug"}

# Chair context: max distance to associated desk/table (tightened).
CHAIR_DESK_ASSOC_DIST = 1.2  # meters
# Minimum confidence for desk/table to be considered for chair association
CHAIR_DESK_MIN_CONF = 0.3

# Minimum number of frame detections to finalize an object (Session 10, Fix D)
# Single-detection objects are likely false positives (depth noise, transient
# misclassifications). Requiring ≥2 stabilizes counts across scans.
MIN_DETECTIONS_TO_FINALIZE = 2
SINGLE_VIEW_KEEP_MIN_SCORE_BY_CAT = {
    "monitor": 0.78,
    "chair": 0.78,
}
SINGLE_VIEW_KEEP_MIN_FP_BY_CAT = {
    "monitor": 14,
    "chair": 12,
}
SINGLE_VIEW_KEEP_MIN_FP_RATIO = 0.12

# Voxel size for temporal deduplication and density normalization
FP_VOXEL_DEDUP = 0.02  # 2cm — temporal weight counting
FP_VOXEL_DENSITY = 0.05  # 5cm — density normalization for centroid

# Only snap to room axes when layout axis confidence is strong enough.
AXIS_SNAP_MIN_CONFIDENCE = 0.62
AXIS_SNAP_RELAXED_MIN_CONF_BY_CAT = {
    "bed": 0.58,
    "monitor": 0.55,
    "desk": 0.55,
    "chair": 0.58,
}

# Conservative wall snapping controls for metric-depth runs.
WALL_SNAP_DIST_BY_CAT = {
    "monitor": 0.35,
    "poster": 0.30,
    "nightstand": 0.40,
    "dresser": 0.45,
    "bookshelf": 0.45,
    "desk": 0.45,
    "bed": 0.50,
    "sofa": 0.50,
}
WALL_SNAP_MAX_SHIFT_M = 0.30
WALL_SNAP_BLEND_ALPHA = 0.60
WALL_SNAP_OFFSET_CAP_M = 0.40

# Context rescue (for sparse-but-valid detections)
CONTEXT_RESCUE_CHAIR_MIN_SCORE = 0.72
CONTEXT_RESCUE_MONITOR_ANCHOR_MIN_SCORE = 0.60
CONTEXT_RESCUE_CHAIR_MAX_ANCHOR_DIST_M = 1.8
CONTEXT_RESCUE_DESK_PROXY_MIN_SCORE = 0.28
CONTEXT_RESCUE_DESK_MIN_MONITORS = 2
CONTEXT_RESCUE_DESK_MONITOR_MIN_FP = 8
CONTEXT_RESCUE_DESK_MAX_RADIUS_M = 1.15
CONTEXT_RESCUE_DESK_MAX_PAIRWISE_P75_M = 1.55
CONTEXT_RESCUE_DESK_MONITOR_CHAIR_MIN_DIST_M = 0.35
CONTEXT_RESCUE_DESK_MONITOR_CHAIR_MAX_DIST_M = 1.9
CONTEXT_RESCUE_DESK_MONITOR_CHAIR_BIAS_TO_MONITOR = 0.7
CONTEXT_RESCUE_CHAIR_ANCHOR_MIN_SCORE = 0.72
CONTEXT_RESCUE_CHAIR_ANCHOR_MIN_FP = 10
FRAGMENTATION_RESCUE_MAX_OBJECTS = 3
FRAGMENTATION_RESCUE_MIN_SCORE_BY_CAT = {
    "monitor": 0.74,
    "chair": 0.60,
    "bookshelf": 0.60,
    "dresser": 0.80,
}
FRAGMENTATION_RESCUE_MIN_FP_BY_CAT = {
    "monitor": 12,
    "chair": 6,
    "bookshelf": 6,
    "dresser": 16,
}
FRAGMENTATION_RESCUE_MIN_FP_RATIO_BY_CAT = {
    "monitor": 0.12,
    "chair": 0.08,
    "bookshelf": 0.08,
    "dresser": 0.14,
}
SMALL_OBJECT_CATS = {"monitor", "chair", "poster"}

# Lift-step diagnostics for downstream quality gates.
_lift_diagnostics: Dict = {}
_cluster_same_frame_merge_events: List[Dict] = []


def get_lift_diagnostics() -> Dict:
    """Return diagnostics collected during the last lift_objects() run."""
    return dict(_lift_diagnostics)


def _dist_xz(a: Dict, b: Dict) -> float:
    dx = float(a.get("x", 0.0)) - float(b.get("x", 0.0))
    dz = float(a.get("z", 0.0)) - float(b.get("z", 0.0))
    return sqrt(dx * dx + dz * dz)


def _monitor_anchor_quality_ok(o: Dict) -> bool:
    """Check whether a monitor candidate is strong enough to anchor desk rescue."""
    score = float(o.get("score", 0.0))
    n_det = int(o.get("n_detections", 1))
    fp_count = int(o.get("fp_count", 0))
    support_views = int(o.get("support_views", 0))
    return (
        score >= max(CONTEXT_RESCUE_MONITOR_ANCHOR_MIN_SCORE, 0.65)
        and n_det >= 2
        and fp_count >= CONTEXT_RESCUE_DESK_MONITOR_MIN_FP
        and support_views >= 2
    )


def _pairwise_p75(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    dists: List[float] = []
    for i in range(len(points)):
        x0, z0 = points[i]
        for j in range(i + 1, len(points)):
            x1, z1 = points[j]
            dx = x1 - x0
            dz = z1 - z0
            dists.append(sqrt(dx * dx + dz * dz))
    if not dists:
        return 0.0
    dists.sort()
    idx = int(0.75 * (len(dists) - 1))
    return float(dists[idx])


def _rescue_context_objects(clustered: List[Dict], dropped: List[Dict]) -> List[Dict]:
    """
    Recover key functional objects lost by the min-observation filter.

    Targets:
    - `desk`: inferred from >=2 strong monitor anchors when no desk/table survives.
    - `chair`: restore one high-confidence chair singleton near desk/monitor context.
    """
    rescued: List[Dict] = []
    present = {str(o.get("category", "")) for o in clustered}
    pool = list(clustered) + list(dropped)

    # ── Desk inference from monitor cluster ────────────────────────────────
    has_desk = ("desk" in present) or ("table" in present)
    monitor_anchors = [
        o for o in pool
        if str(o.get("category")) == "monitor"
        and _monitor_anchor_quality_ok(o)
    ]
    if not has_desk and len(monitor_anchors) >= CONTEXT_RESCUE_DESK_MIN_MONITORS:
        top = sorted(monitor_anchors, key=lambda o: -float(o.get("score", 0.0)))[:4]
        points = [(float(o.get("x", 0.0)), float(o.get("z", 0.0))) for o in top]
        mx = float(sum(p[0] for p in points) / max(1, len(points)))
        mz = float(sum(p[1] for p in points) / max(1, len(points)))
        max_radius = max(sqrt((x - mx) ** 2 + (z - mz) ** 2) for x, z in points) if points else 0.0
        pairwise_p75 = _pairwise_p75(points)
        if (
            max_radius <= CONTEXT_RESCUE_DESK_MAX_RADIUS_M
            and pairwise_p75 <= CONTEXT_RESCUE_DESK_MAX_PAIRWISE_P75_M
        ):
            mscore = float(sum(float(o.get("score", 0.0)) for o in top) / max(1, len(top)))
            proxy_score = max(CONTEXT_RESCUE_DESK_PROXY_MIN_SCORE, min(0.68, mscore * 0.55))
            desk = {
                "category": "desk",
                "score": round(proxy_score, 3),
                "x": mx,
                "y": 0.0,
                "z": mz,
                "frame_idx": int(top[0].get("frame_idx", 0)),
                "n_detections": 1,
                "detections": [],
                "width_2d": 0.35,
                "height_2d": 0.20,
                "depth_median": 1.2,
                "depth_mad": 0.0,
                "depth_sample_count": 0,
                "position_source": "context_inference",
                "fp_count": 0,
                "fp_variance": 0.0,
                "frames_with_fp_hit": 0,
                "frames_total": 0,
                "support_views": 1,
                "position_uncertainty_m": 0.65,
                "_high_uncertainty": True,
                "_pca_world_points": [],
                "_pca_cam_positions": [],
                "_rescue_context": True,
                "_rescue_reason": "desk_inferred_from_monitor_cluster",
            }
            rescued.append(desk)
            logger.info(
                "  RESCUED (context): desk inferred from %d monitors at (%.3f, %.3f), "
                "score=%.2f radius=%.2f p75=%.2f",
                len(top), mx, mz, proxy_score, max_radius, pairwise_p75
            )
            present.add("desk")
        else:
            logger.info(
                "  SKIP context desk rescue: monitor anchors too dispersed "
                "(radius=%.2f, pairwise_p75=%.2f)",
                max_radius,
                pairwise_p75,
            )

    # ── Desk inference from monitor+chair tabletop geometry ────────────────
    if not has_desk and "desk" not in present:
        chairs_for_desk = [
            o for o in pool
            if str(o.get("category")) == "chair"
            and (
                float(o.get("score", 0.0)) >= CONTEXT_RESCUE_CHAIR_ANCHOR_MIN_SCORE
                or int(o.get("fp_count", 0)) >= CONTEXT_RESCUE_CHAIR_ANCHOR_MIN_FP
            )
        ]
        top_monitors = sorted(monitor_anchors, key=lambda o: -float(o.get("score", 0.0)))[:4]

        best_pair: Optional[Tuple[Dict, Dict, float]] = None
        best_key = (float("inf"), float("inf"), float("inf"))
        for m in top_monitors:
            for c in chairs_for_desk:
                dist = _dist_xz(m, c)
                if (
                    dist < CONTEXT_RESCUE_DESK_MONITOR_CHAIR_MIN_DIST_M
                    or dist > CONTEXT_RESCUE_DESK_MONITOR_CHAIR_MAX_DIST_M
                ):
                    continue
                key = (
                    abs(dist - 0.95),  # prefer typical chair-to-monitor spacing
                    -float(m.get("score", 0.0)),
                    -float(c.get("score", 0.0)),
                )
                if key < best_key:
                    best_key = key
                    best_pair = (m, c, dist)

        if best_pair is not None:
            m, c, pair_dist = best_pair
            mx, mz = float(m.get("x", 0.0)), float(m.get("z", 0.0))
            cx, cz = float(c.get("x", 0.0)), float(c.get("z", 0.0))
            ux = mx - cx
            uz = mz - cz
            norm = sqrt(ux * ux + uz * uz)
            if norm > 1e-5:
                ux /= norm
                uz /= norm
                # Desk center is usually slightly behind monitor toward wall side,
                # while still between monitor and chair.
                bias = CONTEXT_RESCUE_DESK_MONITOR_CHAIR_BIAS_TO_MONITOR
                desk_x = bias * mx + (1.0 - bias) * cx - 0.18 * ux
                desk_z = bias * mz + (1.0 - bias) * cz - 0.18 * uz

                mscore = float(m.get("score", 0.0))
                cscore = float(c.get("score", 0.0))
                proxy_score = max(
                    CONTEXT_RESCUE_DESK_PROXY_MIN_SCORE,
                    min(0.66, 0.45 + 0.18 * mscore + 0.08 * cscore),
                )

                desk = {
                    "category": "desk",
                    "score": round(proxy_score, 3),
                    "x": desk_x,
                    "y": 0.0,
                    "z": desk_z,
                    "frame_idx": int(m.get("frame_idx", 0)),
                    "n_detections": 1,
                    "detections": [],
                    "width_2d": 0.42,
                    "height_2d": 0.24,
                    "depth_median": 1.2,
                    "depth_mad": 0.0,
                    "depth_sample_count": 0,
                    "position_source": "context_inference",
                    "fp_count": 0,
                    "fp_variance": 0.0,
                    "frames_with_fp_hit": 0,
                    "frames_total": 0,
                    "support_views": 1,
                    "position_uncertainty_m": 0.65,
                    "_high_uncertainty": True,
                    "_pca_world_points": [],
                    "_pca_cam_positions": [],
                    "_rescue_context": True,
                    "_rescue_reason": "desk_inferred_from_monitor_chair_geometry",
                    "_rescue_monitor_track": m.get("track_id"),
                    "_rescue_chair_track": c.get("track_id"),
                }
                rescued.append(desk)
                present.add("desk")
                logger.info(
                    "  RESCUED (context): desk inferred from monitor+chair geometry "
                    "(dist=%.2f, monitor=%.2f, chair=%.2f) at (%.3f, %.3f), score=%.2f",
                    pair_dist, mscore, cscore, desk_x, desk_z, proxy_score,
                )

    # ── Chair rescue near desk/monitor context ─────────────────────────────
    has_chair = "chair" in present
    if not has_chair:
        anchors = [
            o for o in (clustered + rescued + dropped)
            if str(o.get("category")) in {"desk", "table", "monitor"}
            and float(o.get("score", 0.0)) >= CONTEXT_RESCUE_MONITOR_ANCHOR_MIN_SCORE
        ]
        chair_candidates = [
            o for o in dropped
            if str(o.get("category")) == "chair"
            and float(o.get("score", 0.0)) >= CONTEXT_RESCUE_CHAIR_MIN_SCORE
        ]

        best = None
        best_key = (float("inf"), float("inf"), float("inf"))
        for c in chair_candidates:
            if anchors:
                dist = min(_dist_xz(c, a) for a in anchors)
            else:
                dist = float("inf")
            if dist > CONTEXT_RESCUE_CHAIR_MAX_ANCHOR_DIST_M:
                continue
            # Prefer closer to desk/monitor context, then higher score/fp support.
            key = (
                dist,
                -float(c.get("score", 0.0)),
                -float(c.get("fp_count", 0.0)),
            )
            if key < best_key:
                best_key = key
                best = c

        if best is not None:
            rescued_chair = dict(best)
            rescued_chair["_rescue_context"] = True
            rescued_chair["_rescue_reason"] = "chair_singleton_near_context_anchor"
            rescued.append(rescued_chair)
            logger.info(
                "  RESCUED (context): chair score=%.2f at (%.3f, %.3f), anchor_dist=%.2f",
                float(rescued_chair.get("score", 0.0)),
                float(rescued_chair.get("x", 0.0)),
                float(rescued_chair.get("z", 0.0)),
                best_key[0],
            )

    return rescued


def _required_observation_count(obj: Dict) -> Tuple[int, str]:
    """
    Category-aware observation threshold.

    Monitors/chairs can survive with a single frame if that frame is strongly
    feature-point-supported. This reduces false deletion when tracking misses
    one link in sparse scans.
    """
    cat = str(obj.get("category", ""))
    if cat not in SINGLE_VIEW_KEEP_MIN_SCORE_BY_CAT:
        return MIN_DETECTIONS_TO_FINALIZE, "default"

    src = str(obj.get("position_source", ""))
    score = float(obj.get("score", 0.0))
    fp_count = int(obj.get("fp_count", 0) or 0)
    fp_ratio = float(obj.get("fp_in_bbox_ratio_max", 0.0) or 0.0)
    if (
        src == "feature_points"
        and score >= float(SINGLE_VIEW_KEEP_MIN_SCORE_BY_CAT.get(cat, 1.0))
        and fp_count >= int(SINGLE_VIEW_KEEP_MIN_FP_BY_CAT.get(cat, 999))
        and fp_ratio >= SINGLE_VIEW_KEEP_MIN_FP_RATIO
    ):
        return 1, "single_view_fp_strong"

    return MIN_DETECTIONS_TO_FINALIZE, "default"


def _suppress_weak_duplicate_beds(clustered: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Suppress weak duplicate bed candidates when a strong bed track exists.

    This targets false second-bed artifacts caused by fragmented/misaligned
    trajectories while preserving genuinely strong multiple-bed evidence.
    """
    beds = [o for o in clustered if str(o.get("category")) == "bed"]
    if len(beds) <= 1:
        return clustered, []

    def _is_strong_bed(o: Dict) -> bool:
        return (
            int(o.get("n_detections", 1)) >= 5
            and int(o.get("fp_count", 0)) >= 25
            and float(o.get("fp_in_bbox_ratio_max", 0.0)) >= 0.35
        )

    strong_exists = any(_is_strong_bed(b) for b in beds)
    if not strong_exists:
        return clustered, []

    kept: List[Dict] = []
    dropped: List[Dict] = []
    for o in clustered:
        if str(o.get("category")) != "bed":
            kept.append(o)
            continue

        n_det = int(o.get("n_detections", 1))
        fp_count = int(o.get("fp_count", 0))
        fp_ratio = float(o.get("fp_in_bbox_ratio_max", 0.0))
        src = str(o.get("position_source", ""))
        weak_dup = (
            src == "feature_points"
            and n_det <= 3
            and fp_count < 10
            and fp_ratio < 0.55
        )
        if weak_dup:
            o["_dropped_reason"] = "weak_duplicate_bed"
            dropped.append(o)
        else:
            kept.append(o)

    # Never drop all beds.
    if dropped and not any(str(o.get("category")) == "bed" for o in kept):
        return clustered, []
    return kept, dropped


def _fragmentation_rescue_objects(clustered: List[Dict], dropped: List[Dict]) -> List[Dict]:
    """
    Recover a small number of likely true positives when tracking fragmented
    detections into many singletons.
    """
    if not dropped:
        return []
    n_before = len(clustered) + len(dropped)
    dropped_singletons = [o for o in dropped if int(o.get("n_detections", 1)) <= 1]
    if n_before < 10 or len(dropped_singletons) < 6:
        return []
    if len(clustered) > max(2, int(0.4 * n_before)):
        return []

    present = {str(o.get("category", "")) for o in clustered}
    cat_priority = {"monitor": 0, "chair": 1, "bookshelf": 2, "dresser": 3}

    candidates = []
    for o in dropped_singletons:
        cat = str(o.get("category", ""))
        if cat in present:
            continue
        if cat not in cat_priority:
            continue
        if str(o.get("position_source", "")) != "feature_points":
            continue
        score = float(o.get("score", 0.0))
        fp_count = int(o.get("fp_count", 0))
        fp_ratio = float(o.get("fp_in_bbox_ratio_max", 0.0))
        min_score = float(FRAGMENTATION_RESCUE_MIN_SCORE_BY_CAT.get(cat, 0.60))
        min_fp = int(FRAGMENTATION_RESCUE_MIN_FP_BY_CAT.get(cat, 6))
        min_ratio = float(FRAGMENTATION_RESCUE_MIN_FP_RATIO_BY_CAT.get(cat, 0.08))
        if score < min_score or fp_count < min_fp or fp_ratio < min_ratio:
            continue
        candidates.append((cat_priority[cat], -score, -fp_count, o))

    candidates.sort()
    rescued: List[Dict] = []
    for _, _, _, c in candidates[:FRAGMENTATION_RESCUE_MAX_OBJECTS]:
        x = dict(c)
        x["_rescue_context"] = True
        x["_rescue_reason"] = "fragmentation_singleton_recovery"
        rescued.append(x)
        present.add(str(x.get("category", "")))

    if rescued:
        logger.info(
            "  RESCUED (fragmentation): restored %d singleton objects",
            len(rescued),
        )
    return rescued


def _weighted_median(values: "np.ndarray", weights: "np.ndarray") -> float:
    """Compute weighted median: value at cumulative weight = 50%."""
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_wts = weights[order]
    cumsum = np.cumsum(sorted_wts)
    cutoff = cumsum[-1] * 0.5
    idx = int(np.searchsorted(cumsum, cutoff))
    idx = min(idx, len(sorted_vals) - 1)
    return float(sorted_vals[idx])


# ---------------------------------------------------------------------------
# False-positive filters (reflection + wall-mounted misdetection)
# ---------------------------------------------------------------------------

_FLOOR_STANDING_CATEGORIES = {"bed", "sofa", "desk", "table", "dresser", "bookshelf", "chair"}


def _is_likely_reflection(
    obj: Dict,
    room_width: float,
    room_length: float,
    room_center: Optional[Tuple[float, float]],
    feature_cloud: Optional["np.ndarray"],
    cfg: "PipelineConfig",
) -> Optional[str]:
    """Check if object is likely a mirror reflection. Returns reason string or None."""
    if not getattr(cfg, "enable_reflection_filter", True):
        return None

    margin = getattr(cfg, "reflection_room_margin_m", 0.5)
    min_fp = getattr(cfg, "reflection_min_fp_density", 3)
    fp_radius = getattr(cfg, "reflection_fp_radius_m", 0.5)
    max_conf = getattr(cfg, "reflection_max_confidence", 0.6)

    score = float(obj.get("score", 1.0))
    if score >= max_conf:
        return None  # high-confidence objects are trusted

    x = float(obj.get("x", 0.0))
    z = float(obj.get("z", 0.0))

    # Check 1: Outside room bounding box
    if room_center is not None:
        cx, cz = room_center
        half_w = room_width / 2.0 + margin
        half_l = room_length / 2.0 + margin
        if abs(x - cx) > half_w or abs(z - cz) > half_l:
            return "outside_room_hull"

    # Check 2: Low feature-point density around object position
    if feature_cloud is not None and len(feature_cloud) > 0:
        # feature_cloud is Nx3 array of world-space points
        diffs = feature_cloud[:, [0, 2]] - np.array([[x, z]])
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        nearby_count = int(np.sum(dists <= fp_radius))
        if nearby_count < min_fp:
            return "low_fp_density"

    return None


def _is_wall_mounted_misdetection(
    obj: Dict,
    walls: List[Dict],
    floor_height: float,
    cfg: "PipelineConfig",
) -> bool:
    """Check if a floor-standing category is detected as wall-mounted."""
    if not getattr(cfg, "enable_wall_mounted_suppression", True):
        return False

    category = str(obj.get("category", ""))
    if category not in _FLOOR_STANDING_CATEGORIES:
        return False

    min_height = getattr(cfg, "wall_mounted_min_height_above_floor_m", 0.8)
    max_wall_dist = getattr(cfg, "wall_mounted_max_wall_dist_m", 0.3)

    y_pos = float(obj.get("y", 0.0))
    if y_pos - floor_height <= min_height:
        return False  # object is near floor, OK

    # Check distance to nearest wall
    x = float(obj.get("x", 0.0))
    z = float(obj.get("z", 0.0))

    for wall in walls:
        start = wall.get("start", [0, 0])
        end = wall.get("end", [0, 0])
        # Point-to-line-segment distance
        wx, wz = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
        px, pz = x - float(start[0]), z - float(start[1])
        seg_len_sq = wx * wx + wz * wz
        if seg_len_sq < 1e-9:
            continue
        t = max(0.0, min(1.0, (px * wx + pz * wz) / seg_len_sq))
        closest_x = float(start[0]) + t * wx
        closest_z = float(start[1]) + t * wz
        dist = ((x - closest_x)**2 + (z - closest_z)**2)**0.5
        if dist <= max_wall_dist:
            return True

    return False


def lift_objects(
    tracks: List[Track],
    depth_maps: List[Optional["np.ndarray"]],
    shots: List[ShotMetadata],
    room_width: float,
    room_length: float,
    room_height: float,
    walls: List[Dict],
    cfg: PipelineConfig,
    global_points: Optional["np.ndarray"] = None,
    feature_cloud: Optional["np.ndarray"] = None,
    floor_height: float = 0.0,
    room_center: Optional[Tuple[float, float]] = None,
    axis_confidence: float = 0.5,
    depth_reliability: float = 1.0,
    triangulation_priors: Optional[Dict[int, Dict]] = None,
) -> List[Dict]:
    """
    Lift tracked 2D detections into 3D positioned objects.

    Session 11: Feature-point-first positioning.
      Primary: ARKit feature-point centroid (metric)
      Fallback: Depth back-projection (legacy)

    Args:
        tracks: Tracked detections across frames
        depth_maps: Per-frame depth maps
        shots: Per-frame camera metadata
        room_width, room_length, room_height: Room dimensions
        walls: Wall definitions from layout
        cfg: Pipeline configuration
        global_points: Optional pre-built depth point cloud
        feature_cloud: Global ARKit feature point cloud (metric)
        floor_height: Y coordinate of the floor plane
        room_center: (x, z) center of the room

    Returns:
        List of RoomObject dicts ready for the payload
    """
    global _lift_diagnostics, _cluster_same_frame_merge_events
    _lift_diagnostics = {
        "dropped_depth_outliers_total": 0,
        "dropped_depth_outliers": [],
        "depth_outlier_filter_active": True,
        "rescued_context_total": 0,
        "rescued_context": [],
        "suppressed_inferred_objects_total": 0,
        "suppressed_inferred_objects": [],
    }
    _cluster_same_frame_merge_events = []

    if np is None:
        return []

    if not tracks:
        logger.info("No tracks to lift")
        return []

    logger.info("Lifting %d tracks into 3D (floor=%.3f)", len(tracks), floor_height)

    # Step 1: Estimate 3D position for each track
    raw_objects = []
    for track in tracks:
        obj = _lift_single_track(track, depth_maps, shots, cfg,
                                 floor_height=floor_height,
                                 triangulation_prior=(triangulation_priors or {}).get(track.track_id))
        if obj is not None:
            raw_objects.append(obj)

    logger.info("Lifted %d objects from %d tracks", len(raw_objects), len(tracks))

    # Log every raw object before clustering so we can see what's going in
    for ro in raw_objects:
        logger.debug("  PRE-CLUSTER: track=%d %s at (%.3f, %.3f, %.3f) score=%.2f n_det=%d",
                     ro.get("track_id", -1), ro["category"],
                     ro["x"], ro["y"], ro["z"],
                     ro["score"], ro.get("n_detections", 1))

    if not raw_objects:
        return []

    # Snapshot raw positions for diagnostic chain (pre-cluster)
    for ro in raw_objects:
        ro["_pos_chain"] = [
            {"stage": "lift", "x": round(ro["x"], 4), "z": round(ro["z"], 4)}
        ]

    # Step 2: Cluster duplicate detections of the same object
    clustered = _cluster_objects(raw_objects, cfg)
    logger.info("Clustered to %d objects", len(clustered))

    # Snapshot post-cluster positions
    for co in clustered:
        chain = co.get("_pos_chain", [])
        chain.append({"stage": "cluster", "x": round(co["x"], 4), "z": round(co["z"], 4)})
        co["_pos_chain"] = chain

    # Log every clustered object so we can see what survived
    for co in clustered:
        logger.debug("  POST-CLUSTER: %s at (%.3f, %.3f, %.3f) score=%.2f n_det=%d",
                     co["category"], co["x"], co["y"], co["z"],
                     co["score"], co.get("n_detections", 1))

    # Step 2a-2: Suppress weak duplicate beds when strong bed evidence exists.
    clustered, dropped_weak_beds = _suppress_weak_duplicate_beds(clustered)
    if dropped_weak_beds:
        for d in dropped_weak_beds:
            logger.info(
                "  DROPPED (weak_duplicate_bed): score=%.2f n_det=%d fp=%d fp_ratio=%.2f at (%.3f, %.3f, %.3f)",
                float(d.get("score", 0.0)),
                int(d.get("n_detections", 1)),
                int(d.get("fp_count", 0)),
                float(d.get("fp_in_bbox_ratio_max", 0.0)),
                float(d.get("x", 0.0)),
                float(d.get("y", 0.0)),
                float(d.get("z", 0.0)),
            )
        logger.info(
            "Weak-duplicate bed suppression: dropped %d candidates",
            len(dropped_weak_beds),
        )

    # Step 2b: Filter out low-observation objects (Session 10, Fix D)
    # Single-detection objects are likely false positives or transient
    # misclassifications. Require ≥ MIN_DETECTIONS_TO_FINALIZE frames.
    before_filter = len(clustered)
    dropped = []
    kept = []
    for o in clustered:
        required_obs, obs_rule = _required_observation_count(o)
        o["_obs_required"] = int(required_obs)
        o["_obs_rule"] = obs_rule
        if int(o.get("n_detections", 1)) >= required_obs:
            kept.append(o)
        else:
            dropped.append(o)
    clustered = kept
    if dropped:
        for d in dropped:
            logger.info(
                "  DROPPED (n_det<%d rule=%s): %s score=%.2f n_det=%d fp=%d src=%s at (%.3f, %.3f, %.3f)",
                int(d.get("_obs_required", MIN_DETECTIONS_TO_FINALIZE)),
                str(d.get("_obs_rule", "default")),
                d["category"],
                d["score"],
                d.get("n_detections", 1),
                int(d.get("fp_count", 0)),
                str(d.get("position_source", "?")),
                d["x"],
                d["y"],
                d["z"],
            )
        logger.info("Observation filter: %d → %d objects (%d dropped)",
                     before_filter, len(clustered), len(dropped))

    if not clustered:
        # Recovery path: if all objects were single-view tracks, keep a small set
        # of high-confidence candidates instead of returning an empty room.
        rescue = sorted(dropped, key=lambda o: -o.get("score", 0.0))
        rescue = [o for o in rescue if o.get("score", 0.0) >= 0.45][:12]
        if rescue:
            logger.warning(
                "Observation filter dropped all objects; restoring %d high-confidence single-view objects",
                len(rescue),
            )
            clustered = rescue
        else:
            logger.warning("All objects filtered out by min-observations rule")
            return []

    # Step 2b-2: Context-aware rescue for sparse desk/chair evidence.
    rescued_context = _rescue_context_objects(clustered, dropped)
    if rescued_context:
        clustered.extend(rescued_context)
        logger.info(
            "Context rescue: restored %d objects (post-observation filter total=%d)",
            len(rescued_context),
            len(clustered),
        )
    rescued_fragmentation = _fragmentation_rescue_objects(clustered, dropped)
    if rescued_fragmentation:
        clustered.extend(rescued_fragmentation)
        logger.info(
            "Fragmentation rescue: restored %d objects (post-observation filter total=%d)",
            len(rescued_fragmentation),
            len(clustered),
        )
    rescue_entries = []
    for r in (rescued_context + rescued_fragmentation):
        rescue_entries.append({
            "category": r.get("category", "unknown"),
            "reason": r.get("_rescue_reason"),
            "score": round(float(r.get("score", 0.0)), 3),
            "x": round(float(r.get("x", 0.0)), 3),
            "z": round(float(r.get("z", 0.0)), 3),
        })
    _lift_diagnostics["rescued_context_total"] = len(rescue_entries)
    _lift_diagnostics["rescued_context"] = rescue_entries

    # Step 2c: Placement stability guard (Session 11: FP-aware)
    # For FP-positioned objects: require fp_count >= 3 and fp_variance < 1.0
    # For depth-positioned objects: adapt thresholds by depth reliability.
    depth_reliability = max(0.0, min(1.0, float(depth_reliability)))
    depth_mad_thresh = 0.15 + (1.0 - depth_reliability) * 0.12  # 0.15 .. 0.27
    depth_min_samples = max(10, int(round(25 - (1.0 - depth_reliability) * 10)))  # 25 .. 15
    before_stab = len(clustered)
    unstable = []
    stable = []
    for o in clustered:
        if bool(o.get("_rescue_context", False)):
            stable.append(o)
            continue
        src = o.get("position_source", "depth_backproject")
        if src == "feature_points":
            # FP-positioned: trust if we have enough supporting points
            fpc = o.get("fp_count", 0)
            fpv = o.get("fp_variance", 999)
            if fpc < 3 or fpv > 1.0:
                unstable.append(o)
            else:
                stable.append(o)
        elif src == "triangulated_rays":
            # Triangulated priors come from multi-view ray intersections and should
            # not be judged by depth MAD/sample rules.
            tri_views = int(o.get("triangulation_views", 0))
            tri_err = float(o.get("triangulation_median_ray_error_m", 999.0))
            if tri_views < 2 or tri_err > 0.45:
                unstable.append(o)
            else:
                stable.append(o)
        else:
            # Depth-positioned: original guard
            mad = o.get("depth_mad", 0)
            samples = o.get("depth_sample_count", 9999)
            if mad > depth_mad_thresh or samples < depth_min_samples:
                unstable.append(o)
            else:
                stable.append(o)
    if unstable:
        for u in unstable:
            logger.info("  DROPPED (unstable): %s score=%.2f src=%s fp_count=%d fp_var=%.3f",
                         u["category"], u["score"], u.get("position_source", "?"),
                         u.get("fp_count", 0), u.get("fp_variance", 0))
        logger.info("Stability filter: %d → %d objects (%d unstable dropped)",
                     before_stab, len(stable), len(unstable))
    clustered = stable if stable else clustered  # Keep all if everything is unstable

    # Step 3: Estimate orientation for each object
    for obj in clustered:
        obj["rotation_y"] = _estimate_orientation(
            obj, walls, room_width, room_length, depth_maps, shots,
            all_objects=clustered, room_center=room_center,
            axis_confidence=axis_confidence,
        )

    # Step 4: Estimate dimensions
    for obj in clustered:
        dims = _estimate_dimensions(obj, depth_maps, shots, cfg)
        obj["width"], obj["length"], obj["height"] = dims

    # Step 5: Apply dimension sanity checks
    for obj in clustered:
        _sanity_check_dimensions(
            obj,
            room_width=room_width,
            room_length=room_length,
            room_height=room_height,
        )

    # Snapshot pre-snap positions
    for obj in clustered:
        chain = obj.get("_pos_chain", [])
        chain.append({"stage": "pre_snap", "x": round(obj["x"], 4), "z": round(obj["z"], 4)})
        obj["_pos_chain"] = chain

    # Step 6: Snap to walls where appropriate
    _snap_to_walls(clustered, walls, room_width, room_length,
                   room_center=room_center)

    # Snapshot post-snap positions
    for obj in clustered:
        chain = obj.get("_pos_chain", [])
        chain.append({"stage": "post_snap", "x": round(obj["x"], 4), "z": round(obj["z"], 4)})
        obj["_pos_chain"] = chain

    # Step 7: Drop extreme depth-only outliers before clamp
    clustered, dropped_outliers = _drop_depth_outliers_before_clamp(
        clustered,
        room_width,
        room_length,
        room_center=room_center,
        depth_reliability=depth_reliability,
    )
    depth_outlier_entries = []
    for d in dropped_outliers:
        if d.get("position_source") != "depth_backproject":
            continue
        depth_outlier_entries.append({
            "category": d.get("category", "unknown"),
            "score": round(float(d.get("score", 0.0)), 3),
            "outside_dist_m": float(d.get("_dropped_outside_dist", 0.0)),
            "outside_margin_m": float(d.get("_dropped_outside_margin", 0.0)),
        })
    _lift_diagnostics["dropped_depth_outliers_total"] = len(depth_outlier_entries)
    _lift_diagnostics["dropped_depth_outliers"] = depth_outlier_entries

    if dropped_outliers:
        logger.info(
            "Depth outlier filter: dropped %d objects before room clamp",
            len(dropped_outliers),
        )

    # Step 8: Clamp positions within room bounds
    _clamp_to_room(clustered, room_width, room_length, room_height,
                   room_center=room_center)

    # Snapshot post-clamp positions (final)
    for obj in clustered:
        chain = obj.get("_pos_chain", [])
        chain.append({"stage": "post_clamp", "x": round(obj["x"], 4), "z": round(obj["z"], 4)})
        obj["_pos_chain"] = chain

    # Step 9a: Suppress likely mirror reflections and wall-mounted misdetections.
    suppressed_reflection: List[Dict] = []
    suppressed_wallmount: List[Dict] = []
    pre_filter_clustered: List[Dict] = []
    for obj in clustered:
        reflection_reason = _is_likely_reflection(
            obj, room_width, room_length, room_center, feature_cloud, cfg,
        )
        if reflection_reason:
            obj["_suppressed_reason"] = f"likely_reflection:{reflection_reason}"
            suppressed_reflection.append({
                "category": str(obj.get("category", "unknown")),
                "reason": f"likely_reflection:{reflection_reason}",
                "score": round(float(obj.get("score", 0.0)), 3),
            })
            logger.info(
                "  SUPPRESSED (reflection): %s score=%.2f reason=%s",
                obj.get("category", "unknown"),
                float(obj.get("score", 0.0)),
                reflection_reason,
            )
            continue

        if _is_wall_mounted_misdetection(obj, walls, floor_height, cfg):
            obj["_suppressed_reason"] = "wall_mounted_floor_category"
            suppressed_wallmount.append({
                "category": str(obj.get("category", "unknown")),
                "reason": "wall_mounted_floor_category",
                "score": round(float(obj.get("score", 0.0)), 3),
                "y": round(float(obj.get("y", 0.0)), 3),
            })
            logger.info(
                "  SUPPRESSED (wall-mounted): %s score=%.2f y=%.2f",
                obj.get("category", "unknown"),
                float(obj.get("score", 0.0)),
                float(obj.get("y", 0.0)),
            )
            continue

        pre_filter_clustered.append(obj)
    if pre_filter_clustered:
        clustered = pre_filter_clustered
    if suppressed_reflection:
        logger.info("Reflection filter suppressed %d objects", len(suppressed_reflection))
    if suppressed_wallmount:
        logger.info("Wall-mounted filter suppressed %d objects", len(suppressed_wallmount))

    # Step 9b: Suppress weak inferred context objects by default.
    # These are useful as debug hints but frequently reduce placement precision.
    suppressed_inferred: List[Dict] = []
    filtered_clustered: List[Dict] = []
    for obj in clustered:
        pos_src = str(obj.get("position_source", ""))
        pos_unc = float(obj.get("position_uncertainty_m", 0.0) or 0.0)
        if pos_src == "context_inference" and pos_unc >= 0.55:
            obj["_suppressed_reason"] = "high_uncertainty_context_inference"
            suppressed_inferred.append({
                "category": str(obj.get("category", "unknown")),
                "reason": "high_uncertainty_context_inference",
                "position_source": pos_src,
                "position_uncertainty_m": round(pos_unc, 4),
                "score": round(float(obj.get("score", 0.0)), 3),
            })
            logger.info(
                "  SUPPRESSED (context): %s score=%.2f pos_unc=%.2f",
                obj.get("category", "unknown"),
                float(obj.get("score", 0.0)),
                pos_unc,
            )
            continue
        filtered_clustered.append(obj)
    if filtered_clustered:
        clustered = filtered_clustered
    _lift_diagnostics["suppressed_inferred_objects_total"] = int(len(suppressed_inferred))
    _lift_diagnostics["suppressed_inferred_objects"] = suppressed_inferred

    # Step 10: Assign IDs and build final dicts
    result = _build_room_objects(clustered)

    logger.info("Final output: %d objects", len(result))
    _debug_output(clustered)

    return result


def _lift_single_track(
    track: Track,
    depth_maps: List[Optional["np.ndarray"]],
    shots: List[ShotMetadata],
    cfg: PipelineConfig,
    floor_height: float = 0.0,
    triangulation_prior: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Estimate 3D position for a single track.

    Session 11: Feature-point-first positioning.
      Primary: ARKit feature-point centroid (metric, weighted, filtered)
      Fallback: Depth back-projection (legacy)

    Implements 5 corrections for production quality:
      1. Temporal confidence weighting (points seen in more frames → higher weight)
      2. Depth consistency filter (in associate_fp_to_detection)
      3. Multi-view spatial consistency (points must appear in ≥2 frames for PCA)
      4. Feature density normalization (voxel-grid before centroid)
      5. Accumulated centroid (not per-frame)
    """
    best = track.best_detection
    candidates = [best] + [d for d in sorted(track.detections, key=lambda x: -x.score)
                           if d is not best]

    # ── Step 1: Collect feature points across ALL frames ──────────────
    # For each detection, associate ARKit feature points that project inside it
    all_fp_world = []       # (x, y, z) world-space points
    all_fp_frame_idx = []   # which frame each point came from
    all_fp_quality_w = []   # shot-level usability weights per point
    pca_cam_positions = []  # camera positions for PCA disambiguation

    per_frame_fp_counts = []  # Debug: per-frame FP-in-bbox counts
    for d in candidates:
        fi = d.frame_idx
        if fi >= len(shots):
            per_frame_fp_counts.append({
                "frame": fi, "shot_fp": -1, "in_bbox": "skip:no_shot",
                "bbox": list(d.bbox), "image_size": list(d.input_size),
                "intrinsics_image_size": None, "fp_visible": 0, "fp_behind": 0,
                "bbox_space": None, "fp_uv_bounds": None,
                "in_mask": 0,
                "pixel_space_contract": None,
            })
            continue
        shot = shots[fi]
        if shot.intrinsics is None:
            n_fp_raw = len(shot.feature_points_w) if shot.feature_points_w else 0
            per_frame_fp_counts.append({
                "frame": fi, "shot_fp": n_fp_raw, "in_bbox": "skip:no_intrinsics",
                "bbox": list(d.bbox), "image_size": list(d.input_size),
                "intrinsics_image_size": [shot.intrinsics.width, shot.intrinsics.height]
                    if shot.intrinsics else None,
                "fp_visible": 0, "fp_behind": 0,
                "bbox_space": None, "fp_uv_bounds": None,
                "in_mask": 0,
                "pixel_space_contract": None,
            })
            continue

        n_fp_raw = len(shot.feature_points_w) if shot.feature_points_w else 0
        fp_points, fp_diag = associate_fp_to_detection(d, shot, return_diag=True)
        uv_bounds = fp_diag.get("fp_uv_bounds", None)
        uv_area_ratio = None
        if (
            uv_bounds is not None
            and isinstance(uv_bounds, list)
            and len(uv_bounds) == 4
            and shot.intrinsics is not None
        ):
            try:
                u_span = max(0.0, float(uv_bounds[1]) - float(uv_bounds[0]))
                v_span = max(0.0, float(uv_bounds[3]) - float(uv_bounds[2]))
                img_area = max(1.0, float(shot.intrinsics.width * shot.intrinsics.height))
                uv_area_ratio = (u_span * v_span) / img_area
            except Exception:
                uv_area_ratio = None
        q = shot.quality
        texture_score = float(q.texture_score) if q is not None else 0.0
        blur_score = float(q.blur_score) if q is not None else 0.0
        edge_density = float(q.edge_density) if q is not None else 0.0
        match_seed = float(q.matchability_seed) if q is not None else 0.0
        quality_norm = 0.0
        quality_norm += max(0.0, min(1.0, texture_score))
        quality_norm += max(0.0, min(1.0, edge_density * 14.0))
        quality_norm += max(0.0, min(1.0, blur_score / 120.0))
        quality_norm += max(0.0, min(1.0, match_seed / 140.0))
        quality_norm /= 4.0
        low_w = float(getattr(cfg, "fp_usability_low_quality_weight", 0.45))
        hi_w = float(getattr(cfg, "fp_usability_high_quality_weight", 1.15))
        frame_usability_w = low_w + (hi_w - low_w) * quality_norm
        min_uv_area = float(getattr(cfg, "fp_usability_min_uv_area_ratio", 0.02))
        if uv_area_ratio is not None and uv_area_ratio < min_uv_area:
            frame_usability_w *= 0.72
        if shot.capture_step_m is not None and float(shot.capture_step_m) > 1.25:
            frame_usability_w *= 0.82
        frame_usability_w = max(0.2, min(1.3, frame_usability_w))
        per_frame_fp_counts.append({
            "frame": fi,
            "shot_fp": n_fp_raw,
            "in_bbox": len(fp_points),
            "bbox": list(d.bbox),
            "image_size": list(d.input_size),
            "intrinsics_image_size": [shot.intrinsics.width, shot.intrinsics.height]
                if shot.intrinsics else None,
            "fp_visible": fp_diag.get("fp_visible", 0),
            "fp_behind": fp_diag.get("fp_behind", 0),
            "bbox_space": fp_diag.get("bbox_space", None),
            "fp_uv_bounds": fp_diag.get("fp_uv_bounds", None),
            "in_mask": fp_diag.get("in_mask", 0),
            "uv_area_ratio": round(float(uv_area_ratio), 4) if uv_area_ratio is not None else None,
            "frame_usability_w": round(float(frame_usability_w), 3),
            "pixel_space_contract": fp_diag.get("pixel_space_contract", None),
        })
        if len(fp_points) > 0:
            all_fp_world.extend(fp_points.tolist())
            all_fp_frame_idx.extend([fi] * len(fp_points))
            all_fp_quality_w.extend([frame_usability_w] * len(fp_points))

            # Camera position for disambiguation
            if shot.pose is not None and len(shot.pose) == 16:
                mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
                pca_cam_positions.append(mat[:3, 3].tolist())

    # ── Step 2: Temporal confidence weighting ─────────────────────────
    # Hash each point to a 2cm voxel, count how many frames each voxel appears in
    fp_count = len(all_fp_world)
    position_source = "feature_points"
    temporal_weights = None
    pca_world_points = []

    if fp_count >= 5:
        fp_arr = np.array(all_fp_world, dtype=np.float64)  # (N, 3)
        fp_frames = np.array(all_fp_frame_idx, dtype=np.int32)
        quality_w = np.array(all_fp_quality_w, dtype=np.float64)

        # Voxel keys for temporal counting (2cm grid)
        voxel_coords = np.floor(fp_arr / FP_VOXEL_DEDUP).astype(np.int64)
        voxel_keys = (voxel_coords[:, 0] * 1_000_000 +
                      voxel_coords[:, 1] * 1_000 +
                      voxel_coords[:, 2])

        # For each voxel, count unique frame indices
        voxel_frame_count = {}
        for i in range(len(voxel_keys)):
            vk = int(voxel_keys[i])
            fi = int(fp_frames[i])
            if vk not in voxel_frame_count:
                voxel_frame_count[vk] = set()
            voxel_frame_count[vk].add(fi)

        # Assign temporal weight to each point = number of frames its voxel was seen in
        temporal_weights = np.array([
            len(voxel_frame_count.get(int(voxel_keys[i]), set()))
            for i in range(len(fp_arr))
        ], dtype=np.float64)
        temporal_weights = temporal_weights * np.clip(quality_w, 0.2, 1.5)

        # ── PCA points: only those seen in ≥2 frames (Correction 3) ──
        multi_frame_mask = temporal_weights >= 2
        if np.sum(multi_frame_mask) >= 10:
            pca_world_points = fp_arr[multi_frame_mask].tolist()
        else:
            pca_world_points = fp_arr.tolist()

        # ── Step 3: Feature density normalization (Correction 4) ──────
        # Voxel-grid normalize at 5cm to prevent bias toward textured regions
        voxel_density = np.floor(fp_arr / FP_VOXEL_DENSITY).astype(np.int64)
        density_keys = (voxel_density[:, 0] * 1_000_000 +
                        voxel_density[:, 1] * 1_000 +
                        voxel_density[:, 2])
        _, unique_idx = np.unique(density_keys, return_index=True)
        fp_normed = fp_arr[unique_idx]
        weights_normed = temporal_weights[unique_idx]

        if len(fp_normed) < 3:
            fp_normed = fp_arr
            weights_normed = temporal_weights

        # ── Step 4: MAD outlier rejection ─────────────────────────────
        centroid_rough = np.median(fp_normed, axis=0)
        dists = np.linalg.norm(fp_normed - centroid_rough, axis=1)
        mad = float(np.median(dists))
        if mad > 0.001:
            keep = dists < 2.5 * mad
            if np.sum(keep) >= 3:
                fp_normed = fp_normed[keep]
                weights_normed = weights_normed[keep]

        # ── Step 5: Weighted centroid computation (Correction 1) ──────
        centroid_x = _weighted_median(fp_normed[:, 0], weights_normed)
        centroid_z = _weighted_median(fp_normed[:, 2], weights_normed)
        centroid_y = floor_height  # from layout

        # Variance for diagnostics
        fp_variance = float(np.mean(np.var(fp_normed, axis=0)))
        mean_temporal_weight = float(np.mean(temporal_weights))

        world_x, world_y, world_z = centroid_x, centroid_y, centroid_z

    else:
        fp_variance = 0.0
        mean_temporal_weight = 0.0

        # ── Fallback A: multi-view ray triangulation prior ────────────
        if triangulation_prior is not None:
            position_source = "triangulated_rays"
            world_x = float(triangulation_prior.get("x", 0.0))
            world_y = float(triangulation_prior.get("y", floor_height))
            world_z = float(triangulation_prior.get("z", 0.0))
            result = None
            det = best
            frame_idx = best.frame_idx
        else:
            # ── Fallback B: depth back-projection ─────────────────────
            position_source = "depth_backproject"
            result = None
            det = None
            frame_idx = None
            for d in candidates:
                fi = d.frame_idx
                if fi >= len(depth_maps) or depth_maps[fi] is None:
                    continue
                if fi >= len(shots):
                    continue
                r = robust_lift_detection(d, depth_maps[fi], shots[fi])
                if r is not None:
                    result = r
                    det = d
                    frame_idx = fi
                    break

            if result is None:
                return None

            world_x, world_y, world_z = result.world_pos

    # ── Collect camera yaws from ALL valid detections for median ──────
    all_yaws = []
    for d in candidates:
        fi = d.frame_idx
        if fi >= len(shots) or shots[fi].pose is None:
            continue
        pose = shots[fi].pose
        if pose is None or len(pose) != 16:
            continue
        mat = np.array(pose, dtype=np.float64).reshape(4, 4)
        cam_fwd_x = -mat[0, 2]
        cam_fwd_z = -mat[2, 2]
        yaw = float(np.arctan2(cam_fwd_x, cam_fwd_z))
        all_yaws.append(yaw)

    if all_yaws:
        sin_vals = np.array([sin(y) for y in all_yaws])
        cos_vals = np.array([cos(y) for y in all_yaws])
        median_yaw = float(np.arctan2(np.median(sin_vals), np.median(cos_vals)))
    else:
        median_yaw = None

    # ── 2D bbox size fractions (for _estimate_dimensions) ─────────────
    det = best
    frame_idx = det.frame_idx
    width_2d = 0.0
    height_2d = 0.0
    depth_median = 0.0
    depth_mad_val = 0.0
    depth_samples = 0

    if frame_idx < len(depth_maps) and depth_maps[frame_idx] is not None:
        dm = depth_maps[frame_idx]
        img_h, img_w = dm.shape[:2]
        x1, y1, x2, y2 = det.bbox
        det_w, det_h = det.input_size
        sx = img_w / det_w
        sy = img_h / det_h
        bx1 = int(max(0, x1 * sx))
        by1 = int(max(0, y1 * sy))
        bx2 = int(min(img_w, x2 * sx))
        by2 = int(min(img_h, y2 * sy))
        width_2d = (bx2 - bx1) / img_w if img_w > 0 else 0
        height_2d = (by2 - by1) / img_h if img_h > 0 else 0

        # Get depth stats for diagnostics even if not using for position
        if position_source == "depth_backproject" and result is not None:
            depth_median = result.depth_value
            depth_mad_val = result.depth_mad
            depth_samples = result.valid_sample_count

    # Trim PCA points
    MAX_PCA_POINTS = 500
    if len(pca_world_points) > MAX_PCA_POINTS:
        pca_world_points = pca_world_points[:MAX_PCA_POINTS]

    # Log per-frame FP association details
    logger.debug(
        "Lift %s (track=%d): %s, fp_count=%d, world=(%.3f,%.3f,%.3f), pca_pts=%d",
        track.category, track.track_id, position_source, fp_count,
        world_x, world_y, world_z, len(pca_world_points),
    )
    for pf in per_frame_fp_counts:
        logger.debug("  frame %d: shot_fp=%s, in_bbox=%s, visible=%s, behind=%s",
                     pf["frame"], pf["shot_fp"], pf["in_bbox"],
                     pf.get("fp_visible", "?"), pf.get("fp_behind", "?"))

    fp_ratio_vals = []
    frames_with_fp_hit = 0
    frames_total = 0
    for pf in per_frame_fp_counts:
        in_bbox = pf.get("in_bbox")
        shot_fp = pf.get("shot_fp")
        if not isinstance(in_bbox, int):
            continue
        if not isinstance(shot_fp, int):
            continue
        frames_total += 1
        if in_bbox > 0:
            frames_with_fp_hit += 1
        denom = max(1, shot_fp)
        fp_ratio_vals.append(float(in_bbox) / float(denom))
    fp_ratio_mean = float(np.mean(fp_ratio_vals)) if fp_ratio_vals else 0.0
    fp_ratio_max = float(np.max(fp_ratio_vals)) if fp_ratio_vals else 0.0

    if position_source == "feature_points":
        support_views = int(max(frames_with_fp_hit, 1))
        position_uncertainty_m = float(max(0.06, min(0.9, sqrt(max(fp_variance, 1e-6)))))
    elif position_source == "triangulated_rays":
        tri_err = float(triangulation_prior.get("median_ray_error_m", 0.18)) if triangulation_prior else 0.18
        support_views = int(max(1, triangulation_prior.get("views", 0) if triangulation_prior else 1))
        position_uncertainty_m = float(max(0.08, min(1.2, tri_err * 1.25)))
    elif position_source == "context_inference":
        support_views = int(max(1, len(track.detections)))
        position_uncertainty_m = 0.65
    else:
        support_views = int(max(1, len(track.detections)))
        position_uncertainty_m = float(max(0.2, min(1.5, 0.1 + 2.5 * float(depth_mad_val))))

    high_uncertainty = bool(
        support_views < 2
        or (position_source in {"depth_backproject", "context_inference"} and len(track.detections) <= 1)
    )

    return {
        "category": track.category,
        "x": world_x,
        "y": world_y,
        "z": world_z,
        "score": track.avg_score,
        "track_id": track.track_id,
        "depth_median": depth_median,
        "bbox": det.bbox,
        "frame_idx": frame_idx,
        "width_2d": width_2d,
        "height_2d": height_2d,
        "n_detections": len(track.detections),
        "camera_yaw": all_yaws[0] if all_yaws else None,
        "median_camera_yaw": median_yaw,
        "depth_mad": depth_mad_val,
        "depth_sample_count": depth_samples,
        # Feature-point diagnostics (Session 11)
        "position_source": position_source,
        "fp_count": fp_count,
        "fp_variance": round(fp_variance, 6),
        "fp_temporal_mean_weight": round(mean_temporal_weight, 2),
        "fp_in_bbox_ratio_mean": round(fp_ratio_mean, 4),
        "fp_in_bbox_ratio_max": round(fp_ratio_max, 4),
        "frames_with_fp_hit": frames_with_fp_hit,
        "frames_total": frames_total,
        "support_views": support_views,
        "position_uncertainty_m": round(float(position_uncertainty_m), 4),
        "_high_uncertainty": high_uncertainty,
        "triangulation_views": int(triangulation_prior.get("views", 0))
            if triangulation_prior is not None else 0,
        "triangulation_median_ray_error_m": float(
            triangulation_prior.get("median_ray_error_m", 0.0)
        ) if triangulation_prior is not None else 0.0,
        # Per-frame FP breakdown (Session 12 debug)
        "_per_frame_fp": per_frame_fp_counts,
        # Per-detection diagnostics (Session 13: placement debugging)
        "_detection_details": [
            {
                "frame_idx": d.frame_idx,
                "bbox": list(d.bbox),
                "score": round(d.score, 3),
                "category": d.category,
                "image_size": list(d.input_size),
                "tracking_quality": shots[d.frame_idx].tracking_quality
                    if d.frame_idx < len(shots) else "unknown",
            }
            for d in track.detections
        ],
        # Multi-frame PCA data (now from FP, not depth)
        "_pca_world_points": pca_world_points,
        "_pca_cam_positions": pca_cam_positions,
        # Placeholders
        "width": 1.0,
        "length": 1.0,
        "height": 1.0,
        "rotation_y": 0.0,
    }


def _bbox_iou(a: List[float], b: List[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(1e-6, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    area_b = max(1e-6, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
    return float(inter / max(1e-6, area_a + area_b - inter))


def _bbox_containment(a: List[float], b: List[float]) -> float:
    """Intersection over smaller box area."""
    if len(a) < 4 or len(b) < 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(1e-6, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    area_b = max(1e-6, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
    return float(inter / max(1e-6, min(area_a, area_b)))


def _best_det_by_frame(obj: Dict) -> Dict[int, Dict]:
    by_frame: Dict[int, Dict] = {}
    for d in obj.get("_detection_details", []):
        fi = d.get("frame_idx")
        if fi is None:
            continue
        prev = by_frame.get(int(fi))
        if prev is None or float(d.get("score", 0.0)) > float(prev.get("score", 0.0)):
            by_frame[int(fi)] = d
    return by_frame


def _same_frame_duplicate_evidence(
    a: Dict,
    b: Dict,
    cfg: PipelineConfig,
) -> Tuple[bool, float, float]:
    """
    Return whether same-frame detections look like duplicates.

    True means the pair can be merged even when they co-exist in the same frame.
    """
    by_a = _best_det_by_frame(a)
    by_b = _best_det_by_frame(b)
    shared = sorted(set(by_a.keys()) & set(by_b.keys()))
    if not shared:
        return True, 0.0, 0.0
    ious: List[float] = []
    contains: List[float] = []
    for f in shared:
        da = by_a[f]
        db = by_b[f]
        ba = da.get("bbox", [])
        bb = db.get("bbox", [])
        ious.append(_bbox_iou(ba, bb))
        contains.append(_bbox_containment(ba, bb))
    med_iou = float(np.median(np.array(ious, dtype=np.float64))) if np is not None and ious else 0.0
    med_cont = (
        float(np.median(np.array(contains, dtype=np.float64)))
        if np is not None and contains else 0.0
    )
    iou_thr = float(getattr(cfg, "cluster_same_frame_iou_duplicate_min", 0.55))
    cont_thr = float(getattr(cfg, "cluster_same_frame_containment_min", 0.82))
    is_dup = med_iou >= iou_thr or med_cont >= cont_thr
    return is_dup, med_iou, med_cont


def _cluster_objects(objects: List[Dict], cfg: PipelineConfig) -> List[Dict]:
    """
    Merge duplicate detections of the same object across frames.

    Clustering relies on per-category merge distances (config.py) and skips
    size similarity checks for all categories (depth noise makes dimensions
    unreliable across viewpoints). No hard instance caps — rooms vary widely.
    """
    if not objects:
        return []

    # Group by category
    by_cat: Dict[str, List[Dict]] = {}
    for obj in objects:
        by_cat.setdefault(obj["category"], []).append(obj)

    merged = []
    for cat, cat_objs in by_cat.items():
        # Sort by score descending
        cat_objs.sort(key=lambda o: -o["score"])

        merge_dist = cfg.object_merge_distance_by_type.get(cat, cfg.object_merge_distance_m)
        skip_size = cat in SKIP_SIZE_CHECK
        small_crossframe_only = bool(
            getattr(cfg, "cluster_small_object_crossframe_only", True)
        ) and cat in SMALL_OBJECT_CATS

        clusters: List[List[Dict]] = []

        for obj in cat_objs:
            placed = False
            for cluster in clusters:
                rep = cluster[0]  # Representative (highest score)
                dist = _distance_3d(obj, rep)

                if dist <= merge_dist:
                    rep_frames = set(_best_det_by_frame(rep).keys())
                    obj_frames = set(_best_det_by_frame(obj).keys())
                    shared_frames = sorted(rep_frames & obj_frames)

                    # Prevent same-frame merges unless detections clearly duplicate.
                    if shared_frames:
                        dup_ok, med_iou, med_cont = _same_frame_duplicate_evidence(rep, obj, cfg)
                        if not dup_ok:
                            continue
                    else:
                        med_iou, med_cont = 0.0, 0.0

                    # For small objects, require cross-frame support before merge.
                    if small_crossframe_only:
                        merged_frames = rep_frames | obj_frames
                        if len(merged_frames) < 2:
                            continue

                    # Size similarity check (skip for large furniture)
                    if not skip_size:
                        size_ratio = _size_similarity(obj, rep)
                        if size_ratio < cfg.object_merge_size_ratio:
                            continue
                    cluster.append(obj)
                    if shared_frames:
                        _cluster_same_frame_merge_events.append({
                            "category": cat,
                            "track_a": rep.get("track_id"),
                            "track_b": obj.get("track_id"),
                            "shared_frames": [int(f) for f in shared_frames],
                            "distance_m": round(float(dist), 4),
                            "median_iou": round(float(med_iou), 4),
                            "median_containment": round(float(med_cont), 4),
                        })
                    placed = True
                    break

            if not placed:
                clusters.append([obj])

        # Post-clustering instance cap for categories prone to duplication
        _INSTANCE_CAPS = {"bed": 2, "sofa": 2, "desk": 2, "dresser": 2, "monitor": 3}
        cap = _INSTANCE_CAPS.get(cat)
        if cap is not None and len(clusters) > cap:
            clusters = clusters[:cap]  # already sorted by score descending

        for cluster in clusters:
            merged_obj = _merge_cluster(cluster)
            merged.append(merged_obj)

        logger.info("Category '%s': %d tracks → %d clusters", cat, len(cat_objs), len(clusters))

    # Log cluster summary
    cluster_report: Dict[str, int] = {}
    for obj in merged:
        cluster_report[obj["category"]] = cluster_report.get(obj["category"], 0) + 1
    logger.info("Clustered objects: %s",
                ", ".join(f"{k}:{v}" for k, v in sorted(cluster_report.items())))

    return merged


def _distance_3d(a: Dict, b: Dict) -> float:
    """Euclidean distance between two object positions."""
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    dz = a["z"] - b["z"]
    return sqrt(dx * dx + dy * dy + dz * dz)


def _size_similarity(a: Dict, b: Dict) -> float:
    """Compute size similarity ratio between two objects (0 = different, 1 = same)."""
    w_a = a.get("width_2d", 0.1)
    h_a = a.get("height_2d", 0.1)
    w_b = b.get("width_2d", 0.1)
    h_b = b.get("height_2d", 0.1)

    # Ratio of smaller to larger for each dimension
    w_ratio = min(w_a, w_b) / max(w_a, w_b) if max(w_a, w_b) > 0 else 1.0
    h_ratio = min(h_a, h_b) / max(h_a, h_b) if max(h_a, h_b) > 0 else 1.0

    return (w_ratio + h_ratio) / 2.0


def _merge_cluster(cluster: List[Dict]) -> Dict:
    """Merge a cluster of detections into a single object."""
    if len(cluster) == 1:
        return cluster[0]

    # Weighted average position by score
    total_weight = sum(o["score"] for o in cluster)
    if total_weight < 1e-6:
        total_weight = len(cluster)
        weights = [1.0 / total_weight] * len(cluster)
    else:
        weights = [o["score"] / total_weight for o in cluster]

    avg_x = sum(o["x"] * w for o, w in zip(cluster, weights))
    avg_y = sum(o["y"] * w for o, w in zip(cluster, weights))
    avg_z = sum(o["z"] * w for o, w in zip(cluster, weights))

    best = max(cluster, key=lambda o: o["score"])

    # Concatenate detection details from all cluster members
    all_details = []
    for o in cluster:
        all_details.extend(o.get("_detection_details", []))
    support_views = max(int(o.get("support_views", 0)) for o in cluster)
    pos_unc = float(np.mean([float(o.get("position_uncertainty_m", 0.4)) for o in cluster])) if np is not None else float(cluster[0].get("position_uncertainty_m", 0.4))
    high_unc = any(bool(o.get("_high_uncertainty", False)) for o in cluster)

    return {
        **best,
        "x": avg_x,
        "y": avg_y,
        "z": avg_z,
        "score": best["score"],
        "n_detections": sum(o.get("n_detections", 1) for o in cluster),
        "support_views": support_views,
        "position_uncertainty_m": round(float(pos_unc), 4),
        "_high_uncertainty": high_unc,
        "_detection_details": all_details,
    }


def _snap_to_cardinal(angle: float) -> float:
    """Snap an angle to the nearest cardinal direction (0, pi/2, pi, -pi/2).

    This eliminates random/unstable orientations from noisy depth data.
    Most furniture in residential rooms is axis-aligned.
    """
    # Normalize to [-pi, pi]
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi

    # Snap to nearest 90 degrees
    snapped = round(angle / (pi / 2)) * (pi / 2)

    # Re-normalize
    if snapped > pi:
        snapped -= 2 * pi
    elif snapped < -pi:
        snapped += 2 * pi

    return snapped


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle


def _room_axes_from_walls(walls: List[Dict]) -> Optional[Tuple[float, float]]:
    """
    Estimate room principal axes from wall segments.

    Returns:
        (axis_a, axis_b) where axis_b is perpendicular to axis_a.
        Angles follow atan2(x, z), CCW-positive convention.
    """
    longest_len = 0.0
    primary_axis = None

    for wall in walls:
        sx, sz = wall["start"]
        ex, ez = wall["end"]
        dx = ex - sx
        dz = ez - sz
        wall_len = sqrt(dx * dx + dz * dz)
        if wall_len < 0.01:
            continue
        if wall_len > longest_len:
            longest_len = wall_len
            primary_axis = atan2(dx, dz)

    if primary_axis is None:
        return None

    axis_a = _normalize_angle(primary_axis)
    axis_b = _normalize_angle(axis_a + pi / 2)
    return (axis_a, axis_b)


def _snap_to_room_axes(
    angle: float,
    axes: Optional[Tuple[float, float]],
    axis_confidence: float = 1.0,
) -> float:
    """Snap angle to nearest room axis direction (±axis_a or ±axis_b)."""
    if axis_confidence < AXIS_SNAP_MIN_CONFIDENCE:
        return _normalize_angle(angle)

    if axes is None:
        return _snap_to_cardinal(angle)

    axis_a, axis_b = axes
    candidates = [
        axis_a,
        _normalize_angle(axis_a + pi),
        axis_b,
        _normalize_angle(axis_b + pi),
    ]
    best = candidates[0]
    best_diff = float("inf")
    for candidate in candidates:
        diff = abs(atan2(sin(angle - candidate), cos(angle - candidate)))
        if diff < best_diff:
            best_diff = diff
            best = candidate
    return best


def _snap_to_wall_axes(angle: float, wall_normal_angle: float) -> float:
    """Snap an angle to the 4 wall-aligned axes.

    Given a wall with inward normal `wall_normal_angle`, the 4 axis-aligned
    directions are:
        wall_normal_angle           (perpendicular to wall, facing into room)
        wall_normal_angle + π       (perpendicular to wall, facing wall)
        wall_normal_angle + π/2     (parallel to wall, one direction)
        wall_normal_angle - π/2     (parallel to wall, other direction)

    Returns whichever of the 4 is closest to `angle`.
    """
    candidates = [
        wall_normal_angle,
        wall_normal_angle + pi,
        wall_normal_angle + pi / 2,
        wall_normal_angle - pi / 2,
    ]

    # Normalize to [-π, π]
    def _norm(a: float) -> float:
        while a > pi:
            a -= 2 * pi
        while a < -pi:
            a += 2 * pi
        return a

    best = candidates[0]
    best_diff = float("inf")
    for c in candidates:
        cn = _norm(c)
        diff = abs(atan2(sin(angle - cn), cos(angle - cn)))
        if diff < best_diff:
            best_diff = diff
            best = cn

    return best


def _axis_snap_min_conf_for_object(category: str, near_wall: bool) -> float:
    """Category-aware axis-snap confidence threshold."""
    if near_wall and category in AXIS_SNAP_RELAXED_MIN_CONF_BY_CAT:
        return float(AXIS_SNAP_RELAXED_MIN_CONF_BY_CAT[category])
    return AXIS_SNAP_MIN_CONFIDENCE


def _estimate_orientation(
    obj: Dict,
    walls: List[Dict],
    room_width: float,
    room_length: float,
    depth_maps: List[Optional["np.ndarray"]],
    shots: List[ShotMetadata],
    all_objects: Optional[List[Dict]] = None,
    room_center: Optional[Tuple[float, float]] = None,
    axis_confidence: float = 0.5,
) -> float:
    """
    Estimate object rotation_y.

    Room-frame-first orientation logic.

    Priority (for most categories):
      1. PCA: multi-view object points define a principal axis, then snap to
         room axes derived from walls.
      2. Wall normal: for wall-adjacent furniture, use inward wall normal and
         snap to room axes.
      3. Camera yaw fallback: strict last resort, snapped to room axes.
      4. Default: 0.0.

    Chairs have special context: they face the nearest desk/table.

    Symmetric/flat objects (lamp, plant, rug) always return 0.0.

    Returns rotation in radians (CCW positive, atan2(x,z) convention).
    Backend sends this; iOS negates for SceneKit: eulerAngles.y = -rotation_y.
    """
    cat = obj["category"]
    ox, oz = obj["x"], obj["z"]
    orient_trace: List[str] = []

    # ── Symmetric / flat objects: no meaningful facing direction ──
    if cat in SYMMETRIC_TYPES:
        obj["_orient_source"] = "symmetric"
        obj["_orient_trace"] = ["symmetric", "final=0.0deg"]
        return 0.0

    room_axes = _room_axes_from_walls(walls)

    # ── Wall info: inward normal angle + distance ──
    wall_info = _nearest_wall_info(ox, oz, walls, room_center=room_center)
    wall_normal = None
    wall_dist = float("inf")
    if wall_info is not None:
        wall_normal, wall_dist = wall_info

    near_wall = wall_normal is not None and wall_dist <= WALL_PROXIMITY_THRESHOLD
    high_uncertainty = bool(obj.get("_high_uncertainty", False))
    support_views = int(obj.get("support_views", 0))
    axis_snap_min_conf = _axis_snap_min_conf_for_object(cat, bool(near_wall))
    orient_trace.append(
        f"wall_proximity(dist={wall_dist:.2f},near={str(bool(near_wall)).lower()})"
        if wall_dist != float("inf")
        else "wall_proximity(dist=inf,near=false)"
    )

    logger.debug("Orient %s at (%.3f, %.3f): wall_dist=%.3f wall_normal=%.1f° near=%s",
                 cat, ox, oz, wall_dist, wall_normal * 180 / pi if wall_normal is not None else 0,
                 near_wall)

    # ── Camera yaw (median across all observations) ──
    median_yaw = obj.get("median_camera_yaw")

    # Camera-based facing: camera looks at the object front,
    # so object faces opposite to camera direction → median_yaw + π
    cam_facing = None
    if median_yaw is not None:
        cam_facing = median_yaw + pi
        # Normalize to [-π, π]
        while cam_facing > pi:
            cam_facing -= 2 * pi
        while cam_facing < -pi:
            cam_facing += 2 * pi

    def _snap_with_trace(
        angle: float,
        *,
        allow_axis_snap: bool = True,
        skip_reason: Optional[str] = None,
    ) -> float:
        if high_uncertainty and allow_axis_snap:
            orient_trace.append("snap_skipped(high_uncertainty)")
            return _normalize_angle(angle)
        if not allow_axis_snap:
            orient_trace.append(f"snap_skipped({skip_reason or 'weak_source'})")
            return _normalize_angle(angle)
        if axis_confidence < axis_snap_min_conf:
            orient_trace.append(
                f"snap_skipped(axis_conf={axis_confidence:.2f},min={axis_snap_min_conf:.2f})"
            )
            return _normalize_angle(angle)
        snapped = _snap_to_room_axes(
            angle,
            room_axes,
            axis_confidence=max(axis_confidence, AXIS_SNAP_MIN_CONFIDENCE),
        )
        orient_trace.append(
            f"snap_to_axes(axis_conf={axis_confidence:.2f},min={axis_snap_min_conf:.2f})"
        )
        return snapped

    # Helper to store orientation debug metadata on the object dict
    def _return(angle: float, source: str, **extra) -> float:
        obj["_orient_source"] = source
        obj["_orient_wall_dist"] = round(wall_dist, 3) if wall_dist != float("inf") else None
        obj["_orient_wall_normal"] = round(wall_normal, 4) if wall_normal is not None else None
        obj["_orient_median_cam_yaw"] = round(median_yaw, 4) if median_yaw is not None else None
        obj["_orient_axis_confidence"] = round(axis_confidence, 3)
        orient_trace.append(f"source={source}")
        orient_trace.append(f"final={round(float(angle) * 180.0 / pi, 1)}deg")
        obj["_orient_trace"] = orient_trace
        base_uncertainty = {
            "pca": 12.0,
            "wall_room_axes": 18.0,
            "desk_assoc_room_axes": 20.0,
            "camera_last_resort": 42.0,
            "camera_last_resort_neutral": 72.0,
            "default": 58.0,
            "symmetric": 90.0,
        }.get(source, 45.0)
        if high_uncertainty:
            base_uncertainty += 15.0
        if axis_confidence < axis_snap_min_conf:
            base_uncertainty += 8.0
        obj["_orientation_uncertainty_deg"] = round(min(89.0, max(5.0, base_uncertainty)), 2)
        for k, v in extra.items():
            obj[f"_orient_{k}"] = v
        return angle

    def _camera_last_resort_return() -> float:
        orient_trace.append("camera_last_resort")
        if support_views < 2 or high_uncertainty:
            orient_trace.append(
                f"camera_last_resort_neutral(support_views={support_views},high_unc={str(high_uncertainty).lower()})"
            )
            return _return(0.0, "camera_last_resort_neutral", support_views=support_views)
        cam_cardinal = _snap_with_trace(
            cam_facing,
            allow_axis_snap=False,
            skip_reason="camera_last_resort",
        )
        return _return(cam_cardinal, "camera_last_resort", support_views=support_views)

    # ── Chair: context-dependent orientation ──
    if cat == "chair":
        # Priority 1: face nearest desk/table
        if all_objects is not None:
            desk = _find_nearest_desk_or_table(obj, all_objects)
            if desk is not None:
                facing = _angle_from_to(ox, oz, desk["x"], desk["z"])
                orient_trace.append(
                    f"desk_assoc(track={desk.get('track_id')},dist={sqrt((desk['x'] - ox) ** 2 + (desk['z'] - oz) ** 2):.2f})"
                )
                logger.debug("Chair at (%.2f, %.2f) faces desk at (%.2f, %.2f) → %.1f°",
                             ox, oz, desk["x"], desk["z"], facing * 180 / pi)
                return _return(
                               _snap_with_trace(facing),
                               "desk_assoc_room_axes",
                               axis_confidence=round(axis_confidence, 3),
                               desk_target_id=desk.get("track_id"))

        # Priority 2: PCA
        pca_angle = _pca_orientation(
            obj,
            walls,
            room_center=room_center,
            room_axes=room_axes,
            axis_confidence=axis_confidence,
            axis_snap_min_conf=axis_snap_min_conf,
            orient_trace=orient_trace,
        )
        if pca_angle is not None:
            logger.debug("Chair at (%.2f, %.2f) → PCA orientation %.1f°",
                         ox, oz, pca_angle * 180 / pi)
            return _return(pca_angle, "pca")
        pca_ambiguous = "pca_ambiguous" in orient_trace

        # Priority 3: face away from nearest wall (inward normal)
        if near_wall:
            orient_trace.append("use_wall_normal")
            return _return(
                _snap_with_trace(
                    wall_normal,
                    allow_axis_snap=not pca_ambiguous,
                    skip_reason="pca_ambiguous",
                ),
                "wall_room_axes",
            )

        # Priority 4: camera fallback (strict last resort)
        if cam_facing is not None:
            return _camera_last_resort_return()
        return _return(0.0, "default")

    # ── Wall-facing objects ──
    if cat in WALL_FACING_TYPES:
        # Priority 1: PCA (captures actual object geometry direction)
        pca_angle = _pca_orientation(
            obj,
            walls,
            room_center=room_center,
            room_axes=room_axes,
            axis_confidence=axis_confidence,
            axis_snap_min_conf=axis_snap_min_conf,
            orient_trace=orient_trace,
        )
        if pca_angle is not None:
            logger.debug("%s at (%.2f, %.2f) → PCA orientation %.1f° (elong=%.2f)",
                         cat, ox, oz, pca_angle * 180 / pi,
                         obj.get("_pca_elongation", 0))
            return _return(pca_angle, "pca")
        pca_ambiguous = "pca_ambiguous" in orient_trace

        # Priority 2: wall inward normal (if near wall)
        # Wall-facing furniture faces away from wall = inward normal direction
        # This is more reliable than camera yaw for wall-adjacent objects
        if near_wall:
            orient_trace.append("use_wall_normal")
            wall_angle = _snap_with_trace(
                wall_normal,
                allow_axis_snap=not pca_ambiguous,
                skip_reason="pca_ambiguous",
            )
            logger.debug("%s at (%.2f, %.2f) → wall normal %.1f° (dist=%.2fm)",
                         cat, ox, oz, wall_angle * 180 / pi, wall_dist)
            return _return(wall_angle, "wall_room_axes")

        # Priority 3: camera fallback (strict last resort)
        if cam_facing is not None:
            logger.debug("%s at (%.2f, %.2f) → camera fallback %.1f°",
                         cat, ox, oz, cam_facing * 180 / pi)
            return _camera_last_resort_return()
        return _return(0.0, "default")

    # ── Anything else (unknown category) ──
    pca_angle = _pca_orientation(
        obj,
        walls,
        room_center=room_center,
        room_axes=room_axes,
        axis_confidence=axis_confidence,
        axis_snap_min_conf=axis_snap_min_conf,
        orient_trace=orient_trace,
    )
    if pca_angle is not None:
        return _return(pca_angle, "pca")
    pca_ambiguous = "pca_ambiguous" in orient_trace
    if near_wall:
        orient_trace.append("use_wall_normal")
        return _return(
            _snap_with_trace(
                wall_normal,
                allow_axis_snap=not pca_ambiguous,
                skip_reason="pca_ambiguous",
            ),
            "wall_room_axes",
        )
    if cam_facing is not None:
        return _camera_last_resort_return()
    return _return(0.0, "default")


def _nearest_wall_angle(
    ox: float, oz: float, walls: List[Dict]
) -> Optional[float]:
    """Find the angle facing away from the nearest wall."""
    if not walls:
        return None

    min_dist = float("inf")
    best_normal_angle = None

    for wall in walls:
        sx, sz = wall["start"]
        ex, ez = wall["end"]

        # Wall direction
        dx = ex - sx
        dz = ez - sz
        wall_len = sqrt(dx * dx + dz * dz)
        if wall_len < 0.01:
            continue

        # Project point onto wall line
        t = ((ox - sx) * dx + (oz - sz) * dz) / (wall_len * wall_len)
        t = max(0, min(1, t))

        # Closest point on wall
        closest_x = sx + t * dx
        closest_z = sz + t * dz

        dist = sqrt((ox - closest_x) ** 2 + (oz - closest_z) ** 2)

        if dist < min_dist:
            min_dist = dist
            # Normal pointing inward (away from wall, toward room center)
            # Wall direction is (dx, dz), so normal is (-dz, dx) or (dz, -dx)
            # Choose the one pointing toward room center (0, 0)
            n1x, n1z = -dz / wall_len, dx / wall_len
            n2x, n2z = dz / wall_len, -dx / wall_len

            # Pick normal that points more toward center
            d1 = n1x * (-closest_x) + n1z * (-closest_z)
            d2 = n2x * (-closest_x) + n2z * (-closest_z)

            if d1 > d2:
                best_normal_angle = atan2(n1x, n1z)
            else:
                best_normal_angle = atan2(n2x, n2z)

    return best_normal_angle


def _nearest_wall_info(
    ox: float, oz: float, walls: List[Dict],
    room_center: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float]]:
    """
    Find nearest wall and return (inward_normal_angle, distance_meters).

    inward_normal_angle: angle of the vector pointing from wall toward room center
                         (perpendicular to wall surface, pointing inward).
    distance_meters:     perpendicular distance from object to nearest wall.

    All angles use atan2(x, z) convention, CCW positive.
    Returns None if no walls provided.
    """
    if not walls:
        return None

    # Use room center for inward-normal disambiguation (not origin)
    rcx = room_center[0] if room_center else 0.0
    rcz = room_center[1] if room_center else 0.0

    min_dist = float("inf")
    best_normal_angle = 0.0

    for wall in walls:
        sx, sz = wall["start"]
        ex, ez = wall["end"]

        dx = ex - sx
        dz = ez - sz
        wall_len = sqrt(dx * dx + dz * dz)
        if wall_len < 0.01:
            continue

        # Project point onto wall segment
        t = ((ox - sx) * dx + (oz - sz) * dz) / (wall_len * wall_len)
        t = max(0, min(1, t))

        closest_x = sx + t * dx
        closest_z = sz + t * dz
        dist = sqrt((ox - closest_x) ** 2 + (oz - closest_z) ** 2)

        if dist < min_dist:
            min_dist = dist
            # Inward normal (toward room center)
            n1x, n1z = -dz / wall_len, dx / wall_len
            n2x, n2z = dz / wall_len, -dx / wall_len
            # Pick normal that points toward room center
            d1 = n1x * (rcx - closest_x) + n1z * (rcz - closest_z)
            d2 = n2x * (rcx - closest_x) + n2z * (rcz - closest_z)
            if d1 > d2:
                best_normal_angle = atan2(n1x, n1z)
            else:
                best_normal_angle = atan2(n2x, n2z)

    if min_dist == float("inf"):
        return None
    return (best_normal_angle, min_dist)


def _find_nearest_desk_or_table(
    chair_obj: Dict, all_objects: List[Dict], max_dist: float = CHAIR_DESK_ASSOC_DIST
) -> Optional[Dict]:
    """
    Find the nearest desk or table to a chair within max_dist (XZ plane).

    Only considers desks/tables with confidence >= CHAIR_DESK_MIN_CONF to
    prevent chairs facing misclassified objects.

    Returns the desk/table dict, or None if nothing qualifies.
    """
    cx, cz = chair_obj["x"], chair_obj["z"]
    best = None
    best_dist = max_dist
    best_context = None
    best_context_dist = max_dist

    for obj in all_objects:
        if obj is chair_obj:
            continue
        if obj["category"] not in ("desk", "table"):
            continue
        if obj.get("score", 0) < CHAIR_DESK_MIN_CONF:
            continue

        dx = obj["x"] - cx
        dz = obj["z"] - cz
        dist = sqrt(dx * dx + dz * dz)
        is_context = bool(obj.get("_rescue_context", False))
        if is_context:
            # Keep inferred context anchors as fallback only.
            if dist < best_context_dist:
                best_context_dist = dist
                best_context = obj
            continue
        if dist < best_dist:
            best_dist = dist
            best = obj

    if best is not None:
        return best
    # Fallback: use context-inferred desk/table only when reasonably close and confident.
    if best_context is not None and best_context_dist <= min(max_dist, 1.0):
        if float(best_context.get("score", 0.0)) >= 0.7:
            return best_context
    return None


def _angle_from_to(from_x: float, from_z: float, to_x: float, to_z: float) -> float:
    """Return the angle (atan2 convention, CCW positive) from one point toward another."""
    return atan2(to_x - from_x, to_z - from_z)


# ── PCA orientation (Fix C + D) ───────────────────────────────────────────

# Minimum elongation ratio (λ1/λ2) to trust PCA axis
PCA_MIN_ELONGATION = 1.6
# Minimum number of world points for PCA
# Session 12: Lowered from 50 → 15 to let PCA work with typical FP counts
PCA_MIN_POINTS = 15


def _pca_orientation(
    obj: Dict,
    walls: List[Dict],
    room_center: Optional[Tuple[float, float]] = None,
    room_axes: Optional[Tuple[float, float]] = None,
    axis_confidence: float = 0.5,
    axis_snap_min_conf: float = AXIS_SNAP_MIN_CONFIDENCE,
    orient_trace: Optional[List[str]] = None,
) -> Optional[float]:
    """
    Estimate object orientation using PCA on its multi-frame XZ point cloud.

    Returns orientation angle (CCW positive, atan2(x,z) convention) or None
    if PCA is invalid (too few points, not elongated enough).

    Stores diagnostics on obj dict:
        _pca_n_points, _pca_elongation, _pca_disambig
    """
    pts_raw = obj.get("_pca_world_points", [])
    if len(pts_raw) < PCA_MIN_POINTS:
        obj["_pca_n_points"] = len(pts_raw)
        obj["_pca_elongation"] = None
        if orient_trace is not None:
            orient_trace.append(f"pca_skip(n={len(pts_raw)})")
        return None

    pts = np.array(pts_raw)

    # Use XZ plane only (horizontal orientation — ignore height)
    xz = pts[:, [0, 2]]  # (N, 2)

    # Outlier trim: remove points > 2× median absolute deviation from centroid
    centroid = np.median(xz, axis=0)
    dists = np.linalg.norm(xz - centroid, axis=1)
    mad = float(np.median(dists))
    if mad > 0.001:
        keep = dists < 2.5 * mad
        if np.sum(keep) >= PCA_MIN_POINTS:
            xz = xz[keep]

    if len(xz) < PCA_MIN_POINTS:
        obj["_pca_n_points"] = len(xz)
        obj["_pca_elongation"] = None
        if orient_trace is not None:
            orient_trace.append(f"pca_skip_after_trim(n={len(xz)})")
        return None

    # PCA on XZ
    centroid = np.mean(xz, axis=0)
    centered = xz - centroid
    cov = np.cov(centered.T)  # 2×2
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns eigenvalues in ascending order; we want descending
    idx_sort = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    lam1 = float(eigenvalues[0])
    lam2 = float(eigenvalues[1])

    elongation = lam1 / (lam2 + 1e-8)

    obj["_pca_n_points"] = len(xz)
    obj["_pca_elongation"] = round(elongation, 2)
    if orient_trace is not None:
        orient_trace.append(f"pca_ok(n={len(xz)},elong={elongation:.2f})")

    if elongation < PCA_MIN_ELONGATION:
        if orient_trace is not None:
            orient_trace.append("pca_ambiguous")
        return None  # Too round — PCA axis is unreliable

    # Principal axis direction (in XZ plane)
    # eigenvectors[:, 0] = [vx, vz] for the largest eigenvalue
    vx = float(eigenvectors[0, 0])
    vz = float(eigenvectors[1, 0])
    axis_angle = atan2(vx, vz)  # angle of the principal axis

    def _snap_and_trace(angle: float) -> float:
        if axis_confidence < axis_snap_min_conf:
            if orient_trace is not None:
                orient_trace.append(
                    f"snap_skipped(axis_conf={axis_confidence:.2f},min={axis_snap_min_conf:.2f})"
                )
            return _normalize_angle(angle)
        snapped = _snap_to_room_axes(
            angle,
            room_axes,
            axis_confidence=max(axis_confidence, AXIS_SNAP_MIN_CONFIDENCE),
        )
        if orient_trace is not None:
            orient_trace.append(
                f"snap_to_axes(axis_conf={axis_confidence:.2f},min={axis_snap_min_conf:.2f})"
            )
        return snapped

    # ── Disambiguation: PCA gives axis (θ) not direction (θ vs θ+π) ──
    # Fix D: choose the direction using wall adjacency or camera viewpoint

    ox, oz = obj["x"], obj["z"]

    # Try wall adjacency first
    wall_info = _nearest_wall_info(ox, oz, walls, room_center=room_center)
    if wall_info is not None:
        wall_normal_angle, wall_dist = wall_info
        # "Back of object faces wall": choose direction that points AWAY from wall
        # The wall normal points inward (into the room).
        # Object should face in the same direction as wall normal (away from wall).
        # Pick θ or θ+π — whichever is closer to wall_normal_angle
        diff0 = abs(atan2(sin(axis_angle - wall_normal_angle),
                          cos(axis_angle - wall_normal_angle)))
        diff_pi = abs(atan2(sin((axis_angle + pi) - wall_normal_angle),
                            cos((axis_angle + pi) - wall_normal_angle)))
        if diff_pi < diff0:
            axis_angle = axis_angle + pi
            # Normalize to [-π, π]
            if axis_angle > pi:
                axis_angle -= 2 * pi
        obj["_pca_disambig"] = "wall"
        if orient_trace is not None:
            orient_trace.append("disambig=wall")
        return _snap_and_trace(axis_angle)

    # Camera viewpoint: object front faces away from median camera position
    cam_positions = obj.get("_pca_cam_positions", [])
    if cam_positions:
        cam_arr = np.array(cam_positions)
        median_cam_x = float(np.median(cam_arr[:, 0]))
        median_cam_z = float(np.median(cam_arr[:, 2]))

        # Direction from camera to object = "front" of object
        # (people photograph the front)
        front_angle = atan2(ox - median_cam_x, oz - median_cam_z)

        diff0 = abs(atan2(sin(axis_angle - front_angle),
                          cos(axis_angle - front_angle)))
        diff_pi = abs(atan2(sin((axis_angle + pi) - front_angle),
                            cos((axis_angle + pi) - front_angle)))
        if diff_pi < diff0:
            axis_angle = axis_angle + pi
            if axis_angle > pi:
                axis_angle -= 2 * pi
        obj["_pca_disambig"] = "camera"
        if orient_trace is not None:
            orient_trace.append("disambig=camera")
        return _snap_and_trace(axis_angle)

    # No disambiguation signal — just return the axis angle
    obj["_pca_disambig"] = "none"
    if orient_trace is not None:
        orient_trace.append("disambig=none")
    return _snap_and_trace(axis_angle)


def _viewing_angle(obj: Dict, shots: List[ShotMetadata]) -> Optional[float]:
    """Estimate object facing direction from camera viewing angle."""
    frame_idx = obj.get("frame_idx", 0)

    if frame_idx >= len(shots) or shots[frame_idx].pose is None:
        return None

    pose = shots[frame_idx].pose
    if pose is None or len(pose) != 16:
        return None

    mat = np.array(pose, dtype=np.float64).reshape(4, 4)
    cam_pos = mat[:3, 3]

    # Direction from object to camera
    ox, oy, oz = obj["x"], obj["y"], obj["z"]
    to_cam = np.array([cam_pos[0] - ox, 0, cam_pos[2] - oz])
    norm = np.linalg.norm(to_cam)
    if norm < 0.01:
        return None

    # Object faces toward the camera (typical for photos taken facing furniture)
    angle = atan2(float(to_cam[0]), float(to_cam[2]))
    return angle


def _estimate_dimensions(
    obj: Dict,
    depth_maps: List[Optional["np.ndarray"]],
    shots: List[ShotMetadata],
    cfg: PipelineConfig,
) -> Tuple[float, float, float]:
    """Estimate object dimensions from depth map and 2D bbox size."""
    cat = obj["category"]
    defaults = DEFAULT_DIMS.get(cat, (1.0, 1.0, 1.0))

    frame_idx = obj.get("frame_idx", 0)
    if frame_idx >= len(depth_maps) or depth_maps[frame_idx] is None:
        return defaults

    depth = depth_maps[frame_idx]
    median_depth = obj.get("depth_median", 2.0)

    # Get intrinsics
    intr = None
    if frame_idx < len(shots) and shots[frame_idx].intrinsics is not None:
        intr = shots[frame_idx].intrinsics
    if intr is None:
        return defaults

    img_h, img_w = depth.shape[:2]
    intr = intr.scaled_to(img_w, img_h)

    # 2D bbox size fractions
    width_2d = obj.get("width_2d", 0.1)
    height_2d = obj.get("height_2d", 0.1)

    # Convert 2D fraction to meters using depth and focal length
    # width_m ≈ width_pixels * depth / fx
    width_pixels = width_2d * img_w
    height_pixels = height_2d * img_h

    width_m = width_pixels * median_depth / intr.fx
    height_m = height_pixels * median_depth / intr.fy

    # Depth dimension (length) is harder to estimate from a single view
    # Use default proportions
    default_length, default_width, default_height = defaults

    # Scale the default proportions by the observed width
    if default_width > 0:
        scale = width_m / default_width
    else:
        scale = 1.0

    # Clamp scale to reasonable range
    scale = max(0.3, min(3.0, scale))

    est_width = max(0.1, width_m)
    est_length = max(0.1, default_length * scale)
    est_height = max(0.1, height_m)

    # Clamp to reasonable ranges per category
    est_width = max(0.1, min(5.0, est_width))
    est_length = max(0.1, min(5.0, est_length))
    est_height = max(0.05, min(3.0, est_height))

    return (est_width, est_length, est_height)


def _sanity_check_dimensions(
    obj: Dict,
    room_width: Optional[float] = None,
    room_length: Optional[float] = None,
    room_height: Optional[float] = None,
) -> None:
    """
    Fix dimension issues per category (Session 6 + Session 8).

    Session 8: Added per-category dimension clamping.
    Depth estimation is noisy and often produces wildly wrong depth/length dims.
    Instead of trusting the estimated values, clamp to sensible physical ranges
    per furniture type — real furniture has known physical constraints.
    """
    w = obj["width"]
    l = obj["length"]
    h = obj["height"]
    cat = obj["category"]

    # ── Per-category physical constraints (Session 8) ──
    # Clamp dimensions to realistic ranges for each type.
    # Format: (width_range, length_range, height_range)
    DIMENSION_RANGES = {
        "monitor":    ((0.3, 1.2), (0.05, 0.3), (0.2, 0.8)),   # thin screen
        "poster":     ((0.3, 1.5), (0.01, 0.05), (0.3, 1.5)),   # flat on wall
        "lamp_desk":  ((0.15, 0.4), (0.15, 0.4), (0.3, 0.6)),
        "lamp_floor": ((0.2, 0.5), (0.2, 0.5), (1.0, 2.0)),
        "rug":        ((1.0, 4.0), (0.8, 3.0), (0.01, 0.05)),   # flat on floor
        "nightstand": ((0.3, 0.7), (0.3, 0.6), (0.4, 0.7)),
        "chair":      ((0.4, 0.7), (0.4, 0.7), (0.7, 1.2)),
        "bed":        ((1.0, 2.2), (1.8, 2.5), (0.3, 0.8)),
        "sofa":       ((0.7, 1.2), (1.5, 3.0), (0.6, 1.0)),
        "desk":       ((0.5, 0.8), (0.8, 2.0), (0.65, 0.85)),
        "table":      ((0.6, 1.5), (0.6, 2.0), (0.65, 0.85)),
        "dresser":    ((0.4, 0.7), (0.6, 1.5), (0.7, 1.2)),
        "bookshelf":  ((0.25, 0.5), (0.6, 1.2), (1.2, 2.2)),
        "plant":      ((0.2, 0.8), (0.2, 0.8), (0.3, 1.5)),
    }

    ranges = DIMENSION_RANGES.get(cat)
    if ranges is not None:
        (w_min, w_max), (l_min, l_max), (h_min, h_max) = ranges
        obj["width"] = max(w_min, min(w_max, w))
        obj["length"] = max(l_min, min(l_max, l))
        obj["height"] = max(h_min, min(h_max, h))
        w, l, h = obj["width"], obj["length"], obj["height"]

    # ── Generic fallback: ordering check ──
    # For most furniture, height should be the smallest or mid dimension
    tall_types = {"bookshelf", "lamp_floor", "lamp_desk", "plant", "poster"}

    if cat not in tall_types:
        if h > max(w, l):
            dims = sorted([w, l, h])
            obj["width"] = dims[1]
            obj["length"] = dims[2]
            obj["height"] = dims[0]
    else:
        if h < min(w, l):
            dims = sorted([w, l, h])
            obj["width"] = dims[0]
            obj["length"] = dims[1]
            obj["height"] = dims[2]

    # ── Room-relative cap: guard against oversized objects from noisy depth ──
    if room_width is not None and room_length is not None:
        room_plan_max = max(0.8, 0.85 * max(room_width, room_length))
        obj["width"] = min(obj["width"], room_plan_max)
        obj["length"] = min(obj["length"], room_plan_max)
    if room_height is not None:
        room_h_max = max(0.2, min(3.0, 0.95 * room_height))
        obj["height"] = min(obj["height"], room_h_max)


def _snap_to_walls(
    objects: List[Dict],
    walls: List[Dict],
    room_width: float,
    room_length: float,
    snap_distance: float = WALL_PROXIMITY_THRESHOLD,
    room_center: Optional[Tuple[float, float]] = None,
) -> None:
    """Snap wall-adjacent furniture to walls conservatively."""
    wall_types = {"dresser", "bookshelf", "nightstand", "poster", "monitor",
                  "bed", "desk", "sofa"}

    rcx = room_center[0] if room_center else 0.0
    rcz = room_center[1] if room_center else 0.0

    for obj in objects:
        if obj["category"] not in wall_types:
            continue

        cat = obj["category"]
        local_snap_distance = WALL_SNAP_DIST_BY_CAT.get(cat, snap_distance)
        ox, oz = obj["x"], obj["z"]
        min_dist = float("inf")
        snap_x, snap_z = ox, oz

        for wall in walls:
            sx, sz = wall["start"]
            ex, ez = wall["end"]

            dx = ex - sx
            dz = ez - sz
            wall_len = sqrt(dx * dx + dz * dz)
            if wall_len < 0.01:
                continue

            # Project point onto wall line
            t = ((ox - sx) * dx + (oz - sz) * dz) / (wall_len * wall_len)
            t = max(0, min(1, t))

            closest_x = sx + t * dx
            closest_z = sz + t * dz
            dist = sqrt((ox - closest_x) ** 2 + (oz - closest_z) ** 2)

            if dist < min_dist:
                min_dist = dist

                # Normal direction (inward — toward room center)
                nx = -dz / wall_len
                nz = dx / wall_len
                # Check if normal points toward room center
                d1 = nx * (rcx - closest_x) + nz * (rcz - closest_z)
                if d1 < 0:
                    nx, nz = -nx, -nz

                # Target center = wall + bounded inward offset.
                obj_depth = obj.get("length", 0.5) / 2
                target_offset = min(max(0.12, obj_depth), WALL_SNAP_OFFSET_CAP_M)
                snap_x = closest_x + nx * target_offset
                snap_z = closest_z + nz * target_offset

        # If FP-based placement is already strong, don't overwrite it with wall snap.
        src = obj.get("position_source", "depth_backproject")
        fp_count = int(obj.get("fp_count", 0))
        fp_var = float(obj.get("fp_variance", 999.0))
        fp_frames_hit = int(obj.get("frames_with_fp_hit", 0))
        trusted_fp = (
            src == "feature_points"
            and fp_count >= 8
            and fp_var <= 0.25
            and fp_frames_hit >= 2
        )

        if min_dist < local_snap_distance and not trusted_fp:
            shift_dx = snap_x - ox
            shift_dz = snap_z - oz
            shift = sqrt(shift_dx * shift_dx + shift_dz * shift_dz)
            if shift > WALL_SNAP_MAX_SHIFT_M and shift > 1e-6:
                s = WALL_SNAP_MAX_SHIFT_M / shift
                snap_x = ox + shift_dx * s
                snap_z = oz + shift_dz * s

            # Soft snap instead of teleport.
            snap_x = ox + WALL_SNAP_BLEND_ALPHA * (snap_x - ox)
            snap_z = oz + WALL_SNAP_BLEND_ALPHA * (snap_z - oz)
            obj["_wall_snap_applied"] = True
            obj["_wall_snap_dist_before"] = min_dist
            obj["_wall_snap_max_dist"] = local_snap_distance
            obj["_wall_snap_shift"] = round(sqrt((snap_x - ox) ** 2 + (snap_z - oz) ** 2), 3)
            logger.debug("Snap %s from (%.3f, %.3f) to (%.3f, %.3f), dist=%.3f",
                         obj["category"], ox, oz, snap_x, snap_z, min_dist)
            obj["x"] = snap_x
            obj["z"] = snap_z
        else:
            obj["_wall_snap_applied"] = False
            obj["_wall_snap_dist_before"] = min_dist
            obj["_wall_snap_max_dist"] = local_snap_distance
            obj["_wall_snap_shift"] = 0.0
            if trusted_fp:
                obj["_wall_snap_skip_reason"] = "trusted_feature_points"


def _drop_depth_outliers_before_clamp(
    objects: List[Dict],
    room_width: float,
    room_length: float,
    room_center: Optional[Tuple[float, float]] = None,
    depth_reliability: float = 1.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Drop extreme depth-only placements before room clamp.

    Without this, a depth-backproject object that lands far outside the room is
    pulled to the nearest boundary by clamp, creating a plausible-looking but
    meaningless placement.
    """
    if not objects:
        return [], []

    hw = room_width / 2.0
    hl = room_length / 2.0
    rcx = room_center[0] if room_center else 0.0
    rcz = room_center[1] if room_center else 0.0

    reliability = max(0.0, min(1.0, float(depth_reliability)))
    outside_margin = 0.35 + (1.0 - reliability) * 0.45  # 0.35m .. 0.80m

    kept: List[Dict] = []
    dropped: List[Dict] = []

    for obj in objects:
        src = obj.get("position_source", "depth_backproject")
        if src != "depth_backproject":
            kept.append(obj)
            continue

        ox = float(obj.get("x", 0.0))
        oz = float(obj.get("z", 0.0))
        outside_x = max(0.0, abs(ox - rcx) - hw)
        outside_z = max(0.0, abs(oz - rcz) - hl)
        outside_dist = sqrt(outside_x * outside_x + outside_z * outside_z)

        if outside_dist > outside_margin:
            obj["_dropped_reason"] = "depth_outlier_before_clamp"
            obj["_dropped_outside_dist"] = round(outside_dist, 4)
            obj["_dropped_outside_margin"] = round(outside_margin, 4)
            dropped.append(obj)
            logger.info(
                "  DROPPED (depth outlier): %s score=%.2f src=%s outside=%.2fm margin=%.2fm at (%.2f, %.2f)",
                obj.get("category", "?"),
                obj.get("score", 0.0),
                src,
                outside_dist,
                outside_margin,
                ox,
                oz,
            )
        else:
            kept.append(obj)

    # Safety valve: if nearly everything is outside, layout is likely wrong.
    # In that case keep originals and rely on downstream diagnostics instead.
    if dropped and len(dropped) >= max(2, int(0.7 * len(objects))) and len(kept) == 0:
        logger.warning(
            "Depth outlier filter skipped: dropped %d/%d objects (possible layout mismatch)",
            len(dropped),
            len(objects),
        )
        return objects, []

    return kept, dropped


def _clamp_to_room(
    objects: List[Dict],
    room_width: float,
    room_length: float,
    room_height: float,
    room_center: Optional[Tuple[float, float]] = None,
) -> None:
    """Ensure objects are within room bounds.

    Session 12: Fixed to use room_center offset instead of clamping around origin.
    """
    hw = room_width / 2
    hl = room_length / 2
    rcx = room_center[0] if room_center else 0.0
    rcz = room_center[1] if room_center else 0.0

    for obj in objects:
        # Clamp X and Z to room bounds (with some margin for wall-adjacent items)
        margin = 0.1
        obj["x"] = max(rcx - hw + margin, min(rcx + hw - margin, obj["x"]))
        obj["z"] = max(rcz - hl + margin, min(rcz + hl - margin, obj["z"]))

        # Y: place objects on the floor (y=0) unless they're wall-mounted
        cat = obj["category"]
        if cat == "poster":
            # Wall-mounted: place at eye height
            obj["y"] = room_height * 0.55
        elif cat == "monitor":
            # On desk height
            obj["y"] = 0.75
        elif cat == "lamp_floor":
            obj["y"] = 0.0
        elif cat == "lamp_desk":
            obj["y"] = 0.75  # On desk/nightstand
        elif cat == "rug":
            obj["y"] = 0.0
        else:
            # Ground level
            obj["y"] = 0.0


def _build_room_objects(objects: List[Dict]) -> List[Dict]:
    """Convert internal object dicts to RoomObject dicts for the payload."""
    result = []

    for idx, obj in enumerate(objects):
        cat = obj["category"]

        # Map category to asset_id
        asset_id = f"kenney_{cat}"

        metadata = {
            "width": round(obj["width"], 3),
            "length": round(obj["length"], 3),
            "height": round(obj["height"], 3),
            "inferred": bool(
                str(obj.get("position_source", "")) == "context_inference"
                or bool(obj.get("_rescue_context", False))
            ),
        }
        if obj.get("position_uncertainty_m") is not None:
            metadata["position_uncertainty_m"] = round(float(obj.get("position_uncertainty_m", 0.0)), 4)
        if obj.get("_orientation_uncertainty_deg") is not None:
            metadata["orientation_uncertainty_deg"] = round(
                float(obj.get("_orientation_uncertainty_deg", 0.0)), 2
            )
        if obj.get("support_views") is not None:
            metadata["support_views"] = int(obj.get("support_views", 0))
        if obj.get("_suppressed_reason") is not None:
            metadata["suppressed_reason"] = str(obj.get("_suppressed_reason"))

        room_obj = {
            "id": f"{cat}_{idx}",
            "type": cat,
            "asset_id": asset_id,
            "position": {
                "x": round(obj["x"], 3),
                "y": round(obj["y"], 3),
                "z": round(obj["z"], 3),
            },
            "rotation_y": round(obj["rotation_y"], 4),
            "scale": 1.0,
            "confidence": round(obj["score"], 3),
            "metadata": metadata,
            # Internal diagnostics for downstream filtering (not part of schema).
            "_position_source": obj.get("position_source", "unknown"),
            "_fp_count": obj.get("fp_count", 0),
            "_support_views": int(obj.get("support_views", 0)),
        }
        result.append(room_obj)

    return result


def _debug_output(objects: List[Dict]) -> None:
    """Write cluster report for debugging (includes scale + orientation + placement diagnostics)."""
    try:
        import json
        from .pipeline import get_scale_diagnostics

        diag = get_scale_diagnostics()

        # ── Quick-glance triage summary ──────────────────────────────
        red_flags = []
        n_fp_positioned = sum(1 for o in objects if o.get("position_source") == "feature_points")
        n_depth_positioned = sum(1 for o in objects if o.get("position_source") == "depth_backproject")
        n_triangulated_positioned = sum(
            1 for o in objects if o.get("position_source") == "triangulated_rays"
        )
        n_context_positioned = sum(
            1 for o in objects if o.get("position_source") == "context_inference"
        )
        n_total = len(objects)

        if n_depth_positioned > n_fp_positioned and n_total > 0:
            red_flags.append(f"DEPTH_DOMINANT({n_depth_positioned}/{n_total} objects from depth)")

        uv_pts = diag.get("scale_projection_uv_points", 0)
        legacy_pts = diag.get("scale_projection_legacy_points", 0)
        scale_method = str(diag.get("scale_method", "unknown"))
        is_metric_path = (
            scale_method == "metric_direct"
            or str(diag.get("depth_model_type", "")) == "dav2_metric"
        )

        if not is_metric_path:
            if legacy_pts > 0 and uv_pts == 0:
                red_flags.append("SCALE_ALL_LEGACY(client UVs not used for depth scaling)")
            elif uv_pts == 0 and legacy_pts == 0:
                red_flags.append("SCALE_NO_PROJECTION(no FP projected at all)")

            if scale_method in ("none", "feature_points_rejected"):
                red_flags.append(
                    f"SCALE_FAILED(method={scale_method}, "
                    f"reason={diag.get('scale_fallback_reason', '?')[:50]})"
                )
        else:
            metric_rel = diag.get("metric_depth_reliability")
            if metric_rel is not None and float(metric_rel) < 0.45:
                red_flags.append(f"METRIC_DEPTH_LOW_RELIABILITY({float(metric_rel):.2f})")

        for obj in objects:
            cam_dist = None
            cam_positions_for_obj = obj.get("_pca_cam_positions", [])
            if cam_positions_for_obj:
                ox, oz = obj["x"], obj["z"]
                dists = [sqrt((cp[0] - ox) ** 2 + (cp[2] - oz) ** 2)
                         for cp in cam_positions_for_obj if len(cp) >= 3]
                if dists:
                    cam_dist = min(dists)
            if cam_dist is not None and cam_dist > 8.0:
                red_flags.append(f"FAR_OBJECT({obj['category']} at {cam_dist:.1f}m from camera)")

            chain = obj.get("_pos_chain", [])
            if len(chain) >= 2:
                first = chain[0]
                last = chain[-1]
                drift = sqrt((first["x"] - last["x"]) ** 2 + (first["z"] - last["z"]) ** 2)
                if drift > 1.5:
                    red_flags.append(f"LARGE_POS_DRIFT({obj['category']} moved {drift:.1f}m through pipeline)")

        report_data = {
            # Session 14: Quick-glance triage
            "red_flags": red_flags if red_flags else ["NONE"],
            "summary": {
                "total_objects": n_total,
                "fp_positioned": n_fp_positioned,
                "depth_positioned": n_depth_positioned,
                "triangulated_positioned": n_triangulated_positioned,
                "context_positioned": n_context_positioned,
                "scale_uv_points": uv_pts,
                "scale_legacy_points": legacy_pts,
                "scale_method": diag.get("scale_method", "unknown"),
                "scale_factor": diag.get("scale_factor"),
                "same_frame_merge_events": int(len(_cluster_same_frame_merge_events)),
                "dropped_depth_outliers_total": int(
                    _lift_diagnostics.get("dropped_depth_outliers_total", 0)
                ),
                "rescued_context_total": int(
                    _lift_diagnostics.get("rescued_context_total", 0)
                ),
                "suppressed_inferred_objects_total": int(
                    _lift_diagnostics.get("suppressed_inferred_objects_total", 0)
                ),
            },
            "clustering": {
                "same_frame_merge_events": _cluster_same_frame_merge_events,
            },
            # Session 11: Feature-point-anchored diagnostics
            "geometry_source": "feature_points" if diag.get("depth_role") == "refinement_only" else "depth_primary",
            "depth_role": diag.get("depth_role", "unknown"),
            "scale": diag,
            "dropped_depth_outliers": _lift_diagnostics.get("dropped_depth_outliers", []),
            "rescued_context": _lift_diagnostics.get("rescued_context", []),
            "suppressed_inferred_objects": _lift_diagnostics.get("suppressed_inferred_objects", []),
            "objects": [],
        }
        for obj in objects:
            # Compute nearest camera distance for sanity check
            nearest_cam_dist = None
            cam_positions_for_obj = obj.get("_pca_cam_positions", [])
            if cam_positions_for_obj:
                ox, oz = obj["x"], obj["z"]
                dists = [
                    sqrt((cp[0] - ox) ** 2 + (cp[2] - oz) ** 2)
                    for cp in cam_positions_for_obj if len(cp) >= 3
                ]
                if dists:
                    nearest_cam_dist = round(min(dists), 3)

            entry = {
                "category": obj["category"],
                "position": {"x": round(obj["x"], 3), "y": round(obj["y"], 3), "z": round(obj["z"], 3)},
                "rotation_y": round(obj["rotation_y"], 4),
                "rotation_y_deg": round(obj["rotation_y"] * 180.0 / pi, 1),
                "dims": {"w": round(obj["width"], 3), "l": round(obj["length"], 3), "h": round(obj["height"], 3)},
                "score": round(obj["score"], 3),
                "n_detections": obj.get("n_detections", 1),
                "obs_required": obj.get("_obs_required"),
                "obs_rule": obj.get("_obs_rule"),
                # Feature-point position diagnostics (Session 11)
                "position_source": obj.get("position_source", "unknown"),
                "fp_count": obj.get("fp_count", 0),
                "fp_variance": obj.get("fp_variance", 0),
                "fp_temporal_mean_weight": obj.get("fp_temporal_mean_weight", 0),
                "fp_in_bbox_ratio_mean": obj.get("fp_in_bbox_ratio_mean", 0),
                "fp_in_bbox_ratio_max": obj.get("fp_in_bbox_ratio_max", 0),
                "frames_with_fp_hit": obj.get("frames_with_fp_hit", 0),
                "frames_total": obj.get("frames_total", 0),
                "support_views": obj.get("support_views", 0),
                "position_uncertainty_m": obj.get("position_uncertainty_m"),
                "orientation_uncertainty_deg": obj.get("_orientation_uncertainty_deg"),
                # Session 14: Position transform chain (track exact movement)
                "position_chain": obj.get("_pos_chain", []),
                # Session 14: Nearest camera distance (sanity: should be <5m)
                "nearest_cam_dist_xz": nearest_cam_dist,
                # Session 14: Axis confidence used for this object's orientation
                "orient_axis_confidence": obj.get("_orient_axis_confidence"),
                # Orientation diagnostics
                "orient_source": obj.get("_orient_source"),
                "orient_trace": obj.get("_orient_trace", []),
                "orient_wall_dist": obj.get("_orient_wall_dist"),
                "orient_wall_normal_deg": round(obj["_orient_wall_normal"] * 180.0 / pi, 1)
                    if obj.get("_orient_wall_normal") is not None else None,
                "orient_median_cam_yaw_deg": round(obj["_orient_median_cam_yaw"] * 180.0 / pi, 1)
                    if obj.get("_orient_median_cam_yaw") is not None else None,
                # PCA diagnostics
                "pca_n_points": obj.get("_pca_n_points"),
                "pca_elongation": obj.get("_pca_elongation"),
                "pca_disambig": obj.get("_pca_disambig"),
                # Wall snap diagnostics
                "wall_snap_applied": obj.get("_wall_snap_applied", False),
                "wall_snap_dist_before": round(obj["_wall_snap_dist_before"], 3)
                    if obj.get("_wall_snap_dist_before") is not None else None,
                "wall_snap_max_dist": obj.get("_wall_snap_max_dist"),
                "wall_snap_shift": obj.get("_wall_snap_shift"),
                "wall_snap_skip_reason": obj.get("_wall_snap_skip_reason"),
                # Per-frame FP breakdown (Session 12, enriched Session 13)
                "per_frame_fp": obj.get("_per_frame_fp", []),
                # Session 13: Per-detection diagnostics (Q1: what was detected?)
                "detections": obj.get("_detection_details", []),
                # Session 13: Depth diagnostics (Q2: how was 3D position computed?)
                "depth_median": obj.get("depth_median"),
                "depth_mad": obj.get("depth_mad"),
                "depth_sample_count": obj.get("depth_sample_count"),
                "rescue_context": bool(obj.get("_rescue_context", False)),
                "rescue_reason": obj.get("_rescue_reason"),
                # Session 13: Tracking quality summary (Q3: was SLAM reliable?)
                "tracking_quality_per_frame": [
                    d.get("tracking_quality", "unknown")
                    for d in obj.get("_detection_details", [])
                ],
            }
            # Chair-specific: desk target
            if obj.get("_orient_desk_target_id") is not None:
                entry["orient_desk_target_id"] = obj["_orient_desk_target_id"]
            report_data["objects"].append(entry)
        with open("/tmp/decroom_cluster_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
    except Exception as e:
        logger.warning("Failed to write cluster report: %s", e)
