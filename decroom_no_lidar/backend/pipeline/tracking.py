"""
3D-aware object tracking across frames.
Groups detections of the same object across multiple frames into tracks.

Session 9: Upgraded from pure 2D IoU matching to 3D world-space matching.
When depth maps + camera poses are available, each detection is lifted to an
approximate 3D position. Matching is then based on world-space Euclidean
distance, which works even when the camera rotates 360° (same object at
completely different image positions). Falls back to 2D IoU when 3D data
is unavailable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .detection import Detection
from .geometry import associate_fp_to_detection, robust_lift_detection

logger = logging.getLogger("decroom-backend.tracking")

_tracking_diagnostics: Dict = {}


def get_tracking_diagnostics() -> Dict:
    """Return tracking-level diagnostics for reconstruction reporting."""
    return dict(_tracking_diagnostics)


@dataclass
class Track:
    """A tracked object across multiple frames."""
    track_id: int
    category: str
    detections: List[Detection] = field(default_factory=list)
    last_frame_idx: int = -1
    world_pos: Optional[Tuple[float, float, float]] = None  # approximate 3D

    @property
    def best_detection(self) -> Detection:
        return max(self.detections, key=lambda d: d.score)

    @property
    def avg_score(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.score for d in self.detections) / len(self.detections)


# ─── 3D position estimation for a single detection ───────────────────

def _detection_to_3d(
    det: Detection,
    depth_maps: List,
    shots: List,
) -> Optional[Tuple[float, float, float]]:
    """
    Compute approximate world-space position for a detection.

    Session 11: Feature-point-first.
      Primary: ARKit feature point centroid (metric)
      Fallback: Depth back-projection (legacy)

    Returns (x, y, z) in world space, or None if data is missing.
    """
    if np is None:
        return None

    frame_idx = det.frame_idx
    if frame_idx >= len(shots):
        return None

    shot = shots[frame_idx]

    # ── Primary: Feature-point centroid ──
    if shot.pose is not None and shot.intrinsics is not None:
        fp_points = associate_fp_to_detection(det, shot)
        if len(fp_points) >= 3:
            centroid = np.median(fp_points, axis=0)
            return (float(centroid[0]), float(centroid[1]), float(centroid[2]))

    # ── Fallback: Depth back-projection ──
    if frame_idx >= len(depth_maps) or depth_maps[frame_idx] is None:
        return None

    result = robust_lift_detection(det, depth_maps[frame_idx], shot)
    if result is None:
        return None

    return result.world_pos


def _world_dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Euclidean distance between two 3D points."""
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


# ─── 2D matching helpers ─────────────────────────────────────────────

def _iou(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _bbox_center_dist(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    """Normalized center distance between two bboxes (0 = same center, 1 = far)."""
    cx_a = (box_a[0] + box_a[2]) / 2
    cy_a = (box_a[1] + box_a[3]) / 2
    cx_b = (box_b[0] + box_b[2]) / 2
    cy_b = (box_b[1] + box_b[3]) / 2

    diag_a = ((box_a[2] - box_a[0]) ** 2 + (box_a[3] - box_a[1]) ** 2) ** 0.5
    diag_b = ((box_b[2] - box_b[0]) ** 2 + (box_b[3] - box_b[1]) ** 2) ** 0.5
    diag = max(diag_a, diag_b, 1.0)

    dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
    return dist / diag


def _bbox_area_ratio(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    """Smaller/larger bbox area ratio in [0, 1]."""
    wa = max(1e-6, box_a[2] - box_a[0])
    ha = max(1e-6, box_a[3] - box_a[1])
    wb = max(1e-6, box_b[2] - box_b[0])
    hb = max(1e-6, box_b[3] - box_b[1])
    area_a = wa * ha
    area_b = wb * hb
    lo = min(area_a, area_b)
    hi = max(area_a, area_b, 1e-6)
    return float(lo / hi)


def _frame_to_rgb_array(frame) -> Optional["np.ndarray"]:
    """Convert a frame input (UploadFile/path/ndarray) to RGB uint8 array."""
    if np is None:
        return None
    try:
        from PIL import Image
    except Exception:
        Image = None  # type: ignore
    try:
        import cv2
    except Exception:
        cv2 = None  # type: ignore

    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr

    if isinstance(frame, str):
        if cv2 is None:
            return None
        img = cv2.imread(frame, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if hasattr(frame, "file"):
        try:
            f = frame.file
            pos = f.tell()
            f.seek(0)
            data = f.read()
            f.seek(pos)
            if Image is not None:
                import io
                return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            if cv2 is not None:
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
    return None


def _detection_appearance_descriptor(
    det: Detection,
    frame_rgb: Optional["np.ndarray"],
) -> Optional["np.ndarray"]:
    """Compact descriptor from bbox color histogram + shape stats."""
    if np is None or frame_rgb is None:
        return None
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
        return None

    ih, iw = frame_rgb.shape[:2]
    x1, y1, x2, y2 = det.bbox
    det_w, det_h = det.input_size
    if det_w <= 0 or det_h <= 0:
        return None
    sx = iw / float(det_w)
    sy = ih / float(det_h)
    bx1 = int(max(0, min(iw - 1, x1 * sx)))
    by1 = int(max(0, min(ih - 1, y1 * sy)))
    bx2 = int(max(bx1 + 1, min(iw, x2 * sx)))
    by2 = int(max(by1 + 1, min(ih, y2 * sy)))
    roi = frame_rgb[by1:by2, bx1:bx2]
    if roi.size == 0:
        return None

    pix = roi.reshape(-1, 3).astype(np.float32) / 255.0
    hist_r, _ = np.histogram(pix[:, 0], bins=8, range=(0.0, 1.0))
    hist_g, _ = np.histogram(pix[:, 1], bins=8, range=(0.0, 1.0))
    hist_b, _ = np.histogram(pix[:, 2], bins=8, range=(0.0, 1.0))
    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
    total = float(np.sum(hist))
    if total > 1e-6:
        hist /= total
    aspect = float((bx2 - bx1) / max(1.0, float(by2 - by1)))
    area = float((bx2 - bx1) * (by2 - by1) / max(1.0, float(iw * ih)))
    return np.concatenate([hist, np.array([aspect, area], dtype=np.float32)], axis=0)


def _appearance_similarity(
    a: Optional["np.ndarray"],
    b: Optional["np.ndarray"],
) -> float:
    """Similarity in [0,1] from descriptor distance."""
    if np is None or a is None or b is None:
        return 0.0
    if len(a) != len(b):
        return 0.0
    # Histogram part: Bhattacharyya-like score.
    hlen = max(0, len(a) - 2)
    if hlen > 0:
        h1 = np.clip(a[:hlen], 0.0, 1.0)
        h2 = np.clip(b[:hlen], 0.0, 1.0)
        bc = float(np.sum(np.sqrt(h1 * h2)))
    else:
        bc = 0.0
    # Shape part.
    da = float(abs(float(a[-2]) - float(b[-2])))
    db = float(abs(float(a[-1]) - float(b[-1])))
    shape_sim = max(0.0, 1.0 - min(1.0, 0.35 * da + 2.5 * db))
    return max(0.0, min(1.0, 0.7 * bc + 0.3 * shape_sim))


# ─── Accumulated centroid update (Session 11, Correction 5) ──────────

# Cache for associate_fp_to_detection results to avoid recomputation
_fp_cache: Dict[Tuple[int, int], Optional["np.ndarray"]] = {}


def _update_track_pos_from_fp(
    track: Track,
    shots: List,
    det_3d: Dict,
) -> None:
    """
    Update track.world_pos using accumulated feature-point centroid
    across ALL detections in the track (Correction 5).

    Falls back to running average of det_3d positions if FP insufficient.
    """
    if np is None:
        return

    all_fp = []
    for d in track.detections:
        fi = d.frame_idx
        if fi >= len(shots):
            continue
        shot = shots[fi]
        if shot.pose is None or shot.intrinsics is None:
            continue

        # Cache lookup
        cache_key = (fi, id(d))
        if cache_key in _fp_cache:
            cached = _fp_cache[cache_key]
            if cached is not None and len(cached) > 0:
                all_fp.append(cached)
            continue

        fp_points = associate_fp_to_detection(d, shot)
        _fp_cache[cache_key] = fp_points if len(fp_points) > 0 else None
        if len(fp_points) > 0:
            all_fp.append(fp_points)

    if all_fp:
        combined = np.vstack(all_fp)
        if len(combined) >= 3:
            centroid = np.median(combined, axis=0)
            track.world_pos = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
            return

    # Fallback: use latest det_3d position
    latest_det = track.detections[-1]
    det_pos = det_3d.get(id(latest_det))
    if det_pos is not None:
        old_pos = track.world_pos
        if old_pos is not None:
            n = len(track.detections)
            track.world_pos = (
                old_pos[0] + (det_pos[0] - old_pos[0]) / n,
                old_pos[1] + (det_pos[1] - old_pos[1]) / n,
                old_pos[2] + (det_pos[2] - old_pos[2]) / n,
            )
        else:
            track.world_pos = det_pos


# ─── Main tracking function ──────────────────────────────────────────

# Per-category 3D merge thresholds (meters). Objects within this distance
# in world space are considered the same physical object.
_WORLD_DIST_BY_CAT: Dict[str, float] = {
    "bed": 1.6,
    "sofa": 1.4,
    "desk": 1.1,
    "dresser": 1.0,
    "bookshelf": 1.0,
    "table": 1.1,
    "monitor": 0.8,
    "chair": 0.75,
    "nightstand": 0.8,
    "lamp_floor": 0.8,
    "lamp_desk": 0.65,
    "plant": 0.75,
    "poster": 0.9,
    "rug": 1.4,
}

# Tightened 2D fallback gates to avoid wrong merges when 3D match fails.
_IOU_FALLBACK_MIN_BY_CAT: Dict[str, float] = {
    "monitor": 0.08,
    "chair": 0.10,
    "dresser": 0.12,
    "bookshelf": 0.12,
    "bed": 0.10,
    "sofa": 0.10,
    "desk": 0.10,
    "table": 0.10,
}
_CENTER_DIST_FALLBACK_MAX_BY_CAT: Dict[str, float] = {
    "monitor": 0.40,
    "chair": 0.35,
    "dresser": 0.35,
    "bookshelf": 0.35,
    "bed": 0.45,
    "sofa": 0.45,
    "desk": 0.40,
    "table": 0.40,
}
_AREA_RATIO_MIN_STRICT = 0.35
_AREA_RATIO_MIN_LOOSE = 0.20
_MAX_2D_FALLBACK_FRAME_GAP_STRICT = 4   # was 2
_MAX_2D_FALLBACK_FRAME_GAP_LOOSE = 8    # was 4


def track_detections(
    detections: List[Detection],
    frames: Optional[List] = None,
    depth_maps: Optional[List] = None,
    shots: Optional[List] = None,
    iou_threshold: float = 0.2,
    center_dist_threshold: float = 0.5,
    world_dist_default: float = 1.2,
    max_age: int = 30,
    enable_tracklet_stitching: bool = True,
    stitch_max_gap_frames: int = 8,
    stitch_world_dist_factor: float = 1.35,
    stitch_min_score: float = 0.25,
    stitch_max_passes: int = 6,
    enable_track_duplicate_merge: bool = True,
    duplicate_iou_min: float = 0.22,
    duplicate_center_max: float = 0.5,
    duplicate_world_dist_factor: float = 0.9,
    duplicate_min_score: float = 0.2,
    duplicate_max_passes: int = 8,
    appearance_weight: float = 0.25,
    high_similarity_relaxation: float = 1.18,
    fragmentation_alert_threshold: float = 0.65,
    new_track_min_score_by_cat: Optional[Dict[str, float]] = None,
    sfm_registration_ratio: float = 0.0,
    sfm_alignment_rmse_m: Optional[float] = None,
    monitor_weak_sfm_relaxation: float = 1.25,
    enable_singleton_merge: bool = True,
    singleton_merge_world_dist_factor: float = 1.5,
    singleton_merge_min_appearance: float = 0.70,
    singleton_merge_high_conf_threshold: float = 0.80,
    stitch_2d_fallback_max_gap: int = 6,
) -> List[Track]:
    """
    Track detections across frames using 3D world-space matching.

    Primary matching (when depth + pose available):
      Lift each detection to an approximate 3D world position. Match to the
      nearest existing track of the same category within a distance threshold.
      This works even when the camera rotates 360° because the 3D position
      of a stationary object doesn't change.

    Fallback matching (no depth/pose):
      Use 2D IoU + center-distance on the track's best-scoring detection.

    Args:
        detections: All detections from all frames
        depth_maps: Per-frame depth maps (scaled to meters)
        shots: Per-frame ShotMetadata with pose + intrinsics
        iou_threshold: Minimum IoU for 2D fallback matching
        center_dist_threshold: Max normalized center distance for 2D fallback
        world_dist_default: Default 3D matching threshold in meters
        max_age: Max frame gap before retiring a track

    Returns:
        List of Track objects
    """
    global _tracking_diagnostics

    if not detections:
        _tracking_diagnostics = {
            "fragmentation_rate_by_category": {},
            "fragmentation_alerts": [],
            "n_single_frame_tracks": 0,
            "n_tracks": 0,
        }
        _save_tracking_report([], [], [], [], [])
        return []

    # Clear FP association cache for this run
    global _fp_cache
    _fp_cache = {}

    has_3d = depth_maps is not None and shots is not None
    appearance_weight = max(0.0, min(1.0, float(appearance_weight)))
    high_similarity_relaxation = max(1.0, float(high_similarity_relaxation))

    # Sort by frame index
    sorted_dets = sorted(detections, key=lambda d: (d.frame_idx, -d.score))

    # Group by frame
    frames_map: Dict[int, List[Detection]] = {}
    for det in sorted_dets:
        frames_map.setdefault(det.frame_idx, []).append(det)

    frame_indices = sorted(frames_map.keys())

    # Precompute 3D positions for all detections
    det_3d: Dict[int, Optional[Tuple[float, float, float]]] = {}
    det_app: Dict[int, Optional["np.ndarray"]] = {}
    frame_cache: Dict[int, Optional["np.ndarray"]] = {}
    if frames is not None and np is not None:
        for det in sorted_dets:
            fi = det.frame_idx
            if fi not in frame_cache:
                if fi < len(frames):
                    frame_cache[fi] = _frame_to_rgb_array(frames[fi])
                else:
                    frame_cache[fi] = None
            det_app[id(det)] = _detection_appearance_descriptor(det, frame_cache.get(fi))

    if has_3d:
        n_with_3d = 0
        for det in sorted_dets:
            key = id(det)
            pos = _detection_to_3d(det, depth_maps, shots)
            det_3d[key] = pos
            if pos is not None:
                n_with_3d += 1
        logger.info("3D positions computed: %d/%d detections have 3D",
                     n_with_3d, len(sorted_dets))
    else:
        logger.info("No depth/pose data — using 2D-only tracking")

    active_tracks: List[Track] = []
    finished_tracks: List[Track] = []
    assignment_events: List[Dict] = []
    rejection_events: List[Dict] = []
    next_id = 0

    for frame_idx in frame_indices:
        frame_dets = frames_map[frame_idx]

        matched_track_ids = set()
        matched_det_ids = set()

        # Build match candidates: (score, track_idx, det_idx)
        matches: List[Tuple[float, int, int]] = []
        match_meta: Dict[Tuple[int, int], Dict] = {}
        det_candidate_counts: Dict[int, int] = {di: 0 for di in range(len(frame_dets))}

        for ti, track in enumerate(active_tracks):
            if track.category == "":
                continue
            for di, det in enumerate(frame_dets):
                if det.category != track.category:
                    continue

                # ── 3D matching (preferred) ──
                det_pos = det_3d.get(id(det))
                had_3d_pair = False
                app_sim = _appearance_similarity(
                    det_app.get(id(det)),
                    det_app.get(id(track.best_detection)),
                )
                if det_pos is not None and track.world_pos is not None:
                    had_3d_pair = True
                    dist_3d = _world_dist(det_pos, track.world_pos)
                    threshold = _WORLD_DIST_BY_CAT.get(track.category, world_dist_default)
                    weak_sfm = (
                        float(sfm_registration_ratio) < 0.5
                        or (
                            sfm_alignment_rmse_m is not None
                            and float(sfm_alignment_rmse_m) > 0.35
                        )
                    )
                    if weak_sfm and track.category == "monitor":
                        threshold *= max(1.0, float(monitor_weak_sfm_relaxation))
                    threshold_adj = threshold * (
                        high_similarity_relaxation if app_sim >= 0.82 else 1.0
                    )
                    if dist_3d <= threshold_adj:
                        # Score: closer = better. Normalize to 0-2 range.
                        match_score = (
                            2.0 * max(0, 1.0 - dist_3d / max(1e-6, threshold_adj))
                            + appearance_weight * app_sim
                        )
                        matches.append((match_score, ti, di))
                        det_candidate_counts[di] = det_candidate_counts.get(di, 0) + 1
                        match_meta[(ti, di)] = {
                            "reason": "3d_distance",
                            "distance_m": float(dist_3d),
                            "iou": None,
                            "threshold_m": float(threshold_adj),
                            "center_dist": None,
                            "area_ratio": None,
                            "frame_gap": int(frame_idx - track.last_frame_idx),
                            "appearance_similarity": round(float(app_sim), 4),
                            "threshold_relaxed": bool(threshold_adj > threshold + 1e-6),
                        }
                        continue
                    # 3D exists but did not pass threshold.
                    # Fall back to tightly-gated 2D matching to reduce false merges.

                # ── 2D fallback matching ──
                ref_det = track.best_detection
                if ref_det.frame_idx == frame_idx:
                    continue

                iou_score = _iou(ref_det.bbox, det.bbox)
                center_dist = _bbox_center_dist(ref_det.bbox, det.bbox)
                area_ratio = _bbox_area_ratio(ref_det.bbox, det.bbox)
                frame_gap = int(frame_idx - track.last_frame_idx)

                cat_iou_min = _IOU_FALLBACK_MIN_BY_CAT.get(track.category, iou_threshold)
                cat_center_max = _CENTER_DIST_FALLBACK_MAX_BY_CAT.get(track.category, center_dist_threshold)

                if had_3d_pair:
                    strict_gap_limit = _MAX_2D_FALLBACK_FRAME_GAP_STRICT
                    if app_sim >= 0.82:
                        strict_gap_limit += 1
                    if app_sim >= 0.82 and track.category == "monitor":
                        strict_gap_limit += 1
                    # Tighten gap for low-similarity matches to prevent false merges
                    if app_sim < 0.65 and frame_gap > 2:
                        strict_gap_limit = 2
                    fallback_ok = (
                        frame_gap <= strict_gap_limit
                        and iou_score >= cat_iou_min
                        and center_dist <= cat_center_max
                        and area_ratio >= _AREA_RATIO_MIN_STRICT
                    )
                else:
                    loose_gap_limit = _MAX_2D_FALLBACK_FRAME_GAP_LOOSE
                    if app_sim >= 0.82:
                        loose_gap_limit += 1
                    if app_sim >= 0.82 and track.category == "monitor":
                        loose_gap_limit += 1
                    # Tighten gap for low-similarity matches to prevent false merges
                    if app_sim < 0.65 and frame_gap > 2:
                        loose_gap_limit = 2
                    fallback_ok = (
                        frame_gap <= loose_gap_limit
                        and (iou_score >= iou_threshold or center_dist <= center_dist_threshold)
                        and area_ratio >= _AREA_RATIO_MIN_LOOSE
                    )

                if fallback_ok:
                    combined = (
                        iou_score
                        + max(0, 1.0 - center_dist)
                        + appearance_weight * app_sim
                    )
                    matches.append((combined, ti, di))
                    det_candidate_counts[di] = det_candidate_counts.get(di, 0) + 1
                    match_meta[(ti, di)] = {
                        "reason": "2d_iou_fallback",
                        "distance_m": None,
                        "iou": float(iou_score),
                        "threshold_m": None,
                        "center_dist": float(center_dist),
                        "area_ratio": float(area_ratio),
                        "frame_gap": frame_gap,
                        "fallback_mode": "after_3d_reject" if had_3d_pair else "no_3d_pair",
                        "iou_threshold": float(cat_iou_min if had_3d_pair else iou_threshold),
                        "center_dist_threshold": float(cat_center_max if had_3d_pair else center_dist_threshold),
                        "appearance_similarity": round(float(app_sim), 4),
                        "gap_limit_used": int(
                            strict_gap_limit if had_3d_pair else loose_gap_limit
                        ),
                    }

        # Greedy matching (highest score first)
        matches.sort(reverse=True)
        for score, ti, di in matches:
            if ti in matched_track_ids or di in matched_det_ids:
                continue
            det = frame_dets[di]
            track_id = active_tracks[ti].track_id
            meta = match_meta.get((ti, di), {})
            active_tracks[ti].detections.append(det)
            active_tracks[ti].last_frame_idx = frame_idx

            # Update track's world position using accumulated FP centroid
            # (Correction 5: accumulated centroid, not per-frame)
            _update_track_pos_from_fp(active_tracks[ti], shots, det_3d)

            matched_track_ids.add(ti)
            matched_det_ids.add(di)
            assignment_events.append({
                "frame_idx": frame_idx,
                "category": det.category,
                "detection_score": round(float(det.score), 4),
                "matched_track_id": int(track_id),
                "distance_m": round(float(meta["distance_m"]), 4)
                    if meta.get("distance_m") is not None else None,
                "iou": round(float(meta["iou"]), 4)
                    if meta.get("iou") is not None else None,
                "threshold_m": round(float(meta["threshold_m"]), 4)
                    if meta.get("threshold_m") is not None else None,
                "center_dist": round(float(meta["center_dist"]), 4)
                    if meta.get("center_dist") is not None else None,
                "area_ratio": round(float(meta["area_ratio"]), 4)
                    if meta.get("area_ratio") is not None else None,
                "frame_gap": int(meta["frame_gap"]) if meta.get("frame_gap") is not None else None,
                "fallback_mode": meta.get("fallback_mode"),
                "iou_threshold": round(float(meta["iou_threshold"]), 4)
                    if meta.get("iou_threshold") is not None else None,
                "center_dist_threshold": round(float(meta["center_dist_threshold"]), 4)
                    if meta.get("center_dist_threshold") is not None else None,
                "reason": meta.get("reason", "unknown"),
                "appearance_similarity": round(float(meta["appearance_similarity"]), 4)
                    if meta.get("appearance_similarity") is not None else None,
                "threshold_relaxed": bool(meta.get("threshold_relaxed", False)),
            })

        # Start new tracks for unmatched detections
        for di, det in enumerate(frame_dets):
            if di not in matched_det_ids:
                min_new_track_score = float(
                    (new_track_min_score_by_cat or {}).get(det.category, 0.0)
                )
                if float(det.score) < min_new_track_score:
                    rejection_events.append({
                        "frame_idx": int(frame_idx),
                        "category": det.category,
                        "detection_score": round(float(det.score), 4),
                        "reason": "new_track_rejected_low_score",
                        "min_required_score": round(float(min_new_track_score), 4),
                        "candidate_match_count": int(det_candidate_counts.get(di, 0)),
                    })
                    continue
                rejection_events.append({
                    "frame_idx": int(frame_idx),
                    "category": det.category,
                    "detection_score": round(float(det.score), 4),
                    "reason": (
                        "new_track_no_viable_match_candidates"
                        if int(det_candidate_counts.get(di, 0)) == 0
                        else "new_track_existing_candidates_not_selected"
                    ),
                    "candidate_match_count": int(det_candidate_counts.get(di, 0)),
                })
                track = Track(
                    track_id=next_id,
                    category=det.category,
                    detections=[det],
                    last_frame_idx=frame_idx,
                    world_pos=det_3d.get(id(det)),
                )
                active_tracks.append(track)
                assignment_events.append({
                    "frame_idx": frame_idx,
                    "category": det.category,
                    "detection_score": round(float(det.score), 4),
                    "matched_track_id": int(next_id),
                    "distance_m": None,
                    "iou": None,
                    "threshold_m": None,
                    "reason": "new_track",
                    "appearance_similarity": None,
                })
                next_id += 1

        # Retire old tracks
        still_active = []
        for track in active_tracks:
            if frame_idx - track.last_frame_idx > max_age:
                finished_tracks.append(track)
            else:
                still_active.append(track)
        active_tracks = still_active

    # Add remaining active tracks
    finished_tracks.extend(active_tracks)

    dedup_events: List[Dict] = []
    if enable_track_duplicate_merge and len(finished_tracks) >= 2:
        finished_tracks, dedup_events = _merge_duplicate_overlapping_tracks(
            finished_tracks,
            det_3d=det_3d,
            det_app=det_app,
            shots=shots if shots is not None else [],
            world_dist_default=world_dist_default,
            iou_min=max(0.05, float(duplicate_iou_min)),
            center_max=max(0.1, float(duplicate_center_max)),
            world_dist_factor=max(0.5, float(duplicate_world_dist_factor)),
            min_score=float(duplicate_min_score),
            max_passes=max(1, int(duplicate_max_passes)),
        )
        if dedup_events:
            logger.info("Track duplicate-merge: merged %d overlapping pairs", len(dedup_events))

    stitch_events: List[Dict] = []
    if enable_tracklet_stitching and len(finished_tracks) >= 2:
        finished_tracks, stitch_events = _stitch_fragmented_tracks(
            finished_tracks,
            det_3d=det_3d,
            det_app=det_app,
            shots=shots if shots is not None else [],
            world_dist_default=world_dist_default,
            max_gap_frames=max(1, int(stitch_max_gap_frames)),
            world_dist_factor=max(1.0, float(stitch_world_dist_factor)),
            min_score=float(stitch_min_score),
            max_passes=max(1, int(stitch_max_passes)),
            stitch_2d_max_gap=max(1, int(stitch_2d_fallback_max_gap)),
        )
        if stitch_events:
            logger.info("Track stitching: merged %d fragmented pairs", len(stitch_events))

    singleton_events: List[Dict] = []
    if enable_singleton_merge and len(finished_tracks) >= 2:
        finished_tracks, singleton_events = _merge_singleton_tracks(
            finished_tracks,
            det_3d=det_3d,
            det_app=det_app,
            world_dist_default=world_dist_default,
            world_dist_factor=singleton_merge_world_dist_factor,
            min_appearance=singleton_merge_min_appearance,
            high_conf_threshold=singleton_merge_high_conf_threshold,
        )
        if singleton_events:
            logger.info("Singleton merge: absorbed %d singleton tracks", len(singleton_events))

    logger.info(
        "Tracking complete: %d tracks from %d detections across %d frames",
        len(finished_tracks), len(detections), len(frame_indices),
    )

    # Log track summary
    cat_counts: Dict[str, int] = {}
    for t in finished_tracks:
        cat_counts[t.category] = cat_counts.get(t.category, 0) + 1
    if cat_counts:
        logger.info("Tracks by category: %s", ", ".join(f"{k}:{v}" for k, v in sorted(cat_counts.items())))

    # Log individual tracks for debugging
    for t in finished_tracks:
        frames_in = sorted(set(d.frame_idx for d in t.detections))
        pos_str = f"({t.world_pos[0]:.2f}, {t.world_pos[1]:.2f}, {t.world_pos[2]:.2f})" if t.world_pos else "no-3D"
        logger.info("  Track T%d %s: %d dets, frames=%s, pos=%s, score=%.2f",
                     t.track_id, t.category, len(t.detections),
                     frames_in, pos_str, t.avg_score)

    # Tracking fragmentation diagnostics
    frag_by_cat: Dict[str, Dict] = {}
    for t in finished_tracks:
        cat = str(t.category)
        entry = frag_by_cat.setdefault(cat, {"tracks": 0, "single_frame_tracks": 0})
        entry["tracks"] += 1
        n_frames = len(set(d.frame_idx for d in t.detections))
        if n_frames <= 1:
            entry["single_frame_tracks"] += 1
    frag_rates: Dict[str, float] = {}
    frag_severity: Dict[str, Dict] = {}
    alerts: List[str] = []
    for cat, stat in sorted(frag_by_cat.items()):
        tot = max(1, int(stat["tracks"]))
        rate = float(stat["single_frame_tracks"]) / float(tot)
        frag_rates[cat] = round(rate, 4)
        severity = "low"
        if rate >= 0.7:
            severity = "high"
        elif rate >= 0.45:
            severity = "medium"
        frag_severity[cat] = {
            "tracks": int(stat["tracks"]),
            "single_frame_tracks": int(stat["single_frame_tracks"]),
            "rate": round(rate, 4),
            "severity": severity,
        }
        if tot >= 2 and rate >= float(fragmentation_alert_threshold):
            alerts.append(f"{cat}:{rate:.2f}")
    _tracking_diagnostics = {
        "fragmentation_rate_by_category": frag_rates,
        "fragmentation_severity_by_category": frag_severity,
        "fragmentation_alerts": alerts,
        "n_single_frame_tracks": int(
            sum(1 for t in finished_tracks if len(set(d.frame_idx for d in t.detections)) <= 1)
        ),
        "n_tracks": int(len(finished_tracks)),
    }

    # Save tracking debug report
    _save_tracking_report(
        finished_tracks,
        assignment_events,
        rejection_events,
        stitch_events,
        dedup_events,
        fragmentation_diag=_tracking_diagnostics,
        singleton_merges=singleton_events,
    )

    return finished_tracks


def save_tracking_overlays(
    tracks: List[Track],
    frames: List,
    debug_dir: str = "/tmp",
) -> None:
    """
    Draw tracking results on each frame image so you can SEE which detections
    belong to the same track.  Each track gets a unique color; its track_id and
    category are written next to every bbox.

    Output: /tmp/decroom_track_<frame_idx>.png (one per frame)

    Call this from pipeline.py after tracking finishes.
    """
    if np is None:
        return
    try:
        import cv2
    except ImportError:
        logger.debug("cv2 not available, skipping tracking overlays")
        return

    # Build a lookup: frame_idx → [(track, detection)]
    frame_data: Dict[int, List[Tuple[Track, Detection]]] = {}
    for track in tracks:
        for det in track.detections:
            frame_data.setdefault(det.frame_idx, []).append((track, det))

    # Distinct colours per track_id (cycle through a bright palette)
    _PALETTE = [
        (255, 60, 60), (60, 255, 60), (60, 60, 255), (255, 255, 60),
        (255, 60, 255), (60, 255, 255), (255, 160, 60), (160, 60, 255),
        (60, 255, 160), (255, 60, 160), (160, 255, 60), (60, 160, 255),
        (200, 200, 60), (200, 60, 200), (60, 200, 200), (255, 128, 128),
    ]

    for frame_idx, items in frame_data.items():
        if frame_idx >= len(frames):
            continue

        # Load the frame image
        frame = frames[frame_idx]
        if hasattr(frame, 'file'):
            import io
            from PIL import Image
            frame.file.seek(0)
            pil_img = Image.open(io.BytesIO(frame.file.read())).convert("RGB")
            frame.file.seek(0)
            img = np.array(pil_img)
        elif isinstance(frame, (str,)):
            from pathlib import Path
            img = cv2.imread(str(frame))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                continue
        elif isinstance(frame, np.ndarray):
            img = frame.copy()
        else:
            continue

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        overlay = img.copy()
        ih, iw = overlay.shape[:2]

        for track, det in items:
            color = _PALETTE[track.track_id % len(_PALETTE)]

            x1, y1, x2, y2 = det.bbox
            det_w, det_h = det.input_size
            sx = iw / det_w
            sy = ih / det_h
            bx1 = int(max(0, x1 * sx))
            by1 = int(max(0, y1 * sy))
            bx2 = int(min(iw, x2 * sx))
            by2 = int(min(ih, y2 * sy))

            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 3)

            label = f"T{track.track_id} {det.category} {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(overlay, (bx1, by1 - th - 10), (bx1 + tw + 6, by1), color, -1)
            cv2.putText(overlay, label, (bx1 + 3, by1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        n_tracks_in_frame = len(set(t.track_id for t, _ in items))
        info = f"Frame {frame_idx}: {len(items)} detections, {n_tracks_in_frame} tracks"
        cv2.putText(overlay, info, (10, ih - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, info, (10, ih - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        vis_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{debug_dir}/decroom_track_{frame_idx}.png", vis_bgr)

    logger.info("Tracking overlays saved: %d frames to %s/decroom_track_*.png",
                len(frame_data), debug_dir)


def _track_frame_span(track: Track) -> Tuple[int, int]:
    frames = [d.frame_idx for d in track.detections]
    if not frames:
        return (-1, -1)
    return (min(frames), max(frames))


def _stitch_fragmented_tracks(
    tracks: List[Track],
    det_3d: Dict[int, Optional[Tuple[float, float, float]]],
    det_app: Dict[int, Optional["np.ndarray"]],
    shots: List,
    world_dist_default: float,
    max_gap_frames: int,
    world_dist_factor: float,
    min_score: float,
    max_passes: int,
    stitch_2d_max_gap: int = 6,
) -> Tuple[List[Track], List[Dict]]:
    """
    Merge temporally fragmented tracks after primary matching.

    Typical fragmentation case: the same object becomes multiple singleton/
    short tracks due to brief association failure. We stitch only when evidence
    is strong enough (3D proximity preferred, 2D endpoint fallback if needed).
    """
    if len(tracks) < 2:
        return tracks, []

    merged_tracks = list(tracks)
    stitch_events: List[Dict] = []

    for _ in range(max_passes):
        best_pair = None
        best_meta = None
        best_score = min_score

        for i in range(len(merged_tracks)):
            ta = merged_tracks[i]
            if not ta.detections:
                continue
            a_first, a_last = _track_frame_span(ta)
            for j in range(len(merged_tracks)):
                if i == j:
                    continue
                tb = merged_tracks[j]
                if not tb.detections or ta.category != tb.category:
                    continue

                b_first, b_last = _track_frame_span(tb)
                if b_first <= a_last:
                    continue
                gap = b_first - a_last
                if gap < 1 or gap > max_gap_frames:
                    continue

                reason = None
                score = 0.0
                dist_3d = None
                iou_val = None
                center_dist = None
                area_ratio = None
                appearance_similarity = None

                a_tail = max(ta.detections, key=lambda d: d.frame_idx)
                b_head = min(tb.detections, key=lambda d: d.frame_idx)
                appearance_similarity = _appearance_similarity(
                    det_app.get(id(a_tail)),
                    det_app.get(id(b_head)),
                )

                if ta.world_pos is not None and tb.world_pos is not None:
                    thr = _WORLD_DIST_BY_CAT.get(ta.category, world_dist_default) * world_dist_factor
                    d = _world_dist(ta.world_pos, tb.world_pos)
                    if d > thr:
                        continue
                    dist_3d = float(d)
                    # Prefer multi-observation chains and short temporal gaps.
                    support = min(0.4, 0.04 * (len(ta.detections) + len(tb.detections)))
                    score = (
                        (1.0 - (d / max(thr, 1e-6)))
                        + support
                        - 0.03 * max(0, gap - 1)
                        + 0.2 * appearance_similarity
                    )
                    reason = "stitch_3d"
                else:
                    iou_score = _iou(a_tail.bbox, b_head.bbox)
                    cdist = _bbox_center_dist(a_tail.bbox, b_head.bbox)
                    ar = _bbox_area_ratio(a_tail.bbox, b_head.bbox)
                    if not (gap <= stitch_2d_max_gap and iou_score >= 0.12 and cdist <= 0.55 and ar >= 0.22):
                        continue
                    iou_val = float(iou_score)
                    center_dist = float(cdist)
                    area_ratio = float(ar)
                    score = (
                        0.55 * iou_score
                        + 0.45 * max(0.0, 1.0 - cdist)
                        + 0.2 * appearance_similarity
                        - 0.02 * max(0, gap - 1)
                    )
                    reason = "stitch_2d_fallback"

                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
                    best_meta = {
                        "source_track_id": int(ta.track_id),
                        "target_track_id": int(tb.track_id),
                        "category": ta.category,
                        "gap_frames": int(gap),
                        "score": round(float(score), 4),
                        "reason": reason,
                        "distance_m": round(float(dist_3d), 4) if dist_3d is not None else None,
                        "iou": round(float(iou_val), 4) if iou_val is not None else None,
                        "center_dist": round(float(center_dist), 4) if center_dist is not None else None,
                        "area_ratio": round(float(area_ratio), 4) if area_ratio is not None else None,
                        "appearance_similarity": round(float(appearance_similarity), 4),
                        "source_frames": [int(a_first), int(a_last)],
                        "target_frames": [int(b_first), int(b_last)],
                    }

        if best_pair is None or best_meta is None:
            break

        i, j = best_pair
        ta = merged_tracks[i]
        tb = merged_tracks[j]

        ta.detections.extend(tb.detections)
        ta.detections.sort(key=lambda d: (d.frame_idx, -float(d.score)))
        ta.last_frame_idx = max(d.frame_idx for d in ta.detections)

        if ta.world_pos is not None and tb.world_pos is not None:
            na = max(1, len(ta.detections) - len(tb.detections))
            nb = max(1, len(tb.detections))
            ta.world_pos = (
                (ta.world_pos[0] * na + tb.world_pos[0] * nb) / (na + nb),
                (ta.world_pos[1] * na + tb.world_pos[1] * nb) / (na + nb),
                (ta.world_pos[2] * na + tb.world_pos[2] * nb) / (na + nb),
            )
        elif ta.world_pos is None and tb.world_pos is not None:
            ta.world_pos = tb.world_pos

        if shots:
            _update_track_pos_from_fp(ta, shots, det_3d)

        del merged_tracks[j]
        stitch_events.append(best_meta)

    merged_tracks.sort(key=lambda t: t.track_id)
    return merged_tracks, stitch_events


def _merge_duplicate_overlapping_tracks(
    tracks: List[Track],
    det_3d: Dict[int, Optional[Tuple[float, float, float]]],
    det_app: Dict[int, Optional["np.ndarray"]],
    shots: List,
    world_dist_default: float,
    iou_min: float,
    center_max: float,
    world_dist_factor: float,
    min_score: float,
    max_passes: int,
) -> Tuple[List[Track], List[Dict]]:
    """
    Merge duplicate tracks that overlap in time and likely represent the same object.

    This targets failure mode where one object is split into multiple tracks that
    coexist across overlapping frames (e.g., duplicate beds in the final payload).
    """
    if len(tracks) < 2:
        return tracks, []

    merged = list(tracks)
    events: List[Dict] = []

    for _ in range(max_passes):
        best_pair = None
        best_meta = None
        best_score = min_score

        for i in range(len(merged)):
            ta = merged[i]
            if not ta.detections:
                continue
            by_frame_a: Dict[int, Detection] = {}
            for d in ta.detections:
                prev = by_frame_a.get(d.frame_idx)
                if prev is None or float(d.score) > float(prev.score):
                    by_frame_a[d.frame_idx] = d

            for j in range(i + 1, len(merged)):
                tb = merged[j]
                if not tb.detections or ta.category != tb.category:
                    continue

                by_frame_b: Dict[int, Detection] = {}
                for d in tb.detections:
                    prev = by_frame_b.get(d.frame_idx)
                    if prev is None or float(d.score) > float(prev.score):
                        by_frame_b[d.frame_idx] = d

                overlap = sorted(set(by_frame_a.keys()) & set(by_frame_b.keys()))
                if not overlap:
                    continue

                ious: List[float] = []
                centers: List[float] = []
                areas: List[float] = []
                for f in overlap:
                    da = by_frame_a[f]
                    db = by_frame_b[f]
                    ious.append(float(_iou(da.bbox, db.bbox)))
                    centers.append(float(_bbox_center_dist(da.bbox, db.bbox)))
                    areas.append(float(_bbox_area_ratio(da.bbox, db.bbox)))

                med_iou = float(np.median(np.array(ious, dtype=np.float64))) if np is not None else sorted(ious)[len(ious)//2]
                med_center = float(np.median(np.array(centers, dtype=np.float64))) if np is not None else sorted(centers)[len(centers)//2]
                med_area = float(np.median(np.array(areas, dtype=np.float64))) if np is not None else sorted(areas)[len(areas)//2]
                app_sims: List[float] = []
                for f in overlap:
                    da = by_frame_a[f]
                    db = by_frame_b[f]
                    app_sims.append(_appearance_similarity(det_app.get(id(da)), det_app.get(id(db))))
                med_app = float(np.median(np.array(app_sims, dtype=np.float64))) if app_sims else 0.0

                if not (med_iou >= iou_min and med_center <= center_max and med_area >= 0.22):
                    continue

                dist_3d = None
                if ta.world_pos is not None and tb.world_pos is not None:
                    dist_3d = _world_dist(ta.world_pos, tb.world_pos)
                    thr = _WORLD_DIST_BY_CAT.get(ta.category, world_dist_default) * world_dist_factor
                    if dist_3d > thr:
                        continue

                support = min(0.35, 0.03 * min(len(ta.detections), len(tb.detections)))
                score = (
                    0.45 * med_iou
                    + 0.35 * max(0.0, 1.0 - med_center)
                    + 0.2 * min(1.0, med_area)
                    + 0.2 * med_app
                    + support
                )
                if dist_3d is not None:
                    thr = _WORLD_DIST_BY_CAT.get(ta.category, world_dist_default) * world_dist_factor
                    score += 0.15 * max(0.0, 1.0 - (dist_3d / max(thr, 1e-6)))

                if score > best_score:
                    # Merge smaller support into larger support track.
                    if len(ta.detections) >= len(tb.detections):
                        keep_idx, drop_idx = i, j
                        keep_t, drop_t = ta, tb
                    else:
                        keep_idx, drop_idx = j, i
                        keep_t, drop_t = tb, ta
                    best_score = score
                    best_pair = (keep_idx, drop_idx)
                    best_meta = {
                        "keep_track_id": int(keep_t.track_id),
                        "drop_track_id": int(drop_t.track_id),
                        "category": keep_t.category,
                        "overlap_frames": int(len(overlap)),
                        "median_iou": round(med_iou, 4),
                        "median_center_dist": round(med_center, 4),
                        "median_area_ratio": round(med_area, 4),
                        "median_appearance_similarity": round(med_app, 4),
                        "distance_m": round(float(dist_3d), 4) if dist_3d is not None else None,
                        "score": round(float(score), 4),
                        "reason": "duplicate_overlap_merge",
                    }

        if best_pair is None or best_meta is None:
            break

        keep_idx, drop_idx = best_pair
        keep_t = merged[keep_idx]
        drop_t = merged[drop_idx]

        keep_t.detections.extend(drop_t.detections)
        keep_t.detections.sort(key=lambda d: (d.frame_idx, -float(d.score)))
        keep_t.last_frame_idx = max(d.frame_idx for d in keep_t.detections)

        if keep_t.world_pos is not None and drop_t.world_pos is not None:
            nk = max(1, len(keep_t.detections) - len(drop_t.detections))
            nd = max(1, len(drop_t.detections))
            keep_t.world_pos = (
                (keep_t.world_pos[0] * nk + drop_t.world_pos[0] * nd) / (nk + nd),
                (keep_t.world_pos[1] * nk + drop_t.world_pos[1] * nd) / (nk + nd),
                (keep_t.world_pos[2] * nk + drop_t.world_pos[2] * nd) / (nk + nd),
            )
        elif keep_t.world_pos is None and drop_t.world_pos is not None:
            keep_t.world_pos = drop_t.world_pos

        if shots:
            _update_track_pos_from_fp(keep_t, shots, det_3d)

        del merged[drop_idx]
        events.append(best_meta)

    merged.sort(key=lambda t: t.track_id)
    return merged, events


def _merge_singleton_tracks(
    tracks: List[Track],
    det_3d: Dict[int, Optional[Tuple[float, float, float]]],
    det_app: Dict[int, Optional["np.ndarray"]],
    world_dist_default: float,
    world_dist_factor: float = 1.5,
    min_appearance: float = 0.70,
    high_conf_threshold: float = 0.80,
) -> Tuple[List[Track], List[Dict]]:
    """
    Merge singleton tracks (exactly 1 detection) into non-singleton tracks
    of the same category when evidence is strong enough.

    This reduces single-frame track fragmentation without merging two
    singletons together (which would be unreliable).
    """
    if len(tracks) < 2:
        return tracks, []

    singletons: List[int] = []
    non_singletons: List[int] = []
    for idx, t in enumerate(tracks):
        n_frames = len(set(d.frame_idx for d in t.detections))
        if n_frames <= 1:
            singletons.append(idx)
        else:
            non_singletons.append(idx)

    if not singletons or not non_singletons:
        return tracks, []

    merged_indices: set = set()
    events: List[Dict] = []

    for si in singletons:
        st = tracks[si]
        if not st.detections:
            continue
        s_det = st.detections[0]
        s_pos = det_3d.get(id(s_det))
        s_app = det_app.get(id(s_det))

        best_target: Optional[int] = None
        best_score: float = 0.0
        best_reason: Optional[str] = None
        best_dist: Optional[float] = None
        best_app_sim: float = 0.0

        for ni in non_singletons:
            if ni in merged_indices:
                continue
            nt = tracks[ni]
            if nt.category != st.category:
                continue

            # Appearance similarity against the non-singleton's best detection
            nt_app = det_app.get(id(nt.best_detection))
            app_sim = _appearance_similarity(s_app, nt_app)

            matched = False
            match_reason = ""
            match_dist: Optional[float] = None

            # Condition (a): Both have 3D and are close
            if s_pos is not None and nt.world_pos is not None:
                cat_thr = _WORLD_DIST_BY_CAT.get(st.category, world_dist_default) * world_dist_factor
                d3 = _world_dist(s_pos, nt.world_pos)
                if d3 < cat_thr:
                    matched = True
                    match_reason = "singleton_3d_proximity"
                    match_dist = d3

            # Condition (b): High-confidence singleton with good appearance match
            if not matched and float(s_det.score) > high_conf_threshold and app_sim >= min_appearance:
                matched = True
                match_reason = "singleton_high_conf_appearance"

            if matched:
                score = app_sim
                if match_dist is not None:
                    cat_thr = _WORLD_DIST_BY_CAT.get(st.category, world_dist_default) * world_dist_factor
                    score += 0.5 * max(0.0, 1.0 - match_dist / max(cat_thr, 1e-6))
                if score > best_score:
                    best_score = score
                    best_target = ni
                    best_reason = match_reason
                    best_dist = match_dist
                    best_app_sim = app_sim

        if best_target is not None:
            target_track = tracks[best_target]
            # Merge singleton's detections into target
            target_track.detections.extend(st.detections)
            target_track.detections.sort(key=lambda d: (d.frame_idx, -float(d.score)))
            target_track.last_frame_idx = max(d.frame_idx for d in target_track.detections)
            merged_indices.add(si)
            events.append({
                "singleton_track_id": int(st.track_id),
                "target_track_id": int(target_track.track_id),
                "category": st.category,
                "reason": best_reason,
                "distance_m": round(float(best_dist), 4) if best_dist is not None else None,
                "appearance_similarity": round(float(best_app_sim), 4),
                "singleton_score": round(float(s_det.score), 4),
            })

    if merged_indices:
        tracks = [t for idx, t in enumerate(tracks) if idx not in merged_indices]
        tracks.sort(key=lambda t: t.track_id)

    return tracks, events


def _save_tracking_report(
    tracks: List[Track],
    assignments: List[Dict],
    rejections: List[Dict],
    stitches: List[Dict],
    duplicate_merges: List[Dict],
    fragmentation_diag: Optional[Dict] = None,
    singleton_merges: Optional[List[Dict]] = None,
) -> None:
    """
    Save /tmp/decroom_tracking_report.json — shows every track, which frames
    it matched, and how many detections it grouped.
    """
    try:
        report = []
        for t in tracks:
            frames = sorted(set(d.frame_idx for d in t.detections))
            entry = {
                "track_id": t.track_id,
                "category": t.category,
                "n_detections": len(t.detections),
                "frames": frames,
                "best_score": round(t.best_detection.score, 3),
                "avg_score": round(t.avg_score, 3),
            }
            if t.world_pos is not None:
                entry["world_pos"] = {
                    "x": round(t.world_pos[0], 3),
                    "y": round(t.world_pos[1], 3),
                    "z": round(t.world_pos[2], 3),
                }
            report.append(entry)
        report.sort(key=lambda r: (r["category"], r["track_id"]))
        assignments_sorted = sorted(
            assignments,
            key=lambda a: (a.get("frame_idx", -1), a.get("category", ""), a.get("matched_track_id", -1)),
        )
        rejections_sorted = sorted(
            rejections,
            key=lambda a: (a.get("frame_idx", -1), a.get("category", ""), a.get("reason", "")),
        )
        rejection_reason_counts: Dict[str, int] = {}
        for r in rejections_sorted:
            reason = str(r.get("reason", "unknown"))
            rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
        with open("/tmp/decroom_tracking_report.json", "w") as f:
            json.dump({
                "summary": {
                    "n_tracks": int(len(report)),
                    "n_assignments": int(len(assignments_sorted)),
                    "n_rejections": int(len(rejections_sorted)),
                    "n_stitches": int(len(stitches)),
                    "n_duplicate_merges": int(len(duplicate_merges)),
                    "n_singleton_merges": int(len(singleton_merges or [])),
                    "n_single_frame_tracks": int(
                        (fragmentation_diag or {}).get("n_single_frame_tracks", 0)
                    ),
                },
                "fragmentation_rate_by_category": (fragmentation_diag or {}).get(
                    "fragmentation_rate_by_category", {}
                ),
                "fragmentation_severity_by_category": (fragmentation_diag or {}).get(
                    "fragmentation_severity_by_category", {}
                ),
                "fragmentation_alerts": (fragmentation_diag or {}).get(
                    "fragmentation_alerts", []
                ),
                "rejection_reason_counts": rejection_reason_counts,
                "tracks": report,
                "assignments": assignments_sorted,
                "rejections": rejections_sorted,
                "stitches": stitches,
                "duplicate_merges": duplicate_merges,
                "singleton_merges": singleton_merges or [],
            }, f, indent=2)
        logger.info("Tracking report saved to /tmp/decroom_tracking_report.json")
    except Exception as e:
        logger.debug("Failed to save tracking report: %s", e)
