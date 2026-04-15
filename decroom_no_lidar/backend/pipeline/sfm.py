"""Optional SfM + Bundle Adjustment integration for Decroom.

This module wraps a COLMAP CLI workflow and aligns reconstructed camera poses
back to the ARKit metric frame via similarity transform.

Design goals:
- Keep pipeline stable when COLMAP is unavailable (clean fallback).
- Apply SfM/BA only when enough data quality exists.
- Preserve ARKit metric scale/frame by similarity alignment.
"""
from __future__ import annotations

import copy
import logging
import shutil
import subprocess
import tempfile
import time
from math import atan2, pi
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import pycolmap
except ImportError:
    pycolmap = None  # type: ignore

from .metadata import ShotMetadata

logger = logging.getLogger("decroom-backend.sfm")


def _get_vocab_tree_path() -> Optional[Path]:
    """Find COLMAP vocabulary tree for loop detection. Returns None if unavailable."""
    candidates = [
        Path.home() / ".colmap" / "vocab_tree_flickr100K_words256K.bin",
        Path.home() / ".colmap" / "vocab_tree_flickr100K_words32K.bin",
        Path("/opt/homebrew/share/colmap/vocab_tree_flickr100K_words256K.bin"),
        Path("/usr/local/share/colmap/vocab_tree_flickr100K_words256K.bin"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1000:
            logger.info("Found vocab tree: %s", p)
            return p
    logger.warning(
        "Vocab tree not found in any standard location, loop detection disabled. "
        "For better SfM with video captures, download vocab_tree_flickr100K_words256K.bin "
        "from https://demuc.de/colmap/ and place in ~/.colmap/"
    )
    return None


def _qvec_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> "np.ndarray":
    """Convert COLMAP qvec to 3x3 rotation matrix (world->camera)."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q /= n
    qw, qx, qy, qz = q.tolist()
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float64)


def _c2w_from_colmap(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float) -> "np.ndarray":
    """Build camera-to-world matrix from COLMAP world-to-camera pose."""
    R_wc = _qvec_to_rotmat(qw, qx, qy, qz)
    t_wc = np.array([tx, ty, tz], dtype=np.float64)

    R_cw = R_wc.T
    C = -R_wc.T @ t_wc

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R_cw
    mat[:3, 3] = C
    return mat


def _estimate_similarity_umeyama(src: "np.ndarray", dst: "np.ndarray") -> Optional[Tuple[float, "np.ndarray", "np.ndarray"]]:
    """Estimate similarity transform dst ~= s * R * src + t."""
    if np is None or len(src) < 3 or len(dst) < 3 or src.shape != dst.shape:
        return None

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    if var_src <= 1e-12:
        return None

    scale = float(np.trace(np.diag(D) @ S) / var_src)
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def _materialize_frames_for_sfm(frames, out_dir: Path, max_side: int, cfg=None) -> List[str]:
    """Write input frames to disk as sequential JPEG names for COLMAP."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names: List[str] = []

    if np is None:
        return names

    try:
        from PIL import Image
    except Exception:
        Image = None  # type: ignore
    try:
        from PIL import ImageEnhance, ImageOps
    except Exception:
        ImageEnhance = None  # type: ignore
        ImageOps = None  # type: ignore
    try:
        import cv2
    except Exception:
        cv2 = None  # type: ignore

    enhance_enabled = bool(getattr(cfg, "sfm_enable_image_enhancement", True))
    enhance_contrast = float(getattr(cfg, "sfm_enhance_contrast", 1.14))
    enhance_sharpness = float(getattr(cfg, "sfm_enhance_sharpness", 1.18))
    autocontrast_cutoff = float(getattr(cfg, "sfm_enhance_autocontrast_cutoff", 0.5))

    def _enhance_pil(img):
        if ImageEnhance is None:
            return img
        if not enhance_enabled:
            return img
        try:
            if ImageOps is not None:
                img = ImageOps.autocontrast(img, cutoff=max(0.0, float(autocontrast_cutoff)))
        except Exception:
            pass
        try:
            img = ImageEnhance.Contrast(img).enhance(max(1.0, float(enhance_contrast)))
        except Exception:
            pass
        try:
            img = ImageEnhance.Sharpness(img).enhance(max(1.0, float(enhance_sharpness)))
        except Exception:
            pass
        return img

    for idx, frame in enumerate(frames):
        name = f"frame_{idx:04d}.jpg"
        dst = out_dir / name
        saved = False

        # Existing file path
        if isinstance(frame, (str, Path)):
            src = Path(frame)
            if src.exists():
                try:
                    shutil.copy2(src, dst)
                    saved = True
                except Exception:
                    saved = False

        # UploadFile-like object
        if not saved and hasattr(frame, "file"):
            try:
                f = frame.file
                pos = f.tell()
                f.seek(0)
                data = f.read()
                f.seek(pos)
                if Image is not None:
                    img = Image.open(__import__("io").BytesIO(data)).convert("RGB")
                    w, h = img.size
                    scale = min(1.0, float(max_side) / max(float(w), float(h)))
                    if scale < 0.999:
                        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
                    img = _enhance_pil(img)
                    img.save(dst, format="JPEG", quality=92)
                    saved = True
                elif cv2 is not None and np is not None:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        h, w = img.shape[:2]
                        scale = min(1.0, float(max_side) / max(float(w), float(h)))
                        if scale < 0.999:
                            img = cv2.resize(
                                img,
                                (max(1, int(w * scale)), max(1, int(h * scale))),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                        saved = True
            except Exception:
                saved = False

        # Numpy frame
        if not saved and isinstance(frame, np.ndarray):
            try:
                arr = frame
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                h, w = arr.shape[:2]
                if Image is not None:
                    img = Image.fromarray(arr)
                    scale = min(1.0, float(max_side) / max(float(w), float(h)))
                    if scale < 0.999:
                        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
                    img = _enhance_pil(img)
                    img.save(dst, format="JPEG", quality=92)
                    saved = True
                elif cv2 is not None:
                    scale = min(1.0, float(max_side) / max(float(w), float(h)))
                    out = arr
                    if scale < 0.999:
                        out = cv2.resize(
                            out,
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    if out.shape[-1] == 3:
                        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(dst), out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    saved = True
            except Exception:
                saved = False

        if saved:
            names.append(name)

    return names


def _frame_idx_from_name(name: str) -> Optional[int]:
    """Parse frame index from 'frame_XXXX.jpg'."""
    try:
        stem = Path(name).stem
        if "_" not in stem:
            return None
        return int(stem.split("_")[-1])
    except Exception:
        return None


def _pose_xyzyaw(shot: ShotMetadata) -> Optional[Tuple["np.ndarray", float]]:
    """Return (camera center xyz, yaw_rad) from shot pose, if available."""
    if np is None:
        return None
    if shot.pose is None or len(shot.pose) != 16:
        return None
    try:
        mat = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
    except Exception:
        return None
    c = mat[:3, 3].astype(np.float64)
    fwd_x = -float(mat[0, 2])
    fwd_z = -float(mat[2, 2])
    yaw = float(atan2(fwd_x, fwd_z))
    return c, yaw


def _angle_delta_rad(a: float, b: float) -> float:
    """Absolute wrapped angular difference (radians)."""
    d = abs(a - b) % (2.0 * pi)
    return min(d, (2.0 * pi) - d)


def _image_sharpness(path: Path) -> float:
    """
    Blur/sharpness proxy: variance of Laplacian if OpenCV exists.
    Falls back to finite-difference energy.
    """
    if np is None:
        return 0.0
    try:
        import cv2
    except Exception:
        cv2 = None  # type: ignore

    if cv2 is not None:
        try:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return 0.0
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            pass

    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float64)
        gx = np.abs(np.diff(arr, axis=1))
        gy = np.abs(np.diff(arr, axis=0))
        return float(np.mean(gx) + np.mean(gy))
    except Exception:
        return 0.0


def _select_sfm_keyframes(
    names: List[str],
    images_dir: Path,
    shots: List[ShotMetadata],
    cfg,
) -> Tuple[List[str], Dict]:
    """
    Select a robust keyframe subset for SfM.

    Balances:
    - image quality (sharpness),
    - trajectory novelty (translation/yaw),
    - bounded frame count for stable matching.
    """
    diag = {
        "keyframe_prefilter_enabled": bool(getattr(cfg, "sfm_enable_keyframe_prefilter", True)),
        "keyframe_input_frames": int(len(names)),
        "keyframe_selected_frames": int(len(names)),
        "keyframe_skipped_blur": 0,
        "keyframe_skipped_motion": 0,
        "keyframe_skipped_sequence_break": 0,
        "keyframe_sharpness_min": float(getattr(cfg, "sfm_keyframe_sharpness_min", 14.0)),
        "keyframe_mean_sharpness": None,
        "keyframe_mean_sharpness_selected": None,
        "keyframe_mode": "disabled_or_passthrough",
    }
    if np is None or not names:
        return names, diag
    if not bool(getattr(cfg, "sfm_enable_keyframe_prefilter", True)):
        return names, diag

    min_frames = max(4, int(getattr(cfg, "sfm_keyframe_min_frames", 14)))
    max_frames = max(min_frames, int(getattr(cfg, "sfm_keyframe_max_frames", 48)))
    force_stride = max(1, int(getattr(cfg, "sfm_keyframe_force_stride", 4)))
    min_trans = max(0.01, float(getattr(cfg, "sfm_keyframe_min_translation_m", 0.08)))
    min_yaw = max(0.5, float(getattr(cfg, "sfm_keyframe_min_yaw_deg", 6.0))) * pi / 180.0
    sharp_min = float(getattr(cfg, "sfm_keyframe_sharpness_min", 14.0))

    # Build sortable entries with quality + pose cues.
    entries = []
    for name in names:
        idx = _frame_idx_from_name(name)
        if idx is None:
            continue
        if 0 <= idx < len(shots) and bool(getattr(shots[idx], "sequence_break_flag", False)):
            diag["keyframe_skipped_sequence_break"] += 1
            continue
        sharp = _image_sharpness(images_dir / name)
        pose_data = _pose_xyzyaw(shots[idx]) if 0 <= idx < len(shots) else None
        entries.append({
            "name": name,
            "idx": idx,
            "sharp": float(sharp),
            "pose": pose_data[0] if pose_data is not None else None,
            "yaw": pose_data[1] if pose_data is not None else None,
        })
    entries.sort(key=lambda e: int(e["idx"]))
    if len(entries) <= min_frames:
        diag["keyframe_mode"] = "insufficient_for_prefilter"
        diag["keyframe_selected_frames"] = int(len(entries))
        return [str(e["name"]) for e in entries], diag

    sharp_vals = np.array([float(e["sharp"]) for e in entries], dtype=np.float64)
    if len(sharp_vals) > 0:
        diag["keyframe_mean_sharpness"] = round(float(np.mean(sharp_vals)), 4)

    # Stage A: quality gate with stride-based anchors to preserve coverage.
    cand: List[Dict] = []
    skipped_blur = 0
    for k, e in enumerate(entries):
        is_anchor = (k % force_stride == 0) or (k == 0) or (k == len(entries) - 1)
        if e["sharp"] >= sharp_min or is_anchor:
            cand.append(e)
        else:
            skipped_blur += 1
    diag["keyframe_skipped_blur"] = int(skipped_blur)

    # Guarantee lower bound by recovering highest-sharp skipped frames.
    if len(cand) < min_frames:
        taken = set(int(e["idx"]) for e in cand)
        for e in sorted(entries, key=lambda x: float(x["sharp"]), reverse=True):
            if int(e["idx"]) in taken:
                continue
            cand.append(e)
            taken.add(int(e["idx"]))
            if len(cand) >= min_frames:
                break
        cand.sort(key=lambda e: int(e["idx"]))

    # Stage B: motion novelty gate.
    selected: List[Dict] = []
    skipped_motion = 0
    last = None
    for e in cand:
        if not selected:
            selected.append(e)
            last = e
            continue

        keep = False
        if int(e["idx"]) - int(last["idx"]) >= force_stride:
            keep = True
        else:
            if e["pose"] is not None and last["pose"] is not None:
                trans = float(np.linalg.norm(e["pose"] - last["pose"]))
                yaw_d = _angle_delta_rad(float(e["yaw"]), float(last["yaw"]))
                if trans >= min_trans or yaw_d >= min_yaw:
                    keep = True

        if keep:
            selected.append(e)
            last = e
        else:
            skipped_motion += 1

    # Always include last chronological candidate for loop closure chances.
    if selected and selected[-1]["idx"] != cand[-1]["idx"]:
        selected.append(cand[-1])

    # Recover frames if motion gate got too aggressive.
    if len(selected) < min_frames:
        taken = set(int(e["idx"]) for e in selected)
        for e in sorted(cand, key=lambda x: float(x["sharp"]), reverse=True):
            if int(e["idx"]) in taken:
                continue
            selected.append(e)
            taken.add(int(e["idx"]))
            if len(selected) >= min_frames:
                break
        selected.sort(key=lambda e: int(e["idx"]))

    # Cap upper bound with near-uniform temporal sampling.
    if len(selected) > max_frames:
        idxs = np.linspace(0, len(selected) - 1, num=max_frames)
        chosen = sorted(set(int(round(v)) for v in idxs))
        selected = [selected[i] for i in chosen]

    selected_names = [str(e["name"]) for e in selected]
    diag["keyframe_selected_frames"] = int(len(selected_names))
    diag["keyframe_skipped_motion"] = int(skipped_motion)
    diag["keyframe_mode"] = "quality_motion_prefilter"
    if selected:
        sel_sharp = np.array([float(e["sharp"]) for e in selected], dtype=np.float64)
        diag["keyframe_mean_sharpness_selected"] = round(float(np.mean(sel_sharp)), 4)
    return selected_names, diag


def _prefilter_sfm_match_graph(
    images_dir: Path,
    names: List[str],
    cfg,
    shots: Optional[List["ShotMetadata"]] = None,
) -> Tuple[List[str], Dict]:
    """
    Build a lightweight geometric match graph and prune disconnected/isolated frames.

    Keeps the largest connected component (subject to min component size) and drops
    degree-0 nodes to improve COLMAP registration reliability.
    """
    diag = {
        "enabled": bool(getattr(cfg, "sfm_enable_match_graph_prefilter", True)),
        "status": "disabled",
        "connected_components": 0,
        "largest_component_size": 0,
        "node_degree_per_frame": {},
        "pruned_frames": [],
        "pruned_count": 0,
        "frames_in": int(len(names)),
        "frames_out": int(len(names)),
        "edges": 0,
    }
    if np is None or not names:
        diag["status"] = "numpy_missing_or_empty"
        return names, diag
    if not bool(getattr(cfg, "sfm_enable_match_graph_prefilter", True)):
        return names, diag

    try:
        import cv2
    except Exception:
        diag["status"] = "cv2_missing"
        return names, diag

    # Detect sparse captures via median yaw delta between consecutive frames.
    is_sparse = False
    sparse_yaw_thr_deg = float(getattr(cfg, "sfm_sparse_yaw_threshold_deg", 45.0))
    if shots and np is not None:
        yaw_deltas: List[float] = []
        prev_yaw: Optional[float] = None
        for sh in shots:
            pinfo = _pose_xyzyaw(sh)
            if pinfo is not None:
                _, yaw_val = pinfo
                if prev_yaw is not None:
                    yaw_deltas.append(_angle_delta_rad(yaw_val, prev_yaw) * 180.0 / pi)
                prev_yaw = yaw_val
        if yaw_deltas:
            median_yaw_delta = float(np.median(yaw_deltas))
            is_sparse = median_yaw_delta > sparse_yaw_thr_deg
            diag["median_yaw_delta_deg"] = round(median_yaw_delta, 2)
            diag["is_sparse_capture"] = is_sparse

    # Detect dense video captures: many frames with small yaw deltas
    is_dense_video = (not is_sparse) and len(names) > 30 and diag.get("median_yaw_delta_deg", 999) < 15.0
    diag["is_dense_video"] = is_dense_video

    if is_sparse:
        min_inliers = max(8, int(getattr(cfg, "sfm_graph_min_inliers_sparse", 25)))
        min_ratio = max(0.05, float(getattr(cfg, "sfm_graph_min_inlier_ratio_sparse", 0.12)))
        temporal_window = max(1, int(getattr(cfg, "sfm_graph_temporal_window_sparse", 8)))
    elif is_dense_video:
        min_inliers = max(8, int(getattr(cfg, "sfm_graph_min_inliers_video", 20)))
        min_ratio = max(0.05, float(getattr(cfg, "sfm_graph_min_inlier_ratio_video", 0.10)))
        temporal_window = max(1, int(getattr(cfg, "sfm_graph_temporal_window_video", 12)))
    else:
        min_inliers = max(8, int(getattr(cfg, "sfm_graph_min_inliers", 45)))
        min_ratio = max(0.05, float(getattr(cfg, "sfm_graph_min_inlier_ratio", 0.22)))
        temporal_window = max(1, int(getattr(cfg, "sfm_graph_temporal_window", 5)))
    min_component = max(2, int(getattr(cfg, "sfm_graph_min_component_size", 3)))

    orb = cv2.ORB_create(1600)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(names)}
    kp_desc: Dict[str, Tuple[Optional[List], Optional["np.ndarray"]]] = {}
    for n in names:
        img = cv2.imread(str(images_dir / n), cv2.IMREAD_GRAYSCALE)
        if img is None:
            kp_desc[n] = (None, None)
            continue
        k, d = orb.detectAndCompute(img, None)
        kp_desc[n] = (k, d)

    adjacency: Dict[str, set] = {n: set() for n in names}
    # For dense video, also check loop-closure pairs (first N vs last N)
    loop_closure_window = 3 if is_dense_video else 0
    n_frames = len(names)
    for i, a in enumerate(names):
        ka, da = kp_desc.get(a, (None, None))
        if da is None or ka is None or len(ka) < 12:
            continue
        for j in range(i + 1, len(names)):
            is_temporal = (j - i) <= temporal_window
            is_loop = (i < loop_closure_window and j >= n_frames - loop_closure_window)
            if not is_temporal and not is_loop:
                continue
            b = names[j]
            kb, db = kp_desc.get(b, (None, None))
            if db is None or kb is None or len(kb) < 12:
                continue
            matches = bf.match(da, db)
            if len(matches) < 20:
                continue
            matches = sorted(matches, key=lambda m: m.distance)[:220]
            if len(matches) < 8:
                continue
            src = np.float32([ka[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([kb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            _, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
            ratio = float(inliers) / max(1, len(matches))
            if inliers >= min_inliers and ratio >= min_ratio:
                adjacency[a].add(b)
                adjacency[b].add(a)

    # Connected components on names.
    unvisited = set(names)
    comps: List[List[str]] = []
    while unvisited:
        seed = next(iter(unvisited))
        stack = [seed]
        comp = []
        unvisited.remove(seed)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adjacency.get(cur, set()):
                if nb in unvisited:
                    unvisited.remove(nb)
                    stack.append(nb)
        comps.append(sorted(comp, key=lambda x: name_to_idx.get(x, 0)))
    comps.sort(key=lambda c: (-len(c), min(name_to_idx.get(n, 0) for n in c)))
    diag["connected_components"] = int(len(comps))
    diag["largest_component_size"] = int(len(comps[0])) if comps else 0
    diag["edges"] = int(sum(len(v) for v in adjacency.values()) // 2)

    node_degree = {
        str(_frame_idx_from_name(n)): int(len(adjacency.get(n, set())))
        for n in names
    }
    diag["node_degree_per_frame"] = node_degree

    keep_set = set(names)
    if comps:
        largest = comps[0]
        if len(largest) >= min_component:
            keep_set = set(largest)
        else:
            # If all components are tiny, only drop strict isolates.
            keep_set = set(n for n in names if len(adjacency.get(n, set())) > 0)
            if not keep_set:
                keep_set = set(names)

    # Always prune strict degree-0 nodes unless that empties the set.
    degree_nonzero = set(n for n in names if len(adjacency.get(n, set())) > 0)
    if degree_nonzero:
        keep_set &= degree_nonzero
    if len(keep_set) < max(4, min_component):
        # Avoid over-pruning; retain original set if too few frames remain.
        keep_set = set(names)
        diag["status"] = "kept_all_after_safety_floor"
    else:
        diag["status"] = "pruned"

    kept = [n for n in names if n in keep_set]
    pruned = [n for n in names if n not in keep_set]
    pruned_indices: List[int] = []
    for n in pruned:
        idx = _frame_idx_from_name(n)
        if idx is not None:
            pruned_indices.append(int(idx))
    diag["pruned_frames"] = pruned_indices
    diag["pruned_count"] = int(len(pruned))
    diag["frames_out"] = int(len(kept))
    return kept, diag


def _run_colmap_cmd(args: List[str], timeout_s: int = 1200) -> Tuple[bool, str]:
    """Run COLMAP command and return (ok, stderr_or_error)."""
    try:
        proc = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
    except Exception as e:
        return False, str(e)

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        return False, msg[:1200]
    return True, ""


def _parse_colmap_images_txt(images_txt: Path) -> Dict[str, "np.ndarray"]:
    """Parse COLMAP images.txt and return name -> c2w matrix."""
    poses: Dict[str, "np.ndarray"] = {}
    if np is None or not images_txt.exists():
        return poses

    with open(images_txt, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    # images.txt alternates: pose line, points2D line
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 10:
            try:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                name = parts[9]
                poses[name] = _c2w_from_colmap(qw, qx, qy, qz, tx, ty, tz)
            except Exception:
                pass
        i += 2

    return poses


def _choose_best_model_dir(sparse_root: Path) -> Optional[Path]:
    """Pick model folder with highest registered image count."""
    candidates = [p for p in sparse_root.iterdir() if p.is_dir()]
    if not candidates:
        return None

    best = None
    best_count = -1
    for c in candidates:
        images_bin = c / "images.bin"
        images_txt = c / "images.txt"
        count = 0
        if images_txt.exists():
            with open(images_txt, "r") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln and not ln.startswith("#"):
                        count += 0.5  # two-line entries
            count = int(count)
        elif images_bin.exists():
            # unknown without conversion; keep candidate viable
            count = 1
        if count > best_count:
            best_count = count
            best = c

    return best


def _run_colmap_cli_pipeline(
    colmap_cmd: str,
    db_path: Path,
    images_dir: Path,
    sparse_dir: Path,
    sparse_txt: Path,
    names: List[str],
    cfg,
) -> Tuple[bool, Dict[str, "np.ndarray"], str]:
    """
    Run COLMAP CLI SfM with retry attempts.

    Attempts:
    - primary config from cfg
    - relaxed fallback (SIMPLE_RADIAL + exhaustive)
    """
    use_gpu = "1" if bool(getattr(cfg, "sfm_use_gpu", False)) else "0"
    sift_max_features = int(getattr(cfg, "sfm_sift_max_num_features", 12000))
    sift_peak_thr = float(getattr(cfg, "sfm_sift_peak_threshold", 0.004))
    sift_edge_thr = float(getattr(cfg, "sfm_sift_edge_threshold", 10.0))
    use_guided_matching = bool(getattr(cfg, "sfm_matching_guided", False))
    guided_matching = "1" if use_guided_matching else "0"
    max_num_matches = int(getattr(cfg, "sfm_matching_max_num_matches", 32768))
    mapper_min_matches = int(getattr(cfg, "sfm_mapper_min_num_matches", 10))
    mapper_init_inliers = int(getattr(cfg, "sfm_mapper_init_min_num_inliers", 24))
    mapper_abs_pose_inliers = int(getattr(cfg, "sfm_mapper_abs_pose_min_num_inliers", 18))
    mapper_abs_pose_ratio = float(getattr(cfg, "sfm_mapper_abs_pose_min_inlier_ratio", 0.1))
    primary_camera_model = str(getattr(cfg, "sfm_camera_model", "SIMPLE_PINHOLE"))
    primary_matcher = str(getattr(cfg, "sfm_matcher", "sequential")).lower()
    attempts: List[Tuple[str, str]] = [(primary_camera_model, primary_matcher)]
    if primary_matcher != "exhaustive":
        attempts.append((primary_camera_model, "exhaustive"))
    if (primary_camera_model, "exhaustive") != ("SIMPLE_RADIAL", "exhaustive"):
        attempts.append(("SIMPLE_RADIAL", "exhaustive"))

    errors: List[str] = []
    for camera_model, matcher in attempts:
        try:
            if db_path.exists():
                db_path.unlink()
            shutil.rmtree(sparse_dir, ignore_errors=True)
            shutil.rmtree(sparse_txt, ignore_errors=True)
            sparse_dir.mkdir(parents=True, exist_ok=True)

            ok, msg = _run_colmap_cmd([
                colmap_cmd, "feature_extractor",
                "--database_path", str(db_path),
                "--image_path", str(images_dir),
                "--image_list_path", str((images_dir.parent / "image_list.txt")),
                "--ImageReader.camera_model", camera_model,
                "--ImageReader.single_camera", "1",
                "--FeatureExtraction.use_gpu", use_gpu,
                "--SiftExtraction.max_image_size", str(int(getattr(cfg, "sfm_max_image_size", 1600))),
                "--SiftExtraction.max_num_features", str(max(2000, sift_max_features)),
                "--SiftExtraction.peak_threshold", str(max(1e-4, sift_peak_thr)),
                "--SiftExtraction.edge_threshold", str(max(1.0, sift_edge_thr)),
            ])
            if not ok:
                errors.append(f"{camera_model}/{matcher}:feature_extractor:{msg}")
                continue

            if matcher == "sequential":
                matcher_args = [
                    colmap_cmd, "sequential_matcher",
                    "--database_path", str(db_path),
                    "--FeatureMatching.use_gpu", use_gpu,
                    "--SequentialMatching.overlap", str(int(getattr(cfg, "sfm_sequential_overlap", 10))),
                    "--FeatureMatching.max_num_matches", str(max(1000, max_num_matches)),
                ]
                if use_guided_matching:
                    matcher_args.extend(["--FeatureMatching.guided_matching", guided_matching])
                # Loop detection for video walk-around captures
                if getattr(cfg, "sfm_enable_loop_detection", True):
                    vocab_tree = _get_vocab_tree_path()
                    if vocab_tree is not None:
                        matcher_args.extend([
                            "--SequentialMatching.loop_detection", "1",
                            "--SequentialMatching.loop_detection_period",
                                str(getattr(cfg, "sfm_loop_detection_period", 10)),
                            "--SequentialMatching.loop_detection_num_images",
                                str(getattr(cfg, "sfm_loop_detection_num_images", 50)),
                            "--SequentialMatching.vocab_tree_path", str(vocab_tree),
                        ])
                ok, msg = _run_colmap_cmd(matcher_args)
            else:
                matcher_args = [
                    colmap_cmd, "exhaustive_matcher",
                    "--database_path", str(db_path),
                    "--FeatureMatching.use_gpu", use_gpu,
                    "--FeatureMatching.max_num_matches", str(max(1000, max_num_matches)),
                ]
                if use_guided_matching:
                    matcher_args.extend(["--FeatureMatching.guided_matching", guided_matching])
                ok, msg = _run_colmap_cmd(matcher_args)
            if not ok:
                errors.append(f"{camera_model}/{matcher}:matcher:{msg}")
                continue

            ok, msg = _run_colmap_cmd([
                colmap_cmd, "mapper",
                "--database_path", str(db_path),
                "--image_path", str(images_dir),
                "--output_path", str(sparse_dir),
                "--Mapper.ba_refine_focal_length", "0",
                "--Mapper.ba_refine_principal_point", "0",
                "--Mapper.ba_refine_extra_params", "0",
                "--Mapper.min_num_matches", str(max(4, mapper_min_matches)),
                "--Mapper.init_min_num_inliers", str(max(8, mapper_init_inliers)),
                "--Mapper.abs_pose_min_num_inliers", str(max(6, mapper_abs_pose_inliers)),
                "--Mapper.abs_pose_min_inlier_ratio", str(max(0.01, mapper_abs_pose_ratio)),
            ])
            if not ok:
                errors.append(f"{camera_model}/{matcher}:mapper:{msg}")
                continue

            best_model = _choose_best_model_dir(sparse_dir)
            if best_model is None:
                errors.append(f"{camera_model}/{matcher}:no_sparse_model")
                continue

            sparse_txt.mkdir(parents=True, exist_ok=True)
            ok, msg = _run_colmap_cmd([
                colmap_cmd, "model_converter",
                "--input_path", str(best_model),
                "--output_path", str(sparse_txt),
                "--output_type", "TXT",
            ])
            if not ok:
                errors.append(f"{camera_model}/{matcher}:model_converter:{msg}")
                continue

            poses = _parse_colmap_images_txt(sparse_txt / "images.txt")
            if not poses:
                errors.append(f"{camera_model}/{matcher}:no_poses_parsed")
                continue
            return True, poses, ""
        except Exception as e:
            errors.append(f"{camera_model}/{matcher}:failed:{e}")
            continue

    return False, {}, ("colmap_cli_failed_attempts:" + ";".join(errors))[:1800]


def _poses_from_pycolmap_reconstruction(reconstruction) -> Dict[str, "np.ndarray"]:
    """Extract name->c2w poses from a pycolmap Reconstruction."""
    poses: Dict[str, "np.ndarray"] = {}
    if np is None or reconstruction is None:
        return poses

    try:
        reg_ids = list(reconstruction.reg_image_ids())
    except Exception:
        reg_ids = []

    for image_id in reg_ids:
        try:
            image = reconstruction.image(image_id)
            name = str(image.name)
            cam_from_world = image.cam_from_world
            mat_w2c = np.array(cam_from_world.matrix(), dtype=np.float64)
            if mat_w2c.shape == (3, 4):
                tmp = np.eye(4, dtype=np.float64)
                tmp[:3, :4] = mat_w2c
                mat_w2c = tmp
            if mat_w2c.shape != (4, 4):
                continue
            poses[name] = np.linalg.inv(mat_w2c)
        except Exception:
            continue
    return poses


def _run_pycolmap_pipeline(
    db_path: Path,
    images_dir: Path,
    sparse_dir: Path,
    names: List[str],
    cfg,
) -> Tuple[bool, Dict[str, "np.ndarray"], str]:
    """Run SfM/BA via pycolmap bindings."""
    if pycolmap is None:
        return False, {}, "pycolmap_not_installed"

    use_gpu = bool(getattr(cfg, "sfm_use_gpu", False))
    primary_camera_model = str(getattr(cfg, "sfm_camera_model", "SIMPLE_PINHOLE"))
    primary_matcher = str(getattr(cfg, "sfm_matcher", "sequential")).lower()
    attempts: List[Tuple[str, str]] = [(primary_camera_model, primary_matcher)]
    # Recovery attempt: same camera model with exhaustive matching, then full fallback.
    if primary_matcher != "exhaustive":
        attempts.append((primary_camera_model, "exhaustive"))
    if (primary_camera_model, "exhaustive") != ("SIMPLE_RADIAL", "exhaustive"):
        attempts.append(("SIMPLE_RADIAL", "exhaustive"))

    attempt_errors: List[str] = []

    for camera_model, matcher in attempts:
        try:
            if db_path.exists():
                db_path.unlink()
            shutil.rmtree(sparse_dir, ignore_errors=True)
            sparse_dir.mkdir(parents=True, exist_ok=True)

            reader_options = pycolmap.ImageReaderOptions()
            reader_options.camera_model = camera_model

            extraction_options = pycolmap.FeatureExtractionOptions()
            extraction_options.max_image_size = int(getattr(cfg, "sfm_max_image_size", 1600))
            extraction_options.use_gpu = use_gpu
            if hasattr(extraction_options, "max_num_features"):
                extraction_options.max_num_features = int(
                    max(2000, int(getattr(cfg, "sfm_sift_max_num_features", 12000)))
                )
            if hasattr(extraction_options, "peak_threshold"):
                extraction_options.peak_threshold = float(
                    max(1e-4, float(getattr(cfg, "sfm_sift_peak_threshold", 0.004)))
                )
            if hasattr(extraction_options, "edge_threshold"):
                extraction_options.edge_threshold = float(
                    max(1.0, float(getattr(cfg, "sfm_sift_edge_threshold", 10.0)))
                )

            matching_options = pycolmap.FeatureMatchingOptions()
            matching_options.use_gpu = use_gpu
            if hasattr(matching_options, "guided_matching"):
                matching_options.guided_matching = bool(getattr(cfg, "sfm_matching_guided", True))
            if hasattr(matching_options, "max_num_matches"):
                matching_options.max_num_matches = int(
                    max(1000, int(getattr(cfg, "sfm_matching_max_num_matches", 32768)))
                )

            device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

            pycolmap.extract_features(
                str(db_path),
                str(images_dir),
                image_names=names,
                camera_mode=pycolmap.CameraMode.SINGLE,
                camera_model=camera_model,
                reader_options=reader_options,
                extraction_options=extraction_options,
                device=device,
            )

            if matcher == "sequential":
                pairing_options = pycolmap.SequentialPairingOptions()
                pairing_options.overlap = int(getattr(cfg, "sfm_sequential_overlap", 10))
                # Loop detection for video walk-around captures
                if getattr(cfg, "sfm_enable_loop_detection", True):
                    vocab_tree = _get_vocab_tree_path()
                    if vocab_tree is not None:
                        pairing_options.loop_detection = True
                        pairing_options.loop_detection_period = int(
                            getattr(cfg, "sfm_loop_detection_period", 10))
                        pairing_options.loop_detection_num_images = int(
                            getattr(cfg, "sfm_loop_detection_num_images", 50))
                        pairing_options.vocab_tree_path = str(vocab_tree)
                    else:
                        pairing_options.loop_detection = False
                else:
                    pairing_options.loop_detection = False
                pycolmap.match_sequential(
                    str(db_path),
                    matching_options=matching_options,
                    pairing_options=pairing_options,
                    device=device,
                )
            else:
                pairing_options = pycolmap.ExhaustivePairingOptions()
                pycolmap.match_exhaustive(
                    str(db_path),
                    matching_options=matching_options,
                    pairing_options=pairing_options,
                    device=device,
                )

            options = pycolmap.IncrementalPipelineOptions()
            options.ba_refine_focal_length = False
            options.ba_refine_principal_point = False
            options.ba_refine_extra_params = False
            options.ba_use_gpu = use_gpu
            if hasattr(options, "min_num_matches"):
                options.min_num_matches = int(
                    max(4, int(getattr(cfg, "sfm_mapper_min_num_matches", 10)))
                )
            if hasattr(options, "init_min_num_inliers"):
                options.init_min_num_inliers = int(
                    max(8, int(getattr(cfg, "sfm_mapper_init_min_num_inliers", 24)))
                )
            if hasattr(options, "abs_pose_min_num_inliers"):
                options.abs_pose_min_num_inliers = int(
                    max(6, int(getattr(cfg, "sfm_mapper_abs_pose_min_num_inliers", 18)))
                )
            if hasattr(options, "abs_pose_min_inlier_ratio"):
                options.abs_pose_min_inlier_ratio = float(
                    max(0.01, float(getattr(cfg, "sfm_mapper_abs_pose_min_inlier_ratio", 0.1)))
                )
            if hasattr(options, "random_seed"):
                options.random_seed = int(getattr(cfg, "random_seed", 42))

            reconstructions = pycolmap.incremental_mapping(
                str(db_path),
                str(images_dir),
                str(sparse_dir),
                options,
            )
            if not reconstructions:
                attempt_errors.append(f"{camera_model}/{matcher}:mapping_empty")
                continue

            best = None
            best_n = -1
            for rec in reconstructions.values():
                try:
                    n = int(rec.num_reg_images())
                except Exception:
                    n = 0
                if n > best_n:
                    best_n = n
                    best = rec

            if best is None:
                attempt_errors.append(f"{camera_model}/{matcher}:no_best_model")
                continue

            poses = _poses_from_pycolmap_reconstruction(best)
            if not poses:
                attempt_errors.append(f"{camera_model}/{matcher}:no_registered_poses")
                continue
            return True, poses, ""
        except Exception as e:
            attempt_errors.append(f"{camera_model}/{matcher}:failed:{e}")
            continue

    if not attempt_errors:
        return False, {}, "pycolmap_failed"
    return False, {}, ("pycolmap_failed_attempts:" + ";".join(attempt_errors))[:1500]


def _run_sfm_recovery_pass(
    colmap_cmd: str,
    db_path: Path,
    images_dir: Path,
    sparse_dir: Path,
    sparse_txt: Path,
    existing_poses: Dict[str, "np.ndarray"],
    all_names: List[str],
    cfg,
) -> Tuple[int, Dict[str, "np.ndarray"]]:
    """Run COLMAP image_registrator to recover unregistered frames.

    Takes the existing reconstruction and tries to register additional images
    with relaxed absolute-pose thresholds.

    Returns (newly_registered_count, updated_poses_dict).
    """
    if np is None:
        return 0, existing_poses

    n_unregistered = len(all_names) - len(existing_poses)
    min_unreg = int(getattr(cfg, "sfm_recovery_min_unregistered", 3))
    if n_unregistered < min_unreg:
        return 0, existing_poses

    recovery_inliers = int(getattr(cfg, "sfm_recovery_abs_pose_min_num_inliers", 12))
    recovery_ratio = float(getattr(cfg, "sfm_recovery_abs_pose_min_inlier_ratio", 0.06))
    recovery_reproj = float(getattr(cfg, "sfm_recovery_max_reproj_error_px", 2.0))

    # Use the best model from the sparse directory as input
    best_model = _choose_best_model_dir(sparse_dir)
    if best_model is None:
        logger.info("SfM recovery: no existing model found in sparse_dir")
        return 0, existing_poses

    recovery_output = sparse_dir / "recovery"
    shutil.rmtree(recovery_output, ignore_errors=True)
    recovery_output.mkdir(parents=True, exist_ok=True)

    ok, msg = _run_colmap_cmd([
        colmap_cmd, "image_registrator",
        "--database_path", str(db_path),
        "--input_path", str(best_model),
        "--output_path", str(recovery_output),
        "--Mapper.abs_pose_min_num_inliers", str(max(4, recovery_inliers)),
        "--Mapper.abs_pose_min_inlier_ratio", str(max(0.01, recovery_ratio)),
        "--Mapper.abs_pose_max_error", str(max(0.5, recovery_reproj)),
    ])
    if not ok:
        logger.warning("SfM recovery: image_registrator failed: %s", msg[:300])
        return 0, existing_poses

    # Convert recovery output to text for parsing
    recovery_txt = sparse_dir / "recovery_txt"
    shutil.rmtree(recovery_txt, ignore_errors=True)
    recovery_txt.mkdir(parents=True, exist_ok=True)
    ok, msg = _run_colmap_cmd([
        colmap_cmd, "model_converter",
        "--input_path", str(recovery_output),
        "--output_path", str(recovery_txt),
        "--output_type", "TXT",
    ])
    if not ok:
        logger.warning("SfM recovery: model_converter failed: %s", msg[:300])
        return 0, existing_poses

    recovered_poses = _parse_colmap_images_txt(recovery_txt / "images.txt")
    if not recovered_poses:
        return 0, existing_poses

    # Merge: count genuinely new registrations
    new_count = 0
    merged = dict(existing_poses)
    for name, pose in recovered_poses.items():
        if name not in merged:
            merged[name] = pose
            new_count += 1
        else:
            # Update existing with potentially refined pose
            merged[name] = pose

    logger.info(
        "SfM recovery pass: %d newly registered frames (total %d/%d)",
        new_count, len(merged), len(all_names),
    )
    return new_count, merged


def run_sfm_bundle_adjustment(
    frames,
    shots: List[ShotMetadata],
    cfg,
) -> Tuple[List[ShotMetadata], Dict]:
    """Run optional COLMAP SfM + BA and return refined shot poses."""
    refined = [copy.deepcopy(s) for s in shots]

    diag: Dict = {
        "sfm_enabled": bool(getattr(cfg, "enable_sfm_ba", False)),
        "sfm_attempted": False,
        "sfm_applied": False,
        "backend": None,
        "frames_total": int(len(frames)),
        "images_written": 0,
        "registered_images": 0,
        "aligned_poses": 0,
        "registration_ratio": 0.0,
        "similarity_scale": None,
        "alignment_rmse_m": None,
        "workspace": None,
        "fallback_reason": None,
    }

    if np is None:
        diag["fallback_reason"] = "numpy_missing"
        return refined, diag

    if not bool(getattr(cfg, "enable_sfm_ba", False)):
        diag["fallback_reason"] = "sfm_disabled"
        return refined, diag

    min_frames = max(4, int(getattr(cfg, "sfm_min_frames", 12)))
    if len(frames) < min_frames:
        diag["fallback_reason"] = f"insufficient_frames(n={len(frames)},min={min_frames})"
        return refined, diag

    colmap_cmd = str(getattr(cfg, "sfm_colmap_binary", "colmap"))
    has_colmap_cli = shutil.which(colmap_cmd) is not None
    has_pycolmap = pycolmap is not None and bool(getattr(cfg, "enable_pycolmap_backend", True))
    prefer_cli = bool(getattr(cfg, "sfm_prefer_colmap_cli", True))

    if not has_colmap_cli and not has_pycolmap:
        if pycolmap is None:
            diag["fallback_reason"] = (
                f"colmap_not_found({colmap_cmd});pycolmap_not_installed"
            )
        else:
            diag["fallback_reason"] = (
                f"colmap_not_found({colmap_cmd});pycolmap_disabled"
            )
        return refined, diag

    diag["sfm_attempted"] = True

    workspace_root = Path(str(getattr(cfg, "sfm_workspace_root", "/tmp/decroom_sfm")))
    workspace_root.mkdir(parents=True, exist_ok=True)
    keep_workspace = bool(getattr(cfg, "sfm_keep_workspace", False))

    ts = int(time.time())
    work_dir = Path(tempfile.mkdtemp(prefix=f"run_{ts}_", dir=str(workspace_root)))
    diag["workspace"] = str(work_dir)

    images_dir = work_dir / "images"
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_txt = work_dir / "sparse_txt"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    try:
        names_all = _materialize_frames_for_sfm(
            frames,
            images_dir,
            max_side=max(640, int(getattr(cfg, "sfm_max_image_size", 1600))),
            cfg=cfg,
        )
        diag["images_written"] = len(names_all)
        if len(names_all) < min_frames:
            diag["fallback_reason"] = (
                f"insufficient_materialized_images(n={len(names_all)},min={min_frames})"
            )
            return refined, diag

        selected_names, kf_diag = _select_sfm_keyframes(names_all, images_dir, shots, cfg)
        diag.update(kf_diag)
        if len(selected_names) < 4:
            selected_names = list(names_all)
            diag["keyframe_mode"] = "fallback_full_frames_insufficient_selected"
            diag["keyframe_selected_frames"] = int(len(selected_names))
        graph_names, graph_diag = _prefilter_sfm_match_graph(images_dir, selected_names, cfg, shots=shots)
        diag["sfm_graph"] = graph_diag
        diag["sfm_graph_component_size"] = int(graph_diag.get("largest_component_size", 0))
        if len(graph_names) >= 4:
            selected_names = graph_names
            diag["keyframe_mode"] = f"{diag.get('keyframe_mode', 'unknown')}+graph_prefilter"
        else:
            diag["keyframe_mode"] = f"{diag.get('keyframe_mode', 'unknown')}+graph_kept"
        diag["sfm_images_used"] = int(len(selected_names))

        def _write_image_list(name_list: List[str]) -> None:
            p = work_dir / "image_list.txt"
            with open(p, "w") as f:
                for nm in name_list:
                    f.write(f"{nm}\n")

        sfm_poses: Dict[str, "np.ndarray"] = {}
        backend_errors: List[str] = []
        used_name_list: List[str] = list(selected_names)
        required_ratio = float(getattr(cfg, "sfm_required_registration_ratio", 0.25))
        required_aligned = int(getattr(cfg, "sfm_required_aligned_poses", 4))
        partial_enabled = bool(getattr(cfg, "sfm_partial_apply_enabled", True))
        partial_min_registered = int(getattr(cfg, "sfm_partial_apply_min_registered", 4))
        partial_min_aligned = int(getattr(cfg, "sfm_partial_apply_min_aligned_poses", 3))
        partial_max_rmse = float(getattr(cfg, "sfm_partial_apply_max_rmse_m", 0.22))
        full_apply_max_rmse = float(getattr(cfg, "sfm_apply_max_rmse_m", 0.35))
        diag["required_ratio"] = round(float(required_ratio), 4)
        diag["required_aligned_poses"] = int(required_aligned)
        diag["partial_apply_enabled"] = bool(partial_enabled)
        diag["full_apply_max_rmse_m"] = round(float(full_apply_max_rmse), 4)

        backends: List[str]
        if prefer_cli:
            backends = ["colmap_cli", "pycolmap"]
        else:
            backends = ["pycolmap", "colmap_cli"]

        attempt_sets: List[Tuple[str, List[str], int]] = [
            ("prefilter", selected_names, 4)
        ]
        if selected_names != names_all:
            attempt_sets.append(("full", list(names_all), min_frames))
        candidate_runs: List[Dict[str, object]] = []

        def _candidate_alignment_stats(
            poses: Dict[str, "np.ndarray"],
        ) -> Tuple[int, float, Optional[Tuple[float, "np.ndarray", "np.ndarray"]]]:
            sfm_centers: List["np.ndarray"] = []
            arkit_centers: List["np.ndarray"] = []
            for i, shot in enumerate(shots):
                name = f"frame_{i:04d}.jpg"
                mat_sfm = poses.get(name)
                if mat_sfm is None:
                    continue
                if shot.pose is None or len(shot.pose) != 16:
                    continue
                try:
                    mat_arkit = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
                except Exception:
                    continue
                sfm_centers.append(mat_sfm[:3, 3])
                arkit_centers.append(mat_arkit[:3, 3])
            if len(sfm_centers) < 3:
                return len(sfm_centers), float("inf"), None
            src = np.array(sfm_centers, dtype=np.float64)
            dst = np.array(arkit_centers, dtype=np.float64)
            sim = _estimate_similarity_umeyama(src, dst)
            if sim is None:
                return len(sfm_centers), float("inf"), None
            s, R, t = sim
            aligned = (s * (R @ src.T)).T + t
            rmse = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
            return len(sfm_centers), rmse, sim

        for set_tag, names, min_req in attempt_sets:
            if len(names) < min_req:
                continue
            set_has_good_ratio = False
            for backend in backends:
                if backend == "colmap_cli" and not has_colmap_cli:
                    continue
                if backend == "pycolmap" and not has_pycolmap:
                    continue

                if backend == "colmap_cli":
                    _write_image_list(names)
                    ok, poses, msg = _run_colmap_cli_pipeline(
                        colmap_cmd=colmap_cmd,
                        db_path=db_path,
                        images_dir=images_dir,
                        sparse_dir=sparse_dir,
                        sparse_txt=sparse_txt,
                        names=names,
                        cfg=cfg,
                    )
                    if not ok:
                        backend_errors.append(f"{set_tag}:colmap:{msg}")
                        continue
                    ratio = float(len(poses)) / max(1, len(names))
                    aligned_pairs, rmse_est, sim_est = _candidate_alignment_stats(poses)
                    candidate_runs.append({
                        "backend": "colmap_cli",
                        "set_tag": set_tag,
                        "names": list(names),
                        "poses": poses,
                        "registered_images": int(len(poses)),
                        "images_used": int(len(names)),
                        "registration_ratio": round(ratio, 4),
                        "aligned_pairs": int(aligned_pairs),
                        "alignment_rmse_m_est": (
                            round(float(rmse_est), 4) if np.isfinite(rmse_est) else None
                        ),
                        "sim_est": sim_est,
                    })
                    if ratio >= required_ratio:
                        set_has_good_ratio = True
                        break

                if backend == "pycolmap":
                    ok, poses, msg = _run_pycolmap_pipeline(
                        db_path=db_path,
                        images_dir=images_dir,
                        sparse_dir=sparse_dir,
                        names=names,
                        cfg=cfg,
                    )
                    if not ok:
                        backend_errors.append(f"{set_tag}:pycolmap:{msg or 'failed'}")
                        continue
                    ratio = float(len(poses)) / max(1, len(names))
                    aligned_pairs, rmse_est, sim_est = _candidate_alignment_stats(poses)
                    candidate_runs.append({
                        "backend": "pycolmap",
                        "set_tag": set_tag,
                        "names": list(names),
                        "poses": poses,
                        "registered_images": int(len(poses)),
                        "images_used": int(len(names)),
                        "registration_ratio": round(ratio, 4),
                        "aligned_pairs": int(aligned_pairs),
                        "alignment_rmse_m_est": (
                            round(float(rmse_est), 4) if np.isfinite(rmse_est) else None
                        ),
                        "sim_est": sim_est,
                    })
                    if ratio >= required_ratio:
                        set_has_good_ratio = True
                        break
            if set_has_good_ratio:
                break

        diag["backend_attempts"] = [
            {
                "backend": c.get("backend"),
                "set_tag": c.get("set_tag"),
                "registered_images": c.get("registered_images"),
                "images_used": c.get("images_used"),
                "registration_ratio": c.get("registration_ratio"),
                "aligned_pairs": c.get("aligned_pairs"),
                "alignment_rmse_m_est": c.get("alignment_rmse_m_est"),
            }
            for c in candidate_runs
        ]

        if not candidate_runs:
            diag["fallback_reason"] = (
                ";".join(backend_errors) if backend_errors else "sfm_backend_failed"
            )[:1600]
            return refined, diag

        def _candidate_key(c: Dict[str, object]) -> Tuple[float, int, float, int]:
            ratio = float(c.get("registration_ratio", 0.0))
            aligned_pairs = int(c.get("aligned_pairs", 0))
            rmse_est = c.get("alignment_rmse_m_est")
            inv_rmse = 0.0
            if isinstance(rmse_est, (float, int)) and float(rmse_est) > 0.0:
                inv_rmse = 1.0 / float(rmse_est)
            set_bonus = 1 if str(c.get("set_tag", "")) == "full" else 0
            return (ratio, aligned_pairs, inv_rmse, set_bonus)

        best = max(candidate_runs, key=_candidate_key)
        sfm_poses = dict(best["poses"])  # type: ignore[arg-type]
        used_name_list = list(best["names"])  # type: ignore[arg-type]
        diag["backend"] = str(best["backend"])
        diag["keyframe_mode"] = f"{diag.get('keyframe_mode', 'unknown')}->{best.get('set_tag')}"

        diag["registered_images"] = len(sfm_poses)
        diag["sfm_images_used"] = int(len(used_name_list))
        diag["registration_ratio"] = round(len(sfm_poses) / max(1, len(used_name_list)), 4)
        used_idx: List[int] = []
        for nm in used_name_list:
            idx = _frame_idx_from_name(nm)
            if idx is not None:
                used_idx.append(int(idx))
        diag["sfm_frames_used_indices"] = used_idx

        # Recovery pass: try to register additional frames if ratio is below trigger.
        recovery_trigger = float(getattr(cfg, "sfm_recovery_registration_ratio_trigger", 0.6))
        if (
            bool(getattr(cfg, "sfm_enable_recovery_pass", True))
            and has_colmap_cli
            and float(diag["registration_ratio"]) < recovery_trigger
        ):
            recovery_new, sfm_poses = _run_sfm_recovery_pass(
                colmap_cmd=colmap_cmd,
                db_path=db_path,
                images_dir=images_dir,
                sparse_dir=sparse_dir,
                sparse_txt=sparse_txt,
                existing_poses=sfm_poses,
                all_names=used_name_list,
                cfg=cfg,
            )
            if recovery_new > 0:
                diag["recovery_new_frames"] = int(recovery_new)
                diag["registered_images"] = len(sfm_poses)
                diag["registration_ratio"] = round(
                    len(sfm_poses) / max(1, len(used_name_list)), 4
                )
                logger.info(
                    "SfM recovery: ratio improved to %.3f (%d/%d)",
                    diag["registration_ratio"],
                    len(sfm_poses),
                    len(used_name_list),
                )

        # Build ARKit <-> SfM camera center correspondences for similarity alignment.
        sfm_centers = []
        arkit_centers = []
        matched_indices = []
        for i, shot in enumerate(shots):
            name = f"frame_{i:04d}.jpg"
            mat_sfm = sfm_poses.get(name)
            if mat_sfm is None:
                continue
            if shot.pose is None or len(shot.pose) != 16:
                continue
            try:
                mat_arkit = np.array(shot.pose, dtype=np.float64).reshape(4, 4)
            except Exception:
                continue
            sfm_centers.append(mat_sfm[:3, 3])
            arkit_centers.append(mat_arkit[:3, 3])
            matched_indices.append(i)

        if len(sfm_centers) < 3:
            diag["fallback_reason"] = f"insufficient_alignment_pairs(n={len(sfm_centers)})"
            return refined, diag

        src = np.array(sfm_centers, dtype=np.float64)
        dst = np.array(arkit_centers, dtype=np.float64)
        sim = _estimate_similarity_umeyama(src, dst)
        if sim is None:
            diag["fallback_reason"] = "similarity_estimation_failed"
            return refined, diag

        s, R, t = sim
        aligned = (s * (R @ src.T)).T + t
        rmse = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
        diag["alignment_rmse_m"] = round(rmse, 4)
        diag["similarity_scale"] = round(float(s), 6)
        ratio_ok = float(diag["registration_ratio"]) >= required_ratio
        aligned_ok = len(matched_indices) >= required_aligned
        partial_ok = (
            partial_enabled
            and int(diag.get("registered_images", 0)) >= partial_min_registered
            and len(matched_indices) >= partial_min_aligned
            and rmse <= partial_max_rmse
        )
        full_rmse_ok = rmse <= full_apply_max_rmse
        if not ratio_ok and not partial_ok:
            diag["fallback_reason"] = (
                f"registration_ratio_low({diag['registration_ratio']:.2f}<{required_ratio:.2f})"
            )
            diag["sfm_apply_quality_gate"] = {
                "pass": False,
                "reason": diag["fallback_reason"],
            }
            return refined, diag

        if not aligned_ok and not partial_ok:
            diag["fallback_reason"] = (
                f"aligned_pose_count_low(n={len(matched_indices)},min={required_aligned})"
            )
            diag["sfm_apply_quality_gate"] = {
                "pass": False,
                "reason": diag["fallback_reason"],
            }
            return refined, diag
        if not partial_ok and not full_rmse_ok:
            diag["fallback_reason"] = (
                f"apply_rmse_high(rmse={rmse:.4f},max={full_apply_max_rmse:.4f})"
            )
            diag["sfm_apply_quality_gate"] = {
                "pass": False,
                "reason": diag["fallback_reason"],
            }
            return refined, diag
        diag["partial_apply_used"] = bool(partial_ok and (not ratio_ok or not aligned_ok))
        if diag["partial_apply_used"]:
            diag["partial_apply_reason"] = (
                f"registered={int(diag.get('registered_images', 0))},"
                f"aligned={len(matched_indices)},rmse={rmse:.4f}"
            )
        diag["sfm_apply_quality_gate"] = {
            "pass": True,
            "reason": "partial_apply" if diag.get("partial_apply_used") else "full_apply",
            "rmse_m": round(rmse, 4),
            "registration_ratio": diag.get("registration_ratio"),
            "aligned_poses": int(len(matched_indices)),
        }

        # Apply aligned SfM poses back to shots.
        applied = 0
        for i in range(len(refined)):
            name = f"frame_{i:04d}.jpg"
            mat_sfm = sfm_poses.get(name)
            if mat_sfm is None:
                continue

            mat_new = np.eye(4, dtype=np.float64)
            R_sfm = mat_sfm[:3, :3]
            t_sfm = mat_sfm[:3, 3]

            R_new = R @ R_sfm
            # Re-orthonormalize after multiplication to limit drift.
            U, _, Vt = np.linalg.svd(R_new)
            R_new = U @ Vt
            if np.linalg.det(R_new) < 0:
                U[:, -1] *= -1
                R_new = U @ Vt

            t_new = s * (R @ t_sfm) + t

            mat_new[:3, :3] = R_new
            mat_new[:3, 3] = t_new
            refined[i].pose = mat_new.reshape(-1).tolist()
            applied += 1

        diag["aligned_poses"] = int(applied)
        diag["sfm_applied"] = applied > 0
        if not diag["sfm_applied"]:
            diag["fallback_reason"] = "no_pose_applied"

        return refined, diag

    finally:
        if not keep_workspace:
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
