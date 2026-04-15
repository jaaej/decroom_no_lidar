"""Camera intrinsics and shot metadata for the Decroom pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CameraIntrinsics:
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480

    def scaled_to(self, target_w: int, target_h: int) -> "CameraIntrinsics":
        sx = target_w / self.width
        sy = target_h / self.height
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=target_w,
            height=target_h,
        )


@dataclass
class ShotMetadata:
    shot_id: str = ""
    timestamp: float = 0.0
    intrinsics: Optional[CameraIntrinsics] = None
    pose: Optional[List[float]] = None
    gravity_w: Optional[List[float]] = None
    tracking_quality: str = "normal"
    feature_points_w: Optional[List[List[float]]] = None
    feature_points_uv: Optional[List[List[float]]] = None
    plane_anchors: Optional[List["PlaneAnchorMetadata"]] = None
    image_transform: Optional["ImageTransformMetadata"] = None
    quality: Optional["ShotQualityMetadata"] = None
    capture_step_m: Optional[float] = None
    capture_yaw_delta_deg: Optional[float] = None
    sequence_break_flag: Optional[bool] = None


@dataclass
class PlaneAnchorMetadata:
    anchor_id: str = ""
    alignment: str = "unknown"
    transform: Optional[List[float]] = None
    center: Optional[List[float]] = None
    extent: Optional[List[float]] = None


@dataclass
class ImageTransformMetadata:
    rotate_deg: int = 0
    flip_x: bool = False
    flip_y: bool = False
    crop: List[int] = None  # [x, y, w, h]
    resize: List[int] = None  # [W, H]
    letterbox_pad: List[int] = None  # [px, py]

    def __post_init__(self):
        if self.crop is None:
            self.crop = [0, 0, 0, 0]
        if self.resize is None:
            self.resize = [0, 0]
        if self.letterbox_pad is None:
            self.letterbox_pad = [0, 0]


@dataclass
class ShotQualityMetadata:
    texture_score: float = 0.0
    blur_score: float = 0.0
    edge_density: float = 0.0
    matchability_seed: int = 0
    overexposure_ratio: float = 0.0
    motion_overlap_score: Optional[float] = None


def parse_metadata_json(raw: dict) -> tuple[str, List[ShotMetadata]]:
    """Parse the metadata JSON from the iOS app into mode + shot list."""
    mode = raw.get("mode", "guided")
    shots_raw = raw.get("shots", [])
    shots = []
    for s in shots_raw:
        intr_raw = s.get("intrinsics")
        intr = None
        if intr_raw:
            intr = CameraIntrinsics(
                fx=float(intr_raw.get("fx", 500)),
                fy=float(intr_raw.get("fy", 500)),
                cx=float(intr_raw.get("cx", 320)),
                cy=float(intr_raw.get("cy", 240)),
                width=int(intr_raw.get("width", 640)),
                height=int(intr_raw.get("height", 480)),
            )
        pose_raw = s.get("pose", {})
        pose = pose_raw.get("matrix") if isinstance(pose_raw, dict) else None

        fp_w = s.get("feature_points_w")
        fp_uv = s.get("feature_points_uv")
        image_transform = None
        tf_raw = s.get("image_transform")
        if isinstance(tf_raw, dict):
            crop = tf_raw.get("crop")
            resize = tf_raw.get("resize")
            letterbox = tf_raw.get("letterbox_pad")
            if not (isinstance(crop, list) and len(crop) == 4):
                crop = [0, 0, 0, 0]
            if not (isinstance(resize, list) and len(resize) == 2):
                resize = [0, 0]
            if not (isinstance(letterbox, list) and len(letterbox) == 2):
                letterbox = [0, 0]
            image_transform = ImageTransformMetadata(
                rotate_deg=int(tf_raw.get("rotate_deg", 0)),
                flip_x=bool(tf_raw.get("flip_x", False)),
                flip_y=bool(tf_raw.get("flip_y", False)),
                crop=[int(v) for v in crop],
                resize=[int(v) for v in resize],
                letterbox_pad=[int(v) for v in letterbox],
            )
        plane_anchors = None
        plane_raw = s.get("plane_anchors")
        if isinstance(plane_raw, list):
            parsed_planes: List[PlaneAnchorMetadata] = []
            for p in plane_raw:
                if not isinstance(p, dict):
                    continue
                transform = p.get("transform")
                center = p.get("center")
                extent = p.get("extent")
                if transform is not None and (not isinstance(transform, list) or len(transform) != 16):
                    transform = None
                if center is not None and (not isinstance(center, list) or len(center) != 3):
                    center = None
                if extent is not None and (not isinstance(extent, list) or len(extent) != 3):
                    extent = None
                parsed_planes.append(
                    PlaneAnchorMetadata(
                        anchor_id=str(p.get("anchor_id", "")),
                        alignment=str(p.get("alignment", "unknown")).lower(),
                        transform=transform,
                        center=center,
                        extent=extent,
                    )
                )
            if parsed_planes:
                plane_anchors = parsed_planes
        quality = None
        quality_raw = s.get("quality")
        if isinstance(quality_raw, dict):
            motion_overlap_score_raw = quality_raw.get("motion_overlap_score")
            try:
                motion_overlap_score = (
                    float(motion_overlap_score_raw)
                    if motion_overlap_score_raw is not None else None
                )
            except Exception:
                motion_overlap_score = None
            quality = ShotQualityMetadata(
                texture_score=float(quality_raw.get("texture_score", 0.0)),
                blur_score=float(quality_raw.get("blur_score", 0.0)),
                edge_density=float(quality_raw.get("edge_density", 0.0)),
                matchability_seed=int(quality_raw.get("matchability_seed", 0)),
                overexposure_ratio=float(quality_raw.get("overexposure_ratio", 0.0)),
                motion_overlap_score=motion_overlap_score,
            )
        capture_step_m = s.get("capture_step_m")
        capture_yaw_delta_deg = s.get("capture_yaw_delta_deg")
        try:
            capture_step_m = float(capture_step_m) if capture_step_m is not None else None
        except Exception:
            capture_step_m = None
        try:
            capture_yaw_delta_deg = (
                float(capture_yaw_delta_deg) if capture_yaw_delta_deg is not None else None
            )
        except Exception:
            capture_yaw_delta_deg = None
        # Backward/forward-compatible format:
        # feature_points: [{world:[x,y,z], uv:[u,v]}, ...]
        if (fp_w is None or fp_uv is None) and isinstance(s.get("feature_points"), list):
            fp_items = s.get("feature_points", [])
            worlds = []
            uvs = []
            for item in fp_items:
                if not isinstance(item, dict):
                    continue
                w = item.get("world")
                uv = item.get("uv")
                if isinstance(w, list) and len(w) == 3:
                    worlds.append(w)
                    if isinstance(uv, list) and len(uv) == 2:
                        uvs.append(uv)
                    else:
                        uvs.append([float("nan"), float("nan")])
            if fp_w is None:
                fp_w = worlds
            if fp_uv is None:
                fp_uv = uvs

        shots.append(ShotMetadata(
            shot_id=s.get("shot_id", ""),
            timestamp=float(s.get("timestamp", 0)),
            intrinsics=intr,
            pose=pose,
            gravity_w=s.get("gravity_w"),
            tracking_quality=s.get("tracking_quality", "normal"),
            feature_points_w=fp_w,
            feature_points_uv=fp_uv,
            plane_anchors=plane_anchors,
            image_transform=image_transform,
            quality=quality,
            capture_step_m=capture_step_m,
            capture_yaw_delta_deg=capture_yaw_delta_deg,
            sequence_break_flag=bool(s.get("sequence_break_flag", False)),
        ))
    # Keep shots in chronological order for sequential reconstruction logic.
    shots.sort(key=lambda x: x.timestamp)
    return mode, shots
