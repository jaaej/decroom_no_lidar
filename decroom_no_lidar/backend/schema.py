"""
Data models (Pydantic-free) for the Decroom backend.
Matches the RoomPayload JSON schema consumed by the iOS app.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ObjectPosition:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class ObjectMetadata:
    width: float = 1.0
    length: float = 1.0
    height: float = 1.0
    position_uncertainty_m: Optional[float] = None
    orientation_uncertainty_deg: Optional[float] = None
    support_views: Optional[int] = None
    inferred: Optional[bool] = None
    suppressed_reason: Optional[str] = None


@dataclass
class RoomObject:
    id: str = ""
    type: str = ""
    asset_id: str = ""
    position: ObjectPosition = field(default_factory=ObjectPosition)
    rotation_y: float = 0.0
    scale: float = 1.0
    confidence: float = 0.0
    metadata: ObjectMetadata = field(default_factory=ObjectMetadata)

    def to_dict(self) -> dict:
        metadata = {
            "width": self.metadata.width,
            "length": self.metadata.length,
            "height": self.metadata.height,
        }
        if self.metadata.position_uncertainty_m is not None:
            metadata["position_uncertainty_m"] = self.metadata.position_uncertainty_m
        if self.metadata.orientation_uncertainty_deg is not None:
            metadata["orientation_uncertainty_deg"] = self.metadata.orientation_uncertainty_deg
        if self.metadata.support_views is not None:
            metadata["support_views"] = self.metadata.support_views
        if self.metadata.inferred is not None:
            metadata["inferred"] = self.metadata.inferred
        if self.metadata.suppressed_reason is not None:
            metadata["suppressed_reason"] = self.metadata.suppressed_reason
        return {
            "id": self.id,
            "type": self.type,
            "asset_id": self.asset_id,
            "position": {"x": self.position.x, "y": self.position.y, "z": self.position.z},
            "rotation_y": self.rotation_y,
            "scale": self.scale,
            "confidence": self.confidence,
            "metadata": metadata,
        }


@dataclass
class Wall:
    id: str = ""
    start: List[float] = field(default_factory=lambda: [0.0, 0.0])
    end: List[float] = field(default_factory=lambda: [0.0, 0.0])
    has_window: bool = False
    has_door: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "has_window": self.has_window,
            "has_door": self.has_door,
        }


@dataclass
class RoomInfo:
    id: str = ""
    width_m: float = 4.0
    length_m: float = 4.0
    height_m: float = 2.5
    walls: List[Wall] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "width_m": self.width_m,
            "length_m": self.length_m,
            "height_m": self.height_m,
            "walls": [w.to_dict() for w in self.walls],
        }


@dataclass
class RoomPayload:
    room: RoomInfo = field(default_factory=RoomInfo)
    objects: List[RoomObject] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "room": self.room.to_dict(),
            "objects": [o.to_dict() for o in self.objects],
        }
