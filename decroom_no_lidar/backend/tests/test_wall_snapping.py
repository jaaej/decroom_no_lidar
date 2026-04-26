"""Tests for snap_to_walls."""
import math

from pipeline.consensus import snap_to_walls


def _wall(x0, z0, x1, z1):
    return {"start": {"x": x0, "z": z0}, "end": {"x": x1, "z": z1}}


def _obj(type_, x, z, rotation_y=0.0):
    return {"id": type_[0] + "1", "type": type_,
             "position": {"x": x, "y": 0, "z": z},
             "rotation_y": rotation_y,
             "metadata": {"width": 1.6, "length": 2.0, "height": 0.6}}


def test_bed_within_tolerance_snaps_to_nearest_wall():
    walls = [_wall(0, 0, 5, 0)]
    objs = [_obj("bed", 2.0, 0.3)]
    snapped = snap_to_walls(objs, walls)
    assert len(snapped) == 1
    assert abs(snapped[0]["position"]["z"]) < 1e-6
    assert abs(snapped[0]["position"]["x"] - 2.0) < 1e-6  # along-wall unchanged


def test_bed_beyond_tolerance_stays_put():
    walls = [_wall(0, 0, 5, 0)]
    objs = [_obj("bed", 2.0, 3.0)]  # 3m from wall, tolerance 0.8m
    snapped = snap_to_walls(objs, walls)
    assert snapped[0]["position"]["z"] == 3.0


def test_chair_freestanding_does_not_snap():
    walls = [_wall(0, 0, 5, 0)]
    objs = [_obj("chair", 2.0, 0.2)]
    snapped = snap_to_walls(objs, walls)
    assert snapped[0]["position"]["z"] == 0.2


def test_nightstand_uses_tighter_tolerance():
    walls = [_wall(0, 0, 5, 0)]
    # Nightstand tolerance 0.5m: 0.7m away → no snap.
    far = _obj("nightstand", 2.0, 0.7)
    snapped = snap_to_walls([far], walls)
    assert snapped[0]["position"]["z"] == 0.7
    # Within 0.5m → snaps.
    near = _obj("nightstand", 2.0, 0.3)
    snapped2 = snap_to_walls([near], walls)
    assert abs(snapped2[0]["position"]["z"]) < 1e-6


def test_no_walls_no_snap():
    snapped = snap_to_walls([_obj("bed", 2.0, 0.3)], walls=[])
    assert snapped[0]["position"]["z"] == 0.3


def test_snap_sets_finite_rotation():
    walls = [_wall(0, 0, 5, 0)]
    snapped = snap_to_walls([_obj("bed", 2.0, 0.2)], walls)
    assert math.isfinite(snapped[0]["rotation_y"])


def test_snap_picks_nearest_of_two_walls():
    """North wall at z=0, South wall at z=5. Bed at (2, 0.4) → snap to north."""
    walls = [_wall(0, 0, 5, 0), _wall(0, 5, 5, 5)]
    snapped = snap_to_walls([_obj("bed", 2.0, 0.4)], walls)
    assert abs(snapped[0]["position"]["z"]) < 1e-6
