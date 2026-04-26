"""Tests for the Plan 3b consensus module."""
from pipeline.consensus import (
    consensus_merge,
    POSITION_MATCH_DISTANCE_LARGE,
    POSITION_MATCH_DISTANCE_SMALL,
)


def _make_yolo(category, x, y, z, conf=0.8, view_count=3):
    return {"category": category, "position": [x, y, z],
             "peak_confidence": conf, "view_count": view_count,
             "best_frame_index": 0}


def _make_slm(obj_type, x, y, z, width=1.6, length=2.0, height=0.6, rotation_y=0.0):
    return {"type": obj_type,
             "position": {"x": x, "y": y, "z": z},
             "rotation_y": rotation_y,
             "metadata": {"width": width, "length": length, "height": height}}


def test_yolo_and_slm_agree_on_bed_same_position_keeps_slm_full_agreement():
    yolo = [_make_yolo("bed", 2.0, 0.0, 1.5)]
    slm = [_make_slm("bed", 2.1, 0.0, 1.4)]
    merged, agreement = consensus_merge(yolo, slm, walls=[])
    assert len(merged) == 1
    assert merged[0]["type"] == "bed"
    # SLM position preserved (not moved to YOLO's)
    assert merged[0]["position"]["x"] == 2.1
    assert agreement[merged[0]["id"]] == 1.0


def test_yolo_only_bed_no_slm_keeps_yolo_position_half_agreement():
    yolo = [_make_yolo("bed", 2.0, 0.0, 1.5)]
    merged, agreement = consensus_merge(yolo, [], walls=[])
    assert len(merged) == 1
    assert merged[0]["type"] == "bed"
    assert merged[0]["position"]["x"] == 2.0
    assert agreement[merged[0]["id"]] == 0.5


def test_slm_only_bed_no_yolo_keeps_slm_half_agreement():
    slm = [_make_slm("bed", 2.0, 0.0, 1.5)]
    merged, agreement = consensus_merge([], slm, walls=[])
    assert len(merged) == 1
    assert agreement[merged[0]["id"]] == 0.5


def test_two_beds_far_apart_both_kept_full_agreement():
    yolo = [_make_yolo("bed", 2.0, 0.0, 1.5), _make_yolo("bed", 4.0, 0.0, 1.5)]
    slm = [_make_slm("bed", 2.0, 0.0, 1.5), _make_slm("bed", 4.0, 0.0, 1.5)]
    merged, agreement = consensus_merge(yolo, slm, walls=[])
    assert len(merged) == 2
    for o in merged:
        assert agreement[o["id"]] == 1.0


def test_slm_duplicates_one_yolo_collapses_to_one_low_agreement():
    yolo = [_make_yolo("bed", 2.0, 0.0, 1.5)]
    slm = [_make_slm("bed", 2.0, 0.0, 1.5), _make_slm("bed", 2.2, 0.0, 1.6)]
    merged, agreement = consensus_merge(yolo, slm, walls=[])
    assert len(merged) == 1
    assert agreement[merged[0]["id"]] == 0.3


def test_category_conflict_at_same_position_keeps_slm_low_agreement():
    yolo = [_make_yolo("bed", 2.0, 0.0, 1.5)]
    slm = [_make_slm("couch", 2.0, 0.0, 1.5, width=2.2, length=0.9)]
    merged, agreement = consensus_merge(yolo, slm, walls=[])
    assert len(merged) == 1
    # SLM category wins
    assert merged[0]["type"] == "couch"
    assert agreement[merged[0]["id"]] == 0.3


def test_position_match_distance_uses_small_for_chair():
    # 0.5m apart: LARGE distance (0.8m) would match; SMALL (0.4m) wouldn't.
    yolo = [_make_yolo("chair", 0.0, 0.0, 0.0)]
    slm = [_make_slm("chair", 0.5, 0.0, 0.0)]
    merged, agreement = consensus_merge(yolo, slm, walls=[])
    # Chair uses SMALL threshold → 0.5m > 0.4m → treated as NOT matched,
    # so both objects survive: 1 from YOLO (no match), 1 from SLM (no match).
    assert len(merged) == 2


def test_empty_inputs_returns_empty():
    merged, agreement = consensus_merge([], [], walls=[])
    assert merged == []
    assert agreement == {}


def test_match_distance_constants_exist():
    # Sanity check the public constants the test relies on.
    assert POSITION_MATCH_DISTANCE_LARGE > POSITION_MATCH_DISTANCE_SMALL
