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


# ── collapse_duplicates ──────────────────────────────────────────────────────

def test_collapse_duplicates_two_beds_close_collapses_to_one():
    from pipeline.consensus import collapse_duplicates
    objs = [
        {"id": "a", "type": "bed", "position": {"x": 2.0, "y": 0, "z": 1.5},
         "metadata": {}, "_confidence": 0.9},
        {"id": "b", "type": "bed", "position": {"x": 2.3, "y": 0, "z": 1.7},
         "metadata": {}, "_confidence": 0.7},
    ]
    out = collapse_duplicates(objs)
    assert len(out) == 1
    assert out[0]["id"] == "a"  # higher confidence wins


def test_collapse_duplicates_two_beds_far_apart_both_kept():
    from pipeline.consensus import collapse_duplicates
    objs = [
        {"id": "a", "type": "bed", "position": {"x": 2.0, "y": 0, "z": 1.5},
         "metadata": {}, "_confidence": 0.9},
        {"id": "b", "type": "bed", "position": {"x": 5.0, "y": 0, "z": 1.5},
         "metadata": {}, "_confidence": 0.8},
    ]
    out = collapse_duplicates(objs)
    assert len(out) == 2


def test_collapse_duplicates_preserves_different_categories():
    from pipeline.consensus import collapse_duplicates
    objs = [
        {"id": "a", "type": "bed", "position": {"x": 2.0, "y": 0, "z": 1.5},
         "metadata": {}, "_confidence": 0.9},
        {"id": "b", "type": "chair", "position": {"x": 2.0, "y": 0, "z": 1.5},
         "metadata": {}, "_confidence": 0.9},
    ]
    out = collapse_duplicates(objs)
    assert len(out) == 2  # different categories don't collapse


def test_collapse_duplicates_three_chairs_transitively_collapse():
    from pipeline.consensus import collapse_duplicates
    # chairs at (0,0), (0.4,0), (0.8,0) — pairwise within 0.65m
    objs = [
        {"id": "a", "type": "chair", "position": {"x": 0.0, "y": 0, "z": 0},
         "metadata": {}, "_confidence": 0.8},
        {"id": "b", "type": "chair", "position": {"x": 0.4, "y": 0, "z": 0},
         "metadata": {}, "_confidence": 0.9},
        {"id": "c", "type": "chair", "position": {"x": 0.8, "y": 0, "z": 0},
         "metadata": {}, "_confidence": 0.7},
    ]
    out = collapse_duplicates(objs)
    # Iterating until stable: should end at 1 chair (b, highest conf).
    assert len(out) == 1
    assert out[0]["id"] == "b"


def test_collapse_duplicates_empty_input_returns_empty():
    from pipeline.consensus import collapse_duplicates
    assert collapse_duplicates([]) == []


def test_collapse_duplicates_single_input_unchanged():
    from pipeline.consensus import collapse_duplicates
    o = [{"id": "x", "type": "bed", "position": {"x": 0, "y": 0, "z": 0},
           "metadata": {}, "_confidence": 0.5}]
    assert collapse_duplicates(o) == o


# ── YOLO input filter (Plan 3b post-QA hardening) ────────────────────────────

def _wall(x0, z0, x1, z1):
    return {"start": {"x": x0, "z": z0}, "end": {"x": x1, "z": z1}}


def test_consensus_drops_yolo_non_indoor_categories():
    walls = [_wall(0, 0, 5, 0), _wall(0, 0, 0, 5),
             _wall(5, 0, 5, 5), _wall(0, 5, 5, 5)]
    yolo = [
        _make_yolo("person", 2, 0, 2),
        _make_yolo("car", 3, 0, 3),
        _make_yolo("bed", 2, 0, 2),
    ]
    merged, _ = consensus_merge(yolo, [], walls=walls)
    cats = {o["type"] for o in merged}
    assert "person" not in cats
    assert "car" not in cats
    assert "bed" in cats


def test_consensus_drops_yolo_outside_room_bounds():
    walls = [_wall(0, 0, 5, 0), _wall(0, 0, 0, 5),
             _wall(5, 0, 5, 5), _wall(0, 5, 5, 5)]
    yolo = [
        _make_yolo("bed", 2, 0, 2),
        _make_yolo("bed", 10, 0, 10),
    ]
    merged, _ = consensus_merge(yolo, [], walls=walls)
    assert len(merged) == 1
    assert merged[0]["position"]["x"] == 2


def test_consensus_keeps_yolo_inside_margin_of_walls():
    walls = [_wall(0, 0, 5, 0), _wall(0, 0, 0, 5),
             _wall(5, 0, 5, 5), _wall(0, 5, 5, 5)]
    # 0.1m past z=0 wall — within the 0.3m margin -> keep.
    yolo = [_make_yolo("bed", 2, 0, -0.1)]
    merged, _ = consensus_merge(yolo, [], walls=walls)
    assert len(merged) == 1


# ── Asset-id mapping ─────────────────────────────────────────────────────────

def test_asset_id_maps_tv_to_kenney_monitor():
    from pipeline.consensus import asset_id_for
    # Closest bundled substitute: there's no kenney_tv.usdz.
    assert asset_id_for("tv") == "kenney_monitor"


def test_asset_id_maps_couch_to_kenney_sofa():
    from pipeline.consensus import asset_id_for
    assert asset_id_for("couch") == "kenney_sofa"


def test_asset_id_for_all_whitelist_categories_resolves_to_bundled_kenney():
    from pipeline.consensus import (
        INDOOR_CATEGORIES_WHITELIST, asset_id_for, CATEGORY_TO_ASSET,
    )
    bundled = {
        "bed", "bookshelf", "chair", "desk", "dresser",
        "lamp_desk", "lamp_floor", "monitor", "nightstand",
        "plant", "rug", "sofa", "table",
    }
    for cat in INDOOR_CATEGORIES_WHITELIST:
        asset = asset_id_for(cat)
        # Strip 'kenney_' prefix.
        name = asset.removeprefix("kenney_")
        assert name in bundled, \
            f"category {cat!r} -> {asset} but {name}.usdz isn't bundled"


def test_consensus_drops_laptop_no_asset_for_it():
    from pipeline.consensus import consensus_merge
    walls = [
        {"start": {"x": 0, "z": 0}, "end": {"x": 5, "z": 0}},
        {"start": {"x": 0, "z": 5}, "end": {"x": 5, "z": 5}},
        {"start": {"x": 0, "z": 0}, "end": {"x": 0, "z": 5}},
        {"start": {"x": 5, "z": 0}, "end": {"x": 5, "z": 5}},
    ]
    yolo = [_make_yolo("laptop", 2, 0, 2)]
    merged, _ = consensus_merge(yolo, [], walls=walls)
    # laptop has no Kenney asset; whitelist excludes it.
    assert merged == []
