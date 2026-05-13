"""Tests for cross-keyframe NN-merge of lifted detections (Plan 3c Task 9)."""
import pytest

from pipeline.keyframe_clustering import (
    BACKEND_TRACKER_DISTANCES,
    cluster_keyframe_detections,
)
from pipeline.raycast_lift import LiftedDetection


def _det(frame, cat, pos, conf=0.8):
    return LiftedDetection(
        frame_index=frame, category=cat,
        bbox_normalized=(0.4, 0.4, 0.2, 0.2), confidence=conf,
        position_3d=pos,
    )


def test_same_category_near_positions_merge():
    dets = [
        _det(0, "bed", (2.0, 0.0, 1.5)),
        _det(1, "bed", (2.05, 0.0, 1.45)),
        _det(2, "bed", (1.95, 0.0, 1.55)),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 1
    bed = clusters[0]
    assert bed["category"] == "bed"
    assert bed["view_count"] == 3
    assert bed["position"][0] == pytest.approx(2.0, abs=0.1)


def test_same_category_far_positions_stay_separate():
    dets = [
        _det(0, "chair", (1.0, 0.0, 1.0)),
        _det(1, "chair", (3.5, 0.0, 1.0)),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 2


def test_different_categories_stay_separate_at_same_position():
    dets = [
        _det(0, "bed", (2.0, 0.0, 1.5)),
        _det(1, "desk", (2.0, 0.0, 1.5)),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 2


def test_drops_detections_with_no_3d_position():
    dets = [
        _det(0, "bed", None),
        _det(1, "bed", (2.0, 0.0, 1.5)),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 1
    assert clusters[0]["view_count"] == 1


def test_picks_highest_confidence_as_canonical():
    dets = [
        _det(0, "bed", (2.0, 0.0, 1.5), conf=0.5),
        _det(1, "bed", (2.05, 0.0, 1.55), conf=0.9),
        _det(2, "bed", (1.95, 0.0, 1.45), conf=0.6),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 1
    bed = clusters[0]
    assert bed["peak_confidence"] == pytest.approx(0.9)
    assert bed["best_frame_index"] == 1


def test_storage_family_visual_confusion_collapses():
    """A0 gate found wardrobe/dresser/bookshelf often label the same
    physical object across different keyframes (visual ambiguity).
    Within the storage_family resolution group, near positions should
    collapse to one cluster with the highest-confidence category as
    canonical."""
    dets = [
        _det(0, "wardrobe",  (2.0, 0.0, 1.5), conf=0.8),
        _det(1, "dresser",   (2.05, 0.0, 1.55), conf=0.5),
        _det(2, "bookshelf", (1.95, 0.0, 1.45), conf=0.4),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 1
    assert clusters[0]["category"] == "wardrobe"  # highest-confidence wins
    assert clusters[0]["view_count"] == 3


def test_default_threshold_for_unknown_category():
    dets = [
        _det(0, "alien_thing", (2.0, 0, 1.5)),
        _det(1, "alien_thing", (2.05, 0, 1.55)),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 1


def test_tv_and_monitor_canonicalize_and_merge():
    """Plan 3d: detector emits both 'tv' and 'monitor' for the same screen.
    Canonicalize tv → monitor before clustering so they collapse."""
    dets = [
        _det(0, "tv",      (2.0, 0.0, 1.5), conf=0.7),
        _det(1, "monitor", (2.05, 0.0, 1.45), conf=0.8),
        _det(2, "tv",      (1.95, 0.0, 1.55), conf=0.5),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 1
    assert clusters[0]["category"] == "monitor"
    assert clusters[0]["view_count"] == 3


def test_wardrobes_within_1_5m_merge():
    """Plan 3d: bumped wardrobe threshold from 0.6 to 1.5m to account for
    raycast hitting different surfaces of the same physical wardrobe
    across 66 keyframes."""
    dets = [
        _det(0, "wardrobe", (-0.61, -0.07, 0.75), conf=0.5),
        _det(1, "wardrobe", (0.12, -0.42, -0.47), conf=0.6),  # ~1.4m away
        _det(2, "wardrobe", (-0.57, 0.73, -1.52), conf=0.7),  # ~2.4m away — stays separate
    ]
    clusters = cluster_keyframe_detections(dets)
    # The first two are within 1.5m and merge; the third is beyond.
    cats = [c["category"] for c in clusters]
    assert cats == ["wardrobe", "wardrobe"]


def test_lamp_floor_canonicalizes_to_lamp():
    dets = [
        _det(0, "lamp_floor", (2.0, 0.0, 1.5)),
        _det(1, "lamp",       (2.05, 0.0, 1.55)),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 1
    assert clusters[0]["category"] == "lamp"
