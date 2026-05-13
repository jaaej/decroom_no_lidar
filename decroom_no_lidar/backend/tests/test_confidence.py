"""Tests for compute_confidence."""
from pipeline.consensus import compute_confidence


def _obj(type_, width=1.6, length=2.0, height=0.6):
    return {"type": type_,
             "metadata": {"width": width, "length": length, "height": height}}


def test_full_agreement_multiview_plausible_yields_high_confidence():
    c = compute_confidence(_obj("bed"),
                             agreement_score=1.0,
                             yolo_view_count=5,
                             yolo_matched=True)
    # 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 1.0
    assert c == 1.0


def test_slm_only_neutral_view_plausible_mid_confidence():
    c = compute_confidence(_obj("bed"),
                             agreement_score=0.5,
                             yolo_view_count=0,
                             yolo_matched=False)
    # 0.5*0.5 + 0.3*0.5 + 0.2*1.0 = 0.6
    assert abs(c - 0.6) < 1e-6


def test_duplicate_collapse_survivor_low_view_implausible_size_low_confidence():
    c = compute_confidence(_obj("bed", width=3.5, length=5.0, height=2.0),
                             agreement_score=0.3,
                             yolo_view_count=1,
                             yolo_matched=True)
    # 0.5*0.3 + 0.3*0.4 + 0.2*0.3 = 0.33
    assert abs(c - 0.33) < 1e-6


def test_confidence_clamped_to_01_range():
    c = compute_confidence(_obj("bed"),
                             agreement_score=2.0,  # out-of-range input
                             yolo_view_count=100,
                             yolo_matched=True)
    assert 0.0 <= c <= 1.0


def test_multi_view_score_scales_with_view_count_when_matched():
    low = compute_confidence(_obj("bed"),
                               agreement_score=1.0, yolo_view_count=1,
                               yolo_matched=True)
    mid = compute_confidence(_obj("bed"),
                               agreement_score=1.0, yolo_view_count=2,
                               yolo_matched=True)
    high = compute_confidence(_obj("bed"),
                               agreement_score=1.0, yolo_view_count=3,
                               yolo_matched=True)
    assert low < mid < high


def test_unknown_category_uses_neutral_size_score():
    """A category without a known size range gets a neutral-ish score, not 0."""
    c = compute_confidence({"type": "exotic_thing", "metadata": {}},
                             agreement_score=1.0, yolo_view_count=3,
                             yolo_matched=True)
    # 0.5*1.0 + 0.3*1.0 + 0.2*0.7 = 0.94
    assert 0.9 < c < 1.0
