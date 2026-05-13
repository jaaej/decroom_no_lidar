"""Tests for the Grounding DINO HTTP client (Plan 3c Task 4)."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeline.rgb_detection import (
    KeyframeDetection,
    detect_furniture_in_keyframes,
)

FIXTURE = Path(__file__).parent / "fixtures" / "grounding_dino_response.json"


def _fake_post(*args, **kwargs):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json.loads(FIXTURE.read_text())
    return response


def test_detect_returns_detections_per_frame():
    jpegs = [b"FAKE_JPEG_1", b"FAKE_JPEG_2"]
    with patch("pipeline.rgb_detection.requests.post", side_effect=_fake_post):
        detections = detect_furniture_in_keyframes(
            jpegs, prompts=["bed", "desk"],
            endpoint="http://localhost:8100/detect_furniture",
        )
    assert all(isinstance(d, KeyframeDetection) for d in detections)
    for d in detections:
        assert 0 <= d.frame_index < 2
        assert d.confidence >= 0.0
        x, y, w, h = d.bbox_normalized
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        assert w > 0 and h > 0


def test_detect_handles_empty_jpeg_list():
    with patch("pipeline.rgb_detection.requests.post") as mock_post:
        out = detect_furniture_in_keyframes(
            [], prompts=["bed"],
            endpoint="http://localhost:8100/detect_furniture",
        )
    assert out == []
    mock_post.assert_not_called()


def test_detect_handles_503_gracefully():
    response = MagicMock()
    response.status_code = 503
    response.json.return_value = {"error": "gdino not loaded"}
    with patch("pipeline.rgb_detection.requests.post", return_value=response):
        out = detect_furniture_in_keyframes(
            [b"fake"], prompts=["bed"],
            endpoint="http://localhost:8100/detect_furniture",
        )
    assert out == []


def test_detect_splits_composite_labels():
    """A0 gate found Grounding DINO sometimes returns labels like
    'wardrobe bookshelf' or 'bookshelf dresser' that span two prompt
    categories. The client must split these into separate KeyframeDetection
    entries (one per matched whitelist category) so downstream clustering
    treats them correctly."""
    composite_response = {
        "frames": [{
            "frame_index": 0,
            "detections": [
                {"category": "wardrobe bookshelf",
                 "bbox_normalized": [0.1, 0.2, 0.3, 0.4],
                 "confidence": 0.55},
                {"category": "bed",
                 "bbox_normalized": [0.5, 0.5, 0.2, 0.2],
                 "confidence": 0.8},
            ],
        }],
        "elapsed_s": 0.5,
    }
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = composite_response
    with patch("pipeline.rgb_detection.requests.post", return_value=response):
        out = detect_furniture_in_keyframes(
            [b"fake"], prompts=["bed", "wardrobe", "bookshelf"],
            endpoint="http://localhost:8100/detect_furniture",
        )
    cats = sorted(d.category for d in out)
    assert "bed" in cats
    assert "wardrobe" in cats
    assert "bookshelf" in cats
    # The composite-derived ones share bbox + confidence.
    wardrobe = [d for d in out if d.category == "wardrobe"][0]
    bookshelf = [d for d in out if d.category == "bookshelf"][0]
    assert wardrobe.bbox_normalized == bookshelf.bbox_normalized
    assert wardrobe.confidence == bookshelf.confidence


def test_detect_drops_unrecognizable_labels():
    """Malformed labels like 'wardrobehel' (truncated wardrobe) shouldn't
    pass through. Drop anything that doesn't match a prompt category."""
    response_body = {
        "frames": [{
            "frame_index": 0,
            "detections": [
                {"category": "wardrobehel",
                 "bbox_normalized": [0.1, 0.2, 0.3, 0.4],
                 "confidence": 0.5},
                {"category": "bed",
                 "bbox_normalized": [0.5, 0.5, 0.2, 0.2],
                 "confidence": 0.8},
            ],
        }],
        "elapsed_s": 0.5,
    }
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = response_body
    with patch("pipeline.rgb_detection.requests.post", return_value=response):
        out = detect_furniture_in_keyframes(
            [b"fake"], prompts=["bed", "wardrobe"],
            endpoint="http://localhost:8100/detect_furniture",
        )
    cats = [d.category for d in out]
    assert cats == ["bed"]
