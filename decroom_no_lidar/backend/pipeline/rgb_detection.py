"""Plan 3c: HTTP client for the GPU pod's /detect_furniture endpoint.

Pure data layer. No model code lives here; the GPU work happens on the pod.

Also handles two issues observed in Phase A0 gate:
1. Composite labels (e.g. "wardrobe bookshelf") get split into separate
   KeyframeDetection entries, one per matched whitelist category.
2. Malformed labels (e.g. "wardrobehel") are dropped — they pass no
   whitelist match.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import requests

logger = logging.getLogger("decroom-backend.rgb_detection")

DEFAULT_PROMPT_CATEGORIES = [
    "bed", "wardrobe", "bookshelf", "desk", "monitor", "chair",
    "nightstand", "dresser", "lamp", "potted plant", "sofa",
]

DEFAULT_ENDPOINT = "http://localhost:8100/detect_furniture"
DEFAULT_BOX_THRESHOLD = 0.4
DEFAULT_TEXT_THRESHOLD = 0.3
REQUEST_TIMEOUT_S = 60.0


@dataclass
class KeyframeDetection:
    """One Grounding DINO detection in one keyframe (post split + filter)."""
    frame_index: int
    category: str
    bbox_normalized: Tuple[float, float, float, float]  # (x, y, w, h) in [0,1]
    confidence: float


def _split_and_filter_label(
    raw_label: str,
    whitelist: List[str],
) -> List[str]:
    """Split a Grounding DINO label by whitelist matches.

    Examples:
      "wardrobe"            + [bed, wardrobe]            -> ["wardrobe"]
      "wardrobe bookshelf"  + [wardrobe, bookshelf]      -> ["wardrobe", "bookshelf"]
      "bookshelf dresser"   + [bookshelf, dresser]       -> ["bookshelf", "dresser"]
      "wardrobehel"         + [wardrobe]                 -> []  (no clean match)
      "potted plant"        + [potted plant]             -> ["potted plant"]
    """
    label = raw_label.lower().strip()
    if not label:
        return []
    # Exact match → trivial.
    if label in whitelist:
        return [label]
    # Whitespace-tokenize and find which whitelist categories appear as
    # full tokens. Handles "wardrobe bookshelf" → [wardrobe, bookshelf].
    tokens = label.split()
    matches = []
    # Multi-word categories first (e.g., "potted plant") so we don't double-count.
    sorted_cats = sorted(whitelist, key=lambda c: -len(c.split()))
    remaining = label
    for cat in sorted_cats:
        if cat in remaining and cat not in matches:
            matches.append(cat)
            remaining = remaining.replace(cat, "", 1)
    # Filter: only keep matches that appeared as standalone tokens (so we
    # don't match "wardrobehel" → "wardrobe"). Test by tokenizing the
    # original label and checking each match's token is present.
    valid = []
    for cat in matches:
        cat_tokens = cat.split()
        if all(t in tokens for t in cat_tokens):
            valid.append(cat)
    return valid


def detect_furniture_in_keyframes(
    keyframe_jpegs: List[bytes],
    prompts: List[str] = None,
    endpoint: str = DEFAULT_ENDPOINT,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> List[KeyframeDetection]:
    """Send keyframe JPEGs to the GPU pod's /detect_furniture endpoint.

    Returns one KeyframeDetection per accepted detection (after splitting
    composite labels and dropping malformed ones). Frames with no
    detections contribute zero entries. Returns [] on transport failure
    or server error — caller falls back to walls-only RoomPayload.
    """
    if not keyframe_jpegs:
        return []

    cats = prompts or DEFAULT_PROMPT_CATEGORIES
    prompt_text = ". ".join(cats) + "."

    files = [
        ("images", (f"frame_{i}.jpg", jpeg, "image/jpeg"))
        for i, jpeg in enumerate(keyframe_jpegs)
    ]
    data = {
        "prompt": prompt_text,
        "box_threshold": str(box_threshold),
        "text_threshold": str(text_threshold),
    }

    try:
        resp = requests.post(
            endpoint, files=files, data=data, timeout=REQUEST_TIMEOUT_S,
        )
    except requests.RequestException as e:
        logger.warning("rgb_detection transport error: %s", e)
        return []

    if resp.status_code != 200:
        try:
            err = resp.json().get("error", "")
        except Exception:
            err = "<no body>"
        logger.warning(
            "rgb_detection server returned %d: %s", resp.status_code, err,
        )
        return []

    body = resp.json()
    out: List[KeyframeDetection] = []
    n_raw = 0
    n_split_or_dropped = 0
    for frame in body.get("frames", []):
        fi = int(frame.get("frame_index", 0))
        for d in frame.get("detections", []):
            n_raw += 1
            bbox = d.get("bbox_normalized", [0, 0, 0, 0])
            confidence = float(d.get("confidence", 0.0))
            raw_label = str(d.get("category", ""))
            split_cats = _split_and_filter_label(raw_label, cats)
            if not split_cats:
                n_split_or_dropped += 1
                logger.debug(
                    "rgb_detection dropped unparseable label %r", raw_label,
                )
                continue
            if len(split_cats) > 1:
                n_split_or_dropped += 1
            for cat in split_cats:
                out.append(KeyframeDetection(
                    frame_index=fi,
                    category=cat,
                    bbox_normalized=(float(bbox[0]), float(bbox[1]),
                                      float(bbox[2]), float(bbox[3])),
                    confidence=confidence,
                ))
    logger.info(
        "rgb_detection: %d raw -> %d emitted (%d split/dropped) across %d frames (%.1fs server)",
        n_raw, len(out), n_split_or_dropped,
        len(body.get("frames", [])), body.get("elapsed_s", 0),
    )
    return out
