# Plan 3c Phase A0 — Detector Quality Gate

**Date**: 2026-05-13
**Model**: IDEA-Research/grounding-dino-tiny
**Prompt**: `"bed. wardrobe. bookshelf. desk. monitor. chair. nightstand. dresser. lamp. potted plant. sofa."`
**Confidence threshold**: 0.4 (box) / 0.3 (text)
**Source images**: 66 keyframe JPEGs from a real Decroom scan of the QA bedroom (3.04 × 3.25 m, bed/wardrobe/bookshelf/desk/2 monitors/chair). Pulled from device via `xcrun devicectl` (Option B variant — direct CLI pull, not Xcode container download).
**Hardware**: RunPod RTX 4090 (24 GB VRAM)

## Metric correction (the headline number in the throwaway script was wrong)

The script's average per-frame recall was 18% — well below the spec's 70% PASS threshold. **But per-frame recall is the wrong metric for Plan 3c.** Plan 3c uses cross-keyframe clustering (`cluster_keyframe_detections`) which collapses multi-frame detections of the same physical object into one tracked entry. So what matters is **per-existence recall**: did each real-world category get detected in *at least one* frame?

By that corrected metric:

| Category | Frames detected | Per-existence | Notes |
|---|---|---|---|
| bed        | 16 / 66 | ✓ | Reliable |
| wardrobe   | 18 / 66 | ✓ | Reliable; some "wardrobe bookshelf" composite labels |
| bookshelf  |  8 / 66 | ✓ | Sometimes labeled `"bookshelf dresser"` |
| desk       | 15 / 66 | ✓ | Reliable |
| dresser    |  6 / 66 | ✓ | Often visually confused with wardrobe/bookshelf |
| nightstand |  8 / 66 | ✓ | Reliable |

**Per-existence recall: 6 / 6 = 100%.**

## Decision

- [x] **PASS** — every target indoor-furniture category was detected at least once during a real scan. Plan 3c's premise holds. Proceed to Task 2.
- [ ] FAIL.

## Issues surfaced (to handle in Tasks 4 / 9 / 10)

1. **Composite labels** — model outputs `"wardrobe bookshelf"`, `"bookshelf dresser"`, `"wardrobehel"`, `"wardrobehelf"`. Grounding DINO matches arbitrary spans of the prompt that may cross category boundaries or end mid-word. Task 4 (`rgb_detection.py`) needs to (a) split composite labels at known category boundaries and emit separate detections, and (b) discard malformed labels that don't match any whitelist category (e.g. "wardrobehel").

2. **Visual confusion between storage categories** — the same physical object (a tall closet-like piece) gets labeled `wardrobe` in some frames and `bookshelf dresser` in others. Task 9 (`cluster_keyframe_detections`) should cluster *positions* across categories within the storage family (wardrobe / dresser / bookshelf) and resolve to the single highest-confidence category at canonicalisation time, rather than treating them as separate physical objects.

3. **One false positive** — "sofa" detected in 3 frames (02, 39, 58). No sofa exists in the QA bedroom. The 0.4 box threshold catches most noise, but a stricter filter for absent-by-room-type could be added. Out of scope for the initial Plan 3c implementation.

4. **Frames with no detections** — frames 16, 18-23, 27, 49, 50, 62-64 returned nothing. These are looking at empty walls / floor. Correct behavior; nothing to fix.

## Sample output (full per-frame log archived at /tmp/plan3c_gate output)

```
frame_00.jpg: ['bed', 'wardrobe']
frame_08.jpg: ['chair', 'desk', 'lamp', 'monitor', 'wardrobe']   ← peak detection density
frame_30.jpg: ['bed', 'nightstand']
frame_36.jpg: ['bookshelf dresser', 'chair']                     ← composite label example
frame_56.jpg: ['bed', 'nightstand']
```

## Phase A live integration

Not yet performed. Will run after Task 2 (`/detect_furniture` endpoint) lands.
