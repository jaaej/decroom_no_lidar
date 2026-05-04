# Plan 3c — Backend Open-Vocab RGB Detection on Keyframes

**Date**: 2026-05-03
**Author**: jaejeon (with Claude)
**Status**: Design — awaiting implementation plan
**Predecessors**:
- Plan 3a: phone-first backend integration MVP — `docs/superpowers/specs/2026-04-19-plan3a-phone-first-backend-integration-design.md`
- Plan 3b: semantic polish — `docs/superpowers/specs/2026-04-24-plan3b-semantic-polish-design.md`
**Plan 3b QA evidence**:
- Manual scans 2026-04-29 and 2026-05-03 returning 0 furniture bboxes from SpatialLM despite a real bedroom in view
- Probe outputs: `/tmp/decroom_b0_raw_slm.txt`, `/tmp/decroom_b0_parsed_slm.json`
- Grey-stripped ScanNet PLY test 2026-04-29: SpatialLM still detected 5 bboxes → confirmed colors are NOT the bottleneck
- Density study 2026-04-29: phone PLY ≈3,400 pts/m³ inside the room vs ScanNet ≈49,000 pts/m³ — a real distribution gap that is NOT solvable from the backend

---

## Context

Plan 3b's semantic-polish layer (consensus, wall-snapping, duplicate collapse, per-object confidence, door/window propagation) is functional and tested. Manual QA on a real bedroom (3.04 × 3.25 m, with bed / wardrobe / bookshelf / desk / 2 monitors / chair / door) revealed the **detection layer underneath it is the actual bottleneck**, not the polish layer.

### What works today (after Plan 3b)

- ARKit plane-anchor walls → consolidated 4-wall room rectangle ✓
- SpatialLM detects walls accurately on the phone PLY ✓
- SpatialLM detects doors/windows when present, attached to walls ✓ (parser-level, see Task 14)
- The Plan 3b consensus pipeline (consensus_merge → collapse_duplicates → snap_to_walls → compute_confidence) runs cleanly ✓
- Wall has_door / has_window flags propagate from SLM walls to phone walls ✓

### What doesn't work today

- **SpatialLM detects 0 furniture bboxes from phone-built PLYs.** Tested on multiple real-room scans; output is walls-only. Cause is a distribution mismatch — SpatialLM was trained on RGB-D scanner data (ScanNet); the phone uses monocular depth (Depth Anything V2) which produces smoother, normal-poor surfaces. The model's furniture filters don't fire on our cloud's surface character. Confirmed not solvable from the backend (see Plan 3b QA notes).
- **YOLO (on the phone, COCO-80 vocabulary) covers only a fraction of bedroom furniture** — has bed/sofa/chair/dining_table/tv/laptop/potted plant but no wardrobe/desk/bookshelf/dresser/nightstand. Even with a perfect cloud, YOLO can't label what its vocabulary doesn't include.
- **YOLO 3D positions are noisy** — depth-at-bbox-center varies across frames, ObjectTracker creates ghost duplicates of the same physical object (3 monitors at slightly different positions for 1 real monitor). This is a separate iOS-side problem, tracked outside Plan 3c.

### Net effect on the user

In QA scans of a 7-piece bedroom, the rendered RoomPayload contains 1 chair (correctly placed) + 3 ghost monitors. Bed, wardrobe, bookshelf, desk are completely absent. The pipeline reports the room is empty even though it isn't.

This is the gap Plan 3c closes.

---

## Goals

- Detect indoor furniture in real bedroom scans, including categories absent from COCO-80 (wardrobe, desk, bookshelf, dresser, nightstand).
- Replace the failing SpatialLM furniture path with a backend RGB-based detector that operates on the keyframe JPEGs the phone already uploads (10 per scan).
- Lift 2D detections to 3D using camera poses + intrinsics + the merged PLY (raycast through PLY surface). Avoid the per-pixel depth noise that wrecks YOLO's positions today.
- Feed the resulting 3D-tracked objects into the existing Plan 3b consensus pipeline as a third input source alongside YOLO and SpatialLM. Reuse all of Plan 3b's polish (snap, dedup, confidence) — no new pipeline plumbing required.

## Non-Goals

- **Do NOT remove SpatialLM.** It still produces walls + doors + windows correctly. Keep it.
- Do NOT change the phone-side scan flow. No iOS rebuild required for Plan 3c.
- Do NOT replace YOLO on the phone (it's already wired and tracker output is used for fast-feedback consensus). Just supplement it with the better backend detector.
- No correction UI changes — Plan 4a is still where user-facing corrections live.

## Architecture

```
Phone (unchanged)
    │
    ▼
upload payload (room_geometry + PLY + 10 keyframes + yolo_objects + scan_log)
    │
    ▼
Backend
    │
    ├─ SpatialLM inference on phone PLY  → walls + doors + windows
    │   (furniture bboxes will still be empty most of the time —
    │    don't rely on them; Plan 3c covers furniture)
    │
    ├─ NEW: open-vocab RGB detection on each keyframe JPEG
    │   ├─ Run model with text prompt:
    │   │   "bed. wardrobe. bookshelf. desk. monitor. chair.
    │   │    nightstand. dresser. lamp. potted plant. sofa."
    │   ├─ Per keyframe: list of (category, bbox_2d, confidence)
    │   ├─ Lift 2D → 3D via raycast from camera through bbox center
    │   │   into the merged PLY. First-hit voxel = 3D position.
    │   └─ Cluster across keyframes by category + 3D proximity →
    │      list of candidate furniture objects, one per real instance
    │
    ▼
Plan 3b consensus pipeline (UNCHANGED)
    1. consensus_merge(yolo_3d, slm_objects, walls)
       → now also receives the new RGB-detected objects as a third source.
         consensus_merge needs a small extension to accept three input
         lists or to merge RGB+YOLO upstream into a single "phone +
         backend detection" source.
    2. collapse_duplicates → 3. snap_to_walls → 4. compute_confidence
       → unchanged
    │
    ▼
RoomPayload with proper furniture
```

The new detector + lifter is a **new backend module**. Everything downstream of it is Plan 3b code, reused.

## Components

### 1. New: `backend/pipeline/rgb_detection.py`

Runs an open-vocabulary RGB detector on each keyframe JPEG.

**Recommended model: Grounding DINO** (`IDEA-Research/grounding-dino-tiny` on HuggingFace).
- Open-vocab via text prompts. Works on indoor furniture categories not in COCO.
- ~700MB model, ~2GB GPU memory at inference.
- Available via `transformers` library; no separate install pain.
- Inference time: ~1s per keyframe on RTX 4090 → ~10s for 10 keyframes per scan.

**Alternative: OWL-ViT** (`google/owlv2-base-patch16-ensemble`).
- Same open-vocab idea, similar quality.
- Lighter (~400MB).
- Choose based on whichever produces cleaner detections in a sanity test scan.

**Function signature:**

```python
def detect_furniture_in_keyframes(
    keyframe_jpegs: List[bytes],
    keyframe_metas: List[KeyframeMeta],   # has pose + intrinsics
    cfg: Any,
) -> List[KeyframeDetection]:
    """For each keyframe, run open-vocab detection with the indoor-
    furniture prompt list. Returns 2D detections per frame (no
    lifting yet — that's done by lift_detections_to_3d)."""
```

`KeyframeDetection` is a per-frame record: `{frame_index, category, bbox_2d, confidence}`.

### 2. New: `backend/pipeline/raycast_lift.py`

Lifts 2D detections to 3D using camera pose + intrinsics + the merged PLY.

**Algorithm:** for each 2D bbox center,
1. Build a camera-space ray from the optical center through the bbox center pixel.
2. Transform the ray to world space via the keyframe's pose.
3. Find the first PLY voxel the ray intersects (within a reasonable tolerance — e.g. 5cm shell around the ray).
4. That voxel's 3D position is the lifted detection.

If no voxel is found within the shell, drop the detection (probably out-of-room or specular noise).

**Why this beats per-pixel depth:** YOLO on the phone uses single-pixel depth at bbox center. That value is noisy frame-to-frame. PLY raycast uses the entire merged cloud — every contributing depth observation has already been voxel-deduped at 2cm by `PointCloudAccumulator`. Much more stable position.

### 3. New: backend cross-keyframe clustering for RGB detections

A real bed seen in 5 keyframes produces 5 separate RGB detections. Cluster them into one tracked object before feeding consensus.

```python
def cluster_keyframe_detections(
    detections: List[KeyframeDetection_3d],   # each carries lifted 3D position
    merge_distance_by_category: Dict[str, float],
) -> List[TrackedObject]:
    """Same algorithm as on-device ObjectTracker but on the backend.
    Per category: nearest-neighbor merge within the threshold; pick
    highest-confidence detection as canonical, accumulate observation
    count."""
```

Returns the same shape as `yolo_objects.json` so the rest of Plan 3b can consume it identically.

### 4. Modified: `backend/pipeline/phone_first_pipeline.py`

`run_phone_first_pipeline` gains a step between SLM and consensus:

```python
slm_walls, slm_objects = run_spatiallm_pipeline(positions_for_slm, cfg)
# ...existing wall + door/window plumbing...

# Plan 3c: open-vocab RGB detection on keyframes.
rgb_detections_2d = detect_furniture_in_keyframes(
    payload.keyframe_jpegs,
    payload.keyframes_manifest,
    cfg,
)
rgb_detections_3d = lift_detections_to_3d(
    rgb_detections_2d,
    positions,            # full PLY for raycasting
    payload.keyframes_manifest,
)
rgb_objects = cluster_keyframe_detections(
    rgb_detections_3d,
    merge_distance_by_category=BACKEND_TRACKER_DISTANCES,
)

# Feed into consensus alongside YOLO + SLM.
# Easiest: merge YOLO + RGB into a single "phone+backend detection"
# list before consensus_merge, since both are lifted-2D-to-3D.
combined_detections = _merge_phone_and_backend_detections(
    payload.yolo_objects, rgb_objects,
)
final_objects = _build_final_objects(
    slm_objects=slm_objects,
    yolo_objects_3d=combined_detections,   # was just yolo_objects
    phone_walls=phone_walls_with_openings,
)
```

### 5. Modified: `backend/pipeline/consensus.py`

`INDOOR_CATEGORIES_WHITELIST` and `CATEGORY_TO_ASSET` need entries for the new categories the RGB detector returns:
- `wardrobe` → `dresser` (closest substitute among bundled USDZ)
- `nightstand` → `nightstand` ✓
- `desk` → `desk` ✓
- `bookshelf` → `bookshelf` ✓
- `dresser` → `dresser` ✓
- `monitor` → `monitor` ✓ (kenney_monitor.usdz exists)

Most are already in the map. Audit + add any missing.

### 6. Server.py update on the pod

Add a new endpoint `/detect_furniture` that accepts (keyframe_jpegs[], category_prompts) and returns 2D detections per frame. OR run the detector locally on the same machine as the local backend (no GPU pod needed for Grounding DINO if we use ONNX runtime on CPU — slower but simpler). **Decision: run on the pod alongside SpatialLM** to keep all GPU work in one place. Adds a new endpoint; same SSH tunnel forwards it.

Actually — simpler — just add a `/detect_furniture` route to the existing `server.py` and load the Grounding DINO model at startup alongside SpatialLM. ~50 lines of Python. Both models fit in 4090's 24GB VRAM (SLM ≈4GB, Grounding DINO ≈2GB).

## Data shapes

### Backend-internal: `KeyframeDetection`

```python
@dataclass
class KeyframeDetection:
    frame_index: int
    category: str               # arbitrary string (open-vocab)
    bbox_2d: Tuple[float, float, float, float]  # (x, y, w, h) normalized
    confidence: float
    # 3D position added after lift step
    position_3d: Optional[Tuple[float, float, float]] = None
```

### Output: extra entries in RoomPayload.objects

Same `RoomObject` schema as today. Just more entries because RGB found things YOLO didn't.

## Migration plan

### Phase A — backend RGB detector standalone

1. **Model-quality gate (blocking)**. Before any backend plumbing, take 3–5 photos of the actual QA bedroom (the same room that produced the 0-furniture Plan 3b QA scan). Run Grounding DINO with the proposed prompt list against those photos in a standalone notebook or script — no `server.py`, no pipeline integration yet. Document per-category hit/miss in `docs/superpowers/qa/YYYY-MM-DD-plan3c-detector-gate.md`. **Go criterion**: target categories (wardrobe, bookshelf, desk, dresser, nightstand, bed) achieve ≥70% recall at confidence ≥0.4 across the test photos. If the gate fails, stop and revise the spec — either swap to OWL-ViT (test the same way), or accept that some categories are user-placement-only in Plan 4b. Do NOT proceed to step 2 until the gate passes.
2. Add Grounding DINO load to `server.py`. New `/detect_furniture` endpoint.
3. Add `backend/pipeline/rgb_detection.py` that calls the new endpoint over the SSH tunnel.
4. Unit tests with mocked HTTP response.

### Phase B — raycast lifting

4. Implement `lift_detections_to_3d` in `backend/pipeline/raycast_lift.py`.
5. Test on the phone's real PLY + a fake keyframe meta with known 3D ground truth (place a bbox over a known PLY surface, raycast should hit it).

### Phase C — cross-keyframe clustering

6. Backend `cluster_keyframe_detections` (mirrors on-device ObjectTracker logic in Python).
7. Tests.

### Phase D — integration into phone_first_pipeline

8. Wire RGB detection + lifting + clustering into `run_phone_first_pipeline`.
9. Update `consensus.py` whitelist + asset map for any new categories (wardrobe, etc.).
10. Update integration tests.

### Phase E — manual QA

11. Real-room scan, verify bed/wardrobe/desk/bookshelf appear.
12. Document in QA file.

### Phase F — merge

13. Merge to main.

## Risks and open questions

**Risk: Grounding DINO model size.** ~700MB to download to the pod. Already have 2.5GB SpatialLM cached on `/workspace`. Add to setup.sh.
*Mitigation:* one-time download to `/workspace/.cache/huggingface`.

**Risk: pod GPU memory pressure.** Loading both SLM + Grounding DINO. ~6GB total. RTX 4090 has 24GB. Comfortable headroom.

**Risk: backend latency increases.** SLM is ~7s, Grounding DINO on 10 keyframes ~10s. Total scan response time goes from ~12s to ~22s. Acceptable.

**Risk: open-vocab detector finds too much** (e.g., detects "wall" as a "wardrobe" with low confidence). Mitigation: confidence threshold (e.g. 0.4) before accepting into the lift step.

**Risk: raycast fails for objects far from any PLY surface** (occluded, behind another object). Drop these — the detection had no support in the cloud anyway.

**Open question: which prompt list is right?** Spec says: `"bed. wardrobe. bookshelf. desk. monitor. chair. nightstand. dresser. lamp. potted plant. sofa."`. Tune empirically once running.

**Open question: do we still need YOLO on the phone?** Once RGB detection is on the backend, YOLO's role shrinks. Could potentially disable YOLO on the phone (saves battery + thermal). Defer that decision until Plan 3c is shipped and proven.

## File structure

### New files
- `backend/pipeline/rgb_detection.py`
- `backend/pipeline/raycast_lift.py`
- `backend/tests/test_rgb_detection.py`
- `backend/tests/test_raycast_lift.py`
- Updated `backend/spatiallm_server/server.py` (adds `/detect_furniture` endpoint)
- `docs/superpowers/qa/YYYY-MM-DD-plan3c-manual-qa.md`

### Modified files
- `backend/pipeline/phone_first_pipeline.py` (add RGB detection step)
- `backend/pipeline/consensus.py` (whitelist + asset map updates)
- `backend/spatiallm_server/setup.sh` (download Grounding DINO weights)
- `backend/tests/test_phone_first_pipeline.py` (integration tests)

### Untouched
- iOS code (no rebuild needed for Plan 3c)
- Plan 3b consensus pipeline functions (consensus_merge, collapse_duplicates, snap_to_walls, compute_confidence)

## How a fresh conversation should pick this up

Read in this order:
1. **This spec** — context for why Plan 3c exists.
2. `docs/superpowers/specs/2026-04-24-plan3b-semantic-polish-design.md` — Plan 3b for the polish layer that consumes the new detections.
3. `docs/superpowers/specs/2026-04-19-plan3a-phone-first-backend-integration-design.md` — Plan 3a foundation.
4. `memory/runpod_setup.md` — pod setup with SpatialLM. Plan 3c will add Grounding DINO to the same `/workspace/spatiallm-env` venv.
5. `memory/known_issues.md` — running list of detection failures Plan 3c should fix.

Then invoke `/superpowers:writing-plans` against this spec to produce the task-by-task implementation plan.

## Reasoning summary (one paragraph for posterity)

We tried to make SpatialLM detect furniture in our pipeline. We can't. Six full backend fixes (color jitter, subsampling, point-cloud cropping, official inference flow mirror, max-points raise, install version pinning) all confirmed the same thing: SpatialLM was trained on RGB-D scanner data and the surface character of monocular-depth phone clouds is too far from its training distribution. SLM still does walls and openings well — those are coarse features, robust to scanner type. But it returns 0 furniture bboxes consistently from our PLYs. Plan 3c stops trying to coerce SLM into doing furniture detection and adds a different model — an open-vocab RGB detector running on the keyframe photos the phone already uploads. That's the right tool for the job: open vocabulary covers wardrobe/desk/bookshelf/dresser (which COCO-80 doesn't), RGB input doesn't depend on depth-cloud surface character, and 2D-to-3D lifting via PLY raycast gives stable positions where per-pixel depth is noisy. SpatialLM stays in the pipeline for what it does well; the new detector covers what SLM never could on our input.
