# Plan 3b — Semantic Polish Design Spec

**Date**: 2026-04-24
**Author**: jaejeon (with Claude)
**Status**: Design — awaiting implementation plan
**Predecessor**: Plan 3a (`docs/superpowers/specs/2026-04-19-plan3a-phone-first-backend-integration-design.md`)

## Context

Plan 3a shipped the phone-first architecture end-to-end: iOS sends room geometry + PLY + keyframes, backend runs SpatialLM on the phone PLY, returns a `RoomPayload`. What Plan 3a deliberately skipped — to stay MVP-focused — was the **semantic layer on top of raw SpatialLM output**: wall-snapping, duplicate collapse, YOLO/SpatialLM consensus, per-object confidence, door/window handling.

Without those, the current pipeline's failure modes are:

- **Furniture floats in the middle of rooms** — SpatialLM places a bed at (1.5, 0, 1.8) and we render it there, not snapped against a wall.
- **Duplicate objects** — SpatialLM occasionally detects the same bed twice at slightly different positions; we keep both.
- **No cross-check on SpatialLM** — if SpatialLM hallucinates a chair in a corner where there is none, we render the hallucination. No second opinion from on-device YOLO.
- **Flat confidence** — every object has the same `confidence: 0.5`; users can't tell which detections are certain vs shaky.
- **Doors and windows ignored** — `Wall.has_door` / `has_window` always `false`, even when SpatialLM identifies them in its output.

Plan 3b closes these gaps. This is the semantic layer the design originally envisioned for Plan 3, split out of Plan 3a to keep 3a small.

## Goals

- Apply universal priors that make sense for every room: beds go against walls, duplicates get collapsed.
- Use the on-device YOLO detector (already bundled but disabled) as a second vote against SpatialLM's output.
- Give each returned object a meaningful confidence score that Plan 4a's correction UI can surface as "high / medium / low" visual treatment.
- Detect doors and windows on walls, to be rendered in the 3D preview as openings.

## Non-Goals

- No new visual UI. Plan 4a (correction UI) and Plan 4b (furniture library) cover user-facing editing.
- No automated room-dimension re-derivation. Plan 3a's current approach (plane-anchor + cloud-percentile + smaller-wins) stays. Dimension improvements deferred — see `docs/superpowers/potential_adds.md` (histogram peaks entry).
- No new on-device models. YOLOv8s-seg that's already bundled is the only new ML layer; no swap to a different detector.
- No correction UI for furniture. That's Plan 4a.
- No manual furniture library. That's Plan 4b.

## Architecture

Plan 3b is **backend-heavy with a small iOS change**. Nothing user-facing changes about the capture UX. The capture, geometry, point cloud, and keyframe selection all stay exactly as Plan 3a left them. The only iOS difference: on-device YOLO gets re-enabled so its detections can be packaged into the keyframe manifest and sent to the backend.

### Before Plan 3b (current Plan 3a state)

```
Phone (unchanged)  ─┐
                    ▼
              upload payload
                    │
                    ▼
              Backend
                    │
                    ▼
              SpatialLM inference on phone PLY
                    │
                    ▼
              map_spatiallm_objects_to_room_objects()
                    │   (simple 1:1 pass-through)
                    ▼
              RoomPayload (walls + raw objects)
```

### After Plan 3b

```
Phone
  └─ ScanSession now runs YOLO (was disabled in Plan 3a)
      └─ YOLO detections logged per keyframe + accumulated
        └─ keyframes.json now populated with detections[] per frame
  └─ Upload payload: adds YOLO detections per keyframe (new field)
                    │
                    ▼
              Backend
                    │
                    ▼
              SpatialLM inference on phone PLY
                    │
                    ▼
              ──────────────────────
              NEW: Semantic polish pipeline
              ──────────────────────
              1. consensus_merge(yolo_frames, slm_objects, walls)
                 ├─ match YOLO detections to SpatialLM objects by
                 │   category + 3D position proximity
                 ├─ assign agreement_score per object
                 └─ flag orphans (YOLO-only or SLM-only)
              2. collapse_duplicates(objects, category_rules)
                 └─ position-based merge within category-specific
                     distance; no hardcoded instance caps
              3. snap_to_walls(objects, walls, category_rules)
                 ├─ bed/dresser/sofa/bookshelf/wardrobe → wall
                 ├─ nightstand → wall (tighter)
                 ├─ monitor/tv → perpendicular to wall
                 └─ chair/table/lamp/plant/rug → freestanding
              4. compute_confidence(obj)
                 └─ 0.5*agreement + 0.3*multi_view + 0.2*size_plausibility
              5. extract_doors_windows(slm_output, walls)
                 └─ match door/window positions to walls;
                     set Wall.has_door / has_window
                    │
                    ▼
              RoomPayload (walls with has_door/has_window,
                           objects with meaningful confidence)
```

## Components

### 1. iOS — re-enable on-device YOLO

**File:** `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ScanSession.swift`

**Change:** Currently `update(frame:)` has a block gated behind a `false`-only comment (YOLO dispatched to mlQueue is commented out from Plan 3a). Re-enable with a `yoloInFlight` drop-on-busy guard matching the existing `depthInFlight` pattern.

**Frequency:** 1 Hz (every 30 frames). Lower than depth's 2 Hz because YOLO is ~100 ms per frame on iPhone Neural Engine — we don't need more frequent coverage than that for consensus.

**Output:** `YoloDetection` records (category, bbox, confidence, frameIndex). These accumulate in `ObjectTracker` (already in the codebase).

### 2. iOS — serialize YOLO detections to keyframes manifest

**File:** `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/ScanUploadPayload.swift`

**Change:** `KeyframesManifest.FrameEntry.detections` is currently `[String]` and always empty in 3a. Change to a richer structure:

```swift
struct Detection: Codable {
    let category: String      // COCO-80 name e.g. "bed", "chair"
    let confidence: Float
    let bbox: [Float]         // normalized [x, y, w, h]
}
```

And fill the `detections` array at `finalize()` time by asking the `ObjectTracker` for detections whose `bestFrameIndex == keyframe.frameIndex`.

### 3. Backend — consensus merge

**New file:** `backend/pipeline/consensus.py`

**Function:** `consensus_merge(yolo_detections_per_frame, slm_objects, walls) -> (merged_objects, agreement_scores)`

**Algorithm** (per the design spec's table):

| YOLO says | SpatialLM says | Outcome |
|---|---|---|
| Bed at (2, 1.5) | Bed at (2.1, 1.4) | Keep SLM bbox, agreement_score = 1.0 |
| Bed at (2, 1.5) | Nothing near there | Keep at YOLO position (default oriented bbox), agreement_score = 0.5 |
| Nothing | Bed at (2, 1.5) | Keep SLM as-is, agreement_score = 0.5 |
| Bed at (2, 1.5) & (4, 1.5) | Beds at (2, 1.5) & (4, 1.5) | Keep both, agreement_score = 1.0 each |
| Bed at (2, 1.5) | Beds at (2, 1.5) & (2.2, 1.6) | Collapse to 1, agreement_score = 0.3 |

**Position match distance**: 0.8 m for large items (bed, sofa, dining table), 0.4 m for smaller items (chair, nightstand, lamp).

### 4. Backend — position-based duplicate collapse

**New function in** `consensus.py`: `collapse_duplicates(objects, category_rules) -> [RoomObject]`

**Algorithm:** For each category, find all pairs within the merge distance (per-category table below). When two match, keep the one with highest `confidence`; discard the other. Run passes until no pairs within tolerance.

**Per-category merge distances (meters):**
- `bed`: 1.5 (a real room rarely has two beds < 1.5 m apart)
- `sofa`: 1.5
- `chair`: 0.65
- `dining_table`: 1.1
- `desk`: 1.1
- `monitor`: 0.8
- `laptop`: 0.5
- `lamp_floor`: 0.5
- `lamp_desk`: 0.3
- `potted_plant`: 0.6
- `bookshelf`: 1.2
- `dresser`: 1.2
- `default`: 0.6

**Key principle:** no hardcoded `_INSTANCE_CAPS` dict. Rooms can legitimately have 3 beds (hostel), 4 chairs (kitchen), etc. Only collapse if positions genuinely overlap.

### 5. Backend — wall-snapping

**New function in** `consensus.py`: `snap_to_walls(objects, walls) -> [RoomObject]`

**Per-category rules:**

| Category | Snap behavior | Tolerance |
|---|---|---|
| `bed`, `dresser`, `wardrobe`, `sofa`, `bookshelf` | Translate to touch nearest wall; face outward (rotation_y = wall_yaw + π) | 0.8 m |
| `desk` | Snap to wall; if equidistant from 2 walls, pick the wall perpendicular to desk's long axis | 0.8 m |
| `nightstand` | Tight wall-snap | 0.5 m |
| `monitor`, `tv` | Orient perpendicular to wall (facing into room) | 0.8 m |
| `chair`, `stool`, `dining_table`, `coffee_table`, `side_table` | Freestanding, no snap | — |
| `lamp_floor`, `lamp_desk`, `potted_plant`, `rug`, `poster` | Freestanding | — |

**Implementation:** for each wall-snap category object, find nearest wall by perpendicular distance from object position to wall line segment. If within tolerance, project object onto wall line and set `rotation_y` to the wall's perpendicular normal.

### 6. Backend — per-object confidence

**New function in** `consensus.py`: `compute_confidence(obj, yolo_matches, view_count, size_plausible) -> float`

**Formula (from design spec):**

```
confidence = 0.5 * agreement_score
           + 0.3 * multi_view_score
           + 0.2 * size_plausibility
```

- `agreement_score` — 1.0 if YOLO and SLM both detected this obj; 0.5 if only one; 0.3 if this is a duplicate-collapse survivor
- `multi_view_score` — 1.0 if the object's 3D position is within 0.5 m of its recorded projection in 3+ keyframes; scales down linearly for fewer views
- `size_plausibility` — 1.0 if `(width, length, height)` are within typical range for the category (bed: width 1.2-2.2 m, length 1.9-2.2 m, etc.); 0.3 if outside

**Output:** stored in `RoomObject.confidence` (currently hardcoded to 0.5).

### 7. Backend — door and window detection

**File:** `backend/pipeline/phone_first_pipeline.py` (modify `run_phone_first_pipeline`)

**Source:** SpatialLM's output includes door and window layout elements — NOT in the `SPATIALLM_TO_PRD` mapping because they are wall-attached, not furniture.

**Algorithm:**

1. After `parse_spatiallm_output(...)`, also extract door/window entries (they have `class_name == "door"` / `"window"` but weren't mapped in 3a)
2. For each door: find nearest wall segment in `walls` whose line segment passes within 0.5 m of the door's centroid
3. Set that `Wall.has_door = True`
4. Same for windows — set `has_window = True`
5. Log any door/window that couldn't be matched to a wall

**Note:** `Wall` schema already has `has_door: Bool` and `has_window: Bool` fields (from pre-3a days). Plan 3a hardcoded them to `False`; Plan 3b populates them.

## Data shapes

### New field: `keyframes.json` `detections`

```json
{
  "frames": [
    {
      "filename": "frame_00.jpg",
      "timestamp": 1.234,
      "pose_world_from_camera": [...],
      "intrinsics": { ... },
      "detections": [
        {"category": "bed", "confidence": 0.87, "bbox": [0.12, 0.34, 0.56, 0.78]},
        {"category": "lamp", "confidence": 0.62, "bbox": [0.71, 0.22, 0.15, 0.20]}
      ]
    }
  ]
}
```

### Changed field: `RoomObject.confidence`

Previously always 0.5. Now in the range 0.1-1.0 with meaningful distribution:
- 0.8-1.0: YOLO + SpatialLM both detected + multi-view + plausible size
- 0.5-0.8: single-source or moderate agreement
- 0.1-0.5: flagged as low-confidence (duplicate collapse survivor, single view, unusual size)

### Changed field: `Wall.has_door`, `Wall.has_window`

Previously hardcoded false. Now populated from SpatialLM output.

## Error handling

Plan 3b doesn't add new user-facing errors. All behavior is backend-internal. Failure modes:

- YOLO inference fails to load on iOS: log `yolo_runner_load_failed` in scan_log, continue without YOLO detections. Backend still works (agreement_score just drops to 0.5 for everything).
- SpatialLM outputs unexpected door/window format: log warning, skip door/window detection for this scan.
- Object position out of room bounds: log and filter out.

## Testing

### Backend unit tests

Add to `backend/tests/`:

| Test file | Coverage |
|---|---|
| `test_consensus.py` | `consensus_merge` with all 5 outcomes from the table; `collapse_duplicates` with per-category merge distances; edge cases (empty lists, single object). |
| `test_wall_snapping.py` | Bed → wall snap, chair stays freestanding, 2 equidistant walls for desk → wall-perpendicular chosen, objects beyond tolerance stay in place. |
| `test_confidence.py` | Formula sanity: full agreement + multi-view + plausible → ~1.0; opposite extreme → ~0.1. |
| `test_door_window.py` | Door near wall → attached; door far from any wall → logged, skipped. |

### iOS unit tests

| Test | Coverage |
|---|---|
| Updated `ScanUploadPayloadTests` | `keyframes.json` now includes `detections` array with at least one entry; each entry has category/confidence/bbox. |

### Contract test update

`tests/fixtures/phone_first_payload/keyframes.json` — regenerate with realistic detection arrays.

### Manual QA

Requires a live SpatialLM tunnel (RunPod). Steps:

1. Scan a room with a bed against a wall. After backend processing, verify:
   - Bed is touching the wall, not floating in middle (snap worked)
   - Bed's `confidence` is high (>0.7)
2. Scan a room with two chairs. Verify both chairs in the response (no false dedup).
3. Scan a room with a visible door. Verify `Wall.has_door = True` for the wall containing the door.
4. Scan a cluttered room; verify no SpatialLM-generated duplicates at same position survive.

## Migration plan

### Phase A — iOS side (prep)

1. Update `ScanUploadPayload.Detection` struct.
2. Re-enable YOLO in `ScanSession.update(frame:)` with `yoloInFlight` guard.
3. Update `ScanSession.finalize()` to populate each keyframe's `detections` from `ObjectTracker`.
4. Update `ScanUploadPayloadTests` to check detections are populated.
5. Commit + tests pass.

### Phase B — Backend side (new code)

6. Create `backend/pipeline/consensus.py` with `consensus_merge`, `collapse_duplicates`, `snap_to_walls`, `compute_confidence`.
7. Write `test_consensus.py`, `test_wall_snapping.py`, `test_confidence.py`.
8. Modify `phone_first_pipeline.py` to call consensus steps after `map_spatiallm_objects_to_room_objects`.
9. Write `test_door_window.py`.
10. Modify `phone_first_pipeline.py` to extract door/window from SpatialLM output and attach to walls.
11. Full backend test suite passes.

### Phase C — Contract fixture

12. Regenerate `tests/fixtures/phone_first_payload/keyframes.json` with realistic detection arrays.
13. Verify contract tests still pass on both sides.

### Phase D — QA + merge

14. Manual QA with live SpatialLM tunnel — document in `docs/superpowers/qa/YYYY-MM-DD-plan3b-manual-qa.md`.
15. Merge the branch (separate branch from Plan 3a's `feature/phone-first-visual`).

### Rollback paths

- Per-step git revert works without consequence. The old `map_spatiallm_objects_to_room_objects` stays the default if `consensus_merge` is removed.
- If YOLO causes thermal issues on iPhone, disable with a feature flag (keep bundle, skip dispatch).

## Risks and open questions

**Risk**: YOLO re-enabled could trigger the ARKit buffer-pool freeze that Plan 3a fixed via the `depthInFlight` guard.
*Mitigation*: add `yoloInFlight` guard mirroring `depthInFlight`. Test on device during Phase A.

**Risk**: SpatialLM's door/window output format might differ from what we assume (the current Plan 3a mapping only handles furniture categories).
*Mitigation*: run one test scan with SpatialLM live, log raw output, pattern-match. This is a 10-minute task that can happen before writing `test_door_window.py`.

**Risk**: Wall-snapping could move an object to the wrong wall if two walls are equidistant.
*Mitigation*: tie-breaker rules per category (desk perpendicular to long axis); for ambiguous cases, log and don't snap (leave at SpatialLM position).

**Open question**: should `confidence` also account for "this object type makes sense in this room type"? E.g., a toilet in a bedroom is physically plausible but categorically weird.
*Current answer*: no. Out of scope for Plan 3b. Would require room-type classification (bedroom / kitchen / bathroom) which we don't have. Could be a future plan.

**Open question**: should YOLO-only detections (SLM missed them) be added to the final RoomPayload?
*Current answer*: yes, but flagged with `confidence = 0.5` so Plan 4a's correction UI can surface them for user confirmation.

## Dependencies

- SpatialLM running and producing output including doors/windows. Needs RunPod tunnel.
- No new third-party dependencies.
- Plan 3a merged to main first (Plan 3b builds on that branch's code).

## Out of scope (explicitly — tracked for later plans)

- **Correction UI** — Plan 4a. Users tap walls/furniture to manually correct.
- **Furniture library** — Plan 4b. Manual placement of Kenney models.
- **Dimension re-derivation** — see `docs/superpowers/potential_adds.md` (histogram peaks entry).
- **Room-type classification** — no current scope.
- **Better scanning UX** — Plan 3a open question, tracked separately.
- **App size reduction** — dropping bundled Core ML models when we're sure they're safe to strip.

## File structure

### New files

- `backend/pipeline/consensus.py` — the 4 new functions (`consensus_merge`, `collapse_duplicates`, `snap_to_walls`, `compute_confidence`).
- `backend/tests/test_consensus.py`
- `backend/tests/test_wall_snapping.py`
- `backend/tests/test_confidence.py`
- `backend/tests/test_door_window.py`
- `docs/superpowers/plans/2026-04-24-plan3b-semantic-polish.md` — the implementation plan (to be written via `/superpowers:writing-plans`)

### Modified files

- `decroom_no_lidar/decroom_no_lidar/ScanSession.swift` — re-enable YOLO dispatch
- `decroom_no_lidar/decroom_no_lidar/ScanUploadPayload.swift` — `Detection` struct + populate manifest
- `decroom_no_lidar/decroom_no_lidarTests/ScanUploadPayloadTests.swift` — assert detections populated
- `backend/pipeline/phone_first_pipeline.py` — call consensus pipeline, extract doors/windows
- `backend/tests/test_phone_first_pipeline.py` — update expectations for new RoomPayload shape
- `tests/fixtures/phone_first_payload/keyframes.json` — regenerate with detection arrays

## How a new conversation should pick this up

1. Read this spec top-to-bottom
2. Optionally read `docs/superpowers/specs/2026-04-19-plan3a-phone-first-backend-integration-design.md` for architectural background
3. Optionally read `docs/superpowers/potential_adds.md` for related deferred items
4. Check `memory/plan_status.md` for current branch state
5. Check `memory/runpod_setup.md` if SpatialLM infra isn't live yet
6. Invoke `/superpowers:writing-plans` with this spec as input to produce the task-by-task implementation plan
7. Execute via `/superpowers:subagent-driven-development` on a new feature branch off `main` (after Plan 3a has merged)
