# Potential Adds — Parking Lot of Deferred Improvements

Things we considered, prototyped, or partially shipped, then deferred.
Each entry includes:

- What the improvement is
- What problem it addresses
- What's already been done (if anything, with commit refs)
- How to pick it back up
- Known risks or caveats

New items go at the top.

---

## Histogram-peak wall detection for dimensions

**Problem it addresses:** room dimensions derived from point cloud percentiles (5/95, 8/92, or 10/90) over-count or under-count in scenes where the depth model sees past real walls — through a doorway, window, mirror, or open cupboard. Percentile trimming assumes outliers are few; when >10% of the cloud legitimately extends past the room, percentiles can't clip them all.

**The approach:** instead of using a bounding-box percentile, compute a 1-D histogram of dot density along each Manhattan axis. Walls appear as **density peaks** — thousands of depth rays hit real wall surfaces producing a tall narrow spike, while through-wall bleed is a sparse tail that doesn't form a peak. Pick the outermost peak positions as wall positions; room extent = distance between them.

Algorithm outline:
1. Rotate the point cloud into Manhattan frame using the phone's `manhattan_yaw`.
2. Per axis, compute a 5 cm-bin histogram over the 0.5th to 99.5th percentile range.
3. Smooth with a 3-bin moving average to reduce noise.
4. Find peaks where density ≥ 3× the median non-zero bin density.
5. Keep the top 4 peaks by height; room span = distance between the outermost of those 4.
6. Return `(width, length, peaks_ok: Bool)` — caller falls back to other sources when peaks aren't clean.

**Already done:** code written, tested, committed, then reverted.

- Implementation commit: `cc285c5` — adds `dimensions_from_cloud_peaks` to `backend/pipeline/phone_first_pipeline.py` and integrates it into `room_envelope_from_walls` as the primary dimension source (falls back to anchor-vs-cloud reconciliation when peaks don't fire).
- Revert commit: `67914b3`.
- 4 tests included in `cc285c5`: synthetic walls + through-door bleed, sparse cloud fallback, uniform cloud fallback, end-to-end room-envelope preference.

**Why it was reverted:** in manual QA on the user's real room, peaks never fired. The user's point cloud had broader, flatter density distributions rather than tall narrow spikes at walls, so the 3× median threshold wasn't met. Every scan fell back to the `reconcile` path, giving the same numbers as before — the feature added LOC but contributed zero behaviour on that user's data. Kept in git history; not running in the pipeline.

**To pick back up:**

```bash
git cherry-pick cc285c5
```

**What would need to change for it to help in practice:**

1. **Lower the peak threshold** from `min_peak_ratio=3.0` to `2.0` or `1.5`. Risk: accepts furniture density in the middle of the room as a "peak," producing false wall positions.
2. **Use wider histogram bins** (10 cm instead of 5 cm). Smooths out noise, broadens peaks. Risk: reduces precision of the peak position.
3. **Pre-filter the cloud to points near ARKit plane anchors only** — keeps the analysis to "likely wall dots" and discards through-wall leak entirely. Requires passing plane-anchor surfaces from iOS to the backend.
4. **Combine with floor-plane extent as a cross-check** — if peaks say 3.0 m and floor extent says 3.0 m, strongly trust that value.

**Suggested successor approach**: pair the histogram-peak method with better input guards — reject scans that don't have ≥ 4 consolidated walls before they reach the dimension algorithm at all. Peaks work for clean scans; "scan is incomplete" should be caught earlier via UX gating, not by fallback heuristics.

---

## Correction UI (tap-to-adjust room)

**Problem it addresses:** automated room measurement from a 30-60 s iPhone scan without LiDAR has a practical error floor of ~10-15%. For a user who wants a tidy final room, neither "ship with 15% error" nor "keep tuning algorithms" is a good answer. Manual correction lets the user be the final source of truth.

**The approach:** after the 3D preview renders, overlay draggable corner handles on each wall. User drags corners to match reality, walls snap to Manhattan angles, resulting room saves with corrected dimensions. Also: tap furniture item → delete / rotate / move.

**Status:** not started. Explicitly deferred to Plan 4 in the original design spec.

**To pick back up:** originally scoped as Plan 4 / Phase 6 in the architectural design (`docs/superpowers/specs/2026-04-16-phone-first-ar-architecture-design.md`). A focused brainstorm + spec would list the interactions and touch points in `RoomSceneView.swift`.

---

## SpatialLM door and window detection

**Problem it addresses:** the backend currently receives SpatialLM output but the `SPATIALLM_TO_PRD` mapping in `pipeline/spatiallm_layout.py` doesn't include `door` or `window` categories. Even when SpatialLM identifies them, we drop them on the floor. `Wall.has_door` / `has_window` always resolves to `false`.

**The approach:** add `door` and `window` to the category mapping. For each SpatialLM door/window, find the nearest phone wall within ~0.5 m and set that wall's `has_door` / `has_window` flag.

**Status:** not started. Explicitly deferred to Plan 3b.

**To pick back up:** ~30 lines of Python in `pipeline/spatiallm_layout.py` + `pipeline/phone_first_pipeline.py`. Requires that SpatialLM actually outputs doors/windows with recognisable category strings — needs one test scan with the GPU tunnel active to verify category names.

---

## Wall-snapping for furniture

**Problem it addresses:** furniture items from SpatialLM currently pass through untouched. A bed detected at (1.5, 0, 1.8) might be placed 30 cm from the nearest wall; our 3D preview shows it floating in the middle of the room rather than pushed against the wall.

**The approach:** for categories that should be against walls (bed, dresser, sofa, bookshelf, wardrobe), find the nearest phone wall within 0.8 m, translate the object to touch that wall, set `rotation_y` so the "back" of the object faces the wall. For freestanding categories (chair, table, plant, rug) do nothing.

**Status:** not started. Explicitly deferred to Plan 3b (section "Semantic Wall-Snapping Rules" in the design spec).

**To pick back up:** ~40 lines of Python in `pipeline/phone_first_pipeline.py`. Rule table is in the design spec.

---

## Better scanning guidance UX

**Problem it addresses:** users see a coverage arc that says "66%" after scanning a full room and have no idea what's missing. The `.noWallsDetected` and `.insufficientCoverage` alerts fire only at Done time — mid-scan the user has no real-time signal.

**The approach (partial):** a "No walls detected — look at wall edges, slow down" banner fires when ARKit hasn't committed any vertical plane anchors after 15 s of active scanning. This is already shipping (commit `c0cd32a`).

**Still needed:**
- Arrow pointing at the nearest empty yaw bin ("look 45° left")
- Tilt prompts ("look up and down briefly to capture floor/ceiling")
- Lighting warning when ARKit reports low light
- Explicit per-wall feedback ("north wall captured / south wall not yet")

**Status:** partial. Real-time no-walls banner shipped (see `c0cd32a`). Rest deferred — explicitly flagged as an "open question" in the design spec.

**To pick back up:** separate guidance-polish plan. Light work (~1-2 hours per direction hint) but high user-impact when scans consistently fail quality gates.

---

## Floor-plane extent as a third dimension source

**Problem it addresses:** same as histogram peaks — plane-anchor and cloud-percentile dimensions both have systematic errors. A third source of truth would let us vote.

**The approach:** pass ARKit's floor plane anchor extent from iOS to the backend (add to `room_geometry.json`). Use as a cross-check: if floor extent, anchor bbox, and cloud percentile roughly agree, confidence is high and we pick the median.

**Status:** considered, not implemented. ~1 hour (iOS + backend).

**To pick back up:** add `floor_extent_width`, `floor_extent_length` to `RoomGeometry` JSON schema; compute from the largest horizontal plane anchor in `RoomGeometryBuilder`; backend vote logic in `room_envelope_from_walls`.

**Risk:** floor plane extent can also under-size when scan shows only partial floor (under a bed, a couch, rugs). Not a silver bullet; works best as a **third voter**, not a primary source.

---

## Replacing Depth Anything V2 with Apple Depth Pro

**Problem it addresses:** Depth Anything V2 Metric Indoor has ~5-10% per-pixel depth error on indoor benchmarks. Apple's Depth Pro is ~2× more accurate on boundary fidelity and overall δ₁ score.

**Status:** considered, rejected. Depth Pro is ~10× slower and ~20× larger. On iPhone, real-time at 2 Hz with Depth Pro isn't achievable.

**When to reconsider:** if Apple ships a quantised or mobile-optimised Depth Pro, or if we move depth inference to the backend for premium users (trading privacy for accuracy).

---

## YOLO consensus for furniture detection

**Problem it addresses:** SpatialLM's object list is the only furniture signal. If SpatialLM misses an item or places it at the wrong position, we have no cross-check.

**The approach:** re-enable `YoloRunner` on iOS (currently disabled — see `ScanSession.update`). Ship YOLO detection metadata in the keyframes manifest. Backend compares SpatialLM's objects against YOLO detections; requires both models to agree within category + position tolerance before including the object.

**Status:** infrastructure partially there (`YoloRunner` exists, bundled in the app, just not dispatched). Deferred explicitly in Plan 3a.

**To pick back up:** flip the `if false` guard in `ScanSession.update` back on. Populate `keyframes.json` `detections` array. Add consensus logic to `phone_first_pipeline.py`. ~60 lines backend + verify YOLO doesn't degrade scan performance on real devices.

**Risk:** YOLO adds ML pressure on the iPhone's Neural Engine. Make sure the depth + YOLO combo doesn't cause the buffer-pool freeze that Plan 3a's MainActor offload fix addressed.
