# Decroom Clean-Slate Scan Pipeline — Design (2026-07-02)

**Status:** draft, pending user review
**Framing:** clean-slate re-derivation of the scan→3D-room architecture. Past failures are evidence, not constraints. Supersedes the 2026-04-16 phone-first spec where they conflict.
**Constraints:** non-LiDAR iPhone baseline; pre-trained off-the-shelf models only; inputs must be producible from an iPhone scan; 1–3 min post-scan wait budget; output = discrete, manipulable 3D objects (shell + per-furniture). Dual license tracks: commercial-safe and research-best.

---

## 1. Evidence base

From 74 days of project history:
- ARKit poses are metric and reliable; classical SfM (COLMAP) on casual phone video fragments badly. → Trust ARKit, never re-solve cameras.
- PCA on point clouds locks to the camera-path diagonal (the 45° rotation bug). → Align to gravity + wall normals, never PCA.
- Monocular-depth point clouds are smooth/normal-poor vs RGB-D scans; SpatialLM's furniture head never fires on them (6 fix attempts exhausted). → Don't feed RGB-D-trained point-cloud models our clouds and expect furniture.
- COCO-80 vocabulary misses most furniture (wardrobe, bookshelf, nightstand…). → Open-vocabulary detection required.
- Lifting per-frame detections to 3D then clustering centroids cannot resolve duplicates: depth noise scatters the same object's centroid 0.5–1 m, while distinct objects sit closer than that. → Associate instances in image space first.

From fresh model research (2026-07-02, licenses verified on repos/HF cards — citations in §8):
- Most learned 3D models (VGGT, Pi3, MASt3R family, CUT3R/Spann3R/SLAM3R/Fast3R) **discard known poses** and re-estimate their own; most have research-only weights. The pose-consuming, commercial-clean stack is MoGe-2 (MIT) + Open3D TSDF (MIT), with MapAnything-apache (Apache-2.0) as the learned alternative.
- SAM 3 (Meta, 2025) does text→per-instance masks **tracked across video with persistent IDs** in one model; weights gated, license copyleft-ish. Permissive equivalent: LLMDet (Apache-2.0) boxes → SAM 2 (Apache-2.0) video propagation.
- Orient Anything V2 (CC-BY-4.0, DINOv2/Apache backbone): object crop → azimuth, symmetry-aware, multi-view fusable. Commercially clean.
- SpatialLM: every released checkpoint has a CC-BY-NC point encoder (SceneScript in 1.0, Sonata in 1.1) → **not commercially usable**, any variant.
- Generative compositional mesh models matured in 2025: SceneGen (MIT), PartCrafter-Scene (MIT), MIDI (Apache-2.0); per-object single-image generators TRELLIS (MIT), InstantMesh (Apache-2.0). They output complete (watertight, back included) meshes that match the photo — but quality on real cluttered scan crops is **unverified**.
- Apple RoomPlan is LiDAR-only. Hunyuan3D licenses exclude South Korea — disqualified.

## 2. Architecture decision

**Approach A (this design): tracked-instance pipeline + stock furniture swap, with Approach B (generative furniture meshes) as a spike-gated phase 2 behind the same interface.**

Rejected as backbone:
- *SpatialLM/SceneScript-centric holistic:* license-dead commercially; furniture-dead on our clouds (proven); adds nothing over classical shell extraction.
- *SceneGen-style single-image scene generation as backbone:* ignores the multi-view evidence a scan provides; placement accuracy unproven. Right tool as Phase-2 mesh generator, wrong tool as pipeline.

**Division of labor (resolves the on-device/server open question):** the phone does capture + ARKit tracking + live scan feedback only; the backend GPU does all perception. Rationale: the 2026-04-16 "phone does geometry" premise existed because *backend SfM* failed — but with ARKit poses uploaded, backend geometry is not SfM (cameras are already solved). Every chosen model is GPU-scale, the 1–3 min budget permits it, and removing on-device depth/YOLO fusion removes the noise source behind the duplicate bug and a large amount of on-device complexity.

### Data flow

```
PHONE (capture, real-time)
  ARKit session → HEVC video (1080p) + per-frame poses/intrinsics (JSONL, timestamp-keyed)
                + gravity vector + plane anchors + sparse feature points
  Existing VideoRecorderView UX unchanged (coverage arc, upload)
  Upload ~20–40 MB

BACKEND (GPU, perceived-wait target 30–60 s; hard cap 2 min)
  1. Decode video; sample frames (~5 fps track set, ~2 fps depth set)
  2. GEOMETRY   MoGe-2 depth+normals per depth-frame
                → per-frame metric anchor vs ARKit sparse points (median scale)
                → Open3D TSDF fusion (2–4 cm voxel) → dense cloud + mesh
  3. INSTANCES  LLMDet (furniture+door+window vocabulary) on ~1 fps keyframes
                → boxes prompt SAM 2 → masks propagated across video (persistent IDs)
                → per track: masked depth back-projection per frame via ARKit pose
                → median centroid, robust extents, per-frame point sets
                → class-conditioned track merge (same label, centroid < 0.4 m, footprint overlap)
  4. ORIENTATION Orient Anything V2 on 3 best crops per track (largest mask, most frontal)
                → azimuth → world yaw via pose; circular median across crops
                → reconcile with footprint long axis; skip yaw for symmetric categories
  5. SHELL      consolidate ARKit vertical planes (≤5°, ≤10 cm coplanar merge)
                + RANSAC planes on TSDF cloud where anchors are missing
                → gravity = up; Manhattan yaw = dominant orthogonal wall-normal pair
                → floor/ceiling from horizontal planes; rectilinear wall polygon
                → openings: door/window tracks projected onto nearest wall plane
                  (doors touch floor; windows don't)
  6. ASSEMBLY   RoomPayload v2 (single gravity-aligned Manhattan frame, Y-up)

PHONE (render + edit)
  SceneKit viewer + Plan 4a correction UI unchanged
  Stock asset per category (Kenney CC0), scaled to measured extents
```

**Performance budget (user requirement: results in 30–60 s).** Achieved by: (1) streaming the upload during capture so it costs ~0 perceived seconds; (2) running the geometry branch (stages 2) and detection branch (stage 3 pre-lift) concurrently — they are independent until lifting; (3) sampling rates 1 fps depth / 3 fps tracking / 0.5 fps detection, 720p inference for detection+tracking. The two budget risks are SAM 2's per-object scaling on many-object rooms and LLMDet's unpublished per-frame speed — spikes S1/S3 must record wall-clock times, and the fallbacks are lower sampling, OWLv2/Florence-2, or a faster GPU tier. All stage timings above are estimates until Phase 0 measures them.

## 3. The four sub-problems

### 3.1 Depth / 3D structure
Signals used: RGB video + ARKit poses/intrinsics (metric) + gravity + sparse points. MoGe-2 (MIT) predicts metric-ish depth + normals per frame; a single median scale factor per frame against ARKit sparse-point depths removes monocular scale error; Open3D TSDF fusion under known poses integrates frames into one consistent cloud/mesh. No SfM anywhere. Normals from MoGe-2 directly address the "normal-poor cloud" defect. A/B candidates behind a flag: MapAnything-apache (Apache, natively pose-consuming, single-pass), Depth Anything V3 `DA3Metric-Large` (Apache). Research-track benchmark reference: CUT3R (NC).

### 3.2 Detection (instances, not blobs)
Open-vocab detector LLMDet (Apache 2.0, ~42 AP LVIS — best permissive) prompted with a ~40-noun furniture/structure vocabulary, boxes handed to SAM 2 (Apache 2.0) which propagates instance masks across the video with persistent IDs. **Track-then-lift:** the track ID *is* the instance; each physical object is lifted to 3D exactly once, from many views, with median aggregation — duplicates are prevented structurally rather than merged heuristically. New detections that don't overlap an existing propagated mask spawn new tracks (Grounded-SAM-2 pattern). Research track: SAM 3 replaces both models (text → tracked instances natively). Fallback if tracking fails on handheld motion (spike S1): per-keyframe detection + view-consensus mask clustering (MaskClustering pattern).

### 3.3 Orientation
- *Objects:* Orient Anything V2 (CC-BY-4.0) azimuth from best crops → world yaw via the crop frame's ARKit pose → circular median across views → reconciled with the 3D footprint's principal axis (geometry constrains the axis; the model resolves the 180° front/back flip). Symmetric categories (round tables, stools) get axis-only or no yaw.
- *Room:* gravity vector defines up; Manhattan yaw from the dominant orthogonal pair in the wall-normal histogram (ARKit planes + RANSAC planes). PCA is banned. Consistency across capture is inherited from the single ARKit world frame — no stage re-estimates cameras, so nothing can drift between stages.

### 3.4 Assembling the room
Single coordinate frame: ARKit world, re-based to gravity-aligned Manhattan frame, Y-up for SceneKit. RoomPayload v2:

```json
{
  "shell": { "floor_polygon": [[x,z]…], "height": h,
             "walls": [{ "p0": …, "p1": …, "openings": [{type, rect, confidence}] }] },
  "objects": [{ "id", "category", "position", "dimensions", "yaw",
                "confidence", "asset_ref", "best_crops": […], "provenance" }]
}
```

- Furniture geometry = stock USDZ swap (Kenney CC0 now; Objaverse CC0/CC-BY-filtered catalog later), scaled non-uniformly to measured per-instance extents (fixes the dimension-mismatch issue at the source). Real segmented scan geometry is rejected for manipulables: backs are unobserved, so moved objects would expose holes.
- Conflict resolution: objects clipped to the shell (existing wall-clamp logic); overlapping same-category objects already merged at track level; different-category overlap is allowed (monitor on desk). Wall-snapping is demoted to a QA-time option, not a default mutation.
- Doors/windows render as wall cutouts/inserts in the shell, not as movable objects.
- Every object carries confidence + crops so the Plan 4a correction UI can show evidence and allow delete/rotate/move — the user is the last-mile fixer for the known-flaky cases (windows especially).

## 4. Dual license tracks

| Stage | Commercial (default build) | Research-only extras (flagged) |
|---|---|---|
| Geometry | MoGe-2 (MIT) + Open3D TSDF (MIT); MapAnything-apache / DA3Metric-Large (Apache) as A/B | CUT3R, Pow3R as benchmarks |
| Instances | LLMDet (Apache) → SAM 2 (Apache) | SAM 3 (gated, simpler + stronger) |
| Orientation | Orient Anything V2 (CC-BY-4.0) | same |
| Shell | classical (ARKit + RANSAC + gravity) | + SpatialLM 1.1 second opinion for doors/windows (CC-BY-NC) |
| Furniture assets | Kenney CC0; Objaverse CC0/CC-BY subset | same |
| Phase-2 mesh gen | TRELLIS / InstantMesh / SceneGen / PartCrafter-Scene (MIT/Apache) | same (already permissive) |

The default build is the commercial track — the permissive picks are also (except SAM 3/SpatialLM) the technically best ones, so the tracks diverge far less than expected. Avoid: YOLO-World/YOLOE (GPL/AGPL), Grounding DINO 1.5+/DINO-X (API-only), Depth Pro weights (apple-amlr), MASt3R family (NC), Hunyuan3D (Korea-excluded), Amazon Berkeley Objects (contradictory license).

## 5. Open questions from the brief — resolved

1. **LiDAR devices:** yes, separate fast path eventually — RoomPlan's CapturedRoom maps cleanly onto RoomPayload v2 (walls/doors/windows/objects with dims+transforms). Deferred to Phase 3; the scene format is designed to accept it now.
2. **On-device vs server:** phone = capture + ARKit tracking + live feedback; server = all perception (§2 rationale).
3. **Furniture geometry:** stock swap for MVP (complete, movable, license-clean); generative per-object meshes as Phase 2 for photo-fidelity; raw scan geometry rejected (incomplete backs).

## 6. Phasing

- **Phase 0 — spikes (gate everything; ~1 day of GPU work, one real scan of the user's room):**
  - S1: SAM 2 mask propagation quality on real handheld scan video (the architecture's biggest bet).
  - S2: MoGe-2+TSDF cloud vs the current on-device cloud, side by side.
  - S3: LLMDet vocabulary coverage + per-frame runtime on our keyframes.
  - S4 (phase-2 gate, optional): TRELLIS/InstantMesh on 5–10 real furniture crops.
- **Phase 1 — MVP pipeline:** upload changes (video + trajectory), backend pipeline v2 (stages 1–6), RoomPayload v2, viewer glue. Keep the old pipeline runnable until v2 passes device QA.
- **Phase 2 — generative furniture (if S4 passes):** per-object mesh generation replacing/augmenting stock swap, same object interface (`asset_ref` → generated GLB→USDZ).
- **Phase 3 — RoomPlan fast path + asset catalog upgrade + wall/dimension editing hooks (Plans 5/6 land here naturally: measured extents and editable shell are first-class in v2).**

Delta vs current code — kept: VideoRecorderView capture UX, ARKit collection, SceneKit viewer, Plan 4a correction UI, Kenney assets, FastAPI skeleton, RunPod GPU workflow. Retired: on-device YoloRunner/ObjectTracker/depth-fusion as pipeline inputs (DepthRunner may remain for live feedback only), backend consensus/dedup layer (superseded by track-then-lift), SpatialLM as a required component.

## 7. Risks & kill-switches

| Risk | Signal | Response |
|---|---|---|
| SAM 2 tracking breaks on fast handheld motion | S1 | per-keyframe detect + view-consensus clustering fallback |
| MoGe-2+TSDF no better than current cloud | S2 | keep on-device geometry; backend still gains open-vocab instances + orientation |
| LLMDet too slow / misses categories | S3 | OWLv2 / Florence-2 / MM-Grounding-DINO (all permissive) |
| Generative meshes ugly on real crops | S4 | stay on stock swap; revisit in 6 months |
| Upload size hurts on cellular | QA | 720p video or 2 fps frame pack option |
| Windows flaky (known) | QA | correction UI is the mitigation; no model solves this today |

## 8. License citations (verified 2026-07-02)

MoGe-2 MIT (github.com/microsoft/MoGe; hf.co/Ruicheng/moge-2-vitl) · Open3D MIT (github.com/isl-org/Open3D) · MapAnything-apache (hf.co/facebook/map-anything-apache) · Depth Anything V3 Apache for Small/Base/DA3Metric-Large (github.com/bytedance-seed/depth-anything-3) · LLMDet Apache-2.0 (github.com/iSEE-Laboratory/LLMDet) · SAM 2 Apache-2.0 (github.com/facebookresearch/sam2) · SAM 3 "SAM License", gated (hf.co/facebook/sam3) · Orient Anything V1/V2 CC-BY-4.0 (github.com/SpatialVision/Orient-Anything; orient-anythingv2.github.io) · SpatialLM weights CC-BY-NC-4.0 incl. encoder (hf.co/manycore-research/SpatialLM1.1-Qwen-0.5B) · SceneScript CC-BY-NC (github.com/facebookresearch/scenescript) · TRELLIS MIT (github.com/microsoft/TRELLIS) · InstantMesh Apache-2.0 (github.com/TencentARC/InstantMesh) · SceneGen MIT (hf.co/haoningwu/SceneGen) · PartCrafter MIT (hf.co/wgsxm/PartCrafter-Scene) · MIDI Apache-2.0 (VAST-AI) · RoomFormer MIT (github.com/ywyue/RoomFormer) · Kenney CC0 (kenney.nl) · Depth Pro apple-amlr research-only (hf.co/apple/DepthPro) · YOLO-World GPL-v3 / YOLOE AGPL-3.0 · Grounding DINO 1.5+/DINO-X API-only (github.com/IDEA-Research/*-API).
