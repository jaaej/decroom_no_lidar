# Decroom Clean-Slate Scan Pipeline — Design (2026-07-02, rev 2 post-adversarial-review)

**Status:** draft rev 2, pending user review. Rev 2 incorporates a 7-agent adversarial review (3 research tracks + 3 paired devil's advocates + 1 devil's advocate on the synthesis). Changes vs rev 1 are marked ⟲.
**Framing:** clean-slate re-derivation of the scan→3D-room architecture. Past failures are evidence, not constraints. Supersedes the 2026-04-16 phone-first spec where they conflict.
**Constraints:** non-LiDAR iPhone baseline; pre-trained off-the-shelf models only; inputs producible from an iPhone scan; results target 30–60 s perceived (spike-contingent, see §7); output = discrete, manipulable 3D objects (shell + per-furniture). Dual license tracks: commercial-safe and research-best.

**⟲ Honest top-line from the review:** the capture path and the model roster are *validated* (camera-compat and license/access findings all confirmed or corrected below). The architecture's one distinguishing bet — that image-space tracking (track-then-lift) fixes duplicates where 3D-centroid clustering could not — is **unproven and spike-contingent**: image-space association was tried once before (ByteTrack in the old backend) and duplicates still occurred, SAM 2's documented weakest capability is exactly our #1 recorded fragmentation cause (look-away-and-return re-identification), and if spike S1 fails, the fallback degrades toward lift-then-cluster — the mechanism class that already over-counts in the live pipeline. S1 is therefore the gate for the whole plan, not a checkbox.

---

## 1. Evidence base

From project history (verified against live code during review):
- ARKit poses are metric and reliable; classical SfM (COLMAP) on casual phone video fragments badly. → Trust ARKit, never re-solve cameras.
- PCA on point clouds locks to the camera-path diagonal (the 45° rotation bug). → Align to gravity + wall normals, never PCA.
- Monocular-depth point clouds are smooth/normal-poor vs RGB-D scans; SpatialLM's furniture head never fires on them (6 fix attempts exhausted). → Don't feed RGB-D-trained point-cloud models our clouds and expect furniture.
- ⟲ **Open-vocab RGB detection is already implemented and live** (Plan 3c/3d: Grounding DINO on keyframes, `phone_first_pipeline.py:857-882` → `rgb_detection.py` → `raycast_lift.py` → `keyframe_clustering.py`), contrary to stale memory that said it was never built. Its recorded outcome: category coverage works; lift-then-cluster association over-counts (~5 chairs/lamps/monitors for 1 real). → Detection is not the gap; **association is**. v2's detection stage is a retry-with-different-association of a live stage, not new ground — and its scan artifacts can pre-answer parts of spike S3.
- ⟲ Image-space association was also tried before: the pre-phone-first backend used ByteTrack, and duplicates still occurred. SAM 2 is a far stronger instance of the same mechanism class — which raises, not lowers, the burden of proof on S1.
- Lifting per-frame detections to 3D then clustering centroids cannot resolve duplicates at any threshold: depth noise scatters one object's centroid 0.5–1 m; live code carries a whole tuned family of merge distances (0.6 / 0.8 / 0.9 / 1.0 m across `consensus.py`, `keyframe_clustering.py`, `config.py`) and still over-counts.
- COCO-80 vocabulary misses most furniture. → Open-vocabulary detection required.

From the model research + adversarial verification (licenses independently re-verified on repos/HF; citations §8):
- Most learned 3D models (VGGT, Pi3, MASt3R family, CUT3R/Spann3R/SLAM3R/Fast3R) **discard known poses** and re-estimate their own; most have research-only weights. The pose-consuming, commercial-clean stack is MoGe-2 (MIT) + Open3D TSDF (MIT), with MapAnything-apache (Apache-2.0) as the learned alternative.
- SAM 2 (Apache-2.0) video propagation + LLMDet (Apache-2.0) boxes = the permissive text→tracked-instances stack. SAM 3 (gated, custom Meta license with no NC clause — commercial permitted *by omission*, needs a legal read) does it in one model; research-flagged for now.
- ⟲ **Orient Anything V2 is NOT commercially usable**: its repo vendors Meta VGGT and its 5.05 GB checkpoint is near-certainly fine-tuned VGGT-1B (5.03 GB), whose weights are CC-BY-NC-4.0 — the authors' CC-BY-4.0 label cannot relicense that. Commercial track uses **Orient Anything V1** (`Viglong/Orient-Anything`, verified CC-BY-4.0, DINOv2/Apache backbone). Graduation path exists: `facebook/VGGT-1B-Commercial` (gated). Note the asymmetry vs SAM 3 is deliberate conservatism: SAM 3's license is Meta's own grant over its own weights; OAv2 is a third party's permissive label over Meta NC weights — a redistribution-compliance risk, not just a license read.
- SpatialLM: every released checkpoint has a CC-BY-NC point encoder → research-only, any variant.
- ⟲ Generative mesh models, corrected: **MIDI (Apache-2.0, cleanest chain)** and **PartCrafter-Scene (MIT + TripoSG MIT, geometry-only/untextured; its recommended real-photo bridge is a Gemini-API style-transfer step — flag for S4)** lead. **TRELLIS (MIT models) is blocked for commercial GLB export as-is** — the export path runs through nvdiffrast (NVIDIA non-commercial) and diffoctreerast (Inria research license); shipping requires a renderer swap (gsplat etc.), which is real engineering. **InstantMesh is removed** (its Zero123++ stage is CC-BY-NC). SceneGen (code: `Mengmouxu/SceneGen`, MIT) is a TRELLIS derivative sharing the renderer problem.
- Apple RoomPlan is LiDAR-only. Hunyuan3D licenses exclude South Korea — disqualified.

## 2. Architecture decision

**Approach A: tracked-instance pipeline + stock furniture swap, with generative furniture meshes as a spike-gated phase 2 behind the same interface.**

Rejected as backbone: SpatialLM/SceneScript-centric holistic (license-dead commercially; furniture-dead on our clouds; adds nothing over classical shell) and single-image scene generation as backbone (ignores multi-view evidence; placement unproven; right tool as Phase-2 mesh generator only).

**Division of labor:** the phone does capture + ARKit tracking + live scan feedback only; the backend GPU does all perception. The 2026-04-16 "phone does geometry" premise existed because *backend SfM* failed — with ARKit poses uploaded, backend geometry is not SfM. Every chosen model is GPU-scale, and removing on-device depth/YOLO fusion removes both complexity and the noisy-centroid input that the old association drowned in.

**⟲ The central bet, stated plainly:** v2's claimed advantage over the live pipeline is that association happens in image space *before* lifting, so depth noise cannot fragment instances. This is a hypothesis with one prior partial-failure in its mechanism class (ByteTrack). The residual 3D merge (below) remains heuristic — rev 1's claim that duplicates are "prevented structurally rather than merged heuristically" was overstated and is retracted; the correct claim is: **the primary association signal moves to image space, and the residual heuristic merge runs on far cleaner inputs (median-aggregated track centroids instead of per-frame lifts).** Whether that combination actually kills the over-counting is what S1 measures, with pass criteria in §6.

### Data flow

```
PHONE (capture, real-time)
  ARKit session → HEVC video + per-frame JSONL {timestamp, pose, intrinsics, trackingState}
                + plane anchors + sparse feature points (gravity is implicit: world Y-up)
  Existing VideoRecorderView UX unchanged (see §3.4 note on coverage signal)
  Upload ~20–40 MB, streamed as fMP4 segments during capture (spike S6)

BACKEND (GPU; perceived-wait target 30–60 s — spike-contingent, hard cap 2 min)
  1. Demux/decode; join frames↔poses on timestamp; drop trackingState==limited frames
  2. GEOMETRY   MoGe-2 (moge-2-vitl-normal) depth+normals per depth-frame
                → per-frame metric anchor: median vs ARKit sparse points;
                  windowed/global-scale fallback when <~10 points in frame (low-texture walls)
                → Open3D TSDF fusion (2–4 cm voxel) → dense cloud + mesh
  3. INSTANCES  LLMDet (furniture+door+window vocabulary) on detection keyframes
                → boxes prompt SAM 2 → masks propagated across video (persistent IDs;
                  MP4 consumed directly); unmatched detections spawn new tracks
                → per track: masked-depth back-projection per frame via ARKit pose
                → median centroid, robust extents
                → residual merge: same label + footprint-overlap PRIMARY criterion,
                  centroid gate ≥0.8 m for large categories (0.4–0.6 m small), per the
                  live-code lesson (0.6–1.0 m family) — never tighter than recorded evidence
  4. ORIENTATION Orient Anything V1 on 3 best crops per track (largest mask, most frontal,
                EXIF-sharpest) → azimuth → world yaw via pose → circular median
                → footprint-axis reconciliation ONLY where the track's masks covered ≥2 sides
                  of the object (one-sided clusters are surface-biased — per-object PCA on
                  them re-creates the wrong-facing bug); else trust the model or emit no yaw
                → symmetric categories: axis-only or no yaw
  5. SHELL      consolidate ARKit vertical planes (≤5°, ≤10 cm coplanar merge)
                + RANSAC planes on TSDF cloud where anchors missing (load-bearing on
                  non-Pro devices — anchors are weak on bare walls without LiDAR)
                → gravity = up; Manhattan yaw = dominant orthogonal wall-normal pair
                → floor/ceiling from horizontal planes; rectilinear wall polygon
                → openings: door/window tracks projected onto nearest wall plane
                  (doors touch floor; windows don't)
  6. ASSEMBLY   RoomPayload v2 (single gravity-aligned Manhattan frame, Y-up)

PHONE (render + edit)
  SceneKit viewer + Plan 4a correction UI, MIGRATED to decode RoomPayload v2 (§6 Phase 1 —
  explicit work, not free: yaw rename/handedness, dimensions restructure)
  Stock asset per category (Kenney CC0), scaled to measured extents
```

**⟲ Capture requirements (from camera-compat verification — all consumable formats confirmed):**
- **Sync**: key everything by `frame.timestamp` (= video PTS). Never frame index — ARKit drops frames under load.
- **Buffer ownership**: copy `capturedImage` into an encoder-owned pixel-buffer pool before appending. Retaining ARKit's pool buffers under encoder backpressure starves tracking itself — the likeliest silent-degradation mode of this design, spike-priority alongside S6.
- **Pose-quality gating**: record per-frame `ARCamera.trackingState`; backend drops/downweights `.limited` frames from fusion and lifting.
- **One canonical orientation convention** for frames + intrinsics + masks, asserted in Phase 0. `capturedImage` is landscape-right; detector/orientation models need upright images → rotate once server-side + swap intrinsics (the existing CLAUDE.md formula, generalized). SAM 2 must see the same orientation as the detector prompting it.
- **Convention flip**: ARKit camera looks down −Z, OpenCV/Open3D/MapAnything +Z → one fixed 180°-about-X on extrinsics.
- **Rate/format**: append every 2nd frame (~30 fps effective — backend samples ≤5 fps; 60 fps is inherited waste). Encode 4:3 sensor-native (no 16:9 crop — it costs vertical FOV where floor/ceiling live); 1440p-vs-1080p decided by bitrate math in Phase 1 against the 20–40 MB budget. Set explicit color-space tags on AVAssetWriter (full-range 420, correct YCbCr matrix). Leave `videoHDRAllowed` off (keeps 8-bit; HDR costs performance).
- **Mid-scan robustness**: handle session interruption (call/backgrounding) and lost/out-of-order fMP4 segments — resumable upload with segment manifest; a scan that survives capture must survive transport (part of S6).
- Sampling rates (canonical set, tunable by spikes): **2 fps depth / 5 fps tracking / 1 fps detection**, 720p inference for detection+tracking.

## 3. The four sub-problems

### 3.1 Depth / 3D structure
Signals: RGB video + ARKit poses/intrinsics (metric, per-frame) + gravity + sparse points. **MoGe-2 `moge-2-vitl-normal`** (MIT — the base `-vitl` checkpoint has NO normals ⟲) predicts metric-ish depth + normals; feed per-frame FOV from ARKit intrinsics (`fov_x`) instead of letting it estimate. Per-frame median scale vs ARKit sparse-point depths, **with a windowed/global-scale fallback for low-texture frames** (bare walls can drop feature counts near zero ⟲ — a design requirement, not an edge case). Open3D TSDF fusion under known poses. No SfM anywhere. ARKit frames are pinhole-consumable (rectified per community consensus — but this rests on forum-grade evidence, so S2 includes a `rawFeaturePoints` reprojection check ⟲). A/B behind a flag: MapAnything-apache (Apache, natively pose-consuming), DA3 `DA3METRIC-LARGE` (Apache — beware: `DA3-LARGE`/`GIANT` are CC-BY-NC ⟲). Research benchmark: CUT3R (NC).

### 3.2 Detection (instances, not blobs)
LLMDet (`iSEE-Laboratory/llmdet_large` — the `*_hf` names are dead ⟲; native `transformers` ≥4.55 class) prompted with a ~40-noun vocabulary; boxes prompt SAM 2 (`sam2.1_hiera_large`), which propagates instance masks across the MP4 with persistent IDs. Track-then-lift: each track lifts to 3D once, from many views, median-aggregated. **⟲ Qualified claim (see §2):** primary association is image-space and depth-noise-immune; the residual merge stays heuristic but runs on median-aggregated centroids with footprint-overlap as the primary criterion and gates no tighter than the recorded 0.8 m lesson for large categories. Known weak spot, now the S1 pass criterion: **look-away-and-return re-identification** — SAM 2's memory can fail to re-acquire after long gaps (documented in the SAM 2 issue tracker and follow-up literature), which is exactly our recorded #1 fragmentation cause. Research track: SAM 3 (gated) replaces both models. Fallback if S1 fails: per-keyframe detection + view-consensus mask clustering (MaskClustering pattern) — **⟲ honestly labeled: this is a cousin of the lift-then-cluster approach that already over-counts; it votes on multi-view mask overlap rather than 3D centroids, which may or may not be enough. If S1 fails, expect a real fight here, not a soft landing.**

### 3.3 Orientation
- *Objects:* **Orient Anything V1** (CC-BY-4.0 ⟲ — V2 demoted to research track, see §1) azimuth from best crops → world yaw via pose → circular median. **⟲ Footprint-axis reconciliation only where masks covered ≥2 sides**; one-sided clusters are surface-biased and per-object PCA on them re-creates wrong-facing. Symmetric categories get axis-only or no yaw. V1's real-photo quality on cluttered crops is unmeasured → **new spike S5**.
- *Room:* gravity defines up (implicit in ARKit world under `.gravity` alignment); Manhattan yaw from dominant orthogonal wall-normal pair. PCA banned at room level. Consistency inherited from the single ARKit world frame — no stage re-estimates cameras.

### 3.4 Assembling the room
Single frame: ARKit world → gravity-aligned Manhattan, Y-up for SceneKit. RoomPayload v2:

```json
{
  "coordinate_conventions": { "up": "+Y", "yaw": "CCW-positive about +Y, radians;
      SceneKit consumer negates for eulerAngles.y (CW-positive)" },
  "shell": { "floor_polygon": [[x,z]…], "height": h,
             "walls": [{ "p0": …, "p1": …, "openings": [{type, rect, confidence}] }] },
  "objects": [{ "id", "category", "position", "dimensions", "yaw",
                "confidence", "asset_ref", "best_crops": […], "provenance" }]
}
```

- ⟲ Yaw handedness is stated IN the schema (above) — the recorded negate-for-SceneKit rule, made explicit so v2 doesn't re-debug "bed facing wrong way."
- Furniture geometry = stock USDZ swap (Kenney CC0 now; license-filtered Objaverse later), scaled to measured extents. **⟲ Guards:** keep the existing yOffset-to-floor logic (already in the viewer's `loadUSDZModel`); per-axis non-uniform scale is new — clamp per-axis ratios so noisy extents can't stretch a chair into a wardrobe; S2 scores per-category extent accuracy against tape measure before "fixes dimension mismatch" is claimed.
- Conflict resolution: objects clipped to shell (existing wall-clamp logic); same-category overlap handled at merge; cross-category overlap allowed (monitor on desk). Wall-snapping demoted to QA-time option (its original motivation — bad SfM poses — is gone; monocular range error remains, so keep the option).
- Doors/windows are shell cutouts, not movable objects. Every object carries confidence + crops for the correction UI; the user is the last-mile fixer (windows especially).
- **⟲ Viewer/correction-UI migration is explicit Phase-1 work** — RoomPayload v2 renames `rotation_y`→`yaw` and restructures dimensions; the "kept" UI (itself still in Plan-4a QA flux) needs a decode migration, not zero work.
- **⟲ Coverage signal**: the kept coverage arc is a recorded-weak signal. v2 defines "scan complete" as multi-view baseline coverage (every region seen from ≥2 sufficiently-separated poses — what tracking and TSDF actually need), not point counts. UX unchanged; the readiness math behind it changes.

## 4. Dual license tracks

| Stage | Commercial (default build) | Research-only extras (flagged) |
|---|---|---|
| Geometry | MoGe-2 `moge-2-vitl-normal` (MIT) + Open3D TSDF (MIT); A/B: MapAnything-apache, DA3METRIC-LARGE (Apache) | CUT3R, Pow3R benchmarks |
| Instances | LLMDet `llmdet_large` (Apache) → SAM 2.1 (Apache) | SAM 3 (gated; no NC clause — may graduate after legal read ⟲) |
| Orientation | **Orient Anything V1** (CC-BY-4.0) ⟲ | Orient Anything V2 (VGGT-NC chain ⟲) |
| Shell | classical (ARKit + RANSAC + gravity) | + SpatialLM 1.1 door/window second opinion (CC-BY-NC) |
| Furniture assets | Kenney CC0; Objaverse CC0/CC-BY subset | same |
| Phase-2 mesh gen | **MIDI (Apache), PartCrafter-Scene (MIT, untextured; Gemini-API real-photo bridge flagged)** first; TRELLIS only with renderer swap; ~~InstantMesh~~ removed (zero123plus NC); SceneGen = TRELLIS-derivative ⟲ | same |

**⟲ Environment (verified):** one env — torch ≥2.5.1, transformers ≥4.55 (LLMDet native class), sam2 pkg, MoGe-2, Open3D, MapAnything (unpinned) — runs the whole Phase-1 perception stack; ~10–13 GB peak VRAM concurrent (fits RTX 4090 24 GB; A40 48 GB trivially). Phase-2 generators (TRELLIS/SceneGen: torch 2.4/CUDA 11.8 + 5 custom kernels) get an isolated container; PartCrafter/MIDI could share the perception env. LLMDet speed is unpublished — budget 250–400 ms/keyframe until S3 measures. Only gated model anywhere: SAM 3.

Avoid list: YOLO-World/YOLOE (GPL/AGPL), Grounding DINO 1.5+/DINO-X (API-only), Depth Pro weights (research-only), MASt3R family (NC), Hunyuan3D (Korea-excluded), Amazon Berkeley Objects (contradictory license), InstantMesh (NC dep), Orient Anything V2 commercially (NC chain).

## 5. Open questions from the brief — resolved

1. **LiDAR devices:** RoomPlan fast path in Phase 3; RoomPayload v2 designed to accept CapturedRoom mapping now.
2. **On-device vs server:** phone = capture + tracking + feedback; server = all perception (§2).
3. **Furniture geometry:** stock swap MVP; generative Phase 2 (spike-gated); raw scan geometry rejected (incomplete backs).

## 6. Phasing

- **Phase 0 — spikes (gate everything; a few days of GPU work, one real scan of the user's room, plus mined Plan 3c/3d artifacts ⟲):**
  - **S1 (THE gate):** SAM 2 propagation on real handheld scan video. Pass criteria ⟲: (a) look-away-and-return re-identification on a full 360° loop (every object revisited once), (b) measured inter-track centroid spread for one physical object (validates/refutes the ≥0.8 m residual gate), (c) peak VRAM + wall-clock on multi-minute video.
  - **S2:** MoGe-2+TSDF vs current on-device cloud; per-category extent accuracy vs tape measure ⟲; `rawFeaturePoints` reprojection check (rectification assumption) ⟲.
  - **S3:** LLMDet vocabulary coverage + per-frame runtime — pre-seeded with existing Plan 3c/3d Grounding-DINO scan results ⟲.
  - **S4 (Phase-2 gate):** MIDI / PartCrafter-Scene on 5–10 real furniture crops (note PartCrafter's Gemini-API bridge) ⟲.
  - **S5 ⟲:** Orient Anything V1 yaw quality on real cluttered crops (the V2→V1 swap cost, unmeasured).
  - **S6 ⟲:** fMP4 streaming upload during a live ARKit session — including buffer-pool pressure on tracking, thermal watch, interruption/segment-loss recovery.
- **Phase 1 — MVP pipeline:** capture changes (video + JSONL + streaming), backend pipeline v2 (stages 1–6), RoomPayload v2, **viewer + correction-UI decode migration ⟲**. Old pipeline stays runnable until v2 passes device QA. **Plan 3e is killed** (moot against dead code ⟲). **QA must include a non-Pro device** — LiDAR silently upgrades plane anchors on Pro QA phones ⟲.
- **Phase 2 — generative furniture** (if S4 passes): per-object meshes behind the same `asset_ref` interface.
- **Phase 3 — RoomPlan fast path + asset catalog upgrade + wall/dimension editing (Plans 5/6 land here).**

Delta vs current code — kept: VideoRecorderView UX, ARKit collection, SceneKit viewer + correction UI (with v2 decode migration), Kenney assets, FastAPI skeleton, RunPod workflow. Retired: on-device YoloRunner/ObjectTracker/depth-fusion as pipeline inputs, backend consensus/dedup layer, the live Plan 3c/3d lift-then-cluster path (superseded by stage 3), SpatialLM as required component.

## 7. Risks & kill-switches

| Risk | Signal | Response |
|---|---|---|
| **SAM 2 re-id fails on look-away-return (recorded #1 fragmentation cause; documented SAM 2 weakness)** | S1 | fallback = view-consensus mask clustering — a lift-then-cluster cousin; treat S1 failure as a genuine setback requiring redesign of association, not a lever pull ⟲ |
| Encoder buffer retention degrades ARKit tracking | S6 | encoder-owned buffer pool; frame-skip appending; bounded queue |
| MoGe-2+TSDF no better than current cloud | S2 | keep on-device geometry; backend still gains instances + orientation |
| Extent accuracy insufficient for per-axis scaling | S2 | uniform scale + category priors (Plan-6 territory) |
| LLMDet too slow / misses categories | S3 | MM-Grounding-DINO (same transformers class, zero-code swap), OWLv2, Florence-2 |
| Orient Anything V1 poor on real crops | S5 | footprint-axis-only where ≥2 sides seen; else no yaw + correction UI |
| Generative meshes poor on real crops | S4 | stay stock-swap; revisit in 6 months |
| **30–60 s target misses** (every stage timing is currently unmeasured ⟲) | S1/S3 wall-clocks | sampling-rate cuts, smaller SAM 2 variant (base+ = 64 fps vs large 39.5), faster GPU tier; 2 min hard cap |
| Upload size hurts on cellular | QA | 720p or frame-pack option |
| Windows flaky (known) | QA | correction UI is the mitigation; no model solves this today |

## 8. License & access citations (independently verified 2026-07-02, adversarial pass)

MoGe-2 `Ruicheng/moge-2-vitl-normal` MIT ungated (github.com/microsoft/MoGe) · Open3D MIT (github.com/isl-org/Open3D) · `facebook/map-anything-apache` Apache-2.0 ungated 4.91 GB · `depth-anything/DA3METRIC-LARGE` Apache-2.0 (DA3-LARGE/GIANT are CC-BY-NC) · LLMDet `iSEE-Laboratory/llmdet_{base,large}` Apache-2.0 ungated, transformers ≥4.55 native (`*_hf` names dead) · SAM 2.1 Apache-2.0 ungated (`facebook/sam2.1-hiera-large`) · SAM 3 gated-manual, custom Meta license, no NC clause (hf.co/facebook/sam3) · Orient Anything V1 `Viglong/Orient-Anything` CC-BY-4.0 (DINOv2 Apache backbone; OriNet is a redirect; Base ckpt link in README is broken — use `cropbaseEx03`) · Orient Anything V2 `Viglong/OriAnyV2_ckpt` 5.05 GB — vendors VGGT, ckpt ≈ VGGT-1B (CC-BY-NC weights) under a CC-BY-4.0 label; VGGT code itself commercial-OK since 2025-07; `facebook/VGGT-1B-Commercial` gated graduation path · SpatialLM 1.1 CC-BY-NC-4.0 · TRELLIS `microsoft/TRELLIS-image-large` MIT models, GLB export via nvdiffrast (NVIDIA 1-way commercial = NC) + diffoctreerast (Inria research) — FlexiCubes is Apache · InstantMesh Apache own-weights but zero123plus dep CC-BY-NC ("cannot use in a commercial product pipeline") → removed · SceneGen code `Mengmouxu/SceneGen` MIT, TRELLIS-derivative · PartCrafter `wgsxm/PartCrafter(-Scene)` MIT + TripoSG MIT, untextured, Gemini-API real-photo bridge · MIDI `VAST-AI/MIDI-3D` Apache-2.0 · Kenney CC0 · Avoid: Depth Pro (apple-amlr), YOLO-World GPL / YOLOE AGPL, GD-1.5+/DINO-X API-only, MASt3R NC, Hunyuan3D (Korea-excluded), ABO (contradictory).

Caveat carried from the review: HF license tags are uploader-self-declared — before shipping, a human legal read of the actual license files for MoGe-2, LLMDet, SAM 2, and Orient Anything V1 is the final gate.
