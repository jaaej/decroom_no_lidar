# Phase 0 Spike Plan — Clean-Slate Scan Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, on one real scan of the user's room, whether the rev-2 pipeline's central bets hold — gate S1 = end-to-end furniture instance count — before any Phase-1 code is written.

**Architecture:** Throwaway measurement harnesses, not product code. A kill-screen (camera-app video, no iOS work) runs first and can stop everything cheaply. Then a minimal ARKit debug recorder provides the pose-annotated scan that spikes S1/S2/S3/S5 consume on a RunPod GPU. Every spike writes numbers into a single results file; the final task turns them into a go/no-go.

**Tech Stack:** Python 3.11 on RunPod GPU (torch ≥2.5.1, transformers ≥4.55, sam2, MoGe, Open3D), Swift/ARKit for the debug recorder.

## Global Constraints

- Single GPU env for all perception spikes: `torch>=2.5.1`, `transformers>=4.55` (spec §4 Environment).
- Exact checkpoints (spec §8): `iSEE-Laboratory/llmdet_large`, `facebook/sam2.1-hiera-large` (sam2.1_hiera_large.pt), `Ruicheng/moge-2-vitl-normal` (NOT `-vitl`), Orient Anything V1 from `Viglong/Orient-Anything` (Base ckpt file is `cropbaseEx03`, README link is broken).
- ARKit→OpenCV: 180°-about-X flip on camera pose before any OpenCV/Open3D use (spec §2 capture requirements).
- `capturedImage` is landscape-right: rotate frames 90° CW server-side + swap intrinsics before detection/orientation models (spec §2 / CLAUDE.md).
- Residual merge gates: footprint-overlap primary, centroid ≥0.8 m for large categories — never tighter (spec §2 stage 3).
- git: local commits only, never push without explicit user confirmation.
- Do NOT edit `project.pbxproj` for new .swift files (PBXFileSystemSynchronizedRootGroup).
- Spike scripts live in `decroom_no_lidar/backend/spikes/phase0/`; results in `docs/superpowers/plans/phase0-results.md`.
- S1 pass gate (spec §6): ≤1 spurious instance per real object after the FULL chain.

---

### Task 1: GPU environment + checkpoint verification (RunPod)

**Files:**
- Create: `decroom_no_lidar/backend/spikes/phase0/setup_env.sh`
- Create: `decroom_no_lidar/backend/spikes/phase0/verify_env.py`

**Interfaces:**
- Produces: a conda env `phase0` on the pod in which every later script runs; local checkpoint cache under `~/ckpts/`.

The user starts a RunPod pod (A40 48GB preferred, 4090 24GB acceptable — see `runpod_setup.md` memory for pod workflow; this is a NEW env, do not reuse the SpatialLM env).

- [ ] **Step 1: Write `setup_env.sh`**

```bash
#!/usr/bin/env bash
set -euxo pipefail
# Run ON THE POD. Python 3.11, torch >= 2.5.1 (spec constraint).
conda create -y -n phase0 python=3.11
source activate phase0
pip install "torch>=2.5.1" torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.55" accelerate pillow opencv-python-headless open3d numpy scipy av decord huggingface_hub matplotlib
# SAM 2 (video predictor needs the repo package, not just HF weights)
git clone https://github.com/facebookresearch/sam2 ~/sam2 && pip install -e ~/sam2
# MoGe
pip install git+https://github.com/microsoft/MoGe.git
# Orient Anything V1 (repo code, no pip package)
git clone https://github.com/SpatialVision/Orient-Anything ~/orient-anything
mkdir -p ~/ckpts
python - <<'EOF'
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("iSEE-Laboratory/llmdet_large", local_dir="~/ckpts/llmdet_large")
hf_hub_download("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt", local_dir="~/ckpts/sam2")
snapshot_download("Ruicheng/moge-2-vitl-normal", local_dir="~/ckpts/moge2")
# Orient Anything V1 Large ckpt (croplargeEX2); Base fallback file is cropbaseEx03 (README's cropbaseEx032 link is broken)
hf_hub_download("Viglong/Orient-Anything", "croplargeEX2/dino_weight.pt", local_dir="~/ckpts/orient")
EOF
```

- [ ] **Step 2: Write `verify_env.py`** — loads each model once and prints VRAM

```python
import torch, time
def vram(): return f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB"
t=time.time()
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
det = AutoModelForZeroShotObjectDetection.from_pretrained("~/ckpts/llmdet_large", torch_dtype=torch.float16).cuda()
print("LLMDet OK", vram())
from sam2.build_sam import build_sam2_video_predictor
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", "~/ckpts/sam2/sam2.1_hiera_large.pt")
print("SAM2 OK", vram())
from moge.model.v2 import MoGeModel
moge = MoGeModel.from_pretrained("~/ckpts/moge2").cuda()
print("MoGe-2-normal OK", vram(), f"{time.time()-t:.0f}s total")
import open3d; print("Open3D", open3d.__version__)
```

- [ ] **Step 3: scp both files to the pod, run `setup_env.sh` then `verify_env.py`**

Expected: three `OK` lines + a combined VRAM figure. Record the VRAM number in `phase0-results.md` (this is the first data point for spec §4's 10–13GB estimate). If MoGe import path differs (`moge.model.v2` vs `moge.model`), check `~/MoGe/README` usage section and fix the import — record what worked.

- [ ] **Step 4: Commit the two scripts**

```bash
git add decroom_no_lidar/backend/spikes/phase0/ && git commit -m "spike(phase0): env setup + model load verification"
```

---

### Task 2: Kill-screen capture (user action — no code)

**Files:** none (produces `~/scans/killscreen.mp4` on the pod)

**Interfaces:**
- Produces: `killscreen.mp4` — 60–90 s, 1080p+ camera-app video of the user's room.

- [ ] **Step 1: Ask the user to record, with exactly these instructions:**

> Camera app, video, 1080p or 4K. Walk a slow full 360° loop of your room (~60–90 s). Pan slowly (a wall should take ~3 s to cross the frame). Make sure every major furniture item fully leaves the frame and comes back at least once (that's the point of the test). End roughly where you started.

- [ ] **Step 2: Ask the user for a furniture inventory while they're at it** — the ground-truth list: `category, count` for everything scannable (e.g. bed 1, chair 2, monitor 1, desk 1, wardrobe 1, lamp 1). Save it as `decroom_no_lidar/backend/spikes/phase0/ground_truth.json`:

```json
{"objects": [{"category": "bed", "count": 1}, {"category": "chair", "count": 2}]}
```

- [ ] **Step 3: Transfer** — AirDrop to Mac, then `scp killscreen.mp4 root@<pod>:~/scans/`. Commit `ground_truth.json`.

---

### Task 3: S1a kill-screen — fragmentation on camera video (FAIL-only gate)

**Files:**
- Create: `decroom_no_lidar/backend/spikes/phase0/s1a_killscreen.py`

**Interfaces:**
- Consumes: `~/scans/killscreen.mp4`, `ground_truth.json`, Task 1 env.
- Produces: `killscreen_report.json` with `tracks_per_category` and per-track thumbnail grid `killscreen_tracks.png`; reusable functions `detect(frame_bgr, vocab, det, proc) -> list[dict(box,score,label)]` and `run_chain(frames, fps_det, fps_track) -> dict[track_id -> dict(label, frames, masks)]` imported by Task 7.

This is the chain WITHOUT 3D: LLMDet every 1 s → SAM 2 propagation in windows → spawn-on-unmatched. Measures raw track fragmentation. Per spec: camera-app footage is stabilized, so **only a FAIL is decision-grade** (a pass can be optimistic).

- [ ] **Step 1: Write `s1a_killscreen.py`**

```python
"""S1a kill-screen: track fragmentation of LLMDet->SAM2 chain on camera video.
Windowed propagation: at each detection keyframe, match detections to current
SAM2 masks by IoU; unmatched detections spawn new track IDs; then propagate
to the next keyframe. (sam2 issue #185 pattern.)"""
import torch, cv2, json, numpy as np, argparse, time
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2_video_predictor

VOCAB = ["bed","sofa","chair","desk","table","wardrobe","bookshelf","dresser",
         "nightstand","monitor","tv","lamp","rug","mirror","door","window",
         "curtain","shelf","cabinet","plant"]
SPAWN_IOU = 0.3   # detection matches an existing mask if IoU >= this

def mask_iou(a, b):
    inter = np.logical_and(a, b).sum(); union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0

def detect(frame_bgr, vocab, det, proc):
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = proc(images=img, text=[vocab], return_tensors="pt").to("cuda")
    with torch.no_grad(): out = det(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, threshold=0.35, target_sizes=[img.size[::-1]])[0]
    return [{"box": b.tolist(), "score": float(s), "label": vocab[int(l)]}
            for b, s, l in zip(res["boxes"], res["scores"], res["labels"])]

def run_chain(frames_dir, fps_det=1, fps_track=5, native_fps=30):
    det = AutoModelForZeroShotObjectDetection.from_pretrained(
        "~/ckpts/llmdet_large", torch_dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained("~/ckpts/llmdet_large")
    pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml",
                                      "~/ckpts/sam2/sam2.1_hiera_large.pt")
    state = pred.init_state(video_path=frames_dir, offload_video_to_cpu=True,
                            offload_state_to_cpu=True)
    import os
    n_frames = len(os.listdir(frames_dir))
    det_stride = max(1, fps_track // fps_det)   # detection every det_stride track-frames
    tracks, next_id, timings = {}, 0, {"det": [], "prop": []}
    current_masks = {}                           # track_id -> bool mask
    kf_indices = list(range(0, n_frames, det_stride))
    for i, kf in enumerate(kf_indices):
        frame = cv2.imread(os.path.join(frames_dir, f"{kf:05d}.jpg"))
        t0 = time.time(); dets = detect(frame, VOCAB, det, proc); timings["det"].append(time.time()-t0)
        for d in dets:                           # match vs current propagated masks
            x0, y0, x1, y1 = map(int, d["box"])
            dm = np.zeros(frame.shape[:2], bool); dm[y0:y1, x0:x1] = True
            best = max((mask_iou(dm, m) for m in current_masks.values()), default=0.0)
            if best < SPAWN_IOU:                 # unmatched -> new track
                pred.add_new_points_or_box(state, frame_idx=kf, obj_id=next_id,
                                           box=np.array(d["box"], np.float32))
                tracks[next_id] = {"label": d["label"], "born": kf, "frames": []}
                next_id += 1
        t0 = time.time()
        end = kf_indices[i+1] if i+1 < len(kf_indices) else n_frames
        for fidx, obj_ids, logits in pred.propagate_in_video(
                state, start_frame_idx=kf, max_frame_num_to_track=end-kf):
            for oid, lg in zip(obj_ids, logits):
                m = (lg > 0).cpu().numpy().squeeze()
                if m.any():
                    tracks[oid]["frames"].append(int(fidx)); current_masks[oid] = m
        timings["prop"].append(time.time()-t0)
    return tracks, timings

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--fps", type=int, default=5)
    a = ap.parse_args()
    import os, subprocess
    frames = "/tmp/ks_frames"; os.makedirs(frames, exist_ok=True)
    subprocess.run(["ffmpeg","-y","-i",a.video,"-vf",f"fps={a.fps},scale=1280:-2",
                    f"{frames}/%05d.jpg"], check=True)
    tracks, timings = run_chain(frames, fps_det=1, fps_track=a.fps)
    per_cat = {}
    for t in tracks.values():
        if len(t["frames"]) >= 3:                # ignore 1-2 frame ghosts
            per_cat[t["label"]] = per_cat.get(t["label"], 0) + 1
    gt = json.load(open("ground_truth.json"))
    report = {"tracks_per_category": per_cat,
              "ground_truth": {o["category"]: o["count"] for o in gt["objects"]},
              "peak_vram_gb": torch.cuda.max_memory_allocated()/1e9,
              "det_ms_median": float(np.median(timings["det"])*1000),
              "prop_s_total": float(sum(timings["prop"]))}
    json.dump(report, open("killscreen_report.json","w"), indent=2); print(json.dumps(report, indent=2))
```

- [ ] **Step 2: Sanity-run on a 10-second clip first** — `ffmpeg -t 10 -i killscreen.mp4 short.mp4 && python s1a_killscreen.py short.mp4`. Expected: JSON report prints, no crash, at least 1 track. Fix API mismatches here (SAM 2 and transformers signatures drift; the two likely breakpoints are `post_process_grounded_object_detection` arg names and `propagate_in_video` yields).

- [ ] **Step 3: Full run** — `python s1a_killscreen.py ~/scans/killscreen.mp4 --fps 5`. Record in `phase0-results.md`: tracks-per-category vs ground truth, fragmentation ratio (total tracks ÷ total real objects), median LLMDet ms/frame (this is S3's first runtime number), peak VRAM.

- [ ] **Step 4: KILL-SCREEN GATE.** If fragmentation ratio > ~4 (e.g. 8+ tracks per real object) on *stabilized* footage — the chain is dead as designed; STOP, report to user, invoke the S1-failure ladder (spec §6) starting with SAMURAI/DAM4SAM forks before any iOS work. Otherwise proceed. **A good result here is NOT a pass** — record it as "not killed."

- [ ] **Step 5: Commit** — `git add -A decroom_no_lidar/backend/spikes/phase0 && git commit -m "spike(phase0): S1a kill-screen harness + results"`

---

### Task 4: Minimal ARKit debug recorder (iOS)

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/DebugScanRecorder.swift`
- Modify: `decroom_no_lidar/decroom_no_lidar/VideoRecorderView.swift` (hook: start/stop recorder alongside the existing session; add a hidden toggle)

**Interfaces:**
- Consumes: the existing `ARSession` from VideoRecorderView's coordinator (`session.delegate` frames).
- Produces: a share-able folder `scan_<timestamp>/` containing `video.mov` (HEVC, every 2nd frame, buffer-copied) + `frames.jsonl` (one line per appended frame: `{"t": <frame.timestamp>, "pose": [16 floats, column-major ARKit cam-to-world], "K": [fx,fy,cx,cy], "tracking": "normal|limited", "thermal": "nominal|fair|serious|critical"}`) + `points.jsonl` (every 10th frame: `{"t": ..., "pts": [[x,y,z]...]}` from `rawFeaturePoints`, ≤200 pts) + `stats.json` (`{"frames_appended": n, "frames_dropped_by_encoder": n, "thermal_events": [...]}`). This folder is the input format for Tasks 6–10.

This is also **S6-lite**: `stats.json` measures encoder pressure (spec §2 buffer-ownership risk) and thermal. Full fMP4 network streaming stays in Phase 1.

- [ ] **Step 1: Write `DebugScanRecorder.swift`**

```swift
import ARKit
import AVFoundation

/// Phase-0 debug recorder: HEVC video + per-frame pose/intrinsics JSONL.
/// Spike-quality; records every 2nd ARFrame, copies pixel buffers via a
/// pool so ARKit's camera pool is never retained (spec §2).
final class DebugScanRecorder {
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var jsonl: FileHandle?
    private var ptsJsonl: FileHandle?
    private(set) var dir: URL?
    private var frameCounter = 0
    private var appended = 0, dropped = 0
    private var startTime: TimeInterval?

    func start() throws {
        let stamp = Int(Date().timeIntervalSince1970)
        let d = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("scan_\(stamp)", isDirectory: true)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        dir = d
        let w = try AVAssetWriter(outputURL: d.appendingPathComponent("video.mov"), fileType: .mov)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: 1920, AVVideoHeightKey: 1440,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2],
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 8_000_000]]
        let inp = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        inp.expectsMediaDataInRealTime = true
        let ad = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: inp,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String:
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                kCVPixelBufferWidthKey as String: 1920,
                kCVPixelBufferHeightKey as String: 1440])
        w.add(inp)
        w.startWriting()
        writer = w; input = inp; adaptor = ad
        FileManager.default.createFile(atPath: d.appendingPathComponent("frames.jsonl").path, contents: nil)
        jsonl = try FileHandle(forWritingTo: d.appendingPathComponent("frames.jsonl"))
        FileManager.default.createFile(atPath: d.appendingPathComponent("points.jsonl").path, contents: nil)
        ptsJsonl = try FileHandle(forWritingTo: d.appendingPathComponent("points.jsonl"))
    }

    func append(_ frame: ARFrame) {
        frameCounter += 1
        guard frameCounter % 2 == 0, let input = input, let adaptor = adaptor else { return }
        if startTime == nil {
            startTime = frame.timestamp
            writer?.startSession(atSourceTime: CMTime(seconds: 0, preferredTimescale: 600))
        }
        guard input.isReadyForMoreMediaData else { dropped += 1; return }
        guard let pool = adaptor.pixelBufferPool else { dropped += 1; return }
        var copy: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &copy)
        guard let dst = copy else { dropped += 1; return }
        // Copy both planes so ARKit's buffer is released immediately.
        let src = frame.capturedImage
        CVPixelBufferLockBaseAddress(src, .readOnly); CVPixelBufferLockBaseAddress(dst, [])
        for plane in 0..<2 {
            let h = CVPixelBufferGetHeightOfPlane(src, plane)
            let bpr = min(CVPixelBufferGetBytesPerRowOfPlane(src, plane),
                          CVPixelBufferGetBytesPerRowOfPlane(dst, plane))
            for row in 0..<h {
                memcpy(CVPixelBufferGetBaseAddressOfPlane(dst, plane)! + row * CVPixelBufferGetBytesPerRowOfPlane(dst, plane),
                       CVPixelBufferGetBaseAddressOfPlane(src, plane)! + row * CVPixelBufferGetBytesPerRowOfPlane(src, plane), bpr)
            }
        }
        CVPixelBufferUnlockBaseAddress(src, .readOnly); CVPixelBufferUnlockBaseAddress(dst, [])
        let t = frame.timestamp - startTime!
        if adaptor.append(dst, withPresentationTime: CMTime(seconds: t, preferredTimescale: 600)) {
            appended += 1
            let p = frame.camera.transform, K = frame.camera.intrinsics
            let pose = (0..<4).flatMap { c in (0..<4).map { r in p[c][r] } }
            let tracking: String
            switch frame.camera.trackingState {
            case .normal: tracking = "normal"
            default: tracking = "limited"
            }
            let line: [String: Any] = ["t": frame.timestamp, "pose": pose,
                "K": [K[0][0], K[1][1], K[2][0], K[2][1]], "tracking": tracking,
                "thermal": String(describing: ProcessInfo.processInfo.thermalState)]
            if let data = try? JSONSerialization.data(withJSONObject: line) {
                jsonl?.write(data); jsonl?.write("\n".data(using: .utf8)!)
            }
            if frameCounter % 10 == 0, let pts = frame.rawFeaturePoints?.points.prefix(200) {
                let obj: [String: Any] = ["t": frame.timestamp,
                    "pts": pts.map { [$0.x, $0.y, $0.z] }]
                if let data = try? JSONSerialization.data(withJSONObject: obj) {
                    ptsJsonl?.write(data); ptsJsonl?.write("\n".data(using: .utf8)!)
                }
            }
        } else { dropped += 1 }
    }

    func stop(completion: @escaping (URL?) -> Void) {
        guard let w = writer, let d = dir else { completion(nil); return }
        input?.markAsFinished()
        w.finishWriting {
            let stats: [String: Any] = ["frames_appended": self.appended,
                                        "frames_dropped_by_encoder": self.dropped]
            try? JSONSerialization.data(withJSONObject: stats)
                .write(to: d.appendingPathComponent("stats.json"))
            try? self.jsonl?.close(); try? self.ptsJsonl?.close()
            completion(d)
        }
    }
}
```

- [ ] **Step 2: Hook into `VideoRecorderView.swift`** — read the file first; in its ARSession delegate `session(_:didUpdate frame:)` path add `debugRecorder?.append(frame)`; add a long-press gesture (or a hidden Settings toggle if one exists) that creates `DebugScanRecorder`, calls `start()` when recording begins and `stop()` when it ends, then presents a `UIActivityViewController`/ShareLink for the folder so the user can AirDrop it. Follow the file's existing coordinator pattern exactly — do not restructure it.

- [ ] **Step 3: Build** — `xcodebuild -scheme decroom_no_lidar -project decroom_no_lidar/decroom_no_lidar.xcodeproj -destination 'generic/platform=iOS' build`. Expected: BUILD SUCCEEDED.

- [ ] **Step 4: Device smoke test (user)** — 10-second recording; confirm the share sheet delivers a folder with all 4 files, `video.mov` plays, `frames.jsonl` line count ≈ appended count, `frames_dropped_by_encoder` ≈ 0. Record drop count + thermal states in `phase0-results.md` (this is the S6-lite datum).

- [ ] **Step 5: Commit** — `git commit -am "spike(phase0): ARKit debug recorder (video + pose JSONL + S6-lite stats)"`

---

### Task 5: Real scan capture + ground truth (user action)

**Files:**
- Modify: `decroom_no_lidar/backend/spikes/phase0/ground_truth.json` (add dimensions + facing)

- [ ] **Step 1: User records the pose-annotated scan** with the Task-4 recorder — same choreography as Task 2 (slow 360°, every object leaves + returns). 60–90 s. Transfer folder to pod at `~/scans/scan01/`.

- [ ] **Step 2: User tape-measures 5–8 objects** — width × depth × height (cm) and facing direction (which compass-ish way the front faces, relative to a named wall), plus room width × length. Extend `ground_truth.json`:

```json
{"room": {"width_m": 3.4, "length_m": 4.1},
 "objects": [{"category": "bed", "count": 1, "dims_cm": [153, 203, 55], "facing": "toward window wall"}]}
```

- [ ] **Step 3: Note the capture device model** in `phase0-results.md`. If it's a Pro (LiDAR), add the standing caveat: plane-anchor quality overstates non-Pro devices (spec §6 QA note).

- [ ] **Step 4: Commit** the updated ground truth.

---

### Task 6: S2 — geometry spike (MoGe-2 + scale anchor + TSDF + rectification check + QP ladder)

**Files:**
- Create: `decroom_no_lidar/backend/spikes/phase0/s2_geometry.py`

**Interfaces:**
- Consumes: `~/scans/scan01/` (Task 4 format), Task 1 env.
- Produces: `scan01_tsdf.ply` (fused cloud) + `s2_report.json`; reusable functions `load_scan(dir) -> Scan` (frames iterator + per-frame `pose_cv` 4×4 world-from-camera OpenCV convention + `K` + sparse points) and `depth_for_frame(scan, idx, moge) -> (depth_m, normals, scale_used)` imported by Task 7.

- [ ] **Step 1: Write `s2_geometry.py`**

```python
"""S2: MoGe-2 depth + ARKit scale anchor + Open3D TSDF fusion, plus
(a) rawFeaturePoints reprojection check (rectification assumption) and
(b) HEVC QP ladder effect on depth."""
import numpy as np, json, cv2, torch, open3d as o3d, subprocess, os, argparse
from moge.model.v2 import MoGeModel

FLIP_X = np.diag([1.0, -1.0, -1.0, 1.0])   # ARKit(-Z fwd) -> OpenCV(+Z fwd)

class Scan:
    def __init__(self, d):
        self.dir = d
        self.meta = [json.loads(l) for l in open(f"{d}/frames.jsonl")]
        self.pts = [json.loads(l) for l in open(f"{d}/points.jsonl")]
        os.makedirs(f"{d}/frames", exist_ok=True)
        if not os.listdir(f"{d}/frames"):
            subprocess.run(["ffmpeg","-y","-i",f"{d}/video.mov", f"{d}/frames/%05d.jpg"], check=True)
        self.n = min(len(self.meta), len(os.listdir(f"{d}/frames")))
    def frame(self, i):        # landscape-right BGR as recorded
        return cv2.imread(f"{self.dir}/frames/{i+1:05d}.jpg")
    def pose_cv(self, i):      # world-from-camera, OpenCV convention
        m = np.array(self.meta[i]["pose"]).reshape(4,4, order="F")
        return m @ FLIP_X
    def K(self, i):
        fx, fy, cx, cy = self.meta[i]["K"]; return fx, fy, cx, cy
    def sparse_world(self, i): # nearest points.jsonl entry by timestamp
        t = self.meta[i]["t"]
        e = min(self.pts, key=lambda p: abs(p["t"]-t))
        return np.array(e["pts"]) if abs(e["t"]-t) < 0.5 else np.zeros((0,3))

def project(K, pose_cv, pts_w):
    fx, fy, cx, cy = K
    pc = (np.linalg.inv(pose_cv) @ np.c_[pts_w, np.ones(len(pts_w))].T).T[:, :3]
    z = pc[:, 2]; ok = z > 0.1
    u = fx*pc[:,0]/z + cx; v = fy*pc[:,1]/z + cy
    return u[ok], v[ok], z[ok]

def depth_for_frame(scan, i, moge, running_scale=[1.0]):
    img = cv2.cvtColor(scan.frame(i), cv2.COLOR_BGR2RGB)
    fx, _, _, _ = scan.K(i)
    fov_x = np.degrees(2*np.arctan(img.shape[1]/(2*fx)))
    with torch.no_grad():
        out = moge.infer(torch.from_numpy(img).cuda().permute(2,0,1)/255.0, fov_x=fov_x)
    depth = out["depth"].cpu().numpy()
    u, v, z = project(scan.K(i), scan.pose_cv(i), scan.sparse_world(i))
    inb = (u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
    if inb.sum() >= 10:
        ratios = z[inb] / depth[v[inb].astype(int), u[inb].astype(int)]
        s = float(np.median(ratios)); running_scale[0] = 0.8*running_scale[0] + 0.2*s
    else:
        s = running_scale[0]                    # low-texture fallback (spec §3.1)
    return depth*s, out.get("normal"), s, int(inb.sum())

def fuse(scan, moge, stride, qp_note=""):
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.03, sdf_trunc=0.12,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    scales, pt_counts = [], []
    for i in range(0, scan.n, stride):
        if scan.meta[i]["tracking"] != "normal": continue   # pose gating (spec §2)
        d, _, s, npts = depth_for_frame(scan, i, moge); scales.append(s); pt_counts.append(npts)
        img = scan.frame(i)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
            o3d.geometry.Image((d*1000).astype(np.uint16)),
            depth_scale=1000.0, depth_trunc=6.0, convert_rgb_to_intensity=False)
        fx, fy, cx, cy = scan.K(i)
        intr = o3d.camera.PinholeCameraIntrinsic(img.shape[1], img.shape[0], fx, fy, cx, cy)
        vol.integrate(rgbd, intr, np.linalg.inv(scan.pose_cv(i)))
    return vol.extract_point_cloud(), scales, pt_counts

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("scan_dir"); a = ap.parse_args()
    moge = MoGeModel.from_pretrained("~/ckpts/moge2").cuda().eval()
    scan = Scan(a.scan_dir)
    # (a) rectification check: reprojection residual of sparse points vs pinhole
    res = []
    for i in range(0, scan.n, 30):
        pts = scan.sparse_world(i)
        if len(pts) < 10: continue
        u, v, z = project(scan.K(i), scan.pose_cv(i), pts)
        res.append(len(u)/max(len(pts),1))      # in-frustum fraction; low => convention bug
    # (b) main fusion at 2 fps (stride 15 of 30fps-effective recording)
    pcd, scales, pt_counts = fuse(scan, moge, stride=15)
    o3d.io.write_point_cloud(f"{a.scan_dir}/scan01_tsdf.ply", pcd)
    # (c) QP ladder: re-encode at QP 22/32/37, re-run depth on 5 frames, compare
    ladder = {}
    for qp in (22, 32, 37):
        subprocess.run(["ffmpeg","-y","-i",f"{a.scan_dir}/video.mov","-c:v","libx265",
                        "-x265-params",f"qp={qp}", f"/tmp/qp{qp}.mov"], check=True)
        # extract same 5 frames, run MoGe, record mean abs rel diff vs original frames
        # (loop analogous to depth_for_frame on both sources; ~15 lines, same helpers)
    report = {"points": len(pcd.points), "scale_median": float(np.median(scales)),
              "scale_iqr": float(np.subtract(*np.percentile(scales,[75,25]))),
              "frames_below_10pts": int(sum(1 for c in pt_counts if c < 10)),
              "reproj_infrustum_frac": float(np.mean(res))}
    json.dump(report, open(f"{a.scan_dir}/s2_report.json","w"), indent=2); print(report)
```

(The QP-ladder inner loop reuses `depth_for_frame` on frames extracted from each re-encode; implement it inline — comparison metric: mean absolute relative depth difference vs the original frames. Anchors from research: expect ≈0% @QP22, ~+4.5% @QP32, ~+13% @QP37.)

- [ ] **Step 2: Sanity-run on the first 10 s of the scan** (`scan.n` capped) — expected: PLY written, `reproj_infrustum_frac > 0.7` (if ~0, the ARKit→OpenCV flip or intrinsics swap is wrong — fix before anything else; this is the rectification/convention check doing its job).

- [ ] **Step 3: Full run + eyeball** — open the PLY (scp to Mac, view in MeshLab/Open3D viewer) next to a current-pipeline PLY from `Documents/last_payload/` or the backend run archive. Record in `phase0-results.md`: point count, scale IQR (scale stability), % frames on fallback scale, wall-clock, and a subjective side-by-side verdict vs the current on-device cloud (S2's original question).

- [ ] **Step 4: Commit** harness + results.

---

### Task 7: S1 full-chain — the gate

**Files:**
- Create: `decroom_no_lidar/backend/spikes/phase0/s1_fullchain.py`

**Interfaces:**
- Consumes: `run_chain`/`detect` (Task 3), `Scan`/`depth_for_frame` (Task 6), `~/scans/scan01/`, `ground_truth.json`.
- Produces: `s1_report.json` + `s1_tracks.png` (per-track crop grid for manual labeling) + `track_labels.json` (human-labeled track→real-object map).

- [ ] **Step 1: Write `s1_fullchain.py`** — composes the tested pieces; the new logic is lifting + merging + metrics:

```python
"""S1 gate: full chain on the pose-annotated scan.
Chain: run_chain (LLMDet->SAM2, Task 3) -> lift each track to 3D via MoGe depth
(Task 6) -> residual merge (footprint-overlap primary, >=0.8m centroid gate
for large categories) -> metrics."""
import numpy as np, json, cv2, torch, argparse
from s1a_killscreen import run_chain
from s2_geometry import Scan, depth_for_frame, FLIP_X
from moge.model.v2 import MoGeModel

LARGE = {"bed","sofa","desk","table","wardrobe","bookshelf","dresser"}
CENTROID_GATE = {True: 0.8, False: 0.5}       # large : small (never < recorded 0.8 for large)
FOOTPRINT_IOU = 0.25

def lift_track(scan, track, masks_by_frame, moge):
    """Median 3D centroid + XZ footprint polygon from masked depth, per track."""
    pts_all = []
    for fidx in track["frames"][::3]:          # every 3rd mask frame is plenty
        if scan.meta[fidx]["tracking"] != "normal": continue
        d, _, _, _ = depth_for_frame(scan, fidx, moge)
        m = masks_by_frame.get(fidx)
        if m is None: continue
        ys, xs = np.where(m)
        if len(xs) < 50: continue
        sel = np.random.choice(len(xs), min(500, len(xs)), replace=False)
        fx, fy, cx, cy = scan.K(fidx)
        z = d[ys[sel], xs[sel]]
        cam = np.c_[(xs[sel]-cx)*z/fx, (ys[sel]-cy)*z/fy, z, np.ones(len(z))]
        pts_all.append((scan.pose_cv(fidx) @ cam.T).T[:, :3])
    if not pts_all: return None
    pts = np.vstack(pts_all)
    lo, hi = np.percentile(pts, [5, 95], axis=0)           # robust extent
    return {"centroid": np.median(pts, axis=0), "extent": hi-lo,
            "footprint": pts[:, [0, 2]], "n_frames_lifted": len(pts_all)}

def footprint_iou(a, b, cell=0.05):
    ga = {tuple((p//cell).astype(int)) for p in a}
    gb = {tuple((p//cell).astype(int)) for p in b}
    return len(ga & gb) / max(len(ga | gb), 1)

def residual_merge(lifted):
    merged, used = [], set()
    ids = sorted(lifted, key=lambda k: -lifted[k]["lift"]["n_frames_lifted"])
    for i in ids:
        if i in used: continue
        group = [i]; used.add(i)
        for j in ids:
            if j in used or lifted[j]["label"] != lifted[i]["label"]: continue
            gate = CENTROID_GATE[lifted[i]["label"] in LARGE]
            dc = np.linalg.norm(lifted[i]["lift"]["centroid"] - lifted[j]["lift"]["centroid"])
            fiou = footprint_iou(lifted[i]["lift"]["footprint"], lifted[j]["lift"]["footprint"])
            if fiou >= FOOTPRINT_IOU or dc <= gate:        # footprint primary, centroid gate
                group.append(j); used.add(j)
        merged.append(group)
    return merged

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("scan_dir")
    ap.add_argument("--fps_track", type=int, default=5); ap.add_argument("--fps_det", type=int, default=1)
    a = ap.parse_args()
    scan = Scan(a.scan_dir)
    tracks, timings = run_chain(f"{a.scan_dir}/frames", fps_det=a.fps_det, fps_track=a.fps_track)
    moge = MoGeModel.from_pretrained("~/ckpts/moge2").cuda().eval()
    lifted = {}
    for tid, t in tracks.items():
        if len(t["frames"]) < 3: continue
        lift = lift_track(scan, t, t.get("masks", {}), moge)
        if lift: lifted[tid] = {"label": t["label"], "lift": lift}
    groups = residual_merge(lifted)
    per_cat = {}
    for g in groups:
        lab = lifted[g[0]]["label"]; per_cat[lab] = per_cat.get(lab, 0) + 1
    gt = {o["category"]: o["count"] for o in json.load(open("ground_truth.json"))["objects"]}
    report = {
      "fps": {"track": a.fps_track, "det": a.fps_det},
      "raw_tracks": len(lifted),
      "merged_instances_per_category": per_cat,
      "ground_truth": gt,
      "spurious_per_object": sum(max(0, per_cat.get(c,0)-n) for c,n in gt.items()) / max(sum(gt.values()),1),
      "fragmentation_rate": len(lifted) / max(sum(gt.values()), 1),
      "peak_vram_gb": torch.cuda.max_memory_allocated()/1e9,
      "extents_cm": {str(k): (v["lift"]["extent"]*100).round(0).tolist() for k, v in lifted.items()},
    }
    json.dump(report, open(f"{a.scan_dir}/s1_report_{a.fps_track}fps.json","w"), indent=2)
    print(json.dumps(report, indent=2))
```

Note: `run_chain` must be extended (small edit in `s1a_killscreen.py`) to also store per-track `masks: {frame_idx: bool_mask}` for the frames it propagates — add a `keep_masks=True` flag storing every 3rd mask downsampled to half resolution to bound memory.

- [ ] **Step 2: Run the main configuration** — `python s1_fullchain.py ~/scans/scan01 --fps_track 5 --fps_det 1`. Record ALL report fields in `phase0-results.md`.

- [ ] **Step 3: fps sweep (spec S1d)** — rerun with `--fps_track 1`, `3`, `10` (detection stride adjusts automatically). Record `spurious_per_object` and `fragmentation_rate` per fps.

- [ ] **Step 4: Opposite-view footprint measurement (spec S1c) + stale masks (S1e) — manual labeling.** Generate `s1_tracks.png` (grid: 3 crops per track, labeled with track id — ~15 lines with matplotlib, add to the script). The user (or executor viewing the grid) writes `track_labels.json`: `{"<track_id>": "<real object name>"}`. Then compute and record: (a) footprint IoU + centroid distance for every same-object track pair, split by whether their view directions differ by >90° (dot product of camera-to-centroid bearings) — this is the named weak case; (b) stale-mask rate: fraction of mask-frames per track where the camera-to-centroid angle says the object was behind the camera.

- [ ] **Step 5: Extent accuracy (feeds S2 criterion)** — for the 5–8 tape-measured objects, compare `extents_cm` of their labeled tracks vs `ground_truth.json` dims. Record per-category error %.

- [ ] **Step 6: THE GATE.** Pass = `spurious_per_object ≤ 1` at any tested fps AND runtime within budget shape. Record verdict. If FAIL: do NOT redesign here — record which ladder rung applies (spec §6: SAMURAI/DAM4SAM fork → alternative merge criteria incl. view-consensus **with runtime** → association redesign) and stop for user decision.

- [ ] **Step 7: Commit** harness + all reports + labels.

---

### Task 8: S3 — detection coverage + runtime

**Files:**
- Create: `decroom_no_lidar/backend/spikes/phase0/s3_detection.py`

**Interfaces:**
- Consumes: `detect` (Task 3), scan frames, `ground_truth.json`; optionally old Plan 3c/3d keyframes (`/tmp/decroom_keyframes/latest/` on the Mac backend, or the backend run archive) for the Grounding DINO comparison.

- [ ] **Step 1: Write `s3_detection.py`** — run `detect()` on every 1-fps keyframe of scan01; per category record: detected-in-N-frames, best score, and whether every ground-truth category is found at least 3× (coverage). Time each call; report median/p95 ms at 1280px and at 720px input. ~40 lines reusing Task 3's `detect`.

- [ ] **Step 2: Run; record** coverage table (which GT categories were missed — wardrobe/desk/bookshelf/nightstand were the historical misses), median ms/frame at both resolutions, and (if old keyframes available) side-by-side with Plan 3c/3d Grounding DINO output on the same images.

- [ ] **Step 3: S3 verdict** — pass = all GT categories covered + runtime consistent with the 30–60 s budget (≤ ~400 ms/keyframe at ≤ ~60 keyframes). If vocabulary gaps: try prompt synonyms before swapping detectors (MM-Grounding-DINO is the zero-code-swap fallback, spec §7). Commit.

---

### Task 9: S5 — orientation quality (Orient Anything V1)

**Files:**
- Create: `decroom_no_lidar/backend/spikes/phase0/s5_orientation.py`

**Interfaces:**
- Consumes: per-track best crops (from Task 7's `s1_tracks.png` generation — reuse the crop-selection code: largest mask area), `~/orient-anything` repo + `~/ckpts/orient` ckpt, `ground_truth.json` facing annotations, track world-space data for camera-to-world azimuth conversion.

- [ ] **Step 1: Write `s5_orientation.py`** — for each tape-measured object: take its 3 best crops, run Orient Anything V1 inference (adapt the repo's demo/inference script — model is a DINOv2 classifier over azimuth/polar bins; load per repo README), convert predicted azimuth from camera frame to world yaw using the crop frame's `pose_cv`, take circular median. Compare against the ground-truth facing direction (user's annotation mapped to a world yaw using the room's wall directions from the S2 cloud, ±45° tolerance since annotations are coarse).

- [ ] **Step 2: Run; record** per-object yaw error, and separately note symmetric objects (round table etc. — excluded per spec §3.3). S5 verdict: pass = ≥70% of asymmetric objects within ±45°. If fail: footprint-axis-only + no-yaw fallback per spec §7, and the hybrid 12MP crop question (Task 4's recorder could add stills later) becomes live. Commit.

---

### Task 10: S4 — generative mesh spike (OPTIONAL — gates Phase 2 only)

**Files:**
- Create: `decroom_no_lidar/backend/spikes/phase0/s4_genmesh.md` (notes; MIDI/PartCrafter run in a SEPARATE container per spec — do not install into the phase0 env)

- [ ] **Step 1:** In a fresh container: install PartCrafter (`wgsxm/PartCrafter-Scene`, torch 2.5.1+cu124 per its README) or MIDI (`VAST-AI/MIDI-3D`). Run on 5–10 real furniture crops from scan01 (note PartCrafter's real-photo path uses a Gemini-API style-transfer step — record whether skipping it hurts).
- [ ] **Step 2:** Qualitative grid: does the generated mesh resemble the actual object, is the back plausible, runtime per object. Record verdict in results (pass/fail gates Phase 2 only — no effect on Phase 1 go/no-go). Commit notes.

---

### Task 11: Results synthesis + go/no-go

**Files:**
- Create: `docs/superpowers/plans/phase0-results.md` (has been accumulating; finalize)

- [ ] **Step 1: Compile the results table** — one row per spike: measurement, gate, value, verdict (S1a kill-screen; S1 end-to-end count + fragmentation + opposite-view footprint + fps sweep + stale-mask + VRAM/time; S2 cloud quality + scale stability + rectification + QP ladder + extents; S3 coverage + latency; S5 yaw; S6-lite drops/thermal; S4 optional).

- [ ] **Step 2: Sum the measured stage times** into a projected end-to-end pipeline time; compare against the 30–60 s target and 2 min hard cap (spec §7) — state which sampling rates the numbers support.

- [ ] **Step 3: Write the go/no-go recommendation** — GO (S1 passed → proceed to Phase 1 plan), LADDER (S1 failed → name the rung and what to test next), or MIXED (e.g. S1 pass + S5 fail → Phase 1 with no-yaw fallback). Per the standing rule in memory, dispatch a devil's advocate agent on this synthesis before presenting it to the user.

- [ ] **Step 4: Update memory** (`clean_slate_pipeline_2026_07.md`) with the measured numbers, and commit everything.

---

## Self-Review (done at write time)

- **Spec coverage:** S1 (a–f) → Task 7 (+3 kill-screen); S1 ladder → gated stops in Tasks 3/7; S2 → Task 6 + extent step in Task 7; S3 → Task 8; S4 → Task 10; S5 → Task 9; S6-lite → Task 4 (full streaming explicitly deferred to Phase 1 per plan text). Gaps: none against §6.
- **Placeholder scan:** the QP-ladder inner loop in Task 6 is described with its exact metric and reuses a defined helper — acceptable spike-level instruction; no TBDs remain.
- **Type consistency:** `run_chain`/`detect` (Task 3) consumed by Task 7 with matching signatures; `Scan`/`depth_for_frame`/`FLIP_X` (Task 6) consumed by Task 7; `keep_masks` extension is called out where it's needed.
- **Known-drift warning for executors:** SAM 2 and transformers APIs drift; Task 3 Step 2's sanity run exists precisely to absorb signature fixes before real data.
