# Plan 3c — Backend Open-Vocab RGB Detection on Keyframes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failing SpatialLM-furniture detection path with a backend open-vocab RGB detector (Grounding DINO) that runs on the 10 keyframe JPEGs the phone already uploads, lifts 2D detections to 3D via PLY raycast, clusters across keyframes, and feeds into Plan 3b's existing consensus pipeline unchanged.

**Architecture:** A new `/detect_furniture` endpoint on the existing RunPod server.py loads Grounding DINO alongside SpatialLM and returns per-frame 2D detections for a configurable text prompt list. A new `backend/pipeline/rgb_detection.py` calls that endpoint over the SSH tunnel. `backend/pipeline/raycast_lift.py` turns 2D bbox centers into 3D world positions by raycasting through the merged PLY. `backend/pipeline/keyframe_clustering.py` collapses per-frame detections of the same physical object into one tracked entry (same algorithm as on-device ObjectTracker but with materially less-noisy input positions). `phone_first_pipeline.run_phone_first_pipeline` gains a step between SpatialLM inference and `_build_final_objects`: backend RGB detections are merged into the existing `yolo_objects_3d` list before consensus, so Plan 3b's polish pipeline (consensus_merge → collapse_duplicates → snap_to_walls → compute_confidence) consumes both sources without modification.

**Tech Stack:** Python (FastAPI + NumPy + scipy.spatial + transformers + torch), pytest. iOS code is **not** modified in Plan 3c.

**Spec:** `docs/superpowers/specs/2026-05-03-plan3c-backend-rgb-detection-design.md`.

**Prerequisites (MUST be true before starting):**
- Plan 3b merged to `main` (consensus.py, _build_final_objects, _attach_openings_from_slm_walls all live).
- SpatialLM tunnel on RunPod reachable at `localhost:8100` (see `memory/runpod_setup.md`).
- Branch off `main`: `git switch -c feature/plan3c-backend-rgb`.
- Pod has ≥6 GB free GPU memory after SpatialLM is loaded (Grounding DINO ≈2 GB).

**Critical**: **Task 1 (Phase A0 model-quality gate) is BLOCKING.** No further task may start until the gate passes. If Grounding DINO fails the recall criterion on photos of the actual QA bedroom, stop and revise the spec — either swap to OWL-ViT (re-run Task 1 with that model) or accept that some categories are user-placement-only in Plan 4b.

---

## File structure

### New files

| Path | Responsibility |
|---|---|
| `decroom_no_lidar/backend/pipeline/rgb_detection.py` | HTTP client for `/detect_furniture`. Pure function: `(jpegs, prompts) → list[KeyframeDetection]` per frame. No GPU code lives here. |
| `decroom_no_lidar/backend/pipeline/raycast_lift.py` | 2D bbox center + camera pose + intrinsics + PLY → 3D world position (or None if no PLY support within shell tolerance). |
| `decroom_no_lidar/backend/pipeline/keyframe_clustering.py` | Cross-keyframe NN-merge of lifted 3D detections by category + per-category distance threshold. |
| `decroom_no_lidar/backend/tests/test_rgb_detection.py` | HTTP client mocked; verifies request shape, response parsing, error handling. |
| `decroom_no_lidar/backend/tests/test_raycast_lift.py` | Synthetic PLY + known pose → hit; bbox aimed at empty space → no hit. |
| `decroom_no_lidar/backend/tests/test_keyframe_clustering.py` | Same-category near-positions → 1 cluster; far-positions → 2 clusters; different-categories at same position → 2 clusters. |
| `decroom_no_lidar/backend/tests/fixtures/grounding_dino_response.json` | Captured sample `/detect_furniture` response for offline tests. |
| `docs/superpowers/qa/2026-XX-XX-plan3c-detector-gate.md` | Phase A0 gate findings (per-category recall on the QA bedroom photos). |
| `docs/superpowers/qa/2026-XX-XX-plan3c-manual-qa.md` | Phase E end-to-end scan QA results. |

### Modified files

| Path | Change |
|---|---|
| `decroom_no_lidar/backend/spatiallm_server/server.py` | Lazy-load Grounding DINO at startup alongside SpatialLM. New `POST /detect_furniture` endpoint (multipart: list of JPEGs + form field `prompts`). |
| `decroom_no_lidar/backend/spatiallm_server/setup.sh` | Pre-cache Grounding DINO weights (`IDEA-Research/grounding-dino-tiny`) into `$HF_HOME` during pod setup. Install `transformers` already covers it. |
| `decroom_no_lidar/backend/pipeline/phone_first_pipeline.py` | After `run_spatiallm_pipeline`, run `detect_furniture_in_keyframes → lift_detections_to_3d → cluster_keyframe_detections`, merge result with `payload.yolo_objects` into a single combined list, then call `_build_final_objects` with that combined list. |
| `decroom_no_lidar/backend/pipeline/consensus.py` | Audit `CATEGORY_TO_ASSET` for any RGB-prompt category not yet mapped (target list: bed, wardrobe, bookshelf, desk, monitor, chair, nightstand, dresser, lamp, potted plant, sofa). Add missing entries. |
| `decroom_no_lidar/backend/tests/test_phone_first_pipeline.py` | Mock the RGB detection step; assert combined objects flow through consensus. |
| `decroom_no_lidar/backend/tests/test_consensus.py` (no source change, just verify) | Confirm whitelist audit covers every prompt category. |

### Untouched

- All iOS code.
- `consensus.py` core functions (`consensus_merge`, `collapse_duplicates`, `snap_to_walls`, `compute_confidence`) — no signature changes; only `CATEGORY_TO_ASSET` may grow.
- `_build_final_objects` — signature unchanged. The change is upstream: a richer `yolo_objects_3d` argument.

---

## Task list

Tasks run in order. Phase A0 (Task 1) is a manual blocking gate — no plumbing tasks may start until it passes. Phase A (Tasks 2–5) puts Grounding DINO behind a `/detect_furniture` endpoint and writes its HTTP client. Phase B (Tasks 6–8) implements PLY raycast lifting. Phase C (Tasks 9–10) cross-keyframe clusters. Phase D (Tasks 11–13) wires everything into `phone_first_pipeline`. Phase E (Task 14) is end-to-end QA + merge.

---

### Task 1: Phase A0 — model-quality go/no-go gate (BLOCKING)

**Files:**
- Create: `docs/superpowers/qa/<today>-plan3c-detector-gate.md`
- Create (throwaway): `/tmp/plan3c_gate.py` (standalone notebook-style script — not committed to repo)

**Goal:** Verify Grounding DINO can recall the target indoor-furniture categories on the **actual keyframe JPEGs** the phone uploaded during a Plan 3b QA scan, *before* writing any backend plumbing. Using real keyframes (not staged photos taken with a separate camera) tests the model on the exact image quality, framing, and ARKit `capturedImage` orientation it'll see in production. If recall is insufficient on these images, revise the spec.

- [ ] **Step 1: Source the keyframe JPEGs from a real Plan 3b QA scan.**

Use whichever of these is available — in priority order:

**Option A — backend upload archive (preferred).** If the local backend persists uploaded payloads (check `decroom_no_lidar/backend/` for any `uploads/` or `received_scans/` directory, or `/tmp/decroom_uploads/`), grab the 10 keyframes from the most recent QA scan:
```bash
ls -lt /tmp/decroom_uploads 2>/dev/null || \
  find /Users/jaejeon/Desktop/Decroom/decroom_no_lidar -type d -name "uploads" -o -name "received_scans" 2>/dev/null
# Identify the most-recent scan dir, then:
mkdir -p /tmp/plan3c_gate
cp <scan_dir>/keyframes/*.jpg /tmp/plan3c_gate/
```

**Option B — pull from the iOS device's `last_payload`.** ScanUploadPayload writes to `Documents/last_payload/keyframes/`. Pull via Xcode (Devices and Simulators → select device → Decroom container → Documents → last_payload → keyframes), or via Finder if iCloud sync is on. Drop into `/tmp/plan3c_gate/`.

**Option C — re-run one fresh QA scan to regenerate.** Drive the iOS app through one full capture of the QA bedroom (no special handling needed; same flow that produced the original 0-furniture Plan 3b scan). The keyframes land on the device per Option B and the backend per Option A.

Verify before continuing:
```bash
ls /tmp/plan3c_gate/*.jpg | wc -l   # expect ~10
```
If <5 files, source from a different option.

- [ ] **Step 2: Write the standalone gate script.**

Save as `/tmp/plan3c_gate.py`:

```python
"""Phase A0 gate — Grounding DINO recall on real bedroom photos.

NOT committed to repo. Throwaway. Output is the QA markdown report.
"""
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "bed. wardrobe. bookshelf. desk. monitor. chair. nightstand. dresser. lamp. potted plant. sofa."
TARGET_CATS = ["bed", "wardrobe", "bookshelf", "desk", "dresser", "nightstand"]
CONFIDENCE_THRESHOLD = 0.4
PHOTOS_DIR = Path("/tmp/plan3c_gate")

def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    photos = sorted(PHOTOS_DIR.glob("*.jpg"))
    print(f"Testing {len(photos)} keyframes with prompt: {PROMPT}")
    per_photo_hits: dict[str, set[str]] = {}
    for photo_path in photos:
        image = Image.open(photo_path).convert("RGB")
        inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=CONFIDENCE_THRESHOLD,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )[0]
        hits = {label.lower().strip() for label, score in zip(results["labels"], results["scores"]) if score >= CONFIDENCE_THRESHOLD}
        per_photo_hits[photo_path.name] = hits
        print(f"  {photo_path.name}: {sorted(hits)}")

    # Per-target recall: fraction of photos that detected each target category.
    print("\nPer-target recall:")
    summary = {}
    for cat in TARGET_CATS:
        n_hits = sum(1 for hits in per_photo_hits.values() if any(cat in h for h in hits))
        recall = n_hits / max(1, len(photos))
        summary[cat] = recall
        print(f"  {cat:12s}  {recall:.0%}  ({n_hits}/{len(photos)})")

    avg_recall = sum(summary.values()) / len(summary)
    print(f"\nAverage recall across target categories: {avg_recall:.0%}")
    print(f"GO criterion: ≥70% average. Result: {'PASS' if avg_recall >= 0.70 else 'FAIL'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run on the RunPod GPU pod (faster) or locally on CPU (slower but works).**

If running on the pod via SSH:
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 /tmp/plan3c_gate.py root@<IP>:/tmp/plan3c_gate.py
scp -P <PORT> -i ~/.ssh/id_ed25519 -r /tmp/plan3c_gate root@<IP>:/tmp/plan3c_gate
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<IP> 'source /workspace/spatiallm-env/bin/activate && python /tmp/plan3c_gate.py'
```

Locally on CPU (slower, but no SSH needed):
```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
python3 -m venv /tmp/plan3c_venv && source /tmp/plan3c_venv/bin/activate
pip install --quiet transformers torch pillow
python3 /tmp/plan3c_gate.py
```

- [ ] **Step 4: Document findings.**

Create `docs/superpowers/qa/<today>-plan3c-detector-gate.md`:

```markdown
# Plan 3c Phase A0 — Detector Quality Gate

**Date**: <YYYY-MM-DD>
**Model**: IDEA-Research/grounding-dino-tiny
**Prompt**: "bed. wardrobe. bookshelf. desk. monitor. chair. nightstand. dresser. lamp. potted plant. sofa."
**Confidence threshold**: 0.4
**Source images**: N keyframe JPEGs from Plan 3b QA scan of bedroom (3.04 × 3.25 m, bed/wardrobe/bookshelf/desk/2 monitors/chair). Source: <Option A backend archive | Option B device last_payload | Option C fresh re-scan>

## Per-target recall

| Category   | Recall | Notes |
|------------|--------|-------|
| bed        | XX%    |       |
| wardrobe   | XX%    |       |
| bookshelf  | XX%    |       |
| desk       | XX%    |       |
| dresser    | XX%    |       |
| nightstand | XX%    |       |
| **Avg**    | XX%    |       |

## Decision

- [ ] **PASS** (≥70% average): proceed to Task 2.
- [ ] **FAIL**: do not start Task 2. Re-run gate with `google/owlv2-base-patch16-ensemble`. If both fail, stop and revise the spec — categories below 70% become Plan 4b user-placement-only.

## Sample detections per photo

(paste the script's `photo_NN.jpg: [...]` lines here)
```

- [ ] **Step 5: Commit gate findings (only — no implementation yet).**

```bash
git add docs/superpowers/qa/<today>-plan3c-detector-gate.md
git commit -m "qa(plan3c): A0 detector quality gate — <PASS|FAIL>"
```

**STOP if FAIL.** Do not proceed to Task 2.

---

### Task 2: Add Grounding DINO load + `/detect_furniture` to `server.py`

**Files:**
- Modify: `decroom_no_lidar/backend/spatiallm_server/server.py`

- [ ] **Step 1: Add Grounding DINO globals + loader.**

Insert after the existing SpatialLM globals (around line 36):

```python
# ── Grounding DINO globals (Plan 3c) ──
_gdino_model = None
_gdino_processor = None
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_GDINO_PROMPT = (
    "bed. wardrobe. bookshelf. desk. monitor. chair. "
    "nightstand. dresser. lamp. potted plant. sofa."
)


def load_gdino():
    """Load Grounding DINO model + processor (called once at startup)."""
    global _gdino_model, _gdino_processor
    if _gdino_model is not None:
        return
    logger.info("Loading Grounding DINO: %s", GDINO_MODEL_ID)
    t0 = time.time()
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    _gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_MODEL_ID
    ).to(_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _gdino_model.eval()
    logger.info("Grounding DINO loaded in %.1fs", time.time() - t0)
```

- [ ] **Step 2: Wire into startup so it loads alongside SpatialLM.**

Replace the existing `startup` handler (around line 191):

```python
@app.on_event("startup")
async def startup():
    load_model()    # SpatialLM (existing)
    load_gdino()    # Grounding DINO (Plan 3c)
```

- [ ] **Step 3: Add the `/detect_furniture` route.**

Append after the existing `/infer` route (around line 218):

```python
from typing import List
from fastapi import Form

@app.post("/detect_furniture")
async def detect_furniture(
    images: List[UploadFile] = File(...),
    prompt: str = Form(DEFAULT_GDINO_PROMPT),
    box_threshold: float = Form(0.4),
    text_threshold: float = Form(0.3),
):
    """Run open-vocab detection on a list of JPEGs.

    Returns: {"frames": [{"frame_index": int,
                            "detections": [{"category": str,
                                            "bbox_normalized": [x, y, w, h],
                                            "confidence": float}, ...]}, ...]}
    bbox is (x, y, w, h) in normalized [0,1] image coords.
    """
    from PIL import Image
    import io

    if _gdino_model is None or _gdino_processor is None:
        return JSONResponse(status_code=503, content={"error": "gdino not loaded"})

    t0 = time.time()
    frames_out = []
    for idx, upload in enumerate(images):
        raw = await upload.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        W, H = image.size
        inputs = _gdino_processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(_gdino_model.device)
        with torch.no_grad():
            outputs = _gdino_model(**inputs)
        results = _gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(H, W)],
        )[0]
        dets = []
        for label, score, box in zip(
            results["labels"], results["scores"], results["boxes"]
        ):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            dets.append({
                "category": str(label).lower().strip(),
                "bbox_normalized": [x0 / W, y0 / H,
                                     (x1 - x0) / W, (y1 - y0) / H],
                "confidence": float(score),
            })
        frames_out.append({"frame_index": idx, "detections": dets})

    elapsed = time.time() - t0
    logger.info("Detected on %d frames in %.1fs", len(images), elapsed)
    return JSONResponse(content={"frames": frames_out,
                                  "elapsed_s": round(elapsed, 2)})
```

- [ ] **Step 4: Local sanity-check on the pod.**

```bash
# On the pod (after restarting server.py):
curl -X POST -F "images=@/tmp/plan3c_gate/$(ls /tmp/plan3c_gate | head -1)" \
              -F "images=@/tmp/plan3c_gate/$(ls /tmp/plan3c_gate | sed -n 2p)" \
              -F "prompt=bed. wardrobe. desk." \
              http://localhost:8100/detect_furniture | python3 -m json.tool
```

Expected: JSON with 2 frame entries, each with non-empty `detections`. If empty, the gate (Task 1) should have caught this — re-run gate.

- [ ] **Step 5: Commit.**

```bash
git add decroom_no_lidar/backend/spatiallm_server/server.py
git commit -m "feat(server): add Grounding DINO + /detect_furniture endpoint"
```

---

### Task 3: Pre-cache Grounding DINO weights in `setup.sh`

**Files:**
- Modify: `decroom_no_lidar/backend/spatiallm_server/setup.sh`

- [ ] **Step 1: Add weight pre-cache after the existing SpatialLM weight pull.**

Insert after line 78 (after the `snapshot_download('manycore-research/...')` line):

```bash
# Plan 3c: pre-cache Grounding DINO so server.py startup doesn't block on download.
echo ">>> Downloading Grounding DINO weights..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('IDEA-Research/grounding-dino-tiny'); print('Grounding DINO cached.')"
```

- [ ] **Step 2: Verify the existing `transformers` install covers Grounding DINO.**

The existing line:
```bash
pip install --quiet transformers==4.46.1 tokenizers==0.20.3 ...
```
Grounding DINO requires `transformers>=4.36`. 4.46.1 satisfies this. No change needed.

- [ ] **Step 3: Smoke-test setup.sh from a fresh pod (skip if existing pod).**

If iterating on an existing pod, just run the new lines manually:
```bash
source /workspace/spatiallm-env/bin/activate
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('IDEA-Research/grounding-dino-tiny')"
```

- [ ] **Step 4: Commit.**

```bash
git add decroom_no_lidar/backend/spatiallm_server/setup.sh
git commit -m "build(server): pre-cache Grounding DINO weights in setup.sh"
```

---

### Task 4: Backend HTTP client — `rgb_detection.py`

**Files:**
- Create: `decroom_no_lidar/backend/pipeline/rgb_detection.py`
- Create: `decroom_no_lidar/backend/tests/test_rgb_detection.py`
- Create: `decroom_no_lidar/backend/tests/fixtures/grounding_dino_response.json`

- [ ] **Step 1: Capture a real `/detect_furniture` response into the fixture.**

```bash
# On the local machine with the SSH tunnel open (port 8100):
curl -s -X POST -F "images=@/tmp/plan3c_gate/$(ls /tmp/plan3c_gate | head -1)" \
                -F "images=@/tmp/plan3c_gate/$(ls /tmp/plan3c_gate | sed -n 2p)" \
                http://localhost:8100/detect_furniture \
  > decroom_no_lidar/backend/tests/fixtures/grounding_dino_response.json
```

Verify the JSON matches the shape from Task 2 step 3. If empty, raise prompt confidence threshold or re-run.

- [ ] **Step 2: Write the failing test.**

`decroom_no_lidar/backend/tests/test_rgb_detection.py`:

```python
"""Tests for the Grounding DINO HTTP client."""
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
    # All returned items are KeyframeDetection.
    assert all(isinstance(d, KeyframeDetection) for d in detections)
    # Frame indices are within range.
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
    response = MagicMock(); response.status_code = 503
    response.json.return_value = {"error": "gdino not loaded"}
    with patch("pipeline.rgb_detection.requests.post", return_value=response):
        out = detect_furniture_in_keyframes(
            [b"fake"], prompts=["bed"],
            endpoint="http://localhost:8100/detect_furniture",
        )
    # Returns empty list on server error so the upstream pipeline
    # falls back to walls-only (matches existing SpatialLM-failure behavior).
    assert out == []
```

- [ ] **Step 3: Run the test — verify it fails.**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/test_rgb_detection.py -v
```
Expected: ImportError / ModuleNotFoundError on `pipeline.rgb_detection`.

- [ ] **Step 4: Implement `rgb_detection.py`.**

`decroom_no_lidar/backend/pipeline/rgb_detection.py`:

```python
"""Plan 3c: HTTP client for the GPU pod's /detect_furniture endpoint.

Pure data layer. No model code lives here; the GPU work happens on the pod.
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
    """One Grounding DINO detection in one keyframe."""
    frame_index: int
    category: str
    bbox_normalized: Tuple[float, float, float, float]  # (x, y, w, h) in [0,1]
    confidence: float


def detect_furniture_in_keyframes(
    keyframe_jpegs: List[bytes],
    prompts: List[str] = None,
    endpoint: str = DEFAULT_ENDPOINT,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> List[KeyframeDetection]:
    """Send keyframe JPEGs to the GPU pod's /detect_furniture endpoint.

    Returns one KeyframeDetection per accepted detection. Frames with no
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
        logger.warning(
            "rgb_detection server returned %d: %s",
            resp.status_code, resp.json().get("error", ""),
        )
        return []

    body = resp.json()
    out: List[KeyframeDetection] = []
    for frame in body.get("frames", []):
        fi = int(frame.get("frame_index", 0))
        for d in frame.get("detections", []):
            bbox = d.get("bbox_normalized", [0, 0, 0, 0])
            out.append(KeyframeDetection(
                frame_index=fi,
                category=str(d.get("category", "")).lower().strip(),
                bbox_normalized=(float(bbox[0]), float(bbox[1]),
                                  float(bbox[2]), float(bbox[3])),
                confidence=float(d.get("confidence", 0.0)),
            ))
    logger.info(
        "rgb_detection: %d detections across %d frames (%.1fs server)",
        len(out), len(body.get("frames", [])), body.get("elapsed_s", 0),
    )
    return out
```

- [ ] **Step 5: Run the test — verify it passes.**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/test_rgb_detection.py -v
```
Expected: 3 passing.

- [ ] **Step 6: Commit.**

```bash
git add decroom_no_lidar/backend/pipeline/rgb_detection.py \
        decroom_no_lidar/backend/tests/test_rgb_detection.py \
        decroom_no_lidar/backend/tests/fixtures/grounding_dino_response.json
git commit -m "feat(backend): rgb_detection HTTP client for /detect_furniture"
```

---

### Task 5: Live integration smoke test (manual)

**Files:** none (pure verification)

- [ ] **Step 1: Start the pod's server (with both models loaded) and open the SSH tunnel.**

```bash
# On Mac, in a dedicated terminal:
ssh -L 8100:localhost:8100 -p <PORT> -i ~/.ssh/id_ed25519 root@<IP>
# (in the SSH session)
bash /workspace/start-spatiallm.sh
```

Verify both models are loaded:
```bash
curl http://localhost:8100/health
```

- [ ] **Step 2: From the local backend, call the new client end-to-end.**

```bash
cd decroom_no_lidar/backend
python3 -c "
from pathlib import Path
from pipeline.rgb_detection import detect_furniture_in_keyframes
jpegs = [Path(p).read_bytes() for p in sorted(Path('/tmp/plan3c_gate').glob('*.jpg'))]
out = detect_furniture_in_keyframes(jpegs)
for d in out: print(d)
print(f'TOTAL: {len(out)} detections')
"
```
Expected: ≥1 detection per photo on average. If zero, gate (Task 1) regression — investigate before continuing.

- [ ] **Step 3: No commit (verification only). Note observations in `<today>-plan3c-detector-gate.md` under a new "Phase A live integration" section.**

---

### Task 6: Raycast lifting — pure function with synthetic fixtures

**Files:**
- Create: `decroom_no_lidar/backend/pipeline/raycast_lift.py`
- Create: `decroom_no_lidar/backend/tests/test_raycast_lift.py`

- [ ] **Step 1: Write the failing test.**

`decroom_no_lidar/backend/tests/test_raycast_lift.py`:

```python
"""Tests for 2D bbox center → 3D world position via PLY raycast."""
import numpy as np
import pytest

from pipeline.raycast_lift import (
    KeyframeMeta,
    lift_detections_to_3d,
)
from pipeline.rgb_detection import KeyframeDetection


def _synthetic_meta() -> KeyframeMeta:
    """Identity pose, principal point at center, fx=fy=100, 200x200 image."""
    return KeyframeMeta(
        frame_index=0,
        pose_world_from_camera=np.eye(4, dtype=np.float32),
        fx=100.0, fy=100.0, cx=100.0, cy=100.0,
        image_width=200, image_height=200,
    )


def test_centered_bbox_hits_ply_surface():
    # PLY: a wall of 1000 points at z = -2.0, spread over x ∈ [-1, 1], y ∈ [-1, 1].
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1.0, 1.0, 1000).astype(np.float32)
    ys = rng.uniform(-1.0, 1.0, 1000).astype(np.float32)
    zs = np.full(1000, -2.0, dtype=np.float32)
    ply = np.stack([xs, ys, zs], axis=1)

    # Bbox centered at image (100, 100) — principal point — half-image size.
    det = KeyframeDetection(
        frame_index=0, category="bed",
        bbox_normalized=(0.4, 0.4, 0.2, 0.2),  # center at 0.5, 0.5
        confidence=0.8,
    )

    out = lift_detections_to_3d([det], ply, [_synthetic_meta()])
    assert len(out) == 1
    pos = out[0].position_3d
    assert pos is not None
    assert pos[0] == pytest.approx(0.0, abs=0.1)
    assert pos[1] == pytest.approx(0.0, abs=0.1)
    assert pos[2] == pytest.approx(-2.0, abs=0.1)


def test_bbox_aimed_at_empty_space_returns_none():
    # PLY: same wall at z = -2.0, but bbox centered at (0.5, 0.5) on the image
    # while we aim the ray at (-100, 0) where there's no PLY support.
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1.0, 1.0, 1000).astype(np.float32)
    ys = rng.uniform(-1.0, 1.0, 1000).astype(np.float32)
    zs = np.full(1000, -2.0, dtype=np.float32)
    ply = np.stack([xs, ys, zs], axis=1)

    # Detection bbox center maps to image (180, 100); ray exits the wall.
    det = KeyframeDetection(
        frame_index=0, category="bed",
        bbox_normalized=(0.85, 0.4, 0.1, 0.2),  # center near 0.9
        confidence=0.8,
    )
    # With fx=100, image x=180 → camera x = (180-100)/100 * (-2.0) = ... actually
    # the ray sweeps to x = -1.6 at z=-2.0, still within wall (x ∈ [-1, 1]).
    # Make the cy/fy combo that pushes the ray outside — use a tighter wall.
    ply = np.stack([
        rng.uniform(-0.3, 0.3, 1000).astype(np.float32),
        ys, zs,
    ], axis=1)

    out = lift_detections_to_3d([det], ply, [_synthetic_meta()])
    assert len(out) == 1
    assert out[0].position_3d is None


def test_drops_detection_when_frame_meta_missing():
    det = KeyframeDetection(
        frame_index=99,  # no meta for this frame
        category="bed",
        bbox_normalized=(0.5, 0.5, 0.1, 0.1),
        confidence=0.8,
    )
    out = lift_detections_to_3d(
        [det], np.zeros((10, 3), dtype=np.float32), [_synthetic_meta()],
    )
    # Detection passed through with position_3d = None (caller filters).
    assert len(out) == 1
    assert out[0].position_3d is None
```

- [ ] **Step 2: Run — verify failure.**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/test_raycast_lift.py -v
```
Expected: ImportError on `pipeline.raycast_lift`.

- [ ] **Step 3: Implement `raycast_lift.py`.**

`decroom_no_lidar/backend/pipeline/raycast_lift.py`:

```python
"""Plan 3c: lift 2D bbox centers to 3D world positions via PLY raycast.

The phone-built PLY is voxel-deduped at ~2 cm and is materially less noisy
than per-pixel depth at the bbox center (which is what YOLO-on-phone uses).
For each detection, build the camera ray through the bbox center, transform
to world space, and march along it until the nearest PLY voxel is within
`shell_tolerance` meters. That voxel is the lifted position.

If no voxel is within the shell along the entire ray, return position_3d=None.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .rgb_detection import KeyframeDetection

logger = logging.getLogger("decroom-backend.raycast_lift")

RAY_STEP_M = 0.05          # 5 cm per step along the ray
RAY_MAX_M = 8.0            # don't search beyond 8 m (room max diagonal)
SHELL_TOLERANCE_M = 0.08   # accept first PLY point within 8 cm of ray


@dataclass
class KeyframeMeta:
    frame_index: int
    pose_world_from_camera: np.ndarray  # 4x4 row-major
    fx: float; fy: float
    cx: float; cy: float
    image_width: int; image_height: int


@dataclass
class LiftedDetection:
    """KeyframeDetection augmented with a 3D world position (or None)."""
    frame_index: int
    category: str
    bbox_normalized: Tuple[float, float, float, float]
    confidence: float
    position_3d: Optional[Tuple[float, float, float]] = None


def _bbox_center_pixel(
    bbox_normalized: Tuple[float, float, float, float],
    width: int, height: int,
) -> Tuple[float, float]:
    x, y, w, h = bbox_normalized
    return ((x + w / 2.0) * width, (y + h / 2.0) * height)


def _ray_in_world(
    pixel_u: float, pixel_v: float, meta: KeyframeMeta,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (origin, direction) in world space."""
    x_cam = (pixel_u - meta.cx) / meta.fx
    y_cam = (pixel_v - meta.cy) / meta.fy
    # ARKit camera looks down -Z, so direction is (x, y, -1).
    dir_cam = np.array([x_cam, y_cam, -1.0, 0.0], dtype=np.float32)
    pose = meta.pose_world_from_camera
    dir_world = (pose @ dir_cam)[:3]
    norm = np.linalg.norm(dir_world)
    if norm > 0:
        dir_world = dir_world / norm
    origin_world = pose[:3, 3].astype(np.float32)
    return origin_world, dir_world.astype(np.float32)


def lift_detections_to_3d(
    detections: List[KeyframeDetection],
    ply_positions: np.ndarray,
    keyframe_metas: List[KeyframeMeta],
    step_m: float = RAY_STEP_M,
    max_m: float = RAY_MAX_M,
    shell_tolerance_m: float = SHELL_TOLERANCE_M,
) -> List[LiftedDetection]:
    """For each detection, raycast through the bbox center into the PLY.

    Detections whose frame has no matching meta, or whose ray finds no PLY
    support within shell_tolerance along its first max_m, return position_3d=None.
    """
    out: List[LiftedDetection] = []
    if ply_positions is None or len(ply_positions) == 0:
        return [
            LiftedDetection(
                frame_index=d.frame_index, category=d.category,
                bbox_normalized=d.bbox_normalized, confidence=d.confidence,
                position_3d=None,
            )
            for d in detections
        ]

    meta_by_frame = {m.frame_index: m for m in keyframe_metas}
    tree = cKDTree(ply_positions)
    n_steps = int(max_m / step_m)

    for det in detections:
        meta = meta_by_frame.get(det.frame_index)
        if meta is None:
            out.append(LiftedDetection(
                frame_index=det.frame_index, category=det.category,
                bbox_normalized=det.bbox_normalized, confidence=det.confidence,
                position_3d=None,
            ))
            continue

        u, v = _bbox_center_pixel(
            det.bbox_normalized, meta.image_width, meta.image_height,
        )
        origin, direction = _ray_in_world(u, v, meta)

        hit_pos: Optional[Tuple[float, float, float]] = None
        for k in range(1, n_steps + 1):
            sample = origin + direction * (k * step_m)
            dist, idx = tree.query(sample, k=1)
            if dist <= shell_tolerance_m:
                hit = ply_positions[idx]
                hit_pos = (float(hit[0]), float(hit[1]), float(hit[2]))
                break

        out.append(LiftedDetection(
            frame_index=det.frame_index, category=det.category,
            bbox_normalized=det.bbox_normalized, confidence=det.confidence,
            position_3d=hit_pos,
        ))

    n_hits = sum(1 for d in out if d.position_3d is not None)
    logger.info(
        "raycast_lift: %d/%d detections lifted to 3D", n_hits, len(out),
    )
    return out
```

- [ ] **Step 4: Run — verify pass.**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/test_raycast_lift.py -v
```
Expected: 3 passing.

- [ ] **Step 5: Commit.**

```bash
git add decroom_no_lidar/backend/pipeline/raycast_lift.py \
        decroom_no_lidar/backend/tests/test_raycast_lift.py
git commit -m "feat(backend): raycast_lift — bbox center → 3D via PLY raycast"
```

---

### Task 7: Verify scipy is available in backend

**Files:** none (env check)

- [ ] **Step 1: Confirm scipy is already an installed dependency.**

```bash
cd decroom_no_lidar/backend && python3 -c "import scipy; print(scipy.__version__)"
```
Expected: a version prints. If ModuleNotFoundError, add scipy to whatever requirements file the backend uses (check `requirements.txt` or `pyproject.toml` in backend root) and install.

If not installed:
```bash
pip install scipy
# also add to requirements.txt
```

Then commit any requirements update.

---

### Task 8: Wire raycast input from `keyframes_manifest`

**Files:**
- Modify: `decroom_no_lidar/backend/pipeline/raycast_lift.py` (small helper)
- Modify: `decroom_no_lidar/backend/tests/test_raycast_lift.py` (one new test)

- [ ] **Step 1: Write the failing test for the manifest adapter.**

Append to `tests/test_raycast_lift.py`:

```python
def test_keyframe_metas_from_manifest_parses_pose_intrinsics():
    from pipeline.raycast_lift import keyframe_metas_from_manifest

    manifest = {
        "frames": [
            {
                "filename": "kf_000.jpg",
                "timestamp": 0.0,
                "pose_world_from_camera": [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                ],
                "intrinsics": {
                    "fx": 100.0, "fy": 100.0,
                    "cx": 50.0, "cy": 50.0,
                    "width": 100, "height": 100,
                },
                "detections": [],
            },
            {
                "filename": "kf_001.jpg",
                "timestamp": 1.0,
                "pose_world_from_camera": [
                    0, 1, 0, 5,
                    1, 0, 0, 0,
                    0, 0, -1, -3,
                    0, 0, 0, 1,
                ],
                "intrinsics": {
                    "fx": 200.0, "fy": 200.0,
                    "cx": 100.0, "cy": 75.0,
                    "width": 200, "height": 150,
                },
                "detections": [],
            },
        ]
    }
    metas = keyframe_metas_from_manifest(manifest)
    assert len(metas) == 2
    assert metas[0].frame_index == 0
    assert metas[0].fx == 100.0
    assert metas[1].frame_index == 1
    assert metas[1].image_width == 200
    # pose row 0 of frame 1 is [0,1,0,5] → first row of 4x4
    assert list(metas[1].pose_world_from_camera[0]) == [0, 1, 0, 5]
```

- [ ] **Step 2: Run — verify failure.**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/test_raycast_lift.py::test_keyframe_metas_from_manifest_parses_pose_intrinsics -v
```
Expected: ImportError on `keyframe_metas_from_manifest`.

- [ ] **Step 3: Add the helper.**

In `pipeline/raycast_lift.py`, append:

```python
def keyframe_metas_from_manifest(manifest: dict) -> List[KeyframeMeta]:
    """Parse the keyframes.json dict the phone uploads into KeyframeMeta list.

    The phone serializes pose row-major (per ScanUploadPayload.encodeKeyframesManifest),
    so reshape to (4,4) directly.
    """
    out: List[KeyframeMeta] = []
    for i, f in enumerate(manifest.get("frames", [])):
        pose_flat = f.get("pose_world_from_camera", [])
        if len(pose_flat) != 16:
            logger.warning("frame %d: bad pose length %d", i, len(pose_flat))
            continue
        pose = np.array(pose_flat, dtype=np.float32).reshape(4, 4)
        intr = f.get("intrinsics", {})
        out.append(KeyframeMeta(
            frame_index=i,
            pose_world_from_camera=pose,
            fx=float(intr.get("fx", 0)), fy=float(intr.get("fy", 0)),
            cx=float(intr.get("cx", 0)), cy=float(intr.get("cy", 0)),
            image_width=int(intr.get("width", 0)),
            image_height=int(intr.get("height", 0)),
        ))
    return out
```

- [ ] **Step 4: Run — verify pass.**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/test_raycast_lift.py -v
```
Expected: 4 passing.

- [ ] **Step 5: Commit.**

```bash
git add decroom_no_lidar/backend/pipeline/raycast_lift.py \
        decroom_no_lidar/backend/tests/test_raycast_lift.py
git commit -m "feat(backend): keyframe_metas_from_manifest helper"
```

---

### Task 9: Cross-keyframe clustering — pure function

**Files:**
- Create: `decroom_no_lidar/backend/pipeline/keyframe_clustering.py`
- Create: `decroom_no_lidar/backend/tests/test_keyframe_clustering.py`

- [ ] **Step 1: Write the failing test.**

`decroom_no_lidar/backend/tests/test_keyframe_clustering.py`:

```python
"""Tests for cross-keyframe NN-merge of lifted detections."""
import pytest

from pipeline.keyframe_clustering import (
    cluster_keyframe_detections,
    BACKEND_TRACKER_DISTANCES,
)
from pipeline.raycast_lift import LiftedDetection


def _det(frame, cat, pos, conf=0.8):
    return LiftedDetection(
        frame_index=frame, category=cat,
        bbox_normalized=(0.4, 0.4, 0.2, 0.2), confidence=conf,
        position_3d=pos,
    )


def test_same_category_near_positions_merge():
    dets = [
        _det(0, "bed", (2.0, 0.0, 1.5)),
        _det(1, "bed", (2.05, 0.0, 1.45)),  # 7 cm away
        _det(2, "bed", (1.95, 0.0, 1.55)),  # similar
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 1
    bed = clusters[0]
    assert bed["category"] == "bed"
    assert bed["view_count"] == 3
    # Position is from the highest-confidence (or first) detection.
    assert bed["position"][0] == pytest.approx(2.0, abs=0.1)


def test_same_category_far_positions_stay_separate():
    dets = [
        _det(0, "chair", (1.0, 0.0, 1.0)),
        _det(1, "chair", (3.5, 0.0, 1.0)),  # 2.5 m away — beyond chair threshold
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 2


def test_different_categories_stay_separate_at_same_position():
    dets = [
        _det(0, "bed", (2.0, 0.0, 1.5)),
        _det(1, "desk", (2.0, 0.0, 1.5)),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 2


def test_drops_detections_with_no_3d_position():
    dets = [
        _det(0, "bed", None),
        _det(1, "bed", (2.0, 0.0, 1.5)),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 1
    assert clusters[0]["view_count"] == 1


def test_picks_highest_confidence_as_canonical():
    dets = [
        _det(0, "bed", (2.0, 0.0, 1.5), conf=0.5),
        _det(1, "bed", (2.05, 0.0, 1.55), conf=0.9),
        _det(2, "bed", (1.95, 0.0, 1.45), conf=0.6),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 1
    bed = clusters[0]
    assert bed["peak_confidence"] == pytest.approx(0.9)
    assert bed["best_frame_index"] == 1


def test_threshold_lookup_falls_back_to_default():
    assert BACKEND_TRACKER_DISTANCES.get("__missing__") is None
    # cluster_keyframe_detections must not crash on unknown category.
    dets = [
        _det(0, "alien_thing", (2.0, 0, 1.5)),
        _det(1, "alien_thing", (2.05, 0, 1.55)),
    ]
    clusters = cluster_keyframe_detections(dets)
    assert len(clusters) == 1  # default threshold merges them
```

- [ ] **Step 2: Run — verify failure.**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/test_keyframe_clustering.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement.**

`decroom_no_lidar/backend/pipeline/keyframe_clustering.py`:

```python
"""Plan 3c: cross-keyframe NN-merge of lifted detections.

A real bed seen in 5 keyframes produces 5 LiftedDetection entries with
similar 3D positions. This module collapses them into one tracked
object before they enter Plan 3b's consensus pipeline.

Output shape mirrors the on-device ObjectTracker output (yolo_objects.json),
so the merged list can be appended to payload.yolo_objects unchanged.
"""
from __future__ import annotations

import logging
import math
import uuid
from typing import Dict, List

from .raycast_lift import LiftedDetection

logger = logging.getLogger("decroom-backend.keyframe_clustering")

# Per-category 3D NN-merge distance for cross-keyframe clustering. Looser
# than consensus.py's collapse_duplicates because we're merging multiple
# views of the SAME physical object (high overlap expected); collapse
# distances are tighter to avoid merging two real adjacent objects.
BACKEND_TRACKER_DISTANCES: Dict[str, float] = {
    "bed":          0.6,
    "sofa":         0.6,
    "couch":        0.6,
    "chair":        0.4,
    "stool":        0.4,
    "dining_table": 0.5,
    "desk":         0.5,
    "monitor":      0.5,
    "tv":           0.5,
    "laptop":       0.3,
    "lamp":         0.3,
    "lamp_floor":   0.3,
    "lamp_desk":    0.3,
    "potted plant": 0.3,
    "potted_plant": 0.3,
    "plant":        0.3,
    "bookshelf":    0.5,
    "dresser":      0.5,
    "wardrobe":     0.6,
    "nightstand":   0.4,
    "rug":          0.5,
}
DEFAULT_MERGE_DISTANCE = 0.5


def _distance(a, b) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def cluster_keyframe_detections(
    detections: List[LiftedDetection],
    merge_distance_by_category: Dict[str, float] = None,
) -> List[Dict]:
    """NN-merge same-category detections within per-category 3D distance.

    Returns a list of dicts in the on-device yolo_objects.json shape:
      {category, position [x,y,z], peak_confidence, view_count, best_frame_index}

    Detections with position_3d=None are dropped.
    """
    thresholds = merge_distance_by_category or BACKEND_TRACKER_DISTANCES

    # Drop unlifted detections.
    valid = [d for d in detections if d.position_3d is not None]

    # Sort by confidence desc so the highest-confidence detection seeds
    # each cluster (canonical).
    valid.sort(key=lambda d: d.confidence, reverse=True)

    clusters: List[Dict] = []
    for det in valid:
        threshold = thresholds.get(det.category, DEFAULT_MERGE_DISTANCE)
        merged = False
        for c in clusters:
            if c["category"] != det.category:
                continue
            if _distance(c["position"], det.position_3d) <= threshold:
                c["view_count"] += 1
                if det.confidence > c["peak_confidence"]:
                    c["peak_confidence"] = det.confidence
                    c["best_frame_index"] = det.frame_index
                    c["position"] = list(det.position_3d)
                merged = True
                break
        if not merged:
            clusters.append({
                "id": f"rgb_{uuid.uuid4().hex[:6]}",
                "category": det.category,
                "position": list(det.position_3d),
                "peak_confidence": det.confidence,
                "view_count": 1,
                "best_frame_index": det.frame_index,
                "_source": "rgb",
            })

    logger.info(
        "keyframe_clustering: %d detections -> %d clusters",
        len(valid), len(clusters),
    )
    return clusters
```

- [ ] **Step 4: Run — verify pass.**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/test_keyframe_clustering.py -v
```
Expected: 6 passing.

- [ ] **Step 5: Commit.**

```bash
git add decroom_no_lidar/backend/pipeline/keyframe_clustering.py \
        decroom_no_lidar/backend/tests/test_keyframe_clustering.py
git commit -m "feat(backend): cross-keyframe NN-merge of lifted detections"
```

---

### Task 10: Audit `CATEGORY_TO_ASSET` for prompt-list coverage

**Files:**
- Modify: `decroom_no_lidar/backend/pipeline/consensus.py` (only if a category is missing)
- Add to: `decroom_no_lidar/backend/tests/test_consensus.py` (small assert test)

- [ ] **Step 1: Write the failing test.**

Append to `tests/test_consensus.py`:

```python
def test_every_rgb_prompt_category_resolves_to_asset():
    """Plan 3c: every category in the default Grounding DINO prompt list
    must have a CATEGORY_TO_ASSET mapping so consensus doesn't drop them."""
    from pipeline.consensus import CATEGORY_TO_ASSET, asset_id_for
    from pipeline.rgb_detection import DEFAULT_PROMPT_CATEGORIES

    missing = [c for c in DEFAULT_PROMPT_CATEGORIES
               if c not in CATEGORY_TO_ASSET
               and c.replace(" ", "_") not in CATEGORY_TO_ASSET]
    assert missing == [], (
        f"Plan 3c prompt categories without asset mapping: {missing}. "
        f"Add to consensus.CATEGORY_TO_ASSET (substitute is fine — "
        f"e.g. wardrobe → dresser)."
    )
```

- [ ] **Step 2: Run — likely passes (current map already has the categories), but verify.**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/test_consensus.py::test_every_rgb_prompt_category_resolves_to_asset -v
```

If FAIL: add the missing key(s) to `pipeline/consensus.py`'s `CATEGORY_TO_ASSET` dict, mapping to the closest bundled Kenney asset (e.g., `"sectional": "sofa"`). Re-run.

- [ ] **Step 3: Commit.**

```bash
git add decroom_no_lidar/backend/tests/test_consensus.py decroom_no_lidar/backend/pipeline/consensus.py
git commit -m "test(consensus): assert RGB prompt categories all map to assets"
```

---

### Task 11: Wire RGB detection into `phone_first_pipeline`

**Files:**
- Modify: `decroom_no_lidar/backend/pipeline/phone_first_pipeline.py`

- [ ] **Step 1: Add the helper that merges phone YOLO objects with backend RGB clusters.**

Insert near `_build_final_objects` (around line 636):

```python
def _merge_phone_and_backend_detections(
    phone_yolo: List[Dict[str, Any]],
    backend_rgb: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Combine on-device YOLO tracks with backend RGB clusters into a
    single 'phone+backend detection' list.

    Both inputs share the yolo_objects.json shape:
      {category, position[x,y,z], peak_confidence, view_count, best_frame_index}

    No deduplication here — collapse_duplicates inside _build_final_objects
    handles cross-source duplicates.
    """
    merged = list(phone_yolo or [])
    merged.extend(backend_rgb or [])
    return merged
```

- [ ] **Step 2: Add the RGB-detection step inside `run_phone_first_pipeline`.**

Replace the call to `_build_final_objects` (around line 821) with:

```python
    # Plan 3c: backend open-vocab RGB detection on the keyframe JPEGs.
    # SpatialLM can't detect furniture from phone-built monocular-depth
    # PLYs (training distribution mismatch); Grounding DINO on the RGB
    # keyframes does. Lifted to 3D via PLY raycast and clustered across
    # frames before feeding into the consensus pipeline.
    from .rgb_detection import detect_furniture_in_keyframes
    from .raycast_lift import lift_detections_to_3d, keyframe_metas_from_manifest
    from .keyframe_clustering import cluster_keyframe_detections

    rgb_clusters: List[Dict[str, Any]] = []
    try:
        rgb_dets_2d = detect_furniture_in_keyframes(payload.keyframe_jpegs)
        if rgb_dets_2d:
            metas = keyframe_metas_from_manifest(payload.keyframes_manifest)
            lifted = lift_detections_to_3d(
                rgb_dets_2d, positions, metas,
            )
            rgb_clusters = cluster_keyframe_detections(lifted)
            logger.info(
                "Plan 3c: %d 2D detections -> %d 3D clusters",
                len(rgb_dets_2d), len(rgb_clusters),
            )
    except Exception as e:
        logger.warning("Plan 3c RGB detection failed: %s -- continuing without", e)

    combined_detections = _merge_phone_and_backend_detections(
        payload.yolo_objects, rgb_clusters,
    )

    # Plan 3b consensus pipeline: take the SLM-mapped objects and the
    # combined phone-YOLO + backend-RGB 3D detections, run them through
    # the four polish functions, and emit final RoomObjects.
    objects = _build_final_objects(
        slm_objects or [],
        combined_detections,
        phone_walls_with_openings,
    )
```

- [ ] **Step 3: Run the existing pipeline tests — confirm none break (mocks may be needed).**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/test_phone_first_pipeline.py -v
```

If existing tests fail because they don't mock `detect_furniture_in_keyframes`: the function is called but transport will fail (no tunnel in test env), the `except` swallows it, and `rgb_clusters` stays empty. Tests should still pass. If they do fail, fix the test fixtures in Task 12.

- [ ] **Step 4: Commit.**

```bash
git add decroom_no_lidar/backend/pipeline/phone_first_pipeline.py
git commit -m "feat(backend): wire RGB detection into phone_first_pipeline"
```

---

### Task 12: Update `test_phone_first_pipeline.py` with RGB mock

**Files:**
- Modify: `decroom_no_lidar/backend/tests/test_phone_first_pipeline.py`

- [ ] **Step 1: Inspect the existing test setup for `run_spatiallm_pipeline` mocking pattern.**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/test_phone_first_pipeline.py -v --collect-only
```

Identify where existing tests patch `run_spatiallm_pipeline`. New tests will use the same `unittest.mock.patch` style.

- [ ] **Step 2: Add a test that backend RGB clusters reach `_build_final_objects`.**

Append to `tests/test_phone_first_pipeline.py`:

```python
def test_rgb_clusters_merge_into_combined_detections(monkeypatch, tmp_path):
    """Plan 3c: when /detect_furniture returns clusters, they appear in
    the combined detection list passed to _build_final_objects."""
    from pipeline import phone_first_pipeline as pfp
    from pipeline.rgb_detection import KeyframeDetection
    from pipeline.raycast_lift import LiftedDetection

    fake_2d = [
        KeyframeDetection(frame_index=0, category="wardrobe",
                          bbox_normalized=(0.4, 0.4, 0.2, 0.2),
                          confidence=0.7),
    ]
    fake_lifted = [
        LiftedDetection(frame_index=0, category="wardrobe",
                        bbox_normalized=(0.4, 0.4, 0.2, 0.2),
                        confidence=0.7,
                        position_3d=(2.0, 0.0, 1.5)),
    ]

    monkeypatch.setattr(pfp, "run_spatiallm_pipeline",
                         lambda points, cfg: ([], []))

    captured = {}
    real_build = pfp._build_final_objects

    def spy_build(slm_objects, yolo_objects_3d, phone_walls):
        captured["combined"] = yolo_objects_3d
        return real_build(slm_objects, yolo_objects_3d, phone_walls)

    monkeypatch.setattr(pfp, "_build_final_objects", spy_build)

    # Patch the new RGB stack inline.
    import pipeline.rgb_detection as rgb_mod
    import pipeline.raycast_lift as lift_mod
    monkeypatch.setattr(rgb_mod, "detect_furniture_in_keyframes",
                         lambda *a, **k: fake_2d)
    monkeypatch.setattr(lift_mod, "lift_detections_to_3d",
                         lambda *a, **k: fake_lifted)

    # Use the existing fixture payload with one phone YOLO object.
    payload = _make_minimal_payload(
        yolo_objects=[{
            "category": "chair",
            "position": [1.0, 0.0, 1.0],
            "peak_confidence": 0.9,
            "view_count": 3,
            "best_frame_index": 0,
        }],
    )
    cfg = _stub_cfg()
    pfp.run_phone_first_pipeline(payload, cfg)

    cats = sorted(o.get("category") for o in captured["combined"])
    assert "chair" in cats
    assert "wardrobe" in cats
```

`_make_minimal_payload` and `_stub_cfg` are existing helpers in this file. Reuse them; if naming differs in the current code, match the actual helpers.

- [ ] **Step 3: Run — verify pass.**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/test_phone_first_pipeline.py -v
```

- [ ] **Step 4: Commit.**

```bash
git add decroom_no_lidar/backend/tests/test_phone_first_pipeline.py
git commit -m "test(pipeline): RGB clusters merge into combined detections"
```

---

### Task 13: Full backend test suite green

**Files:** none (verification)

- [ ] **Step 1: Run the entire backend test suite.**

```bash
cd decroom_no_lidar/backend && python3 -m pytest tests/ -v
```
Expected: all tests pass. If anything is red, fix before continuing.

- [ ] **Step 2: Commit any fix-ups.**

If fixes were made, commit with descriptive messages.

---

### Task 14: Phase E — manual end-to-end QA + merge

**Files:**
- Create: `docs/superpowers/qa/<today>-plan3c-manual-qa.md`

- [ ] **Step 1: Pod readiness.**

```bash
ssh -L 8100:localhost:8100 -p <PORT> -i ~/.ssh/id_ed25519 root@<IP>
# in pod:
bash /workspace/start-spatiallm.sh
# verify both models loaded:
curl http://localhost:8100/health
```

- [ ] **Step 2: Start the local backend.**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
export DECROOM_ENABLE_SPATIALLM=true
export DECROOM_SPATIALLM_API_URL=http://localhost:8100/infer
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

- [ ] **Step 3: Run a full scan of the QA bedroom on device. Capture results.**

Drive the iOS app through the existing capture flow (no iOS changes). Wait for the backend RoomPayload response. Inspect the rendered SceneKit view.

- [ ] **Step 4: Document findings.**

Create `docs/superpowers/qa/<today>-plan3c-manual-qa.md`:

```markdown
# Plan 3c Manual QA — End-to-End Scan

**Date**: <YYYY-MM-DD>
**Room**: 3.04 × 3.25 m bedroom (same as Plan 3b QA)
**Real furniture present**: bed, wardrobe, bookshelf, desk, 2 monitors, chair, door

## Detected objects (after Plan 3c full pipeline)

| Category | Count | Notes (placement, snap, orientation) |
|----------|-------|--------------------------------------|
| bed      |       |                                      |
| wardrobe |       |                                      |
| bookshelf|       |                                      |
| desk     |       |                                      |
| monitor  |       |                                      |
| chair    |       |                                      |

## Walls / openings
- Wall count: 4 ✓ / ✗
- Door propagated to correct wall: ✓ / ✗

## Backend log highlights
(paste the `Plan 3c: N 2D detections -> M 3D clusters` line and any warnings)

## Comparison to Plan 3b QA
- Plan 3b: 1 chair + 3 ghost monitors, 0 wardrobe/bookshelf/desk/bed.
- Plan 3c: <fill in>

## Verdict
- [ ] Furniture detection materially improved → merge.
- [ ] Detection improved but ghost duplicates worse than 3b → tune merge_distance_by_category in keyframe_clustering.
- [ ] Detection regressed → investigate before merge.
```

- [ ] **Step 5: If verdict = merge, merge to main.**

```bash
git checkout main
git pull
git merge --no-ff feature/plan3c-backend-rgb -m "Merge Plan 3c — backend open-vocab RGB detection"
git push origin main
```

- [ ] **Step 6: Update memory.**

Update `memory/plan_status.md` to mark Plan 3c complete + merged. Update `memory/known_issues.md` to remove the SpatialLM-furniture-detection entry now that it's bypassed (but keep the diagnostic detail in a "Resolved" section).

- [ ] **Step 7: Commit doc + memory updates.**

```bash
git add docs/superpowers/qa/<today>-plan3c-manual-qa.md
# Memory updates live outside the repo (~/.claude/projects/.../memory/) — no git add for those.
git commit -m "qa(plan3c): manual QA results — <verdict>"
git push
```

---

## Self-review

Spec coverage check:
- ✓ Phase A0 model-quality gate (spec §Migration plan Phase A step 1, expanded after spec edit) — Task 1.
- ✓ Backend RGB detector standalone (spec §Components 1, §Migration A) — Tasks 2–5.
- ✓ Raycast lifting (spec §Components 2, §Migration B) — Tasks 6–8.
- ✓ Cross-keyframe clustering (spec §Components 3, §Migration C) — Task 9.
- ✓ Phone+backend detection merge into consensus (spec §Components 4) — Task 11 (`_merge_phone_and_backend_detections`).
- ✓ Whitelist + asset map audit (spec §Components 5) — Task 10.
- ✓ Server.py + setup.sh changes (spec §Components 6) — Tasks 2 + 3.
- ✓ Migration phases A–F — Tasks map 1:1.
- ✓ Risks: GPU memory pressure (Tasks 2–3 verify), latency increase (acceptable per spec), open-vocab false positives (default box_threshold=0.4), raycast misses (drop with `position_3d=None`).
- ✓ Open question: prompt list — `DEFAULT_PROMPT_CATEGORIES` in `rgb_detection.py` is the source of truth and tunable post-Phase A0.
- ✓ Open question: drop on-device YOLO — explicitly deferred per spec; `_merge_phone_and_backend_detections` keeps both sources alive.

Placeholder scan: no TBD/TODO; all code blocks are concrete; date placeholders are `<YYYY-MM-DD>` / `<today>` (filled at write-time, common convention).

Type/symbol consistency:
- `KeyframeDetection` defined in Task 4 (rgb_detection.py), consumed in Task 6.
- `LiftedDetection` defined in Task 6 (raycast_lift.py), consumed in Task 9.
- `KeyframeMeta` defined in Task 6, populated by `keyframe_metas_from_manifest` in Task 8.
- `cluster_keyframe_detections` output dict shape matches `payload.yolo_objects` shape consumed by `_build_final_objects` (verified against `phone_first_pipeline.py` line 638-690 in current code).
- `_merge_phone_and_backend_detections` argument order (phone_yolo, backend_rgb) matches the call site in Task 11 step 2.
- `DEFAULT_PROMPT_CATEGORIES` exported by Task 4, asserted in Task 10.
- `BACKEND_TRACKER_DISTANCES` exported by Task 9, asserted in Task 9 test.

Done.
