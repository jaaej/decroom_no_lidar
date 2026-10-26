# Decroom — Room Scanning iOS App

## Project Structure
- **iOS app**: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/` (SwiftUI + ARKit + RealityKit + SceneKit)
- **Backend**: `decroom_no_lidar/decroom_no_lidar/backend/` (FastAPI + Python)
- **SpatialLM GPU server**: `decroom_no_lidar/decroom_no_lidar/backend/spatiallm_server/`
- **GitHub**: https://github.com/jaaej/decroom_no_lidar.git

## Build Commands
```bash
# iOS app
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -scheme decroom_no_lidar -project /Users/jaejeon/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj -destination 'generic/platform=iOS' build

# Backend
cd /Users/jaejeon/Decroom/decroom_no_lidar/decroom_no_lidar/backend
export DECROOM_ENABLE_SPATIALLM=true
export DECROOM_SPATIALLM_API_URL=http://localhost:8100/infer
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Critical Rules
- ARKit `capturedImage` is ALWAYS landscape-right — apply `.oriented(.right)` and swap intrinsics for portrait
- SceneKit `eulerAngles.y` is CW positive — negate backend `rotation_y` (CCW positive)
- USDZ Kenney models need bounding box measurement + median scale factor + Y-offset to floor
- Do NOT make changes the user didn't ask for — no extra refactoring, no unsolicited docstrings
- **If the user is wrong, say so directly. If a task is low quality or has a better approach, say so once — without being preachy — then do the task if the user confirms. Never agree with something incorrect to avoid friction.**
- **NEVER recommend destructive iOS actions without first stating exactly what user data will be lost.** This includes: "Delete App" (wipes the entire app sandbox — `Documents/decroom_saved_rooms.json` and `Documents/last_payload/` are GONE), "Reset Content & Settings" on the device, force-erasing a simulator, or any `xcrun simctl erase`. Decroom has NO cloud backup — once these are deleted, they cannot be recovered. Default to non-destructive alternatives (Clean Build Folder, `xcodebuild clean`, just hitting Run again to reinstall over the top, etc.). If a destructive action genuinely is the right move, first offer to pull a backup via `xcrun devicectl device copy from` and wait for the user to confirm before proceeding.

## Anti-sycophancy self-checks (run these AFTER any non-trivial recommendation, BEFORE replying)
The user has caught a recurring pattern: confidently endorsing my own plan, then later finding a better one. To break the consistency bias that causes this, ask myself these three questions explicitly after every plan or recommendation. Surface anything they reveal — do not bury it.

1. **What would change my mind about this?** Enumerate concrete conditions under which this plan would be wrong. If I can't name 2-3, I haven't actually evaluated it — I've just defended it.
2. **If I came to this cold with no prior context, would I still recommend it?** This catches inherited-premise blindness. If the answer is "no, I'd recommend X instead," surface X.
3. **Why isn't there a better option?** Forces a search outside the current frame. If I haven't checked existing memory files (especially `pipeline_improvements_research.md` and `reasoning_anti_patterns.md`), do that before answering.

When the user asks "is this good?" — treat that as "stress-test this," not "validate me." Lead with uncertainties and gaps, not with confidence.

## Verify before claiming

Any claim about a value, behavior, or current state — whether sourced from an agent report, code reading, a paper, web search, or memory — is a HYPOTHESIS until verified by running a command or reading actual data. Before including such a claim in a recommendation, tradeoff table, or "issues to fix" list, run the verification first and report what was actually observed.

Code-reading (mine or an agent's) is hypothesis generation, not evidence. Two agents independently reading the same code produce two hypotheses, not convergent evidence — they may both be wrong in the same way.

Verification is cheap. If it takes <5 minutes (print a value, grep a file, run a one-liner), just do it. If it takes longer, surface the cost explicitly and let the user decide whether to verify or proceed on the hypothesis.

Specific failure mode this prevents: synthesizing an agent's "X is broken" finding into "fix X" without running the 30-second check that would have shown X is fine. Recurring pattern flagged across multiple sessions — including the 2026-05-24 BoxFusion audit where I claimed an intrinsics rotation bug existed and a gravity convention was wrong, both of which collapsed on a 30-second Python print.

## Architecture (Phone-First — decided 2026-04-16)
- **Design direction**: phone does geometry (ARKit planes + on-device depth), backend adds semantics only
- **Current MVP capture**: VideoRecorderView (existing capture UI with coverage arc + upload)
- **Backend**: SpatialLM for semantic labeling, YOLO for detection refinement
- COLMAP SfM disabled when SpatialLM is enabled
- 13 Kenney USDZ furniture models (CC0), mapped via `kenney_{type}.usdz`
- **Design spec**: `docs/superpowers/specs/2026-04-16-phone-first-ar-architecture-design.md`
- **Plan 1** (foundation): MERGED to main — DeviceCapabilities, ScanLogger, DepthRunner, RoomGeometryBuilder, PointCloudAccumulator, KeyframeSelector
- **Plan 2** (visual capture): branch `feature/phone-first-visual` — code done but visual UI deferred, Tasks 11-12 pending
- **Plans 3-4**: not started (backend integration + correction UI)

## Important: Do NOT re-attempt these scanning UIs
Multiple visual approaches for the scanning experience were tried and ALL failed:
1. Panoramic mosaic (discrete tiles) — not seamless
2. 3D wireframe grid shell — confusing, too small
3. Floating AR reticle — too abstract
4. Equirectangular panorama buffer — images distorted/sideways
5. Flat panorama with edge blending — still looked tiled
6. Text prompts + arrows — ambiguous
See memory file `visual_ui_failures.md` for details. The current approach is functional MVP (VideoRecorderView).

## Worktree Rules
- When recreating worktrees, add to `.gitignore`
- Copy any `.env` files into the worktree

## Key Gotchas
- Object merge distance must be generous (0.8m+ for monitors) due to depth noise
- SpatialLM requires Z-up coordinates: ARKit (X,Y,Z) → SpatialLM (X,-Z,Y)
- SpatialLM server needs `spatiallm.model.spatiallm_qwen` import path (not `spatiallm.model`)
- `trust_remote_code=True` required for SpatialLM tokenizer
- RunPod GPU server: upload via SCP, run via Jupyter terminal, SSH tunnel for port 8100
