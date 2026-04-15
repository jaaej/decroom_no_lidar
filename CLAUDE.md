# Decroom — Room Scanning iOS App

## Project Structure
- **iOS app**: `decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/` (SwiftUI + ARKit + RealityKit + SceneKit)
- **Backend**: `decroom_no_lidar/decroom_no_lidar/backend/` (FastAPI + Python)
- **SpatialLM GPU server**: `decroom_no_lidar/decroom_no_lidar/backend/spatiallm_server/`
- **GitHub**: https://github.com/jaaej/decroom_no_lidar.git

## Build Commands
```bash
# iOS app
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj -destination 'generic/platform=iOS' build

# Backend
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/backend
export DECROOM_ENABLE_SPATIALLM=true
export DECROOM_SPATIALLM_API_URL=http://localhost:8100/infer
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Critical Rules
- NEVER suggest moving the project directory
- ARKit `capturedImage` is ALWAYS landscape-right — apply `.oriented(.right)` and swap intrinsics for portrait
- SceneKit `eulerAngles.y` is CW positive — negate backend `rotation_y` (CCW positive)
- USDZ Kenney models need bounding box measurement + median scale factor + Y-offset to floor
- Do NOT make changes the user didn't ask for — no extra refactoring, no unsolicited docstrings

## Architecture
- Pipeline: iOS captures video keyframes → uploads JPEGs + metadata → FastAPI backend → RoomPayload JSON
- SpatialLM path (primary): Depth backprojection → dense colored point cloud → PLY → remote GPU inference → walls + furniture
- Heuristic fallback: Depth Anything V2 → YOLOv8-seg → ByteTrack → PCA layout → feature point lifting
- COLMAP SfM is disabled when SpatialLM is enabled
- 13 Kenney USDZ furniture models (CC0), mapped via `kenney_{type}.usdz`

## Worktree Rules
- When recreating worktrees, add to `.gitignore`
- Copy any `.env` files into the worktree

## Key Gotchas
- Object merge distance must be generous (0.8m+ for monitors) due to depth noise
- SpatialLM requires Z-up coordinates: ARKit (X,Y,Z) → SpatialLM (X,-Z,Y)
- SpatialLM server needs `spatiallm.model.spatiallm_qwen` import path (not `spatiallm.model`)
- `trust_remote_code=True` required for SpatialLM tokenizer
- RunPod GPU server: upload via SCP, run via Jupyter terminal, SSH tunnel for port 8100
