# Decroom (no LiDAR) — scanning lab

**Status: paused (October 2026).** Product development has moved to the new app, [`jaaej/decroom`](https://github.com/jaaej/decroom), a cosy 3D room designer. This repository is kept as the research lab for one-shot room scanning: one scan of a whole room, without LiDAR, producing a 3D room model with each piece of furniture as a separate object. The goal is to bring that into the new app later.

The frozen state is tagged `scanning-lab-2026-10`.

## What's here

- `decroom_no_lidar/decroom_no_lidar/`: the iOS app (SwiftUI, ARKit, SceneKit). It handles video capture with coverage guidance, upload, the 3D room viewer and the furniture correction UI.
- `decroom_no_lidar/backend/`: the FastAPI backend and the SpatialLM GPU server.
- `docs/superpowers/`: design specs and implementation plans.
- `CLAUDE.md`: build commands, critical rules and gotchas.

## State when paused

- Furniture detection works, but association over-counts: one real object can come out as several.
- `docs/superpowers/specs/2026-07-02-clean-slate-scan-pipeline-design.md` proposes a rebuilt pipeline. It was **never reviewed or approved**, and its spike S1 (instance identity) is the gate.
- Six scanning UIs have been tried and abandoned. See `CLAUDE.md` before attempting another.
- Known security gaps, such as plaintext LAN transport, are recorded in the project's Claude memory notes.
