# Plan 3a Manual QA — 2026-04-24

## Context

Pre-merge QA for Plan 3a (phone-first backend integration). Branch:
`feature/phone-first-visual`. Latest commit: `67914b3` (revert of
histogram peaks, so the reconciliation is back to anchor + cloud 8/92
percentile + smaller-wins).

Design spec: `docs/superpowers/specs/2026-04-19-plan3a-phone-first-backend-integration-design.md`
Plan doc: `docs/superpowers/plans/2026-04-19-plan3a-phone-first-backend-integration.md`

## Automated checks

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | Swift full test suite (simulator, iPhone 17 Pro) | **PASS** | Last run `bqg2nl7nw` — 62 passed, 0 failed. UI-test harness warnings pre-existing, not caused by Plan 3a. |
| 2 | Backend Python test suite | **PASS** | 20 passed, 0 failed (after histogram-peaks revert). |
| 3 | Device build (generic/iOS, CODE_SIGNING_ALLOWED=NO) | **PASS** | xcodebuild build succeeds. |

## Happy-path device checklist

Execute on a physical iPhone (XS or newer = `.full` / `.reduced` tier).

- [ ] App launches without crash
- [ ] Scan tab shows only "Video Scan" and "Scan Furniture" (no Photo Capture)
- [ ] Tap "Start Scan" → clean live-camera view (no panorama overlay, no grid)
- [ ] Slowly rotate ~60-90 s covering all 4 corners
- [ ] Coverage arc fills progressively
- [ ] No freeze at any point during the 60 s scan
- [ ] Thermal state stays `.nominal` or `.fair` (check Xcode organizer)
- [ ] Tap Done → app advances to Upload state
- [ ] Tap Upload → progress indicator shows "Uploading N frames..." with correct N
- [ ] Upload completes within 60 s
- [ ] 3D preview renders a closed rectangular room
- [ ] Room dimensions within ±15% of reality
- [ ] `scan_log.json` appears in `~/Documents/scan_logs/` after scan
- [ ] `last_payload/` directory exists in `~/Documents/` after successful upload

## Unsupported device checklist

Execute on iPhone X (A11) or older if available.

- [ ] App launches
- [ ] Scan screen shows "This iPhone doesn't support Decroom's scanning. Minimum: iPhone XS."
- [ ] Scan button is disabled / absent
- [ ] No crash

## Error-path checklist — each deliberately triggered

| Error | How to trigger | Expected alert text | Retry button behaviour |
|---|---|---|---|
| `.depthModelLoadFailed` | Dev build with renamed `DepthModel.mlpackage` | "Can't start scan: the depth engine isn't available. Please reinstall the app." | No retry; scan button disabled next launch |
| `.arKitTrackingLimited` | Cover the camera or move very fast | Yellow banner "Lost tracking — slow down, more light" | Auto-clears when tracking resumes |
| `.noWallsDetected` | Scan a pitch-black room for 20 s then tap Done | "No walls found. Point at the walls of your room and keep scanning." | Resume Scan preserves frames |
| `.insufficientCoverage` | Tap Done after 3 s of scanning | "Only part of the room was captured. Rotate 360° and try again." | Resume Scan preserves frames |
| `.uploadNoServer` | Stop backend, tap Upload | "Can't reach the Decroom server. Check WiFi and tap Retry." | Retry re-attempts upload |
| `.uploadTimeout` | Backend with forced 90 s sleep before response | "Server took too long. Try again in a moment." | Retry re-attempts |
| `.backendSchemaError` | Dev build that corrupts `room_geometry.json` before upload | "Scan data was rejected. Please report this. Scan ID: <id>" | No retry; scan ID shown |
| `.backendServerError` | Backend feature flag returning 500 | "Server error. Try again in a moment." | Retry re-attempts |
| `.noFurnitureDetected` | Scan an empty room / SpatialLM tunnel down | Info toast: "Scan complete — no furniture found." | N/A — scan saves anyway |

## Use Last Capture QA

- [ ] After one successful upload, return to the scan screen idle state
- [ ] "Use Last Capture" button visible alongside "Start Scan"
- [ ] Tap "Use Last Capture"
- [ ] "Uploading N frames..." shows the correct N (matches original scan's keyframe count)
- [ ] Re-upload completes and 3D preview renders
- [ ] Dimensions match the original scan's dimensions (same payload → same result)

## Resume-preserves-frames QA

- [ ] Start a scan
- [ ] Scan ~50% coverage then tap Done
- [ ] When it fails `.insufficientCoverage`, tap "Resume Scan"
- [ ] Recorder picks up where it left off (keyframe count does not reset to 0)
- [ ] Continue scan, tap Done again, upload succeeds

## Real-time no-walls warning

- [ ] Start a scan pointed at a plain ceiling or floor only (no walls in view)
- [ ] After ~15 seconds, orange banner "No walls detected — look at wall edges, slow down" appears
- [ ] Aim at a wall edge; ARKit commits a plane anchor within a few seconds
- [ ] Banner disappears automatically

## Payload fixture

- [ ] Save a real successful scan's payload to `tests/fixtures/phone_first_payload/` for future contract tests. Files:
  - [ ] `room_geometry.json`
  - [ ] `point_cloud.ply`
  - [ ] `keyframes.json`
  - [ ] `frame_00.jpg` through `frame_NN.jpg`
  - [ ] `scan_log.json`

## Known limitations as of this QA

- Dimension accuracy on non-empty rooms: typically 10-20% over real measurements. Both ARKit plane anchors and point cloud percentile-bbox tend to over-extend (extent bleed, through-wall depth). Tuning done to 8/92 percentile + smaller-wins reconciliation; further improvements deferred — see `docs/superpowers/potential_adds.md`.
- Doors and windows not detected. `Wall.has_door` / `has_window` stay `false`. Deferred to Plan 3b.
- Wall-snapping for furniture not applied backend-side. SpatialLM output goes through `SPATIALLM_TO_PRD` mapping only. Deferred to Plan 3b.
- Coverage arc sometimes shows partial fill on full-rotation scans because keyframe capture requires feature points ≥ 5 and blur score ≥ 45. Plain / dimly-lit walls fail these gates.
- `.fallback` tier devices (iPhone X / A11 or older) can't scan at all.

## Observations / follow-ups

(Fill in during QA.)

### Dimension check

Real room: ____ × ____ m

Scan attempt 1:
- Reported dimensions: ____ × ____ m
- Backend log `dimensions ... reconcile` line(s):
  ```

  ```
- Error vs real: width ____%, length ____%

Scan attempt 2 (different room or same room with better technique):
- Reported dimensions: ____ × ____ m
- Error vs real: width ____%, length ____%

### Issues surfaced

- [ ] Issue: ____
  - Repro: ____
  - Proposed fix or follow-up plan: ____

## Sign-off

- Manual QA performed by: ______
- On device: ______ (iPhone model + iOS version)
- Date: 2026-04-24
- Ready to merge to main: ☐ yes ☐ no — with follow-ups
