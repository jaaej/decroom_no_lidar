# Plan 4a — Correction UI Design Spec

**Date**: 2026-05-13
**Author**: jaejeon (with Claude)
**Status**: Design — awaiting implementation plan
**Predecessors**:
- Plan 3b: semantic polish — `docs/superpowers/specs/2026-04-24-plan3b-semantic-polish-design.md`
- Plan 3c: backend RGB detection — `docs/superpowers/specs/2026-05-03-plan3c-backend-rgb-detection-design.md`
- Plan 3d: clustering retune (committed `bd9e446`)

## Context

After Plan 3c + 3d the pipeline produces a RoomPayload that's categorically complete (bed, wardrobe, bookshelf, desk, nightstand all present where they were previously absent) but quantitatively noisy: a real ~7-piece bedroom currently renders as ~25 objects, with 3-5× duplicates in chair / lamp / monitor / wardrobe and occasional category false-positives (e.g., "sofa" in a bedroom).

Pipeline-side improvements (Plan 3e, Plan 5, Plan 6) are queued in `memory/plan4a_deferred_issues.md` and will continue reducing the over-count. But the long-tail of detection errors will never be zero, and users need a way to fix the scene after the fact.

Plan 4a adds that capability: an in-scene editing UI that lets users **delete**, **move**, and **rotate** the detected furniture in the rendered 3D room. No new capture flow, no new ML models, no changes to the backend pipeline.

## Goals

- Let users delete spurious / duplicate objects directly from the rendered 3D scene.
- Let users move and rotate misplaced objects to better match reality.
- Provide single-step undo to recover from accidental actions.
- Keep the interaction model mobile-friendly: large tap targets, discoverable verbs, no hidden gestures.

## Non-Goals

- **No wall editing.** Marking door/window cutouts is deferred to Plan 4c.
- **No category change.** Reclassifying an object (e.g., "this is a desk not a wardrobe") is deferred — covered by Plan 4a-v2 if needed; v1 ships without it.
- **No persistent corrections.** Edits live in memory only. Closing the app or scanning a new room resets to the freshly-detected RoomPayload. Persistent edits are Plan 4d if it becomes a real pain.
- **No furniture library / manual placement.** That's Plan 4b — add furniture from a catalog. Plan 4a only edits what was detected.
- **No multi-step undo or redo.** Single previous-state snapshot covers ~90% of accidental actions. Adding history is Plan 4d if needed.
- **No backend changes.** All work is iOS-side. The RoomPayload schema is unchanged.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Existing: RoomSceneView (renders walls + furniture)        │
│                                                              │
│  Adds:                                                       │
│   • UITapGestureRecognizer → notifies edit model            │
│   • Selection highlight on SCNNode (emissive outline)       │
│   • Drag gesture (active only when editModel.activeAction   │
│     == .move)                                                │
│   • Two-finger rotation gesture (active only when           │
│     editModel.activeAction == .rotate)                      │
│   • Disables SCNView.allowsCameraControl while activeAction │
│     is set, re-enables when nil                             │
└─────────────────────────────────────────────────────────────┘
              │                       ▲
              │ tap → object ID       │ binds to roomPayload,
              ▼                       │ selectedObjectID,
                                      │ activeAction, canUndo
┌─────────────────────────────────────────────────────────────┐
│  New: RoomEditModel (ObservableObject)                      │
│                                                              │
│  Single source of truth for editing state.                  │
│                                                              │
│   @Published roomPayload: RoomPayload                       │
│   @Published selectedObjectID: String?                      │
│   @Published activeAction: EditAction?  // .move | .rotate  │
│   @Published canUndo: Bool                                  │
│   private var previousObjects: [RoomObject]?                │
│                                                              │
│   select(_:) / delete(_:) / move(_:to:gestureBegan:) /      │
│   rotate(_:toRadians:gestureBegan:) / undo()                │
└─────────────────────────────────────────────────────────────┘
              ▲
              │ @ObservedObject
              │
┌─────────────────────────────────────────────────────────────┐
│  New: RoomEditMenuView (SwiftUI overlay)                    │
│                                                              │
│   Bottom sheet (when selectedObjectID != nil):              │
│     [Delete] [Move] [Rotate] [Done]                         │
│   Toolbar Undo button (when canUndo)                        │
└─────────────────────────────────────────────────────────────┘
```

Single source of truth for selection + mutations + undo is `RoomEditModel`. `RoomSceneView` reports taps and applies visual feedback. `RoomEditMenuView` is pure SwiftUI bound to the model. SwiftUI's `@ObservedObject` triggers re-render whenever the model publishes.

## Components

### 1. New: `RoomEditModel.swift` (~120 lines)

```swift
import Foundation
import simd

enum EditAction {
    case move
    case rotate
}

@MainActor
final class RoomEditModel: ObservableObject {
    @Published private(set) var roomPayload: RoomPayload
    @Published var selectedObjectID: String?
    @Published var activeAction: EditAction?
    @Published private(set) var canUndo: Bool = false

    private var previousObjects: [RoomObject]?

    init(initial: RoomPayload) { self.roomPayload = initial }

    func select(_ id: String?)
    func delete(_ id: String)
    func move(_ id: String, to xz: SIMD2<Float>, gestureBegan: Bool)
    func rotate(_ id: String, toRadians y: Float, gestureBegan: Bool)
    func undo()
}
```

Selection setter validates against the current `roomPayload.objects` and clears `selectedObjectID` if the ID is stale (race-after-delete safety).

All mutations snapshot `previousObjects` exactly once per logical user action:
- `delete` snapshots before removing.
- `move` snapshots only when `gestureBegan == true` (first tick of a drag); subsequent ticks of the same drag do not re-snapshot.
- `rotate` same as move.

`undo()` restores `previousObjects` into `roomPayload.objects`, clears the snapshot, clears selection, sets `canUndo = false`. No-op if `previousObjects == nil`.

### 2. New: `RoomEditMenuView.swift` (~80 lines)

```swift
struct RoomEditMenuView: View {
    @ObservedObject var editModel: RoomEditModel

    var body: some View {
        ZStack {
            // Top-trailing: Undo button visible when editModel.canUndo
            // Bottom: action sheet visible when editModel.selectedObjectID != nil
        }
    }
}
```

Action sheet UI:
- Default state (selectedObjectID set, no activeAction): horizontal row `[Delete] [Move] [Rotate] [Done]`.
- `.move` active: hint text "Drag to reposition. Tap Done to finish." with [Done] button.
- `.rotate` active: hint text "Twist with two fingers. Tap Done to finish." with [Done] button.

Tap mappings:
- `[Delete]` → `editModel.delete(selectedID)` (clears selection automatically).
- `[Move]` → `editModel.activeAction = .move`.
- `[Rotate]` → `editModel.activeAction = .rotate`.
- `[Done]` → context-dependent:
  - if `activeAction != nil`: `editModel.activeAction = nil` (end the move/rotate, return to default state, keep selection).
  - if `activeAction == nil`: `editModel.select(nil)` (close menu, deselect).
- Outside the sheet tap → `editModel.select(nil)` (handled by `RoomSceneView`).

### 3. New: `SceneSelection.swift` (~50 lines)

Utility module isolating the SceneKit-specific concerns from view code.

```swift
enum SceneSelection {
    /// Returns the RoomObject.id of the topmost furniture node hit by the
    /// tap, or nil. Uses a 22pt-radius retry pattern for small objects.
    static func hitTest(_ scnView: SCNView, at point: CGPoint) -> String?

    /// Applies or removes the "selected" emissive outline to the SCNNode
    /// owning that RoomObject.id.
    static func setSelected(_ id: String?, in rootNode: SCNNode)

    /// Project a screen point onto the floor plane (y=0) in world coords.
    /// Returns nil if the ray is parallel to the floor or points away.
    static func floorPoint(forScreen point: CGPoint, in scnView: SCNView) -> SIMD2<Float>?
}
```

Selection highlight: thin emissive cyan outline on the selected USDZ. Achieved by setting `material.emission.contents = UIColor.cyan` on a duplicated outer node of the SCNNode, rendered with `material.lightingModel = .constant` so it ignores scene lighting.

### 4. Modified: `RoomSceneView.swift`

Three additions to the existing 697-line file:

1. **Hold `@ObservedObject var editModel: RoomEditModel`** as an injected dependency.
2. **Tap recognizer** on the SCNView. Calls `SceneSelection.hitTest` → `editModel.select(id)`. Empty-space tap → `editModel.select(nil)`.
3. **Reactive gesture installation** keyed off `editModel.activeAction`:
   - `.move`: install `UIPanGestureRecognizer`. On each update, raycast finger position to the floor plane via `SceneSelection.floorPoint`. For wall-snap categories (bed/wardrobe/dresser/bookshelf/sofa) clamp to nearest wall. For freestanding categories (chair/table/lamp/plant) use raw raycast point. Call `editModel.move(id, to: xz, gestureBegan: (state == .began))`.
   - `.rotate`: install `UIRotationGestureRecognizer`. On each update, snap gesture rotation to nearest 90° increment. Call `editModel.rotate(id, toRadians: snapped, gestureBegan: (state == .began))`.
   - `nil`: remove pan/rotation recognizers, re-enable `scnView.allowsCameraControl`.

When an action is active, `scnView.allowsCameraControl = false` to prevent SceneKit's built-in camera gestures from consuming the same multi-touch events.

The render code reacts to `editModel.roomPayload` changes via SwiftUI's `@ObservedObject` mechanism — no manual scene-graph diffing.

## Data flow

### A. Delete

1. User taps wardrobe → `SceneSelection.hitTest` → `"obj_wardrobe_3"` → `editModel.select(...)`.
2. `selectedObjectID` published; `RoomEditMenuView` shows action sheet; `RoomSceneView` renders highlight.
3. User taps `[Delete]` → `editModel.delete("obj_wardrobe_3")`:
   - Snapshot `previousObjects`.
   - Remove from `roomPayload.objects`.
   - Clear `selectedObjectID`.
   - Set `canUndo = true`.
4. SwiftUI re-renders: wardrobe gone from scene, sheet hidden, Undo button appears.

### B. Move

1. User taps chair → menu appears → taps `[Move]` → `editModel.activeAction = .move`.
2. `RoomSceneView` reacts: disables camera control, installs pan recognizer, sheet shrinks to hint.
3. User drags. First gesture tick (`.began`):
   - `editModel.move(id, to: xz, gestureBegan: true)` → snapshot taken once.
4. Subsequent gesture ticks: `gestureBegan: false` → no re-snapshot, just apply position.
5. User taps `[Done]` → `editModel.activeAction = nil` → pan recognizer removed, camera re-enabled.

For wall-snap categories, the XZ position passed in is the snapped wall position. The `editModel.move` itself is category-agnostic — the snap is computed in the gesture handler in `RoomSceneView`.

### C. Rotate

Same as Move but with rotation gesture. Snap-to-90° happens in the gesture handler. The model sees only the final snapped angle.

### D. Undo

Tap `[Undo]` (only visible when `canUndo == true`):
1. `roomPayload.objects` restored from `previousObjects`.
2. `previousObjects = nil`.
3. `canUndo = false`.
4. `selectedObjectID = nil`.
5. Scene re-renders, Undo button hides.

## Gesture-handling details

### Camera vs object gesture conflict

`SCNView.allowsCameraControl` defaults to true and claims pan/pinch/rotation. Solution:

```swift
editModel.$activeAction.sink { action in
    scnView.allowsCameraControl = (action == nil)
}
```

Single-tap selection works in both states — taps don't conflict with orbit-camera.

### Tap accuracy on small objects

Far-away objects (lamp at 4m, 30px on screen) miss Apple's 44pt minimum touch target. `SceneSelection.hitTest` does a center hit first, and if empty retries at 8 offset points within a 22pt radius:

```swift
let offsets: [(Int, Int)] = [(11,0), (-11,0), (0,11), (0,-11),
                              (8,8), (-8,8), (8,-8), (-8,-8)]
```

Returns the first non-empty hit. If all miss → nil → user gets empty-space behavior (clear selection).

### Move drag → floor raycast

`SCNHitTest` returns objects; we want the floor (y=0 plane). Manual unproject + ray-plane intersection in `SceneSelection.floorPoint`:

```swift
let near = scnView.unprojectPoint(SCNVector3(x, y, 0))
let far  = scnView.unprojectPoint(SCNVector3(x, y, 1))
let dir  = far - near
// y=0 plane intersection
let t = -near.y / dir.y
let world = near + dir * t
return SIMD2(world.x, world.z)
```

Returns nil if `dir.y` is ~0 (ray parallel to floor) or `t < 0` (intersection behind camera). Caller ignores nil ticks.

For wall-snap categories, the gesture handler then clamps the floor point to the nearest wall using the same per-category rules from Plan 3b's `snap_to_walls`. The Swift port lives in the gesture handler (or in a shared util).

If the resulting position is outside the room footprint, clamp to the nearest footprint edge — don't reject. "Drag ignored" feels broken. The footprint is assumed rectangular (4 walls forming a closed rectangle, which is what `RoomGeometryBuilder` always produces today); clamp = independently bound x to `[x_min, x_max]` and z to `[z_min, z_max]`. If non-rectangular footprints land later (e.g., L-shaped rooms via Plan 5/LiDAR), this needs to become a polygon clamp.

### Rotation snap

Snap **live** during the gesture, not just on `.ended`, so the bed visibly rotates in 90° clicks rather than spinning continuously then jumping. Snap math:

```swift
let snapped = round(gr.rotation / (.pi / 2)) * (.pi / 2)
```

After `.ended`, reset `gr.rotation = 0` so the next gesture starts from zero delta.

Note the SceneKit/backend rotation convention from CLAUDE.md: SceneKit `eulerAngles.y` is CW positive, backend `rotation_y` is CCW positive. The model stores backend-convention; rendering code negates when applying to SCNNode (existing behavior).

## Error handling

| Failure | Behavior |
|---|---|
| Tap hits no object | No selection; menu stays hidden |
| Tap empty space while selection is active | Clear selection; menu hides |
| Move drag's floor raycast returns nil (off-camera) | Ignore that tick; no mutation |
| User drags object past room footprint | Clamp to nearest footprint edge |
| Rotation gesture `.cancelled` (e.g., phone call) | Treat as `.ended` with no commit; snapshot already taken on `.began` so undo recovers initial state |
| Selected object ID no longer exists (race after delete) | Selection setter validates; sets to nil silently |
| Undo with `previousObjects == nil` | Early return; no-op |
| App backgrounded or killed | All Plan 4a state in-memory; reopening returns to original detected RoomPayload. Intentional v1 scope |

No alerts, no toasts, no error sheets. Fail silently, fall back to a clean state.

## Testing

### Unit tests — `RoomEditModelTests.swift` (~14 tests, XCTest, no SceneKit)

| Test | What it verifies |
|---|---|
| `test_select_setsSelectedID` | select(id) updates `selectedObjectID` |
| `test_select_nil_clearsSelection` | select(nil) clears |
| `test_select_cancelsActiveAction` | Changing selection clears `activeAction` |
| `test_select_invalidID_setsNil` | Selection setter rejects IDs not in `roomPayload.objects` |
| `test_delete_removesFromObjects` | After delete, objects list no longer contains that id |
| `test_delete_snapshotsPreviousObjects` | `previousObjects` captures pre-delete state |
| `test_delete_clearsSelection` | After delete, `selectedObjectID == nil` |
| `test_delete_setsCanUndo` | `canUndo == true` post-delete |
| `test_move_began_snapshots` | First call with `gestureBegan: true` snapshots |
| `test_move_subsequentUpdates_dontReSnapshot` | Calls with `gestureBegan: false` leave snapshot unchanged |
| `test_move_appliesXZ_preservesY` | Object's `position.y` is preserved across move |
| `test_rotate_began_snapshots` | Same as move but rotation |
| `test_rotate_appliesRadians` | Object's `rotation_y` updates to passed value |
| `test_undo_restoresPreviousObjects` | After mutation + undo, objects match pre-mutation state |
| `test_undo_clearsCanUndo` | `canUndo == false` after undo |
| `test_undo_clearsSelection` | `selectedObjectID == nil` after undo |
| `test_undo_noOpWhenNothingToUndo` | undo with `previousObjects == nil` doesn't crash |

### Utility tests — `SceneSelectionTests.swift` (~5 tests)

| Test | Verifies |
|---|---|
| `test_floorPoint_centerScreen_returnsCorrectWorld` | Unproject math correct for known camera + point |
| `test_floorPoint_rayParallelToFloor_returnsNil` | Ray with `dir.y ≈ 0` returns nil |
| `test_floorPoint_rayBehindCamera_returnsNil` | `t < 0` returns nil |
| `test_wallSnap_bedCategory_snapsToNearestWall` | Wall-snap clamp logic ported from Plan 3b |
| `test_wallSnap_chairCategory_doesNotSnap` | Freestanding categories pass through |
| `test_footprintClamp_outsideRoom_clampsToEdge` | Position outside 4-wall footprint clamps to nearest edge |

### Integration — `RoomSceneViewEditTests.swift` (XCUITest, 1 happy-path)

| Test | Verifies |
|---|---|
| `test_tapDeleteRemovesObjectFromScene` | Launch with stub RoomPayload → tap chair → tap Delete → assert chair gone from scene-graph |

SceneKit's hit-test, gesture recognizers, and camera control are well-trodden Apple code — we don't test those. We test that we wire them up correctly.

### Manual QA on device

Documented in `docs/superpowers/qa/<YYYY-MM-DD>-plan4a-manual-qa.md` after implementation lands. Checklist:

- Tap accuracy on small objects (lamps, monitors at depth)
- Gesture conflict with camera (camera should be inert during move/rotate)
- Drag-to-wall snap feel (bed snaps cleanly to nearest wall)
- Rotation snap feel (90° clicks visible during twist)
- Undo behavior on delete / move / rotate
- Empty-space tap clears selection
- Selection highlight visible against the rendered USDZ

## File structure

### New files

| Path | Responsibility |
|---|---|
| `decroom_no_lidar/decroom_no_lidar/RoomEditModel.swift` | Selection + mutation + undo state. |
| `decroom_no_lidar/decroom_no_lidar/RoomEditMenuView.swift` | Bottom sheet + Undo button SwiftUI overlay. |
| `decroom_no_lidar/decroom_no_lidar/SceneSelection.swift` | SceneKit-specific hit-test, floor raycast, selection-highlight helpers. |
| `decroom_no_lidar/decroom_no_lidarTests/RoomEditModelTests.swift` | Model unit tests. |
| `decroom_no_lidar/decroom_no_lidarTests/SceneSelectionTests.swift` | Utility unit tests. |
| `decroom_no_lidar/decroom_no_lidarUITests/RoomSceneViewEditTests.swift` | One end-to-end XCUITest. |
| `docs/superpowers/qa/<YYYY-MM-DD>-plan4a-manual-qa.md` | Post-implementation manual QA results. |

### Modified files

| Path | Change |
|---|---|
| `decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift` | Accept `@ObservedObject editModel`; add tap recognizer; reactive pan/rotation recognizer install; disable camera control when active action set. Approx +200 LoC. |
| The view that hosts `RoomSceneView` (likely `HomeBlockListView.swift` or `ContentView.swift`) | Construct and own `RoomEditModel`; overlay `RoomEditMenuView` on top of `RoomSceneView`. |

### Untouched

- All backend code (Python pipeline).
- All capture-side iOS code (`ScanSession`, `VideoRecorderView`, `UploadService`).
- RoomPayload schema (no new fields).
- USDZ assets (no new models).

## Migration plan

### Phase A — Model layer (TDD)

1. Create `RoomEditModel.swift` with full API, all mutations stubbed.
2. Write `RoomEditModelTests.swift` (all 14 tests fail).
3. Implement mutations + undo to make tests pass.
4. Commit.

### Phase B — Utility layer

5. Create `SceneSelection.swift` with hit-test, floor raycast, wall-snap, footprint-clamp helpers.
6. Write `SceneSelectionTests.swift` (all 5 tests fail).
7. Implement helpers to make tests pass.
8. Commit.

### Phase C — View layer

9. Add `@ObservedObject editModel: RoomEditModel` injection point to `RoomSceneView`.
10. Wire tap recognizer → `editModel.select(...)`.
11. Add reactive gesture-install observer for `activeAction`.
12. Add selection-highlight rendering hook.
13. Create `RoomEditMenuView.swift`.
14. Mount the overlay in the hosting view alongside `RoomSceneView`.
15. Commit.

### Phase D — Integration test + manual QA

16. Write `RoomSceneViewEditTests.swift` happy-path XCUITest.
17. Build, deploy, do manual QA on device. Document in QA doc.
18. Commit QA doc.

### Phase E — Merge

19. Merge to main.

## Risks and open questions

**Risk: SceneKit camera control fights with our recognizers despite the `allowsCameraControl` toggle.** SceneKit's gesture recognizers can persist beyond the property toggle in some iOS versions. Mitigation: detect during manual QA; fall back to explicit recognizer-removal via `scnView.gestureRecognizers?.forEach { ... }` if needed.

**Risk: 90° rotation snap feels jerky / unnatural on touch.** Some users may expect free rotation. Mitigation: ship 90° default per the brainstorming decision, watch QA feedback, add long-press → free-rotate in v2 if a real pain point.

**Risk: Wall-snap during drag feels "sticky" near corners.** Bed dragged near a corner could snap-flicker between two walls. Mitigation: hysteresis in the snap selection — once snapped to wall A, require moving > 0.3m further toward wall B before switching.

**Risk: Selection highlight invisible against bright USDZ models.** Emissive cyan may wash out against light wood textures. Mitigation: test against the actual Kenney USDZ bundle during Phase C; pick a color/style that works on all 13 models.

**Open question: Should undo persist after the user navigates away from the scene and back?** Current spec: no — the model lives only as long as the view does. Simpler. If users want session-long undo, surface it as Plan 4d.

## How a fresh conversation should pick this up

Read in this order:
1. This spec.
2. `docs/superpowers/specs/2026-05-03-plan3c-backend-rgb-detection-design.md` — context for the over-count Plan 4a addresses.
3. `decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift` — the file being modified.
4. `memory/visual_ui_failures.md` — what previous UI attempts ran into (mostly capture-time UI; not directly applicable but worth knowing).
5. `memory/plan4a_deferred_issues.md` — the pipeline-quality issues parked for post-Plan-4a.

Then invoke `/superpowers:writing-plans` against this spec to produce the task-by-task implementation plan.
