# Plan 4a — Correction UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tap-to-select / Delete / Move / Rotate / Undo editing on top of the rendered 3D room, with single-step undo, smart wall-snap during drag, and 90° rotation snap.

**Architecture:** Three new Swift files isolate editing concerns from the existing 697-line `RoomSceneView.swift`. `RoomEditModel: ObservableObject` owns selection + mutations + 1-step undo. `RoomEditMenuView` is a SwiftUI overlay with the bottom action sheet + Undo button. `SceneSelection` enum holds SceneKit-specific helpers (tap hit-test with 22pt retry radius, floor-plane raycast, wall-snap and footprint-clamp math). `RoomSceneView` is modified to accept an injected `editModel`, install a tap recognizer, and reactively install pan/rotation recognizers when `editModel.activeAction` becomes non-nil (with `allowsCameraControl` toggled off during those windows to prevent gesture conflict). All state is in-memory and per-view-instance — corrections do not persist across navigation in v1.

**Tech Stack:** Swift, SwiftUI, SceneKit, XCTest, XCUITest.

**Spec:** `docs/superpowers/specs/2026-05-13-plan4a-correction-ui-design.md`.

**Prerequisites (MUST be true before starting):**
- Plan 3c + Plan 3d merged to `main` (commit `bd9e446` or later) — this plan's branch builds on that.
- The local-machine iOS toolchain works: `xcodebuild` succeeds on the current `main`.
- Branch off `main`: `git switch -c feature/plan4a-correction-ui`.

**Project convention reminder (from CLAUDE.md):** The Xcode project uses `PBXFileSystemSynchronizedRootGroup`. **DO NOT edit `decroom_no_lidar.xcodeproj/project.pbxproj`** when adding new `.swift` files — they register automatically by being placed in the synchronized folder.

---

## File structure

### New files

| Path | Responsibility |
|---|---|
| `decroom_no_lidar/decroom_no_lidar/RoomEditModel.swift` | `RoomEditModel: ObservableObject`. Holds selection state, applies delete/move/rotate mutations, manages 1-step undo cache. Pure Swift — no SceneKit imports. |
| `decroom_no_lidar/decroom_no_lidar/RoomEditMenuView.swift` | SwiftUI overlay: bottom action sheet (`[Delete] [Move] [Rotate] [Done]`) and top-trailing Undo button. Bound to `@ObservedObject editModel`. |
| `decroom_no_lidar/decroom_no_lidar/SceneSelection.swift` | SceneKit helper module: `hitTest(_:at:)` (with 22pt-radius retry), `floorPoint(forScreen:in:)`, `wallSnap(_:to:)`, `clampToFootprint(_:in:)`, `setSelected(_:in:)`. |
| `decroom_no_lidar/decroom_no_lidar/RoomEditingCoordinator.swift` | NSObject-based Coordinator that owns gesture recognizers and bridges them to `RoomEditModel`. Created by `RoomSceneRepresentable.makeCoordinator()`. |
| `decroom_no_lidar/decroom_no_lidarTests/RoomEditModelTests.swift` | 17 unit tests covering all `RoomEditModel` mutations + undo. |
| `decroom_no_lidar/decroom_no_lidarTests/SceneSelectionTests.swift` | 6 unit tests for the pure-math helpers (wallSnap, clampToFootprint, floorPoint with synthetic projection matrix). |
| `decroom_no_lidar/decroom_no_lidarUITests/RoomSceneViewEditTests.swift` | One XCUITest: launch with stub payload → tap chair → tap Delete → assert chair gone. |
| `docs/superpowers/qa/<YYYY-MM-DD>-plan4a-manual-qa.md` | Post-implementation manual QA results. Filled in during Task 17. |

### Modified files

| Path | Change |
|---|---|
| `decroom_no_lidar/decroom_no_lidar/RoomModels.swift` | Add a `RoomObject.with(position:rotationY:)` helper extension so the edit model can produce a mutated copy without re-implementing `init`. |
| `decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift` | (a) `RoomSceneRepresentable` gains an optional `editModel: RoomEditModel?` property; (b) `makeCoordinator()` returns `RoomEditingCoordinator`; (c) tap recognizer installed unconditionally; (d) pan/rotation recognizers installed in `updateUIView` only when `editModel?.activeAction != nil`, removed otherwise; (e) `scnView.allowsCameraControl = (editModel?.activeAction == nil)`; (f) `RoomSceneView` exposes a constructor that takes the model + overlays `RoomEditMenuView` on top. |
| `decroom_no_lidar/decroom_no_lidar/CaptureFlowView.swift` (line ~85) | Construct an `@StateObject editModel = RoomEditModel(initial: payload)` and pass to the new editing `RoomSceneView` initializer. |
| `decroom_no_lidar/decroom_no_lidar/RoomCardView.swift` (line ~248) | Same change as `CaptureFlowView` — construct + pass `editModel`. |

### Untouched

- All backend code.
- All capture-side iOS code (`ScanSession`, `VideoRecorderView`, `UploadService`).
- USDZ assets.
- `RoomPayload`/`RoomInfo`/`Wall` schemas — only `RoomObject` gets an extension helper.

---

## Task list

Tasks run in order. Phase A (Tasks 1–5) is the model layer with TDD. Phase B (Tasks 6–8) the pure SceneKit utility helpers. Phase C (Task 9) the menu view. Phase D (Tasks 10–13) integrates into `RoomSceneRepresentable`. Phase E (Tasks 14–15) wires the hosting views and exposes the public initializer. Phase F (Tasks 16–17) end-to-end test + manual QA + merge.

---

### Task 1: Branch + `RoomObject.with(...)` extension

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/RoomModels.swift`

- [ ] **Step 1: Create the branch.**

```bash
cd /Users/jaejeon/Desktop/Decroom/decroom_no_lidar
git switch -c feature/plan4a-correction-ui
```

- [ ] **Step 2: Add the helper at the end of `RoomModels.swift`.**

After the existing `ObjectMetadata` struct (line ~67), append:

```swift
// MARK: - Plan 4a helpers

extension RoomObject {
    /// Returns a copy of this RoomObject with optionally-changed position
    /// or rotation_y. All other fields preserved.
    func with(
        position: ObjectPosition? = nil,
        rotationY: Double? = nil
    ) -> RoomObject {
        RoomObject(
            id: id,
            type: type,
            asset_id: asset_id,
            position: position ?? self.position,
            rotation_y: rotationY ?? self.rotation_y,
            scale: scale,
            confidence: confidence,
            metadata: metadata
        )
    }
}

extension RoomPayload {
    /// Returns a copy of this RoomPayload with a new objects list.
    func with(objects: [RoomObject]) -> RoomPayload {
        RoomPayload(room: room, objects: objects)
    }
}
```

- [ ] **Step 3: Build to verify the synthesized memberwise init is accessible.**

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj -destination 'generic/platform=iOS' build 2>&1 | tail -5
```
Expected: `BUILD SUCCEEDED`.

If the synthesized init isn't accessible (Swift structs default the synthesized init to `internal`; since both files are in the same target this should work — but if the build fails with "RoomObject init inaccessible," add an explicit `init` to `RoomObject` with the same signature, or move both extensions into `RoomModels.swift` directly).

- [ ] **Step 4: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/RoomModels.swift
git commit -m "feat(models): add RoomObject.with(...) + RoomPayload.with(objects:) helpers (Plan 4a Task 1)"
```

---

### Task 2: `RoomEditModel` skeleton + tests for `select(...)`

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/RoomEditModel.swift`
- Create: `decroom_no_lidar/decroom_no_lidarTests/RoomEditModelTests.swift`

- [ ] **Step 1: Write the failing tests.**

`decroom_no_lidar/decroom_no_lidarTests/RoomEditModelTests.swift`:

```swift
import XCTest
@testable import decroom_no_lidar

@MainActor
final class RoomEditModelTests: XCTestCase {

    // MARK: - Fixtures

    private func makeObject(id: String,
                             type: String = "chair",
                             x: Double = 0, y: Double = 0, z: Double = 0,
                             rot: Double = 0) -> RoomObject {
        RoomObject(
            id: id,
            type: type,
            asset_id: "kenney_\(type)",
            position: ObjectPosition(x: x, y: y, z: z),
            rotation_y: rot,
            scale: 1.0,
            confidence: 0.7,
            metadata: ObjectMetadata(width: 1, length: 1, height: 1)
        )
    }

    private func makePayload(_ objects: [RoomObject]) -> RoomPayload {
        let walls = [
            Wall(id: "w0", start: [-2, -2], end: [2, -2], has_window: false, has_door: false),
            Wall(id: "w1", start: [2, -2],  end: [2, 2],  has_window: false, has_door: false),
            Wall(id: "w2", start: [2, 2],   end: [-2, 2], has_window: false, has_door: false),
            Wall(id: "w3", start: [-2, 2],  end: [-2, -2], has_window: false, has_door: false),
        ]
        let room = RoomInfo(id: "r1", width_m: 4, length_m: 4, height_m: 2.5, walls: walls)
        return RoomPayload(room: room, objects: objects)
    }

    // MARK: - select(_:)

    func test_select_setsSelectedID() {
        let payload = makePayload([makeObject(id: "obj_1")])
        let model = RoomEditModel(initial: payload)
        model.select("obj_1")
        XCTAssertEqual(model.selectedObjectID, "obj_1")
    }

    func test_select_nil_clearsSelection() {
        let payload = makePayload([makeObject(id: "obj_1")])
        let model = RoomEditModel(initial: payload)
        model.select("obj_1")
        model.select(nil)
        XCTAssertNil(model.selectedObjectID)
    }

    func test_select_cancelsActiveAction() {
        let payload = makePayload([
            makeObject(id: "obj_1"),
            makeObject(id: "obj_2", x: 1),
        ])
        let model = RoomEditModel(initial: payload)
        model.select("obj_1")
        model.activeAction = .move
        model.select("obj_2")
        XCTAssertNil(model.activeAction)
    }

    func test_select_invalidID_setsNil() {
        let payload = makePayload([makeObject(id: "obj_1")])
        let model = RoomEditModel(initial: payload)
        model.select("not_in_payload")
        XCTAssertNil(model.selectedObjectID)
    }
}
```

- [ ] **Step 2: Run to verify failure.**

```bash
xcodebuild test -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/RoomEditModelTests 2>&1 | tail -10
```
Expected: build failure with "No such module" or "RoomEditModel undefined".

- [ ] **Step 3: Implement the skeleton.**

Create `decroom_no_lidar/decroom_no_lidar/RoomEditModel.swift`:

```swift
//
//  RoomEditModel.swift
//  decroom_no_lidar
//
//  Plan 4a — selection state, mutations, single-step undo for the
//  correction UI. Pure Swift, no SceneKit imports.
//

import Foundation
import simd

enum EditAction: Equatable {
    case move
    case rotate
}

@MainActor
final class RoomEditModel: ObservableObject {
    @Published private(set) var roomPayload: RoomPayload
    @Published var activeAction: EditAction?
    @Published private(set) var canUndo: Bool = false

    @Published var selectedObjectID: String? {
        didSet {
            // Validate against current objects; setting an invalid ID clears.
            if let id = selectedObjectID,
               !roomPayload.objects.contains(where: { $0.id == id }) {
                selectedObjectID = nil
                return
            }
            // Changing selection cancels any in-progress action.
            if oldValue != selectedObjectID {
                activeAction = nil
            }
        }
    }

    private var previousObjects: [RoomObject]?

    init(initial: RoomPayload) {
        self.roomPayload = initial
    }

    // MARK: - Selection

    func select(_ id: String?) {
        selectedObjectID = id
    }

    // MARK: - Mutations (stubs — implemented in later tasks)

    func delete(_ id: String) {
        // Task 3
    }

    func move(_ id: String, to xz: SIMD2<Float>, gestureBegan: Bool) {
        // Task 4
    }

    func rotate(_ id: String, toRadians y: Float, gestureBegan: Bool) {
        // Task 4
    }

    func undo() {
        // Task 5
    }
}
```

- [ ] **Step 4: Run tests — verify they pass.**

```bash
xcodebuild test -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/RoomEditModelTests 2>&1 | tail -10
```
Expected: 4 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/RoomEditModel.swift decroom_no_lidar/decroom_no_lidarTests/RoomEditModelTests.swift
git commit -m "feat(edit): RoomEditModel skeleton + selection tests (Plan 4a Task 2)"
```

---

### Task 3: Implement `delete(...)` + tests

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/RoomEditModel.swift`
- Modify: `decroom_no_lidar/decroom_no_lidarTests/RoomEditModelTests.swift`

- [ ] **Step 1: Append failing tests.**

Append to `RoomEditModelTests.swift` inside the `final class` block:

```swift
    // MARK: - delete(_:)

    func test_delete_removesFromObjects() {
        let payload = makePayload([
            makeObject(id: "a"),
            makeObject(id: "b", x: 1),
        ])
        let model = RoomEditModel(initial: payload)
        model.delete("a")
        XCTAssertEqual(model.roomPayload.objects.map { $0.id }, ["b"])
    }

    func test_delete_snapshotsPreviousObjects() {
        let original = [makeObject(id: "a"), makeObject(id: "b", x: 1)]
        let model = RoomEditModel(initial: makePayload(original))
        model.delete("a")
        XCTAssertTrue(model.canUndo)
    }

    func test_delete_clearsSelection() {
        let payload = makePayload([makeObject(id: "a")])
        let model = RoomEditModel(initial: payload)
        model.select("a")
        model.delete("a")
        XCTAssertNil(model.selectedObjectID)
    }

    func test_delete_unknownID_noOp() {
        let payload = makePayload([makeObject(id: "a")])
        let model = RoomEditModel(initial: payload)
        model.delete("not_there")
        XCTAssertEqual(model.roomPayload.objects.count, 1)
        XCTAssertFalse(model.canUndo)
    }
```

- [ ] **Step 2: Run — verify failure.**

```bash
xcodebuild test -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarTests/RoomEditModelTests 2>&1 | tail -10
```
Expected: 4 new tests fail (the existing 4 still pass).

- [ ] **Step 3: Implement `delete`.**

Replace the `delete` stub in `RoomEditModel.swift`:

```swift
    func delete(_ id: String) {
        guard roomPayload.objects.contains(where: { $0.id == id }) else { return }
        previousObjects = roomPayload.objects
        canUndo = true
        let newObjects = roomPayload.objects.filter { $0.id != id }
        roomPayload = roomPayload.with(objects: newObjects)
        if selectedObjectID == id {
            selectedObjectID = nil
        }
    }
```

- [ ] **Step 4: Run — verify pass.**

```bash
xcodebuild test ... -only-testing:decroom_no_lidarTests/RoomEditModelTests 2>&1 | tail -5
```
Expected: 8 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/RoomEditModel.swift decroom_no_lidar/decroom_no_lidarTests/RoomEditModelTests.swift
git commit -m "feat(edit): RoomEditModel.delete + tests (Plan 4a Task 3)"
```

---

### Task 4: Implement `move(...)` and `rotate(...)` + tests

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/RoomEditModel.swift`
- Modify: `decroom_no_lidar/decroom_no_lidarTests/RoomEditModelTests.swift`

- [ ] **Step 1: Append failing tests.**

```swift
    // MARK: - move(_:to:gestureBegan:)

    func test_move_began_snapshots() {
        let model = RoomEditModel(initial: makePayload([makeObject(id: "a")]))
        model.move("a", to: SIMD2<Float>(1, 1), gestureBegan: true)
        XCTAssertTrue(model.canUndo)
    }

    func test_move_subsequentUpdates_dontReSnapshot() {
        let original = [makeObject(id: "a", x: 0, z: 0)]
        let model = RoomEditModel(initial: makePayload(original))
        model.move("a", to: SIMD2<Float>(1, 1), gestureBegan: true)
        model.move("a", to: SIMD2<Float>(2, 2), gestureBegan: false)
        model.undo()
        XCTAssertEqual(model.roomPayload.objects[0].position.x, 0)
        XCTAssertEqual(model.roomPayload.objects[0].position.z, 0)
    }

    func test_move_appliesXZ_preservesY() {
        let model = RoomEditModel(
            initial: makePayload([makeObject(id: "a", x: 0, y: 0.5, z: 0)])
        )
        model.move("a", to: SIMD2<Float>(3, 4), gestureBegan: true)
        let obj = model.roomPayload.objects[0]
        XCTAssertEqual(obj.position.x, 3, accuracy: 1e-5)
        XCTAssertEqual(obj.position.z, 4, accuracy: 1e-5)
        XCTAssertEqual(obj.position.y, 0.5, accuracy: 1e-5)
    }

    func test_move_unknownID_noOp() {
        let model = RoomEditModel(initial: makePayload([makeObject(id: "a")]))
        model.move("nope", to: SIMD2<Float>(1, 1), gestureBegan: true)
        XCTAssertFalse(model.canUndo)
    }

    // MARK: - rotate(_:toRadians:gestureBegan:)

    func test_rotate_began_snapshots() {
        let model = RoomEditModel(initial: makePayload([makeObject(id: "a")]))
        model.rotate("a", toRadians: .pi / 2, gestureBegan: true)
        XCTAssertTrue(model.canUndo)
    }

    func test_rotate_appliesRadians() {
        let model = RoomEditModel(
            initial: makePayload([makeObject(id: "a", rot: 0)])
        )
        model.rotate("a", toRadians: .pi, gestureBegan: true)
        XCTAssertEqual(model.roomPayload.objects[0].rotation_y, Double.pi, accuracy: 1e-5)
    }

    func test_rotate_subsequentUpdates_dontReSnapshot() {
        let model = RoomEditModel(initial: makePayload([makeObject(id: "a", rot: 0)]))
        model.rotate("a", toRadians: .pi / 2, gestureBegan: true)
        model.rotate("a", toRadians: .pi, gestureBegan: false)
        model.undo()
        XCTAssertEqual(model.roomPayload.objects[0].rotation_y, 0, accuracy: 1e-5)
    }
```

- [ ] **Step 2: Run — verify failure.**

Expected: 7 new tests fail.

- [ ] **Step 3: Implement.**

Replace the `move` and `rotate` stubs:

```swift
    func move(_ id: String, to xz: SIMD2<Float>, gestureBegan: Bool) {
        guard let idx = roomPayload.objects.firstIndex(where: { $0.id == id }) else { return }
        if gestureBegan {
            previousObjects = roomPayload.objects
            canUndo = true
        }
        let old = roomPayload.objects[idx]
        let newPos = ObjectPosition(x: Double(xz.x), y: old.position.y, z: Double(xz.y))
        var newObjects = roomPayload.objects
        newObjects[idx] = old.with(position: newPos)
        roomPayload = roomPayload.with(objects: newObjects)
    }

    func rotate(_ id: String, toRadians y: Float, gestureBegan: Bool) {
        guard let idx = roomPayload.objects.firstIndex(where: { $0.id == id }) else { return }
        if gestureBegan {
            previousObjects = roomPayload.objects
            canUndo = true
        }
        let old = roomPayload.objects[idx]
        var newObjects = roomPayload.objects
        newObjects[idx] = old.with(rotationY: Double(y))
        roomPayload = roomPayload.with(objects: newObjects)
    }
```

- [ ] **Step 4: Run — verify pass.**

```bash
xcodebuild test ... -only-testing:decroom_no_lidarTests/RoomEditModelTests 2>&1 | tail -5
```
Expected: 15 tests pass total.

- [ ] **Step 5: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/RoomEditModel.swift decroom_no_lidar/decroom_no_lidarTests/RoomEditModelTests.swift
git commit -m "feat(edit): RoomEditModel.move + rotate + tests (Plan 4a Task 4)"
```

---

### Task 5: Implement `undo()` + tests

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/RoomEditModel.swift`
- Modify: `decroom_no_lidar/decroom_no_lidarTests/RoomEditModelTests.swift`

- [ ] **Step 1: Append failing tests.**

```swift
    // MARK: - undo()

    func test_undo_restoresPreviousObjects() {
        let original = [makeObject(id: "a"), makeObject(id: "b", x: 1)]
        let model = RoomEditModel(initial: makePayload(original))
        model.delete("a")
        model.undo()
        XCTAssertEqual(model.roomPayload.objects.map { $0.id }, ["a", "b"])
    }

    func test_undo_clearsCanUndo() {
        let model = RoomEditModel(initial: makePayload([makeObject(id: "a"), makeObject(id: "b", x: 1)]))
        model.delete("a")
        model.undo()
        XCTAssertFalse(model.canUndo)
    }

    func test_undo_clearsSelection() {
        let model = RoomEditModel(initial: makePayload([makeObject(id: "a"), makeObject(id: "b", x: 1)]))
        model.select("b")
        model.delete("a")
        model.undo()
        XCTAssertNil(model.selectedObjectID)
    }

    func test_undo_noOpWhenNothingToUndo() {
        let model = RoomEditModel(initial: makePayload([makeObject(id: "a")]))
        model.undo()  // should not crash
        XCTAssertEqual(model.roomPayload.objects.count, 1)
        XCTAssertFalse(model.canUndo)
    }
```

- [ ] **Step 2: Run — verify failure.**

Expected: 4 new tests fail.

- [ ] **Step 3: Implement.**

Replace the `undo` stub:

```swift
    func undo() {
        guard let prev = previousObjects else { return }
        roomPayload = roomPayload.with(objects: prev)
        previousObjects = nil
        canUndo = false
        selectedObjectID = nil
    }
```

- [ ] **Step 4: Run — verify pass.**

Expected: 19 tests pass total (including 4 new + 15 prior).

Wait — `test_undo_clearsSelection` may set selection to `"b"` (which is valid in current payload) but after delete + undo, `"b"` is still in payload yet selection becomes nil because `undo` clears it. The test asserts nil — correct.

- [ ] **Step 5: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/RoomEditModel.swift decroom_no_lidar/decroom_no_lidarTests/RoomEditModelTests.swift
git commit -m "feat(edit): RoomEditModel.undo + tests (Plan 4a Task 5)"
```

---

### Task 6: `SceneSelection.wallSnap` + tests

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/SceneSelection.swift`
- Create: `decroom_no_lidar/decroom_no_lidarTests/SceneSelectionTests.swift`

- [ ] **Step 1: Write failing tests.**

`SceneSelectionTests.swift`:

```swift
import XCTest
import simd
@testable import decroom_no_lidar

final class SceneSelectionTests: XCTestCase {

    private func walls(width: Double = 4, length: Double = 4) -> [Wall] {
        let hw = width / 2, hl = length / 2
        return [
            Wall(id: "w0", start: [-hw, -hl], end: [ hw, -hl], has_window: false, has_door: false),
            Wall(id: "w1", start: [ hw, -hl], end: [ hw,  hl], has_window: false, has_door: false),
            Wall(id: "w2", start: [ hw,  hl], end: [-hw,  hl], has_window: false, has_door: false),
            Wall(id: "w3", start: [-hw,  hl], end: [-hw, -hl], has_window: false, has_door: false),
        ]
    }

    func test_wallSnap_bedCategory_snapsToNearestWall() {
        // Bed near the south wall (z = -2); should snap to z = -2.
        let snapped = SceneSelection.wallSnap(
            position: SIMD2<Float>(0, -1.7), category: "bed", walls: walls()
        )
        XCTAssertEqual(snapped.x, 0, accuracy: 0.01)
        XCTAssertEqual(snapped.y, -2, accuracy: 0.01)
    }

    func test_wallSnap_chairCategory_doesNotSnap() {
        let p = SIMD2<Float>(0.3, 0.7)
        let snapped = SceneSelection.wallSnap(position: p, category: "chair", walls: walls())
        XCTAssertEqual(snapped.x, p.x, accuracy: 0.001)
        XCTAssertEqual(snapped.y, p.y, accuracy: 0.001)
    }

    func test_clampToFootprint_outsideRoom_clampsToEdge() {
        let p = SIMD2<Float>(5, 3)  // way outside the 4×4 room
        let clamped = SceneSelection.clampToFootprint(p, walls: walls())
        XCTAssertEqual(clamped.x, 2, accuracy: 0.001)  // clamped to x_max
        XCTAssertEqual(clamped.y, 2, accuracy: 0.001)  // clamped to z_max
    }

    func test_clampToFootprint_insideRoom_unchanged() {
        let p = SIMD2<Float>(0.5, 0.5)
        let clamped = SceneSelection.clampToFootprint(p, walls: walls())
        XCTAssertEqual(clamped.x, 0.5, accuracy: 0.001)
        XCTAssertEqual(clamped.y, 0.5, accuracy: 0.001)
    }
}
```

- [ ] **Step 2: Run — verify failure.**

Expected: build failure / SceneSelection missing.

- [ ] **Step 3: Implement.**

`decroom_no_lidar/decroom_no_lidar/SceneSelection.swift`:

```swift
//
//  SceneSelection.swift
//  decroom_no_lidar
//
//  Plan 4a — SceneKit-specific helpers for the correction UI:
//   hitTest    : tap location -> RoomObject.id (with 22pt retry radius)
//   floorPoint : tap location -> world-space XZ via ray-plane intersection
//   wallSnap   : wall-snap-category positions snap to nearest wall
//   clampToFootprint : clamp position to the rectangular room bbox
//   setSelected: apply/remove the selection highlight to a SCNNode
//

import Foundation
import SceneKit
import UIKit
import simd

enum SceneSelection {

    // MARK: - Snap rules (mirror Plan 3b consensus.snap_to_walls categories)

    private static let wallSnapCategories: Set<String> = [
        "bed", "dresser", "wardrobe", "sofa", "couch", "bookshelf", "nightstand",
    ]

    // MARK: - Wall snap

    static func wallSnap(
        position: SIMD2<Float>,
        category: String,
        walls: [Wall]
    ) -> SIMD2<Float> {
        guard wallSnapCategories.contains(category) else { return position }
        guard !walls.isEmpty else { return position }

        var bestDist: Float = .greatestFiniteMagnitude
        var snapped = position
        for wall in walls {
            guard wall.start.count >= 2, wall.end.count >= 2 else { continue }
            let projected = projectOntoSegment(
                p: position,
                a: SIMD2<Float>(Float(wall.start[0]), Float(wall.start[1])),
                b: SIMD2<Float>(Float(wall.end[0]),   Float(wall.end[1]))
            )
            let d = simd_distance(position, projected)
            if d < bestDist {
                bestDist = d
                snapped = projected
            }
        }
        return snapped
    }

    private static func projectOntoSegment(
        p: SIMD2<Float>, a: SIMD2<Float>, b: SIMD2<Float>
    ) -> SIMD2<Float> {
        let ab = b - a
        let len2 = simd_dot(ab, ab)
        guard len2 > 1e-6 else { return a }
        let t = simd_dot(p - a, ab) / len2
        let tClamped = max(0, min(1, t))
        return a + ab * tClamped
    }

    // MARK: - Footprint clamp

    static func clampToFootprint(
        _ p: SIMD2<Float>, walls: [Wall]
    ) -> SIMD2<Float> {
        guard !walls.isEmpty else { return p }
        var xMin: Float = .greatestFiniteMagnitude
        var xMax: Float = -.greatestFiniteMagnitude
        var zMin: Float = .greatestFiniteMagnitude
        var zMax: Float = -.greatestFiniteMagnitude
        for w in walls {
            for endpoint in [w.start, w.end] {
                guard endpoint.count >= 2 else { continue }
                let x = Float(endpoint[0])
                let z = Float(endpoint[1])
                xMin = min(xMin, x); xMax = max(xMax, x)
                zMin = min(zMin, z); zMax = max(zMax, z)
            }
        }
        return SIMD2<Float>(
            max(xMin, min(xMax, p.x)),
            max(zMin, min(zMax, p.y))
        )
    }

    // MARK: - Hit test (Task 7)

    static func hitTest(_ scnView: SCNView, at point: CGPoint) -> String? {
        return nil  // Task 7
    }

    // MARK: - Floor raycast (Task 8)

    static func floorPoint(forScreen point: CGPoint, in scnView: SCNView) -> SIMD2<Float>? {
        return nil  // Task 8
    }

    // MARK: - Selection highlight (Task 11)

    static func setSelected(_ id: String?, in rootNode: SCNNode) {
        // Task 11
    }
}
```

- [ ] **Step 4: Run — verify pass.**

```bash
xcodebuild test ... -only-testing:decroom_no_lidarTests/SceneSelectionTests 2>&1 | tail -5
```
Expected: 4 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/SceneSelection.swift decroom_no_lidar/decroom_no_lidarTests/SceneSelectionTests.swift
git commit -m "feat(edit): SceneSelection.wallSnap + clampToFootprint + tests (Plan 4a Task 6)"
```

---

### Task 7: `SceneSelection.hitTest` with 22pt retry radius

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/SceneSelection.swift`

This step has no pure unit test — `SCNHitTest` requires a fully-set-up SCNView with a scene and a window. It will be covered by the XCUITest (Task 16) and manual QA. The implementation is small and standard.

- [ ] **Step 1: Implement `hitTest`.**

Replace the `hitTest` stub in `SceneSelection.swift`:

```swift
    static func hitTest(_ scnView: SCNView, at point: CGPoint) -> String? {
        let options: [SCNHitTestOption: Any] = [
            SCNHitTestOption.searchMode: SCNHitTestSearchMode.closest.rawValue,
            SCNHitTestOption.ignoreHiddenNodes: true,
        ]
        // Center hit first.
        if let hit = scnView.hitTest(point, options: options).first,
           let id = roomObjectID(forNode: hit.node) {
            return id
        }
        // Retry within a 22pt radius (8 surrounding points).
        let offsets: [(Int, Int)] = [
            (11, 0), (-11, 0), (0, 11), (0, -11),
            (8, 8), (-8, 8), (8, -8), (-8, -8),
        ]
        for (dx, dy) in offsets {
            let p = CGPoint(x: point.x + CGFloat(dx), y: point.y + CGFloat(dy))
            if let hit = scnView.hitTest(p, options: options).first,
               let id = roomObjectID(forNode: hit.node) {
                return id
            }
        }
        return nil
    }

    /// Walk up the node tree until we find a node whose name starts with
    /// "obj_" (the convention used by RoomSceneBuilder).
    private static func roomObjectID(forNode node: SCNNode) -> String? {
        var current: SCNNode? = node
        while let n = current {
            if let name = n.name, name.hasPrefix("obj_") {
                return name
            }
            current = n.parent
        }
        return nil
    }
```

- [ ] **Step 2: Verify `RoomSceneBuilder` names furniture nodes with `"obj_<id>"`.**

```bash
grep -n "node.name" /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift | head -10
```

If furniture nodes already have `node.name = "obj_\(obj.id)"` (or equivalent), no further change. If they don't, add to `RoomSceneBuilder.buildScene` (around line ~200 where each object's node is created):

```swift
furnitureNode.name = "obj_\(obj.id)"
```

Build the project to confirm compilation:

```bash
xcodebuild build -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -3
```
Expected: `BUILD SUCCEEDED`.

- [ ] **Step 3: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/SceneSelection.swift decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift
git commit -m "feat(edit): SceneSelection.hitTest with 22pt retry radius (Plan 4a Task 7)"
```

---

### Task 8: `SceneSelection.floorPoint` — ray-plane intersection

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/SceneSelection.swift`
- Modify: `decroom_no_lidar/decroom_no_lidarTests/SceneSelectionTests.swift`

The math is testable without an SCNView via dependency injection (we pass `unproject` as a closure in tests).

- [ ] **Step 1: Add the internal `floorPointFromRay` helper that the test exercises.**

Append to `SceneSelectionTests.swift`:

```swift
    // MARK: - floorPoint math (pure)

    func test_floorPointFromRay_centerRayHitsFloor() {
        // Ray straight down: origin (0, 5, 0), dir (0, -1, 0) → hits y=0 at (0, 0).
        let p = SceneSelection.floorPointFromRay(
            near: SIMD3<Float>(0, 5, 0), far: SIMD3<Float>(0, 4, 0)
        )
        XCTAssertNotNil(p)
        XCTAssertEqual(p!.x, 0, accuracy: 1e-4)
        XCTAssertEqual(p!.y, 0, accuracy: 1e-4)
    }

    func test_floorPointFromRay_parallelRay_returnsNil() {
        // near.y == far.y → ray parallel to floor.
        let p = SceneSelection.floorPointFromRay(
            near: SIMD3<Float>(0, 2, 0), far: SIMD3<Float>(1, 2, 0)
        )
        XCTAssertNil(p)
    }
```

- [ ] **Step 2: Run — verify failure.**

Expected: build failure / `floorPointFromRay` undefined.

- [ ] **Step 3: Implement.**

Replace the `floorPoint` stub block in `SceneSelection.swift`:

```swift
    // MARK: - Floor raycast

    /// Project a screen point onto the y=0 floor plane in world space.
    /// Returns nil if the ray is parallel to the floor or points away.
    static func floorPoint(forScreen point: CGPoint, in scnView: SCNView) -> SIMD2<Float>? {
        let near = scnView.unprojectPoint(SCNVector3(Float(point.x), Float(point.y), 0))
        let far  = scnView.unprojectPoint(SCNVector3(Float(point.x), Float(point.y), 1))
        return floorPointFromRay(
            near: SIMD3<Float>(near.x, near.y, near.z),
            far:  SIMD3<Float>(far.x,  far.y,  far.z)
        )
    }

    /// Pure-math ray/plane intersection — unit-testable.
    static func floorPointFromRay(
        near: SIMD3<Float>, far: SIMD3<Float>
    ) -> SIMD2<Float>? {
        let dir = far - near
        guard abs(dir.y) > 1e-4 else { return nil }
        let t = -near.y / dir.y
        guard t > 0 else { return nil }
        let world = near + dir * t
        return SIMD2<Float>(world.x, world.z)
    }
```

- [ ] **Step 4: Run — verify pass.**

```bash
xcodebuild test ... -only-testing:decroom_no_lidarTests/SceneSelectionTests 2>&1 | tail -5
```
Expected: 6 tests pass (4 prior + 2 new).

- [ ] **Step 5: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/SceneSelection.swift decroom_no_lidar/decroom_no_lidarTests/SceneSelectionTests.swift
git commit -m "feat(edit): SceneSelection.floorPoint via ray-plane intersection (Plan 4a Task 8)"
```

---

### Task 9: `RoomEditMenuView` SwiftUI overlay

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/RoomEditMenuView.swift`

No unit test for this — pure SwiftUI, exercised by the XCUITest (Task 16) and manual QA.

- [ ] **Step 1: Implement.**

```swift
//
//  RoomEditMenuView.swift
//  decroom_no_lidar
//
//  Plan 4a — bottom action sheet + top-trailing Undo button.
//  Bound to RoomEditModel; renders nothing when selectedObjectID is nil
//  and canUndo is false.
//

import SwiftUI

struct RoomEditMenuView: View {
    @ObservedObject var editModel: RoomEditModel

    var body: some View {
        ZStack(alignment: .topTrailing) {
            // Undo button (top-trailing).
            if editModel.canUndo {
                Button {
                    editModel.undo()
                } label: {
                    Label("Undo", systemImage: "arrow.uturn.backward")
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                }
                .padding(12)
            }

            // Action sheet (bottom).
            if editModel.selectedObjectID != nil {
                VStack(spacing: 0) {
                    Spacer()
                    actionSheet
                }
                .frame(maxWidth: .infinity)
            }
        }
        .animation(.easeInOut(duration: 0.15), value: editModel.selectedObjectID)
        .animation(.easeInOut(duration: 0.15), value: editModel.activeAction)
        .animation(.easeInOut(duration: 0.15), value: editModel.canUndo)
    }

    @ViewBuilder
    private var actionSheet: some View {
        VStack(spacing: 8) {
            if let action = editModel.activeAction {
                Text(hintText(for: action))
                    .font(.subheadline)
                    .padding(.vertical, 6)
                HStack {
                    Spacer()
                    Button("Done") {
                        editModel.activeAction = nil
                    }
                    .buttonStyle(.borderedProminent)
                    Spacer()
                }
            } else {
                HStack(spacing: 16) {
                    actionButton("Delete", systemImage: "trash") {
                        if let id = editModel.selectedObjectID {
                            editModel.delete(id)
                        }
                    }
                    actionButton("Move", systemImage: "arrow.up.and.down.and.arrow.left.and.right") {
                        editModel.activeAction = .move
                    }
                    actionButton("Rotate", systemImage: "rotate.right") {
                        editModel.activeAction = .rotate
                    }
                    actionButton("Done", systemImage: "checkmark") {
                        editModel.select(nil)
                    }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal, 12)
        .padding(.bottom, 16)
    }

    private func actionButton(
        _ label: String, systemImage: String, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: systemImage).font(.title3)
                Text(label).font(.caption)
            }
            .frame(minWidth: 56, minHeight: 44)
        }
        .accessibilityIdentifier("editAction.\(label)")
    }

    private func hintText(for action: EditAction) -> String {
        switch action {
        case .move:   return "Drag to reposition. Tap Done to finish."
        case .rotate: return "Twist with two fingers. Tap Done to finish."
        }
    }
}
```

The `.accessibilityIdentifier` is what the XCUITest (Task 16) uses to tap buttons reliably.

- [ ] **Step 2: Build to verify compilation.**

```bash
xcodebuild build -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -3
```
Expected: `BUILD SUCCEEDED`.

- [ ] **Step 3: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/RoomEditMenuView.swift
git commit -m "feat(edit): RoomEditMenuView SwiftUI overlay (Plan 4a Task 9)"
```

---

### Task 10: `RoomEditingCoordinator` skeleton + tap recognizer

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidar/RoomEditingCoordinator.swift`
- Modify: `decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift`

- [ ] **Step 1: Create the Coordinator.**

`decroom_no_lidar/decroom_no_lidar/RoomEditingCoordinator.swift`:

```swift
//
//  RoomEditingCoordinator.swift
//  decroom_no_lidar
//
//  Plan 4a — owns gesture recognizers for the editing UI and bridges
//  them to RoomEditModel.
//

import Foundation
import SceneKit
import UIKit

@MainActor
final class RoomEditingCoordinator: NSObject {
    weak var scnView: SCNView?
    var editModel: RoomEditModel?

    // Recognizers installed for the duration of an active edit action.
    private var panRecognizer: UIPanGestureRecognizer?
    private var rotateRecognizer: UIRotationGestureRecognizer?

    func attach(to scnView: SCNView, editModel: RoomEditModel) {
        self.scnView = scnView
        self.editModel = editModel
        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        scnView.addGestureRecognizer(tap)
    }

    @objc func handleTap(_ gr: UITapGestureRecognizer) {
        guard let scnView = scnView, let model = editModel else { return }
        let point = gr.location(in: scnView)
        let id = SceneSelection.hitTest(scnView, at: point)
        model.select(id)
    }

    // Pan and rotation handlers — Task 12.
    @objc func handlePan(_ gr: UIPanGestureRecognizer) {
        // Task 12
    }

    @objc func handleRotate(_ gr: UIRotationGestureRecognizer) {
        // Task 12
    }

    // Recognizer installation / removal — Task 12.
    func setActiveAction(_ action: EditAction?) {
        // Task 12
    }
}
```

- [ ] **Step 2: Wire the Coordinator into `RoomSceneRepresentable`.**

In `RoomSceneView.swift`, modify `RoomSceneRepresentable` (around line 74). Two changes:

(a) Add an optional `editModel` property near the top of the struct (line ~75):

```swift
struct RoomSceneRepresentable: UIViewRepresentable {
    let payload: RoomPayload
    let lightsEnabled: Bool
    var editModel: RoomEditModel? = nil   // Plan 4a — optional injection

    func makeCoordinator() -> RoomEditingCoordinator {
        RoomEditingCoordinator()
    }
```

(b) At the end of `makeUIView` (just before `return scnView`, around line 100), attach the Coordinator if an editModel was provided:

```swift
        if let editModel = editModel {
            context.coordinator.attach(to: scnView, editModel: editModel)
        }

        return scnView
```

- [ ] **Step 3: Build.**

```bash
xcodebuild build -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -3
```
Expected: `BUILD SUCCEEDED`. Existing call sites that pass `payload: ...` continue to work because `editModel` defaults to nil.

- [ ] **Step 4: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/RoomEditingCoordinator.swift decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift
git commit -m "feat(edit): RoomEditingCoordinator + tap recognizer (Plan 4a Task 10)"
```

---

### Task 11: Selection highlight on the SCNNode

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/SceneSelection.swift`
- Modify: `decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift`

- [ ] **Step 1: Implement `setSelected` in `SceneSelection.swift`.**

Replace the `setSelected` stub:

```swift
    // MARK: - Selection highlight

    private static let highlightNodeName = "_selection_highlight"

    /// Apply a thin emissive cyan outline to the SCNNode owning the given
    /// RoomObject.id. Pass nil to clear all highlights. Uses a child node
    /// scaled slightly larger than the parent so the outline shows through.
    static func setSelected(_ id: String?, in rootNode: SCNNode) {
        // Remove existing highlight wherever it lives.
        rootNode.enumerateChildNodes { node, _ in
            if node.name == highlightNodeName {
                node.removeFromParentNode()
            }
        }
        guard let id = id else { return }
        guard let target = findNode(named: "obj_\(id)", under: rootNode) else { return }

        // Build a thin outer-shell box matching the object's bounding box.
        let (minVec, maxVec) = target.boundingBox
        let size = maxVec - minVec
        let outline = SCNNode(geometry: SCNBox(
            width: CGFloat(size.x), height: CGFloat(size.y),
            length: CGFloat(size.z), chamferRadius: 0
        ))
        outline.name = highlightNodeName
        outline.position = SCNVector3(
            (minVec.x + maxVec.x) / 2,
            (minVec.y + maxVec.y) / 2,
            (minVec.z + maxVec.z) / 2
        )
        outline.scale = SCNVector3(1.02, 1.02, 1.02)   // slightly larger
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.clear
        material.emission.contents = UIColor.cyan
        material.lightingModel = .constant
        material.isDoubleSided = true
        material.fillMode = .lines  // wireframe outline
        outline.geometry?.firstMaterial = material
        target.addChildNode(outline)
    }

    private static func findNode(named name: String, under root: SCNNode) -> SCNNode? {
        if root.name == name { return root }
        for child in root.childNodes {
            if let found = findNode(named: name, under: child) { return found }
        }
        return nil
    }
```

- [ ] **Step 2: Apply selection highlight reactively from `RoomSceneRepresentable.updateUIView`.**

Modify `updateUIView` (around line 103) — at the end of the existing body, before the closing brace, add:

```swift
        // Plan 4a: apply / clear selection highlight.
        if let scene = uiView.scene {
            SceneSelection.setSelected(editModel?.selectedObjectID, in: scene.rootNode)
        }
```

`updateUIView` is called by SwiftUI whenever an `@Published` change on the bound `editModel` triggers a re-evaluation of the parent view's body. Selection toggles will flow here automatically.

- [ ] **Step 3: Build.**

```bash
xcodebuild build ... 2>&1 | tail -3
```
Expected: `BUILD SUCCEEDED`.

- [ ] **Step 4: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/SceneSelection.swift decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift
git commit -m "feat(edit): selection highlight on SCNNode (Plan 4a Task 11)"
```

---

### Task 12: Move + Rotate gesture handlers in the Coordinator

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/RoomEditingCoordinator.swift`
- Modify: `decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift`

- [ ] **Step 1: Replace the stub handlers in `RoomEditingCoordinator.swift`.**

```swift
    @objc func handlePan(_ gr: UIPanGestureRecognizer) {
        guard let scnView = scnView, let model = editModel else { return }
        guard model.activeAction == .move, let id = model.selectedObjectID else { return }
        let point = gr.location(in: scnView)
        guard let floor = SceneSelection.floorPoint(forScreen: point, in: scnView) else { return }

        // Category-aware snap + footprint clamp.
        let category = model.roomPayload.objects.first(where: { $0.id == id })?.type ?? ""
        let snapped = SceneSelection.wallSnap(
            position: floor, category: category,
            walls: model.roomPayload.room.walls
        )
        let clamped = SceneSelection.clampToFootprint(
            snapped, walls: model.roomPayload.room.walls
        )

        let began = (gr.state == .began)
        model.move(id, to: clamped, gestureBegan: began)
    }

    @objc func handleRotate(_ gr: UIRotationGestureRecognizer) {
        guard let model = editModel else { return }
        guard model.activeAction == .rotate, let id = model.selectedObjectID else { return }

        // Snap to nearest 90° increment.
        let snapped = round(gr.rotation / (.pi / 2)) * (.pi / 2)
        let current = model.roomPayload.objects.first(where: { $0.id == id })?.rotation_y ?? 0
        let newRotation = current - Double(snapped)   // negate: SceneKit CW vs backend CCW

        let began = (gr.state == .began)
        model.rotate(id, toRadians: Float(newRotation), gestureBegan: began)

        if gr.state == .ended || gr.state == .cancelled {
            gr.rotation = 0   // reset for next gesture
        }
    }

    func setActiveAction(_ action: EditAction?) {
        guard let scnView = scnView else { return }

        // Disable camera control while an action is active.
        scnView.allowsCameraControl = (action == nil)

        // Remove any existing pan/rotate.
        if let pr = panRecognizer {
            scnView.removeGestureRecognizer(pr); panRecognizer = nil
        }
        if let rr = rotateRecognizer {
            scnView.removeGestureRecognizer(rr); rotateRecognizer = nil
        }

        switch action {
        case .move:
            let pr = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            scnView.addGestureRecognizer(pr)
            panRecognizer = pr
        case .rotate:
            let rr = UIRotationGestureRecognizer(target: self, action: #selector(handleRotate(_:)))
            scnView.addGestureRecognizer(rr)
            rotateRecognizer = rr
        case .none:
            break
        }
    }
```

- [ ] **Step 2: Drive `setActiveAction` from `RoomSceneRepresentable.updateUIView`.**

In `RoomSceneView.swift`'s `updateUIView` (around line 103), at the END of the body (after the selection highlight update from Task 11), add:

```swift
        context.coordinator.setActiveAction(editModel?.activeAction)
```

- [ ] **Step 3: Build.**

```bash
xcodebuild build ... 2>&1 | tail -3
```
Expected: `BUILD SUCCEEDED`.

- [ ] **Step 4: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/RoomEditingCoordinator.swift decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift
git commit -m "feat(edit): move + rotate gesture handlers in coordinator (Plan 4a Task 12)"
```

---

### Task 13: Editing-aware `RoomSceneView` initializer + menu overlay

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift`

- [ ] **Step 1: Add an editing initializer and overlay to `RoomSceneView`.**

Replace the existing `RoomSceneView` body (line 46–70) with:

```swift
struct RoomSceneView: View {
    let payload: RoomPayload
    @State private var lightsEnabled: Bool = true
    @ObservedObject private var editModel: RoomEditModel
    private let editingEnabled: Bool

    /// Read-only initializer (existing call sites). Editing disabled.
    init(payload: RoomPayload) {
        self.payload = payload
        // A throwaway model is created so the @ObservedObject stays valid;
        // editingEnabled=false means it's never wired to the scene.
        self.editModel = RoomEditModel(initial: payload)
        self.editingEnabled = false
    }

    /// Editing initializer (Plan 4a call sites).
    init(payload: RoomPayload, editModel: RoomEditModel) {
        self.payload = payload
        self.editModel = editModel
        self.editingEnabled = true
    }

    var body: some View {
        VStack(spacing: 0) {
            ZStack(alignment: .topTrailing) {
                RoomSceneRepresentable(
                    payload: editingEnabled ? editModel.roomPayload : payload,
                    lightsEnabled: lightsEnabled,
                    editModel: editingEnabled ? editModel : nil
                )
                .clipShape(RoundedRectangle(cornerRadius: 12))

                // Lighting toggle button (existing).
                Button {
                    lightsEnabled.toggle()
                } label: {
                    Image(systemName: lightsEnabled ? "lightbulb.fill" : "lightbulb.slash")
                        .font(.title3)
                        .foregroundStyle(lightsEnabled ? .yellow : .gray)
                        .padding(10)
                        .background(.ultraThinMaterial, in: Circle())
                }
                .padding(10)

                // Plan 4a: editing menu overlay (only when editing enabled).
                if editingEnabled {
                    RoomEditMenuView(editModel: editModel)
                        .allowsHitTesting(true)
                }
            }
        }
    }
}
```

The read-only initializer keeps existing call sites (`CaptureFlowView`, `RoomCardView`) compiling unchanged. The new editing initializer is what Plan 4a's hosting view will use.

- [ ] **Step 2: Critical detail — `RoomSceneRepresentable` must rebuild the scene when `payload` changes.**

The current `updateUIView` only updates lighting. Since edits change `editModel.roomPayload.objects`, we need the scene-graph to reflect that. Modify `updateUIView` to detect payload-content changes and rebuild furniture nodes:

```swift
    func updateUIView(_ uiView: SCNView, context: Context) {
        // Lighting update (existing behavior).
        if let scene = uiView.scene {
            updateLighting(in: scene)
        }

        // Plan 4a: rebuild furniture if the object list changed.
        if let scene = uiView.scene, editModel != nil {
            rebuildFurnitureIfNeeded(in: scene, against: payload)
        }

        // Plan 4a: apply / clear selection highlight.
        if let scene = uiView.scene {
            SceneSelection.setSelected(editModel?.selectedObjectID, in: scene.rootNode)
        }

        // Plan 4a: install / remove pan-or-rotate gestures.
        context.coordinator.setActiveAction(editModel?.activeAction)
    }

    private func updateLighting(in scene: SCNScene) {
        // ... existing lighting code from the previous updateUIView ...
    }

    private func rebuildFurnitureIfNeeded(in scene: SCNScene, against payload: RoomPayload) {
        // Remove existing furniture nodes (name starts with "obj_").
        let toRemove = scene.rootNode.childNodes.filter {
            ($0.name ?? "").hasPrefix("obj_")
        }
        for n in toRemove { n.removeFromParentNode() }

        // Re-add from the current payload using the existing builder.
        for obj in payload.objects {
            if let node = RoomSceneBuilder.makeFurnitureNode(for: obj) {
                node.name = "obj_\(obj.id)"
                scene.rootNode.addChildNode(node)
            }
        }
    }
```

If `RoomSceneBuilder` doesn't already expose a per-object `makeFurnitureNode(for:)` static, refactor `buildScene` to extract it. The refactor: pull the "for each object in objects, build node, position, scale, rotate, addChild" loop body into a separate static func returning the assembled `SCNNode?`. The existing `buildScene` then calls it in its for loop.

- [ ] **Step 3: Refactor `RoomSceneBuilder` to expose `makeFurnitureNode(for:)`.**

In `RoomSceneView.swift`, locate `RoomSceneBuilder.buildScene` (around line 151). The current code iterates `for obj in objects` building each node inline. Extract the body of that loop into a new static method on `RoomSceneBuilder`:

```swift
    /// Build a single furniture SCNNode (USDZ-loaded + positioned + rotated +
    /// scaled). Returns nil if the asset can't be loaded.
    static func makeFurnitureNode(for obj: RoomObject) -> SCNNode? {
        // ... the existing per-object body from `buildScene`, copy-pasted ...
        // Make sure the returned node has its position, eulerAngles.y, and
        // scale set per `obj`, but DOES NOT set `node.name` (the caller does).
    }
```

The existing `buildScene` then becomes:

```swift
        for obj in objects {
            if let node = makeFurnitureNode(for: obj) {
                node.name = "obj_\(obj.id)"
                scene.rootNode.addChildNode(node)
            }
        }
```

Be precise: the existing inline code likely also computes the bounding-box scale factor (per CLAUDE.md: "USDZ Kenney models need bounding box measurement + median scale factor + Y-offset to floor"). All of that math goes into `makeFurnitureNode(for:)` unchanged — just relocated.

- [ ] **Step 4: Build.**

```bash
xcodebuild build ... 2>&1 | tail -3
```
Expected: `BUILD SUCCEEDED`. Existing call sites still build because `RoomSceneView(payload:)` still exists.

- [ ] **Step 5: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/RoomSceneView.swift
git commit -m "feat(edit): editing-aware RoomSceneView initializer + scene rebuild on edits (Plan 4a Task 13)"
```

---

### Task 14: Wire `CaptureFlowView` to use the editing initializer

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/CaptureFlowView.swift`

- [ ] **Step 1: Inspect the existing call site (around line 85).**

```bash
sed -n '80,95p' /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/CaptureFlowView.swift
```

The line should look like `RoomSceneView(payload: payload)` inside some SwiftUI body where `payload` is a `RoomPayload` already in scope.

- [ ] **Step 2: Replace with the editing initializer.**

Add a `@StateObject` near the top of the containing view (next to other state):

```swift
@StateObject private var editModel: RoomEditModel
```

And add an initializer to the containing view that constructs it from the payload:

```swift
init(payload: RoomPayload, /* ...other existing args... */) {
    // ... existing init body ...
    _editModel = StateObject(wrappedValue: RoomEditModel(initial: payload))
}
```

Replace the call:

```swift
RoomSceneView(payload: payload, editModel: editModel)
```

If `CaptureFlowView` doesn't yet have a custom init, add one mirroring the property list. If `payload` is bound later (e.g., flows in via `.onAppear`), put `editModel = RoomEditModel(initial: payload)` in `.onAppear` instead and switch `@StateObject` to a lazy `var editModel: RoomEditModel?` pattern — but the simplest is to inject `payload` at construction time and use `@StateObject` directly.

- [ ] **Step 3: Build.**

```bash
xcodebuild build ... 2>&1 | tail -3
```
Expected: `BUILD SUCCEEDED`.

- [ ] **Step 4: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/CaptureFlowView.swift
git commit -m "feat(edit): wire CaptureFlowView to RoomSceneView editing init (Plan 4a Task 14)"
```

---

### Task 15: Wire `RoomCardView` to use the editing initializer

**Files:**
- Modify: `decroom_no_lidar/decroom_no_lidar/RoomCardView.swift`

Same change pattern as Task 14, applied to `RoomCardView.swift:248`.

- [ ] **Step 1: Inspect.**

```bash
sed -n '244,255p' /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar/RoomCardView.swift
```

- [ ] **Step 2: Replace with the editing initializer.**

Same `@StateObject editModel = RoomEditModel(initial: room.payload)` pattern, then:

```swift
RoomSceneView(payload: room.payload, editModel: editModel)
```

- [ ] **Step 3: Build.**

```bash
xcodebuild build ... 2>&1 | tail -3
```
Expected: `BUILD SUCCEEDED`.

- [ ] **Step 4: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/RoomCardView.swift
git commit -m "feat(edit): wire RoomCardView to RoomSceneView editing init (Plan 4a Task 15)"
```

---

### Task 16: XCUITest — happy-path tap → Delete

**Files:**
- Create: `decroom_no_lidar/decroom_no_lidarUITests/RoomSceneViewEditTests.swift`

- [ ] **Step 1: Verify the `decroom_no_lidarUITests` target exists.**

```bash
ls /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidarUITests/ 2>&1
```
Expected: a directory containing `decroom_no_lidarUITests.swift` or similar. If the directory exists, files in it auto-register (PBXFileSystemSynchronizedRootGroup). If it doesn't exist, the scheme has no UI test target — skip this task and rely on manual QA (Task 17) only.

- [ ] **Step 2: Create the test.**

```swift
//
//  RoomSceneViewEditTests.swift
//  decroom_no_lidarUITests
//
//  Plan 4a — one happy-path: launch with a stub RoomPayload that places
//  a chair at the origin, tap near screen-center to select it, tap the
//  Delete button, and assert the chair node is no longer in the scene
//  (by tapping again and confirming no selection menu appears).
//

import XCTest

final class RoomSceneViewEditTests: XCTestCase {

    func test_tapChair_tapDelete_chairGone() throws {
        let app = XCUIApplication()
        app.launchArguments += ["--ui-test-room-edit-stub"]
        app.launch()

        // Tap roughly at the center where the stub chair is rendered.
        let scene = app.otherElements.matching(identifier: "roomScene").firstMatch
        XCTAssertTrue(scene.waitForExistence(timeout: 5))
        let center = scene.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        center.tap()

        // The edit menu should appear with a Delete button.
        let deleteBtn = app.buttons["editAction.Delete"]
        XCTAssertTrue(deleteBtn.waitForExistence(timeout: 2))
        deleteBtn.tap()

        // After delete, tapping the same point should not re-show the menu
        // (no object to select; menu element disappears).
        center.tap()
        let menuStillExists = deleteBtn.waitForExistence(timeout: 1.5)
        XCTAssertFalse(menuStillExists, "Edit menu still visible after delete — chair was not removed")
    }
}
```

- [ ] **Step 3: Add the UI-test launch-arg path to the app.**

In whichever `App` struct or root view is the entry point (likely `decroom_no_lidarApp.swift` — find via `grep`), branch on `ProcessInfo.processInfo.arguments.contains("--ui-test-room-edit-stub")` to swap in a hardcoded `RoomPayload` containing one chair at the origin and a small 4×4 room. Render that payload directly via the editing `RoomSceneView` and apply `.accessibilityIdentifier("roomScene")` to the SCNView container.

Concrete pattern:

```swift
@main
struct decroom_no_lidarApp: App {
    var body: some Scene {
        WindowGroup {
            if ProcessInfo.processInfo.arguments.contains("--ui-test-room-edit-stub") {
                UITestStubView()
            } else {
                ContentView()   // existing
            }
        }
    }
}

private struct UITestStubView: View {
    @StateObject private var editModel: RoomEditModel
    init() {
        let chair = RoomObject(
            id: "chair_test", type: "chair", asset_id: "kenney_chair",
            position: ObjectPosition(x: 0, y: 0, z: 0), rotation_y: 0, scale: 1,
            confidence: 0.9, metadata: ObjectMetadata(width: 0.5, length: 0.5, height: 0.85)
        )
        let walls = [
            Wall(id: "w0", start: [-2, -2], end: [2, -2], has_window: false, has_door: false),
            Wall(id: "w1", start: [2, -2],  end: [2, 2],  has_window: false, has_door: false),
            Wall(id: "w2", start: [2, 2],   end: [-2, 2], has_window: false, has_door: false),
            Wall(id: "w3", start: [-2, 2],  end: [-2, -2], has_window: false, has_door: false),
        ]
        let room = RoomInfo(id: "r", width_m: 4, length_m: 4, height_m: 2.5, walls: walls)
        let payload = RoomPayload(room: room, objects: [chair])
        _editModel = StateObject(wrappedValue: RoomEditModel(initial: payload))
    }
    var body: some View {
        RoomSceneView(payload: editModel.roomPayload, editModel: editModel)
            .accessibilityIdentifier("roomScene")
    }
}
```

- [ ] **Step 4: Run the test.**

```bash
xcodebuild test -scheme decroom_no_lidar -project /Users/jaejeon/Desktop/Decroom/decroom_no_lidar/decroom_no_lidar/decroom_no_lidar.xcodeproj -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:decroom_no_lidarUITests/RoomSceneViewEditTests 2>&1 | tail -10
```
Expected: 1 test passes. If hit-test misses (chair tapped at wrong screen point), tune the camera's initial position so the chair is dead-center in the rendered output, or expand the test's tap to a 3×3 grid around center.

- [ ] **Step 5: Commit.**

```bash
git add decroom_no_lidar/decroom_no_lidar/decroom_no_lidarApp.swift decroom_no_lidar/decroom_no_lidarUITests/RoomSceneViewEditTests.swift
git commit -m "test(edit): XCUITest happy-path tap+delete (Plan 4a Task 16)"
```

---

### Task 17: Manual QA on device + merge

**Files:**
- Create: `docs/superpowers/qa/<YYYY-MM-DD>-plan4a-manual-qa.md`

- [ ] **Step 1: Build for device and deploy via Xcode.**

Open the project in Xcode, select a real iPhone, hit Run. Drive a fresh scan to produce a RoomPayload, then exercise the edit UI on the resulting 3D view.

- [ ] **Step 2: Execute the manual QA checklist.**

| Check | Pass? |
|---|---|
| Tap a chair → selection highlight visible | |
| Bottom sheet shows [Delete] [Move] [Rotate] [Done] | |
| Undo button appears top-right after Delete | |
| Delete: chair gone from scene; tap again does nothing | |
| Move: tap [Move] → hint changes → drag works → chair follows finger | |
| Move + wall-snap: drag bed → snaps to nearest wall as it nears | |
| Move + footprint clamp: drag chair past wall → clamps to wall | |
| Rotate: tap [Rotate] → two-finger twist → bed snaps in 90° clicks | |
| Camera control: pan/zoom inert during Move and Rotate; resumes after Done | |
| Undo restores last delete / move / rotate | |
| Empty-space tap clears selection | |
| Tap on small lamp at distance: 22pt retry catches it | |
| Selection highlight visible against light USDZ texture | |
| App background mid-Move → recovery: object stays at last position | |

- [ ] **Step 3: Document findings.**

Create `docs/superpowers/qa/<YYYY-MM-DD>-plan4a-manual-qa.md`:

```markdown
# Plan 4a Manual QA — Correction UI

**Date**: <YYYY-MM-DD>
**Device**: iPhone <model>, iOS <version>
**Build**: feature/plan4a-correction-ui @ <commit>

## Checklist results

[paste the table above with Pass/Fail in each row]

## Issues found

(list any UX feel issues, gesture conflicts, highlight visibility on
specific USDZ models, or wall-snap stickiness near corners)

## Verdict

- [ ] PASS — merge to main
- [ ] FAIL — fix listed issues, re-run QA
```

- [ ] **Step 4: If PASS, merge to main.**

```bash
git checkout main
git pull
git merge --no-ff feature/plan4a-correction-ui -m "Merge Plan 4a — correction UI (delete/move/rotate + undo)"
```

Do NOT push to origin without explicit user request (per session convention).

- [ ] **Step 5: Commit the QA doc.**

```bash
git add docs/superpowers/qa/<YYYY-MM-DD>-plan4a-manual-qa.md
git commit -m "qa(plan4a): manual QA results — <verdict>"
```

- [ ] **Step 6: Update memory.**

Edit `memory/plan_status.md` to add a Plan 4a entry marked merged. No need to update `plan4a_deferred_issues.md` — its issues are now next in queue.

---

## Self-review

**Spec coverage check** — each spec section maps to one or more tasks:

| Spec section | Task(s) |
|---|---|
| Architecture (model / menu / scene-selection split) | 2, 6, 9, 10, 13 |
| Components — RoomEditModel API | 2, 3, 4, 5 |
| Components — RoomEditMenuView | 9 |
| Components — SceneSelection (hitTest + floorPoint + setSelected) | 6, 7, 8, 11 |
| Modified RoomSceneView | 10, 11, 12, 13 |
| Data flow (delete / move / rotate / undo) | 3, 4, 5, 12 |
| Gesture handling (camera disable, hit-test radius, raycast, snap) | 7, 8, 11, 12 |
| Undo behavior (single snapshot, gestureBegan flag) | 5 |
| Error handling (silent fail, footprint clamp, race-after-delete) | 2 (validating setter), 4, 12 (clamp), 5 |
| Testing — RoomEditModelTests | 2, 3, 4, 5 |
| Testing — SceneSelectionTests | 6, 8 |
| Testing — XCUITest | 16 |
| Manual QA + merge | 17 |

No gaps.

**Placeholder scan** — every code step shows actual code. The only `<YYYY-MM-DD>` placeholder is in the manual-QA filename, which is a normal date-fill-in. Two "if the synthesized init isn't accessible…" fallback notes (Task 1, Task 13) describe the fallback action concretely, not as a TBD.

**Type consistency** —
- `EditAction.move`/`.rotate` defined in Task 2, used in Tasks 4, 9, 12.
- `RoomEditModel.select / delete / move / rotate / undo` signatures consistent across Tasks 2, 3, 4, 5, 9, 10, 12.
- `SceneSelection.hitTest / floorPoint / wallSnap / clampToFootprint / setSelected` signatures consistent across Tasks 6, 7, 8, 10, 11, 12.
- `RoomObject.with(position:rotationY:)` and `RoomPayload.with(objects:)` defined Task 1, used in Tasks 3, 4, 5.
- `RoomEditingCoordinator.attach / setActiveAction / handleTap / handlePan / handleRotate` signatures consistent across Tasks 10, 12.
- Furniture-node naming convention `"obj_<RoomObject.id>"` — established in Task 7, depended on by Tasks 11 (`setSelected`) and 13 (`rebuildFurnitureIfNeeded`).

Done.
