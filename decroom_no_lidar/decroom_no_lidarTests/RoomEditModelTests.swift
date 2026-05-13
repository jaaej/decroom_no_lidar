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
}
