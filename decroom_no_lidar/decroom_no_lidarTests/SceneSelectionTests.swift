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

    func test_wallSnap_deskCategory_snapsToNearestWall() {
        let snapped = SceneSelection.wallSnap(
            position: SIMD2<Float>(0, -1.7), category: "desk", walls: walls()
        )
        XCTAssertEqual(snapped.x, 0, accuracy: 0.01)
        XCTAssertEqual(snapped.y, -2, accuracy: 0.01)
    }

    func test_wallSnap_tvCategory_snapsToNearestWall() {
        let snapped = SceneSelection.wallSnap(
            position: SIMD2<Float>(0, -1.7), category: "tv", walls: walls()
        )
        XCTAssertEqual(snapped.x, 0, accuracy: 0.01)
        XCTAssertEqual(snapped.y, -2, accuracy: 0.01)
    }
}
