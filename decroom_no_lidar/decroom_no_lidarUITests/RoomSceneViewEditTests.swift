//
//  RoomSceneViewEditTests.swift
//  decroom_no_lidarUITests
//
//  Plan 4a — happy-path XCUITest: tap chair, tap Delete, assert chair gone.
//

import XCTest

final class RoomSceneViewEditTests: XCTestCase {

    func test_tapChair_tapDelete_chairGone() throws {
        let app = XCUIApplication()
        app.launchArguments += ["--ui-test-room-edit-stub"]
        app.launch()

        let scene = app.otherElements.matching(identifier: "roomScene").firstMatch
        XCTAssertTrue(scene.waitForExistence(timeout: 5))
        let center = scene.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        center.tap()

        let deleteBtn = app.buttons["editAction.Delete"]
        XCTAssertTrue(deleteBtn.waitForExistence(timeout: 2))
        deleteBtn.tap()

        center.tap()
        let menuStillExists = deleteBtn.waitForExistence(timeout: 1.5)
        XCTAssertFalse(menuStillExists, "Edit menu still visible after delete — chair was not removed")
    }
}
