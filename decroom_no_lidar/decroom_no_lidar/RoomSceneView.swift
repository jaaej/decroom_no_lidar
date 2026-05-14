//
//  RoomSceneView.swift
//  decroom_no_lidar
//
//  Interactive 3D room viewer using SceneKit.
//  Renders walls, floor, and placed furniture from a RoomPayload.
//

import SwiftUI
import SceneKit

// MARK: - USDZ Model Cache (loaded once, reused across objects)

private final class USDZCache {
    static let shared = USDZCache()
    private var cache: [String: SCNScene] = [:]
    private let lock = NSLock()

    func scene(named name: String) -> SCNScene? {
        lock.lock()
        defer { lock.unlock() }

        if let cached = cache[name] {
            return cached
        }

        guard let url = Bundle.main.url(forResource: name, withExtension: "usdz") else {
            return nil
        }

        do {
            let scene = try SCNScene(url: url, options: [
                .checkConsistency: false
            ])
            cache[name] = scene
            return scene
        } catch {
            print("[USDZCache] Failed to load '\(name)': \(error)")
            return nil
        }
    }
}

// MARK: - SwiftUI Wrapper

struct RoomSceneView: View {
    let payload: RoomPayload
    @State private var lightsEnabled: Bool = true

    var body: some View {
        VStack(spacing: 0) {
            ZStack(alignment: .topTrailing) {
                RoomSceneRepresentable(payload: payload, lightsEnabled: lightsEnabled)
                    .clipShape(RoundedRectangle(cornerRadius: 12))

                // Lighting toggle button
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
            }
        }
    }
}

// MARK: - UIKit Representable

struct RoomSceneRepresentable: UIViewRepresentable {
    let payload: RoomPayload
    let lightsEnabled: Bool

    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = false
        scnView.antialiasingMode = .multisampling2X  // 2X is much cheaper than 4X
        scnView.backgroundColor = UIColor.systemBackground
        scnView.preferredFramesPerSecond = 30  // 30fps is fine for a preview

        // Build scene in background, assign when ready
        let env = payload.envelope
        let objects = payload.objects
        let lights = lightsEnabled

        Task.detached(priority: .userInitiated) {
            let scene = RoomSceneBuilder.buildScene(
                envelope: env, objects: objects, lightsEnabled: lights
            )
            await MainActor.run {
                scnView.scene = scene
            }
        }

        return scnView
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        // Only update lighting — do NOT rebuild the entire scene
        guard let scene = uiView.scene else { return }

        // Remove existing lights and re-add with new setting
        let lightsToRemove = scene.rootNode.childNodes.filter { $0.light != nil }
        for node in lightsToRemove {
            node.removeFromParentNode()
        }

        if lightsEnabled {
            let ambientNode = SCNNode()
            let ambient = SCNLight()
            ambient.type = .ambient
            ambient.intensity = 600
            ambient.color = UIColor.white
            ambientNode.light = ambient
            ambientNode.name = "_light_ambient"
            scene.rootNode.addChildNode(ambientNode)

            let dirNode = SCNNode()
            let dirLight = SCNLight()
            dirLight.type = .directional
            dirLight.intensity = 800
            dirLight.castsShadow = true
            dirLight.shadowMode = .deferred
            dirLight.shadowSampleCount = 4
            dirNode.light = dirLight
            dirNode.eulerAngles = SCNVector3(-Float.pi / 4, Float.pi / 6, 0)
            dirNode.name = "_light_dir"
            scene.rootNode.addChildNode(dirNode)
        } else {
            let ambientNode = SCNNode()
            let ambient = SCNLight()
            ambient.type = .ambient
            ambient.intensity = 1000
            ambient.color = UIColor.white
            ambientNode.light = ambient
            ambientNode.name = "_light_ambient"
            scene.rootNode.addChildNode(ambientNode)
        }
    }
}

// MARK: - Scene Builder (can run off main thread)

enum RoomSceneBuilder {

    static func buildScene(
        envelope env: RoomInfo,
        objects: [RoomObject],
        lightsEnabled: Bool
    ) -> SCNScene {
        print("\n[Decroom] ════════════════════════════════════════════════════")
        print("[Decroom] BUILD SCENE — room=\(String(format: "%.1f", env.width))×\(String(format: "%.1f", env.length))×\(String(format: "%.1f", env.height))m  objects=\(objects.count)  walls=\(env.walls.count)")
        print("[Decroom] ════════════════════════════════════════════════════")

        let scene = SCNScene()

        // Lighting
        if lightsEnabled {
            let ambientNode = SCNNode()
            let ambient = SCNLight()
            ambient.type = .ambient
            ambient.intensity = 600
            ambient.color = UIColor.white
            ambientNode.light = ambient
            scene.rootNode.addChildNode(ambientNode)

            let dirNode = SCNNode()
            let dirLight = SCNLight()
            dirLight.type = .directional
            dirLight.intensity = 800
            dirLight.castsShadow = true
            dirLight.shadowMode = .deferred
            dirLight.shadowSampleCount = 4
            dirNode.light = dirLight
            dirNode.eulerAngles = SCNVector3(-Float.pi / 4, Float.pi / 6, 0)
            scene.rootNode.addChildNode(dirNode)
        } else {
            let ambientNode = SCNNode()
            let ambient = SCNLight()
            ambient.type = .ambient
            ambient.intensity = 1000
            ambient.color = UIColor.white
            ambientNode.light = ambient
            scene.rootNode.addChildNode(ambientNode)
        }

        // Floor
        let floor = SCNFloor()
        floor.reflectivity = 0.05
        floor.firstMaterial?.diffuse.contents = UIColor.systemGray5
        let floorNode = SCNNode(geometry: floor)
        scene.rootNode.addChildNode(floorNode)

        // Walls
        for wall in env.walls {
            if let wallNode = makeWallNode(wall: wall, height: env.height) {
                scene.rootNode.addChildNode(wallNode)
            }
        }

        // Bounding box if no walls
        if env.walls.isEmpty {
            let boundingBox = makeBoundingBox(width: env.width, length: env.length, height: env.height)
            scene.rootNode.addChildNode(boundingBox)
        }

        // Furniture objects
        for obj in objects {
            let objNode = makeObjectNode(obj: obj)
            scene.rootNode.addChildNode(objNode)
        }

        // Camera
        let cameraNode = SCNNode()
        let camera = SCNCamera()
        camera.fieldOfView = 50
        camera.automaticallyAdjustsZRange = true
        cameraNode.camera = camera

        let maxDim = max(env.width, env.length, env.height)
        let cameraDistance = Float(maxDim) * 1.8

        let centerX: Float
        let centerZ: Float
        if !env.walls.isEmpty {
            let allX = env.walls.flatMap { [Float($0.start[0]), Float($0.end[0])] }
            let allZ = env.walls.flatMap { [Float($0.start[1]), Float($0.end[1])] }
            centerX = ((allX.min() ?? 0) + (allX.max() ?? 0)) / 2
            centerZ = ((allZ.min() ?? 0) + (allZ.max() ?? 0)) / 2
        } else {
            centerX = Float(env.width) / 2
            centerZ = Float(env.length) / 2
        }

        cameraNode.position = SCNVector3(
            centerX,
            cameraDistance * 0.6,
            centerZ + cameraDistance * 0.5
        )
        cameraNode.look(at: SCNVector3(
            centerX,
            Float(env.height) * 0.3,
            centerZ
        ))
        scene.rootNode.addChildNode(cameraNode)

        return scene
    }

    // MARK: - Wall Construction

    private static func makeWallNode(wall: Wall, height: Double) -> SCNNode? {
        guard wall.start.count >= 2, wall.end.count >= 2 else { return nil }

        let sx = Float(wall.start[0])
        let sz = Float(wall.start[1])
        let ex = Float(wall.end[0])
        let ez = Float(wall.end[1])

        let dx = ex - sx
        let dz = ez - sz
        let length = sqrt(dx * dx + dz * dz)

        guard length > 0.01 else { return nil }

        let wallThickness: Float = 0.08
        let wallHeight = Float(height)

        let wallGeometry = SCNBox(
            width: CGFloat(length),
            height: CGFloat(wallHeight),
            length: CGFloat(wallThickness),
            chamferRadius: 0
        )

        let material = SCNMaterial()
        material.diffuse.contents = UIColor.systemGray3
        material.transparency = 0.85

        if wall.has_door {
            material.diffuse.contents = UIColor.systemBrown.withAlphaComponent(0.3)
        }
        if wall.has_window {
            material.diffuse.contents = UIColor.systemCyan.withAlphaComponent(0.25)
        }

        wallGeometry.firstMaterial = material

        let wallNode = SCNNode(geometry: wallGeometry)

        let midX = (sx + ex) / 2
        let midZ = (sz + ez) / 2
        wallNode.position = SCNVector3(midX, wallHeight / 2, midZ)

        // SCNBox width extends along local X axis.
        // atan2(dz, dx) = angle from +X to direction vector. Negate for SceneKit CW convention.
        let angle = atan2(dz, dx)
        wallNode.eulerAngles.y = -angle

        if wall.has_door {
            let doorWidth: Float = 0.9
            let doorHeight: Float = min(2.1, wallHeight * 0.85)
            let doorGeometry = SCNBox(
                width: CGFloat(doorWidth),
                height: CGFloat(doorHeight),
                length: CGFloat(wallThickness + 0.02),
                chamferRadius: 0
            )
            let doorMaterial = SCNMaterial()
            doorMaterial.diffuse.contents = UIColor.systemBrown.withAlphaComponent(0.6)
            doorGeometry.firstMaterial = doorMaterial

            let doorNode = SCNNode(geometry: doorGeometry)
            doorNode.position = SCNVector3(0, (doorHeight - wallHeight) / 2, 0)
            wallNode.addChildNode(doorNode)
        }

        if wall.has_window {
            let windowWidth: Float = min(1.2, length * 0.4)
            let windowHeight: Float = min(1.0, wallHeight * 0.35)
            let windowGeometry = SCNBox(
                width: CGFloat(windowWidth),
                height: CGFloat(windowHeight),
                length: CGFloat(wallThickness + 0.02),
                chamferRadius: 0.02
            )
            let windowMaterial = SCNMaterial()
            windowMaterial.diffuse.contents = UIColor.systemCyan.withAlphaComponent(0.4)
            windowMaterial.transparency = 0.5
            windowGeometry.firstMaterial = windowMaterial

            let windowNode = SCNNode(geometry: windowGeometry)
            windowNode.position = SCNVector3(0, wallHeight * 0.15, 0)
            wallNode.addChildNode(windowNode)
        }

        return wallNode
    }

    // MARK: - Bounding Box (when no walls)

    private static func makeBoundingBox(width: Double, length: Double, height: Double) -> SCNNode {
        let container = SCNNode()
        let hw = Float(width) / 2
        let hl = Float(length) / 2
        let h = Float(height)

        let corners: [(Float, Float, Float)] = [
            (-hw, 0, -hl), (hw, 0, -hl), (hw, 0, hl), (-hw, 0, hl),
            (-hw, h, -hl), (hw, h, -hl), (hw, h, hl), (-hw, h, hl),
        ]

        let edges: [(Int, Int)] = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7),
        ]

        for (a, b) in edges {
            let from = corners[a]
            let to = corners[b]
            let edgeNode = makeLine(from: SCNVector3(from.0, from.1, from.2),
                                     to: SCNVector3(to.0, to.1, to.2))
            container.addChildNode(edgeNode)
        }

        return container
    }

    private static func makeLine(from: SCNVector3, to: SCNVector3) -> SCNNode {
        let dx = to.x - from.x
        let dy = to.y - from.y
        let dz = to.z - from.z
        let dist = sqrt(dx*dx + dy*dy + dz*dz)

        let cylinder = SCNCylinder(radius: 0.015, height: CGFloat(dist))
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.systemGray2
        cylinder.firstMaterial = material

        let node = SCNNode(geometry: cylinder)

        node.position = SCNVector3(
            (from.x + to.x) / 2,
            (from.y + to.y) / 2,
            (from.z + to.z) / 2
        )

        let up = SCNVector3(0, 1, 0)
        let dir = SCNVector3(dx, dy, dz)
        let cross = SCNVector3(
            up.y * dir.z - up.z * dir.y,
            up.z * dir.x - up.x * dir.z,
            up.x * dir.y - up.y * dir.x
        )
        let crossLen = sqrt(cross.x*cross.x + cross.y*cross.y + cross.z*cross.z)
        let dot = up.x*dir.x + up.y*dir.y + up.z*dir.z

        if crossLen > 0.0001 {
            let angle = atan2(crossLen, dot)
            let axis = SCNVector3(cross.x/crossLen, cross.y/crossLen, cross.z/crossLen)
            node.rotation = SCNVector4(axis.x, axis.y, axis.z, angle)
        } else if dot < 0 {
            node.eulerAngles.z = .pi
        }

        return node
    }

    // MARK: - Object (Furniture) Construction

    private static func makeObjectNode(obj: RoomObject) -> SCNNode {
        let containerNode = SCNNode()
        containerNode.name = "obj_\(obj.id)"

        containerNode.position = SCNVector3(
            Float(obj.position.x),
            Float(obj.position.y),
            Float(obj.position.z)
        )

        // Backend rotation_y: CCW positive. SceneKit eulerAngles.y: CW positive. Negate.
        containerNode.eulerAngles.y = -Float(obj.rotation_y)

        let assetName = resolveAssetName(type: obj.type, assetId: obj.asset_id)
        let usedUSDZ: Bool
        if let modelNode = loadUSDZModel(named: assetName, targetSize: obj.metadata) {
            containerNode.addChildNode(modelNode)
            usedUSDZ = true
        } else {
            let fallback = makeFallbackGeometry(obj: obj)
            containerNode.addChildNode(fallback)
            usedUSDZ = false
        }

        // ── Per-object placement audit log ─────────────────────────────
        let rotDeg = -Float(obj.rotation_y) * 180.0 / .pi
        print("[Decroom] makeObjectNode: \(obj.type) id=\(obj.id) pos=(\(String(format: "%.3f", obj.position.x)), \(String(format: "%.3f", obj.position.y)), \(String(format: "%.3f", obj.position.z))) rotation_y=\(String(format: "%.4f", obj.rotation_y)) (scene_y=\(String(format: "%.1f", rotDeg))°) asset=\(assetName) usdz=\(usedUSDZ) dims=(\(String(format: "%.3f", obj.metadata.width)), \(String(format: "%.3f", obj.metadata.height)), \(String(format: "%.3f", obj.metadata.length)))")

        return containerNode
    }

    private static func resolveAssetName(type: String, assetId: String) -> String {
        let kenneyName = "kenney_\(type.lowercased())"
        if Bundle.main.url(forResource: kenneyName, withExtension: "usdz") != nil {
            return kenneyName
        }
        return assetId
    }

    // ── Deterministic axis correction lookup ──────────────────────────
    // All Kenney assets are Z-up (authored in Blender). Floor plane is Z=0.
    // SceneKit is Y-up, so ALL Kenney models need a -90° X rotation.
    // This lookup eliminates the old heuristic that compared bbox ratios
    // against (noisy) backend dimensions, which caused "face down" artifacts.
    //
    // Audited 2025-01 via bounding-box measurement of every kenney_*.usdz:
    //   All 13 models have bbMin.z=0 (Blender Z-up convention).
    //   After -π/2 X rotation, models sit correctly on the Y=0 ground plane.
    private static let zUpAssets: Set<String> = [
        "kenney_bed", "kenney_bookshelf", "kenney_chair", "kenney_desk",
        "kenney_dresser", "kenney_lamp_desk", "kenney_lamp_floor",
        "kenney_monitor", "kenney_nightstand", "kenney_plant",
        "kenney_rug", "kenney_sofa", "kenney_table"
    ]

    private static func loadUSDZModel(named name: String, targetSize: ObjectMetadata? = nil) -> SCNNode? {
        // Use cached USDZ scene (avoids re-parsing from disk)
        guard let cachedScene = USDZCache.shared.scene(named: name) else {
            return nil
        }

        // Wrapper node: holds the raw model content, handles axis correction
        let contentNode = SCNNode()
        for child in cachedScene.rootNode.childNodes {
            contentNode.addChildNode(child.clone())
        }

        // ── Deterministic axis correction ──────────────────────────────
        // Asset coordinate system is a property of the asset, not the scan.
        // All Kenney models are Z-up (Blender export). Rotate to Y-up for SceneKit.
        let needsAxisCorrection = zUpAssets.contains(name)

        let (rawMin, rawMax) = contentNode.boundingBox
        let rawW = rawMax.x - rawMin.x
        let rawH = rawMax.y - rawMin.y
        let rawD = rawMax.z - rawMin.z

        print("[Decroom] loadUSDZModel: \(name) raw_bbox=(\(String(format: "%.3f", rawW)), \(String(format: "%.3f", rawH)), \(String(format: "%.3f", rawD))) needsAxisCorrection=\(needsAxisCorrection) (deterministic)")

        if needsAxisCorrection {
            contentNode.eulerAngles.x = -.pi / 2  // Rotate Z-up to Y-up
        }

        // Re-measure after potential axis correction
        // Wrap in a parent to get the effective bounding box
        let wrapperNode = SCNNode()
        wrapperNode.addChildNode(contentNode)

        let (bbMin, bbMax) = wrapperNode.boundingBox
        let modelW = bbMax.x - bbMin.x
        let modelH = bbMax.y - bbMin.y
        let modelL = bbMax.z - bbMin.z

        // Scale model to fit the backend's estimated dimensions
        if let target = targetSize {
            let targetW = Float(max(target.width, 0.1))
            let targetH = Float(max(target.height, 0.1))
            let targetL = Float(max(target.length, 0.1))

            if modelW > 0.001 && modelH > 0.001 && modelL > 0.001 {
                let scaleX = targetW / modelW
                let scaleY = targetH / modelH
                let scaleZ = targetL / modelL
                let scales = [scaleX, scaleY, scaleZ].sorted()
                let uniformScale = scales[1]  // median
                wrapperNode.scale = SCNVector3(uniformScale, uniformScale, uniformScale)
            }
        }

        // Position so bottom sits on y=0
        let (scaledMin, _) = wrapperNode.boundingBox
        let actualMinY = scaledMin.y * wrapperNode.scale.y
        wrapperNode.position = SCNVector3(0, -actualMinY, 0)

        // ── Axis correction audit log ──────────────────────────────────
        if let target = targetSize {
            let finalScale = wrapperNode.scale.x  // uniform
            print("[Decroom] loadUSDZModel: \(name) post_correction_bbox=(\(String(format: "%.3f", modelW)), \(String(format: "%.3f", modelH)), \(String(format: "%.3f", modelL))) target=(\(String(format: "%.3f", target.width)), \(String(format: "%.3f", target.height)), \(String(format: "%.3f", target.length))) scale=\(String(format: "%.3f", finalScale)) yOffset=\(String(format: "%.4f", -actualMinY))")
        }

        return wrapperNode
    }

    private static func makeFallbackGeometry(obj: RoomObject) -> SCNNode {
        let w = CGFloat(max(Float(obj.metadata.width), 0.1))
        let h = CGFloat(max(Float(obj.metadata.height), 0.1))
        let l = CGFloat(max(Float(obj.metadata.length), 0.1))

        let container = SCNNode()
        let color = colorForType(obj.type)

        let box = SCNBox(width: w, height: h, length: l, chamferRadius: min(0.03, w * 0.05))
        let mat = SCNMaterial()
        mat.diffuse.contents = color
        mat.transparency = 0.7
        box.firstMaterial = mat

        let wireframeMat = SCNMaterial()
        wireframeMat.diffuse.contents = UIColor.white.withAlphaComponent(0.2)
        wireframeMat.fillMode = .lines
        box.materials = [mat, wireframeMat]

        let shapeNode = SCNNode(geometry: box)
        shapeNode.position = SCNVector3(0, Float(h) / 2, 0)

        container.addChildNode(shapeNode)

        let text = SCNText(string: obj.type, extrusionDepth: 0.005)
        text.font = UIFont.systemFont(ofSize: 0.06, weight: .medium)
        text.firstMaterial?.diffuse.contents = UIColor.white
        let textNode = SCNNode(geometry: text)
        textNode.position = SCNVector3(-Float(w) / 2, Float(h) + 0.08, 0)
        textNode.scale = SCNVector3(1, 1, 1)
        container.addChildNode(textNode)

        return container
    }

    private static func colorForType(_ type: String) -> UIColor {
        switch type.lowercased() {
        case "bed":
            return .systemBlue
        case "sofa", "couch":
            return .systemTeal
        case "chair":
            return .systemOrange
        case "table", "desk":
            return .systemBrown
        case "bookshelf", "shelf":
            return .systemIndigo
        case "dresser", "cabinet":
            return .systemPurple
        case "lamp":
            return .systemYellow
        case "plant":
            return .systemGreen
        case "rug":
            return .systemPink
        case "monitor", "tv":
            return .darkGray
        default:
            return .systemGray
        }
    }
}

// MARK: - Static Snapshot Renderer (for card thumbnails)

/// Renders a RoomPayload into a UIImage off-screen. Much cheaper than a live
/// SCNView — one render pass, no continuous GPU updates, no per-frame USDZ loads.
enum RoomSnapshotRenderer {
    /// Render a room to a UIImage at the given point size.
    /// Returns nil if rendering fails.
    static func render(payload: RoomPayload, size: CGSize) -> UIImage? {
        let scene = RoomSceneBuilder.buildScene(
            envelope: payload.envelope,
            objects: payload.objects,
            lightsEnabled: true
        )

        let renderer = SCNRenderer(device: nil, options: nil)
        renderer.scene = scene
        renderer.autoenablesDefaultLighting = false

        let scale = UIScreen.main.scale
        let pixelSize = CGSize(width: size.width * scale, height: size.height * scale)

        let image = renderer.snapshot(
            atTime: 0,
            with: pixelSize,
            antialiasingMode: .multisampling2X
        )
        return image
    }
}

/// In-memory cache for room card snapshots, keyed by room ID.
@MainActor
final class RoomSnapshotCache: ObservableObject {
    static let shared = RoomSnapshotCache()

    private var cache: [String: UIImage] = [:]
    private var pending: Set<String> = []

    func image(for room: SavedRoom, size: CGSize) -> UIImage? {
        return cache[room.id]
    }

    func ensureSnapshot(for room: SavedRoom, size: CGSize) {
        guard cache[room.id] == nil, !pending.contains(room.id) else { return }
        pending.insert(room.id)

        let payload = room.payload
        let roomId = room.id
        Task.detached(priority: .utility) {
            let img = RoomSnapshotRenderer.render(payload: payload, size: size)
            await MainActor.run { [weak self] in
                self?.cache[roomId] = img
                self?.pending.remove(roomId)
            }
        }
    }

    func invalidate(roomId: String) {
        cache.removeValue(forKey: roomId)
    }
}

// MARK: - Preview

#Preview {
    RoomSceneView(
        payload: RoomPayload(
            room: RoomInfo(
                id: "preview",
                width_m: 5.0,
                length_m: 4.0,
                height_m: 2.5,
                walls: [
                    Wall(id: "w1", start: [-2.5, -2.0], end: [2.5, -2.0], has_window: false, has_door: true),
                    Wall(id: "w2", start: [2.5, -2.0], end: [2.5, 2.0], has_window: true, has_door: false),
                    Wall(id: "w3", start: [2.5, 2.0], end: [-2.5, 2.0], has_window: false, has_door: false),
                    Wall(id: "w4", start: [-2.5, 2.0], end: [-2.5, -2.0], has_window: true, has_door: false),
                ]
            ),
            objects: [
                RoomObject(
                    id: "obj1",
                    type: "bed",
                    asset_id: "kenney_bed",
                    position: .init(x: 0.5, y: 0, z: 0.0),
                    rotation_y: 0,
                    scale: 1.0,
                    confidence: 0.95,
                    metadata: ObjectMetadata(width: 1.5, length: 2.0, height: 0.6)
                )
            ]
        )
    )
    .frame(height: 400)
}
