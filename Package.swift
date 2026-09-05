// swift-tools-version:6.0
import PackageDescription

// ModelCartography's app target is built through project.yml/Xcode (it needs
// the local SwiftSci Interp dependency for the dense-LLM path). This
// manifest exists solely so the model-agnostic core — the ModelAdapter
// protocol and the Attribution/Verification pipeline stages, which are pure
// Foundation with no model-specific code — can be exercised with plain
// `swift test` against a controlled mock adapter. It points straight at the
// real files under Sources/Core and Sources/Pipeline rather than copying
// them, so there's no drift between what's tested and what ships.
let package = Package(
    name: "ModelCartographyCore",
    platforms: [.macOS(.v14)],
    targets: [
        .target(
            name: "CartographyCore",
            path: "Sources",
            exclude: ["App", "Adapters", "Core/SeededRandom.swift"]
        ),
        .testTarget(
            name: "CartographyCoreTests",
            dependencies: ["CartographyCore"]
        ),
    ]
)
