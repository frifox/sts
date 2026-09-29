// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "STSApp",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "STSApp", path: "Sources/STSApp"),
    ]
)
