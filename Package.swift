// swift-tools-version: 5.10
// CaptureInterp: a single-file app plus the vendored RifeMetal package (GPU frame interpolation).
import PackageDescription
let package = Package(
    name: "CaptureInterp",
    platforms: [.macOS("26.0")],
    dependencies: [.package(path: "ThirdParty/RifeMetal")],
    targets: [.executableTarget(name: "CaptureInterp",
                                dependencies: [.product(name: "RifeMetal", package: "RifeMetal")],
                                path: ".", sources: ["main.swift"],
                                swiftSettings: [.unsafeFlags(["-swift-version", "5"])])]
)
