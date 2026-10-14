// swift-tools-version: 6.0
// Opens as-is in Xcode; builds with the Command Line Tools alone (`swift build`).
import PackageDescription

let package = Package(
    name: "Mp4ToTranscript",
    platforms: [.macOS("26.0")],
    targets: [
        .executableTarget(name: "Mp4ToTranscript", path: "Sources/Mp4ToTranscript"),
    ]
)
