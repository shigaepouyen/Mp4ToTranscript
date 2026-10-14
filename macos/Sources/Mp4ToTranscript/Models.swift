import Foundation

/// Whisper models in the Hugging Face cache, shared with the command-line tool.
struct WhisperModel: Identifiable, Hashable {
    let id: String          // Hugging Face repository
    let title: String
    let summary: String
    let approximateSize: Int64?

    static let catalog = [
        WhisperModel(id: "mlx-community/whisper-large-v3-mlx", title: "Précision · large v3",
                     summary: "Le plus précis, le plus lent", approximateSize: 3_100_000_000),
        WhisperModel(id: "mlx-community/whisper-large-v3-turbo", title: "Rapide · large v3 turbo",
                     summary: "Précision proche, plusieurs fois plus rapide", approximateSize: 1_600_000_000),
        WhisperModel(id: "mlx-community/whisper-medium-mlx", title: "Léger · medium",
                     summary: "Moins précis, pour les Mac avec peu de mémoire", approximateSize: 1_500_000_000),
    ]

    /// Same resolution as `huggingface_hub`: HF_HUB_CACHE, then HF_HOME/hub, then ~/.cache.
    static var hubCache: URL {
        let environment = ProcessInfo.processInfo.environment
        if let cache = environment["HF_HUB_CACHE"] { return URL(filePath: cache) }
        if let home = environment["HF_HOME"] { return URL(filePath: home).appending(path: "hub") }
        return FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/huggingface/hub")
    }

    var folder: URL {
        Self.hubCache.appending(path: "models--" + id.replacingOccurrences(of: "/", with: "--"))
    }

    /// Bytes on disk, partial downloads included; nil when the model was never downloaded.
    var bytesOnDisk: Int64? {
        guard let files = FileManager.default.enumerator(
            at: folder, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { return nil }
        var total: Int64 = 0
        for case let file as URL in files {
            let values = try? file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if values?.isRegularFile == true { total += Int64(values?.fileSize ?? 0) }
        }
        return total
    }

    /// Catalog models, plus any other Whisper model found in the cache.
    static func installedOrKnown() -> [WhisperModel] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: hubCache.path)) ?? []
        let extra = names
            .filter { $0.hasPrefix("models--") && $0.localizedCaseInsensitiveContains("whisper") }
            .map { String($0.dropFirst("models--".count)).replacingOccurrences(of: "--", with: "/") }
            .filter { repo in !catalog.contains { $0.id == repo } }
            .sorted()
            .map { WhisperModel(id: $0, title: $0.components(separatedBy: "/").last ?? $0,
                                summary: $0, approximateSize: nil) }
        return catalog + extra
    }

    /// Exact download size of the files the worker fetches (`desktop_worker.resolve_model`).
    func downloadSize() async -> Int64? {
        guard let url = URL(string: "https://huggingface.co/api/models/\(id)/tree/main"),
              let (data, _) = try? await URLSession.shared.data(from: url),
              let files = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return nil }
        let wanted: Set<String> = ["config.json", "weights.safetensors", "weights.npz"]
        let sizes = files.filter { wanted.contains($0["path"] as? String ?? "") }
            .compactMap { ($0["size"] as? NSNumber)?.int64Value }
        return sizes.isEmpty ? nil : sizes.reduce(0, +)
    }
}

extension Int64 {
    var fileSize: String { ByteCountFormatter.string(fromByteCount: self, countStyle: .file) }
}

/// Polls the model folder while the worker downloads it: the JSON-line protocol
/// has no byte-level progress, and it stays unchanged.
@MainActor
@Observable
final class DownloadMonitor {
    let model: WhisperModel
    private(set) var downloaded: Int64 = 0
    private(set) var total: Int64?
    private var timer: Timer?

    init(model: WhisperModel) {
        self.model = model
        downloaded = model.bytesOnDisk ?? 0
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            Task { @MainActor [weak self] in self?.poll() }
        }
        Task { [weak self] in
            let size = await model.downloadSize() ?? model.approximateSize
            self?.total = size
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    private func poll() {
        downloaded = model.bytesOnDisk ?? 0
        TranscriptionEngine.shared.updateDock()
    }
}
