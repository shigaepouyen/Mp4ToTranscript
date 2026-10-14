import AVFoundation
import Foundation

/// A file remembered by bookmark, so a moved or renamed file is still found.
struct StoredFile: Codable, Hashable {
    var name: String
    var path: String
    var bookmark: Data?

    init(_ url: URL) {
        name = url.lastPathComponent
        path = url.path
        bookmark = try? url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
    }

    /// The current location, or nil when the file no longer exists.
    func resolve() -> URL? {
        if let bookmark {
            var stale = false
            if let url = try? URL(resolvingBookmarkData: bookmark, options: [.withoutUI],
                                  relativeTo: nil, bookmarkDataIsStale: &stale),
               FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        return FileManager.default.fileExists(atPath: path) ? URL(filePath: path) : nil
    }
}

struct HistoryEntry: Codable, Identifiable, Hashable {
    enum State: String, Codable {
        case running, done, failed, cancelled, interrupted
    }

    let id: UUID
    var source: StoredFile
    var outputs: [StoredFile] = []
    var date: Date
    /// Length of the recording, when AVFoundation can read it.
    var mediaDuration: Double?
    var processingTime: Double?
    var profile: String
    var language: String
    var state: State
    var message: String?

    var profileTitle: String { Profile(rawValue: profile)?.title ?? profile }
}

@MainActor
@Observable
final class HistoryStore {
    static let shared = HistoryStore()
    static let limit = 200

    /// Newest first.
    private(set) var entries: [HistoryEntry] = []

    private let fileURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appending(path: "Mp4ToTranscript/history.json")

    init() {
        guard let data = try? Data(contentsOf: fileURL),
              let saved = try? JSONDecoder().decode([HistoryEntry].self, from: data) else { return }
        // An entry still « running » at launch belongs to a session that ended abruptly.
        entries = saved.map { entry in
            var entry = entry
            if entry.state == .running { entry.state = .interrupted }
            return entry
        }
    }

    subscript(id: UUID?) -> HistoryEntry? {
        entries.first { $0.id == id }
    }

    func begin(source: URL, options: JobOptions) -> UUID {
        let entry = HistoryEntry(id: UUID(), source: StoredFile(source), date: .now,
                                 profile: options.profile.rawValue,
                                 language: options.language.rawValue, state: .running)
        entries.insert(entry, at: 0)
        entries = Array(entries.prefix(Self.limit))
        save()
        Task {
            let duration = try? await AVURLAsset(url: source).load(.duration)
            if let seconds = duration?.seconds, seconds.isFinite, seconds > 0 {
                update(entry.id) { $0.mediaDuration = seconds }
            }
        }
        return entry.id
    }

    func update(_ id: UUID, _ change: (inout HistoryEntry) -> Void) {
        guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
        change(&entries[index])
        save()
    }

    func remove(_ ids: Set<UUID>) {
        entries.removeAll { ids.contains($0.id) }
        save()
    }

    private func save() {
        do {
            try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(entries).write(to: fileURL, options: .atomic)
        } catch {
            NSLog("Historique non enregistré : \(error)")
        }
    }
}

extension HistoryEntry {
    var formattedDuration: String? {
        guard let mediaDuration else { return nil }
        let pattern: Duration.TimeFormatStyle.Pattern = mediaDuration >= 3600 ? .hourMinuteSecond : .minuteSecond
        return Duration.seconds(mediaDuration.rounded()).formatted(.time(pattern: pattern))
    }

    var stateTitle: String {
        switch state {
        case .running: "En cours"
        case .done: "Terminé"
        case .failed: "Échec"
        case .cancelled: "Arrêté"
        case .interrupted: "Interrompu"
        }
    }

    var stateSymbol: String {
        switch state {
        case .running: "hourglass"
        case .done: "checkmark.circle.fill"
        case .failed: "exclamationmark.triangle.fill"
        case .cancelled: "stop.circle"
        case .interrupted: "exclamationmark.circle"
        }
    }
}
