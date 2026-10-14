import Foundation

/// Runs the bundled Python worker for one file and decodes its JSON-line events.
/// Protocol (unchanged, see `mp4_to_transcript/desktop_worker.py`): one request line
/// on stdin, then `status` / `model` / `done` / `error` events on stdout.
@MainActor
@Observable
final class TranscriptionEngine {
    enum Phase: Equatable {
        case idle
        case running(step: Int, title: String, detail: String)
        case finished(HistoryEntry.ID)
        case failed(String)
        case cancelled
    }

    struct Notice: Identifiable {
        let id = UUID()
        let title: String
        let message: String
    }

    static let shared = TranscriptionEngine()

    private(set) var phase: Phase = .idle {
        didSet {
            CommandState.shared.canStop = isRunning && !cancelling
            updateDock()
        }
    }
    /// Share of the audio already transcribed, read from Whisper's progress bar on stderr.
    private(set) var audioFraction: Double?
    private(set) var source: URL?
    private(set) var modelNotice = ""
    private(set) var startedAt: Date?
    /// The history entry of the current or last job.
    private(set) var entryID: UUID?
    /// Set while the worker downloads the model.
    private(set) var download: DownloadMonitor?
    private var modelID = Preference.defaultModel
    /// Set while asking whether the transcript may be sent to OpenAI.
    var pendingOpenAI: URL? {
        didSet { if pendingOpenAI == nil { pendingOptions = nil } }
    }
    private var pendingOptions: JobOptions?
    var notice: Notice?

    private var process: Process?
    private var stdoutBuffer = Data()
    private var stderrTail = ""
    private(set) var cancelling = false {
        didSet { CommandState.shared.canStop = isRunning && !cancelling }
    }

    static let cacheDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appending(path: "Mp4ToTranscript")

    var isRunning: Bool { if case .running = phase { true } else { false } }

    /// Whole-job progress for the Dock: five steps, refined during download and transcription.
    var overallFraction: Double? {
        guard case let .running(step, _, _) = phase else { return nil }
        if let download, let total = download.total, total > 0 {
            return (1 + Double(download.downloaded) / Double(total)) / 5
        }
        if step == 4, let audioFraction { return (3 + audioFraction) / 5 }
        return Double(step) / 5
    }

    func updateDock() {
        DockProgress.show(overallFraction)
    }

    /// The PyInstaller worker bundled in `Contents/Helpers`, or `MP4_WORKER_PATH` in development.
    static var workerURL: URL? {
        if let path = ProcessInfo.processInfo.environment["MP4_WORKER_PATH"] {
            return URL(filePath: path)
        }
        let url = Bundle.main.bundleURL
            .appending(path: "Contents/Helpers/Mp4ToTranscriptWorker.app/Contents/MacOS/mp4-transcript-worker")
        return FileManager.default.isExecutableFile(atPath: url.path) ? url : nil
    }

    /// Every entry point (window, Dock icon, « Ouvrir avec », Cmd+O, Retranscrire) goes through
    /// here: one file at a time, started at once unless OpenAI needs a confirmation.
    /// `profile` and `language` override the toolbar, for Retranscrire.
    func submit(_ urls: [URL], profile: Profile? = nil, language: Language? = nil) {
        guard let file = urls.first else { return }
        if isRunning {
            notice = Notice(title: "Une transcription est déjà en cours",
                            message: "Arrêtez-la (⌘.) avant d’en lancer une autre.")
        } else if urls.count > 1 {
            notice = Notice(title: "Un fichier à la fois",
                            message: "Déposez un seul fichier audio ou vidéo.")
        } else if !SupportedMedia.accepts(file) {
            notice = Notice(title: "« \(file.lastPathComponent) » n’est pas pris en charge",
                            message: "Formats acceptés : " + SupportedMedia.extensions.sorted().joined(separator: ", ") + ".")
        } else {
            var options = JobOptions.current()
            if let profile { options.profile = profile }
            if let language { options.language = language }
            if !options.sendsToOpenAI {
                start(file, options: options)
            } else if options.apiKey.isEmpty {
                notice = Notice(title: "Clé OpenAI manquante",
                                message: "Renseignez une clé dans les Réglages ou désactivez l’enrichissement OpenAI.")
            } else {
                pendingOpenAI = file
                pendingOptions = options
            }
        }
    }

    func confirmOpenAI() {
        guard let file = pendingOpenAI, let options = pendingOptions else { return }
        pendingOpenAI = nil
        start(file, options: options)
    }

    private func start(_ file: URL, options: JobOptions) {
        guard !isRunning else { return }
        guard let worker = Self.workerURL else {
            notice = Notice(title: "Moteur introuvable",
                            message: "Le moteur de transcription est absent de l’application. Réinstallez-la.")
            return
        }
        source = file
        audioFraction = nil
        modelID = options.model
        modelNotice = ""
        stdoutBuffer = Data()
        stderrTail = ""
        cancelling = false
        startedAt = .now
        phase = .running(step: 0, title: "Démarrage du traitement…", detail: "Préparation du moteur de transcription.")
        entryID = HistoryStore.shared.begin(source: file, options: options)
        Notifier.shared.requestPermissionIfNeeded()

        let request: [String: Any] = [
            "source": file.path,
            "cache_dir": Self.cacheDirectory.path,
            "options": options.json,
        ]

        let process = Process()
        process.executableURL = worker
        var environment = ProcessInfo.processInfo.environment
        environment["MallocStackLogging"] = nil
        environment["MallocStackLoggingNoCompact"] = nil
        process.environment = environment
        let input = Pipe(), output = Pipe(), errors = Pipe()
        process.standardInput = input
        process.standardOutput = output
        process.standardError = errors

        output.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            Task { @MainActor in TranscriptionEngine.shared.receive(stdout: data) }
        }
        errors.fileHandleForReading.readabilityHandler = { handle in
            let text = String(decoding: handle.availableData, as: UTF8.self)
            Task { @MainActor in TranscriptionEngine.shared.receive(stderr: text) }
        }
        process.terminationHandler = { finished in
            let status = finished.terminationStatus
            output.fileHandleForReading.readabilityHandler = nil
            errors.fileHandleForReading.readabilityHandler = nil
            let rest = output.fileHandleForReading.readDataToEndOfFile()
            Task { @MainActor in
                TranscriptionEngine.shared.receive(stdout: rest)
                TranscriptionEngine.shared.terminated(status: status)
            }
        }

        do {
            try process.run()
            self.process = process
            var line = try JSONSerialization.data(withJSONObject: request)
            line.append(0x0A)
            try input.fileHandleForWriting.write(contentsOf: line)
            try input.fileHandleForWriting.close()
        } catch {
            fail("Impossible de lancer le moteur : \(error.localizedDescription)")
        }
    }

    func cancel() {
        guard let process, process.isRunning else { return }
        cancelling = true
        process.terminate()
        let pid = process.processIdentifier
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
            if process.isRunning { kill(pid, SIGKILL) }
        }
    }

    /// Called when the app quits: the worker must not outlive it.
    func terminateNow() {
        guard let process, process.isRunning else { return }
        cancelling = true
        process.terminate()
        for _ in 0..<20 where process.isRunning { usleep(100_000) }
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        record(.cancelled)
    }

    func reset() {
        guard !isRunning else { return }
        phase = .idle
        source = nil
    }

    /// A worker killed during `save_cache` can leave a `*.tmp` next to the cache entries.
    private func removeTemporaryCacheFiles() {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: Self.cacheDirectory, includingPropertiesForKeys: nil)) ?? []
        for file in files where file.pathExtension == "tmp" {
            try? FileManager.default.removeItem(at: file)
        }
    }

    private func receive(stdout data: Data) {
        stdoutBuffer.append(data)
        while let newline = stdoutBuffer.firstIndex(of: 0x0A) {
            let line = stdoutBuffer[stdoutBuffer.startIndex..<newline]
            stdoutBuffer.removeSubrange(stdoutBuffer.startIndex...newline)
            handle(line: Data(line))
        }
    }

    private func receive(stderr text: String) {
        stderrTail = String((stderrTail + text).suffix(12_000))
        // tqdm from mlx_whisper: « 412/1044 [00:02<00:03, …frames/s] ». Last update wins.
        guard case .running(4, _, _) = phase,
              let match = text.matches(of: /(\d+)\/(\d+) \[/).last,
              let done = Double(match.1), let total = Double(match.2), total > 0 else { return }
        audioFraction = min(1, done / total)
        updateDock()
    }

    private func handle(line: Data) {
        guard let event = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
              let kind = event["event"] as? String else { return }
        switch kind {
        case "status":
            if (event["step"] as? Int ?? 0) > 2 { stopDownload() }
            phase = .running(step: event["step"] as? Int ?? 0,
                             title: event["text"] as? String ?? "",
                             detail: event["detail"] as? String ?? "")
        case "model":
            modelNotice = event["text"] as? String ?? ""
            // `desktop_worker.resolve_model` announces a download with this sentence.
            if modelNotice.contains("téléchargement nécessaire") {
                let known = WhisperModel.catalog.first { $0.id == modelID }
                download = DownloadMonitor(model: known ?? WhisperModel(
                    id: modelID, title: modelID, summary: modelID, approximateSize: nil))
            } else {
                stopDownload()
            }
        case "done":
            let outputs = (event["outputs"] as? [String] ?? []).map { URL(filePath: $0) }
            record(.done, outputs: outputs)
            if let entryID { phase = .finished(entryID) }
        case "error":
            fail(event["text"] as? String ?? "Erreur inconnue.")
        default:
            break
        }
    }

    private func stopDownload() {
        download?.stop()
        download = nil
    }

    private func terminated(status: Int32) {
        process = nil
        stopDownload()
        if cancelling {
            phase = .cancelled
            record(.cancelled)
            removeTemporaryCacheFiles()
        } else if isRunning {
            // Exited without a `done` or `error` event: show the end of stderr.
            let tail = stderrTail.split(separator: "\n").suffix(6).joined(separator: "\n")
            fail("Le moteur s’est arrêté de façon inattendue (code \(status)).\n\(tail)")
        }
    }

    private func fail(_ message: String) {
        phase = .failed(message)
        record(.failed, message: message)
    }

    private func record(_ state: HistoryEntry.State, outputs: [URL] = [], message: String? = nil) {
        guard let entryID else { return }
        let elapsed = startedAt.map { Date.now.timeIntervalSince($0) }
        HistoryStore.shared.update(entryID) { entry in
            entry.state = state
            entry.outputs = outputs.map(StoredFile.init)
            entry.processingTime = elapsed
            entry.message = message
        }
        if state == .done || state == .failed {
            Notifier.shared.jobEnded(source: source?.lastPathComponent ?? "",
                                     output: outputs.first, failure: message)
        }
    }
}
