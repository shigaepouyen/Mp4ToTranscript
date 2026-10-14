import AppKit
import QuickLook
import SwiftUI
import UniformTypeIdentifiers

/// Fenêtre unique : barre d'outils (Profil, Langue, Historique), historique repliable
/// et trois états, dépôt → progression → résultat.
struct ContentView: View {
    let engine: TranscriptionEngine
    let history: HistoryStore
    @AppStorage(Preference.profile) private var profile = Profile.clean
    @AppStorage(Preference.language) private var language = Language.french
    @State private var columns = NavigationSplitViewVisibility.detailOnly
    @State private var selection: HistoryEntry.ID?
    @State private var previewURL: URL?
    @State private var importing = false
    @State private var targeted = false
    @ObservedObject private var commands = CommandState.shared
    @Environment(\.colorSchemeContrast) private var contrast

    var body: some View {
        NavigationSplitView(columnVisibility: $columns) {
            HistorySidebar(history: history, selection: $selection, actions: actions,
                           busy: engine.isRunning)
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 340)
                .toolbar(removing: .sidebarToggle)
        } detail: {
            stage
                .frame(minWidth: 440, maxWidth: .infinity, minHeight: 300, maxHeight: .infinity)
                .dropDestination(for: URL.self) { urls, _ in
                    guard !engine.isRunning else { return false }
                    engine.submit(urls)
                    return true
                } isTargeted: { targeted = $0 }
        }
        .toolbar { toolbar }
        .quickLookPreview($previewURL)
        .onChange(of: engine.entryID) { selection = nil }
        .onChange(of: commands.openRequests) { importing = true }
        .onChange(of: commands.historyToggles) { toggleHistory() }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.audio, .movie]) { result in
            if case let .success(file) = result { engine.submit([file]) }
        }
        .alert(engine.notice?.title ?? "", isPresented: noticeShown, presenting: engine.notice) { _ in
            Button("OK") {}
        } message: { notice in
            Text(notice.message)
        }
        .confirmationDialog("Envoyer le texte à OpenAI ?", isPresented: openAIShown) {
            Button("Transcrire et envoyer") { engine.confirmOpenAI() }
            // Sending data to a third party must never be the Return-key default.
            Button("Annuler", role: .cancel) { engine.pendingOpenAI = nil }
                .keyboardShortcut(.defaultAction)
        } message: {
            Text("L’enrichissement OpenAI est activé : le texte transcrit de « \(engine.pendingOpenAI?.lastPathComponent ?? "") » sera envoyé à OpenAI, et des frais d’API peuvent s’appliquer.")
        }
    }

    private var actions: EntryActions {
        EntryActions(
            preview: { previewURL = $0 },
            retranscribe: { entry in
                guard let source = entry.source.resolve() else {
                    engine.notice = .init(title: "Fichier source introuvable",
                                          message: "« \(entry.source.name) » a été supprimé ou n’est plus accessible.")
                    return
                }
                // Same profile and language as the original transcription.
                engine.submit([source], profile: Profile(rawValue: entry.profile),
                              language: Language(rawValue: entry.language))
            },
            remove: { ids in
                if let selection, ids.contains(selection) { self.selection = nil }
                history.remove(ids)
            })
    }

    // MARK: - Barre d'outils

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            Button(action: toggleHistory) {
                Label("Historique", systemImage: "sidebar.left")
            }
            .help("Afficher ou masquer l’historique")
        }
        ToolbarItemGroup(placement: .primaryAction) {
            Menu {
                ForEach(Profile.allCases) { option in
                    Toggle(isOn: Binding(get: { profile == option }, set: { if $0 { profile = option } })) {
                        // Two Text views: the second becomes the menu item's subtitle.
                        Text(option.title)
                        Text(option.summary)
                    }
                }
            } label: {
                Text(profile.title)
            }
            .disabled(engine.isRunning)
            .help("Profil : \(profile.summary)")
            .accessibilityLabel("Profil : \(profile.title)")

            Menu {
                Picker("Langue", selection: $language) {
                    ForEach(Language.allCases) { language in
                        Text(language.title).tag(language)
                    }
                }
                .pickerStyle(.inline)
                .labelsHidden()
            } label: {
                Text(language.short)
            }
            .disabled(engine.isRunning)
            .help("Langue de l’enregistrement : \(language.title)")
            .accessibilityLabel("Langue : \(language.title)")
        }
    }

    /// Opening the sidebar widens a narrow window to the left, so the result card keeps its width.
    private func toggleHistory() {
        let opening = columns == .detailOnly
        withAnimation { columns = opening ? .all : .detailOnly }
        // From the menu bar the window may not be key: fall back to the visible main window.
        guard opening, let window = NSApp.keyWindow ?? NSApp.mainWindow
                ?? NSApp.windows.first(where: { $0.isVisible && $0.canBecomeMain }) else { return }
        let wanted: CGFloat = 780
        guard window.frame.width < wanted else { return }
        var frame = window.frame
        let visible = window.screen?.visibleFrame ?? frame
        frame.origin.x = max(visible.minX, frame.maxX - wanted)
        frame.size.width = min(wanted, visible.maxX - frame.origin.x)
        window.setFrame(frame, display: true, animate: true)
    }

    // MARK: - États

    @ViewBuilder
    private var stage: some View {
        if case let .running(step, title, detail) = engine.phase {
            progress(step: step, title: title, detail: detail)
        } else if let entry = history[selection] {
            entryView(entry)
        } else {
            switch engine.phase {
            case .idle, .running:
                dropZone
            case let .finished(id):
                if let entry = history[id] { entryView(entry) } else { dropZone }
            case let .failed(message):
                ending("exclamationmark.triangle", "La transcription a échoué", message)
            case .cancelled:
                ending("stop.circle", "Transcription arrêtée", "Aucun fichier n’a été créé.")
            }
        }
    }

    @ViewBuilder
    private func entryView(_ entry: HistoryEntry) -> some View {
        VStack(spacing: 14) {
            if entry.state == .done {
                ResultCard(entry: entry, actions: actions, busy: engine.isRunning)
            } else {
                EntryStatus(entry: entry, actions: actions, busy: engine.isRunning)
            }
            Button("Nouveau fichier…") {
                selection = nil
                engine.reset()
                importing = true
            }
            .buttonStyle(.link)
            .font(.callout)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .overlay { if targeted { dropHighlight } }
    }

    private var dropZone: some View {
        VStack(spacing: 12) {
            Image(systemName: "waveform")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(targeted ? Color.accentColor : .secondary)
                .symbolEffect(.bounce, value: targeted)
            Text("Déposez un fichier audio ou vidéo")
                .font(.title3.weight(.medium))
            Text("La transcription démarre aussitôt.")
                .foregroundStyle(.secondary)
            Button("Choisir un fichier…") { importing = true }
                .help("Choisir un fichier audio ou vidéo (⌘O)")
                .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background {
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(targeted ? Color.accentColor
                                  : Color.secondary.opacity(contrast == .increased ? 1 : 0.35),
                              style: StrokeStyle(lineWidth: 1.5, dash: [6, 4]))
                .fill(targeted ? Color.accentColor.opacity(0.08) : .clear)
        }
        .padding(20)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
        .accessibilityHint("Déposez un fichier ou choisissez-en un pour lancer la transcription.")
    }

    private func progress(step: Int, title: String, detail: String) -> some View {
        VStack(spacing: 10) {
            Text(engine.source?.lastPathComponent ?? "")
                .font(.headline)
                .lineLimit(1)
                .truncationMode(.middle)
            if let download = engine.download {
                downloadProgress(download)
            } else if step == 4, let fraction = engine.audioFraction {
                ProgressView(value: fraction)
                    .accessibilityLabel("Transcription de l’audio")
                    .accessibilityValue("\(Int(fraction * 100)) %")
                Text("\(Int(fraction * 100)) % de l’audio transcrit")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            } else {
                ProgressView(value: Double(step), total: 5)
                    .accessibilityLabel("Étape \(step) sur 5")
            }
            Text(engine.cancelling ? "Arrêt en cours…" : title)
            Text(detail)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 6) {
                if !engine.modelNotice.isEmpty {
                    Text(engine.modelNotice)
                    Text("·")
                }
                if let startedAt = engine.startedAt {
                    Text(startedAt, style: .timer).monospacedDigit()
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            Button("Arrêter", role: .cancel) { engine.cancel() }
                .help("Arrêter la transcription (⌘.)")
                .disabled(engine.cancelling)
                .padding(.top, 4)
        }
        .padding(28)
    }

    @ViewBuilder
    private func downloadProgress(_ download: DownloadMonitor) -> some View {
        if let total = download.total, total > 0 {
            ProgressView(value: Double(min(download.downloaded, total)), total: Double(total))
                .accessibilityLabel("Téléchargement du modèle")
                .accessibilityValue("\(download.downloaded.fileSize) sur \(total.fileSize)")
            Text("\(download.downloaded.fileSize) sur \(total.fileSize)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        } else {
            ProgressView()
                .progressViewStyle(.linear)
                .accessibilityLabel("Téléchargement du modèle")
            Text("\(download.downloaded.fileSize) téléchargés")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    private func ending(_ symbol: String, _ title: String, _ message: String) -> some View {
        VStack(spacing: 10) {
            Image(systemName: symbol)
                .font(.system(size: 32))
                .foregroundStyle(.secondary)
            Text(title).font(.headline)
            ScrollView {
                Text(message)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity)
            }
            .frame(maxHeight: 110)
            .fixedSize(horizontal: false, vertical: true)
            Button("Choisir un autre fichier…") { engine.reset(); importing = true }
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .overlay { if targeted { dropHighlight } }
    }

    private var dropHighlight: some View {
        RoundedRectangle(cornerRadius: 16)
            .strokeBorder(Color.accentColor, lineWidth: 2)
            .padding(8)
            .allowsHitTesting(false)
    }

    private var noticeShown: Binding<Bool> {
        Binding(get: { engine.notice != nil }, set: { if !$0 { engine.notice = nil } })
    }

    private var openAIShown: Binding<Bool> {
        Binding(get: { engine.pendingOpenAI != nil }, set: { if !$0 { engine.pendingOpenAI = nil } })
    }
}
