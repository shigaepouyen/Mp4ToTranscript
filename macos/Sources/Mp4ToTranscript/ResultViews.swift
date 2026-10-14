import AppKit
import SwiftUI

/// What can be done with a history entry, from the result card or the sidebar.
struct EntryActions {
    var preview: (URL) -> Void
    var retranscribe: (HistoryEntry) -> Void
    var remove: (Set<HistoryEntry.ID>) -> Void

    func open(_ url: URL) { NSWorkspace.shared.open(url) }
    func reveal(_ url: URL) { NSWorkspace.shared.activateFileViewerSelecting([url]) }
}

/// The result of a finished job: the file, Coup d'œil, Ouvrir, Afficher dans le Finder.
/// No built-in text preview: transcripts are read in another app.
struct ResultCard: View {
    let entry: HistoryEntry
    let actions: EntryActions
    let busy: Bool

    var body: some View {
        let outputs = entry.outputs.compactMap { $0.resolve() }
        VStack(alignment: .leading, spacing: 16) {
            if let primary = outputs.first {
                HStack(spacing: 14) {
                    Image(nsImage: NSWorkspace.shared.icon(forFile: primary.path))
                        .resizable()
                        .frame(width: 56, height: 56)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(primary.lastPathComponent)
                            .font(.headline)
                            .lineLimit(2)
                            .truncationMode(.middle)
                        Text(entry.source.name)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Text(entry.details)
                    }
                    .font(.callout)
                    .foregroundStyle(.secondary)
                }
                HStack {
                    Button("Coup d’œil", systemImage: "eye") { actions.preview(primary) }
                        .keyboardShortcut(.space, modifiers: [])
                        .help("Aperçu rapide (Espace)")
                    Button("Ouvrir", systemImage: "arrow.up.forward.app") { actions.open(primary) }
                        .keyboardShortcut(.defaultAction)
                    Button("Afficher dans le Finder", systemImage: "folder") { actions.reveal(primary) }
                }
                ForEach(outputs.dropFirst(), id: \.self) { other in
                    HStack(spacing: 6) {
                        Text("Aussi : \(other.lastPathComponent)")
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(.secondary)
                        Button("Ouvrir") { actions.open(other) }
                            .buttonStyle(.link)
                    }
                    .font(.callout)
                }
            } else {
                Label("« \(entry.outputs.first?.name ?? entry.source.name) » est introuvable : il a été supprimé ou n’est plus accessible.",
                      systemImage: "questionmark.folder")
                    .foregroundStyle(.secondary)
                Button("Retranscrire", systemImage: "arrow.clockwise") { actions.retranscribe(entry) }
                    .help("Relancer avec le profil d’origine : \(entry.profileTitle)")
                    .disabled(busy)
            }
        }
        .padding(20)
        .frame(maxWidth: 440, alignment: .leading)
        .background(.background.secondary, in: .rect(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(.separator))
    }
}

/// A history entry that did not produce a file: failed, stopped or interrupted.
struct EntryStatus: View {
    let entry: HistoryEntry
    let actions: EntryActions
    let busy: Bool

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: entry.stateSymbol)
                .font(.system(size: 30))
                .foregroundStyle(.secondary)
            Text(entry.source.name)
                .font(.headline)
                .lineLimit(1)
                .truncationMode(.middle)
            Text("\(entry.stateTitle) · \(entry.details)")
                .foregroundStyle(.secondary)
            if let message = entry.message {
                ScrollView {
                    Text(message)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity)
                }
                .frame(maxHeight: 90)
                .fixedSize(horizontal: false, vertical: true)
            }
            Button("Retranscrire", systemImage: "arrow.clockwise") { actions.retranscribe(entry) }
                    .help("Relancer avec le profil d’origine : \(entry.profileTitle)")
                .disabled(busy)
                .padding(.top, 4)
        }
    }
}

struct HistorySidebar: View {
    let history: HistoryStore
    @Binding var selection: HistoryEntry.ID?
    let actions: EntryActions
    let busy: Bool

    var body: some View {
        List(selection: $selection) {
            ForEach(history.entries) { entry in
                HistoryRow(entry: entry)
                    .tag(entry.id)
                    .contextMenu { menu(for: entry) }
            }
        }
        .onDeleteCommand {
            if let selection { actions.remove([selection]) }
        }
        .overlay {
            if history.entries.isEmpty {
                VStack(spacing: 6) {
                    Image(systemName: "clock").font(.title2)
                    Text("Aucune transcription").font(.headline)
                    Text("Les fichiers traités apparaîtront ici.")
                        .font(.callout)
                        .multilineTextAlignment(.center)
                }
                .foregroundStyle(.secondary)
                .padding()
            }
        }
    }

    @ViewBuilder
    private func menu(for entry: HistoryEntry) -> some View {
        let output = entry.outputs.lazy.compactMap { $0.resolve() }.first
        Button("Coup d’œil", systemImage: "eye") { output.map(actions.preview) }
            .disabled(output == nil)
        Button("Ouvrir", systemImage: "arrow.up.forward.app") { output.map(actions.open) }
            .disabled(output == nil)
        Button("Afficher dans le Finder", systemImage: "folder") {
            if let url = output ?? entry.source.resolve() { actions.reveal(url) }
        }
        Divider()
        Button("Retranscrire", systemImage: "arrow.clockwise") {
            actions.retranscribe(entry)
        }
        .disabled(busy || entry.state == .running)
        Divider()
        Button("Retirer de l’historique", systemImage: "minus.circle") { actions.remove([entry.id]) }
            .disabled(entry.state == .running)
    }
}

struct HistoryRow: View {
    let entry: HistoryEntry

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Group {
                if entry.state == .running {
                    ProgressView().controlSize(.mini)
                } else {
                    Image(systemName: entry.stateSymbol)
                        .foregroundStyle(tint)
                }
            }
            .frame(width: 16)
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.source.name)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(entry.date, format: .dateTime.day().month(.abbreviated).hour().minute())
                Text(entry.details)
            }
            .font(.callout)
            .foregroundStyle(.primary)
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(entry.source.name), \(entry.stateTitle)")
        .accessibilityValue(entry.details)
    }

    private var tint: Color {
        switch entry.state {
        case .done: .green
        case .failed: .orange
        default: .secondary
        }
    }
}

extension HistoryEntry {
    /// « 1:31 · Texte nettoyé », the duration when known.
    var details: String {
        [formattedDuration, profileTitle].compactMap { $0 }.joined(separator: " · ")
    }
}
