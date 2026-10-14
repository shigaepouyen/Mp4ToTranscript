import AppKit
import SwiftUI

/// Réglages (⌘,) : appliqués immédiatement, sans bouton de validation.
struct SettingsView: View {
    var body: some View {
        TabView {
            Tab("Général", systemImage: "gearshape") { GeneralSettings() }
            Tab("Modèles", systemImage: "cpu") { ModelSettings(engine: .shared) }
            Tab("Enrichissement", systemImage: "sparkles") { EnrichmentSettings() }
        }
        .frame(width: 560)
    }
}

private struct GeneralSettings: View {
    @AppStorage(Preference.format) private var format = "md"
    @AppStorage(Preference.output) private var output = ""
    @AppStorage(Preference.timestamps) private var timestamps = false
    @AppStorage(Preference.prompt) private var prompt = ""
    @State private var choosingFolder = false

    private static let chooseTag = "\u{0}choose"

    var body: some View {
        Form {
            Picker("Format :", selection: $format) {
                Text("Markdown (.md)").tag("md")
                Text("Texte (.txt)").tag("txt")
                Text("Les deux").tag("both")
            }
            .fixedSize()

            Picker("Destination :", selection: destination) {
                Text("Dossier « transcripts » à côté du fichier").tag("")
                if !output.isEmpty {
                    Label(URL(filePath: output).lastPathComponent, systemImage: "folder").tag(output)
                }
                Divider()
                Text("Choisir un dossier…").tag(Self.chooseTag)
            }
            .fixedSize()
            .fileImporter(isPresented: $choosingFolder, allowedContentTypes: [.folder]) { result in
                if case let .success(folder) = result { output = folder.path }
            }
            if !output.isEmpty {
                LabeledContent("") {
                    Text((output as NSString).abbreviatingWithTildeInPath)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: 340, alignment: .leading)
                        .help(output)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }

            LabeledContent("Transcription :") {
                Toggle("Inclure les repères de temps", isOn: $timestamps)
            }

            TextField("Contexte :", text: $prompt,
                      prompt: Text("Noms propres, jargon, sujet de la réunion…"), axis: .vertical)
                .lineLimit(2...4)
            LabeledContent("") {
                Text("Aide Whisper à reconnaître les noms et le vocabulaire. S’applique à toutes les transcriptions.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .formStyle(.columns)
        .padding(24)
    }

    /// The « Choisir un dossier… » item opens the folder picker instead of being selected.
    private var destination: Binding<String> {
        Binding(get: { output }, set: { value in
            if value == Self.chooseTag { choosingFolder = true } else { output = value }
        })
    }
}

private struct ModelSettings: View {
    let engine: TranscriptionEngine
    @AppStorage(Preference.model) private var defaultModel = Preference.defaultModel
    @State private var models: [(model: WhisperModel, size: Int64?)] = []
    @State private var deleting: WhisperModel?

    var body: some View {
        Form {
            Picker("Modèle par défaut :", selection: $defaultModel) {
                ForEach(WhisperModel.catalog) { model in
                    Text(model.title).tag(model.id)
                }
            }
            .fixedSize()
            LabeledContent("") {
                Text(WhisperModel.catalog.first { $0.id == defaultModel }?.summary ?? "")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            LabeledContent("Sur ce Mac :") {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(models, id: \.model.id) { item in
                        row(item.model, size: item.size)
                    }
                }
            }

            LabeledContent("") {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Un modèle absent est téléchargé à la première transcription. Le dossier est partagé avec l’outil en ligne de commande.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Button("Afficher dans le Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([WhisperModel.hubCache])
                    }
                    .buttonStyle(.link)
                    .font(.caption)
                }
            }
        }
        .formStyle(.columns)
        .padding(24)
        .onAppear(perform: refresh)
        .confirmationDialog("Supprimer « \(deleting?.title ?? "") » ?", isPresented: deletingShown,
                            presenting: deleting) { model in
            Button("Supprimer", role: .destructive) { delete(model) }
            Button("Annuler", role: .cancel) {}
        } message: { model in
            Text("\(model.bytesOnDisk?.fileSize ?? "") seront libérés. Le modèle sera de nouveau téléchargé s’il est utilisé.")
        }
    }

    private func row(_ model: WhisperModel, size: Int64?) -> some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.title)
                Text(size.map { $0.fileSize } ?? "Non téléchargé"
                     + (model.approximateSize.map { " · environ \($0.fileSize)" } ?? ""))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Spacer(minLength: 16)
            if size != nil {
                Button("Supprimer…") { deleting = model }
                    .disabled(engine.isRunning)
                    .help(engine.isRunning ? "Impossible pendant une transcription" : "Supprimer ce modèle du Mac")
            }
        }
        .frame(width: 330)
        .accessibilityElement(children: .contain)
    }

    private func refresh() {
        models = WhisperModel.installedOrKnown().map { ($0, $0.bytesOnDisk) }
    }

    private func delete(_ model: WhisperModel) {
        try? FileManager.default.removeItem(at: model.folder)
        refresh()
    }

    private var deletingShown: Binding<Bool> {
        Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })
    }
}

private struct EnrichmentSettings: View {
    @AppStorage(Preference.cloud) private var cloud = false
    @State private var key = ""
    @State private var stored = false

    var body: some View {
        Form {
            LabeledContent("OpenAI :") {
                Toggle("Enrichir les comptes rendus avec OpenAI", isOn: $cloud)
            }
            SecureField("Clé d’API :", text: $key, prompt: Text("sk-…"))
                .disabled(!cloud)
                .onChange(of: key) { stored = Keychain.save(key) && !key.isEmpty }
            LabeledContent("") {
                VStack(alignment: .leading, spacing: 6) {
                    if stored {
                        Label("Clé enregistrée dans le Trousseau", systemImage: "key.fill")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Text("Ne concerne que le profil Compte rendu. Le texte transcrit est alors envoyé à OpenAI et des frais d’API peuvent s’appliquer. Une confirmation est demandée avant chaque envoi. Sans cette option, le compte rendu est produit sur votre Mac, par règles.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .formStyle(.columns)
        .padding(24)
        .onAppear {
            // Read only when enabled: an ad hoc build may trigger a Keychain prompt.
            if cloud {
                key = Keychain.read()
                stored = !key.isEmpty
            }
        }
        .onChange(of: cloud) {
            if cloud, key.isEmpty {
                key = Keychain.read()
                stored = !key.isEmpty
            }
        }
    }
}
