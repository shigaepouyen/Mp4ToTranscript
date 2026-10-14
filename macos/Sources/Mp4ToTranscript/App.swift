import AppKit
import SwiftUI

@main
struct Mp4ToTranscriptApp: App {
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate

    init() {
        Preference.register()
    }

    var body: some Scene {
        Window("Mp4ToTranscript", id: "main") {
            ContentView(engine: .shared, history: .shared)
        }
        .defaultSize(width: 520, height: 340)
        .windowResizability(.contentMinSize)
        .windowToolbarStyle(.unified)
        .commands { TranscriptionCommands(engine: .shared) }

        Settings {
            SettingsView()
        }
    }
}

/// Receives files dropped on the Dock icon or sent with « Ouvrir avec ».
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        Notifier.shared.setUp()
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        MainActor.assumeIsolated {
            NSApp.activate()
            TranscriptionEngine.shared.submit(urls)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        MainActor.assumeIsolated {
            let engine = TranscriptionEngine.shared
            guard engine.isRunning else { return .terminateNow }
            let alert = NSAlert()
            alert.messageText = "Arrêter la transcription en cours ?"
            alert.informativeText = "« \(engine.source?.lastPathComponent ?? "") » n’est pas terminé. Aucun fichier ne sera créé."
            alert.addButton(withTitle: "Arrêter et quitter")
            alert.addButton(withTitle: "Annuler")
            guard alert.runModal() == .alertFirstButtonReturn else { return .terminateCancel }
            engine.terminateNow()
            return .terminateNow
        }
    }
}

/// A menu key equivalent: AppKit turns a plain Cmd+. into `cancelOperation:`
/// before a SwiftUI button shortcut can see it.
struct TranscriptionCommands: Commands {
    let engine: TranscriptionEngine
    /// Menus do not track `@Observable` state: they observe this mirror instead.
    @ObservedObject private var state = CommandState.shared

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("Ouvrir…") { state.openRequests += 1 }
                .keyboardShortcut("o")
                .disabled(state.canStop)
        }
        CommandGroup(after: .newItem) {
            Button("Arrêter la transcription") { engine.cancel() }
                // `.custom`: keep ⌘. on every keyboard layout (AZERTY would remap it to ⌘;).
                .keyboardShortcut(".", modifiers: .command, localization: .custom)
                .disabled(!state.canStop)
        }
        CommandGroup(replacing: .sidebar) {
            Button("Afficher ou masquer l’historique") { state.historyToggles += 1 }
                .keyboardShortcut("s", modifiers: [.control, .command])
        }
        CommandGroup(replacing: .help) {
            Button("Aide de Mp4ToTranscript") {
                NSWorkspace.shared.open(URL(string: "https://github.com/shigaepouyen/Mp4ToTranscript#readme")!)
            }
        }
    }
}

/// What the menu bar needs to know about the engine, as an `ObservableObject`.
@MainActor
final class CommandState: ObservableObject {
    static let shared = CommandState()
    @Published var canStop = false
    /// Incremented by the menu; the window reacts to the change.
    @Published var openRequests = 0
    @Published var historyToggles = 0
}
