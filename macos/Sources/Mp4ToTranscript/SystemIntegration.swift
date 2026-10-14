import AppKit
import UserNotifications

/// End-of-job notifications, only when the app is in the background.
/// Clicking one opens the result.
final class Notifier: NSObject, UNUserNotificationCenterDelegate, @unchecked Sendable {
    static let shared = Notifier()

    func setUp() {
        UNUserNotificationCenter.current().delegate = self
    }

    @MainActor
    func jobEnded(source: String, output: URL?, failure: String?) {
        guard !NSApp.isActive else { return }
        let content = UNMutableNotificationContent()
        if let output {
            content.title = "Transcription terminée"
            content.body = output.lastPathComponent
            content.userInfo = ["path": output.path]
        } else {
            content.title = "La transcription a échoué"
            content.body = "\(source) : \(failure ?? "erreur inconnue")"
        }
        content.sound = .default
        let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        Task {
            let center = UNUserNotificationCenter.current()
            // Asked at the first background completion, when the benefit is obvious.
            do {
                guard try await center.requestAuthorization(options: [.alert, .sound]) else {
                    NSLog("Notifications refusées par l’utilisateur")
                    return
                }
                try await center.add(request)
            } catch {
                NSLog("Notification impossible : \(error)")
            }
        }
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter,
                                didReceive response: UNNotificationResponse) async {
        guard let path = response.notification.request.content.userInfo["path"] as? String else { return }
        await MainActor.run { _ = NSWorkspace.shared.open(URL(filePath: path)) }
    }
}

/// The app icon with a progress bar, in the Dock, while a job runs.
@MainActor
enum DockProgress {
    private static let view = DockTileView()

    static func show(_ fraction: Double?) {
        let tile = NSApp.dockTile
        guard let fraction else {
            if tile.contentView != nil {
                tile.contentView = nil
                tile.display()
            }
            return
        }
        view.fraction = max(0, min(1, fraction))
        if tile.contentView == nil { tile.contentView = view }
        tile.display()
    }
}

private final class DockTileView: NSView {
    var fraction = 0.0

    override func draw(_ dirtyRect: NSRect) {
        NSApp.applicationIconImage?.draw(in: bounds)
        let track = NSRect(x: bounds.width * 0.14, y: bounds.height * 0.1,
                           width: bounds.width * 0.72, height: bounds.height * 0.09)
        let radius = track.height / 2
        NSColor.black.withAlphaComponent(0.55).setFill()
        NSBezierPath(roundedRect: track, xRadius: radius, yRadius: radius).fill()
        var filled = track.insetBy(dx: 2, dy: 2)
        filled.size.width = max(filled.height, filled.width * fraction)
        NSColor.controlAccentColor.setFill()
        NSBezierPath(roundedRect: filled, xRadius: filled.height / 2, yRadius: filled.height / 2).fill()
    }
}
