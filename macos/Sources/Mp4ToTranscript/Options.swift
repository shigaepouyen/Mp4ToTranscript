import Foundation

enum Profile: String, CaseIterable, Identifiable {
    case clean
    case raw
    case meetingPlus = "meeting-plus"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .clean: "Texte nettoyé"
        case .raw: "Transcription brute"
        case .meetingPlus: "Compte rendu"
        }
    }

    /// Shown as the menu item's subtitle: short enough to stay on one line.
    var summary: String {
        switch self {
        case .clean: "Retire hésitations et répétitions, sans résumer"
        case .raw: "Le texte reconnu par Whisper, sans retouche"
        case .meetingPlus: "Résumé, décisions et actions, avec le verbatim"
        }
    }

    var symbol: String {
        switch self {
        case .clean: "text.badge.checkmark"
        case .raw: "text.alignleft"
        case .meetingPlus: "list.bullet.clipboard"
        }
    }
}

enum Language: String, CaseIterable, Identifiable {
    case french = "fr"
    case english = "en"
    case automatic = ""

    var id: String { rawValue }

    var title: String {
        switch self {
        case .french: "Français"
        case .english: "Anglais"
        case .automatic: "Automatique"
        }
    }

    var short: String {
        switch self {
        case .french: "FR"
        case .english: "EN"
        case .automatic: "Auto"
        }
    }
}

/// `UserDefaults` keys shared by the window toolbar and the Settings scene (étape 4).
enum Preference {
    static let profile = "profile"
    static let language = "language"
    static let format = "format"
    static let model = "model"
    static let timestamps = "timestamps"
    static let prompt = "prompt"
    static let output = "output"
    static let cloud = "cloud"

    static let defaultModel = "mlx-community/whisper-large-v3-mlx"

    static func register() {
        UserDefaults.standard.register(defaults: [
            profile: Profile.clean.rawValue, language: Language.french.rawValue,
            format: "md", model: defaultModel, timestamps: false,
            prompt: "", output: "", cloud: false,
        ])
    }
}

/// The options object of the worker request, read from the saved preferences.
struct JobOptions {
    var profile: Profile
    var language: Language
    var format: String
    var model: String
    var timestamps: Bool
    var prompt: String
    var output: String
    var cloud: Bool
    var apiKey: String

    /// OpenAI only applies to the « Compte rendu » profile, as in the Qt app.
    var sendsToOpenAI: Bool { cloud && profile == .meetingPlus }

    static func current() -> JobOptions {
        let defaults = UserDefaults.standard
        return JobOptions(
            profile: Profile(rawValue: defaults.string(forKey: Preference.profile) ?? "") ?? .clean,
            language: Language(rawValue: defaults.string(forKey: Preference.language) ?? "fr") ?? .french,
            format: defaults.string(forKey: Preference.format) ?? "md",
            model: defaults.string(forKey: Preference.model) ?? Preference.defaultModel,
            timestamps: defaults.bool(forKey: Preference.timestamps),
            prompt: defaults.string(forKey: Preference.prompt) ?? "",
            output: defaults.string(forKey: Preference.output) ?? "",
            cloud: defaults.bool(forKey: Preference.cloud),
            apiKey: defaults.bool(forKey: Preference.cloud)
                ? (Keychain.read().nilIfEmpty ?? ProcessInfo.processInfo.environment["OPENAI_API_KEY"] ?? "")
                : "")
    }

    var json: [String: Any] {
        ["model": model, "language": language.rawValue, "prompt": prompt,
         "format": format, "mode": profile.rawValue, "timestamps": timestamps,
         "output": output, "cloud": sendsToOpenAI, "api_key": sendsToOpenAI ? apiKey : ""]
    }
}

enum SupportedMedia {
    /// Same list as `cli.SUPPORTED_EXTENSIONS` and the Info.plist document types.
    static let extensions: Set<String> = ["aac", "flac", "m4a", "m4v", "mkv", "mov", "mp3", "mp4",
                                          "mpeg", "mpga", "ogg", "opus", "wav", "webm", "wma"]

    static func accepts(_ url: URL) -> Bool {
        url.isFileURL && extensions.contains(url.pathExtension.lowercased())
    }
}

extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
