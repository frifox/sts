import Foundation

/// One of sts's command-line options, as build.sh exported it from the
/// bundled sts (Resources/options.json).
struct STSOption: Decodable {
    enum Kind: String, Decodable {
        case text, choice, flag
        case optionalBool = "optional-bool"  // --x / --no-x
    }

    let key: String
    let flag: String
    let kind: Kind
    let choices: [String]
    let type: String  // "int", "float" or "str"
    let `default`: String
    /// What the app passes when the form leaves this at Default, where that
    /// differs from sts's own default (none at the moment: see APP_DEFAULTS in export_options.py).
    let appDefault: String?
    let help: String

    static let all: [STSOption] = {
        guard let url = Bundle.main.url(forResource: "options", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let options = try? JSONDecoder().decode([STSOption].self, from: data)
        else { return [] }
        return options
    }()
}

/// The bundled Python with sts installed, and where sts keeps what it
/// remembers (sts.db, voices/, certs/ - paths relative to where it runs).
enum STSEnvironment {
    static var python: URL {
        Bundle.main.resourceURL!.appendingPathComponent("python/bin/python3")
    }

    static var dataDirectory: URL {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return support.appendingPathComponent("Live Translate", isDirectory: true)
    }
}

/// The options set in the Configuration form; anything not set is left to
/// the app's default for it, if it has one, or else sts's own. Values are what the form shows: text as typed, a
/// choice, or "on"/"off" for switches.
struct LaunchConfig: Codable, Equatable {
    var values: [String: String] = [:]

    private static let defaultsKey = "stsOptions"

    static func load() -> LaunchConfig {
        guard let data = UserDefaults.standard.data(forKey: defaultsKey),
              let config = try? JSONDecoder().decode(LaunchConfig.self, from: data)
        else { return LaunchConfig() }
        return config
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.defaultsKey)
        }
    }

    /// sts's arguments for the set options, in the form's order.
    var arguments: [String] {
        STSOption.all.flatMap { option -> [String] in
            guard let value = values[option.key].flatMap({ $0.isEmpty ? nil : $0 }) ?? option.appDefault
            else { return [] }
            switch option.kind {
            case .text, .choice: return [option.flag, value]
            case .flag: return value == "on" ? [option.flag] : []
            case .optionalBool:
                return [value == "on" ? option.flag : "--no-" + option.flag.dropFirst(2)]
            }
        }
    }

    /// Shown in the console when sts starts.
    var commandLine: String {
        (["sts"] + arguments.map { $0.contains(" ") ? "'\($0)'" : $0 }).joined(separator: " ")
    }

    /// The caption page on this Mac, on the --web-port in use.
    var pageURL: URL {
        let port = values["web_port"].flatMap { Int($0) } ?? 80
        return URL(string: port == 80 ? "http://127.0.0.1/" : "http://127.0.0.1:\(port)/")!
    }
}
