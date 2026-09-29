import Foundation

/// How sts is started: `<venv>/bin/sts <arguments>`, run by a login shell
/// (so the user's PATH and profile apply, and quoting works as in Terminal)
/// from a working directory, and where its caption page is served.
struct LaunchConfig: Codable, Equatable {
    var workingDirectory: String
    /// A virtualenv in the working directory, e.g. ".venv-mlx".
    var venv: String
    var arguments: String
    /// Empty: http://127.0.0.1 on the command's --web-port (default 80).
    var pageURL: String

    private static let defaultsKey = "launchConfig"

    static var defaultConfig: LaunchConfig {
        // build.sh records the checkout the app was built from.
        let repo = Bundle.main.object(forInfoDictionaryKey: "STSRepoDirectory") as? String
        return LaunchConfig(
            workingDirectory: repo ?? NSHomeDirectory(),
            venv: ".venv-mlx",
            arguments: "",
            pageURL: ""
        )
    }

    static func load() -> LaunchConfig {
        guard let data = UserDefaults.standard.data(forKey: defaultsKey),
              let config = try? JSONDecoder().decode(LaunchConfig.self, from: data)
        else { return defaultConfig }
        return config
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.defaultsKey)
        }
    }

    var command: String {
        let sts = "\(venv)/bin/sts"
        let quoted = sts.contains(" ") ? "'\(sts)'" : sts
        return arguments.isEmpty ? quoted : "\(quoted) \(arguments)"
    }

    /// The virtualenvs in `directory` that have sts installed.
    static func venvs(in directory: String) -> [String] {
        let path = (directory as NSString).expandingTildeInPath
        let names = (try? FileManager.default.contentsOfDirectory(atPath: path)) ?? []
        return names
            .filter { $0.hasPrefix(".venv") || $0.hasPrefix("venv") }
            .filter { FileManager.default.isExecutableFile(atPath: "\(path)/\($0)/bin/sts") }
            .sorted()
    }

    var expandedWorkingDirectory: String {
        (workingDirectory as NSString).expandingTildeInPath
    }

    /// The page to show: `pageURL` if set, otherwise localhost on the
    /// `--web-port` given in the command.
    var resolvedPageURL: URL? {
        let explicit = pageURL.trimmingCharacters(in: .whitespaces)
        if !explicit.isEmpty { return URL(string: explicit) }
        return URL(string: port == 80 ? "http://127.0.0.1/" : "http://127.0.0.1:\(port)/")
    }

    private var port: Int {
        let words = arguments.split(whereSeparator: \.isWhitespace).map(String.init)
        for (i, word) in words.enumerated() {
            if word.hasPrefix("--web-port="), let p = Int(word.dropFirst("--web-port=".count)) { return p }
            if word == "--web-port", i + 1 < words.count, let p = Int(words[i + 1]) { return p }
        }
        return 80
    }
}
