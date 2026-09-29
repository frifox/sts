import AppKit

/// Edits the launch configuration as a sheet on the main window; saving
/// restarts sts with it.
final class ConfigSheet: NSWindowController {
    private let directoryField = NSTextField()
    private let venvPopup = NSPopUpButton()
    private let argumentsField = NSTextField()
    private let urlField = NSTextField()
    private var onSave: ((LaunchConfig) -> Void)?

    init() {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 600, height: 200),
                              styleMask: [.titled], backing: .buffered, defer: false)
        super.init(window: window)

        argumentsField.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        directoryField.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        urlField.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        urlField.placeholderString = "automatic: http://127.0.0.1 on the command's --web-port"
        argumentsField.placeholderString = "e.g. --web-port 8765 -v"
        directoryField.target = self
        directoryField.action = #selector(directoryChanged)
        for field in [directoryField, argumentsField, urlField] {
            field.lineBreakMode = .byTruncatingTail
            field.cell?.isScrollable = true
            field.widthAnchor.constraint(greaterThanOrEqualToConstant: 420).isActive = true
        }

        let choose = NSButton(title: "Choose…", target: self, action: #selector(chooseDirectory))
        let directoryRow = NSStackView(views: [directoryField, choose])
        let hint = NSTextField(wrappingLabelWithString:
            "Runs <venv>/bin/sts <arguments> in a login shell (zsh -lc) from the working directory, so "
            + "quoting and your PATH work as in Terminal. See `sts --help` for the arguments.")
        hint.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        hint.textColor = .secondaryLabelColor

        let grid = NSGridView(views: [
            [label("Working directory:"), directoryRow],
            [label("Virtualenv:"), venvPopup],
            [label("Arguments:"), argumentsField],
            [label("Page URL:"), urlField],
            [NSGridCell.emptyContentView, hint],
        ])
        grid.column(at: 0).xPlacement = .trailing
        grid.rowAlignment = .firstBaseline
        grid.rowSpacing = 10
        grid.columnSpacing = 8

        let defaults = NSButton(title: "Defaults", target: self, action: #selector(resetDefaults))
        let cancel = NSButton(title: "Cancel", target: self, action: #selector(cancel))
        cancel.keyEquivalent = "\u{1b}"
        let save = NSButton(title: "Save & Restart sts", target: self, action: #selector(save))
        save.keyEquivalent = "\r"
        let spacer = NSView()
        spacer.setContentHuggingPriority(.init(1), for: .horizontal)
        let buttons = NSStackView(views: [defaults, spacer, cancel, save])

        let stack = NSStackView(views: [grid, buttons])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 18
        stack.edgeInsets = NSEdgeInsets(top: 20, left: 20, bottom: 20, right: 20)
        buttons.widthAnchor.constraint(equalTo: grid.widthAnchor).isActive = true
        window.contentView = stack
    }

    required init?(coder: NSCoder) { fatalError() }

    private func label(_ text: String) -> NSTextField { NSTextField(labelWithString: text) }

    func begin(on parent: NSWindow, config: LaunchConfig, onSave: @escaping (LaunchConfig) -> Void) {
        self.onSave = onSave
        fill(config)
        parent.beginSheet(window!)
        window?.makeFirstResponder(argumentsField)
    }

    private func fill(_ config: LaunchConfig) {
        directoryField.stringValue = config.workingDirectory
        fillVenvs(selecting: config.venv)
        argumentsField.stringValue = config.arguments
        urlField.stringValue = config.pageURL
    }

    /// The venvs found in the working directory, plus `selection` even if it
    /// isn't there (so a saved choice isn't silently swapped).
    private func fillVenvs(selecting selection: String) {
        var venvs = LaunchConfig.venvs(in: directoryField.stringValue)
        if !selection.isEmpty, !venvs.contains(selection) { venvs.append(selection) }
        venvPopup.removeAllItems()
        venvPopup.addItems(withTitles: venvs)
        if venvs.contains(selection) { venvPopup.selectItem(withTitle: selection) }
    }

    @objc private func directoryChanged() {
        fillVenvs(selecting: venvPopup.titleOfSelectedItem ?? LaunchConfig.defaultConfig.venv)
    }

    @objc private func chooseDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.directoryURL = URL(fileURLWithPath: (directoryField.stringValue as NSString).expandingTildeInPath)
        panel.beginSheetModal(for: window!) { [weak self] response in
            if response == .OK, let url = panel.url {
                self?.directoryField.stringValue = url.path
                self?.directoryChanged()
            }
        }
    }

    @objc private func resetDefaults() {
        fill(.defaultConfig)
    }

    @objc private func cancel() {
        close()
    }

    @objc private func save() {
        let config = LaunchConfig(
            workingDirectory: directoryField.stringValue.trimmingCharacters(in: .whitespaces),
            venv: venvPopup.titleOfSelectedItem ?? "",
            arguments: argumentsField.stringValue.trimmingCharacters(in: .whitespaces),
            pageURL: urlField.stringValue.trimmingCharacters(in: .whitespaces)
        )
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: config.expandedWorkingDirectory, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            complain("The working directory doesn't exist.")
            return
        }
        guard FileManager.default.isExecutableFile(atPath: "\(config.expandedWorkingDirectory)/\(config.venv)/bin/sts") else {
            complain("No sts in that virtualenv.")
            return
        }
        guard config.resolvedPageURL != nil else {
            complain("The page URL isn't a valid URL.")
            return
        }
        close()
        onSave?(config)
    }

    private func complain(_ message: String) {
        let alert = NSAlert()
        alert.messageText = message
        alert.beginSheetModal(for: window!)
    }

    override func close() {
        if let window, let parent = window.sheetParent { parent.endSheet(window) }
    }
}
