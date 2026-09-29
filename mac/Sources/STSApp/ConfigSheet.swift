import AppKit

/// The Configuration sheet: one row per sts option (generated from the
/// bundled sts's own options), each with sts's default beside it. A row left
/// at Default passes nothing, so sts's default applies.
final class ConfigSheet: NSWindowController {
    private var controls: [String: NSControl] = [:]
    private var onSave: ((LaunchConfig) -> Void)?
    private static let defaultTitle = "Default"

    init() {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 760, height: 620),
                              styleMask: [.titled, .resizable], backing: .buffered, defer: false)
        window.minSize = NSSize(width: 640, height: 320)
        super.init(window: window)

        let grid = NSGridView(numberOfColumns: 3, rows: 0)
        grid.rowSpacing = 8
        grid.columnSpacing = 10
        grid.rowAlignment = .firstBaseline
        for option in STSOption.all {
            let label = NSTextField(labelWithString: option.flag)
            label.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
            let control = makeControl(for: option)
            let fallback = NSTextField(labelWithString: "default: \(option.default)")
            fallback.textColor = .secondaryLabelColor
            fallback.lineBreakMode = .byTruncatingTail
            for view in [label, control, fallback] { view.toolTip = option.help }
            controls[option.key] = control
            grid.addRow(with: [label, control, fallback])
        }
        grid.column(at: 0).xPlacement = .trailing
        grid.column(at: 1).width = 260

        let document = FlippedView()
        grid.translatesAutoresizingMaskIntoConstraints = false
        document.translatesAutoresizingMaskIntoConstraints = false
        document.addSubview(grid)
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.drawsBackground = false
        scroll.documentView = document
        NSLayoutConstraint.activate([
            grid.leadingAnchor.constraint(equalTo: document.leadingAnchor, constant: 20),
            grid.trailingAnchor.constraint(lessThanOrEqualTo: document.trailingAnchor, constant: -20),
            grid.topAnchor.constraint(equalTo: document.topAnchor, constant: 16),
            grid.bottomAnchor.constraint(equalTo: document.bottomAnchor, constant: -16),
            document.widthAnchor.constraint(equalTo: scroll.contentView.widthAnchor),
        ])

        let hint = NSTextField(wrappingLabelWithString:
            "Leave an option empty (or at Default) to use sts's default. Hover an option for what it does. "
            + "Saving restarts sts if it's running. Voices, settings and certificates are kept in "
            + STSEnvironment.dataDirectory.path + ".")
        hint.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        hint.textColor = .secondaryLabelColor

        let reset = NSButton(title: "Reset All to Defaults", target: self, action: #selector(resetAll))
        let cancel = NSButton(title: "Cancel", target: self, action: #selector(cancel))
        cancel.keyEquivalent = "\u{1b}"
        let save = NSButton(title: "Save", target: self, action: #selector(save))
        save.keyEquivalent = "\r"
        let spacer = NSView()
        spacer.setContentHuggingPriority(.init(1), for: .horizontal)
        let buttons = NSStackView(views: [reset, spacer, cancel, save])

        let footer = NSStackView(views: [hint, buttons])
        footer.orientation = .vertical
        footer.alignment = .leading
        footer.spacing = 12
        footer.edgeInsets = NSEdgeInsets(top: 12, left: 20, bottom: 20, right: 20)
        buttons.widthAnchor.constraint(equalTo: footer.widthAnchor, constant: -40).isActive = true
        hint.widthAnchor.constraint(equalTo: footer.widthAnchor, constant: -40).isActive = true

        let content = NSStackView(views: [scroll, NSBox.separator(), footer])
        content.orientation = .vertical
        content.spacing = 0
        for view in content.arrangedSubviews {
            view.widthAnchor.constraint(equalTo: content.widthAnchor).isActive = true
        }
        window.contentView = content
    }

    required init?(coder: NSCoder) { fatalError() }

    private func makeControl(for option: STSOption) -> NSControl {
        switch option.kind {
        case .text:
            let field = NSTextField()
            field.placeholderString = option.default
            field.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
            field.cell?.isScrollable = true
            return field
        case .choice, .flag, .optionalBool:
            let popup = NSPopUpButton()
            popup.addItem(withTitle: Self.defaultTitle)
            popup.menu?.addItem(.separator())
            switch option.kind {
            case .choice: popup.addItems(withTitles: option.choices)
            case .flag: popup.addItems(withTitles: ["on"])
            default: popup.addItems(withTitles: ["on", "off"])
            }
            return popup
        }
    }

    func begin(on parent: NSWindow, config: LaunchConfig, onSave: @escaping (LaunchConfig) -> Void) {
        self.onSave = onSave
        fill(config)
        parent.beginSheet(window!)
    }

    private func fill(_ config: LaunchConfig) {
        for option in STSOption.all {
            let value = config.values[option.key] ?? ""
            switch controls[option.key] {
            case let field as NSTextField: field.stringValue = value
            case let popup as NSPopUpButton:
                popup.selectItem(withTitle: value)
                if popup.selectedItem == nil || value.isEmpty { popup.selectItem(at: 0) }
            default: break
            }
        }
    }

    private var entered: LaunchConfig {
        var config = LaunchConfig()
        for option in STSOption.all {
            let value: String
            switch controls[option.key] {
            case let field as NSTextField: value = field.stringValue.trimmingCharacters(in: .whitespaces)
            case let popup as NSPopUpButton:
                let title = popup.titleOfSelectedItem ?? ""
                value = title == Self.defaultTitle ? "" : title
            default: value = ""
            }
            if !value.isEmpty { config.values[option.key] = value }
        }
        return config
    }

    @objc private func resetAll() {
        fill(LaunchConfig())
    }

    @objc private func cancel() {
        close()
    }

    @objc private func save() {
        let config = entered
        for option in STSOption.all {
            guard let value = config.values[option.key] else { continue }
            let valid = option.type == "int" ? Int(value) != nil
                : option.type == "float" ? Double(value) != nil : true
            if !valid {
                complain("\(option.flag) needs a number (\(option.type == "int" ? "whole" : "decimal")), not “\(value)”.")
                window?.makeFirstResponder(controls[option.key])
                return
            }
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

/// Lays the form out from the top of the scroll view.
private final class FlippedView: NSView {
    override var isFlipped: Bool { true }
}

private extension NSBox {
    static func separator() -> NSBox {
        let box = NSBox()
        box.boxType = .separator
        return box
    }
}
