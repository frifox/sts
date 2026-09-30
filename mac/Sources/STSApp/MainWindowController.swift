import AppKit
import WebKit

/// The window: the caption page on top, the console as a collapsible panel
/// at the bottom (like an IDE's terminal).
final class MainWindowController: NSWindowController, NSWindowDelegate {
    let sts = STSProcess()
    private(set) var config = LaunchConfig.load()

    /// Set by the app delegate: the Edit Config button.
    var onEditConfig: (() -> Void)?

    private let splitController = NSSplitViewController()
    private let pageController = PageViewController()
    private let console = ConsoleView()
    private var consoleItem: NSSplitViewItem!
    private var pollGeneration = 0
    private var userStopped = false

    private static let consoleVisibleKey = "consoleVisible"

    init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false
        )
        window.title = "STS"
        window.minSize = NSSize(width: 480, height: 320)
        super.init(window: window)
        window.delegate = self

        let consoleController = NSViewController()
        consoleController.view = console
        splitController.splitView.isVertical = false
        splitController.splitView.dividerStyle = .thin
        let pageItem = NSSplitViewItem(viewController: pageController)
        pageItem.minimumThickness = 150
        consoleItem = NSSplitViewItem(viewController: consoleController)
        consoleItem.canCollapse = true
        consoleItem.minimumThickness = 80
        consoleItem.holdingPriority = .defaultLow + 1  // window resizes go to the page
        splitController.addSplitViewItem(pageItem)
        splitController.addSplitViewItem(consoleItem)
        splitController.splitView.autosaveName = "STSMainSplit"
        consoleItem.isCollapsed = !UserDefaults.standard.bool(forKey: Self.consoleVisibleKey)

        window.contentViewController = splitController
        window.setContentSize(NSSize(width: 1200, height: 800))
        window.setFrameAutosaveName("STSMainWindow")
        if window.frame.origin == .zero { window.center() }

        pageController.onEditConfig = { [weak self] in self?.onEditConfig?() }
        pageController.onStart = { [weak self] in self?.startSTS() }
        pageController.showHome()

        sts.onOutput = { [weak self] text in self?.console.append(text) }
        sts.onExit = { [weak self] status in self?.stsExited(status) }
    }

    required init?(coder: NSCoder) { fatalError() }

    // MARK: sts

    func startSTS() {
        guard !sts.isRunning else { return }
        userStopped = false
        console.append("\n── \(config.commandLine)  (in \(STSEnvironment.dataDirectory.path)) ──\n")
        do {
            try sts.start(config)
        } catch {
            console.append("couldn't start: \(error.localizedDescription)\n")
            pageController.showHome(status: "Couldn't start sts: \(error.localizedDescription)")
            setConsoleVisible(true)
            return
        }
        pageController.showStarting(config.commandLine)
        waitForPage()
    }

    func stopSTS(completion: (() -> Void)? = nil) {
        userStopped = true
        pollGeneration += 1
        sts.stop(completion: completion)
    }

    func restartSTS() {
        stopSTS { [weak self] in self?.startSTS() }
    }

    /// Saves the configuration; a running sts restarts with it.
    func apply(_ newConfig: LaunchConfig) {
        newConfig.save()
        config = newConfig
        if sts.isRunning { restartSTS() }
    }

    private func stsExited(_ status: Int32) {
        pollGeneration += 1
        console.append("── sts exited (\(status)) ──\n")
        if userStopped {
            pageController.showHome(status: "sts stopped.")
        } else {
            pageController.showHome(status: "sts exited with code \(status) - the console shows why.")
            setConsoleVisible(true)
        }
    }

    /// The page is served once sts has its web server up (before the models
    /// finish loading, which the page shows itself): poll until it answers.
    private func waitForPage() {
        let url = config.pageURL
        pollGeneration += 1
        let generation = pollGeneration
        func poll() {
            guard generation == pollGeneration, sts.isRunning else { return }
            var request = URLRequest(url: url)
            request.timeoutInterval = 1
            URLSession.shared.dataTask(with: request) { [weak self] _, response, _ in
                DispatchQueue.main.async {
                    guard let self, generation == self.pollGeneration else { return }
                    if response is HTTPURLResponse {
                        self.pageController.load(url)
                    } else {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5, execute: poll)
                    }
                }
            }.resume()
        }
        poll()
    }

    func reloadPage() {
        pageController.reload()
    }

    // MARK: console

    var isConsoleVisible: Bool { !consoleItem.isCollapsed }

    func setConsoleVisible(_ visible: Bool) {
        guard visible != isConsoleVisible else { return }
        if visible { console.showTail() }
        consoleItem.animator().isCollapsed = !visible
        UserDefaults.standard.set(visible, forKey: Self.consoleVisibleKey)
    }

    func clearConsole() {
        console.clear()
    }

    // MARK: window

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        NSApp.terminate(nil)  // one window: closing it quits (and stops sts)
        return false
    }
}

/// The web view, and - until sts serves its page - the home screen:
/// Edit Config / Start STS, or a spinner while sts starts.
final class PageViewController: NSViewController, WKUIDelegate, WKNavigationDelegate {
    var onEditConfig: (() -> Void)?
    var onStart: (() -> Void)?

    private var webView: WKWebView!
    private let home = NSStackView()
    private let statusLabel = NSTextField(wrappingLabelWithString: "")
    private let spinner = NSProgressIndicator()
    private let buttons = NSStackView()

    override func loadView() {
        let configuration = WKWebViewConfiguration()
        configuration.mediaTypesRequiringUserActionForPlayback = []  // the page streams the voice
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.uiDelegate = self
        webView.navigationDelegate = self
        webView.setValue(false, forKey: "drawsBackground")

        let title = NSTextField(labelWithString: "STS")
        title.font = .systemFont(ofSize: 34, weight: .semibold)
        let subtitle = NSTextField(labelWithString: "Live Russian → English speech translation")
        subtitle.textColor = .secondaryLabelColor
        statusLabel.alignment = .center
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        spinner.style = .spinning
        spinner.controlSize = .small

        let edit = NSButton(title: "Edit Config", target: self, action: #selector(editConfig))
        let start = NSButton(title: "Start STS", target: self, action: #selector(startSTS))
        start.keyEquivalent = "\r"
        for button in [edit, start] {
            button.controlSize = .large
            button.widthAnchor.constraint(greaterThanOrEqualToConstant: 130).isActive = true
        }
        buttons.addArrangedSubview(edit)
        buttons.addArrangedSubview(start)
        buttons.spacing = 12

        home.orientation = .vertical
        home.alignment = .centerX
        home.spacing = 8
        for view in [title, subtitle] as [NSView] { home.addArrangedSubview(view) }
        home.setCustomSpacing(28, after: subtitle)
        for view in [buttons, spinner, statusLabel] as [NSView] { home.addArrangedSubview(view) }
        home.setCustomSpacing(16, after: buttons)

        let container = NSView()
        container.wantsLayer = true
        container.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        for view in [webView!, home] as [NSView] {
            view.translatesAutoresizingMaskIntoConstraints = false
            container.addSubview(view)
        }
        NSLayoutConstraint.activate([
            webView.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            webView.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            webView.topAnchor.constraint(equalTo: container.topAnchor),
            webView.bottomAnchor.constraint(equalTo: container.bottomAnchor),
            home.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            home.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            home.widthAnchor.constraint(lessThanOrEqualTo: container.widthAnchor, constant: -40),
        ])
        view = container
    }

    /// The home screen, with the buttons; `status` says what happened last.
    func showHome(status: String = "") {
        showHomeScreen(status: status, starting: false)
    }

    /// The home screen while sts starts: no buttons, a spinner.
    func showStarting(_ commandLine: String) {
        showHomeScreen(status: "Starting \(commandLine)…", starting: true)
    }

    private func showHomeScreen(status: String, starting: Bool) {
        _ = view
        statusLabel.stringValue = status
        statusLabel.isHidden = status.isEmpty
        buttons.isHidden = starting
        spinner.isHidden = !starting
        starting ? spinner.startAnimation(nil) : spinner.stopAnimation(nil)
        home.isHidden = false
        webView.isHidden = true
        webView.loadHTMLString("", baseURL: nil)  // drop the old page's connection to sts
    }

    func load(_ url: URL) {
        spinner.stopAnimation(nil)
        home.isHidden = true
        webView.isHidden = false
        webView.load(URLRequest(url: url))
    }

    func reload() {
        if !webView.isHidden { webView.reload() }
    }

    @objc private func editConfig() { onEditConfig?() }
    @objc private func startSTS() { onStart?() }

    // MARK: WKUIDelegate - the page's alert/confirm/prompt and mic

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let alert = NSAlert()
        alert.messageText = message
        alert.beginSheetModal(for: view.window!) { _ in completionHandler() }
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let alert = NSAlert()
        alert.messageText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        alert.beginSheetModal(for: view.window!) { completionHandler($0 == .alertFirstButtonReturn) }
    }

    func webView(_ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String,
                 defaultText: String?, initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping (String?) -> Void) {
        let alert = NSAlert()
        alert.messageText = prompt
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 280, height: 24))
        field.stringValue = defaultText ?? ""
        alert.accessoryView = field
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        alert.beginSheetModal(for: view.window!) {
            completionHandler($0 == .alertFirstButtonReturn ? field.stringValue : nil)
        }
    }

    // The page's own "use this browser's mic" - it's our sts, so allow it.
    func webView(_ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin,
                 initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
                 decisionHandler: @escaping (WKPermissionDecision) -> Void) {
        decisionHandler(.grant)
    }

    // target=_blank links open in the default browser.
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = navigationAction.request.url { NSWorkspace.shared.open(url) }
        return nil
    }
}
