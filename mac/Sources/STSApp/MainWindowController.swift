import AppKit
import WebKit

/// The window: the caption page on top, the console as a collapsible panel
/// at the bottom (like an IDE's terminal).
final class MainWindowController: NSWindowController, NSWindowDelegate {
    let sts = STSProcess()
    private(set) var config = LaunchConfig.load()

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

        sts.onOutput = { [weak self] text in self?.console.append(text) }
        sts.onExit = { [weak self] status in self?.stsExited(status) }
    }

    required init?(coder: NSCoder) { fatalError() }

    // MARK: sts

    func startSTS() {
        guard !sts.isRunning else { return }
        userStopped = false
        console.append("\n── \(config.command)  (in \(config.expandedWorkingDirectory)) ──\n")
        do {
            try sts.start(config)
        } catch {
            console.append("couldn't start: \(error.localizedDescription)\n")
            pageController.showStatus("Couldn't start sts", detail: error.localizedDescription)
            setConsoleVisible(true)
            return
        }
        pageController.showStatus("Starting sts…", detail: config.command, busy: true)
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

    func apply(_ newConfig: LaunchConfig) {
        newConfig.save()
        config = newConfig
        restartSTS()
    }

    private func stsExited(_ status: Int32) {
        pollGeneration += 1
        console.append("── sts exited (\(status)) ──\n")
        if userStopped {
            pageController.showStatus("sts is stopped", detail: "Server ▸ Start sts (⌘⇧R) to start it.")
        } else {
            pageController.showStatus("sts exited with code \(status)", detail: "See the console for why.")
            setConsoleVisible(true)
        }
    }

    /// The page is served once sts has its web server up (before the models
    /// finish loading, which the page shows itself): poll until it answers.
    private func waitForPage() {
        guard let url = config.resolvedPageURL else {
            pageController.showStatus("Bad page URL", detail: config.pageURL)
            return
        }
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

/// The web view, with a status overlay while there's no page to show.
final class PageViewController: NSViewController, WKUIDelegate, WKNavigationDelegate {
    private var webView: WKWebView!
    private let overlay = NSStackView()
    private let titleLabel = NSTextField(labelWithString: "")
    private let detailLabel = NSTextField(wrappingLabelWithString: "")
    private let spinner = NSProgressIndicator()

    override func loadView() {
        let configuration = WKWebViewConfiguration()
        configuration.mediaTypesRequiringUserActionForPlayback = []  // the page streams the voice
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.uiDelegate = self
        webView.navigationDelegate = self
        webView.setValue(false, forKey: "drawsBackground")

        titleLabel.font = .systemFont(ofSize: 18, weight: .medium)
        detailLabel.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        detailLabel.textColor = .secondaryLabelColor
        detailLabel.alignment = .center
        spinner.style = .spinning
        spinner.controlSize = .small
        overlay.orientation = .vertical
        overlay.alignment = .centerX
        overlay.spacing = 10
        overlay.addArrangedSubview(spinner)
        overlay.addArrangedSubview(titleLabel)
        overlay.addArrangedSubview(detailLabel)

        let container = NSView()
        container.wantsLayer = true
        container.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        for view in [webView!, overlay] as [NSView] {
            view.translatesAutoresizingMaskIntoConstraints = false
            container.addSubview(view)
        }
        NSLayoutConstraint.activate([
            webView.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            webView.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            webView.topAnchor.constraint(equalTo: container.topAnchor),
            webView.bottomAnchor.constraint(equalTo: container.bottomAnchor),
            overlay.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            overlay.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            overlay.widthAnchor.constraint(lessThanOrEqualTo: container.widthAnchor, constant: -40),
        ])
        view = container
    }

    func showStatus(_ title: String, detail: String = "", busy: Bool = false) {
        _ = view
        titleLabel.stringValue = title
        detailLabel.stringValue = detail
        busy ? spinner.startAnimation(nil) : spinner.stopAnimation(nil)
        spinner.isHidden = !busy
        overlay.isHidden = false
        webView.isHidden = true
    }

    func load(_ url: URL) {
        overlay.isHidden = true
        webView.isHidden = false
        webView.load(URLRequest(url: url))
    }

    func reload() {
        if !webView.isHidden { webView.reload() }
    }

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
