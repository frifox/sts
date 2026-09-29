import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuItemValidation {
    private var main: MainWindowController!
    private lazy var configSheet = ConfigSheet()

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.mainMenu = buildMenu()
        main = MainWindowController()
        main.onEditConfig = { [weak self] in self?.editConfiguration(nil) }
        main.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Stop sts (Ctrl+C, so it shuts its pipeline down) before quitting.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard main.sts.isRunning else { return .terminateNow }
        main.stopSTS { NSApp.reply(toApplicationShouldTerminate: true) }
        return .terminateLater
    }

    // MARK: actions

    @objc func editConfiguration(_ sender: Any?) {
        guard let window = main.window, window.attachedSheet == nil else { return }
        configSheet.begin(on: window, config: main.config) { [weak self] in self?.main.apply($0) }
    }

    @objc func startSTS(_ sender: Any?) { main.startSTS() }
    @objc func stopSTS(_ sender: Any?) { main.stopSTS() }
    @objc func restartSTS(_ sender: Any?) { main.restartSTS() }
    @objc func toggleConsole(_ sender: Any?) { main.setConsoleVisible(!main.isConsoleVisible) }
    @objc func clearConsole(_ sender: Any?) { main.clearConsole() }
    @objc func reloadPage(_ sender: Any?) { main.reloadPage() }

    func validateMenuItem(_ item: NSMenuItem) -> Bool {
        switch item.action {
        case #selector(startSTS(_:)): return !main.sts.isRunning
        case #selector(stopSTS(_:)), #selector(restartSTS(_:)): return main.sts.isRunning
        case #selector(toggleConsole(_:)):
            item.title = main.isConsoleVisible ? "Hide Console" : "Show Console"
            return true
        default: return true
        }
    }

    // MARK: menu

    private func buildMenu() -> NSMenu {
        let menu = NSMenu()

        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "About STS", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Configuration…", action: #selector(editConfiguration(_:)), keyEquivalent: ",")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Hide STS", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        let hideOthers = appMenu.addItem(withTitle: "Hide Others", action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        appMenu.addItem(withTitle: "Show All", action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit STS", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        add(appMenu, titled: "STS", to: menu)

        // Copy/paste in the page's text fields and the configuration sheet go through these.
        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        add(edit, titled: "Edit", to: menu)

        let view = NSMenu(title: "View")
        let console = view.addItem(withTitle: "Show Console", action: #selector(toggleConsole(_:)), keyEquivalent: "j")
        console.keyEquivalentModifierMask = [.command]
        let clear = view.addItem(withTitle: "Clear Console", action: #selector(clearConsole(_:)), keyEquivalent: "k")
        clear.keyEquivalentModifierMask = [.command]
        view.addItem(.separator())
        view.addItem(withTitle: "Reload Page", action: #selector(reloadPage(_:)), keyEquivalent: "r")
        view.addItem(.separator())
        let fullScreen = view.addItem(withTitle: "Enter Full Screen", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
        fullScreen.keyEquivalentModifierMask = [.command, .control]
        add(view, titled: "View", to: menu)

        let server = NSMenu(title: "Server")
        server.addItem(withTitle: "Start sts", action: #selector(startSTS(_:)), keyEquivalent: "")
        server.addItem(withTitle: "Stop sts", action: #selector(stopSTS(_:)), keyEquivalent: ".")
        server.addItem(withTitle: "Restart sts", action: #selector(restartSTS(_:)), keyEquivalent: "R")
        server.addItem(.separator())
        server.addItem(withTitle: "Configuration…", action: #selector(editConfiguration(_:)), keyEquivalent: "")
        add(server, titled: "Server", to: menu)

        let window = NSMenu(title: "Window")
        window.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        window.addItem(withTitle: "Zoom", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        add(window, titled: "Window", to: menu)
        NSApp.windowsMenu = window

        return menu
    }

    private func add(_ submenu: NSMenu, titled title: String, to menu: NSMenu) {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        item.submenu = submenu
        menu.addItem(item)
    }
}
