import Foundation

/// The running sts: started through a login shell (`zsh -lc "exec <command>"`,
/// so the shell is replaced and signals reach sts itself), stdout and stderr
/// merged and handed to `onOutput` on the main queue.
final class STSProcess {
    var onOutput: ((String) -> Void)?
    var onExit: ((Int32) -> Void)?

    private var process: Process?
    private var stopping = false

    var isRunning: Bool { process?.isRunning ?? false }

    func start(_ config: LaunchConfig) throws {
        guard !isRunning else { return }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", "exec \(config.command)"]
        process.currentDirectoryURL = URL(fileURLWithPath: config.expandedWorkingDirectory)
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"  // console output as it happens, not in 4 KB blocks
        process.environment = env

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        process.standardInput = FileHandle.nullDevice
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else {
                handle.readabilityHandler = nil
                return
            }
            let text = String(decoding: data, as: UTF8.self)
            DispatchQueue.main.async { self?.onOutput?(text) }
        }
        process.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                guard let self, self.process === proc else { return }
                self.process = nil
                self.stopping = false
                self.onExit?(proc.terminationStatus)
            }
        }
        stopping = false
        try process.run()
        self.process = process
    }

    /// Ctrl+C first (sts stops its pipeline cleanly), then SIGTERM, then SIGKILL.
    func stop(completion: (() -> Void)? = nil) {
        guard let process, process.isRunning else {
            completion?()
            return
        }
        stopping = true
        let pid = process.processIdentifier
        kill(pid, SIGINT)
        let deadline = Date()
        func poll() {
            if !process.isRunning {
                completion?()
                return
            }
            let waited = Date().timeIntervalSince(deadline)
            if waited > 8 { kill(pid, SIGKILL) } else if waited > 5 { kill(pid, SIGTERM) }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2, execute: poll)
        }
        poll()
    }
}
