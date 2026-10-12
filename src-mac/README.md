# Live Translate.app (macOS)

A native macOS window for sts: the caption page on top, sts's output in a
console panel below, and a form for every sts option. It carries its own
Python with sts and the MLX voice engine (Qwen3-TTS on Apple's MLX) inside,
so it runs without a checkout or a virtualenv.

Apple Silicon, macOS 13 or later.

## Build and install

From the repository root:

```bash
make                    # build fresh and install in /Applications
make build              # build only: src-mac/build/Live Translate.app (~1.6 GB)
make install            # install the last build
```

Quit the app first: both steps replace an app whose Python may be running
(`make` checks and stops if it is). The Claude desktop app's server list has
a **make** entry that does the same and shows the build's output live on a
page (`make_page.py`).

Needs the Xcode command line tools (`swift`, `codesign`, `iconutil`). The
first build downloads a standalone Python and every dependency (several
minutes, a few GB); later builds reuse them and only reinstall sts itself.

## What's here

| File | |
|---|---|
| `build.sh` | The build: Python, sts, the Swift app, icon, Info.plist, signing |
| `Package.swift`, `Sources/STSApp/` | The app (Swift, AppKit + WebKit) |
| `export_options.py` | Writes sts's command-line options as `options.json` for the Configuration form |
| `make_icon.swift` | Draws the app icon |
| `make_page.py` | Runs `make` and shows its output on a web page (the Claude app's **make** entry) |
| `build/` | Build output and the kept Python (ignored by git) |

### `Sources/STSApp/`

| File | |
|---|---|
| `main.swift`, `AppDelegate.swift` | Start-up and the menus |
| `MainWindowController.swift` | The window: home screen / caption page (web view) and the console panel |
| `STSProcess.swift` | Runs sts and collects its output; stopping sends Ctrl+C, then forces it after 5–8 s |
| `ConsoleView.swift` | The console: follows the latest output unless scrolled up |
| `ConfigSheet.swift`, `LaunchConfig.swift` | The Configuration form, and turning it into sts's arguments |

## How the build works

1. **Python**: a relocatable CPython 3.11
   ([python-build-standalone](https://github.com/astral-sh/python-build-standalone),
   pinned in `build.sh`), kept in `build/python` between builds. A virtualenv
   wouldn't do: it points at the Python it was made from by absolute path.
2. **sts**: `pip install "<repo>[mlx]"` into that Python, then sts itself
   reinstalled every build (an unchanged version number would otherwise
   count as installed).
3. **The app**: `swift build -c release`, the binary copied in as
   `Contents/MacOS/LiveTranslate`, the icon drawn, `options.json` exported
   from the bundled sts, the Python copied into `Contents/Resources/python`,
   and `Info.plist` written.
4. **Signing**: ad hoc (`codesign --sign -`), then registered with Launch
   Services so Finder and the Dock pick up the new build.

## What the app does

- **Starts sts when it opens**: the bundled Python runs `-m sts.cli` with the
  options set in the Configuration form, in the data folder. A "Starting…"
  screen stays up until the caption page has loaded (a failed load is
  retried every second); if sts stops or fails to start, **Edit Config** /
  **Start** come back.
- **Data folder**: `~/Library/Application Support/Live Translate` - `sts.db`
  (remembered settings), `voices/` (the voice library) and `certs/` (the
  self-signed HTTPS certificate). sts run from the terminal uses the same
  `sts.db` and `voices/` on macOS. Models download to `~/.cache/huggingface`.
- **Configuration** (⌘,): one row per sts option, generated from the bundled
  sts's own options, with its default beside it; a row left at Default
  passes nothing. Saving restarts sts if it's running. Kept in the app's
  preferences (bundle id `com.frifox.sts`).
- **Microphone**: sts runs as the app's child process, so macOS asks for
  microphone access for the app itself.

### Menus

| | |
|---|---|
| Live Translate ▸ Configuration… | ⌘, |
| View ▸ Show / Hide Console | ⌘J |
| View ▸ Clear Console | ⌘K |
| View ▸ Reload Page | ⌘R |
| Server ▸ Start sts / Stop sts / Restart sts | — / ⌘. / ⇧⌘R |

Closing the window quits the app, which stops sts first (Ctrl+C, so its
pipeline shuts down cleanly).

## Notes

- **Renamed**: the app was STS.app, then Translate Live.app; `make install`
  removes those from /Applications if they're still there.
- **Rebuilding while it runs** would pull files out from under the running
  sts (its Python lives inside the app) - hence the check in `make`. Run the
  app from /Applications, not from `build/`.
- **After an update**, macOS may ask for microphone access again: each build
  is signed anew.
