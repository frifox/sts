# Live Translate (sts) — Bitfocus Companion module

Companion buttons that toggle sts's Caption, Translate and Voice and show their
state, kept in sync with the sts caption page in both directions.

It talks to sts over its remote-control websocket, `ws://<host>:<port>/control`
(no authentication yet):

| Direction | Message |
|---|---|
| sts → module | `{"type": "status", "active", "translation_enabled", "voice_enabled", "voice_stopping", ...}` on connect and on every change |
| sts → module | `{"type": "stage", "text", "ready"}` start-up progress ("Listening" once ready) |
| module → sts | `{"action": "start" \| "stop" \| "translate_on" \| "translate_off" \| "voice_on" \| "voice_off"}` |
| module → sts | `{"action": "get_status"}` — the status again |

Built on `@companion-module/base` 1.10: runs on Companion 3.4 and later.

## Install

```bash
cd companion-sts
npm install
npm run package          # makes pkg.tgz
```

In Companion: **Modules ▸ Import module package**, pick `pkg.tgz`, then add a
**Live Translate** connection and set its host and port (the sts caption
page's: `127.0.0.1`, `80` on the same computer). sts must be started with
`--all-interfaces` if Companion runs on another computer.

For development, point Companion's **Developer modules path** (Settings ▸
Advanced) at the folder containing `companion-sts`; it reloads the module on
changes.
