## Live Translate (sts)

Buttons that turn sts's **Caption**, **Translate** and **Voice** on and off, each lit by its current state —
whether it was changed here or on the sts caption page.

### Setup

- **Host / Port**: where sts serves its caption page — `127.0.0.1` and `80` when sts runs on this computer.
  From another computer, use the sts computer's address (sts accepts connections from the network unless
  started with `--local-only`).
- The connection status shows on the module in the Connections list; it reconnects by itself when sts restarts.

### Buttons

Presets ▸ **Toggles**: Caption, Translate, Voice — press to toggle.

Presets ▸ **Voices**: **Voice 125%** — switches between the *Default 1.25* voice (lit) and *Default* (off).
Amber while the voice it switched to is being prepared (the previous one speaks until then).

| Color | Meaning |
|---|---|
| black | off |
| green (#468259) | on |
| amber | Voice off was pressed: it finishes the sentence being spoken, then goes off (press again to keep it on) |
| dark red | not connected to sts |

### Actions

**Caption / Translate / Voice: toggle / on / off** — Toggle, or set it on or off.

**Voice: switch between two voices** — toggle between an "on" and an "off" voice, or pick one. Voices are given by
their id in the sts voice library (the folder name: `default`, `default-1-25`, …).

### Feedbacks

- **Caption / Translate / Voice is on**
- **Voice is finishing** — off after the sentence being spoken
- **Voice in use is …** — that library voice is speaking
- **Voice … is being prepared** — chosen, not speaking yet
- **Not connected to sts**

### Variables

`$(sts-live-translate:caption)`, `translate`, `voice` (`on` / `off` / `finishing`), `speaker` (library voice id),
`connection`, `stage`
