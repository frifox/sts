# STS: Speech To Speech

Live Russian → English speech translation: listens to a speaker, shows live
captions in both languages, and speaks the translation in a natural voice.
Everything runs locally on-device - no audio leaves the machine, and it
works offline once the models are downloaded. Runs on an NVIDIA GPU
(`--device cuda`) or an Apple Silicon Mac (`--device mps`).

## How it works

```
sound card / browser mic -> GigaAM captions (ru) -> MADLAD-400 (en) -> XTTS / Qwen3-TTS -> sound card / browser
                                    \___________________ caption page (websocket) ___/
```

- **Captions** — [GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3)
  re-transcribes the unsettled audio every 500 ms, so text appears as it's
  spoken and can still be corrected. A sentence becomes final once it has
  its end punctuation and hasn't changed (text or punctuation) for 3 s, or
  once nothing new has been heard for 7 s, end punctuation or not
  (`--final-after-unchanged`, `--final-after-silence`).
  `--final-force-after 30` also finalizes a sentence 30 s after its first
  word however much it's still changing (off by default).
  Only final sentences are translated and spoken.
- **Translation** — MADLAD-400 10B, one sentence at a time. Picked by
  comparing models on a recorded sermon (see `translate.py`): it renders
  Bible passages in standard English (mostly KJV) wording on its own.
- **Voice** — XTTS v2 (built-in voices, zero-shot cloning, streamed so a
  sentence starts playing ~1 s in) or Qwen3-TTS (cloning, preset voices,
  voices designed from a text description). Every sentence is brought to
  the same loudness.
- **Bible references** — a spoken reference ("Матфея 1:15") shows a badge
  with the reference; clicking it opens the verse in Russian (Synodal) and
  English (KJV) from a bundled offline dataset.

## Caption page

`http://<host>` (`https://<host>` for the browser mic) - open it on any
number of screens (projector, phones); they stay in sync. Landscape shows original over translation; Portrait
shows them side by side, one row per sentence.

Hover a header item to open its settings:

| Control | Click | Dropdown |
|---|---|---|
| Input meter | - | Input device; use this browser's mic instead (needs HTTPS, see Usage) |
| **Caption** | Recognition on/off | Show/hide not-yet-final text |
| **Translate** | Translation on/off | Landscape / Portrait |
| **Voice** | Speech on/off (starts off) | Voice list; Train, Save, Create (Qwen3), Delete voice |
| Output meter | - | Output device, volume, stream voice to this browser, test output |
| **Reset** | Clear captions, drop text in flight, forget the cloned voice | - |

**Train voice** records the live speaker (up to 18 s) and clones them
immediately; **Save Voice** keeps it in the voice library (`voices/`) for
later sessions. Devices, volume and the active voice are remembered across
restarts (`sts.db`).

## Hardware

| | Memory |
|---|---|
| GigaAM ASR | ~1 GB |
| MADLAD-400 10B / 7B / 3B (bf16) | ~21 / 15 / 6 GB |
| XTTS / Qwen3-TTS 1.7B | ~2 / ~4 GB |

The defaults need ~25 GB: a 32 GB GPU or an Apple Silicon Mac with 32 GB+.
With `--translation-model google/madlad400-3b-mt` a 12 GB GPU works.
Qwen3-TTS on pre-Ampere NVIDIA cards falls back to float16 automatically.

## Setup

Python 3.10 or 3.11 (Coqui TTS doesn't support 3.12+). The TTS options
need conflicting `transformers` versions, so each gets its own virtualenv:

```bash
git clone https://github.com/frifox/sts.git && cd sts
python3.11 -m venv .venv-xtts && .venv-xtts/bin/pip install -e ".[xtts,dev]"
python3.11 -m venv .venv-qwen && .venv-qwen/bin/pip install -e ".[qwen3,dev]"   # optional
python3.11 -m venv .venv-mlx && .venv-mlx/bin/pip install -e ".[mlx,dev]"       # optional, Apple Silicon
```

`.venv-mlx` runs Qwen3-TTS on Apple's MLX instead of PyTorch: the same voices
at about 2.5x the speed on Apple Silicon (1.7B model: ~0.27x real time vs
~0.72x). Zero-shot cloned voices only - no trained LoRA voices.

- **NVIDIA:** install the driver, then PyTorch from the CUDA index before
  the project (`pip install torch --index-url https://download.pytorch.org/whl/cu124`).
- **Linux:** also `apt install libportaudio2`.
- **macOS:** allow Microphone access for the terminal app you run sts from.
  Without it macOS silently records nothing; sts detects this and opens the
  setting.

Models download on first run.

## Usage

```bash
.venv-xtts/bin/sts                          # XTTS voice (.venv-mlx/bin/sts or .venv-qwen/bin/sts: Qwen3-TTS)
sts --list-devices                          # find input/output device indexes
sts --input-device 2 --output-device 3
sts --translation-model google/madlad400-3b-mt   # less memory
sts --default-voice "Craig Gutsy"           # starting built-in voice
sts --clone speaker.wav --clone-name "Ivan" # clone from a 10-20 s clip and save it
sts --web-port 8765 --https-port 8443      # other ports (defaults: 80 and 443)
sts --https-port 0                          # plain HTTP only
sts --all-interfaces                        # let other devices open the page (default: this computer only)
sts -v                                      # log timings and device events
```

The caption page is served over HTTP (port 80) and HTTPS (443), to this
computer only unless `--all-interfaces` is given. HTTPS is what lets other
devices use their browser mic - browsers only allow the mic on a
secure page, which plain HTTP is only on localhost. It uses a self-signed
certificate, `certs/cert.pem` + `certs/key.pem`, made on first start (with
`openssl`); each device shows a one-time warning to click through.
`sts --help` lists everything else (per-stage devices, caption timing,
pauses, Qwen3 options).

## Mac app

`mac/` builds **Translate Live.app**: a native window for sts with its own Python and
the MLX voice engine (Qwen3-TTS) inside, so it needs no checkout or
virtualenv to run.

```bash
make                    # builds Translate Live.app fresh and installs it in /Applications
make build              # only builds it, in mac/build/ (~1.6 GB)
make install            # installs the last build
```

Quit the app first: both steps replace an app that may be running (`make` stops
and says so if it is).

The first build downloads a standalone Python 3.11
([python-build-standalone](https://github.com/astral-sh/python-build-standalone))
and installs `sts[mlx]` into it; later builds reuse it and reinstall sts.

- sts starts as soon as the app opens; the window shows the caption page
  once sts serves it. **Edit Config** / **Start** come back if it's
  stopped or fails to start.
- **Edit Config** (also Translate Live ▸ Configuration…, ⌘,) lists every sts option,
  one per line, with its default beside it; anything left at Default isn't
  passed. Saving restarts sts if it's running.
- **View ▸ Show/Hide Console (⌘J)**: sts's output in a panel at the bottom.
- **Server**: start, stop (⌘.), restart (⇧⌘R).

The app's sts keeps `sts.db`, `voices/` and `certs/` in
`~/Library/Application Support/Translate Live` - and on macOS sts run from the terminal
uses the same `sts.db` and `voices/`, so settings and voices are shared; models still download to
`~/.cache/huggingface`. macOS asks for Microphone access for the app itself.

## Tools

- **`sts-caption`** — captions only, in the terminal; for checking recognition.
- **`sts-add-voices`** — adds a batch of voices: put video or audio files
  (one speaker each, named after them) in the voice library's `new/` folder
  (on macOS `~/Library/Application Support/Translate Live/voices/new/`, the
  library the app uses too; elsewhere `voices/new/`) and run it. It picks
  the best ~18 s of speech from each, transcribes it, saves it as a voice
  (like Clone) and moves the file to `new/done/`. `--start/--end`
  limit it to part of a file; `--replace` redoes an existing voice.
- **`translate.py`** — compares translation models (NLLB, MADLAD,
  HY-MT1.5, TranslateGemma) on the first minutes of a recording and writes
  each sentence with every model's English. Run from `.venv-qwen`.
- **`benchmark.py`** — times each stage on this machine; results go to `benchmark_results/`.
- **`training/qwen3_lora/`** — trains a Qwen3-TTS LoRA voice on rented GPU
  compute; `sts-import-voice` adds it to the library. See its
  [README](training/qwen3_lora/README.md).

## Tests

```bash
.venv-xtts/bin/pytest
```

Covers the logic that doesn't need models or hardware (caption locking,
Bible lookup, controls, voice library, devices, playback, loudness).
