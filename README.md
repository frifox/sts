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
  spoken and can still be corrected. A sentence becomes final once there's
  text after it, once it hasn't changed for 5 s, or 30 s after its first
  word however much it's still changing (`--final-after-unchanged`,
  `--final-force-after`). Only final sentences are translated and spoken.
- **Translation** — MADLAD-400 10B, one sentence at a time. Picked by
  comparing models on a recorded sermon (see `translate.py`): it renders
  Bible passages in standard English (mostly KJV) wording on its own.
- **Voice** — XTTS v2 (built-in voices, zero-shot cloning, streamed so a
  sentence starts playing ~1 s in) or Qwen3-TTS (cloning, preset voices,
  voices designed from a text description). Every sentence is brought to
  the same loudness.
- **Bible references** — a spoken reference ("Матфея 1:15") shows the verse
  in Russian (Synodal) and English (KJV) from a bundled offline dataset.

## Caption page

`http://<host>:8765` - open it on any number of screens (projector, phones);
they stay in sync. Landscape shows original over translation; Portrait
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

Python 3.10 or 3.11 (Coqui TTS doesn't support 3.12+). The two TTS backends
need conflicting `transformers` versions, so each gets its own virtualenv:

```bash
git clone https://github.com/frifox/sts.git && cd sts
python3.11 -m venv .venv-xtts && .venv-xtts/bin/pip install -e ".[xtts,dev]"
python3.11 -m venv .venv-qwen && .venv-qwen/bin/pip install -e ".[qwen3,dev]"   # optional
```

- **NVIDIA:** install the driver, then PyTorch from the CUDA index before
  the project (`pip install torch --index-url https://download.pytorch.org/whl/cu124`).
- **Linux:** also `apt install libportaudio2`.
- **macOS:** allow Microphone access for the terminal app you run sts from.
  Without it macOS silently records nothing; sts detects this and opens the
  setting.

Models download on first run.

## Usage

```bash
.venv-xtts/bin/sts                          # XTTS voice (.venv-qwen/bin/sts for Qwen3-TTS)
sts --list-devices                          # find input/output device indexes
sts --input-device 2 --output-device 3
sts --translation-model google/madlad400-3b-mt   # less memory
sts --default-voice "Craig Gutsy"           # starting built-in voice
sts --clone speaker.wav --clone-name "Ivan" # clone from a 10-20 s clip and save it
sts --https-port 443                        # HTTPS too, for browser-mic input
sts -v                                      # log timings and device events
```

`--https-port` uses `certs/cert.pem` and `certs/key.pem`; make a self-signed pair with
`openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/key.pem -out certs/cert.pem -days 3650 -subj '/CN=sts'`.
`sts --help` lists everything else (per-stage devices, caption timing,
pauses, Qwen3 options).

## Tools

- **`sts-caption`** — captions only, in the terminal; for checking recognition.
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
