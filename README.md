# sts

Live speech-to-speech pipeline: listens to a sound card feed of a Russian
speaker (pulpit audio, Baptist Christian sermon context), transcribes,
translates to English, and speaks the translation back in a natural voice.

Everything runs locally against on-device models — no audio leaves the
machine, no per-minute API costs, and it works without internet once the
models are downloaded. Target hardware is a PC/laptop with an NVIDIA GPU.

## Pipeline

```
sound card -> VAD segmenter -> Whisper ASR (ru) -> NLLB translation (ru->en) -> XTTS (en) -> sound card out
                                       |                        |
                                       v                        v
                                  original text          translated text
                                        \______________________/
                                                   |
                                             websocket -> browser (live captions)
```

- **Capture** (`sts.audio.capture`) — reads raw audio off an input device via `sounddevice`.
- **VAD segmentation** (`sts.audio.vad`) — Silero VAD buffers speech and cuts a
  segment on trailing silence, so ASR sees whole utterances instead of
  arbitrary fixed windows.
- **ASR** (`sts.asr.whisper_asr`) — `faster-whisper`, `large-v3`, Russian,
  primed with a Baptist/religious-vocabulary prompt to bias decoding.
- **Translation** (`sts.translate.translator`) — NLLB-200 (`ru->en`). A
  glossary (`sts.translate.glossary`) force-substitutes known Baptist/Christian
  terms (grace, repentance, communion, sanctification, ...) so the model
  can't mistranslate them, matching inflected Russian forms via stemming.
- **TTS** (`sts.tts`) — two swappable backends, picked via `--tts-backend`:
  - `xtts` (default) — Coqui XTTS v2. Ships natural built-in speakers, plus
    zero-shot cloning from a short reference clip.
  - `qwen3` — Qwen3-TTS-Base (`pip install -e '.[qwen3]'`), Apache-2.0,
    self-hosted. Also zero-shot cloning, and officially supports
    cross-lingual cloning (Russian reference clip -> English output) with a
    reusable "clone prompt" object built from the clip once. It has no
    XTTS-style built-in default voice, so synthesis is silent until a voice
    is picked or trained. Unverified against real hardware in this repo (no
    GPU here) — see `sts/tts/qwen3.py` for exactly what the integration
    assumes, taken from the upstream README's example code.
  Both back a persistent, cross-session **voice library**
  (`sts.tts.voice_library`, dropdown on the caption page — see Operator
  controls below) rather than a one-off in-memory clone.

All stages run on separate threads connected by queues (`sts.pipeline`), so a
slow translation or TTS call doesn't stall audio capture.

- **Live captions** (`sts.web`) — an optional FastAPI/uvicorn server (`--web`)
  that serves a full-screen caption page and pushes recognized/translated
  text to it over a websocket as it's produced. The page is a top half
  (original) / bottom half (translated) split; each new line fades in at the
  bottom and older lines are dropped once they no longer fit on screen, so
  the page keeps auto-scrolling without accumulating unbounded DOM/memory
  over a long service.
- **Bible reference detection** (`sts.bible`) — the Russian transcript is
  scanned for a spoken reference ("Матфея 1:15", "Первое послание к
  Коринфянам, глава 13, стих 4", "Плач Иеремии 3:22-23", chapter/verse as
  digits or Russian number words). A match looks up the exact verse in a
  bundled, fully offline Russian Synodal + English KJV (Pure Cambridge
  Edition) dataset and pushes it to the caption page as a banner that slides
  in over both panels for ~25s. Book-name matching is stem-based (all 66
  books) rather than an exhaustive alias list, and Psalms gets its chapter
  number converted between the Synodal (Septuagint) and KJV (Masoretic)
  numbering traditions so it doesn't show the wrong psalm — see
  `sts/bible/lookup.py` for what that does and doesn't cover. This is
  inherently best-effort: it depends on the ASR getting the reference right
  and only covers phrasing the parser recognizes.
- **Operator controls** (`sts.control`, buttons on the caption page) — a live
  feed runs across multiple speakers/performances in a session (music,
  announcements, one voice at a time when someone *is* speaking), so
  recognition defaults to **off** and an operator drives it:
  - **Start / Stop** — turns ASR/translation/TTS on or off. Audio capture and
    VAD segmentation keep running either way; Stop just means nothing gets
    transcribed, so switching a long musical number or announcement doesn't
    waste GPU time or produce junk captions.
  - **Reset** — for a new speaker/performance: clears both caption panels on
    every connected screen, drops any text already in flight through the
    pipeline, and forgets the current cloned voice (falls back to the
    default TTS voice).
  - **Train voice** — type a speaker name, then start capturing their live
    audio for a voice clone; stops automatically at 18s or on a second
    click, whichever comes first. What's captured is transcribed (for a
    higher-quality clone on backends that use a reference transcript) and
    saved into the voice library under that name — reusable from the
    **speaker dropdown** in later sessions, not just this one. Click Reset
    before the next speaker so their audio doesn't get folded into the
    previous one's training sample.
  - **Speaker dropdown** — picks a previously trained/saved voice from the
    library (or "— no cloned voice —" to fall back to the default) without
    retraining.

  All of this is relayed over the same websocket the captions arrive on
  (`{"action": "start"}`, `{"action": "train_start", "name": "..."}`,
  `{"action": "select_speaker", "speaker_id": "..."}`, etc.); the server
  broadcasts current state and the library list back to every connected
  screen so multiple open tabs stay in sync. Library entries live under
  `--voice-library-dir` (default `voices/`) as `<speaker-id>/reference.wav`
  + `meta.json` — the raw reference clip and transcript, not a backend's
  in-memory prompt object, so they're portable across restarts and backend
  swaps and a clone prompt only needs recomputing (seconds) when a speaker
  is actually selected.

  This is zero-shot clone caching, not true model fine-tuning — a
  from-scratch per-speaker model needs far more data (10-30 min of clean
  audio) and training time than a live session realistically gives you.

## Setup

Requires Python 3.10+ and an NVIDIA GPU with CUDA for real-time performance
(CPU works but will lag).

```bash
pip install -e .
```

Model weights (Whisper, NLLB, XTTS, Silero VAD) download automatically on
first run — that first run needs internet access even though normal
operation doesn't.

`src/sts/bible/data/bible.json.gz` bundles the Russian Synodal (1876) and
KJV Pure Cambridge Edition (1769) texts, both public domain, sourced from
[scrollmapper/bible_databases](https://github.com/scrollmapper/bible_databases)
(MIT-licensed compilation).

## Usage

```bash
sts --list-devices                 # find your sound card's input/output index
sts --input-device 2 --output-device 3
sts --clone reference.wav          # stretch goal: clone the speaker's voice
                                    # from a 10-20s reference clip instead of
                                    # using the default TTS voice
sts --web --web-port 8765          # also serve live captions at :8765
                                    # (open it full-screen / kiosk mode on a
                                    # second monitor or projector)
sts --web --tts-backend qwen3 --voice-library-dir /data/sts-voices
                                    # Qwen3-TTS backend + a custom voice
                                    # library location
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The glossary term-matching, Bible reference detection/lookup, the
start/stop/reset/train control state machine (`sts.control`), and the voice
library (`sts.tts.voice_library`) are unit tested here — the audio/ASR/
translation/TTS stages need real hardware (mic, GPU) and aren't exercised by
the test suite.
