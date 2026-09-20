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
- **TTS** (`sts.tts.xtts`) — Coqui XTTS v2. Ships with natural built-in
  speakers for day-to-day use, and directly supports the stretch goal:
  zero-shot voice cloning from a short reference clip of the actual speaker.

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
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The glossary term-matching and Bible reference detection/lookup are unit
tested here — the audio/ASR/translation/TTS stages need real hardware (mic,
GPU) and aren't exercised by the test suite.
