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

## Setup

Requires Python 3.10+ and an NVIDIA GPU with CUDA for real-time performance
(CPU works but will lag).

```bash
pip install -e .
```

Model weights (Whisper, NLLB, XTTS, Silero VAD) download automatically on
first run — that first run needs internet access even though normal
operation doesn't.

## Usage

```bash
sts --list-devices                 # find your sound card's input/output index
sts --input-device 2 --output-device 3
sts --clone reference.wav          # stretch goal: clone the speaker's voice
                                    # from a 10-20s reference clip instead of
                                    # using the default TTS voice
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Only the glossary term-matching logic is unit tested here — the audio/ASR/
translation/TTS stages need real hardware (mic, GPU) and aren't exercised by
the test suite.
