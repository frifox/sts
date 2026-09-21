# sts

Live speech-to-speech pipeline: listens to a sound card feed of a Russian
speaker (pulpit audio, Baptist Christian sermon context), transcribes,
translates to English, and speaks the translation back in a natural voice.

Everything runs locally against on-device models — no audio leaves the
machine, no per-minute API costs, and it works without internet once the
models are downloaded. Primary target hardware is a PC/laptop with an NVIDIA
GPU (`--device cuda`); Apple Silicon (`--device mps`) is also supported —
see **Apple Silicon / Metal** below.

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
  - `qwen3` — Qwen3-TTS-Base (`pip install -e '.[qwen3]'`, in its own virtualenv - see Setup), Apache-2.0,
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
  announcements, one voice at a time when someone *is* speaking).
  Recognition/translation/captions run **by default** - captions are silent,
  so leaving them running through a musical number is harmless. TTS
  *voicing* is different (audio played over speakers during music or an
  announcement would not be harmless) and defaults to **off**, opt-in per
  button:
  - **Start / Stop** — pauses/resumes ASR/translation/captions, for when even
    the captions aren't wanted (e.g. a long musical number). Audio capture
    and VAD segmentation keep running either way; Stop just means nothing
    gets transcribed, so it doesn't waste GPU time or produce junk captions.
  - **Voice on / off** — independently gates whether translated text is
    actually synthesized and played out loud; starts off regardless of
    Start/Stop, so captions can run continuously while voicing is switched
    on only when someone is actually preaching.
  - **Reset** — for a new speaker/performance: clears both caption panels on
    every connected screen, drops any text already in flight through the
    pipeline, and forgets the current cloned voice (falls back to the
    default TTS voice).
  - **Train voice** — start capturing the live speaker's audio for a voice
    clone; stops automatically at 18s or on a second click, whichever comes
    first. What's captured is transcribed (for a higher-quality clone on
    backends that use a reference transcript) and the clone becomes active
    right away, so you can listen to it. It is **not** saved yet.
  - **Create Voice** (Qwen3 backend only) — designs a brand-new voice from a
    written description, no recording needed. Describe the voice, hit
    **Generate**, then paste any text into the second box and **Play sample**
    (plays on the output device) to hear it. Not right? Tweak the description
    and Generate again - each generation is a *different* voice, so keep
    regenerating until one fits. When it does, name it and **Save**: it's
    stored in the voice library like a trained voice and appears in the
    voice dropdown (Custom group). Uses the Qwen3-TTS VoiceDesign model, which is loaded
    only while the dialog is open (first use downloads it).
  - **Save Voice** — enabled once a trained voice is active. Asks for a name
    and, on Save, stores it in the voice library so it's reusable from the
    **voice dropdown** (Custom group) in later sessions. Naming an existing voice replaces
    it. Skip it to keep the voice for this session only. Click Reset before
    the next speaker so their audio doesn't get folded into the previous
    one's training sample.
  - **Voice dropdown** — a single list with two groups:
    - **Default** — the built-in (non-trained) voices. XTTS offers 13
      (defaults to Adde Michal; `--default-voice "Craig Gutsy"` or
      `male`/`female` sets the starting choice). Qwen3-TTS offers the preset
      voices Aiden and Ryan, spoken in a calm "preacher" style. The list
      lives in `TTSConfig.voice_groups` (`sts/config.py`); any of XTTS's 58
      speakers can be added there.
    - **Custom** — voices you trained or designed and saved, reusable across
      sessions. A just-trained voice that isn't saved yet shows up here as
      "New voice (unsaved)".

    Picking any voice switches to it immediately; picking a Default voice
    drops any cloned/trained one.
  - **Delete** — appears to the right of the voice dropdown while a Custom
    voice is selected. Asks for confirmation, then removes that voice from
    the library on disk (its folder under `--voice-library-dir`). If it was
    the active voice, the default voice takes over.

  All of this is relayed over the same websocket the captions arrive on
  (`{"action": "start"}`, `{"action": "voice_on"}`,
  `{"action": "train_start"}`, `{"action": "save_voice", "name": "..."}`,
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
  audio) and training time than a live session realistically gives you. For
  that, see **True LoRA training**, below.

## GPU requirements

Estimates from published model sizes, not benchmarked against real hardware
in this repo (no GPU in dev). All three components load simultaneously
regardless of which caption-page toggles are on, so the last row is what
actually matters for picking hardware.

| Part | Models | Rough VRAM |
|---|---|---|
| Captions (ASR + translation) | faster-whisper `large-v3` + NLLB-200-1.3B | ~5-6GB int8, ~8GB float16 |
| TTS, zero-shot | XTTS v2 (~460M) or Qwen3-TTS-Base (0.6B/1.7B) | ~3-4GB |
| TTS, trained (LoRA) | same base model, adapter merged in | same as zero-shot per speaker; each additional *cached* trained speaker is a full extra model load, not a small increment (see `sts/tts/qwen3.py`) |
| **Whole pipeline together** | all of the above loaded at once | **~7-9GB at int8 Whisper, ~10-14GB at float16** |

A 12GB+ card (RTX 3060 12GB, RTX 4070, ...) is a comfortable floor. 8-11GB
cards (see **Older NVIDIA GPUs**, below, for a GTX 1080 Ti specifically) are
workable but tight, especially with Qwen3-TTS's 1.7B variant or several
cached trained speakers — the 0.6B variant, XTTS, and `int8` Whisper (the
default off-CUDA already) all buy back headroom.

## True LoRA training (rented GPU, e.g. an H100/H200 droplet)

`training/qwen3_lora/` trains a real, reusable Qwen3-TTS LoRA adapter from a
recorded speaker file — meant to run on rented GPU compute, not live. Full
workflow (create droplet, upload recording, one script per step, download
adapter, `sts-import-voice` it into the local library) is in
[`training/qwen3_lora/README.md`](training/qwen3_lora/README.md).

**Training needs the big GPU; using the trained voice doesn't.** LoRA
weights are small (a few hundred MB at most) applied on top of the same base
model already used for zero-shot cloning — what needs an H100/H200 is
training itself (backprop + optimizer state across the whole base model,
even though only a small LoRA delta is updated). Inference with a trained
adapter is a forward pass through that same base model, same cost as
zero-shot cloning already is: an RTX 3090/4090-class consumer GPU runs it
fine. A trained speaker shows up in the caption page's dropdown exactly like
a zero-shot one; switching to it is a few seconds to load, not a retrain.

This isn't officially supported by the Qwen team — Qwen3-TTS's own
`finetuning/` only does full fine-tuning, no LoRA. The LoRA path comes from
a third-party community patch
([cheeweijie/qwen3-tts-lora-finetuning](https://github.com/cheeweijie/qwen3-tts-lora-finetuning),
Apache-2.0) pinned to a specific upstream commit; `sts/tts/qwen3.py`'s LoRA
loading code mirrors that patch's inference script line-for-line for the
same reason the rest of the Qwen3-TTS integration does: it's unverified
against real hardware in this repo (no GPU here), so it's built to match
known-working reference code rather than guessed at.

## Apple Silicon / Metal

`--device mps` runs translation and TTS on the Apple GPU. One component
can't join it: **faster-whisper's CTranslate2 backend has no Metal support**
(only CPU and CUDA), so ASR always runs on CPU on a Mac regardless of this
flag — `sts.devices.resolve_devices` handles that split automatically
(ASR -> cpu, translation/TTS -> mps), and sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` so any PyTorch op MPS doesn't yet implement
falls back to CPU instead of crashing, at some speed cost for that op. Use
`--asr-device cuda` instead only if you actually have an eGPU or are
otherwise not really on Apple Silicon.

Qwen3-TTS uses float16 + SDPA attention on `mps` rather than CUDA's
bfloat16 + flash-attention (flash-attn has no Metal build; MPS's bfloat16
support has historically been the less mature of the two float types) — see
`_dtype_for_device`/`_attn_impl_for_device` in `sts/tts/qwen3.py`. XTTS has
no such split; `.to("mps")` is all it needs.

This hasn't been run on real Apple hardware in this repo (no Mac available
in dev) - assuming a high-end chip (M5 Max-class: plenty of GPU cores and
unified memory), there's no reason to downgrade model sizes (Whisper
`large-v3`, Qwen3-TTS 1.7B) the way you might on a smaller Mac; unified
memory also means there's no separate VRAM ceiling to budget against like
on a discrete GPU. What's unverified is per-op MPS support and real-world
throughput, not whether the models fit.

```bash
sts --web --device mps                          # XTTS backend
sts --web --device mps --tts-backend qwen3       # Qwen3-TTS backend
```

## Older NVIDIA GPUs (pre-Ampere) and Windows

`--device cuda` works on Windows the same as Linux — PyTorch, CTranslate2
(faster-whisper), Coqui TTS, and `sounddevice` all support it natively.

The one thing that varies by *which* GPU, not the OS: Qwen3-TTS's CUDA path
picks bfloat16 + flash-attention by default, but both need Ampere or newer
(compute capability >= 8) - a GTX 10-series card (Pascal, compute 6.1) or a
20-series card (Turing, compute 7.5) supports neither properly.
`sts.tts.qwen3` checks the actual GPU's compute capability at load time
(`sts.devices.supports_bf16_and_flash_attention`) and falls back to
float16 + SDPA attention automatically on anything older than Ampere - same
reasoning, and the same fallback, as the Apple Silicon case above. No flag
needed; this is automatic. (XTTS has no such split either way.)

Flash-attention wheels are also less reliably available for Windows even on
GPUs new enough to use them (Ampere+) - if `pip install flash-attn` fails on
Windows on a card that should otherwise support it, the SDPA fallback above
only triggers based on compute capability, not install failures, so you may
need to fall back manually; nothing else in the pipeline depends on
flash-attn being present.

On VRAM: an 11GB card (e.g. a GTX 1080 Ti) is workable for the whole
pipeline but tight if you also pick Qwen3-TTS's 1.7B variant - the 0.6B
variant, or the XTTS backend, plus `int8` Whisper (already the default for
non-CUDA ASR devices), leaves more headroom. See the GPU requirements
breakdown earlier in this README for the per-component numbers this is
based on.

## Setup

Requires Python 3.10 or 3.11 (not 3.12+: Coqui TTS does not support it) and a GPU for real-time performance — NVIDIA/CUDA
(`--device cuda`), Apple Silicon (`--device mps`, see above), or CPU (works,
but will lag). Supported on Windows, macOS, and Linux.

```bash
git clone https://github.com/frifox/sts.git
cd sts
pip install -e ".[xtts]"
```

If you already have Python, git, and (for `--device cuda`) a working CUDA
install of PyTorch on this machine, that's the whole setup — skip to
**Usage** below. The rest of this section is a from-zero walkthrough per OS,
for a machine that has none of that yet.

### Fresh Windows install

For a Windows PC that's never had a Python dev environment on it (this is
the path a Windows + NVIDIA GPU setup - e.g. a GTX 1080 Ti - would follow):

1. **NVIDIA driver** (skip if using `--device cpu`). Download the latest
   Studio or Game Ready driver for your card from
   [nvidia.com/drivers](https://www.nvidia.com/drivers) and install it, then
   reboot. You do **not** need to separately install the full CUDA Toolkit -
   PyTorch's pip wheels bundle the CUDA runtime they need. Verify the driver
   installed correctly by opening Command Prompt and running:
   ```
   nvidia-smi
   ```
   This should list your GPU and driver version. If it doesn't, the rest of
   this won't detect a GPU either - fix that first.

2. **Python 3.10 or 3.11** (not 3.12+). Download the installer from
   [python.org/downloads](https://www.python.org/downloads/windows/) and run
   it. On the first installer screen, check **"Add python.exe to PATH"**
   before clicking Install - easy to miss, and everything below assumes it's
   checked. Verify in a fresh Command Prompt:
   ```
   python --version
   ```

3. **Git for Windows** (to clone the repo; skip if you'll download a zip
   instead). Install from [git-scm.com](https://git-scm.com/download/win)
   with default options.

4. **Clone the repo and set up a virtual environment:**
   ```
   git clone https://github.com/frifox/sts.git
   cd sts
   python -m venv .venv
   .venv\Scripts\activate
   ```
   You should see `(.venv)` at the start of the prompt from here on - every
   command below assumes it's still active (reactivate with the same command
   if you close and reopen the terminal).

5. **Install PyTorch with CUDA support explicitly, before anything else.**
   Whether a plain `pip install torch` gets you a CUDA-capable build or a
   CPU-only one has changed across PyTorch releases and isn't something to
   rely on either way - install explicitly from the CUDA index, matching a
   CUDA version your driver supports (12.4 is a safe current choice for most
   recent drivers; use the selector at
   [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
   if you need a different one):
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
   Verify it can see the GPU before moving on:
   ```
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   This should print `True` and your GPU's name. If it prints `False`,
   stop here and recheck the driver (step 1) and that you installed the
   `cu124` (or matching) build, not plain `torch`.

6. **Install the project:**
   ```
   pip install -e ".[xtts]"
   ```
   The Qwen3-TTS backend needs a separate virtualenv (`pip install -e ".[qwen3]"`; its
   transformers requirement conflicts with XTTS's)
   — see **Older NVIDIA GPUs (pre-Ampere) and Windows** above for what that
   backend needs on a card like a GTX 1080 Ti specifically.

7. **Known rough edge: faster-whisper (ASR) on Windows + CUDA.** Its
   CTranslate2 backend needs cuBLAS/cuDNN, and Windows wheels have
   historically been less consistent than Linux about bundling them. If
   `sts` fails specifically when loading the Whisper model (not at the
   PyTorch step above, which is a separate library), that's almost always
   this - check [faster-whisper's own install
   docs](https://github.com/SYSTRAN/faster-whisper#gpu) for current
   Windows-specific guidance, since this varies by faster-whisper version
   more than anything else here. Forcing `--asr-device cpu` (ASR still runs
   fine on CPU, just slower) is the immediate workaround while sorting it
   out.

8. **Run it** - see **First run, on any OS** at the end of this section.
   Two Windows-specific prompts to expect the first time you do: a
   **Windows Defender Firewall**
   prompt when the caption web server binds its port (allow it, at least for
   Private networks), and possibly a **microphone permission**
   prompt/block under Settings -> Privacy & security -> Microphone if
   `sounddevice` can't see your input device - enable access for desktop
   apps there if `--list-devices` doesn't show your mic.

### Fresh macOS install

For Apple Silicon (see **Apple Silicon / Metal** above for what `--device
mps` does and doesn't cover) or Intel Macs with no dev environment yet:

1. **Xcode Command Line Tools** (gives you `git` and basic build tools;
   skip if you already have Homebrew or Xcode installed):
   ```
   xcode-select --install
   ```

2. **Homebrew**, if you don't have it - follow the install command at
   [brew.sh](https://brew.sh/). Then:
   ```
   brew install python@3.11
   ```
   (Use 3.10 or 3.11 - Coqui TTS does not install on 3.12+. `python.org`'s own macOS installer is an equally fine
   alternative to Homebrew here.)

3. **Clone the repo and set up a virtual environment:**
   ```
   git clone https://github.com/frifox/sts.git
   cd sts
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
   `(.venv)` should appear at the start of the prompt - reactivate with the
   same `source` command in any new terminal tab/window.

4. **Install the project:**
   ```
   pip install -e ".[xtts]"
   ```
   `pip install torch` on macOS installs a universal build with Apple
   Silicon (MPS) support already included - no separate CUDA-style index-url
   step like Windows/Linux. Verify MPS is available on Apple Silicon:
   ```
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```
   This should print `True`. On Intel Macs it'll print `False` - expected,
   use `--device cpu` there.

5. **`sounddevice`'s PortAudio dependency is bundled in its wheel** - no
   separate `brew install portaudio` needed in the normal case. If capture
   fails to find your input device anyway, that's the fallback to try.

6. **Run it** - see **First run, on any OS** at the end of this section.
   Two macOS-specific prompts to expect the first time: a
   **"Accept incoming network connections?"** firewall prompt for the
   caption web server (allow it), and a **microphone permission** prompt for
   Terminal/iTerm under System Settings -> Privacy & Security -> Microphone
   if `--list-devices` doesn't show your input device.

### Fresh Linux install

Commands below are for Debian (current stable; `apt`) specifically - not
Ubuntu, which diverges on the driver step in particular. Substitute your
distro's package manager (`dnf`, `pacman`, ...) if you're on something else;
the Python side is identical everywhere.

1. **Python 3.10 or 3.11 (not 3.12+), git, and PortAudio.** Linux is the one OS here where
   `sounddevice`'s PortAudio dependency is **not** bundled in the wheel and
   needs installing separately, or audio capture will fail to import.
   Debian 12 (bookworm) ships Python 3.11, which works. Debian 13 and newer
   ship 3.13, which Coqui TTS can't install on - use `uv` or `pyenv` to get
   3.11 there (e.g. `uv venv --python 3.11 .venv`).
   ```
   sudo apt update
   sudo apt install python3 python3-venv python3-pip git libportaudio2
   ```

2. **NVIDIA driver** (skip if using `--device cpu`). Debian keeps
   proprietary drivers out of its default repos, so `contrib` and
   `non-free-firmware` need enabling first. Check your release's codename
   with `cat /etc/os-release` (`VERSION_CODENAME=`), then edit
   `/etc/apt/sources.list` (or the file(s) under
   `/etc/apt/sources.list.d/`) so each `deb` line for that release includes
   them, e.g. for a line that currently reads
   `deb http://deb.debian.org/debian <codename> main`:
   ```
   deb http://deb.debian.org/debian <codename> main contrib non-free non-free-firmware
   ```
   Then:
   ```
   sudo apt update
   sudo apt install nvidia-driver linux-headers-$(uname -r)
   sudo reboot
   ```
   `nvidia-driver` pulls in the kernel module via DKMS, built against
   the headers package for your currently running kernel - if you update
   the kernel later, `linux-headers-$(uname -r)` needs reinstalling for the
   new one so the module rebuilds. After reboot, verify:
   ```
   nvidia-smi
   ```
   If that fails to find a device, check `lsmod | grep nouveau` - the
   open-source driver conflicting with the proprietary one is the usual
   cause; the `nvidia-driver` package normally blacklists it automatically,
   but a leftover load from before that took effect needs a further reboot
   (or `sudo modprobe -r nouveau` if nothing has it in use yet). As on
   Windows, you don't need the full CUDA Toolkit - just the driver;
   PyTorch's wheel brings its own CUDA runtime.

3. **Clone the repo and set up a virtual environment:**
   ```
   git clone https://github.com/frifox/sts.git
   cd sts
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install PyTorch with CUDA support explicitly**, same reasoning as the
   Windows step - install from the CUDA index rather than relying on
   whatever a plain `pip install torch` happens to resolve to:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Should print `True` and your GPU's name; if `False`, recheck the driver
   (step 2) before going further.

5. **Install the project:**
   ```
   pip install -e ".[xtts]"
   ```
   The Qwen3-TTS backend needs its own virtualenv (`pip install -e ".[qwen3]"`), because
   its transformers requirement conflicts with XTTS's.

6. **Run it** - see **First run, on any OS** below. Linux generally has no
   OS-level app-permission prompts for mic/network access
   the way Windows/macOS do; if `--list-devices` doesn't list your input
   device, check `arecord -l` (or your audio server's equivalent) to confirm
   the OS itself sees the device, and that your user is in the `audio`
   group on distros that gate device access that way. If you have `ufw`
   enabled and can't reach the caption page from another machine on the
   network, `sudo ufw allow 8765` (or your `--web-port`).

### First run, on any OS

```bash
sts --list-devices                 # confirm your mic/speakers show up
sts --web
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
sts --default-voice "Craig Gutsy"   # a different built-in voice (default: Adde Michal)
sts --accent-voice "Andrew Chipper" --accent-strength 1.0
                                    # steer cloned voices' accent toward a built-in speaker
sts --clone reference.wav          # stretch goal: clone the speaker's voice
                                    # from a 10-20s reference clip instead of
                                    # using the default TTS voice (one-off,
                                    # not saved anywhere)
sts --clone reference.wav --clone-name "Pastor Ivan"
                                    # same, but also saves it into the voice
                                    # library so it's picked from the
                                    # dropdown in later sessions without
                                    # reprocessing the file - zero-shot is
                                    # often good enough on its own
sts --web --web-port 8765          # also serve live captions at :8765
                                    # (open it full-screen / kiosk mode on a
                                    # second monitor or projector)
sts --web --tts-backend qwen3 --voice-library-dir /data/sts-voices
                                    # Qwen3-TTS backend + a custom voice
                                    # library location

# import a LoRA adapter trained via training/qwen3_lora/ (see its README):
sts-import-voice "Pastor Ivan" pastor_ivan ./checkpoint-epoch-9 \
  --lora-scale 0.3 --voice-library-dir /data/sts-voices
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The glossary term-matching, Bible reference detection/lookup, the
start/stop/reset/train control state machine (`sts.control`), the voice
library (`sts.tts.voice_library`, including LoRA entries), and the
cuda/cpu/mps device-mapping logic (`sts.devices`) are unit tested here — the
audio/ASR/translation/TTS stages, and everything under
`training/qwen3_lora/`, need real hardware (mic, GPU) and aren't exercised
by the test suite.
