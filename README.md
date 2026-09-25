# STS: Speech To Speech

Live speech-to-speech pipeline: listens to a sound card feed of a Russian
speaker (pulpit audio, Baptist Christian sermon context), transcribes,
translates to English, and speaks the translation back in a natural voice.

Everything runs locally against on-device models — no audio leaves the
machine, no per-minute API costs, and it works without internet once the
models are downloaded. It runs on an NVIDIA GPU (`--device cuda`) or an
Apple Silicon Mac (`--device mps`) — see **Apple Silicon / Metal** below.

## Pipeline

```
sound card or browser mic -> live captions (ru) -> MADLAD-400 translation (en) -> XTTS / Qwen3-TTS (en) -> sound card
                                    |                          |                             \-> browser (optional)
                                    v                          v
                              original text             translated text
                                    \__________________________/
                                                 |
                                   websocket -> caption page (any number of screens)
```

- **Capture** (`sts.audio.capture`) — reads the input device via
  `sounddevice`, resampling if the device can't do 16 kHz itself. Instead of
  a local device, a browser tab can send its own microphone (see Operator
  controls). On macOS, sts checks the Microphone privacy permission at
  start-up: without it macOS hands every input device's audio over as
  silence, so sts prints how to fix it, opens System Settings > Privacy &
  Security > Microphone, and shows the same warning on the caption page.
- **Captions** (`sts.transcript`, `sts.asr.gigaam_asr`) — Sber's
  [GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3) `e2e_rnnt`, a
  Russian-only speech recognizer with punctuation. Captions don't wait for
  the speaker to pause: every 500 ms (`--caption-interval-ms`) the audio that
  isn't settled yet is transcribed again, so the model keeps hearing more
  context and can revise what it heard. A sentence is **locked** once two
  more sentences have followed it, or 30 s after it ended
  (`--final-after-sentences N`, `--final-after-seconds S`; 0 turns either
  rule off), and everything still open is locked once nothing new has been
  recognized for 10 s (`ASRConfig.final_after_silence_seconds`; the caption
  page counts down the last 5 s). A locked sentence's audio is dropped, so
  it's never reworded again. If the model starts a sentence with a lowercase
  letter, it's joined back onto the previous one (the speaker only paused) -
  even a locked one, so a locked sentence becomes **final** once the
  sentence after it is locked too. Only final sentences go on to translation
  and voicing; everything before that is shown on the caption page as it's
  heard. The price of waiting for text to settle is delay: in testing, a
  sentence became final about 10-20 s after it first appeared (with the
  defaults), and was translated within a second after that.
- **Translation** (`sts.translate.translator`) — MADLAD-400 10B (`ru->en`),
  chosen over NLLB-200, HY-MT1.5 and others after comparing them on a
  recorded sermon (see **Comparing translation models**): it renders Bible
  passages in standard English wording (mostly KJV) without the reading
  being announced, and was the most accurate on the preacher's own speech.
  `--translation-model google/madlad400-3b-mt` (or `-7b-mt`) trades a little
  accuracy for memory. Text is translated one sentence at a time; common
  one-word replies ("Аминь", "Аллилуйя") use a fixed table. The glossary
  (`sts.translate.glossary`, placeholder substitution of Baptist/Christian
  terms) is off by default: MADLAD gets those terms right on its own, and
  mangles the placeholders.
- **TTS** (`sts.tts`) — two swappable backends. `--tts-backend` picks one; by
  default it's whichever is installed in the Python environment you run from
  (they need conflicting `transformers` versions, so they live in separate
  virtualenvs): `.venv-xtts/bin/sts` uses XTTS and `.venv-qwen/bin/sts` uses
  Qwen3-TTS.
  - `xtts` (default) — Coqui XTTS v2. Natural built-in speakers, plus
    zero-shot cloning from a short reference clip. Live playback streams:
    each sentence starts playing about a second after synthesis begins.
  - `qwen3` — Qwen3-TTS (Apache-2.0). Zero-shot cloning, including
    cross-lingual (Russian reference clip -> English speech), preset voices
    (Aiden, Ryan) from its CustomVoice model, and **Create Voice** from a
    written description (its VoiceDesign model). A clip is generated whole
    before it plays.

  Both back a persistent, cross-session **voice library**
  (`sts.tts.voice_library`, the Voice dropdown on the caption page — see
  Operator controls below) rather than a one-off in-memory clone.

All stages run on separate threads connected by queues (`sts.pipeline`), so a
slow translation or TTS call doesn't stall audio capture. On Apple Silicon,
GPU work from those threads is serialized through one lock (`sts.gpu`),
since PyTorch's Metal backend aborts the process when two threads submit
work at once.

- **Caption page** (`sts.web`) — a FastAPI/uvicorn server, on by default
  (`--no-web` turns it off; port 8765, `--web-port`), serves a full-screen
  caption page and pushes text to it over a websocket as it's produced. Any
  number of screens can have it open; they stay in sync.
  - **Landscape** (default): original text in the top half, translation in
    the bottom half, each a flowing paragraph. **Portrait**: side-by-side
    columns, one row per sentence so each translation lines up with its
    original (for a portrait-mounted screen). Picked per screen in the
    Translate dropdown.
  - The newest final sentence in each half is bright and older ones fade.
    The text that isn't final yet appears faded below the final sentences,
    updated as it's heard; the Caption dropdown's **Unfinished text:
    Show/Hide** turns it off on that screen. A "Finalizing in N s" countdown
    appears while nothing new is being recognized.
  - Older lines are dropped once they no longer fit, so the page never
    accumulates unbounded DOM over a long service. A half is hidden while
    its stage is off, and "System Idle" shows when both are.
  - The header shows input and output level meters and, per stage, a
    real-time factor (processing time / audio duration; at 1.0x or above
    that stage is falling behind).
- **Bible reference detection** (`sts.bible`) — the Russian transcript is
  scanned for a spoken reference ("Матфея 1:15", "Первое послание к
  Коринфянам, глава 13, стих 4", "Плач Иеремии 3:22-23", chapter/verse as
  digits or Russian number words). A match looks up the exact verse in a
  bundled, fully offline Russian Synodal + English KJV (Pure Cambridge
  Edition) dataset and shows it as a banner over each half for 25 s; a chip
  in the corner brings the last one back. Book-name matching is stem-based
  (all 66 books) rather than an exhaustive alias list, and Psalms gets its
  chapter number converted between the Synodal (Septuagint) and KJV
  (Masoretic) numbering — see `sts/bible/lookup.py`. This is best-effort:
  it only fires when the preacher says the reference, the ASR gets it
  right, and the parser recognizes the phrasing. A passage read without
  announcing it is still translated well, because MADLAD recognizes
  Scripture on its own.
- **Operator controls** (`sts.control`, the caption page's header) — a live
  feed runs across multiple speakers/performances in a session (music,
  announcements, one voice at a time when someone *is* speaking). Captions
  and translation run **by default** - they're silent, so leaving them
  running through a musical number is harmless. *Voicing* is different
  (audio played during music or an announcement would not be harmless) and
  defaults to **off**. Hovering a button or a level meter opens its
  dropdown:
  - **Input meter** — the input device (microphone / line in), switched
    live, and **Use this browser's mic** to stream that browser tab's
    microphone in instead. Browsers only allow the microphone on a secure
    page, so for another device on the network serve the page over HTTPS
    too (`--https-port`, see Usage).
  - **Caption** — turns recognition on/off (e.g. through a long musical
    number). Turning it off makes whatever was already heard final, so it's
    still translated, and then nothing more is transcribed. Dropdown: model
    and device in use, and Unfinished text Show/Hide.
  - **Translate** — turns translation on/off, independently of captions.
    Dropdown: model and device, and the Landscape/Portrait choice.
  - **Voice** — turns synthesized speech on/off; starts off, so captions can
    run all service while voicing is switched on only when someone is
    preaching. Turning it off cuts the current sentence short and drops
    anything queued. Dropdown:
    - **Voice** list, in two groups. **Default**: the bundled **Default**
      voice (a calm English preacher voice designed with Qwen3's Create
      Voice, shipped in `src/sts/bundled_voices/default/`; the voice Reset
      returns to), then the built-in voices - 13 for XTTS
      (`TTSConfig.voice_groups`, from XTTS's 58), Aiden and Ryan for Qwen3.
      **Custom**: voices you trained or designed and saved; an unsaved
      just-trained voice shows as "New voice (unsaved)". Picking a voice
      switches to it immediately. A session starts on whichever voice was
      last picked (remembered via `--config-db`) unless `--default-voice` or
      `--clone` says otherwise.
    - **Train voice** — captures the live speaker's audio for a zero-shot
      clone; stops at 18 s or on a second click. The clone becomes active
      right away so you can listen to it, but isn't saved yet.
    - **Save Voice** — names the active trained voice and stores it in the
      library, reusable in later sessions (an existing name is replaced).
      Click Reset before the next speaker so their audio doesn't get folded
      into the previous one's sample.
    - **Create Voice** (Qwen3 only) — designs a new voice from a written
      description: **Generate**, try it on any text with **Play sample**,
      regenerate until one fits (each generation is a different voice), then
      name it and **Save**. The VoiceDesign model is loaded only while the
      dialog is open.
    - **Delete** (while a Custom voice is selected) — removes it from the
      library on disk after confirmation; the default voice takes over if it
      was active. Saving your own voice named "Default" replaces the bundled
      one; deleting that copy brings the bundled one back.
  - **Output meter** — the output device, switched live (the next sentence
    plays on it); a **Volume** slider for the local sound card; **Stream
    voice to this browser**, to hear the voice in that browser tab too
    (remembered per browser); and **Test Output**, a short pre-recorded clip
    played on the output device even with Voice off.
  - **Reset** — for a new speaker/performance: clears the captions on every
    connected screen, drops any text in flight through the pipeline, and
    forgets the current cloned voice.

  Device lists refresh each time a dropdown opens, so a device plugged in
  after start-up shows up. The input/output devices, volume and active voice
  are remembered across restarts (`--config-db`, default `sts.db`); an
  explicit `--input-device` / `--output-device` always wins over what was
  saved. Per-screen choices (Landscape/Portrait, Unfinished text, voice
  streaming) are stored in that browser.

  All of this is relayed over the same websocket the captions arrive on
  (`{"action": "start"}`, `{"action": "translate_off"}`,
  `{"action": "voice_on"}`, `{"action": "save_voice", "name": "..."}`,
  `{"action": "select_speaker", "speaker_id": "..."}`, etc.). Library
  entries live under `--voice-library-dir` (default `voices/`) as
  `<speaker-id>/reference.wav` + `meta.json` — the raw reference clip and
  transcript, not a backend's in-memory prompt object, so they're portable
  across restarts and backend swaps.

  This is zero-shot clone caching, not true model fine-tuning — a
  per-speaker model needs far more data (10-30 min of clean audio) and
  training time than a live session gives you. For that, see **True LoRA
  training**, below.

## GPU requirements

All three components load simultaneously regardless of which caption-page
toggles are on, so the last row is what actually matters for picking
hardware.

| Part | Models | Rough VRAM |
|---|---|---|
| Captions (ASR + translation) | GigaAM-v3 (~240M, ~0.8GB) + MADLAD-400 10B (bf16; 3B with `--translation-model google/madlad400-3b-mt`) | ~22GB (3B: ~7GB) |
| TTS, zero-shot | XTTS v2 (~460M) or Qwen3-TTS-Base (0.6B/1.7B) | ~2-4GB |
| TTS, trained (LoRA) | same base model, adapter merged in | same as zero-shot per speaker; each additional *cached* trained speaker is a full extra model load, not a small increment (see `sts/tts/qwen3.py`) |
| **Whole pipeline together** | all of the above loaded at once | **~25GB with MADLAD 10B + XTTS (~10GB with MADLAD 3B); more with Qwen3-TTS 1.7B** |

These are estimates from parameter counts; the only measured figure is
from before MADLAD (~6GB with NLLB-200-1.3B + XTTS on an 8GB RTX 2070). An
8GB card no longer fits the pipeline even with MADLAD 3B: use a 12GB+ card
with `--translation-model google/madlad400-3b-mt`, and a 32GB card (or Apple
Silicon with 32GB+ of unified memory) for the default 10B. Qwen3-TTS's
1.7B variant or several cached trained speakers need a few GB more; its
0.6B variant, or XTTS, buys back headroom.

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
loading code mirrors that patch's inference script line-for-line, so it
matches known-working reference code rather than guessing at the API.

## Apple Silicon / Metal

`--device mps` runs ASR, translation, and TTS on the Apple GPU, and sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` so any PyTorch op MPS doesn't yet implement
falls back to CPU instead of crashing, at some speed cost for that op.
`--asr-device cpu` keeps just ASR on the CPU if GigaAM turns out not to
run well on MPS.

Qwen3-TTS uses float16 + SDPA attention on `mps` rather than CUDA's
bfloat16 + flash-attention (flash-attn has no Metal build; MPS's bfloat16
support has historically been the less mature of the two float types) — see
`_dtype_for_device`/`_attn_impl_for_device` in `sts/tts/qwen3.py`. XTTS has
no such split; `.to("mps")` is all it needs.

The whole pipeline runs on an Apple Silicon Mac with the default models
(MADLAD-400 10B, XTTS or Qwen3-TTS 1.7B); unified memory means there's no
separate VRAM ceiling, but the models need about 25GB of it together, so a
Mac with 32GB+ is the floor for the defaults (use MADLAD 3B on a smaller
one). Two Mac-specific behaviors:

- GPU work from the pipeline's threads is serialized through one lock
  (`sts.gpu`): Metal aborts the process ("A command encoder is already
  encoding...") when two threads submit work at once.
- macOS gives an app without Microphone permission silence instead of an
  error. The permission belongs to the app sts was started from (Terminal,
  iTerm, an IDE), and sts checks it at start-up - see Capture, above.

```bash
sts --device mps                                # XTTS backend
sts --device mps --tts-backend qwen3             # Qwen3-TTS backend (or just run it from .venv-qwen)
```

## Older NVIDIA GPUs (pre-Ampere) and Windows

`--device cuda` works on Windows the same as Linux — PyTorch, Coqui TTS,
and `sounddevice` all support it natively.

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

On VRAM: an 11GB card (e.g. a GTX 1080 Ti) can't hold the default MADLAD
10B; with `--translation-model google/madlad400-3b-mt` and XTTS it's
workable but tight. See the GPU requirements breakdown earlier in this
README for the per-component numbers.

## Setup

Requires Python 3.10 or 3.11 (not 3.12+: Coqui TTS does not support it) and a GPU for real-time performance — NVIDIA/CUDA
(`--device cuda`), Apple Silicon (`--device mps`, see above), or CPU (works,
but will lag). Supported on Windows, macOS, and Linux.

```bash
git clone https://github.com/frifox/sts.git
cd sts
python3.11 -m venv .venv-xtts && .venv-xtts/bin/pip install -e ".[xtts,dev]"
python3.11 -m venv .venv-qwen && .venv-qwen/bin/pip install -e ".[qwen3,dev]"   # optional: the Qwen3-TTS backend
```

The two TTS backends need conflicting `transformers` versions, so each gets
its own virtualenv; run `sts` from whichever one has the backend you want.
`translate.py` needs `.venv-qwen` (a newer `transformers`).

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
   python -m venv .venv-xtts
   .venv-xtts\Scripts\activate
   ```
   You should see `(.venv-xtts)` at the start of the prompt from here on - every
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
   The Qwen3-TTS backend needs a separate virtualenv (`.venv-qwen`,
   `pip install -e ".[qwen3]"`; its transformers requirement conflicts with
   XTTS's)
   — see **Older NVIDIA GPUs (pre-Ampere) and Windows** above for what that
   backend needs on a card like a GTX 1080 Ti specifically.

7. **Run it** - see **First run, on any OS** at the end of this section.
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
   python3.11 -m venv .venv-xtts
   source .venv-xtts/bin/activate
   ```
   `(.venv-xtts)` should appear at the start of the prompt - reactivate with the
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
   the terminal app you run sts from. If access was denied earlier, macOS
   won't ask again and the input records silence: sts detects this, says
   so, and opens System Settings -> Privacy & Security -> Microphone, where
   you enable it for that app and restart sts.

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
   3.11 there (e.g. `uv venv --python 3.11 .venv-xtts`).
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
   python3 -m venv .venv-xtts
   source .venv-xtts/bin/activate
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
   The Qwen3-TTS backend needs its own virtualenv (`.venv-qwen`,
   `pip install -e ".[qwen3]"`), because its transformers requirement
   conflicts with XTTS's.

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
sts
```

Model weights (GigaAM, MADLAD-400, XTTS, Silero VAD) download automatically on
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
sts -v                              # log pipeline activity: segments, timings, device events
sts --translation-model google/madlad400-3b-mt
                                    # a smaller translation model (see GPU requirements)
sts --asr-device cpu --tts-device cuda:1
                                    # pin single stages to a device (also --translation-device)
sts --default-voice "Craig Gutsy"   # a different built-in voice (default: Adde Michal)
sts --accent-voice "Andrew Chipper" --accent-strength 1.0
                                    # steer cloned voices' accent toward a built-in speaker
sts --pause-s 1.0 --sentence-pause-s 0.35
                                    # silence between spoken translations / between the
                                    # sentences within one (each sentence is spoken separately)
sts --final-after-sentences 1 --final-after-seconds 15
                                    # finalize captions sooner - less delay before
                                    # translation/voice, more chance a recognition
                                    # error is locked in
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
sts --web-port 8765                # live captions are served at :8765 by default
                                    # (open it full-screen / kiosk mode on a
                                    # second monitor or projector); --no-web
                                    # runs without the page, printing text only
sts --https-port 443               # also serve the page over HTTPS, needed to use
                                    # a phone's or laptop's browser mic as the input
                                    # (browsers only allow the mic on a secure page).
                                    # Self-signed cert in certs/ by default (--tls-cert,
                                    # --tls-key); make one with:
                                    #   openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/key.pem \
                                    #     -out certs/cert.pem -days 3650 -subj '/CN=sts'
sts --voice-library-dir /data/sts-voices
                                    # a custom voice library location
sts --config-db /data/sts.db       # where the operator's input/output
                                    # device, volume and active voice are remembered across
                                    # restarts (default: sts.db in the working
                                    # directory; pass '' to disable persistence)

# import a LoRA adapter trained via training/qwen3_lora/ (see its README):
sts-import-voice "Pastor Ivan" pastor_ivan ./checkpoint-epoch-9 \
  --lora-scale 0.3 --voice-library-dir /data/sts-voices
```

## Sound quality of spoken clips

Both TTS backends' output is brought to one speech level
(`TTSConfig.target_level`, default 0.10 RMS, about -20 dBFS; `None` disables
it), so a short phrase doesn't sound distant next to a long sentence. Qwen3
clips are measured and matched whole. XTTS streams, so a sentence starts
playing before most of it exists: its gain comes from the running level of
that voice's earlier sentences (XTTS speaks a given voice at a steady
level), kept per voice, and stays fixed through the sentence. Only a voice's
first sentence is judged from its own first 0.8 s, and a clip shorter than
that (e.g. "Amen.") is matched exactly.

With Qwen3, a cloned voice also copies its reference's recording conditions: a
muffled or distant sample gives a muffled voice. Voices are best trained from a
close, clean recording. `--qwen-quality-retries N` (off by default) regenerates a
take that is clearly duller than that voice's own recent ones, at the cost of an
extra generation each time.

### Streaming and catch-up

With XTTS, each sentence starts playing about one second of audio after
synthesis begins, and the rest is generated while it plays, instead of the
whole sentence being rendered first.

**Currently disabled** (`catchup_max_rate = 1.0` in `AudioConfig`): all
speech plays at normal speed. Set it back to `1.35` to re-enable what
follows.

If the voice runs late - sentences start being spoken long after their
captions became final - playback speeds up, keeping the pitch, until it has
caught up, then returns to normal speed. Lateness is measured in seconds
from the moment a sentence became final, so the wait before a caption is
final doesn't count, and neither does a long sentence simply waiting its
turn. It starts once a sentence is 8 s late, ramps from 1.1x to at most
1.35x as lateness grows, and stops once a sentence starts within 3 s. The
speed is chosen at the start of each sentence and kept for all of it. In the
rare case that a full minute of synthesized speech is waiting even so, the
oldest waiting sentences are skipped. The thresholds are the `catchup_*`
and `max_backlog_s` settings in `AudioConfig` (`sts/config.py`).

## Caption diagnostic (sts-caption)

`sts-caption` runs the live pipeline's captions on their own, in a terminal:
no web page, translation or voice. Final sentences are printed once, above
a blank line; below it is what may still change, one sentence per line,
redrawn in place. Press `x` to clear and start over. It takes the same
`--final-after-sentences`, `--final-after-seconds` and `--interval-ms`
settings as `sts` (named `--caption-interval-ms` there), and runs from the
same virtualenv:

```bash
.venv-xtts/bin/sts-caption --list-devices
.venv-xtts/bin/sts-caption --input-device 7
```

Piped to a file, only final sentences are written.

## Comparing translation models (translate.py)

`translate.py` transcribes the first minutes of a recording with GigaAM
(the live pipeline's ASR), splits it into sentences the way the live
translator does, and translates every sentence with each candidate model -
NLLB-200 (600M distilled, 1.3B distilled, 1.3B, 3.3B), MADLAD-400 (3B, 7B,
10B), Tencent HY-MT1.5 (1.8B, 7B) and Google TranslateGemma (4B, 12B, 27B).
The result is one text file with each Russian sentence followed by every
model's English, plus each model's load time and seconds per sentence:

```bash
.venv-qwen/bin/python translate.py                       # videos/k_neyman.mkv, first 5 min, every model
.venv-qwen/bin/python translate.py --models madlad hy-mt  # only labels containing these
.venv-qwen/bin/python translate.py --video other.mkv --minutes 10
```

Models are compared raw (no glossary, no context from earlier sentences,
greedy decoding). Results are cached per model in
`benchmark_results/translate/`, so a rerun only translates what's missing
(`--force` redoes them). All the sizes together are ~200 GB of downloads;
`--delete-downloads` removes each one it downloaded after its run.
TranslateGemma is gated: accept its license on huggingface.co and
`hf auth login` first. This comparison is what picked MADLAD-400 10B.

## Benchmarking hardware

`benchmark.py` times each pipeline stage - ASR (GigaAM-v3),
translation (MADLAD-400), and TTS (both backends), whichever backends are installed - using the exact
same model configs and device-selection logic `sts` itself runs, so the
numbers reflect real pipeline performance, not a synthetic benchmark of the
frameworks in isolation. Point of it: run it on different machines/OSes (an
old GPU vs a new one, a Mac's MPS backend, CPU-only) and diff the results to
see where one is actually faster - this is what decided the GPU upgrade
options in the church-service deployment this was built for.

```bash
python benchmark.py                       # auto-detect device, benchmark everything installed
python benchmark.py --device cpu          # force a device
python benchmark.py --asr-device cpu      # override just ASR (e.g. to match a VRAM-constrained deployment)
python benchmark.py --repeats 5           # timed repetitions per stage (default 3, plus 1 untimed warmup call)
python benchmark.py --tts-backend qwen3   # only benchmark one TTS backend instead of both
```

Each stage reports model load time, a warmup call (excluded from the timed
numbers, so first-call CUDA/MPS init doesn't skew them), and the
timed runs with mean/median - plus a real-time factor (processing time /
audio duration) for ASR and TTS, since that's what actually determines
whether a stage keeps up with a live speaker. A backend that isn't installed
in the current environment is reported as skipped, not an error. Results are
also written to `benchmark_results/<hostname>_<timestamp>.json` for
diffing between machines.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Run it from `.venv-xtts` (the XTTS-specific tests skip themselves
elsewhere). Covered without models or hardware: caption sentence locking
(`sts.transcript`), Bible reference detection and lookup, the operator
control state, the voice library (including LoRA entries), saved settings,
device mapping, capture resampling and device recovery, the browser mic,
playback, catch-up, loudness matching of streamed XTTS audio, the glossary
and text cleanup. Loading and running the actual ASR, translation and TTS
models, and everything under `training/qwen3_lora/`, needs the models and a
GPU, so the suite doesn't do it - use `benchmark.py` and `translate.py` for
those.
