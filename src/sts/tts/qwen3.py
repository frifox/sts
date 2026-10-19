"""Synthesizes English speech with Qwen3-TTS, either zero-shot cloning a
reference clip or using a real trained LoRA adapter for a repeat speaker.

Zero-shot API per github.com/QwenLM/Qwen3-TTS (Apache-2.0, `pip install
qwen-tts`): `model.create_voice_clone_prompt(ref_audio=..., ref_text=...,
x_vector_only_mode=...)` builds a reusable prompt once; `generate_voice_clone`
consumes it. `ref_text` is the reference clip's own transcript and
materially improves quality; `x_vector_only_mode` drops that requirement at
some quality cost, and is the fallback when no transcript is available.

LoRA loading (`_load_lora_model` and its helpers below) mirrors
`finetuning/infer_lora_custom_voice.py` from
github.com/cheeweijie/qwen3-tts-lora-finetuning (Apache-2.0), the community
patch used by `training/qwen3_lora/` to actually train these adapters - see
that directory for the training side. A LoRA speaker is selected with
`generate_custom_voice(speaker=...)`, not `generate_voice_clone`: the
trained checkpoint bakes the voice into a named speaker slot (with its own
embedding) inside the model rather than being conditioned on a reference
clip at synthesis time.

Each cached LoRA speaker is a second, full model load (base weights + the
adapter merged in) - unlike the zero-shot path, which reuses one base model
for every reference clip. On a memory-constrained consumer GPU, keeping more
than a couple of trained speakers active at once will add up; there's no
cache eviction here, only what `reset_voice()`/picking a different speaker
overwrites as the *active* one (cached entries stay loaded for fast
re-selection).

When no cloned/trained voice is active, a preset speaker from Qwen3-TTS's
separate CustomVoice model is used instead (default: Aiden, with a calm
"preacher" style instruction), so no reference clip is needed to get sound.
It's a second model load alongside the Base model.

This backend is unverified against real hardware in this repo (no GPU
available in dev) - the call shapes come directly from the two upstream
projects' example code, not from having actually run it.

Device support: `cfg.device` is passed straight through as `device_map`, so
"cuda", "cpu", and "mps" (Apple Silicon) all work as far as loading the
model goes. bfloat16 and flash-attention aren't requirements of the model
itself, though, and aren't even a safe default for every CUDA GPU: both need
Ampere or newer (compute capability >= 8) - an older card like a GTX 1080 Ti
(Pascal, compute 6.1) supports neither properly, and would fail to load or
run badly if forced onto that path. `_dtype_for_device`/`_attn_impl_for_device`
below check the actual GPU's compute capability (via
`sts.devices.supports_bf16_and_flash_attention`, kept import-light so it's
unit-testable without a GPU) and fall back to float16 + SDPA attention on
mps or an older/absent GPU, same reasoning as the Metal case either way.
"""

import gc
import hashlib
import re
import statistics
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:  # .venv-mlx: Qwen3-TTS runs on MLX instead (sts.tts.qwen3_mlx)
    Qwen3TTSModel = None

from sts.config import TTSConfig
from sts.devices import supports_bf16_and_flash_attention
from sts.audioquality import (
    clarity_db,
    level_gain,
    longest_pause_s,
    match_level,
    match_tone,
    tone_filter,
    tone_gains,
    tone_profile,
    trim_silence,
)
from sts.gpu import gpu_lock
from sts.log import vlog
from sts.tts.numbers import spell_numbers
from sts.tts.accent import ACCENT_DONORS, BUNDLED_DONOR
from sts.tts.english_reference import (  # noqa: F401 (clone_conditioning, transcript_is_english: used by callers)
    DONOR_TEXT,
    ENGLISH_BRIDGE_TEXT,
    accent_donor,
    cached_reference_path,
    clone_conditioning,
    transcript_is_english,
)
from sts.tts.qwen3_engine import resolve_engine
from sts.tts.voice_library import VoiceEntry


ENGLISH_BRIDGE_TAKES = 3
# How much like the speaker a take with an accent donor must still sound (cosine
# similarity of speaker embeddings: takes that kept the speaker's voice measured
# 0.97-0.99, ones that slid into the donor's 0.91-0.93).
MIN_SPEAKER_SIMILARITY = 0.96
BRIDGE_SECONDS_PER_CHAR = 0.058  # the passage's usual pace (about 11 s)


# Streaming (MLX engine, TTSConfig.qwen3_stream): a sentence is played while
# it's still being generated, rather than once it's whole. Nothing plays for
# the first STREAM_HOLD_S: a short sentence is often whole by then (generating
# runs ~3x faster than speech), and is checked and, if bad, regenerated before
# anyone hears it; a longer one starts playing then and streams the rest.
STREAM_HOLD_S = 2.0
STREAM_PIECE_S = 0.32  # how much audio each streamed piece holds (4 codec frames)
# A take that runs past RAMBLE_FACTOR x (+ RAMBLE_MARGIN_S) the length its text
# should take is rambling - babbling on past the end of the text. Caught while
# still held back it's regenerated; once playing, it's cut off with a short fade.
# A take that falls silent for STALL_S once it has started talking, before
# STALL_UNDER of its expected length is spoken, has stopped talking partway -
# Drama 3 sometimes does, generating silence to the end instead: regenerated if
# still held back, cut off (no silence played) if already playing.
STALL_S = 1.5
STALL_UNDER = 0.75
RAMBLE_FACTOR = 1.6
RAMBLE_MARGIN_S = 0.8
STREAM_FADE_S = 0.1  # kept back from each piece so a cut-off can fade out
TYPICAL_SECONDS_PER_CHAR = 0.062  # until the voice has a pace of its own (see _pace_history)

CODEC_FRAMES_PER_SECOND = 12  # the 12 Hz codec: 12 generated tokens per second of audio
MIN_CAP_SECONDS = 8.0  # the length cap for very short texts ("Amen.")
MIN_CHARS_TO_JUDGE = 20  # shorter texts vary too much in pace to judge


def min_plausible_seconds_per_char(pace_history, floor: float) -> float:
    """Shortest believable time per character. After a few accepted sentences the
    bar is 75% of the voice's own median pace (sentence-to-sentence pace varies
    by roughly +-10%), never below `floor`."""
    if len(pace_history) < 5:
        return floor
    return max(floor, 0.75 * statistics.median(pace_history))


def clarity_bar(history, margin_db: float) -> float | None:
    """The clarity a take must reach to be kept: this voice's own median minus a
    margin, or None (no judgement) until it has a few takes of its own. A voice
    cloned from a muffled recording is muffled every time and mustn't be regenerated
    forever, and clarity varies by voice and sentence, so there is no absolute bar."""
    if len(history) < 3:
        return None
    return statistics.median(history) - margin_db


def speech_length_plausible(text: str, seconds: float, min_spc: float, max_spc: float) -> bool:
    """False when the audio is far shorter (words skipped) or longer (babble)
    than `text` could plausibly take to say. Very short texts are exempt."""
    chars = len(text.strip())
    if chars < MIN_CHARS_TO_JUDGE:
        return seconds > 0
    return min_spc * chars <= seconds <= max_spc * chars


def _cuda_supports_bf16_and_flash_attention() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return supports_bf16_and_flash_attention(major)


def _dtype_for_device(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if _cuda_supports_bf16_and_flash_attention() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _attn_impl_for_device(device: str) -> str:
    if device == "cuda" and _cuda_supports_bf16_and_flash_attention():
        return "flash_attention_2"
    return "sdpa"


class Qwen3TTSSynthesizer:
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.engine = resolve_engine(cfg.qwen3_engine)
        if self.engine == "mlx":
            from sts.tts.qwen3_mlx import MLXQwen3TTSModel

            self._model_class = MLXQwen3TTSModel
        elif Qwen3TTSModel is None:
            raise ImportError("qwen_tts isn't installed (pip install -e '.[qwen3]'); "
                              "or use the MLX engine from .venv-mlx (pip install -e '.[mlx]')")
        else:
            self._model_class = Qwen3TTSModel
        vlog("tts", f"Qwen3-TTS engine: {self.engine}")
        self.model = self._model_class.from_pretrained(
            cfg.qwen3_model_name,
            device_map=cfg.device,
            dtype=_dtype_for_device(cfg.device),
            attn_implementation=_attn_impl_for_device(cfg.device),
        )
        self._custom_model = self._load_custom_model() if cfg.qwen3_preload_presets else None
        self.default_voice = cfg.resolve_voice(cfg.default_voice)
        # MPS/CUDA models aren't safe to run from two threads at once, and the
        # voice-design dialog synthesizes previews while the live pipeline runs.
        # (On mps this is the process-wide GPU lock shared with translation.)
        self._lock = gpu_lock(cfg.device) if cfg.device == "mps" else threading.Lock()
        self._pace_history: deque = deque(maxlen=20)  # seconds/character of recently accepted sentences
        self._clarity_history: deque = deque(maxlen=20)  # clarity (dB) of recently accepted takes
        self._tone_history: deque = deque(maxlen=20)  # tone_profile of recently voiced sentences
        self._design_model = None
        self._design_load_lock = threading.Lock()
        self._prompt = None
        self.cloned_speaker_wav: str | None = None  # set whenever a zero-shot clone is active
        self._lora_models: dict[str, tuple[Qwen3TTSModel, str]] = {}  # speaker_id -> (model, speaker_name)
        self._active_lora_speaker_id: str | None = None

    def memory_bytes(self) -> int:
        """Every Qwen3 model currently loaded: the base cloning model, the
        preset-voice model, the voice designer, and trained (LoRA) voices."""
        from sts.gpu import model_bytes

        loras = [model for model, _ in self._lora_models.values()]
        return model_bytes(self.model, self._custom_model, self._design_model, *loras)

    def _sampling(self, text: str) -> dict:
        """Sampling settings that keep delivery consistent between sentences, and
        a length cap: a take that babbles on past the end of its text would
        otherwise run to the model's default 2048 codec frames (almost 3 minutes
        of audio, many minutes of generating) before the length check sees it."""
        seconds = max(MIN_CAP_SECONDS, len(text) * self.cfg.qwen3_max_seconds_per_char)
        return dict(
            temperature=self.cfg.qwen3_temperature,
            subtalker_temperature=self.cfg.qwen3_temperature,
            top_k=self.cfg.qwen3_top_k,
            subtalker_top_k=self.cfg.qwen3_top_k,
            max_new_tokens=int(seconds * CODEC_FRAMES_PER_SECOND),
        )

    # -- voice design ("Create Voice" dialog) --------------------------------

    @property
    def design_loaded(self) -> bool:
        return self._design_model is not None

    def design_reference(self, instruct: str, ref_text: str) -> tuple[np.ndarray, int]:
        """Invents a voice from `instruct` and returns it speaking `ref_text`.

        Each call designs a *different* voice, so the returned clip is used as
        the reference for cloning (`preview`, and the saved library voice)
        rather than re-running the description for every sentence.
        """
        with self._design_load_lock:
            if self._design_model is None:
                # Download first, outside the GPU lock, so a multi-GB first-time
                # download doesn't stall live translation/synthesis.
                from huggingface_hub import snapshot_download

                snapshot_download(self.cfg.qwen3_design_model_name)
                with self._lock:
                    self._design_model = self._model_class.from_pretrained(
                        self.cfg.qwen3_design_model_name,
                        device_map=self.cfg.device,
                        dtype=_dtype_for_device(self.cfg.device),
                        attn_implementation=_attn_impl_for_device(self.cfg.device),
                    )
        with self._lock:
            wavs, sample_rate = self._design_model.generate_voice_design(
                text=ref_text, instruct=instruct, language="English"
            )
        return np.asarray(wavs[0], dtype=np.float32), sample_rate

    def unload_designer(self) -> None:
        """Frees the VoiceDesign model when the dialog closes (it's a multi-GB model)."""
        with self._design_load_lock:
            self._design_model = None
        gc.collect()
        with self._lock:
            if self.engine == "mlx":
                from sts.tts.qwen3_mlx import clear_cache

                clear_cache()
            elif self.cfg.device == "mps":
                torch.mps.empty_cache()
            elif self.cfg.device == "cuda":
                torch.cuda.empty_cache()

    def preview(self, text: str, reference_wav_path: str, ref_text: str) -> tuple[np.ndarray, int]:
        """Speaks `text` in the voice of a reference clip, without changing the active voice."""
        with self._lock:
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=reference_wav_path, ref_text=ref_text, x_vector_only_mode=False
            )
            wavs, sample_rate = self.model.generate_voice_clone(
                text=text, language="English", voice_clone_prompt=prompt, non_streaming_mode=True, **self._sampling(text)
            )
        return np.asarray(wavs[0], dtype=np.float32), sample_rate

    def clone_voice(self, reference_wav_path: str, ref_text: str | None = None) -> None:
        self._pace_history.clear()
        self._clarity_history.clear()
        self._tone_history.clear()
        conditioning = clone_conditioning(self.cfg.qwen3_clone_mode, ref_text)
        if conditioning == "bridge":
            prompt_wav, prompt_text = str(self._english_reference(reference_wav_path)), ENGLISH_BRIDGE_TEXT
        else:
            prompt_wav, prompt_text = reference_wav_path, ref_text
        with self._lock:
            self._prompt = self.model.create_voice_clone_prompt(
                ref_audio=prompt_wav, ref_text=prompt_text, x_vector_only_mode=conditioning == "xvector"
            )
        vlog("tts", "voice conditioning: " + {
            "icl": "clip + transcript",
            "bridge": "an English clip in the voice + its text",
            "xvector": "speaker embedding only",
        }[conditioning])
        self.cloned_speaker_wav = reference_wav_path
        self._active_lora_speaker_id = None

    def _english_reference(self, reference_wav_path: str) -> Path:
        """The voice of a non-English reference clip speaking ENGLISH_BRIDGE_TEXT
        in an American accent, for cloning from instead of the clip itself
        (zero-shot either way).

        Cloned from a Russian clip + its Russian transcript, the model carries on
        as that Russian speaker: English words read with Russian sounds, as if
        spelled out letter by letter, digits read out in Russian, stray Russian-
        sounding syllables before a sentence. So the voice first says an English
        passage; that clip and its text are then the reference, and every
        sentence imitates English speech in the speaker's voice.

        With an accent donor (`cfg.qwen3_accent_donor`, one of ACCENT_DONORS),
        the passage is spoken with the speaker's embedding (their voice) but
        imitating that preset American voice saying DONOR_TEXT: with the
        embedding alone (no donor), the accent is left to chance, and a
        reference made that way once came out British. A take can drift into
        the donor's own voice instead, so one that sounds less like the speaker
        than MIN_SPEAKER_SIMILARITY is dropped; if every take does, the
        embedding alone is used. Cached by the reference clip's contents and
        the donor.
        """
        donor_name = accent_donor(self.cfg)
        cached = cached_reference_path(self.cfg, reference_wav_path)
        folder = cached.parent
        if cached.is_file():
            return cached
        vlog("tts", "making an English reference for this voice (once; cached)")
        folder.mkdir(parents=True, exist_ok=True)
        with self._lock:
            speaker = self.model.create_voice_clone_prompt(
                ref_audio=reference_wav_path, ref_text=None, x_vector_only_mode=True
            )
        target = self._speaker_vector(speaker)
        takes = []
        if donor_name:
            donor_clip, donor_text = self._accent_donor_clip(folder, donor_name)
            with self._lock:
                donor = self.model.create_voice_clone_prompt(ref_audio=str(donor_clip), ref_text=donor_text)
            takes = self._bridge_takes(self._with_speaker(donor, speaker), folder, target)
            takes = [take for take in takes if take[0] >= MIN_SPEAKER_SIMILARITY]
            if not takes:
                vlog("tts", f"  every take with {donor_name}'s accent drifted from the speaker's voice; "
                            "using the speaker embedding alone")
        if not takes:
            takes = self._bridge_takes(speaker, folder, target)
        _, _, _, audio, sample_rate = max(takes, key=lambda take: take[1:3])
        import soundfile

        soundfile.write(str(cached), match_level(audio, sample_rate, 0.1), sample_rate)
        return cached

    def _bridge_takes(self, prompt, folder: Path, target: np.ndarray) -> list:
        """ENGLISH_BRIDGE_TAKES takes of the passage from `prompt`, each as
        (similarity to the speaker, no stall mid-passage, closeness to the usual
        pace, audio, sample rate)."""
        import soundfile

        takes = []
        for _ in range(ENGLISH_BRIDGE_TAKES):
            with self._lock:
                wavs, sample_rate = self.model.generate_voice_clone(
                    text=ENGLISH_BRIDGE_TEXT, language="English", voice_clone_prompt=prompt,
                    non_streaming_mode=True, **self._sampling(ENGLISH_BRIDGE_TEXT)
                )
            # Silence at the ends of a reference teaches the model to trail on
            # after a sentence (takes that run to the length cap), so it's cut.
            audio = trim_silence(np.asarray(wavs[0], dtype=np.float32), sample_rate)
            scratch = folder / "take.wav"
            soundfile.write(str(scratch), audio, sample_rate)
            with self._lock:
                vector = self._speaker_vector(self.model.create_voice_clone_prompt(
                    ref_audio=str(scratch), ref_text=None, x_vector_only_mode=True
                ))
            scratch.unlink(missing_ok=True)
            similarity = float(vector @ target / (np.linalg.norm(vector) * np.linalg.norm(target) + 1e-9))
            pause = longest_pause_s(audio, sample_rate)
            seconds_per_char = len(audio) / sample_rate / len(ENGLISH_BRIDGE_TEXT)
            takes.append((similarity, pause <= 1.0, -abs(seconds_per_char - BRIDGE_SECONDS_PER_CHAR), audio, sample_rate))
            vlog("tts", f"  English reference take: {len(audio) / sample_rate:.1f}s, longest pause {pause:.1f}s, "
                        f"speaker similarity {similarity:.2f}")
        return takes

    def set_accent_donor(self, name: str) -> None:
        """Which voice new English references take their accent from (ACCENT_DONORS; "": none).
        Takes effect on the next clone (the pipeline re-clones the active voice)."""
        self.cfg.qwen3_accent_donor = name if name in ACCENT_DONORS else ""

    def _accent_donor_clip(self, folder: Path, donor_name: str) -> tuple[Path, str]:
        """The clip (and its text) whose delivery English references imitate:
        the bundled Default voice's own recording, or a preset voice of the
        CustomVoice model saying DONOR_TEXT (made once, cached)."""
        if donor_name == BUNDLED_DONOR:
            import json

            from sts.tts.voice_library import BUNDLED_VOICES_DIR

            folder = BUNDLED_VOICES_DIR / "default"
            return folder / "reference.wav", json.loads((folder / "meta.json").read_text())["ref_text"]
        key = hashlib.sha1((donor_name + DONOR_TEXT + self.cfg.qwen3_custom_model_name).encode()).hexdigest()[:16]
        path = folder / f"donor-{key}.wav"
        if path.is_file():
            return path, DONOR_TEXT
        if self._custom_model is None:
            self._custom_model = self._load_custom_model()
        with self._lock:
            wavs, sample_rate = self._custom_model.generate_custom_voice(
                text=DONOR_TEXT, speaker=donor_name.lower() if self.engine == "torch" else donor_name,
                language="English", **self._sampling(DONOR_TEXT)
            )
        import soundfile

        soundfile.write(str(path), trim_silence(np.asarray(wavs[0], dtype=np.float32), sample_rate), sample_rate)
        return path, DONOR_TEXT

    def _with_speaker(self, prompt, speaker):
        """`prompt` (a clip + its text: how to speak) with `speaker`'s embedding (whose voice)."""
        if self.engine == "mlx":
            return self.model.with_speaker(prompt, speaker)
        from dataclasses import replace

        return [replace(item, ref_spk_embedding=voice.ref_spk_embedding) for item, voice in zip(prompt, speaker)]

    def _speaker_vector(self, prompt) -> np.ndarray:
        """The speaker embedding of a clone prompt, for comparing voices."""
        if self.engine == "mlx":
            return self.model.speaker_vector(prompt)
        return prompt[0].ref_spk_embedding.detach().float().cpu().numpy().reshape(-1)

    def make_english_reference(self, reference_wav_path: str) -> str:
        """Makes (or finds) the clip's English reference without changing the
        active voice - run in a helper process while the live one speaks."""
        return str(self._english_reference(reference_wav_path))

    def clone_from_entry(self, entry: VoiceEntry) -> None:
        if entry.kind == "lora":
            self._activate_lora(entry)
        else:
            self.clone_voice(str(entry.wav_path), entry.ref_text)

    def _activate_lora(self, entry: VoiceEntry) -> None:
        self._pace_history.clear()
        self._clarity_history.clear()
        self._tone_history.clear()
        if entry.speaker_id not in self._lora_models:
            self._lora_models[entry.speaker_id] = _load_lora_model(self.cfg, entry)
        self._active_lora_speaker_id = entry.speaker_id
        self._prompt = None
        self.cloned_speaker_wav = None

    def set_default_voice(self, voice: str) -> None:
        voice = self.cfg.resolve_voice(voice)
        if voice not in self.cfg.all_voices():
            raise ValueError(f"unknown preset voice {voice!r}; choose from {self.cfg.all_voices()}")
        self._pace_history.clear()
        self._clarity_history.clear()
        self._tone_history.clear()
        self.default_voice = voice

    def forget_voice(self, speaker_id: str) -> None:
        """Drops a deleted voice's cached LoRA model (a multi-GB model load)."""
        if self._lora_models.pop(speaker_id, None) is not None:
            gc.collect()

    def reset_voice(self) -> None:
        self._pace_history.clear()
        self._clarity_history.clear()
        self._tone_history.clear()
        self._prompt = None
        self.cloned_speaker_wav = None
        self._active_lora_speaker_id = None

    def _load_custom_model(self):
        return self._model_class.from_pretrained(
            self.cfg.qwen3_custom_model_name,
            device_map=self.cfg.device,
            dtype=_dtype_for_device(self.cfg.device),
            attn_implementation=_attn_impl_for_device(self.cfg.device),
        )

    def synthesize_stream(self, text: str):
        """Yields (audio, sample_rate) for `text` - one translated sentence,
        as shown on the caption page - generated whole, in one go, however
        long: cut into pieces, the next piece often wasn't ready when a short
        one finished playing (a pause mid-sentence), and each piece got its
        own intonation. (A generator for the interface the pipeline uses.)"""
        text = spell_numbers(text.strip())
        if not text:
            return
        audio, sample_rate = self._synthesize_checked(text)
        if self.cfg.qwen3_tone_match:
            audio = self._even_out_tone(audio, sample_rate)
        if self.cfg.target_level:
            audio = match_level(audio, sample_rate, self.cfg.target_level)
        yield audio, sample_rate

    # -- streaming ------------------------------------------------------------

    def stream_chunks(self, text: str):
        """Yields (audio, sample_rate, seamless) for `text` - one translated
        sentence - streamed as it's generated (see STREAM_HOLD_S): the first
        piece not seamless (a pause may go before it), the rest seamless.
        Without streaming (the PyTorch engine, a trained LoRA voice, or
        TTSConfig.qwen3_stream off) the whole sentence, as synthesize_stream."""
        if not self._can_stream():
            for audio, sample_rate in self.synthesize_stream(text):
                yield audio, sample_rate, False
            return
        text = spell_numbers(text.strip())
        if text:
            yield from self._stream_sentence(text)

    def _can_stream(self) -> bool:
        return (self.cfg.qwen3_stream and self.engine == "mlx" and not self._active_lora_speaker_id
                and hasattr(self.model, "stream_voice_clone"))

    def _expected_seconds(self, text: str) -> float:
        pace = statistics.median(self._pace_history) if len(self._pace_history) >= 5 else TYPICAL_SECONDS_PER_CHAR
        return max(0.8, len(text) * pace)

    # A streamed take is also held back until this share of the text's expected
    # length is generated (0: just STREAM_HOLD_S), and a take that ends short of
    # CUT_SHORT_FRACTION of it is regenerated, up to STREAM_ATTEMPTS times in all
    # (None: TTSConfig.qwen3_max_retries + 1). Qwen3 needs neither; Drama 3 does.
    RELEASE_FRACTION = 0.0
    CUT_SHORT_FRACTION = 0.0
    RAMBLE_FACTOR = RAMBLE_FACTOR
    STREAM_ATTEMPTS: int | None = None

    def _stream_sentence(self, text: str):
        expected = self._expected_seconds(text)
        ramble_s = expected * self.RAMBLE_FACTOR + RAMBLE_MARGIN_S
        hold_s = self.cfg.qwen3_stream_hold_s
        release_s = expected * self.RELEASE_FRACTION
        attempts = self.STREAM_ATTEMPTS or self.cfg.qwen3_max_retries + 1
        for attempt in range(attempts):
            last = attempt == attempts - 1
            raw: list[np.ndarray] = []  # the whole take as generated, for the checks and the voice's history
            held: list[np.ndarray] = []
            generated_s = 0.0
            post = None  # _StreamPost once playing
            rambled = stalled = False
            sample_rate = getattr(self.model, "sample_rate", 24_000)
            speech = _SpeechTracker(sample_rate)
            started = time.monotonic()
            with self._lock:
                pieces = self._stream_generate(text, attempt)
                try:
                    for piece in pieces:
                        raw.append(piece)
                        generated_s += len(piece) / sample_rate
                        speech.add(piece)
                        if generated_s > ramble_s:
                            rambled = True
                            break
                        if speech.silent_for >= STALL_S and speech.spoken_s < STALL_UNDER * expected:
                            stalled = True  # stopped talking partway through the text
                            break
                        if post is None:
                            held.append(piece)
                            if time.monotonic() - started >= hold_s and speech.spoken_s >= release_s:
                                post = self._stream_post(np.concatenate(held), sample_rate)
                                out = post.process(np.concatenate(held))
                                held = []
                                vlog("tts", f"streaming: {generated_s:.1f}s generated in the first {hold_s:.1f}s")
                                yield out, sample_rate, False
                        else:
                            out = post.process(piece)
                            if out.size:
                                yield out, sample_rate, True
                finally:
                    pieces.close()
            if stalled:
                said = f"{speech.spoken_s:.1f}s of speech for {len(text)} characters, expected about {expected:.1f}s"
                if post is None and not last:
                    vlog("tts", f"stopped talking partway ({said}); caught before playing, regenerating: {text[:60]!r}")
                    continue
                if post is not None:
                    vlog("tts", f"stopped talking partway ({said}); the rest of the take cut off: {text[:60]!r}")
                    yield post.fade_out(), sample_rate, True
                    return
                audio = trim_silence(np.concatenate(raw), sample_rate)  # every try stalled: what was said
                yield self._finish_whole(audio, sample_rate), sample_rate, False
                return
            if rambled:
                if post is None and not last:
                    vlog("tts", f"rambling ({generated_s:.1f}s for {len(text)} characters, expected about "
                                f"{expected:.1f}s); caught before playing, regenerating: {text[:60]!r}")
                    continue
                if post is not None:
                    vlog("tts", f"rambling ({generated_s:.1f}s for {len(text)} characters, expected about "
                                f"{expected:.1f}s); cut off: {text[:60]!r}")
                    yield post.fade_out(), sample_rate, True
                    return
                # Rambled on every try before playing: what it said in the expected time, faded out.
                audio = np.concatenate(raw)[: int(ramble_s * sample_rate)]
                fade = min(len(audio), int(STREAM_FADE_S * sample_rate))
                audio[len(audio) - fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
                yield self._finish_whole(audio, sample_rate), sample_rate, False
                return
            if post is not None:  # streamed to the end
                tail = post.finish()
                if tail.size:
                    yield tail, sample_rate, True
                self._remember_take(np.concatenate(raw), sample_rate, text)
                self._spoken_whole(text)
                return
            # Whole within the hold: checked like an unstreamed take, regenerated if bad.
            audio = np.concatenate(raw) if raw else np.zeros(0, dtype=np.float32)
            min_spc = min_plausible_seconds_per_char(self._pace_history, self.cfg.qwen3_min_seconds_per_char)
            plausible = speech_length_plausible(text, len(audio) / sample_rate, min_spc, self.cfg.qwen3_max_seconds_per_char)
            if plausible and len(text) >= MIN_CHARS_TO_JUDGE and speech.spoken_s < self.CUT_SHORT_FRACTION * expected:
                plausible = False  # the speech ended well short of the text: the rest wasn't said
            if not plausible and not last:
                vlog("tts", f"suspicious speech length ({len(audio) / sample_rate:.1f}s for {len(text)} characters), "
                            f"caught before playing, regenerating: {text[:60]!r}")
                continue
            self._remember_take(audio, sample_rate, text)
            if plausible:
                self._spoken_whole(text)
            yield self._finish_whole(audio, sample_rate), sample_rate, False
            return

    def _spoken_whole(self, text: str) -> None:
        """`text` was just spoken in full, by the take last generated (not one
        cut off, or let through only because every try was bad). Drama 3 keeps
        it as context for the next sentence; Qwen3 has no use for it."""

    def _stream_generate(self, text: str, attempt: int = 0):
        """The model's pieces of `text` (np.float32), as they're generated
        (`attempt`: which try at this sentence, from 0)."""
        if self._prompt is not None:
            return self.model.stream_voice_clone(
                text, self._prompt, STREAM_PIECE_S, language="English", **self._sampling(text)
            )
        if self._custom_model is None:
            self._custom_model = self._load_custom_model()
        return self._custom_model.stream_custom_voice(
            text, self.default_voice.lower(), STREAM_PIECE_S, language="English",
            instruct=self.cfg.qwen3_style_instruct or None, **self._sampling(text),
        )

    def _stream_post(self, start: np.ndarray, sample_rate: int) -> "_StreamPost":
        """Level and tone for a streamed take, worked out from its start (what
        was generated while held back - usually a few seconds), then applied
        unchanged to the rest, so the pieces join without a seam."""
        gain = level_gain(start, sample_rate, self.cfg.target_level) if self.cfg.target_level else 1.0
        fir = None
        profile = tone_profile(start, sample_rate) if self.cfg.qwen3_tone_match else None
        if profile is not None and len(self._tone_history) >= 3:
            fir = tone_filter(tone_gains(profile, np.mean(self._tone_history, axis=0)), sample_rate)
        return _StreamPost(sample_rate, gain, fir)

    def _finish_whole(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Tone and level for a whole take (as synthesize_stream)."""
        if self.cfg.qwen3_tone_match:
            audio = self._even_out_tone(audio, sample_rate, remember=False)
        if self.cfg.target_level:
            audio = match_level(audio, sample_rate, self.cfg.target_level)
        return audio

    def _remember_take(self, audio: np.ndarray, sample_rate: int, text: str) -> None:
        """A finished take counts toward the voice's pace, clarity and tone history."""
        chars = len(text.strip())
        if chars >= MIN_CHARS_TO_JUDGE and len(audio):
            self._pace_history.append(len(audio) / sample_rate / chars)
            self._clarity_history.append(clarity_db(audio, sample_rate))
        profile = tone_profile(audio, sample_rate)
        if profile is not None:
            self._tone_history.append(profile)

    def _even_out_tone(self, audio: np.ndarray, sample_rate: int, remember: bool = True) -> np.ndarray:
        """Equalizes the take toward this voice's recent average tone (see
        sts.audioquality.match_tone), once there are a few takes to average."""
        profile = tone_profile(audio, sample_rate)
        if profile is None:
            return audio
        if len(self._tone_history) >= 3:
            audio = match_tone(audio, sample_rate, profile, np.mean(self._tone_history, axis=0))
        if remember:
            self._tone_history.append(profile)  # the take as generated, so the average is the voice's own
        return audio

    def synthesize(self, text: str, pause_s: float = 0.35) -> tuple[np.ndarray, int]:
        """Whole text as one clip."""
        pieces = list(self.synthesize_stream(text))
        if not pieces:
            return np.zeros(0, dtype=np.float32), 24_000
        sample_rate = pieces[0][1]
        gap = np.zeros(int(pause_s * sample_rate), dtype=np.float32)
        audio = np.concatenate([part for i, (a, _) in enumerate(pieces) for part in ((gap, a) if i else (a,))])
        return audio, sample_rate

    def _synthesize_checked(self, text: str) -> tuple[np.ndarray, int]:
        """One piece, regenerated if the take looks like skipped or runaway speech,
        or is noticeably duller than this voice normally sounds. Keeps the best take."""
        chars = len(text.strip())
        # (audio, sample_rate, length_was_believable, clarity in dB or None if too short to judge)
        takes: list[tuple[np.ndarray, int, bool, float | None]] = []
        previous_seconds: float | None = None
        previous_clarity: float | None = None
        for attempt in range(self.cfg.qwen3_max_retries + 1):
            audio, sample_rate = self._generate_once(text)
            seconds = len(audio) / sample_rate
            min_spc = min_plausible_seconds_per_char(self._pace_history, self.cfg.qwen3_min_seconds_per_char)
            plausible = speech_length_plausible(text, seconds, min_spc, self.cfg.qwen3_max_seconds_per_char)
            # A skipped middle is a random failure, so it comes out a different length
            # each time. The same length twice running means the model is consistently
            # reading the whole text (just at this pace); retrying again would only delay audio.
            repeated = previous_seconds is not None and abs(seconds - previous_seconds) <= 0.1 * previous_seconds
            if not plausible and not repeated:
                vlog("tts", f"suspicious speech length ({seconds:.1f}s for {chars} characters, "
                            f"expected at least {min_spc * chars:.1f}s), regenerating (attempt {attempt + 1}): {text[:60]!r}")
                previous_seconds = seconds
                takes.append((audio, sample_rate, False, None))
                continue

            clarity = clarity_db(audio, sample_rate) if chars >= MIN_CHARS_TO_JUDGE else None
            takes.append((audio, sample_rate, True, clarity))
            bar = clarity_bar(self._clarity_history, self.cfg.qwen3_clarity_margin_db)
            if clarity is None or bar is None or clarity >= bar or attempt >= self.cfg.qwen3_quality_retries:
                break
            if previous_clarity is not None and abs(clarity - previous_clarity) < 1.5:
                break  # about as dull as the last take: this voice/sentence is just like that, more tries won't help
            previous_clarity = clarity
            vlog("tts", f"dull take ({clarity:.1f} dB; this voice's bar is {bar:.1f} dB), "
                        f"regenerating (attempt {attempt + 1}): {text[:60]!r}")
            previous_seconds = seconds

        believable = [t for t in takes if t[2]]
        if not believable:
            # Every take skipped words (they come out too short), so the longest is the best fallback.
            audio, sample_rate, _, _ = max(takes, key=lambda t: len(t[0]))
            return audio, sample_rate
        audio, sample_rate, _, clarity = max(believable, key=lambda t: t[3] if t[3] is not None else 0.0)
        if clarity is not None:
            self._clarity_history.append(clarity)
            self._pace_history.append(len(audio) / sample_rate / chars)
        return audio, sample_rate

    def _generate_once(self, text: str) -> tuple[np.ndarray, int]:
        with self._lock:
            if self._active_lora_speaker_id:
                lora_model, speaker_name = self._lora_models[self._active_lora_speaker_id]
                wavs, sample_rate = lora_model.generate_custom_voice(
                    text=text, speaker=speaker_name, language="English", **self._sampling(text)
                )
            elif self._prompt is not None:
                wavs, sample_rate = self.model.generate_voice_clone(
                    text=text, language="English", voice_clone_prompt=self._prompt,
                    non_streaming_mode=True, **self._sampling(text)
                )
            else:
                if self._custom_model is None:
                    self._custom_model = self._load_custom_model()
                wavs, sample_rate = self._custom_model.generate_custom_voice(
                    text=text,
                    speaker=self.default_voice.lower(),
                    language="English",
                    instruct=self.cfg.qwen3_style_instruct or None,
                    **self._sampling(text),
                )
        return np.asarray(wavs[0], dtype=np.float32), sample_rate


class _SpeechTracker:
    """Where the speech is in a take that arrives in pieces: how far into it
    the last speech goes (`spoken_s`) and how long it's been silent since
    (`silent_for`). Speech: 20 ms frames above 3% of the loudest so far."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.frame = max(1, sample_rate // 50)
        self.frames = 0
        self.peak = 0.0
        self.last_speech = None  # frames into the take where the last speech ends
        self.pending = np.zeros(0, dtype=np.float32)

    def add(self, piece: np.ndarray) -> None:
        audio = np.concatenate([self.pending, piece])
        count = len(audio) // self.frame
        self.pending = audio[count * self.frame:]
        if not count:
            return
        rms = np.sqrt(np.mean(audio[: count * self.frame].reshape(count, self.frame) ** 2, axis=1))
        self.peak = max(self.peak, float(rms.max()))
        loud = np.nonzero(rms > max(0.003, 0.03 * self.peak))[0]
        if loud.size:
            self.last_speech = self.frames + int(loud[-1]) + 1
        self.frames += count

    @property
    def spoken_s(self) -> float:
        return (self.last_speech or 0) * self.frame / self.sample_rate

    @property
    def silent_for(self) -> float:
        if self.last_speech is None:
            return 0.0  # not started talking yet
        return (self.frames - self.last_speech) * self.frame / self.sample_rate


class _StreamPost:
    """Level and tone for a streamed take's pieces: a fixed gain and an FIR
    equalizer whose state carries from piece to piece, so the pieces join
    seamlessly. The last STREAM_FADE_S is kept back each time, so the take can
    be faded out if it's cut off (rambling)."""

    def __init__(self, sample_rate: int, gain: float, fir: np.ndarray | None):
        self.gain = gain
        self.fir = fir
        self.state = np.zeros(len(fir) - 1) if fir is not None else None
        self.keep = int(STREAM_FADE_S * sample_rate)
        self.tail = np.zeros(0, dtype=np.float32)

    def _filter(self, audio: np.ndarray) -> np.ndarray:
        if self.fir is not None:
            from scipy.signal import lfilter

            audio, self.state = lfilter(self.fir, [1.0], audio, zi=self.state)
        return np.clip(audio * self.gain, -0.99, 0.99).astype(np.float32)

    def process(self, audio: np.ndarray) -> np.ndarray:
        out = np.concatenate([self.tail, self._filter(audio)])
        self.tail = out[-self.keep:] if len(out) > self.keep else out
        return out[: len(out) - len(self.tail)]

    def finish(self) -> np.ndarray:
        """The rest: the kept-back tail, and what the filter still holds."""
        rest = self._filter(np.zeros(len(self.fir) - 1)) if self.fir is not None else np.zeros(0, np.float32)
        out, self.tail = np.concatenate([self.tail, rest]), np.zeros(0, dtype=np.float32)
        return out

    def fade_out(self) -> np.ndarray:
        """Cut off: the kept-back tail, faded to silence."""
        out, self.tail = self.tail.copy(), np.zeros(0, dtype=np.float32)
        out *= np.linspace(1, 0, len(out), dtype=np.float32)
        return out


def _load_lora_model(cfg: TTSConfig, entry: VoiceEntry) -> tuple[Qwen3TTSModel, str]:
    if Qwen3TTSModel is None or resolve_engine(cfg.qwen3_engine) == "mlx":
        raise RuntimeError(f"voice {entry.name!r} is a trained (LoRA) voice, which needs the PyTorch "
                           "engine (.venv-qwen); the MLX engine clones voices zero-shot only")
    from peft import PeftModel

    base = Qwen3TTSModel.from_pretrained(
        cfg.qwen3_model_name,
        device_map=cfg.device,
        dtype=_dtype_for_device(cfg.device),
        attn_implementation=_attn_impl_for_device(cfg.device),
    )
    peft_model = PeftModel.from_pretrained(base.model, str(entry.adapter_dir))
    _set_lora_scale(peft_model, entry.lora_scale)
    base.model = peft_model.merge_and_unload()

    core_model = _resolve_core_model(base.model)
    _apply_speaker_config(core_model, str(entry.adapter_dir), entry.speaker_name)
    _apply_speaker_embedding(core_model, str(entry.adapter_dir), entry.speaker_name)
    core_model.eval()

    return base, entry.speaker_name


def _resolve_core_model(model):
    if hasattr(model, "talker"):
        return model
    if hasattr(model, "model") and hasattr(model.model, "talker"):
        return model.model
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "talker"):
            return base
        if hasattr(base, "model") and hasattr(base.model, "talker"):
            return base.model
    return model


def _apply_speaker_config(core_model, adapter_dir: str, speaker_name: str, speaker_id: int = 3000) -> None:
    import json
    import os

    config_path = os.path.join(adapter_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)
        talker_config = config_dict.get("talker_config", {})
        spk_id_map = talker_config.get("spk_id", {})
        spk_is_dialect = talker_config.get("spk_is_dialect", {})
        if isinstance(spk_id_map, dict):
            core_model.config.talker_config.spk_id.update(spk_id_map)
        if isinstance(spk_is_dialect, dict):
            core_model.config.talker_config.spk_is_dialect.update(spk_is_dialect)

    if speaker_name not in core_model.config.talker_config.spk_id:
        core_model.config.talker_config.spk_id[speaker_name] = speaker_id
        core_model.config.talker_config.spk_is_dialect[speaker_name] = False

    core_model.config.tts_model_type = "custom_voice"
    core_model.tts_model_type = "custom_voice"
    core_model.supported_speakers = core_model.config.talker_config.spk_id.keys()


def _apply_speaker_embedding(core_model, adapter_dir: str, speaker_name: str) -> None:
    import os

    from safetensors.torch import load_file

    embedding_path = os.path.join(adapter_dir, "speaker_embedding.safetensors")
    payload = load_file(embedding_path)
    spk_id = core_model.config.talker_config.spk_id[speaker_name]
    weight = core_model.talker.model.codec_embedding.weight
    weight.data[spk_id] = payload["target_speaker_embedding"].to(device=weight.device, dtype=weight.dtype)


def _set_lora_scale(peft_model, scale: float) -> None:
    if scale == 1.0:
        return
    from peft.tuners.lora import LoraLayer

    adapter = getattr(peft_model, "active_adapter", None)
    for module in peft_model.modules():
        if isinstance(module, LoraLayer):
            if adapter is None:
                module.scale_layer(scale)
            elif isinstance(adapter, (list, tuple)):
                for name in adapter:
                    module.set_scale(name, scale)
            else:
                module.set_scale(adapter, scale)
