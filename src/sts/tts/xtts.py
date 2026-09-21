"""Synthesizes English speech with Coqui XTTS v2.

XTTS supports both a set of built-in speakers and zero-shot voice cloning
from a short reference clip, which covers the stretch goal (clone the live
speaker's voice from ~10-20s of their own audio) with no separate model.

`cfg.device` is passed straight to `.to(...)`, so "cuda", "cpu", and "mps"
(Apple Silicon) all work as far as PyTorch is concerned - unlike Qwen3-TTS,
there's no flash-attention/bfloat16 choice baked into this backend to worry
about per device.
"""

import re

import numpy as np
import soundfile as sf
import torch
import torchaudio
from TTS.api import TTS
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig

from sts.config import TTSConfig
from sts.tts.voice_library import VoiceEntry

SAMPLE_RATE = 24_000  # XTTS v2 output rate

# Coqui TTS 0.22 predates PyTorch 2.6's weights_only=True default for
# torch.load, so its own checkpoint (which pickles these config classes)
# fails to load unless they're allowlisted.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])


# XTTS asserts on more than 400 tokens per sentence and warns past 250 characters
# for English, so long unpunctuated sentences are cut at clause boundaries first.
MAX_CHARS_PER_CHUNK = 220


def split_for_xtts(text: str, limit: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    # Prefer sentence ends, then commas/semicolons, then plain spaces.
    for part in re.split(r"(?<=[.!?])\s+|(?<=[,;:])\s+", text):
        for word_run in _hard_wrap(part, limit):
            if current and len(current) + 1 + len(word_run) > limit:
                chunks.append(current)
                current = word_run
            else:
                current = f"{current} {word_run}".strip()
    if current:
        chunks.append(current)
    return chunks


def _hard_wrap(text: str, limit: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > limit:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _load_with_soundfile(path, *args, **kwargs):
    """Drop-in for `torchaudio.load` returning (channels x frames float32 tensor, sample rate).

    torchaudio 2.9+ dropped its own decoders and now requires torchcodec (and
    ffmpeg) for `load`. Coqui XTTS calls `torchaudio.load` to read the
    reference clip for cloning, and our reference clips are plain wav files
    that soundfile (already a dependency) reads fine.
    """
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sample_rate


torchaudio.load = _load_with_soundfile


class XTTSSynthesizer:
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.tts = TTS(cfg.model_name).to(cfg.device)
        self.cloned_speaker_wav: str | None = cfg.speaker_wav
        self.default_voice = cfg.default_voice

    def set_default_voice(self, voice: str) -> None:
        """Picks which built-in voice ("male"/"female") is used when no cloned voice is active."""
        if voice not in self.cfg.builtin_voices:
            raise ValueError(f"unknown built-in voice {voice!r}; choose from {list(self.cfg.builtin_voices)}")
        self.default_voice = voice

    def clone_voice(self, reference_wav_path: str, ref_text: str | None = None) -> None:
        """Registers a reference clip (10-20s is enough) to clone for
        subsequent synthesis. `ref_text` is accepted for interface parity
        with `Qwen3TTSSynthesizer` but unused - XTTS's cloning doesn't need
        a transcript of the reference clip."""
        self.cloned_speaker_wav = reference_wav_path

    def clone_from_entry(self, entry: VoiceEntry) -> None:
        self.clone_voice(str(entry.wav_path), entry.ref_text)

    def reset_voice(self) -> None:
        self.cloned_speaker_wav = None

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        pieces = [self._synthesize_piece(chunk) for chunk in split_for_xtts(text)]
        if not pieces:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE
        return np.concatenate(pieces), SAMPLE_RATE

    def _synthesize_piece(self, text: str) -> np.ndarray:
        if self.cloned_speaker_wav:
            wav = self.tts.tts(
                text=text, speaker_wav=self.cloned_speaker_wav, language=self.cfg.language
            )
        else:
            wav = self.tts.tts(
                text=text, speaker=self.cfg.builtin_voices[self.default_voice], language=self.cfg.language
            )
        return np.asarray(wav, dtype=np.float32)
