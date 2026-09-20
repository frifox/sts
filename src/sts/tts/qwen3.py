"""Synthesizes English speech with Qwen3-TTS (`Qwen/Qwen3-TTS-12Hz-*-Base`),
cloning a speaker's voice from a saved reference clip.

API per github.com/QwenLM/Qwen3-TTS (Apache-2.0, `pip install qwen-tts`):
`model.create_voice_clone_prompt(ref_audio=..., ref_text=..., x_vector_only_mode=...)`
builds a reusable prompt object once; `model.generate_voice_clone(text=...,
language=..., voice_clone_prompt=...)` returns `(wavs, sample_rate)`.
`ref_text` is the reference clip's own transcript (in its own language -
here, Russian) and materially improves clone quality; `x_vector_only_mode`
drops that requirement at some quality cost, and is the fallback when no
transcript is available.

Unlike XTTS, the Base model has no built-in default speaker - it always
needs a reference clip, so `synthesize()` returns silence with a warning
until `clone_voice()`/`clone_from_entry()` has been called at least once
(typically by picking or training a speaker in `sts.tts.voice_library`).

This backend is unverified against real hardware in this repo (no GPU
available in dev) - the call shapes come directly from the upstream
README's example code, not from having actually run it.
"""

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from sts.config import TTSConfig
from sts.tts.voice_library import VoiceEntry


class Qwen3TTSSynthesizer:
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.model = Qwen3TTSModel.from_pretrained(
            cfg.qwen3_model_name,
            device_map=cfg.device,
            dtype=torch.bfloat16,
        )
        self._prompt = None
        self.cloned_speaker_wav: str | None = None  # set whenever a clone is active; mirrors XTTSSynthesizer

    def clone_voice(self, reference_wav_path: str, ref_text: str | None = None) -> None:
        self._prompt = self.model.create_voice_clone_prompt(
            ref_audio=reference_wav_path,
            ref_text=ref_text,
            x_vector_only_mode=ref_text is None,
        )
        self.cloned_speaker_wav = reference_wav_path

    def clone_from_entry(self, entry: VoiceEntry) -> None:
        self.clone_voice(str(entry.wav_path), entry.ref_text)

    def reset_voice(self) -> None:
        self._prompt = None
        self.cloned_speaker_wav = None

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            return np.zeros(0, dtype=np.float32), 24_000

        if self._prompt is None:
            print("[qwen3-tts] no voice cloned yet - pick or train a speaker first; skipping synthesis")
            return np.zeros(0, dtype=np.float32), 24_000

        wavs, sample_rate = self.model.generate_voice_clone(
            text=text, language="English", voice_clone_prompt=self._prompt
        )
        return np.asarray(wavs[0], dtype=np.float32), sample_rate
