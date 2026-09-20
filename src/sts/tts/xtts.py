"""Synthesizes English speech with Coqui XTTS v2.

XTTS supports both a set of built-in speakers and zero-shot voice cloning
from a short reference clip, which covers the stretch goal (clone the live
speaker's voice from ~10-20s of their own audio) with no separate model.
"""

import numpy as np
from TTS.api import TTS

from sts.config import TTSConfig
from sts.tts.voice_library import VoiceEntry

SAMPLE_RATE = 24_000  # XTTS v2 output rate


class XTTSSynthesizer:
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.tts = TTS(cfg.model_name).to(cfg.device)
        self.cloned_speaker_wav: str | None = cfg.speaker_wav

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
        if not text.strip():
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE

        if self.cloned_speaker_wav:
            wav = self.tts.tts(
                text=text, speaker_wav=self.cloned_speaker_wav, language=self.cfg.language
            )
        else:
            wav = self.tts.tts(
                text=text, speaker=self.cfg.default_speaker, language=self.cfg.language
            )
        return np.asarray(wav, dtype=np.float32), SAMPLE_RATE
