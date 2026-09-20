"""Transcribes Russian speech segments with faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel

from sts.config import ASRConfig


class WhisperASR:
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self.model = WhisperModel(
            cfg.model_size, device=cfg.device, compute_type=cfg.compute_type
        )

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.source_language,
            initial_prompt=self.cfg.initial_prompt,
            vad_filter=False,  # segmentation is already handled upstream
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
