"""Transcribes Russian speech segments with faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel

from sts.config import ASRConfig


class WhisperASR:
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        device = cfg.device
        if device == "mps":
            # CTranslate2 (faster-whisper's backend) has no Metal/MPS support,
            # only CPU and CUDA - fail loud with a clear reason rather than
            # letting ctranslate2 raise its own cryptic error deep inside.
            # sts.cli already routes ASR to cpu automatically for --device
            # mps; this only fires if ASRConfig.device was set to mps directly.
            print("[asr] faster-whisper has no Metal/MPS backend - falling back to cpu")
            device = "cpu"
        self.model = WhisperModel(cfg.model_size, device=device, compute_type=cfg.compute_type)

    def transcribe(self, audio: np.ndarray | str) -> str:
        # faster-whisper accepts either a raw array (live-captured segments)
        # or a file path (e.g. a pre-supplied --clone reference clip) directly.
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.source_language,
            initial_prompt=self.cfg.initial_prompt,
            vad_filter=False,  # segmentation is already handled upstream
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
