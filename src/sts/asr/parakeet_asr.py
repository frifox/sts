"""Transcribes Russian speech segments with NVIDIA's Parakeet TDT, via NeMo.

Unlike Whisper, Parakeet emits its own punctuation and capitalization, so
there's no initial_prompt to nudge vocabulary and no per-segment confidence
score to filter hallucinations on - `nemo`'s plain `transcribe()` doesn't
expose one without switching to its `return_hypotheses=True` mode. Not
implemented here since it hasn't shown the hallucination-on-silence problem
Whisper has in testing; revisit if that changes.
"""

import numpy as np

from sts.config import ASRConfig
from sts.textclean import clean_transcript

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


class ParakeetASR:
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        import nemo.collections.asr as nemo_asr  # heavy (NeMo + lightning); only paid for when selected

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        if cfg.device == "cuda":
            # Not the fp32 the model loads in by default: Turing and newer
            # have native fp16 tensor cores, measured ~13% less VRAM and
            # ~40% faster than fp32 on an RTX 2070 with no quality loss.
            self.model = self.model.half()
        else:
            self.model = self.model.to(cfg.device).float()
        self.model.eval()

    def transcribe(self, audio: np.ndarray | str) -> str:
        # Accepts a raw float32 array (live-captured segments) or a file path
        # (e.g. a pre-supplied --clone reference clip) directly, same as
        # WhisperASR.transcribe.
        out = self.model.transcribe(audio, verbose=False)
        text = out[0].text if hasattr(out[0], "text") else out[0]
        return clean_transcript(text.strip())
