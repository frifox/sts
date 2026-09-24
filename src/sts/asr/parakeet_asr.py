"""Transcribes Russian speech segments with NVIDIA's Parakeet TDT, via NeMo.

Parakeet emits its own punctuation and capitalization. There's no prompt to
nudge its vocabulary, and no per-segment confidence score to filter
hallucinated text on - `nemo`'s plain `transcribe()` doesn't expose one
without switching to its `return_hypotheses=True` mode. Not implemented here
since it hasn't shown a hallucination-on-silence problem in testing; revisit
if that changes.
"""

import numpy as np
import torch

from sts.asr import TimedWord
from sts.config import ASRConfig
from sts.textclean import clean_transcript, restore_unk_quotes

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


class ParakeetASR:
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        import nemo.collections.asr as nemo_asr  # heavy (NeMo + lightning); only paid for when selected

        # Loaded onto the CPU first, then converted, then moved: from_pretrained
        # would otherwise put the fp32 checkpoint (~2.4GB) straight onto the
        # GPU before .half() halves it - a peak that doesn't fit on an 8GB
        # card next to translation and TTS, and exactly how a live switch to
        # Parakeet ran out of memory in real use.
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=MODEL_NAME, map_location=torch.device("cpu")
        )
        if cfg.device.startswith("cuda"):
            # Not the fp32 the model loads in by default: Turing and newer
            # have native fp16 tensor cores, measured ~13% less VRAM and
            # ~40% faster than fp32 on an RTX 2070 with no quality loss.
            self.model = self.model.half().to(cfg.device)
        else:
            self.model = self.model.to(cfg.device).float()
        self.model.eval()

    def close(self) -> None:
        """Releases the model's GPU memory now, even if something (a
        traceback, a stray reference) still holds on to this object."""
        self.model.to("cpu")

    def transcribe(self, audio: np.ndarray | str) -> str:
        # Accepts a raw float32 array (live-captured segments) or a file path
        # (e.g. a pre-supplied --clone reference clip) directly.
        out = self.model.transcribe(audio, verbose=False)
        text = out[0].text if hasattr(out[0], "text") else out[0]
        # Its 8192-token vocabulary has no token for any style of quotation
        # mark, so it emits its "unknown token" placeholder for one instead;
        # restored to a plain '"' before disfluency cleanup sees it.
        return clean_transcript(restore_unk_quotes(text.strip()))

    def transcribe_words(self, audio: np.ndarray) -> list[TimedWord]:
        """Each word with its timing (see TimedWord). Not disfluency-cleaned:
        the caller decides what a sentence is before cleaning its text."""
        out = self.model.transcribe([audio], timestamps=True, verbose=False)
        return [
            TimedWord(restore_unk_quotes(w["word"]), float(w["start"]), float(w["end"]))
            for w in out[0].timestamp["word"]
        ]
