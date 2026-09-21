"""Transcribes Russian speech segments with faster-whisper."""

import re

import numpy as np
from faster_whisper import WhisperModel

from sts.config import ASRConfig
from sts.log import vlog
from sts.textclean import clean_transcript

# Whisper was trained on lots of subtitled video, so on silence/music/noise it
# sometimes emits subtitle credits nobody spoke. These are stock phrases from
# that hallucination family; a pastor has no reason to say any of them.
_HALLUCINATION_PATTERNS = re.compile(
    r"субтитры\b|субтитров|"
    r"продолжение следует|подписывайтесь на канал|"
    r"thanks for watching|subtitles by|amara\.org",
    re.IGNORECASE,
)


def _words(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def is_hallucination(text: str, no_speech_prob: float, avg_logprob: float, cfg: ASRConfig) -> bool:
    if _HALLUCINATION_PATTERNS.search(text):
        return True
    # On noise Whisper sometimes just echoes the vocabulary hint it was given.
    words = _words(text).split()
    if len(words) >= 3 and cfg.initial_prompt and set(words) <= set(_words(cfg.initial_prompt).split()):
        return True
    # Whisper's own "this probably isn't speech" signal, only trusted when the
    # decoder was also unsure of what it wrote.
    return no_speech_prob > cfg.no_speech_threshold and avg_logprob < cfg.min_avg_logprob


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
            # Stops one hallucinated line from snowballing into the next segment's decoding.
            condition_on_previous_text=False,
        )
        kept = []
        for segment in segments:
            text = segment.text.strip()
            if is_hallucination(text, segment.no_speech_prob, segment.avg_logprob, self.cfg):
                vlog("asr", f"dropped likely hallucination: {text!r} "
                            f"(no_speech={segment.no_speech_prob:.2f}, logprob={segment.avg_logprob:.2f})")
                continue
            kept.append(text)
        return clean_transcript(" ".join(kept))
