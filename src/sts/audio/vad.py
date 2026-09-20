"""Turns a stream of raw audio frames into discrete speech segments.

Uses Silero VAD frame-by-frame, buffering speech and cutting a segment once
trailing silence exceeds `min_silence_ms` (or the segment runs too long).
This is what lets the ASR stage work on whole utterances instead of
arbitrary fixed-size chunks.
"""

from collections.abc import Iterator

import numpy as np
import torch

from sts.config import AudioConfig, VADConfig


class SpeechSegmenter:
    def __init__(self, audio_cfg: AudioConfig, vad_cfg: VADConfig):
        self.audio_cfg = audio_cfg
        self.vad_cfg = vad_cfg
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        self.model.eval()

        self._buffer: list[np.ndarray] = []
        self._silence_ms = 0.0
        self._speech_ms = 0.0

    def _is_speech(self, frame: np.ndarray) -> bool:
        tensor = torch.from_numpy(frame)
        with torch.no_grad():
            prob = self.model(tensor, self.audio_cfg.sample_rate).item()
        return prob >= self.vad_cfg.threshold

    def _flush(self) -> np.ndarray | None:
        if not self._buffer or self._speech_ms < self.vad_cfg.min_speech_ms:
            self._buffer = []
            self._speech_ms = 0.0
            self._silence_ms = 0.0
            return None
        segment = np.concatenate(self._buffer)
        self._buffer = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        return segment

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed one `block_ms` frame in; returns a finished segment, if any."""
        speech = self._is_speech(frame)
        frame_ms = self.audio_cfg.block_ms

        if speech:
            self._buffer.append(frame)
            self._speech_ms += frame_ms
            self._silence_ms = 0.0
            if self._speech_ms >= self.vad_cfg.max_segment_ms:
                return self._flush()
            return None

        if self._buffer:
            self._buffer.append(frame)  # keep a little trailing silence for natural prosody
            self._silence_ms += frame_ms
            if self._silence_ms >= self.vad_cfg.min_silence_ms:
                return self._flush()
        return None

    def stream(self, frames: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        for frame in frames:
            segment = self.process(frame)
            if segment is not None:
                yield segment
