"""Plays synthesized speech out to a sound card output."""

import numpy as np
import sounddevice as sd

from sts.config import AudioConfig


class AudioPlayback:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg

    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
        sd.play(audio, samplerate=sample_rate, device=self.cfg.output_device)
        if blocking:
            sd.wait()
