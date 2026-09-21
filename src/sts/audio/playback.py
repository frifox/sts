"""Plays synthesized speech out to a sound card output."""

import numpy as np
import sounddevice as sd

from sts.config import AudioConfig
from sts.log import vlog


class AudioPlayback:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg

    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
        info = sd.query_devices(self.cfg.output_device, "output")
        vlog("audio", f"playing {len(audio) / sample_rate:.1f}s of speech on output [{info['index']}] {info['name']}")
        sd.play(audio, samplerate=sample_rate, device=self.cfg.output_device)
        if blocking:
            sd.wait()
            vlog("audio", "playback finished")
