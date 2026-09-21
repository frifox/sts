"""Plays synthesized speech out to a sound card output.

Uses one persistent output stream that clips are written into back to back,
so consecutive sentences play without the gap that starting a fresh
`sd.play()` for each one introduces.
"""

import threading

import numpy as np
import sounddevice as sd

from sts.config import AudioConfig
from sts.log import vlog


class AudioPlayback:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._stream: sd.OutputStream | None = None
        self._stream_rate: int | None = None
        self._lock = threading.Lock()

    def _ensure_stream(self, sample_rate: int) -> sd.OutputStream:
        if self._stream is None or self._stream_rate != sample_rate:
            self._close_stream()
            self._stream = sd.OutputStream(
                samplerate=sample_rate, channels=1, dtype="float32", device=self.cfg.output_device
            )
            self._stream.start()
            self._stream_rate = sample_rate
            info = sd.query_devices(self.cfg.output_device, "output")
            vlog("audio", f"output stream open on [{info['index']}] {info['name']} at {sample_rate} Hz")
        return self._stream

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None
            self._stream_rate = None

    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
        """Queues `audio` on the output stream. The write blocks only until the
        stream's buffer has room, so back-to-back calls play continuously."""
        vlog("audio", f"playing {len(audio) / sample_rate:.1f}s of speech")
        samples = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1, 1)
        with self._lock:
            stream = self._ensure_stream(sample_rate)
        stream.write(samples)

    def stop(self) -> None:
        """Cuts off whatever is currently playing (used when the operator resets)."""
        with self._lock:
            self._close_stream()
