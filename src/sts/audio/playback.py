"""Plays synthesized speech out to a sound card output.

Uses one persistent output stream that clips are written into back to back,
so consecutive sentences play without the gap that starting a fresh
`sd.play()` for each one introduces.
"""

import threading
import time

import numpy as np
import sounddevice as sd

from sts.config import AudioConfig
from sts.log import vlog


def pause_needed(now: float, scheduled_end: float, gap: float) -> float:
    """Silence to insert so a clip starts at least `gap` seconds after the
    previous one ends. `scheduled_end` is when already-queued audio finishes
    (in the past once playback has caught up); 0 means nothing has played yet."""
    if scheduled_end <= 0:
        return 0.0
    idle = now - scheduled_end  # negative while earlier audio is still playing
    return max(0.0, gap - max(idle, 0.0))


class AudioPlayback:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._stream: sd.OutputStream | None = None
        self._stream_rate: int | None = None
        self._lock = threading.Lock()
        self._scheduled_end = 0.0  # monotonic time at which queued audio finishes playing

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

    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True, pause_before: float = 0.0) -> None:
        """Queues `audio` on the output stream. The write blocks only until the
        stream's buffer has room, so back-to-back calls play continuously.

        `pause_before` is the minimum silence to leave after whatever played
        before (0 for continuation chunks of the same text)."""
        samples = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1, 1)
        with self._lock:
            stream = self._ensure_stream(sample_rate)
            now = time.monotonic()
            pad = pause_needed(now, self._scheduled_end, pause_before) if pause_before > 0 else 0.0
            self._scheduled_end = max(now, self._scheduled_end) + pad + len(samples) / sample_rate
            if pad > 0:
                samples = np.concatenate([np.zeros((int(pad * sample_rate), 1), dtype=np.float32), samples])
        vlog("audio", f"playing {len(audio) / sample_rate:.1f}s of speech" + (f" after a {pad:.1f}s pause" if pad > 0 else ""))
        stream.write(samples)

    def stop(self) -> None:
        """Cuts off whatever is currently playing (used when the operator resets)."""
        with self._lock:
            self._close_stream()
            self._scheduled_end = 0.0
