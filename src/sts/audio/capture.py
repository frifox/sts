"""Reads raw audio off a sound card input and pushes fixed-size frames onto a queue."""

import queue
import threading

import numpy as np
import sounddevice as sd

from sts.config import AudioConfig
from sts.log import vlog


def list_devices() -> str:
    return str(sd.query_devices())


class AudioCapture:
    """Streams mono float32 frames from an input device.

    Frames are `block_ms` long, matching what the VAD expects, and are pushed
    onto `frames` as they arrive so capture never blocks on downstream work.
    """

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.frames: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._stop = threading.Event()

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Overflows/underflows show up here; surfaced rather than raised
            # since dropping a frame is preferable to killing capture.
            print(f"[audio] input status: {status}")
        self.frames.put(indata[:, 0].copy())

    def start(self) -> None:
        block_size = int(self.cfg.sample_rate * self.cfg.block_ms / 1000)
        self._stream = sd.InputStream(
            device=self.cfg.input_device,
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            blocksize=block_size,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        info = sd.query_devices(self.cfg.input_device, "input")
        vlog("audio", f"listening on input [{info['index']}] {info['name']} "
                      f"({self.cfg.sample_rate} Hz, {self.cfg.block_ms} ms frames)")

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            vlog("audio", "stopped listening")
