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
        """Opens the configured source. A no-op for `source == "browser"`:
        frames arrive via `push_browser_frame` instead of a local stream."""
        if self.cfg.source == "browser":
            vlog("audio", "waiting for browser microphone frames")
            return
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

    def switch_to_alsa(self, device: int | None = None) -> None:
        """Moves capture to a local ALSA input device, switching away from the
        browser-mic source if that was active. `device=None` keeps whatever
        device was already configured - used when only the source itself
        needs to change back, not the device index. If the device won't
        open, capture goes back to the previous source/device and the error
        is raised."""
        previous_device = self.cfg.input_device
        previous_source = self.cfg.source
        self.stop()
        self.cfg.source = "alsa"
        if device is not None:
            self.cfg.input_device = device
        try:
            self.start()
        except Exception:
            self.cfg.input_device = previous_device
            self.cfg.source = previous_source
            self.start()
            raise
        finally:
            with self.frames.mutex:
                self.frames.queue.clear()  # audio queued from the old source isn't part of the new one

    def switch_to_browser(self) -> None:
        """Moves capture to accepting frames pushed from a connected
        browser's microphone (see `push_browser_frame`) instead of a local
        ALSA device."""
        self.stop()
        self.cfg.source = "browser"
        with self.frames.mutex:
            self.frames.queue.clear()

    def push_browser_frame(self, frame: np.ndarray) -> None:
        """Feeds one block_ms frame captured by a connected browser's
        microphone. Ignored unless "browser" is the currently selected
        source (e.g. a frame that was already in flight when the operator
        switched back to ALSA)."""
        if self.cfg.source == "browser":
            self.frames.put(frame)

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            vlog("audio", "stopped listening")
