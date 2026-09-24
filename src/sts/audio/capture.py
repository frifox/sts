"""Reads raw audio off a sound card input and pushes fixed-size frames onto a queue."""

import queue
import threading
import time

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

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
        self._last_frame_at = time.monotonic()
        self._capture_rate = cfg.sample_rate  # set for real in start(); see _resolve_capture_rate
        self._resample_buffer = np.empty(0, dtype=np.float32)

    def is_stalled(self, timeout_s: float) -> bool:
        """True if a local input stream is supposed to be running but has
        stopped delivering frames for `timeout_s` - what happens to it when
        the machine sleeps and wakes, or its device goes away. A live stream
        delivers a frame every block_ms whether or not anyone is speaking, so
        silence in the room never looks like a stall. Always False for the
        browser microphone (frames come over the websocket) or when no
        stream is open."""
        if self._stream is None or self.cfg.source == "browser":
            return False
        try:
            if not self._stream.active:
                return True
        except Exception:
            return True  # asking a broken stream whether it's running can itself raise
        return time.monotonic() - self._last_frame_at > timeout_s

    def _callback(self, indata, frames, time_info, status):
        self._last_frame_at = time.monotonic()
        if status:
            # Overflows/underflows show up here; surfaced rather than raised
            # since dropping a frame is preferable to killing capture.
            print(f"[audio] input status: {status}")
        mono = indata[:, 0]
        if self._capture_rate == self.cfg.sample_rate:
            self.frames.put(mono.copy())
            return
        # Captured at the device's own native rate (see _resolve_capture_rate)
        # - resample down to what the VAD/ASR need, buffering across callbacks
        # since the ratio rarely divides the callback's block size evenly, and
        # slice off exact block_ms frames (Silero needs exactly 512 samples at
        # 16kHz, not "close enough").
        resampled = resample_poly(mono, self.cfg.sample_rate, self._capture_rate).astype(np.float32)
        self._resample_buffer = np.concatenate([self._resample_buffer, resampled])
        block_size = int(self.cfg.sample_rate * self.cfg.block_ms / 1000)
        while len(self._resample_buffer) >= block_size:
            self.frames.put(self._resample_buffer[:block_size].copy())
            self._resample_buffer = self._resample_buffer[block_size:]

    def _resolve_capture_rate(self, device: int | None) -> int:
        """The rate to actually open the input stream at: cfg.sample_rate if
        the device accepts it directly, else the device's own default rate.
        PortAudio's raw ALSA "hw" access (no "plug" layer) means many real
        sound cards - an onboard Line-In codec, say - only accept a couple of
        fixed hardware rates like 44100/48000 and flatly reject the
        pipeline's 16kHz; captured audio is resampled back down to
        cfg.sample_rate in _callback, same idea as AudioPlayback._target_rate
        for output."""
        try:
            sd.check_input_settings(
                device=device, channels=self.cfg.channels, samplerate=self.cfg.sample_rate, dtype="float32"
            )
            return self.cfg.sample_rate
        except Exception:
            info = sd.query_devices(device, "input")
            return int(info["default_samplerate"])

    def start(self) -> None:
        """Opens the configured source. A no-op for `source == "browser"`:
        frames arrive via `push_browser_frame` instead of a local stream."""
        if self.cfg.source == "browser":
            vlog("audio", "waiting for browser microphone frames")
            return
        self._capture_rate = self._resolve_capture_rate(self.cfg.input_device)
        self._resample_buffer = np.empty(0, dtype=np.float32)
        block_size = int(self._capture_rate * self.cfg.block_ms / 1000)
        self._stream = sd.InputStream(
            device=self.cfg.input_device,
            samplerate=self._capture_rate,
            channels=self.cfg.channels,
            blocksize=block_size,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._last_frame_at = time.monotonic()
        info = sd.query_devices(self.cfg.input_device, "input")
        note = f" (resampled from {self._capture_rate} Hz)" if self._capture_rate != self.cfg.sample_rate else ""
        vlog("audio", f"listening on input [{info['index']}] {info['name']} "
                      f"({self.cfg.sample_rate} Hz, {self.cfg.block_ms} ms frames){note}")

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
            stream, self._stream = self._stream, None
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                # A stream whose device vanished (or that the machine slept
                # under) can raise while being torn down - it's being thrown
                # away regardless, and this is also how it's cleaned up before
                # the audio system is re-initialized.
                vlog("audio", f"input stream didn't close cleanly ({exc!r}); discarded anyway")
            vlog("audio", "stopped listening")
