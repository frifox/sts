"""Reads raw audio off a sound card input and pushes fixed-size frames onto a queue."""

import queue
import threading
import time

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from sts.audio.permission import DENIED_MESSAGE, is_blocked, mic_permission, open_mic_settings
from sts.config import AudioConfig
from sts.log import vlog


def list_devices() -> str:
    return str(sd.query_devices())


# switch_to_alsa's default: distinct from `None`, which is a real, valid
# input_device value (PortAudio/ALSA's own "system default" device) - using
# it as the "no device given" sentinel too meant explicitly picking "System
# default" from the UI (which sends None for exactly that) silently turned
# into "keep whatever device was already open" instead, since None and None
# were indistinguishable.
_KEEP_CURRENT_DEVICE = object()


# Input buffer depth - see AudioCapture._open_stream.
CAPTURE_BUFFER_S = 0.5


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
        # Resampling (when the device doesn't support cfg.sample_rate
        # directly) runs on its own thread, never inside _callback - see
        # _resample_worker for why.
        self._raw_frames: queue.Queue[np.ndarray] = queue.Queue()
        self._resample_thread: threading.Thread | None = None
        self._resample_stop = threading.Event()
        self._overflow_count = 0
        self._last_overflow_log = 0.0
        # macOS microphone permission as of the last local-device start() (see sts.audio.permission).
        self.mic_permission: str | None = None
        self._opened_mic_settings = False

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
        # Kept deliberately trivial: this runs on PortAudio's real-time ALSA
        # thread, which has to be serviced every block_ms or the driver's own
        # buffer overflows (dropped audio, logged as "input status: input
        # overflow" - not a warning to ignore, it means the pipeline missed
        # that audio entirely). resample_poly's filter design/convolution is
        # too slow to risk here - it's what caused that overflow in practice
        # when it ran inline - so a device needing resampling just hands the
        # raw frame to a separate thread (_resample_worker) instead.
        now = time.monotonic()
        self._last_frame_at = now
        if status:
            # print() is itself a blocking syscall (a pipe to journald under
            # systemd) - calling it here on *every* overflow was observed to
            # cascade into hundreds more within the same second, each print
            # delaying the callback's return past the next deadline. Throttled
            # to once/sec so a burst gets reported without becoming the cause
            # of a bigger one.
            self._overflow_count += 1
            if now - self._last_overflow_log >= 1.0:
                print(f"[audio] input status: {status} ({self._overflow_count}x in the last "
                      f"{now - self._last_overflow_log:.1f}s)")
                self._overflow_count = 0
                self._last_overflow_log = now
        mono = indata[:, 0].copy()
        if self._capture_rate == self.cfg.sample_rate:
            self.frames.put(mono)
        else:
            self._raw_frames.put(mono)

    def _resample_worker(self, stop_event: threading.Event) -> None:
        """Off PortAudio's real-time thread (see _callback): resamples from
        the device's native rate down to cfg.sample_rate, buffering across
        callbacks since the ratio rarely divides the callback's block size
        evenly, and slices off exact block_ms frames (Silero needs exactly
        512 samples at 16kHz, not "close enough"). Runs until `stop_event` is
        set - a fresh Event per start(), not the shared self._stop, since
        that one is never cleared once set and this thread is recreated on
        every device switch."""
        buffer = np.empty(0, dtype=np.float32)
        block_size = int(self.cfg.sample_rate * self.cfg.block_ms / 1000)
        while not stop_event.is_set():
            try:
                raw = self._raw_frames.get(timeout=0.5)
            except queue.Empty:
                continue
            resampled = resample_poly(raw, self.cfg.sample_rate, self._capture_rate).astype(np.float32)
            buffer = np.concatenate([buffer, resampled])
            while len(buffer) >= block_size:
                self.frames.put(buffer[:block_size].copy())
                buffer = buffer[block_size:]

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

    def _open_stream(self, block_size: int) -> sd.InputStream:
        """Opens and starts the InputStream. No retry here on purpose: a
        failing open on this box's hardware (two ALSA "hw:" devices on the
        same physical codec, tried right after one another) took PortAudio
        itself several seconds per attempt to give up on internally, so
        retrying a few times turned a single failure into a 30+ second hang
        before the operator saw anything - worse than just failing fast and
        letting them try a different device."""
        stream = sd.InputStream(
            device=self.cfg.input_device,
            samplerate=self._capture_rate,
            channels=self.cfg.channels,
            blocksize=block_size,
            # PortAudio's default latency (tuned for low-latency pro
            # interfaces) is too tight for real hardware codecs - a real
            # Line-In here overflowed constantly at the default even with a
            # trivial callback. Its "high" preset isn't much better on raw
            # ALSA "hw" devices: ~35 ms of buffer, which any moment the
            # callback can't get the GIL for that long overruns - e.g.
            # while NeMo transcribes a long buffer, measured as a steady
            # stream of overflows (dropped audio, garbled words) in
            # sts-caption. The buffer is only headroom: blocks are still
            # delivered every block_ms as they're captured, so a deep one
            # costs no delay unless the callback actually falls behind.
            latency=CAPTURE_BUFFER_S,
            dtype="float32",
            callback=self._callback,
        )
        stream.start()
        return stream

    def start(self) -> None:
        """Opens the configured source. A no-op for `source == "browser"`:
        frames arrive via `push_browser_frame` instead of a local stream."""
        if self.cfg.source == "browser":
            vlog("audio", "waiting for browser microphone frames")
            return
        self._check_mic_permission()
        self._capture_rate = self._resolve_capture_rate(self.cfg.input_device)
        resampling = self._capture_rate != self.cfg.sample_rate
        # A forced blocksize not evenly related to the hardware's own period
        # size (e.g. 1411 samples/32ms at 44100 Hz - not a "nice" number for
        # ALSA) made a real Line-In codec overflow continuously even with
        # generous latency; _resample_worker already buffers arbitrary chunk
        # sizes, so when resampling let PortAudio pick its own natural period
        # (0) instead of dictating one it may not handle cleanly.
        block_size = 0 if resampling else int(self._capture_rate * self.cfg.block_ms / 1000)
        self._stream = self._open_stream(block_size)
        self._last_frame_at = time.monotonic()
        if resampling:
            with self._raw_frames.mutex:
                self._raw_frames.queue.clear()
            self._resample_stop = threading.Event()
            self._resample_thread = threading.Thread(
                target=self._resample_worker, args=(self._resample_stop,), daemon=True
            )
            self._resample_thread.start()
        info = sd.query_devices(self.cfg.input_device, "input")
        note = f" (resampled from {self._capture_rate} Hz)" if self._capture_rate != self.cfg.sample_rate else ""
        vlog("audio", f"listening on input [{info['index']}] {info['name']} "
                      f"({self.cfg.sample_rate} Hz, {self.cfg.block_ms} ms frames){note}")

    def _check_mic_permission(self) -> None:
        """Warns - and, once per run, opens the setting - when macOS won't
        let this process hear any input device."""
        self.mic_permission = mic_permission()
        vlog("audio", f"macOS microphone permission: {self.mic_permission}")
        if not is_blocked(self.mic_permission):
            return
        print(f"[audio] {DENIED_MESSAGE}")
        if not self._opened_mic_settings:
            self._opened_mic_settings = True
            open_mic_settings()

    def switch_to_alsa(self, device: int | None | object = _KEEP_CURRENT_DEVICE) -> None:
        """Moves capture to a local ALSA input device, switching away from the
        browser-mic source if that was active. Omitting `device` entirely
        keeps whatever device was already configured - used when only the
        source itself needs to change back, not the device index (e.g.
        mic_stop). Pass `device=None` to mean PortAudio/ALSA's own "system
        default" input, same as any other explicit device value - it is one,
        not the absence of one. If the device won't open, capture goes back
        to the previous source/device and the error is raised."""
        previous_device = self.cfg.input_device
        previous_source = self.cfg.source
        self.stop()
        self.cfg.source = "alsa"
        if device is not _KEEP_CURRENT_DEVICE:
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
        self._resample_stop.set()
        if self._resample_thread is not None:
            self._resample_thread.join(timeout=1)
            self._resample_thread = None
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
