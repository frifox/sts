"""Plays synthesized speech out to a sound card output.

Uses one persistent output stream that clips are written into back to back,
so consecutive sentences play without the gap that starting a fresh
`sd.play()` for each one introduces.
"""

import threading
import time

import numpy as np
import sounddevice as sd
import soxr

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
        self._resolved_rates: dict[int, int] = {}  # clip rate -> device-accepted rate, probed once each
        # Converts clip rate -> device rate when they differ, as one
        # continuous stream across the seamless chunks of a piece (see
        # play()'s `seamless`), so chunk joins resample exactly like the
        # piece would have in one go - no filter-edge seam every chunk.
        # It holds back a few ms at the end until flush() or the next clip.
        self._resampler: soxr.ResampleStream | None = None
        self._resampler_rates: tuple[int, int] | None = None

    def _target_rate(self, requested: int) -> int:
        """The rate to actually open the stream at: `requested` if the output
        device accepts it directly, else the device's own default rate
        (opened this way - PortAudio's raw ALSA "hw" access, no "plug" layer
        - many real sound cards only accept a couple of fixed hardware rates,
        e.g. 44100/48000, and flatly reject a TTS model's own output rate,
        typically 24kHz).

        Cached per requested rate: `check_output_settings` briefly opens the
        device itself to probe it, which collides ("AlsaOpen" failure) with
        the persistent stream this class already keeps open on the same
        exclusive-access hw device if done on every clip instead of once."""
        cached = self._resolved_rates.get(requested)
        if cached is not None:
            return cached
        try:
            sd.check_output_settings(device=self.cfg.output_device, samplerate=requested)
            target = requested
        except Exception:
            info = sd.query_devices(self.cfg.output_device, "output")
            target = int(info["default_samplerate"])
        self._resolved_rates[requested] = target
        return target

    def _ensure_stream(self, sample_rate: int) -> tuple[sd.OutputStream, int]:
        target = self._target_rate(sample_rate)
        if self._stream is None or self._stream_rate != target:
            self._close_stream()
            self._stream = sd.OutputStream(
                samplerate=target, channels=1, dtype="float32", device=self.cfg.output_device
            )
            self._stream.start()
            self._stream_rate = target
            info = sd.query_devices(self.cfg.output_device, "output")
            note = f" (resampled from {sample_rate} Hz)" if target != sample_rate else ""
            vlog("audio", f"output stream open on [{info['index']}] {info['name']} at {target} Hz{note}")
        return self._stream, target

    def _close_stream(self) -> None:
        self._resampler = None
        if self._stream is not None:
            stream, self._stream, self._stream_rate = self._stream, None, None
            try:
                stream.abort()
                stream.close()
            except Exception as exc:
                # A stream that went bad (device gone, machine slept) can
                # raise while being torn down; it's being discarded anyway,
                # and a fresh one is opened on the next clip.
                vlog("audio", f"output stream didn't close cleanly ({exc!r}); discarded anyway")

    def forget_devices(self) -> None:
        """Drops everything learned about the output device - to be called
        after the audio system is re-initialized (see
        reinitialize_audio_system), since the accepted-rate probes were made
        against devices that may no longer be the same. Closes the stream."""
        with self._lock:
            self._close_stream()
            self._resolved_rates.clear()
            self._scheduled_end = 0.0

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        blocking: bool = True,
        pause_before: float = 0.0,
        should_continue=None,
        on_chunk=None,
        seamless: bool = False,
    ) -> None:
        """Queues `audio` on the output stream. The write blocks only until the
        stream's buffer has room, so back-to-back calls play continuously.

        `pause_before` is the minimum silence to leave after whatever played
        before (0 for continuation chunks of the same text).

        `should_continue`, if given, is polled every ~50ms of audio and lets
        the caller cut *this* clip short mid-playback (e.g. the operator
        unchecking Voice) - without calling `stop()`, which aborts/closes the
        stream from whatever thread calls it. Since `write()` below blocks
        the *this* method's own thread, not the caller's, aborting the
        stream from a different thread while this one is mid-write has been
        observed to segfault PortAudio's ALSA backend; polling in small
        chunks on this same thread gets the same prompt cutoff safely.

        `on_chunk`, if given, is called with each chunk as it's written (used
        to report output level for the volume-bar meter) - safe to do inline
        here since, unlike audio *capture*, this runs on a plain Python
        thread (the caller's), not a PortAudio real-time callback.

        Always written in ~50ms chunks (whether or not `should_continue` is
        given) rather than as one big write, so `cfg.volume` - read fresh per
        chunk, not baked in once up front - takes effect within ~50ms of the
        operator dragging the volume slider, even mid-clip, instead of only
        on the next clip pulled off the queue.

        `seamless` marks a streamed chunk that continues the previous clip
        mid-piece: it's resampled as a continuation of it. Otherwise the
        previous piece's held-back resampler tail is played out first.

        With the local output turned off (`cfg.output_enabled` False) nothing
        is written anywhere, but the clip still takes its real time - see
        _play_silently."""
        if not self.cfg.output_enabled:
            self._play_silently(audio, sample_rate, pause_before, should_continue, on_chunk)
            return
        with self._lock:
            stream, target_rate = self._ensure_stream(sample_rate)
            duration = len(audio) / sample_rate
            tail = np.zeros(0, dtype=np.float32)
            if not seamless or self._resampler_rates != (sample_rate, target_rate):
                tail = self._take_resampler_tail()
            if target_rate != sample_rate:
                if self._resampler is None:
                    self._resampler = soxr.ResampleStream(sample_rate, target_rate, 1, dtype="float32", quality="HQ")
                    self._resampler_rates = (sample_rate, target_rate)
                audio = self._resampler.resample_chunk(np.ascontiguousarray(audio, dtype=np.float32))
                sample_rate = target_rate
            samples = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1, 1)
            now = time.monotonic()
            pad = pause_needed(now, self._scheduled_end, pause_before) if pause_before > 0 else 0.0
            self._scheduled_end = max(now, self._scheduled_end) + pad + len(samples) / sample_rate
            if pad > 0:
                samples = np.concatenate([np.zeros((int(pad * sample_rate), 1), dtype=np.float32), samples])
            if tail.size:
                samples = np.concatenate([tail.reshape(-1, 1), samples])
        vlog("audio", f"playing {duration:.1f}s of speech" + (f" after a {pad:.1f}s pause" if pad > 0 else ""))
        chunk_frames = max(1, int(0.05 * sample_rate))
        for start in range(0, len(samples), chunk_frames):
            if should_continue is not None and not should_continue():
                with self._lock:
                    self._scheduled_end = 0.0  # what got cut short isn't "queued" anymore
                    self._resampler = None
                vlog("audio", "playback cut short")
                return
            chunk = samples[start : start + chunk_frames] * self.cfg.volume
            stream.write(chunk)
            if on_chunk:
                on_chunk(chunk)

    def _play_silently(self, audio, sample_rate, pause_before, should_continue, on_chunk) -> None:
        """play() with no output device: waits out the clip (and its lead-in
        pause) in real time instead of writing it - so everything paced by
        playback (the next clip's timing, catch-up, pending_s, the output
        meter, voice streamed to browsers) behaves as if it had played."""
        samples = np.ascontiguousarray(audio, dtype=np.float32)
        with self._lock:
            self._resampler = self._resampler_rates = None  # nothing held back for a device
            now = time.monotonic()
            pad = pause_needed(now, self._scheduled_end, pause_before) if pause_before > 0 else 0.0
            start = max(now, self._scheduled_end) + pad
            self._scheduled_end = start + len(samples) / sample_rate
        chunk_frames = max(1, int(0.05 * sample_rate))
        time.sleep(max(0.0, start - time.monotonic()))
        for i in range(0, len(samples), chunk_frames):
            if should_continue is not None and not should_continue():
                with self._lock:
                    self._scheduled_end = 0.0
                return
            if on_chunk:
                on_chunk(samples[i : i + chunk_frames] * self.cfg.volume)
            time.sleep(max(0.0, start + (i + chunk_frames) / sample_rate - time.monotonic()))

    def _take_resampler_tail(self) -> np.ndarray:
        """Ends the current resampling stream and returns what it was still
        holding back. Call with _lock held."""
        resampler, self._resampler, self._resampler_rates = self._resampler, None, None
        if resampler is None:
            return np.zeros(0, dtype=np.float32)
        return resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)

    def flush(self) -> None:
        """Plays out the last few ms of the previous piece that the
        resampler was holding back in case a seamless chunk followed it -
        to be called once nothing more of that piece is coming soon."""
        with self._lock:
            tail = self._take_resampler_tail()
            stream = self._stream
        if tail.size and stream is not None:
            stream.write((tail * self.cfg.volume).reshape(-1, 1))

    @property
    def holding_tail(self) -> bool:
        return self._resampler is not None

    def pending_s(self) -> float:
        """Seconds of already-written audio (and its lead-in pause) still to
        play out - what's left of the current clip."""
        return max(0.0, self._scheduled_end - time.monotonic())

    def stop(self) -> None:
        """Cuts off whatever is currently playing (used when the operator resets)."""
        with self._lock:
            self._close_stream()
            self._scheduled_end = 0.0
