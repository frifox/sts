"""Covers the bug this was written to fix: switching to a real hardware input
(e.g. a motherboard's Line-In codec, opened via PortAudio's raw ALSA "hw"
access) that doesn't support the pipeline's 16kHz directly used to fail
outright with "Invalid sample rate" and silently revert the operator's
selection. AudioCapture now falls back to the device's own rate and
resamples; these tests exercise that path without real hardware.
"""

import threading
import time

import numpy as np
import pytest

from sts.audio.capture import AudioCapture
from sts.config import AudioConfig


class _StubStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


def _capture() -> AudioCapture:
    return AudioCapture(AudioConfig(sample_rate=16_000, block_ms=32))


def test_resolve_capture_rate_uses_the_pipeline_rate_when_the_device_accepts_it(monkeypatch):
    monkeypatch.setattr("sts.audio.capture.sd.check_input_settings", lambda **kw: None)
    capture = _capture()
    assert capture._resolve_capture_rate(4) == 16_000


def test_resolve_capture_rate_falls_back_to_the_devices_native_rate(monkeypatch):
    def reject_16k(**kw):
        if kw["samplerate"] == 16_000:
            raise Exception("Invalid sample rate [PaErrorCode -9997]")

    monkeypatch.setattr("sts.audio.capture.sd.check_input_settings", reject_16k)
    monkeypatch.setattr(
        "sts.audio.capture.sd.query_devices",
        lambda device, kind: {"index": device, "name": "Line In", "default_samplerate": 44_100.0},
    )
    capture = _capture()
    assert capture._resolve_capture_rate(4) == 44_100


def test_open_stream_fails_fast_without_retrying(monkeypatch):
    """A retry loop was tried here at one point (for two ALSA "hw:" devices
    on the same physical card refusing to open back-to-back), but on this
    box's hardware each failing attempt took PortAudio itself several
    seconds to give up on internally - retrying multiplied that into a 30+
    second hang before the operator saw anything. A single fast failure (so
    they can immediately try a different device) beats that."""
    calls = {"n": 0}

    def always_fails(**kw):
        calls["n"] += 1
        raise RuntimeError("device genuinely unusable")

    monkeypatch.setattr("sts.audio.capture.sd.InputStream", always_fails)
    capture = _capture()

    with pytest.raises(RuntimeError, match="genuinely unusable"):
        capture._open_stream(block_size=512)
    assert calls["n"] == 1


def test_start_opens_the_stream_at_the_resolved_rate_not_always_16k(monkeypatch):
    opened = {}

    def fake_stream(**kwargs):
        opened.update(kwargs)
        return _StubStream(**kwargs)

    monkeypatch.setattr("sts.audio.capture.sd.InputStream", fake_stream)
    monkeypatch.setattr("sts.audio.capture.sd.check_input_settings", lambda **kw: (_ for _ in ()).throw(Exception()))
    monkeypatch.setattr(
        "sts.audio.capture.sd.query_devices",
        lambda device, kind: {"index": device or 0, "name": "Line In", "default_samplerate": 44_100.0},
    )
    capture = _capture()
    capture.cfg.input_device = 4
    capture.start()

    assert opened["samplerate"] == 44_100
    # 0 (PortAudio's own choice), not a dictated size like 1411 (32ms at
    # 44100 Hz) - forcing a period size unrelated to the hardware's own
    # caused a real Line-In codec to overflow continuously; _resample_worker
    # buffers whatever size actually arrives, so there's no need to dictate one.
    assert opened["blocksize"] == 0
    capture.stop()  # release the resample thread test_start spawned


def test_callback_passes_native_rate_frames_through_unchanged(monkeypatch):
    monkeypatch.setattr("sts.audio.capture.sd.check_input_settings", lambda **kw: None)
    capture = _capture()
    capture._capture_rate = 16_000  # matches cfg.sample_rate - no resampling
    frame = np.linspace(-1, 1, 512, dtype=np.float32).reshape(-1, 1)

    capture._callback(frame, 512, None, None)

    queued = capture.frames.get_nowait()
    assert len(queued) == 512
    assert np.allclose(queued, frame[:, 0])


def test_callback_hands_off_to_the_resample_worker_instead_of_resampling_inline():
    """The regression this guards: resample_poly's filter design/convolution
    ran directly inside _callback originally, which is PortAudio's real-time
    ALSA thread - too slow for that budget in practice, and it caused
    genuine "input status: input overflow" (dropped audio) under sustained
    use. _callback must stay trivial and hand resampling to another thread."""
    capture = _capture()
    capture._capture_rate = 44_100
    chunk = np.zeros((1411, 1), dtype=np.float32)

    capture._callback(chunk, 1411, None, None)

    assert capture.frames.empty()
    assert capture._raw_frames.qsize() == 1


def test_resample_worker_buffers_across_chunks_into_exact_block_size_frames():
    capture = _capture()
    capture._capture_rate = 44_100
    block_size_16k = int(16_000 * 32 / 1000)  # 512
    # 44100 Hz doesn't divide evenly into 32ms chunks that resample to exactly
    # 512 samples every single call - the point of this test is that frames
    # queued are still always exactly 512 samples regardless, via buffering.
    native_block = int(44_100 * 32 / 1000)  # 1411 - not evenly resampling to 512
    tone = np.sin(2 * np.pi * 440 * np.arange(native_block * 6) / 44_100).astype(np.float32)
    for i in range(6):
        capture._raw_frames.put(tone[i * native_block : (i + 1) * native_block])

    stop_event = threading.Event()
    worker = threading.Thread(target=capture._resample_worker, args=(stop_event,), daemon=True)
    worker.start()
    deadline = time.monotonic() + 2
    while capture._raw_frames.qsize() > 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    stop_event.set()
    worker.join(timeout=2)

    produced = []
    while not capture.frames.empty():
        produced.append(capture.frames.get_nowait())

    assert len(produced) >= 5  # ~6 * 1411 native samples resampled to 16kHz gives several full 512-sample frames
    for frame in produced:
        assert len(frame) == block_size_16k
        assert frame.dtype == np.float32


def test_check_input_falls_back_to_the_devices_own_rate(monkeypatch):
    from sts.audio.devices import check_input

    calls = []

    def fake_check(**kw):
        calls.append(kw["samplerate"])
        if kw["samplerate"] == 16_000:
            raise Exception("Invalid sample rate [PaErrorCode -9997]")

    monkeypatch.setattr("sts.audio.devices.sd.check_input_settings", fake_check)
    monkeypatch.setattr(
        "sts.audio.devices.sd.query_devices",
        lambda device, kind: {"default_samplerate": 44_100.0},
    )

    check_input(device=4, sample_rate=16_000, channels=1)  # doesn't raise

    assert calls == [16_000, 44_100.0]


def test_check_input_still_raises_when_even_the_devices_own_rate_fails(monkeypatch):
    from sts.audio.devices import check_input

    monkeypatch.setattr(
        "sts.audio.devices.sd.check_input_settings", lambda **kw: (_ for _ in ()).throw(Exception("broken device"))
    )
    monkeypatch.setattr("sts.audio.devices.sd.query_devices", lambda device, kind: {"default_samplerate": 44_100.0})

    with pytest.raises(Exception, match="broken device"):
        check_input(device=4, sample_rate=16_000, channels=1)


def test_check_output_falls_back_to_the_devices_own_rate(monkeypatch):
    """Same bug as check_input, on the output side: reported switching the
    motherboard's Line Out gave the same "Invalid sample rate" error, since
    check_output had no fallback even though AudioPlayback._target_rate
    already resamples for exactly this case at actual playback time."""
    from sts.audio.devices import check_output

    calls = []

    def fake_check(**kw):
        calls.append(kw["samplerate"])
        if kw["samplerate"] == 24_000:
            raise Exception("Invalid sample rate [PaErrorCode -9997]")

    monkeypatch.setattr("sts.audio.devices.sd.check_output_settings", fake_check)
    monkeypatch.setattr(
        "sts.audio.devices.sd.query_devices",
        lambda device, kind: {"default_samplerate": 44_100.0},
    )

    check_output(device=0, sample_rate=24_000)  # doesn't raise

    assert calls == [24_000, 44_100.0]


def test_check_output_still_raises_when_even_the_devices_own_rate_fails(monkeypatch):
    from sts.audio.devices import check_output

    monkeypatch.setattr(
        "sts.audio.devices.sd.check_output_settings", lambda **kw: (_ for _ in ()).throw(Exception("broken device"))
    )
    monkeypatch.setattr("sts.audio.devices.sd.query_devices", lambda device, kind: {"default_samplerate": 44_100.0})

    with pytest.raises(Exception, match="broken device"):
        check_output(device=0, sample_rate=24_000)
