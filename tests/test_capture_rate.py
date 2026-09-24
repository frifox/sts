"""Covers the bug this was written to fix: switching to a real hardware input
(e.g. a motherboard's Line-In codec, opened via PortAudio's raw ALSA "hw"
access) that doesn't support the pipeline's 16kHz directly used to fail
outright with "Invalid sample rate" and silently revert the operator's
selection. AudioCapture now falls back to the device's own rate and
resamples; these tests exercise that path without real hardware.
"""

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
    # block_ms worth of samples at the *capture* rate, not the pipeline's 16kHz.
    assert opened["blocksize"] == int(44_100 * 32 / 1000)


def test_callback_passes_native_rate_frames_through_unchanged(monkeypatch):
    monkeypatch.setattr("sts.audio.capture.sd.check_input_settings", lambda **kw: None)
    capture = _capture()
    capture._capture_rate = 16_000  # matches cfg.sample_rate - no resampling
    frame = np.linspace(-1, 1, 512, dtype=np.float32).reshape(-1, 1)

    capture._callback(frame, 512, None, None)

    queued = capture.frames.get_nowait()
    assert len(queued) == 512
    assert np.allclose(queued, frame[:, 0])


def test_callback_resamples_and_buffers_across_multiple_calls():
    capture = _capture()
    capture._capture_rate = 44_100
    block_size_16k = int(16_000 * 32 / 1000)  # 512
    # 44100 Hz doesn't divide evenly into 32ms chunks that resample to exactly
    # 512 samples every single call - the point of this test is that frames.
    # queued are still always exactly 512 samples regardless, via buffering.
    native_block = int(44_100 * 32 / 1000)  # 1411 - not evenly resampling to 512
    tone = np.sin(2 * np.pi * 440 * np.arange(native_block * 6) / 44_100).astype(np.float32)

    for i in range(6):
        chunk = tone[i * native_block : (i + 1) * native_block].reshape(-1, 1)
        capture._callback(chunk, native_block, None, None)

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
