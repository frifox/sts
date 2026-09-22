import struct
import threading

import numpy as np
import pytest

from sts.audio.capture import AudioCapture
from sts.config import AudioConfig


class _StubStream:
    """Stands in for `sounddevice.InputStream` so ALSA-touching paths can be
    unit tested without real hardware (consistent with how this box's own
    tests never open real audio devices elsewhere)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def stub_sounddevice(monkeypatch):
    monkeypatch.setattr("sts.audio.capture.sd.InputStream", lambda **kw: _StubStream(**kw))
    monkeypatch.setattr(
        "sts.audio.capture.sd.query_devices", lambda device, kind: {"index": device or 0, "name": "Stub Device"}
    )


def _capture() -> AudioCapture:
    return AudioCapture(AudioConfig(sample_rate=16_000, block_ms=32))


def test_capture_defaults_to_alsa_source():
    assert AudioConfig().source == "alsa"


def test_switch_to_browser_is_a_no_op_start_with_no_hardware():
    capture = _capture()
    capture.switch_to_browser()
    assert capture.cfg.source == "browser"
    capture.start()  # must not try to open a real ALSA stream
    assert capture._stream is None


def test_push_browser_frame_only_queues_while_browser_is_the_source():
    capture = _capture()
    frame = np.zeros(512, dtype=np.float32)

    capture.push_browser_frame(frame)  # still "alsa" by default - dropped
    assert capture.frames.empty()

    capture.switch_to_browser()
    capture.push_browser_frame(frame)
    assert capture.frames.get_nowait() is frame


def test_switch_to_browser_clears_frames_queued_from_the_previous_source():
    capture = _capture()
    capture.frames.put(np.zeros(512, dtype=np.float32))
    capture.switch_to_browser()
    assert capture.frames.empty()


def test_switch_to_alsa_updates_source_and_device(stub_sounddevice):
    capture = _capture()
    capture.switch_to_browser()
    capture.switch_to_alsa(device=3)
    assert capture.cfg.source == "alsa"
    assert capture.cfg.input_device == 3
    assert capture._stream is not None


def test_switch_to_alsa_without_a_device_argument_keeps_the_configured_index(stub_sounddevice):
    capture = _capture()
    capture.cfg.input_device = 7
    capture.switch_to_browser()
    capture.switch_to_alsa()  # no explicit device - falls back to whatever was already configured
    assert capture.cfg.source == "alsa"
    assert capture.cfg.input_device == 7


def test_switch_to_alsa_restores_previous_source_and_device_on_failure(stub_sounddevice, monkeypatch):
    capture = _capture()
    capture.cfg.input_device = 7
    capture.switch_to_browser()

    calls = {"n": 0}

    def flaky_stream(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("device busy")
        return _StubStream(**kw)

    monkeypatch.setattr("sts.audio.capture.sd.InputStream", flaky_stream)
    with pytest.raises(RuntimeError):
        capture.switch_to_alsa(device=9)
    # Failed to open device 9 - back to being a browser source with the old index intact.
    assert capture.cfg.source == "browser"
    assert capture.cfg.input_device == 7


def _frame_bytes(sample_rate: int, samples: np.ndarray) -> bytes:
    return struct.pack("<I", sample_rate) + samples.astype("<f4").tobytes()


def _mic_pipeline():
    """A Pipeline stub with a real AudioCapture (browser mode - no hardware
    needed) so `_handle_mic_frame`'s parsing/validation can be exercised."""
    from sts.config import PipelineConfig
    from sts.pipeline import Pipeline

    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.capture = AudioCapture(p.cfg.audio)
    p.capture.switch_to_browser()
    return p


def test_handle_mic_frame_queues_a_well_formed_frame():
    p = _mic_pipeline()
    samples = np.linspace(-1, 1, 512, dtype=np.float32)
    p._handle_mic_frame(_frame_bytes(16_000, samples))
    queued = p.capture.frames.get_nowait()
    assert queued.dtype == np.float32
    assert len(queued) == 512
    assert np.allclose(queued, samples)


def test_handle_mic_frame_drops_wrong_sample_rate():
    p = _mic_pipeline()
    p._handle_mic_frame(_frame_bytes(44_100, np.zeros(512, dtype=np.float32)))
    assert p.capture.frames.empty()


def test_handle_mic_frame_drops_wrong_frame_length():
    p = _mic_pipeline()
    p._handle_mic_frame(_frame_bytes(16_000, np.zeros(256, dtype=np.float32)))
    assert p.capture.frames.empty()


def test_handle_mic_frame_drops_truncated_payload():
    p = _mic_pipeline()
    p._handle_mic_frame(b"\x00\x01")  # shorter than the 4-byte header alone
    assert p.capture.frames.empty()


def test_mic_start_and_stop_switch_the_capture_source(stub_sounddevice):
    from sts.config import PipelineConfig
    from sts.pipeline import Pipeline

    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.capture = AudioCapture(p.cfg.audio)
    p._device_lock = threading.Lock()
    p.broadcaster = None

    p._handle_device_action("mic_start", {})
    assert p.cfg.audio.source == "browser"

    p._handle_device_action("mic_stop", {})
    assert p.cfg.audio.source == "alsa"
