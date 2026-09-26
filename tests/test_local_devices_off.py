"""Turning the local input/output device off (clicking the device in use in
the caption page's device list) - the browser mic and voice streaming keep
working without them."""

import threading
import time

import numpy as np
import pytest

from sts.audio.capture import AudioCapture
from sts.audio.playback import AudioPlayback
from sts.config import AudioConfig, PipelineConfig
from sts.persistent_config import PersistentConfig
from sts.pipeline import Pipeline


class _StubStream:
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def no_hardware(monkeypatch):
    monkeypatch.setattr("sts.audio.capture.sd.InputStream", lambda **kw: _StubStream(**kw))
    monkeypatch.setattr(
        "sts.audio.capture.sd.query_devices",
        lambda device, kind: {"index": device or 0, "name": "Stub", "default_samplerate": 16_000.0},
    )
    monkeypatch.setattr("sts.audio.capture.sd.check_input_settings", lambda **kw: None)
    monkeypatch.setattr("sts.audio.devices.check_input", lambda *a, **k: None)
    monkeypatch.setattr("sts.audio.devices.check_output", lambda *a, **k: None)


class _Playback:
    stopped = 0

    def stop(self):
        self.stopped += 1


def _pipeline(tmp_path):
    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.cfg.audio.input_device, p.cfg.audio.output_device = 3, 5
    p.capture = AudioCapture(p.cfg.audio)
    p.playback = _Playback()
    p._device_lock = threading.Lock()
    p._local_source = "alsa"
    p.broadcaster = None
    p._persistent_config = PersistentConfig(str(tmp_path / "sts.db"))
    return p


def test_the_input_can_be_turned_off_and_back_on(tmp_path, no_hardware):
    p = _pipeline(tmp_path)
    p._handle_device_action("set_audio_devices", {"input": "off"})
    assert p.cfg.audio.source == "none"
    assert p.capture._stream is None
    assert p._persistent_config.get("input_device") == "off"
    assert p.cfg.audio.output_enabled  # the other side untouched
    p._handle_device_action("set_audio_devices", {"input": "3"})
    assert p.cfg.audio.source == "alsa" and p.cfg.audio.input_device == 3
    assert p._persistent_config.get("input_device") == "3"


def test_the_browser_mic_goes_back_to_the_input_being_off(tmp_path, no_hardware):
    p = _pipeline(tmp_path)
    p._handle_device_action("set_audio_devices", {"input": "off"})
    p._handle_device_action("mic_start", {})
    assert p.cfg.audio.source == "browser"
    p._handle_device_action("mic_stop", {})
    assert p.cfg.audio.source == "none"


def test_changing_the_output_leaves_the_browser_mic_alone(tmp_path, no_hardware):
    p = _pipeline(tmp_path)
    p._handle_device_action("mic_start", {})
    p._handle_device_action("set_audio_devices", {"output": "7"})
    assert p.cfg.audio.source == "browser"
    assert p.cfg.audio.output_device == 7


def test_the_output_can_be_turned_off(tmp_path, no_hardware):
    p = _pipeline(tmp_path)
    p._handle_device_action("set_audio_devices", {"output": "off"})
    assert not p.cfg.audio.output_enabled
    assert p.cfg.audio.output_device == 5  # remembered, for turning it back on
    assert p.playback.stopped == 1
    assert p._persistent_config.get("output_device") == "off"
    p._handle_device_action("set_audio_devices", {"output": "5"})
    assert p.cfg.audio.output_enabled
    assert p._persistent_config.get("output_device") == "5"


def test_with_the_output_off_a_clip_plays_silently_in_real_time(monkeypatch):
    def no_device(*args, **kwargs):
        raise AssertionError("opened an output device")

    monkeypatch.setattr("sts.audio.playback.sd.OutputStream", no_device)
    playback = AudioPlayback(AudioConfig(output_enabled=False))
    chunks = []
    t0 = time.monotonic()
    playback.play(np.ones(2400, dtype=np.float32), sample_rate=24_000, on_chunk=chunks.append)  # 0.1 s
    elapsed = time.monotonic() - t0
    assert 0.09 <= elapsed < 0.3
    assert sum(len(c) for c in chunks) == 2400  # the output meter still sees it
    assert playback.pending_s() == pytest.approx(0.0, abs=0.02)


def test_a_side_saved_as_off_is_not_taken_for_a_device(tmp_path):
    from sts import cli

    pc = PersistentConfig(str(tmp_path / "sts.db"))
    pc.set("input_device", "off")
    pc.set("output_device", "off")
    assert cli.saved_device(pc, "input_device", lambda device: None) is None  # no device to restore
