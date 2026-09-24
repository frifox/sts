import numpy as np
import pytest

from sts.audio.playback import AudioPlayback, pause_needed
from sts.config import AudioConfig


def test_no_pause_before_first_clip():
    assert pause_needed(now=100.0, scheduled_end=0.0, gap=1.0) == 0.0


def test_full_pause_when_playback_is_catching_up():
    # earlier audio is still playing (ends in the future) -> a full gap after it
    assert pause_needed(now=100.0, scheduled_end=103.0, gap=1.0) == 1.0


def test_partial_pause_when_previous_just_ended():
    assert abs(pause_needed(now=100.4, scheduled_end=100.0, gap=1.0) - 0.6) < 1e-9


def test_no_pause_after_a_long_natural_gap():
    assert pause_needed(now=110.0, scheduled_end=100.0, gap=1.0) == 0.0


class _StubOutputStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.written: list[np.ndarray] = []

    def start(self):
        pass

    def stop(self):
        pass

    def abort(self):
        pass

    def close(self):
        pass

    def write(self, samples):
        self.written.append(samples.copy())


@pytest.fixture
def stub_output(monkeypatch):
    """Stands in for `sounddevice.OutputStream` so play()'s chunking/
    interruption logic can be unit tested without real audio hardware."""
    holder: dict[str, _StubOutputStream] = {}

    def make_stream(**kwargs):
        holder["stream"] = _StubOutputStream(**kwargs)
        return holder["stream"]

    monkeypatch.setattr("sts.audio.playback.sd.OutputStream", make_stream)
    monkeypatch.setattr("sts.audio.playback.sd.check_output_settings", lambda **kw: None)
    monkeypatch.setattr(
        "sts.audio.playback.sd.query_devices", lambda device, kind: {"index": device or 0, "name": "Stub Output"}
    )
    return holder


def test_play_without_should_continue_still_writes_the_whole_clip(stub_output):
    """Written in chunks even without should_continue (see the volume test
    below for why), but nothing is skipped or cut short."""
    pb = AudioPlayback(AudioConfig())
    audio = np.zeros(24_000 * 2, dtype=np.float32)  # 2s clip
    pb.play(audio, sample_rate=24_000)
    stream = stub_output["stream"]
    assert len(stream.written) > 1
    assert sum(len(c) for c in stream.written) == len(audio)


def test_play_applies_the_configured_volume_as_a_gain(stub_output):
    """The Output dropdown's volume slider sets cfg.volume - play() must
    scale samples by it (a local-only gain; a connected browser's own stream
    gets the unscaled clip, see broadcast_audio's call site)."""
    pb = AudioPlayback(AudioConfig(volume=0.5))
    audio = np.full(100, 0.8, dtype=np.float32)
    pb.play(audio, sample_rate=24_000)
    written = np.concatenate(stub_output["stream"].written).reshape(-1)
    assert np.allclose(written, 0.4)


def test_volume_change_mid_clip_takes_effect_on_the_next_chunk(stub_output):
    """The bug this guards: volume used to be multiplied into the whole clip
    once, up front, so dragging the slider mid-sentence had no audible effect
    until the *next* queued clip started - up to several seconds later. It's
    now read fresh from cfg.volume per ~50ms chunk instead."""
    cfg = AudioConfig()
    pb = AudioPlayback(cfg)
    chunk_frames = int(0.05 * 24_000)  # 1200 samples/chunk at 24kHz
    audio = np.full(chunk_frames * 3, 0.8, dtype=np.float32)  # exactly 3 chunks

    calls = {"n": 0}

    def should_continue():
        calls["n"] += 1
        if calls["n"] == 2:
            cfg.volume = 0.25  # the operator drags the slider between chunks 1 and 2
        return True

    pb.play(audio, sample_rate=24_000, should_continue=should_continue)

    written = stub_output["stream"].written
    assert len(written) == 3
    assert np.allclose(written[0], 0.8)   # written before the drag, at the default volume (1.0)
    assert np.allclose(written[1], 0.2)   # 0.8 * 0.25 - applied to this chunk already, not the next play() call
    assert np.allclose(written[2], 0.2)


def test_should_continue_false_cuts_playback_short_instead_of_finishing_the_clip(stub_output):
    """The regression this guards: unchecking Voice mid-sentence must stop
    the sound actually coming out, not just prevent the *next* sentence from
    starting - without calling stream.abort()/close() from a different
    thread than the one blocked in write(), which can segfault PortAudio's
    ALSA backend (reproduced directly against real hardware while
    investigating this)."""
    pb = AudioPlayback(AudioConfig())
    audio = np.zeros(24_000 * 2, dtype=np.float32)  # 2s clip

    calls = {"n": 0}

    def should_continue():
        calls["n"] += 1
        return calls["n"] <= 1  # let exactly one chunk through, then cut

    pb.play(audio, sample_rate=24_000, should_continue=should_continue)

    stream = stub_output["stream"]
    total_written = sum(len(chunk) for chunk in stream.written)
    assert 0 < total_written < len(audio)  # some audio played, but nowhere near the whole clip
    assert len(stream.written) == 1


def test_should_continue_true_throughout_writes_the_whole_clip(stub_output):
    pb = AudioPlayback(AudioConfig())
    audio = np.zeros(24_000 * 2, dtype=np.float32)
    pb.play(audio, sample_rate=24_000, should_continue=lambda: True)
    stream = stub_output["stream"]
    assert sum(len(c) for c in stream.written) == len(audio)
