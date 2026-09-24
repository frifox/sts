"""Speeding speech playback up when it falls behind, and back down once caught up."""

import queue

import numpy as np
import pytest

from sts.audio.tempo import time_stretch
from sts.config import PipelineConfig
from sts.pipeline import Pipeline, _Clip

SR = 24_000


def _tone(seconds: float, freq: float = 150.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dominant_hz(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio))
    return float(np.fft.rfftfreq(len(audio), 1 / SR)[np.argmax(spectrum)])


# -- time_stretch --------------------------------------------------------------


@pytest.mark.parametrize("rate", [1.1, 1.25, 1.35])
def test_time_stretch_shortens_without_changing_pitch(rate):
    audio = _tone(3.0)
    out = time_stretch(audio, SR, rate)
    assert len(out) == pytest.approx(len(audio) / rate, abs=1)
    assert _dominant_hz(out[SR // 2 : SR // 2 + SR]) == pytest.approx(150, abs=2)
    # No clicks: the overlapped frames line up, so no sample jumps further
    # than the original tone's own steepest step.
    assert np.abs(np.diff(out)).max() <= np.abs(np.diff(audio)).max() * 1.05


def test_time_stretch_leaves_rate_one_and_tiny_clips_alone():
    audio = _tone(1.0)
    assert len(time_stretch(audio, SR, 1.0)) == len(audio)
    tiny = _tone(0.01)
    assert len(time_stretch(tiny, SR, 1.3)) == len(tiny)


# -- Pipeline._catch_up --------------------------------------------------------


class _Playback:
    def __init__(self, pending=0.0):
        self.pending = pending

    def pending_s(self):
        return self.pending


def _pipeline(pending=0.0):
    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.playback = _Playback(pending)
    p._audio = queue.Queue()
    return p


def _queue(p, *seconds, seamless=False):
    for s in seconds:
        p._audio.put(_Clip(_tone(s), SR, 0.0, seamless=seamless))


def test_no_speed_up_while_playback_keeps_pace():
    p = _pipeline(pending=1.0)
    _queue(p, 2.0)
    clip = _Clip(_tone(2.0), SR, 1.0)
    assert p._catch_up(clip) is clip
    assert not p._catching_up


def test_speeds_up_when_behind_then_returns_to_normal_once_caught_up():
    cfg = PipelineConfig().audio
    p = _pipeline()
    _queue(p, 4.0, 4.0, 4.0)  # 12s waiting: past catchup_start_s
    clip = _Clip(_tone(3.0), SR, 1.0)

    faster = p._catch_up(clip)

    assert p._catching_up
    assert len(faster.audio) < len(clip.audio)
    assert faster.pause_before < clip.pause_before  # pauses shrink too
    rate = len(clip.audio) / len(faster.audio)
    assert cfg.catchup_min_rate - 0.01 <= rate <= cfg.catchup_max_rate + 0.01

    # Still behind, but under the start threshold: keeps catching up (hysteresis).
    p._audio = queue.Queue()
    _queue(p, 5.0)
    assert len(p._catch_up(_Clip(_tone(3.0), SR)).audio) < 3.0 * SR
    assert p._catching_up

    # Down to catchup_stop_s: normal speed again.
    p._audio = queue.Queue()
    _queue(p, 1.0)
    clip = _Clip(_tone(3.0), SR)
    assert p._catch_up(clip) is clip
    assert not p._catching_up


def test_speed_up_grows_with_the_backlog():
    small, large = _pipeline(), _pipeline()
    _queue(small, 9.0)
    _queue(large, 30.0)
    clip = _Clip(_tone(3.0), SR)
    assert len(large._catch_up(clip).audio) < len(small._catch_up(clip).audio)


def test_waiting_chunks_of_the_same_piece_are_stretched_together():
    p = _pipeline()
    _queue(p, 1.0, 1.0, seamless=True)  # the rest of the piece, already synthesized
    _queue(p, 8.0)  # the next sentence
    out = p._catch_up(_Clip(_tone(1.0), SR, 0.35))
    assert p._audio.qsize() == 1  # both chunks merged into this clip
    assert len(out.audio) < 3.0 * SR and len(out.audio) > 2.0 * SR


def test_skips_the_oldest_speech_when_hopelessly_behind():
    p = _pipeline()
    p.cfg.audio.max_backlog_s = 20.0
    _queue(p, 10.0, 10.0, 10.0)
    first = _Clip(_tone(10.0, freq=300.0), SR)
    out = p._catch_up(first)
    assert _dominant_hz(out.audio) != pytest.approx(300, abs=5)  # the oldest clip was skipped
    assert p._audio.qsize() == 2  # 20s still waiting: back within max_backlog_s


def test_test_output_clip_is_never_sped_up():
    p = _pipeline()
    _queue(p, 30.0)
    clip = _Clip(_tone(1.0), SR, force=True)
    assert p._catch_up(clip) is clip
