"""Speeding speech playback up when it falls behind, and back down once caught up."""

import queue
import time

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


def _sentence(seconds, late_s, freq=150.0):
    """The first clip of a sentence whose caption became final `late_s` ago."""
    return _Clip(_tone(seconds, freq), SR, 0.35, sentence_start=True, finalized_at=time.monotonic() - late_s)


def _rate(original, played):
    return len(original.audio) / len(played.audio)


def test_a_long_sentence_waiting_its_turn_is_not_lateness():
    """What went wrong before: one long sentence queued behind the one
    playing counted as "8 s behind" and sped the voice up, though nothing
    was running late."""
    p = _pipeline(pending=6.0)
    _queue(p, 15.0)  # a long sentence already synthesized, waiting
    clip = _sentence(3.0, late_s=1.5)
    assert p._catch_up(clip) is clip
    assert not p._catching_up


def test_speeds_up_while_sentences_start_late_then_returns_to_normal():
    cfg = PipelineConfig().audio
    p = _pipeline()

    late = _sentence(3.0, late_s=10.0)
    faster = p._catch_up(late)
    assert p._catching_up
    assert cfg.catchup_min_rate - 0.01 <= _rate(late, faster) <= cfg.catchup_max_rate + 0.01
    assert faster.pause_before < late.pause_before  # pauses shrink too

    # The rest of the same sentence keeps the speed chosen at its start,
    # however long it has been since the sentence became final by now.
    rest = _Clip(_tone(3.0), SR, 0.35, finalized_at=late.finalized_at)
    assert _rate(rest, p._catch_up(rest)) == pytest.approx(_rate(late, faster), rel=0.01)

    # Next sentence 5 s late: under the start threshold, over the stop one - keeps going (hysteresis).
    assert _rate(late, p._catch_up(_sentence(3.0, late_s=5.0))) > 1.0

    # One that starts within catchup_stop_s: normal speed again, for it and the rest of it.
    on_time = _sentence(3.0, late_s=1.0)
    assert p._catch_up(on_time) is on_time
    assert not p._catching_up
    rest = _Clip(_tone(3.0), SR, 0.35)
    assert p._catch_up(rest) is rest


def test_speed_up_grows_with_how_late_a_sentence_starts():
    a, b = _pipeline(), _pipeline()
    slightly = a._catch_up(_sentence(3.0, late_s=9.0))
    very = b._catch_up(_sentence(3.0, late_s=30.0))
    assert len(very.audio) < len(slightly.audio)


def test_waiting_chunks_of_the_same_piece_are_stretched_together():
    p = _pipeline()
    _queue(p, 1.0, 1.0, seamless=True)  # the rest of the piece, already synthesized
    _queue(p, 8.0)  # the next sentence
    out = p._catch_up(_sentence(1.0, late_s=12.0))
    assert p._audio.qsize() == 1  # both chunks merged into this clip
    assert 2.0 * SR < len(out.audio) < 3.0 * SR


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
