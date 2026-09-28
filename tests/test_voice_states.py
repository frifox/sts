"""The translated sentences' voice-state indicators (Pipeline._set_voice_state and co.)."""

import queue

import numpy as np

from sts.pipeline import Pipeline, _Clip


class _Broadcaster:
    def __init__(self):
        self.states = []

    def broadcast_voice_state(self, seq, state):
        self.states.append((seq, state))


def _pipeline():
    p = object.__new__(Pipeline)
    p._audio = queue.Queue()
    p.broadcaster = _Broadcaster()
    return p


def _clip(seq):
    return _Clip(np.zeros(10, dtype=np.float32), 24_000, seq=seq)


def test_a_sentence_goes_queued_generating_generated_playing_played():
    p = _pipeline()
    p._set_voice_state(1, "queued")
    p._set_voice_state(1, "generating")
    p._audio.put(_clip(1))
    p._voice_generation_done(1)
    clip = p._audio.get()
    p._voice_playing(clip.seq)
    p._voice_clip_done(clip.seq)
    assert [state for _, state in p.broadcaster.states] == ["queued", "generating", "generated", "playing", "played"]


def test_a_streamed_sentence_plays_while_generating_and_is_played_once_both_are_done():
    p = _pipeline()
    p._set_voice_state(2, "generating")
    p._audio.put(_clip(2))
    p._voice_playing(p._audio.get().seq)  # the first chunk plays while the rest is generated
    p._voice_clip_done(2)
    assert p.broadcaster.states[-1] == (2, "playing")  # not done generating yet
    p._voice_generation_done(2)
    assert p.broadcaster.states[-1] == (2, "played")


def test_a_later_sentence_starting_marks_earlier_ones_played():
    p = _pipeline()
    for seq in (3, 4):
        p._set_voice_state(seq, "queued")
    p._set_voice_state(3, "playing")
    p._voice_playing(4)
    assert (3, "played") in p.broadcaster.states
    assert p.broadcaster.states[-1] == (4, "playing")


def test_voice_off_clears_what_was_still_to_be_spoken():
    p = _pipeline()
    p._set_voice_state(5, "queued")
    p._set_voice_state(6, "generating")
    p._clear_voice_states()
    assert p.broadcaster.states[-2:] == [(5, None), (6, None)]
    assert p._voice_states == {}


def test_the_generating_estimate_comes_from_recent_speed_and_pace():
    import pytest

    p = _pipeline()
    assert p._voice_generation_estimate("Hello there.") is None  # nothing voiced yet
    p._voice_rtf, p._voice_pace = 0.41, 0.07  # 0.41x real time, 70 ms of speech per character
    assert p._voice_generation_estimate("x" * 100) == pytest.approx(0.41 * 0.07 * 100)
