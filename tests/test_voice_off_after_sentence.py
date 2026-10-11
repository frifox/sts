"""Voice off lets the sentence being spoken finish, then goes off."""

import queue

from sts.config import PipelineConfig
from sts.control import RecognitionControl
from sts.pipeline import Pipeline, _Clip


class _Broadcaster:
    def __init__(self):
        self.statuses = []

    def broadcast_status(self, **status):
        self.statuses.append(status)

    def broadcast_voice_state(self, seq, state):
        pass

    def broadcast_output_level(self, level):
        pass


def _pipeline():
    p = Pipeline.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.control = RecognitionControl(sample_rate=16_000)
    p.control.enable_voice()
    p.broadcaster = _Broadcaster()
    p.tts = object()
    p._audio = queue.Queue()
    p._pending_voice = None
    p._active_speaker_id = None
    return p


def _clip(seq):
    import numpy as np

    return _Clip(np.zeros(10, dtype="float32"), 24_000, 0.0, seq=seq)


def test_the_sentence_being_spoken_finishes_then_voice_goes_off():
    p = _pipeline()
    p._set_voice_state(5, "generating")
    p._voice_playing(5)  # sentence 5 is being spoken; more of it, and sentence 6, are queued
    p._audio.put(_clip(5))
    p._audio.put(_clip(6))
    p._execute_control({"action": "voice_off"})
    assert p.control.voice_enabled.is_set() and p.broadcaster.statuses[-1]["voice_stopping"]
    assert [clip.seq for clip in p._audio.queue] == [5]  # only the rest of sentence 5
    p._voice_generation_done(5)
    p._voice_clip_done(5)  # a clip of it ends, but more of it is still queued
    assert p.control.voice_enabled.is_set()
    p._audio.get()
    p._voice_playing(5)
    p._voice_clip_done(5)  # its last clip: spoken to the end
    assert not p.control.voice_enabled.is_set() and not p.broadcaster.statuses[-1]["voice_stopping"]


def test_nothing_playing_goes_off_at_once():
    p = _pipeline()
    p._execute_control({"action": "voice_off"})
    assert not p.control.voice_enabled.is_set()


def test_clicking_voice_again_while_it_waits_keeps_it_on():
    p = _pipeline()
    p._voice_playing(5)
    p._execute_control({"action": "voice_off"})
    p._execute_control({"action": "voice_on"})
    assert p.control.voice_enabled.is_set() and not p.broadcaster.statuses[-1]["voice_stopping"]
    p._voice_generation_done(5)
    p._voice_clip_done(5)
    assert p.control.voice_enabled.is_set()  # the sentence ending doesn't turn it off any more
