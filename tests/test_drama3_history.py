"""Drama 3 gets the sentences just spoken in full as history - none cut off,
none from another voice."""

from collections import deque

import numpy as np

from sts.tts import drama3
from sts.tts.drama3 import Drama3Synthesizer


class FakeModel:
    sample_rate = 100
    frames_per_second = 10.0

    def __init__(self):
        self.histories = []
        self.last_turns = []

    def stream(self, text, ref_audio, ref_text, interval_s, max_tokens=0, history=(), **sampling):
        self.histories.append([t for t, _ in history])
        self.last_turns = [(text, "codes")]
        yield np.zeros(10, np.float32)


def synthesizer():
    tts = object.__new__(Drama3Synthesizer)
    tts.model = FakeModel()
    tts._reference = ("ref.wav", "ref text")
    tts._history = deque(maxlen=drama3.HISTORY_SENTENCES)
    tts._sampling = lambda text: {"max_tokens": 100}
    return tts


def test_history_is_the_last_sentences_spoken_whole():
    tts = synthesizer()
    for text in ("one", "two", "three"):
        list(tts._stream_generate(text))
        tts._spoken_whole(text)
    list(tts._stream_generate("cut off"))  # not spoken whole: not kept
    list(tts._stream_generate("four"))
    assert tts.model.histories == [[], ["one"], ["one", "two"], ["two", "three"], ["two", "three"]]


def test_a_new_voice_starts_without_history(monkeypatch):
    tts = synthesizer()
    tts._pace_history, tts._clarity_history, tts._tone_history = deque(), deque(), deque()
    tts.cfg = type("Cfg", (), {"drama3_reference_speed": 1.0})()
    monkeypatch.setattr(tts, "_paced", lambda audio, speed: audio)
    list(tts._stream_generate("one"))
    tts._spoken_whole("one")
    tts._use(("other.wav", "other text"))
    list(tts._stream_generate("two"))
    assert tts.model.histories[-1] == []
