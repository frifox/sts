"""sts.transcript.Transcript as the live pipeline drives it: from two
threads, through long silences, and flushed on Stop. (Sentence-building
and locking rules are covered in test_caption.py.)"""

import threading

import numpy as np
import pytest

from sts.asr import TimedWord
from sts.transcript import Transcript

SR = 16_000


class _Asr:
    """Hears the words of `script` (absolute times) that fall inside the
    audio it's given; the fake samples carry their own absolute index."""

    def __init__(self, script=(), during=None):
        self.script = list(script)
        self.during = during  # called mid-transcription, e.g. to add audio meanwhile
        self.heard = []

    def transcribe_words(self, audio):
        begin, end = audio[0] / SR, (audio[-1] + 1) / SR
        self.heard.append((begin, end))
        if self.during:
            self.during()
        return [TimedWord(text, s - begin, e - begin) for text, s, e in self.script if s >= begin and e <= end]


def _samples(from_s, to_s):
    return np.arange(int(from_s * SR), int(to_s * SR), dtype=np.float32)


def test_silence_does_not_pile_up_in_the_buffer():
    """During a long musical number nothing is said, so nothing locks - the
    audio behind it must still be let go, or every step re-transcribes it all."""
    asr = _Asr()
    t = Transcript(asr)
    for second in range(120):
        t.add(_samples(second, second + 1))
        t.step()
    begin, end = asr.heard[-1]
    assert end == pytest.approx(120)
    assert end - begin <= 1 + Transcript.CONTEXT_S + 1e-6  # just the newest audio, plus context


def test_open_speech_is_kept_from_its_first_word():
    asr = _Asr([("Братья", 30.0, 30.5), ("и", 30.6, 30.7)])
    t = Transcript(asr)
    t.add(_samples(0, 31))
    assert t.step()[1] == ["Братья и"]
    t.add(_samples(31, 32))
    t.step()
    assert asr.heard[-1][0] == pytest.approx(30.0 - Transcript.CONTEXT_S)


def test_audio_arriving_during_a_step_is_kept_for_the_next():
    t = Transcript(None)
    t.asr = _Asr(during=lambda: t.add(_samples(2, 3)))  # capture keeps going while this transcribes
    t.add(_samples(0, 2))
    t.step()
    t.asr.during = None
    t.step()
    # The second step heard through 3 s - nothing lost - from CONTEXT_S
    # before where the first one ended.
    assert t.asr.heard[-1] == pytest.approx((2.0 - Transcript.CONTEXT_S, 3.0))


def test_a_step_across_clear_returns_nothing_of_the_old_speaker():
    t = Transcript(None)
    t.asr = _Asr([("Старое.", 0.1, 0.9)], during=t.clear)
    t.add(_samples(0, 1))
    assert t.step() == ([], [])
    assert not t.has_audio


def test_flush_returns_what_is_still_open_as_final_and_empties_the_buffer():
    asr = _Asr([("Первое.", 0.1, 0.9), ("Второе", 1.0, 1.5)])
    t = Transcript(asr)
    t.add(_samples(0, 2))
    t.step()
    flushed = t.flush()
    assert [s.text for s in flushed] == ["Первое.", "Второе"]
    assert not t.has_audio
    assert t.flush() == []


def test_time_rules_can_be_turned_off():
    asr = _Asr([("Второе", 1.0, 1.5)])
    t = Transcript(asr, after_unchanged=0, force_after=0, after_silence=0)
    t.add(_samples(0, 60))
    t.step()
    t.add(_samples(60, 120))
    assert t.step() == ([], ["Второе"])
    assert t.finalize_in_s is None


def test_add_and_step_from_two_threads():
    t = Transcript(_Asr())
    stop = threading.Event()

    def capture():
        second = 0
        while not stop.is_set():
            t.add(_samples(second * 0.032, (second + 1) * 0.032))
            second += 1

    thread = threading.Thread(target=capture)
    thread.start()
    try:
        for _ in range(200):
            t.step()
    finally:
        stop.set()
        thread.join()
    t.step()
    samples = np.concatenate(t._frames)
    assert np.all(np.diff(samples) == 1)  # contiguous: no frame lost or reordered


def test_finalize_in_counts_down_to_the_first_open_sentence_being_final():
    asr = _Asr([("Второе.", 1.0, 2.0)])
    t = Transcript(asr)
    assert t.finalize_in_s is None
    t.add(_samples(0, 3))
    t.step()  # "Второе" first seen at 3 s
    assert t.finalize_in_s == pytest.approx(5.0)
    t.add(_samples(3, 6))
    t.step()
    assert t.finalize_in_s == pytest.approx(2.0)
    t.add(_samples(6, 8))
    assert [s.text for s in t.step()[0]] == ["Второе."]
    assert t.finalize_in_s is None


def test_finalize_timer_follows_the_rule_that_finalizes_soonest():
    # Speaking, no end mark yet: only the 30 s rule runs (a pause shorter than
    # SILENCE_GRACE_S doesn't count as silence).
    asr = _Asr([("Братья", 1.0, 1.5), ("и", 2.6, 2.8)])
    t = Transcript(asr)
    t.add(_samples(0, 3))
    t.step()
    assert t.finalize_timer == pytest.approx((28.0, 30.0))
    # The speaker stopped: 1 s later the silence rule runs, shown from its start.
    t.add(_samples(3, 3.8))
    t.step()
    assert t.finalize_timer == pytest.approx((9.0, 9.0))
    # An end mark: the 5 s rule starts, and is the soonest.
    asr.script = [("Братья", 1.0, 1.5), ("и", 2.6, 2.8), ("сёстры.", 2.9, 3.3)]
    t.add(_samples(3.8, 4.0))
    t.step()
    assert t.finalize_timer == pytest.approx((5.0, 5.0))
