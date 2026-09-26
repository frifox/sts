"""Verifies the live real-time-factor (RTF) readings shown in the caption
page's header: processing time / audio duration for the ASR, translate, and
voice stages, so an operator can see whether each stage is keeping up with
the live speaker (below 1.0x) or falling behind (at/above 1.0x). Readings are
smoothed over a rolling window (see RTF_WINDOW_S) rather than shown raw, so
one slow/short phrase doesn't spike the display."""

import queue
import threading
import time

import numpy as np
import pytest

from sts.config import PipelineConfig
from sts.control import RecognitionControl
from sts.pipeline import Pipeline


class _Broadcaster:
    def __init__(self):
        self.rtf = []
        self.original = []

    def broadcast_rtf(self, stage, value):
        self.rtf.append((stage, value))

    def broadcast_original(self, text, seq=None, batch=None):
        self.original.append(text)

    def broadcast_translated(self, text, seq=None, batch=None):
        pass


def _rtf_pipeline():
    p = object.__new__(Pipeline)
    p.broadcaster = _Broadcaster()
    p._rtf_window = {}
    return p


def test_broadcast_rtf_sends_processing_over_audio_ratio():
    p = _rtf_pipeline()

    p._broadcast_rtf("asr", processing_s=1.0, audio_s=2.0)

    assert p.broadcaster.rtf == [("asr", 0.5)]


def test_broadcast_rtf_skips_when_audio_duration_is_not_positive():
    p = _rtf_pipeline()

    p._broadcast_rtf("translate", processing_s=1.0, audio_s=0.0)
    p._broadcast_rtf("translate", processing_s=1.0, audio_s=-1.0)

    assert p.broadcaster.rtf == []


def test_broadcast_rtf_is_a_noop_with_no_broadcaster():
    p = _rtf_pipeline()
    p.broadcaster = None

    p._broadcast_rtf("voice", processing_s=1.0, audio_s=1.0)  # must not raise


def test_broadcast_rtf_averages_readings_within_the_window(monkeypatch):
    """The windowed value is total processing time over total audio time
    across every reading still in the window - not a plain mean of the
    per-call ratios - so a big segment's reading isn't drowned out by a
    handful of tiny ones (or vice versa)."""
    p = _rtf_pipeline()
    clock = [0.0]
    monkeypatch.setattr("sts.pipeline.time.monotonic", lambda: clock[0])

    p._broadcast_rtf("asr", processing_s=1.0, audio_s=2.0)  # 0.5x alone
    clock[0] = 5.0
    p._broadcast_rtf("asr", processing_s=0.5, audio_s=1.0)  # 0.5x alone too - same combined ratio

    assert p.broadcaster.rtf[-1] == ("asr", pytest.approx(1.5 / 3.0))


def test_broadcast_rtf_drops_readings_older_than_the_window(monkeypatch):
    p = _rtf_pipeline()
    clock = [0.0]
    monkeypatch.setattr("sts.pipeline.time.monotonic", lambda: clock[0])

    p._broadcast_rtf("voice", processing_s=10.0, audio_s=1.0)  # a spike: 10x
    clock[0] = 31.0  # past RTF_WINDOW_S (30s) - the spike ages out
    p._broadcast_rtf("voice", processing_s=0.1, audio_s=1.0)

    assert p.broadcaster.rtf[-1] == ("voice", pytest.approx(0.1))  # the spike no longer counts


def test_broadcast_rtf_keeps_readings_still_within_the_window(monkeypatch):
    p = _rtf_pipeline()
    clock = [0.0]
    monkeypatch.setattr("sts.pipeline.time.monotonic", lambda: clock[0])

    p._broadcast_rtf("translate", processing_s=3.0, audio_s=1.0)  # a spike: 3x
    clock[0] = 29.0  # still inside the 30s window
    p._broadcast_rtf("translate", processing_s=0.1, audio_s=1.0)

    assert p.broadcaster.rtf[-1] == ("translate", pytest.approx(3.1 / 2.0))  # the spike still counts


def test_caption_step_reports_its_time_over_the_audio_that_arrived_meanwhile(monkeypatch):
    p = _rtf_pipeline()
    p.cfg = PipelineConfig()
    p.cfg.asr.interval_ms = 1
    p._asr_lock = threading.Lock()
    p._stop = threading.Event()
    published = []
    p._publish_captions = lambda final, still_open: published.append((final, still_open))

    class Transcript:
        has_audio = True
        finalize_timer = None

        def step(self):
            time.sleep(0.05)
            return [], ["Привет"]

    p.transcript = Transcript()
    p._finalize_timer = None
    p._last_caption_step = time.monotonic() - 0.5  # half a second of audio since the last step

    Pipeline._caption_step(p)

    stage, value = p.broadcaster.rtf[0]
    assert stage == "asr"
    assert value == pytest.approx(0.1, abs=0.05)  # ~0.05 s of work per ~0.5 s of audio
    assert published == [([], ["Привет"])]


def test_translation_rtf_is_against_how_long_the_sentence_took_to_say():
    p = _rtf_pipeline()
    p._translations = queue.Queue()

    class FakeTranslator:
        def translate_sentences(self, text):
            yield "Translated."

    p.translator = FakeTranslator()

    Pipeline._translate_sentence(p, "Text.", seq=0, spoken_s=4.0)
    Pipeline._translate_sentence(p, "Text.", seq=1, spoken_s=0.0)  # no duration: nothing to compare against

    assert len(p.broadcaster.rtf) == 1
    stage, value = p.broadcaster.rtf[0]
    assert stage == "translate"
    assert 0 <= value < 1.0  # near-instant translation over 4s of source audio


def test_tts_loop_broadcasts_voice_rtf_from_its_own_output_duration():
    p = _rtf_pipeline()
    p.cfg = PipelineConfig()
    p.control = RecognitionControl(16000)
    p.control.enable_voice()
    p._translations = queue.Queue()
    p._audio = queue.Queue()
    p._stop = threading.Event()

    class FakeTts:
        def synthesize(self, text):
            time.sleep(0.05)
            return np.zeros(24_000, dtype=np.float32), 24_000  # 1s of audio at 24kHz

    p.tts = FakeTts()
    p._translations.put(("Hello.", False, None))

    thread = threading.Thread(target=Pipeline._tts_loop, args=(p,), daemon=True)
    thread.start()
    thread.join(timeout=0.5)

    assert len(p.broadcaster.rtf) == 1
    stage, value = p.broadcaster.rtf[0]
    assert stage == "voice"
    assert value == pytest.approx(0.05, abs=0.05)  # ~0.05s processing / 1s of synthesized audio
    assert p._audio.get_nowait().audio.size == 24_000


def test_tts_loop_paces_a_continuation_like_the_same_spoken_thought():
    """The reported bug: a VAD split lands mid-sentence, so the continuation
    was still spoken with the long between-sentences pause and no lead-in -
    audibly introducing it as a fresh, unrelated utterance rather than
    picking up where the (already-spoken) previous sentence left off."""
    p = _rtf_pipeline()
    p.cfg = PipelineConfig()
    p.control = RecognitionControl(16000)
    p.control.enable_voice()
    p._translations = queue.Queue()
    p._audio = queue.Queue()
    p._stop = threading.Event()

    class FakeTts:
        def synthesize(self, text):
            return np.zeros(100, dtype=np.float32), 24_000

    p.tts = FakeTts()
    p._translations.put(("Continues the previous sentence.", True, None))
    p._translations.put(("A fresh, unrelated sentence.", False, None))

    thread = threading.Thread(target=Pipeline._tts_loop, args=(p,), daemon=True)
    thread.start()
    thread.join(timeout=0.5)

    pauses = []
    while not p._audio.empty():
        pauses.append(p._audio.get_nowait().pause_before)
    assert pauses == [p.cfg.audio.pause_within_s, p.cfg.audio.pause_between_s]
