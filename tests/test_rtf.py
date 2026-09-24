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


def test_asr_loop_tracks_audio_duration_and_broadcasts_rtf():
    p = _rtf_pipeline()
    p.bible_lookup = None  # "Привет." has no bible reference, so this is never touched
    p.cfg = PipelineConfig()
    p._asr_lock = threading.Lock()
    p._segments = queue.Queue()
    p._stop = threading.Event()
    p._last_asr_audio_s = 0.0
    p._committed_text = ""
    p._pending_confirmation = None
    p._original_seq = 0
    p._last_sentence_original = None
    p._transcripts = queue.Queue()

    class FakeAsr:
        def transcribe(self, audio):
            time.sleep(0.05)
            return "Привет."

    p.asr = FakeAsr()
    segment = np.zeros(p.cfg.audio.sample_rate * 2, dtype=np.float32)  # 2s of audio
    p._segments.put(("final", segment))

    thread = threading.Thread(target=Pipeline._asr_loop, args=(p,), daemon=True)
    thread.start()
    thread.join(timeout=0.5)  # loop blocks forever on the next get() - daemon thread is fine to leave running

    assert p._last_asr_audio_s == 2.0
    assert len(p.broadcaster.rtf) == 1
    stage, value = p.broadcaster.rtf[0]
    assert stage == "asr"
    assert value == pytest.approx(0.025, abs=0.05)  # ~0.05s processing / 2s audio
    assert p._transcripts.get_nowait() == ("Привет.", 0, None)


def test_translate_segment_uses_the_last_asr_audio_duration():
    p = _rtf_pipeline()
    p._translations = queue.Queue()
    p._last_asr_audio_s = 4.0

    class FakeTranslator:
        def translate_sentences(self, text, context=None):
            yield "Translated."

    p.translator = FakeTranslator()

    Pipeline._translate_segment(p, "Text.")

    assert len(p.broadcaster.rtf) == 1
    stage, value = p.broadcaster.rtf[0]
    assert stage == "translate"
    assert 0 <= value < 1.0  # near-instant translation over 4s of source audio


def test_translate_segment_skips_rtf_before_any_segment_is_transcribed():
    p = _rtf_pipeline()
    p._translations = queue.Queue()
    p._last_asr_audio_s = 0.0  # nothing transcribed yet this session

    class FakeTranslator:
        def translate_sentences(self, text, context=None):
            yield "Translated."

    p.translator = FakeTranslator()

    Pipeline._translate_segment(p, "Text.")

    assert p.broadcaster.rtf == []


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
    p._translations.put("Hello.")

    thread = threading.Thread(target=Pipeline._tts_loop, args=(p,), daemon=True)
    thread.start()
    thread.join(timeout=0.5)

    assert len(p.broadcaster.rtf) == 1
    stage, value = p.broadcaster.rtf[0]
    assert stage == "voice"
    assert value == pytest.approx(0.05, abs=0.05)  # ~0.05s processing / 1s of synthesized audio
    assert p._audio.get_nowait()[0].size == 24_000
