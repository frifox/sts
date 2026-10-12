"""Streamed Qwen3 speech: held back STREAM_HOLD_S, checked if whole by then,
streamed after, cut off with a fade if it rambles."""

import sys
import time
import types

import numpy as np
import pytest

SR = 24_000


@pytest.fixture
def qwen3(monkeypatch):
    if "qwen_tts" not in sys.modules:
        stub = types.ModuleType("qwen_tts")
        stub.Qwen3TTSModel = object
        monkeypatch.setitem(sys.modules, "qwen_tts", stub)
    import sts.tts.qwen3 as q

    return q


class _FakeModel:
    """Generates `seconds` of a tone, a piece every `every_s`, `piece_s` of audio each."""

    sample_rate = SR

    def __init__(self, seconds_per_take, piece_s=0.32, every_s=0.0):
        self.takes = list(seconds_per_take)
        self.piece_s, self.every_s = piece_s, every_s
        self.closed = 0

    def stream_voice_clone(self, text, prompt, interval_s, language="English", **sampling):
        seconds = self.takes.pop(0)
        n = int(self.piece_s * SR)
        made = 0
        try:
            while made < seconds * SR:
                time.sleep(self.every_s)
                piece = (0.1 * np.sin(np.arange(made, made + n) * 0.05)).astype(np.float32)
                made += n
                yield piece
        finally:
            self.closed += 1


def _synth(q, model, hold_s=0.2):
    s = q.Qwen3TTSSynthesizer.__new__(q.Qwen3TTSSynthesizer)
    from collections import deque

    from sts.config import TTSConfig

    s.cfg = TTSConfig(backend="qwen3")
    s.cfg.qwen3_stream_hold_s = hold_s
    s.cfg.qwen3_tone_match = False
    s.cfg.qwen3_max_retries = 2
    s.engine, s.model, s._prompt, s._active_lora_speaker_id = "mlx", model, object(), None
    s._lock = __import__("threading").Lock()
    s._pace_history, s._clarity_history, s._tone_history = deque(maxlen=20), deque(maxlen=20), deque(maxlen=20)
    s._sampling = lambda text: {}
    s.model.stream_voice_clone = model.stream_voice_clone
    return s


def test_a_short_sentence_whole_within_the_hold_plays_as_one_clip(qwen3):
    model = _FakeModel([1.3])  # generated instantly: whole long before the hold ends
    pieces = list(_synth(qwen3, model).stream_chunks("Let us pray."))
    assert len(pieces) == 1 and pieces[0][2] is False


def test_a_long_sentence_streams_after_the_hold(qwen3):
    model = _FakeModel([6.0], every_s=0.03)  # ~10x faster than real time: past the 0.2 s hold mid-take
    pieces = list(_synth(qwen3, model).stream_chunks("At the beginning of my sermon I will read eight verses of John."))
    assert len(pieces) > 3
    assert pieces[0][2] is False and all(seamless for _, _, seamless in pieces[1:])
    total = sum(len(p) for p, _, _ in pieces) / SR
    assert 5.9 < total < 6.4  # all of it (plus the filter's few ms)


def test_a_ramble_caught_in_the_hold_is_regenerated_before_anything_plays(qwen3):
    model = _FakeModel([20.0, 1.0])  # first take babbles on; generated instantly, so caught while held
    pieces = list(_synth(qwen3, model).stream_chunks("Let us pray."))
    assert len(pieces) == 1 and abs(len(pieces[0][0]) / SR - 1.0) < 0.4  # the second, good take
    assert model.closed == 2


def test_a_ramble_while_playing_is_cut_off_with_a_fade(qwen3):
    model = _FakeModel([30.0], every_s=0.02)
    text = "These are the words of our Lord Jesus Christ."
    pieces = list(_synth(qwen3, model).stream_chunks(text))
    audio = np.concatenate([p for p, _, _ in pieces])
    expected = len(text) * qwen3.TYPICAL_SECONDS_PER_CHAR
    assert len(audio) / SR < expected * qwen3.RAMBLE_FACTOR + qwen3.RAMBLE_MARGIN_S + 0.5
    assert np.abs(audio[-50:]).max() < 0.01  # faded out, not chopped
    assert model.closed == 1  # generating stopped


def test_streamed_pieces_filter_seamlessly(qwen3):
    from sts.audioquality import tone_filter

    fir = tone_filter(np.linspace(-3, 3, 18), SR)
    audio = (np.random.default_rng(0).standard_normal(SR * 2) * 0.1).astype(np.float32)
    whole = qwen3._StreamPost(SR, 1.0, fir)
    one = np.concatenate([whole.process(audio), whole.finish()])
    pieced = qwen3._StreamPost(SR, 1.0, fir)
    parts = [pieced.process(chunk) for chunk in np.array_split(audio, 7)] + [pieced.finish()]
    assert np.allclose(np.concatenate(parts), one, atol=1e-6)
