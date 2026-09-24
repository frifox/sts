"""GigaAM backend: chunking of audio longer than the model accepts (no model loaded)."""

import numpy as np
import pytest

from sts.asr import TimedWord, create_asr
from sts.asr.gigaam_asr import MAX_CHUNK_S, MIN_CHUNK_S, SAMPLE_RATE, GigaAMASR, chunk_bounds
from sts.config import ASRConfig


def _speech_with_pauses(seconds: float, pause_every_s: float = 3.7) -> np.ndarray:
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.1).astype(np.float32)
    for t in np.arange(pause_every_s, seconds, pause_every_s):
        audio[int(t * SAMPLE_RATE) : int((t + 0.3) * SAMPLE_RATE)] *= 0.01
    return audio


def test_short_audio_is_one_chunk():
    assert chunk_bounds(np.zeros(10 * SAMPLE_RATE, np.float32)) == [0, 10 * SAMPLE_RATE]


def test_long_audio_is_cut_at_quiet_moments_within_the_limits():
    audio = _speech_with_pauses(70)
    bounds = chunk_bounds(audio)
    assert bounds[0] == 0 and bounds[-1] == len(audio)
    for start, end in zip(bounds, bounds[1:]):
        assert end - start <= MAX_CHUNK_S * SAMPLE_RATE
    for cut in bounds[1:-1]:
        assert audio[cut - 100 : cut + 100].std() < 0.01  # in a pause, not mid-speech
        assert cut - bounds[bounds.index(cut) - 1] >= MIN_CHUNK_S * SAMPLE_RATE


def test_cuts_already_made_never_move_as_the_buffer_grows():
    """What keeps older text from changing: a finished chunk is the same
    audio on every re-run, so it transcribes (and caches) identically."""
    audio = _speech_with_pauses(90)
    earlier = chunk_bounds(audio[: 50 * SAMPLE_RATE])[:-1]
    later = chunk_bounds(audio)
    assert later[: len(earlier)] == earlier


def test_finished_chunks_are_transcribed_once():
    asr = object.__new__(GigaAMASR)
    from collections import OrderedDict

    asr._cache = OrderedDict()
    calls = []
    asr._chunk_words = lambda chunk: calls.append(len(chunk)) or [TimedWord(f"part{len(calls)}.", 0.1, 0.5)]
    audio = _speech_with_pauses(50)

    asr.transcribe_words(audio[: 45 * SAMPLE_RATE])
    first = len(calls)
    words = asr.transcribe_words(audio)
    assert len(calls) == first + 1  # only the still-open last chunk again
    # Each chunk's word times are shifted to where that chunk starts in the audio.
    starts = chunk_bounds(audio)[:-1]
    assert [w.start for w in words] == pytest.approx([b / SAMPLE_RATE + 0.1 for b in starts])


def test_create_asr_builds_gigaam(monkeypatch):
    monkeypatch.setattr("sts.asr.gigaam_asr.GigaAMASR", lambda cfg: "gigaam-instance")
    assert create_asr(ASRConfig()) == "gigaam-instance"


def test_create_asr_missing_dependency_gives_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_gigaam(name, *args, **kwargs):
        if name == "sts.asr.gigaam_asr":
            raise ImportError("No module named 'hydra'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_gigaam)
    with pytest.raises(SystemExit, match=r"pip install -e \."):
        create_asr(ASRConfig())


def test_a_cut_goes_in_a_real_pause_not_a_brief_dip_inside_a_word():
    """A stop consonant's closure is the quietest 20 ms around, but it's
    inside a word ("в Н|ём"); the cut belongs in the actual pause."""
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(30 * SAMPLE_RATE) * 0.1).astype(np.float32)
    dip = int(16.0 * SAMPLE_RATE)
    audio[dip : dip + SAMPLE_RATE // 50] = 0.0  # 20 ms of total silence mid-word
    pause = int(19.0 * SAMPLE_RATE)
    audio[pause : pause + int(0.4 * SAMPLE_RATE)] *= 0.02  # a real, if not perfectly silent, pause
    cut = chunk_bounds(audio)[1]
    assert pause <= cut <= pause + int(0.4 * SAMPLE_RATE)
