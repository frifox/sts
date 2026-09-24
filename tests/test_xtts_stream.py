"""Level matching of XTTS's streamed chunks (the model itself isn't loaded)."""

import numpy as np
import pytest

pytest.importorskip("TTS")

from sts.audioquality import speech_level  # noqa: E402
from sts.config import TTSConfig  # noqa: E402
from sts.tts.xtts import LEVEL_WINDOW_S, SAMPLE_RATE, XTTSSynthesizer  # noqa: E402


def _synth(chunks):
    synth = object.__new__(XTTSSynthesizer)
    synth.cfg = TTSConfig()
    synth._generate_stream = lambda text: iter(chunks)
    return synth


def _speech(seconds, level):
    rng = np.random.default_rng(1)
    return (rng.standard_normal(int(seconds * SAMPLE_RATE)) * level).astype(np.float32)


def test_first_chunk_is_marked_as_a_new_piece_and_the_rest_as_continuing_it():
    out = list(_synth([_speech(0.9, 0.02), _speech(0.9, 0.02), _speech(0.5, 0.02)])._stream_piece("x"))
    assert [seamless for _, _, seamless in out] == [False, True, True]


def test_quiet_speech_is_brought_up_to_the_target_level_from_the_first_window():
    chunks = [_speech(0.9, 0.04) for _ in range(4)]
    out = list(_synth(chunks)._stream_piece("x"))
    audio = np.concatenate([a for a, _, _ in out])
    assert speech_level(audio, SAMPLE_RATE) == pytest.approx(TTSConfig().target_level, rel=0.15)


def test_a_piece_shorter_than_the_level_window_is_leveled_whole():
    short = [_speech(LEVEL_WINDOW_S / 4, 0.04), _speech(LEVEL_WINDOW_S / 4, 0.04)]
    out = list(_synth(short)._stream_piece("Amen."))
    assert len(out) == 1 and out[0][2] is False
    assert speech_level(out[0][0], SAMPLE_RATE) == pytest.approx(TTSConfig().target_level, rel=0.15)
