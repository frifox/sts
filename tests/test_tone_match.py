"""Tone matching evens out a take's tonal balance toward the voice's average."""

import numpy as np
from scipy.signal import butter, sosfilt

from sts.audioquality import match_tone, tone_profile

SR = 24_000


def _speech_like(seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(SR * 3).astype(np.float32)
    envelope = np.repeat(rng.uniform(0.1, 1.0, 30), SR // 10).astype(np.float32)  # louder and quieter stretches
    return noise * envelope * 0.1


def test_a_duller_take_is_brought_back_toward_the_voice_s_tone():
    reference = tone_profile(_speech_like(0), SR)
    dull = sosfilt(butter(2, 2000, fs=SR, output="sos"), _speech_like(1)).astype(np.float32)  # "further away"
    before = tone_profile(dull, SR)
    after = tone_profile(match_tone(dull, SR, before, reference), SR)
    assert np.abs(after - reference).mean() < 0.6 * np.abs(before - reference).mean()


def test_a_take_already_matching_is_left_alone():
    audio = _speech_like(2)
    profile = tone_profile(audio, SR)
    assert np.allclose(match_tone(audio, SR, profile, profile), audio, atol=1e-4)


def test_too_short_to_judge():
    assert tone_profile(np.zeros(1000, dtype=np.float32), SR) is None



def test_silence_is_trimmed_from_the_ends_and_pauses_measured():
    from sts.audioquality import longest_pause_s, trim_silence

    audio = np.zeros(SR * 6, dtype=np.float32)
    audio[SR: 2 * SR] = _speech_like(3)[:SR] + 0.05
    audio[int(3.5 * SR): 4 * SR] = _speech_like(4)[: SR // 2] + 0.05
    assert abs(len(trim_silence(audio, SR)) / SR - 3.3) < 0.05  # 1 s .. 4 s, plus 0.15 s each side
    assert abs(longest_pause_s(audio, SR) - 1.5) < 0.05
