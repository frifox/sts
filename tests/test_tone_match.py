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


def _glide(start_hz, end_hz, seconds=2.0):
    """A voice-like tone holding `start_hz`, gliding to `end_hz` over its last half second."""
    n = int(SR * seconds)
    hz = np.full(n, float(start_hz))
    hz[-SR // 2:] = np.linspace(start_hz, end_hz, SR // 2)
    phase = 2 * np.pi * np.cumsum(hz) / SR
    return sum(np.sin(k * phase) / k for k in range(1, 6)).astype(np.float32) * 0.1


def test_a_falling_ending_is_told_from_a_level_or_rising_one():
    from sts.audioquality import final_pitch_st

    assert final_pitch_st(_glide(150, 100), SR) < -3
    assert final_pitch_st(_glide(150, 150), SR) > -1
    assert final_pitch_st(_glide(150, 190), SR) > 1
