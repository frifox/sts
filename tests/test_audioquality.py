import numpy as np

from sts.audioquality import clarity_db


def speechlike(bright: bool, seconds: float = 3.0, sr: int = 24000) -> np.ndarray:
    """Voiced-speech stand-in: a 120 Hz buzz with harmonics; the 'bright' one keeps its harmonics above 3 kHz."""
    t = np.arange(int(seconds * sr)) / sr
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)                    # syllable-like loudness changes
    top = 60 if bright else 14                                          # harmonic count: 7.2 kHz vs 1.7 kHz reach
    signal = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, top))
    return (0.1 * envelope * signal).astype(np.float32)


def test_muffled_audio_scores_lower_than_full_range_audio():
    assert clarity_db(speechlike(bright=False), 24000) < clarity_db(speechlike(bright=True), 24000) - 10


def test_sample_rate_does_not_change_the_verdict_and_tiny_clips_are_neutral():
    clip16 = speechlike(bright=True, sr=16000)
    clip24 = speechlike(bright=True, sr=24000)
    assert abs(clarity_db(clip16, 16000) - clarity_db(clip24, 24000)) < 2.0
    assert clarity_db(np.zeros(500, dtype=np.float32), 16000) == 0.0
    assert clarity_db(np.zeros(16000, dtype=np.float32), 16000) == 0.0    # digital silence: nothing to judge


def test_short_phrases_are_brought_up_to_the_level_of_long_ones():
    from sts.audioquality import match_level, speech_level

    rng = np.random.default_rng(0)

    def clip(level: float, seconds: float, sr: int = 24000) -> np.ndarray:
        t = np.arange(int(seconds * sr)) / sr
        return (level * np.sqrt(2) * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)

    quiet_short, loud_long = clip(0.04, 1.2), clip(0.12, 6.0)
    matched = [match_level(c, 24000, 0.10) for c in (quiet_short, loud_long)]
    levels = [speech_level(m, 24000) for m in matched]
    assert abs(levels[0] - levels[1]) < 0.02                       # now about the same loudness
    assert all(abs(lv - 0.10) < 0.03 for lv in levels)

    peaky = np.zeros(24000, dtype=np.float32); peaky[100] = 0.9; peaky[5000:5200] = 0.02
    assert np.abs(match_level(peaky, 24000, 0.10)).max() <= 0.95 + 1e-6      # never clips
    silence = np.zeros(24000, dtype=np.float32)
    assert np.array_equal(match_level(silence, 24000, 0.10), silence)         # nothing to scale
    hiss = (0.0002 * rng.standard_normal(24000)).astype(np.float32)
    assert np.abs(match_level(hiss, 24000, 0.10)).max() < 0.01                # no runaway gain on pure noise (cap: +12 dB)
