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
