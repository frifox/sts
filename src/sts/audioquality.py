"""A cheap measure of how clear (close, full-range) a piece of speech sounds.

Voice cloning copies the recording conditions of its reference along with the
voice: cloned from a distant, heavily compressed recording, the output sounds
far away and low-bitrate, and even from a good reference some generations come
out duller than others. Speech from a muffled recording has almost nothing above
3 kHz, so the ratio of energy in 3-7 kHz to energy in 0.3-3 kHz (over the
loudest half of the frames) separates them well.
"""

from math import gcd

import numpy as np
from scipy.signal import resample_poly, stft

_RATE = 16_000


def clarity_db(audio: np.ndarray, sample_rate: int) -> float:
    """Higher = clearer. Typical values measured with this app's voices: about -14 to -16 dB for a clear
    take, -20 dB or lower for a muffled one; muffled or distant recordings sit around -25 dB."""
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != _RATE:
        divisor = gcd(sample_rate, _RATE)
        audio = resample_poly(audio, _RATE // divisor, sample_rate // divisor).astype(np.float32)
    if len(audio) < 2048:
        return 0.0  # too short to judge
    freqs, _, spectrum = stft(audio, fs=_RATE, nperseg=1024, noverlap=768, boundary=None, padded=False)
    power = np.abs(spectrum) ** 2
    low = power[(freqs >= 300) & (freqs < 3000)]
    high = power[(freqs >= 3000) & (freqs < 7000)]
    speech = low.sum(axis=0) > np.percentile(low.sum(axis=0), 50)
    if not speech.any():
        return 0.0
    return float(10 * np.log10(high[:, speech].sum() / (low[:, speech].sum() + 1e-12) + 1e-12))


def speech_level(audio: np.ndarray, sample_rate: int) -> float:
    """RMS of the louder half of the 100 ms frames: the level of the speech itself, ignoring pauses."""
    audio = np.asarray(audio, dtype=np.float32)
    frame = max(1, sample_rate // 10)
    if len(audio) < frame:
        return float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    energy = np.array([np.mean(audio[i: i + frame] ** 2) for i in range(0, len(audio) - frame + 1, frame)])
    return float(np.sqrt(np.mean(energy[energy >= np.median(energy)])))


def match_level(audio: np.ndarray, sample_rate: int, target: float, max_gain_db: float = 12.0, peak: float = 0.95) -> np.ndarray:
    """Scales a clip so its speech sits at `target` RMS, so sentences of different lengths play at the
    same loudness. The model speaks very short phrases about 6 dB quieter than long sentences, which
    sounds like someone far away. The gain is capped (so noise isn't boosted) and never pushes a peak past `peak`."""
    audio = np.asarray(audio, dtype=np.float32)
    level = speech_level(audio, sample_rate)
    top = float(np.abs(audio).max()) if len(audio) else 0.0
    if level < 1e-4 or top == 0.0:
        return audio
    gain = min(target / level, 10 ** (max_gain_db / 20), peak / top)
    return (audio * gain).astype(np.float32)
