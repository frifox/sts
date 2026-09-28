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
from scipy.signal import istft, resample_poly, stft

_RATE = 16_000
# The most match_level() will boost a clip, so noise isn't boosted.
MAX_LEVEL_GAIN_DB = 12.0


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


def level_gain(audio: np.ndarray, sample_rate: int, target: float, max_gain_db: float = MAX_LEVEL_GAIN_DB, peak: float = 0.95) -> float:
    """The gain match_level() would apply to `audio` - see there. 1.0 for silence."""
    audio = np.asarray(audio, dtype=np.float32)
    level = speech_level(audio, sample_rate)
    top = float(np.abs(audio).max()) if len(audio) else 0.0
    if level < 1e-4 or top == 0.0:
        return 1.0
    return min(target / level, 10 ** (max_gain_db / 20), peak / top)


def match_level(audio: np.ndarray, sample_rate: int, target: float, max_gain_db: float = MAX_LEVEL_GAIN_DB, peak: float = 0.95) -> np.ndarray:
    """Scales a clip so its speech sits at `target` RMS, so sentences of different lengths play at the
    same loudness. The model speaks very short phrases about 6 dB quieter than long sentences, which
    sounds like someone far away. The gain is capped (so noise isn't boosted) and never pushes a peak past `peak`."""
    audio = np.asarray(audio, dtype=np.float32)
    gain = level_gain(audio, sample_rate, target, max_gain_db, peak)
    return audio if gain == 1.0 else (audio * gain).astype(np.float32)


# Tone matching: Qwen3 takes of the same voice come out with slightly different
# tonal balance - one boomier and duller (close to the mic, speaking softly),
# the next thinner and brighter (further away, speaking up) - even at the same
# loudness. Each take's long-term spectrum, over the speech frames and with its
# level taken out, is compared with the voice's recent average, and the take is
# gently equalized toward it. Measured on 16-sentence runs, this halved the
# take-to-take spread (about 2.2 -> 1.3 dB average deviation from the voice's
# typical spectrum). Only partially and within limits: part of a sentence's
# spectrum comes from its own vowels and consonants, not the "recording".
_TONE_EDGES = np.geomspace(80, 10_000, 19)  # 18 bands, ~0.4 octave each
_TONE_FFT = 1024


def tone_profile(audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """The shape (dB per band, mean removed) of the speech's long-term spectrum, or None if too short."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < 4 * _TONE_FFT or sample_rate < 2 * _TONE_EDGES[-1]:
        return None
    freqs, _, spectrum = stft(audio, fs=sample_rate, nperseg=_TONE_FFT, noverlap=_TONE_FFT * 3 // 4)
    power = np.abs(spectrum) ** 2
    total = power.sum(axis=0)
    speech = total > np.percentile(total, 50)
    if not speech.any():
        return None
    ltas = power[:, speech].mean(axis=1)
    bands = np.array([ltas[(freqs >= lo) & (freqs < hi)].mean() for lo, hi in zip(_TONE_EDGES[:-1], _TONE_EDGES[1:])])
    db = 10 * np.log10(bands + 1e-20)
    return db - db.mean()


def match_tone(audio: np.ndarray, sample_rate: int, profile: np.ndarray, reference: np.ndarray,
               strength: float = 0.75, max_db: float = 6.0) -> np.ndarray:
    """`audio` (whose tone_profile is `profile`) equalized `strength` of the way toward `reference`,
    no band moved more than `max_db`. The overall level is then off by a little; match_level after."""
    gains = np.clip(strength * (np.asarray(reference) - np.asarray(profile)), -max_db, max_db)
    gains = np.convolve(np.pad(gains, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")  # no sharp steps
    freqs, _, spectrum = stft(audio, fs=sample_rate, nperseg=_TONE_FFT, noverlap=_TONE_FFT * 3 // 4)
    centers = np.sqrt(_TONE_EDGES[:-1] * _TONE_EDGES[1:])
    per_bin = np.interp(np.log(np.maximum(freqs, 1.0)), np.log(centers), gains)  # flat beyond the end bands
    _, out = istft(spectrum * (10 ** (per_bin / 20))[:, None], fs=sample_rate, nperseg=_TONE_FFT, noverlap=_TONE_FFT * 3 // 4)
    return out[: len(audio)].astype(np.float32)


def final_pitch_st(audio: np.ndarray, sample_rate: int, tail_s: float = 0.35) -> float | None:
    """Where the voice's pitch ends up: the average over its last `tail_s` of voiced speech, in semitones
    relative to the whole clip's median pitch, or None if too little of it is voiced to tell. A statement
    normally ends well below its median (-3 st on average in Qwen3's takes); one that ends near or above it
    sounds unfinished, as if a comma came next."""
    import librosa

    hop = 512
    f0, voiced, _ = librosa.pyin(np.asarray(audio, dtype=np.float32), fmin=60, fmax=400, sr=sample_rate,
                                 frame_length=2048, hop_length=hop)
    f0 = f0[voiced & ~np.isnan(f0)]
    tail = max(3, int(tail_s * sample_rate / hop))
    if len(f0) < 3 * tail:
        return None
    semitones = 12 * np.log2(f0 / np.median(f0))
    return float(semitones[-tail:].mean())
