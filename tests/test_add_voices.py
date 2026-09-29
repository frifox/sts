"""sts-add-voices: which part of a recording becomes the voice's reference."""

from pathlib import Path

import numpy as np

from sts.add_voices import SAMPLE_RATE, pick_clip, voice_name


def _audio(seconds, loud=()):
    audio = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    rng = np.random.default_rng(0)
    for start, end, level in loud:
        first, last = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        audio[first:last] = rng.standard_normal(last - first) * level
    return audio


def test_the_longest_continuous_speech_is_picked_and_the_intro_skipped():
    # 10 min recording: speech in the intro, then choppy speech, then a steady passage
    segments = [(5, 25), (65, 70), (72, 73), (80, 84), (200, 204), (204.5, 210), (210.4, 216), (216.8, 222)]
    audio = _audio(600, [(s, e, 0.1) for s, e in segments])
    clip = pick_clip(segments, audio, target_s=18)
    assert clip.start == 200 and clip.speech_s >= 18 * 0.9


def test_a_short_file_keeps_its_start_and_a_long_segment_is_cut():
    segments = [(1, 40)]  # one 39 s stretch of speech, no pauses
    audio = _audio(45, [(1, 40, 0.1)])
    clip = pick_clip(segments, audio, target_s=18)
    assert clip.start == 1 and 18 <= clip.seconds <= 18 * 1.4 + 0.1


def test_too_little_speech_gives_nothing():
    segments = [(1, 3), (10, 12)]
    assert pick_clip(segments, _audio(20, [(1, 3, 0.1), (10, 12, 0.1)]), target_s=18) is None


def test_the_voice_is_named_after_the_file():
    assert voice_name(Path("Neyman_Konstantin.mp4")) == "Neyman Konstantin"
    assert voice_name(Path("Pastor Ivan  2019.mov")) == "Pastor Ivan 2019"
    assert voice_name(Path("goldinov-venya.mp3")) == "Goldinov Venya"
    assert voice_name(Path("McArthur_John.mp4")) == "McArthur John"
