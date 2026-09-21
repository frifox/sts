import numpy as np

from sts.audio.vad import SpeechSegmenter
from sts.config import AudioConfig, VADConfig


def make_segmenter(vad_cfg):
    """A segmenter with scripted speech/silence decisions (no Silero model needed)."""
    seg = object.__new__(SpeechSegmenter)
    seg.audio_cfg = AudioConfig()
    seg.vad_cfg = vad_cfg
    seg._buffer, seg._silence_ms, seg._speech_ms = [], 0.0, 0.0
    seg._script = []
    seg._is_speech = lambda frame: seg._script.pop(0)
    return seg


def feed(seg, pattern):
    """pattern: 'S' speech frame, '.' silence frame. Returns the frame index at which a segment ended."""
    frame = np.zeros(512, dtype=np.float32)
    seg._script = [c == "S" for c in pattern]
    for i in range(len(pattern)):
        if seg.process(frame) is not None:
            return i
    return None


def test_pause_threshold_change_applies_to_the_next_pause():
    vad = VADConfig(min_silence_ms=700, min_speech_ms=200)
    frames_per_pause = lambda ms: -(-ms // AudioConfig().block_ms)  # ceil

    seg = make_segmenter(vad)
    ended = feed(seg, "S" * 20 + "." * 60)
    assert ended == 20 + frames_per_pause(700) - 1        # ends after ~700 ms of silence

    vad.min_silence_ms = 300                               # the operator drags the slider (same object the pipeline shares)
    seg = make_segmenter(vad)
    ended = feed(seg, "S" * 20 + "." * 60)
    assert ended == 20 + frames_per_pause(300) - 1         # now ends after ~300 ms

    # ...and mid-utterance: a pause already in progress is judged against the new value
    seg = make_segmenter(vad)
    vad.min_silence_ms = 1400
    assert feed(seg, "S" * 20 + "." * 30) is None          # 30 frames (~960 ms) is no longer enough
