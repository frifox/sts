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


def test_peek_snapshots_the_in_progress_buffer_without_disturbing_it():
    seg = make_segmenter(VADConfig(min_silence_ms=700, min_speech_ms=200))
    assert seg.peek() is None  # nothing buffered yet

    frame = (np.arange(512, dtype=np.float32) + 1)  # distinct nonzero content, easy to check concatenation
    n = 10  # well over min_speech_ms (200ms / 32ms per frame = ~7), so the segment won't be dropped as noise
    seg._script = [True] * n
    for _ in range(n):
        assert seg.process(frame) is None  # segment hasn't closed

    snapshot = seg.peek()
    assert snapshot is not None and snapshot.shape == (512 * n,)
    assert np.array_equal(snapshot, np.concatenate([frame] * n))

    # peek() must not consume/clear anything - state keeps accumulating, and a
    # real close later still produces the *whole* segment, snapshot included.
    assert seg.peek() is not None
    assert seg._speech_ms == n * seg.audio_cfg.block_ms

    seg._script = [False] * 30  # enough silence to close the segment
    closed = None
    for _ in range(30):
        closed = seg.process(frame)
        if closed is not None:
            break
    assert closed is not None and len(closed) > len(snapshot)  # includes the earlier speech plus trailing silence
    assert seg.peek() is None  # buffer cleared once the segment closed
