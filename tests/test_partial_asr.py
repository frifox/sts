"""Verifies periodic re-transcription of a still-open VAD segment: complete
sentences are released as soon as they're confirmed, a still-growing trailing
fragment is held back, and the mechanism throttles itself to the ASR stage's
own pace instead of building a backlog.

"Confirmed" specifically means: there's real decoded text *after* the
sentence-ending punctuation too, in the same re-run. Given only a short audio
prefix, the model will often end its output with a plausible-looking period
simply because it ran out of audio to decode, not because that's a real
sentence boundary - a single sentence with nothing after it is exactly that
ambiguous case, so it's held back rather than committed early (this was found
by testing against real audio: an early one-sentence guess with an empty tail
was wrong, and - because committed text is never revised - stuck around
uncorrected for the rest of the segment)."""

import queue

import numpy as np

from sts.config import PipelineConfig
from sts.pipeline import Pipeline


def _stub_pipeline():
    class Broadcaster:
        def __init__(self):
            self.original = []
            self.status = None

        def broadcast_original(self, text):
            self.original.append(text)

        def broadcast_status(self, **kwargs):
            self.status = kwargs

    p = object.__new__(Pipeline)
    p.broadcaster = Broadcaster()
    p.bible_lookup = None  # find_references() on plain sentences below finds nothing, so this is never touched
    p._committed_text = ""
    p._transcripts = queue.Queue()
    return p


# -- _commit_new_asr_text --------------------------------------------------


def test_a_lone_sentence_with_nothing_after_it_is_held_back():
    """The exact real-world failure this guards against: an early re-run
    whose growing transcript is just one sentence, full stop, with the model
    having decoded nothing past it yet - too ambiguous to trust."""
    p = _stub_pipeline()

    p._commit_new_asr_text("Первое предложение.", final=False)

    assert p.broadcaster.original == []
    assert p._committed_text == ""  # nothing committed yet


def test_a_sentence_is_released_once_the_model_has_moved_past_it():
    p = _stub_pipeline()
    p._commit_new_asr_text("Первое предложение.", final=False)  # held back, no tail yet
    assert p.broadcaster.original == []

    # more audio: the model has now decoded real content *after* the period -
    # that's what confirms "Первое предложение." really was a full sentence.
    p._commit_new_asr_text("Первое предложение. Второе еще не", final=False)

    assert p.broadcaster.original == ["Первое предложение."]
    assert p._committed_text == "Первое предложение."
    assert list(p._transcripts.queue) == ["Первое предложение."]


def test_second_sentence_released_once_a_third_confirms_it():
    p = _stub_pipeline()
    p._commit_new_asr_text("Первое предложение. Второе еще не", final=False)
    assert p.broadcaster.original == ["Первое предложение."]

    # "Второе..." now ends in a period, but nothing follows it yet - held back,
    # same as the first sentence was.
    p._commit_new_asr_text("Первое предложение. Второе теперь готово.", final=False)
    assert p.broadcaster.original == ["Первое предложение."]
    assert p._committed_text == "Первое предложение."

    # a third sentence starts: that confirms sentence two.
    p._commit_new_asr_text("Первое предложение. Второе теперь готово. Третье начина", final=False)
    assert p.broadcaster.original == ["Первое предложение.", "Второе теперь готово."]
    assert p._committed_text == "Первое предложение. Второе теперь готово."


def test_final_releases_everything_left_over_even_without_a_period():
    """Unlike a partial, the final pass needs no tail-confirmation - the
    segment is genuinely done, so whatever's left over (finished sentence or
    not) is released; there's no "next re-run" left to catch it otherwise."""
    p = _stub_pipeline()
    p._commit_new_asr_text("Первое предложение. Второе тоже", final=False)  # confirms + releases sentence one
    assert p.broadcaster.original == ["Первое предложение."]

    p._commit_new_asr_text("Первое предложение. Второе тоже без точки", final=True)

    assert p.broadcaster.original == ["Первое предложение.", "Второе тоже без точки"]
    assert p._committed_text == ""  # ready for the next segment


def test_final_with_a_single_uncommitted_sentence_still_releases_it():
    """A short segment that closes before any partial ever fired (e.g. the
    whole thing was under one partial_interval_ms) must still get its text
    out via the final pass, tail-confirmation or not."""
    p = _stub_pipeline()

    p._commit_new_asr_text("Только одно предложение.", final=True)

    assert p.broadcaster.original == ["Только одно предложение."]
    assert p._committed_text == ""


def test_final_with_nothing_new_still_resets_for_the_next_segment():
    """The common case: partials already caught everything, so the final pass
    (re-transcribing the whole, now-closed segment) reproduces exactly what
    was already committed. Must not skip resetting state for the next segment."""
    p = _stub_pipeline()
    p._commit_new_asr_text("Всё уже сказано. И тут", final=False)  # confirms + releases "Всё уже сказано."
    before = list(p.broadcaster.original)
    assert before == ["Всё уже сказано."]

    p._commit_new_asr_text("Всё уже сказано. И тут", final=True)

    assert p.broadcaster.original == before + ["И тут"]
    assert p._committed_text == ""


def test_empty_asr_output_is_a_safe_no_op():
    p = _stub_pipeline()
    p._commit_new_asr_text("", final=False)
    assert p.broadcaster.original == [] and p._committed_text == ""
    p._commit_new_asr_text("", final=True)
    assert p.broadcaster.original == [] and p._committed_text == ""

    p._commit_new_asr_text("Кое-что уже есть. И еще", final=False)  # confirms + releases "Кое-что уже есть."
    p._commit_new_asr_text("", final=True)  # e.g. a transient hiccup on the closing pass
    assert p.broadcaster.original == ["Кое-что уже есть."]
    assert p._committed_text == ""  # still reset, ready for the next segment


def test_diverged_partial_is_ignored_until_it_resolves():
    """A later re-run's full self-attention occasionally hears an earlier part
    differently once more context arrives. Mid-segment this is just skipped
    (wait for the next re-run or the final pass) rather than guessed at."""
    p = _stub_pipeline()
    p._commit_new_asr_text("Он сказал одно. И еще", final=False)  # confirms + releases "Он сказал одно."
    assert p.broadcaster.original == ["Он сказал одно."]

    p._commit_new_asr_text("Он сказал другое. И еще", final=False)  # diverges from what's already committed

    assert p.broadcaster.original == ["Он сказал одно."]  # unchanged - nothing new pushed
    assert p._committed_text == "Он сказал одно."  # unchanged, not overwritten either


def test_diverged_final_sends_the_full_text_rather_than_risk_dropping_content():
    """On the final pass, a divergence from committed text means whatever was
    already shown/spoken for this segment can't be trusted to still be a
    clean prefix of the truth. Measured against real audio, requiring an
    exact prefix match here caused the final pass to silently drop every
    sentence after the divergence point - so instead the full, authoritative
    final text is always sent, even though that means the couple of already-
    shown words may appear a second time. Duplication is visibly wrong;
    silently missing content is worse and easy to miss."""
    p = _stub_pipeline()
    p._commit_new_asr_text("Он сказал одно. И еще", final=False)
    assert p.broadcaster.original == ["Он сказал одно."]

    p._commit_new_asr_text("Он сказал на самом деле другое, и продолжил дальше.", final=True)

    assert p.broadcaster.original == ["Он сказал одно.", "Он сказал на самом деле другое, и продолжил дальше."]
    assert p._committed_text == ""  # ready for the next segment


# -- _process_captured_frame (segment loop's per-frame decision) -----------


class FakeSegmenter:
    def __init__(self, close_on_frame=None, peek_value=None):
        self.close_on_frame = close_on_frame
        self.peek_value = peek_value
        self.calls = 0
        self.peeked = 0

    def process(self, frame):
        self.calls += 1
        if self.calls == self.close_on_frame:
            return np.zeros(10, dtype=np.float32)
        return None

    def peek(self):
        self.peeked += 1
        return self.peek_value


class FakeControl:
    class _Flag:
        def __init__(self, value):
            self._value = value

        def is_set(self):
            return self._value

    def __init__(self, active=True, training=False):
        self.active = self._Flag(active)
        self.training = self._Flag(training)


def _stub_frame_pipeline(segmenter, cfg, active=True):
    p = object.__new__(Pipeline)
    p.segmenter = segmenter
    p.cfg = cfg
    p.control = FakeControl(active=active)
    p._segments = queue.Queue()
    p._frames_since_partial = 0
    return p


def test_partial_fires_once_the_interval_elapses_and_then_resets():
    cfg = PipelineConfig()
    cfg.vad.partial_interval_ms = 100  # e.g. ~3 frames at 32ms/frame
    cfg.audio.block_ms = 32
    segmenter = FakeSegmenter(peek_value=np.ones(5, dtype=np.float32))
    p = _stub_frame_pipeline(segmenter, cfg)

    for _ in range(3):  # 96ms - not yet at the 100ms interval
        p._process_captured_frame(np.zeros(1))
    assert p._segments.qsize() == 0

    p._process_captured_frame(np.zeros(1))  # 128ms - now over the interval
    assert p._segments.qsize() == 1
    assert p._segments.get() == ("partial", segmenter.peek_value)
    assert p._frames_since_partial == 0  # counter reset after firing


def test_partial_is_skipped_while_the_asr_stage_has_not_caught_up():
    cfg = PipelineConfig()
    cfg.vad.partial_interval_ms = 32
    cfg.audio.block_ms = 32
    segmenter = FakeSegmenter(peek_value=np.ones(5, dtype=np.float32))
    p = _stub_frame_pipeline(segmenter, cfg)
    p._segments.put(("final", np.zeros(1)))  # ASR stage still has work queued

    p._process_captured_frame(np.zeros(1))

    assert p._segments.qsize() == 1  # no partial added on top
    assert segmenter.peeked == 0  # didn't even bother snapshotting


def test_partial_interval_zero_disables_the_mechanism():
    cfg = PipelineConfig()
    cfg.vad.partial_interval_ms = 0
    segmenter = FakeSegmenter(peek_value=np.ones(5, dtype=np.float32))
    p = _stub_frame_pipeline(segmenter, cfg)

    for _ in range(50):
        p._process_captured_frame(np.zeros(1))

    assert p._segments.qsize() == 0
    assert segmenter.peeked == 0


def test_partial_skipped_while_inactive():
    cfg = PipelineConfig()
    cfg.vad.partial_interval_ms = 32
    segmenter = FakeSegmenter(peek_value=np.ones(5, dtype=np.float32))
    p = _stub_frame_pipeline(segmenter, cfg, active=False)

    p._process_captured_frame(np.zeros(1))

    assert p._segments.qsize() == 0


def test_no_partial_when_nothing_is_buffered_yet():
    cfg = PipelineConfig()
    cfg.vad.partial_interval_ms = 32
    segmenter = FakeSegmenter(peek_value=None)  # e.g. still in leading silence, VAD has nothing buffered
    p = _stub_frame_pipeline(segmenter, cfg)

    p._process_captured_frame(np.zeros(1))

    assert p._segments.qsize() == 0


def test_a_closing_segment_is_pushed_as_final_and_resets_the_partial_timer():
    cfg = PipelineConfig()
    cfg.vad.partial_interval_ms = 32
    segmenter = FakeSegmenter(close_on_frame=1)
    p = _stub_frame_pipeline(segmenter, cfg)
    p._frames_since_partial = 5  # pretend some time had already elapsed

    p._process_captured_frame(np.zeros(1))

    assert p._segments.qsize() == 1
    kind, segment = p._segments.get()
    assert kind == "final" and len(segment) == 10
    assert p._frames_since_partial == 0
