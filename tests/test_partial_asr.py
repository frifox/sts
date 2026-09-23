"""Verifies periodic re-transcription of a still-open VAD segment: complete
sentences are released once confirmed, a still-growing trailing fragment is
held back, and the mechanism throttles itself to the ASR stage's own pace
instead of building a backlog.

"Confirmed" means two things, both of which have to hold before a sentence is
actually released:

1. There's real decoded text *after* the sentence-ending punctuation too, in
   the same re-run. Given only a short audio prefix, the model will often end
   its output with a plausible-looking period simply because it ran out of
   audio to decode, not because that's a real sentence boundary - a single
   sentence with nothing after it is exactly that ambiguous case (found by
   testing against real audio: an early one-sentence guess with an empty tail
   was wrong, and - because committed text is never revised - stuck around
   uncorrected for the rest of the segment).
2. The *next* re-run reproduces the same sentence identically. A sentence's
   first sighting is held in `_pending_confirmation` rather than released
   immediately, because a re-run's first pass at a sentence is sometimes
   revised once a little more audio context is available on the next one -
   not just a reworded ending, occasionally a different opening (found the
   same way: real audio where a later re-run changed "Вопрос:" to "Возникает
   вопрос,", which would otherwise have shown, then shown again differently
   worded once the revision was committed on top of it)."""

import queue

import numpy as np

from sts.config import PipelineConfig
from sts.pipeline import Pipeline


def _stub_pipeline():
    class Broadcaster:
        def __init__(self):
            self.original = []
            self.seqs = []
            self.batches = []
            self.status = None

        def broadcast_original(self, text, seq=None, batch=None):
            self.original.append(text)
            self.seqs.append(seq)
            self.batches.append(batch)

        def broadcast_status(self, **kwargs):
            self.status = kwargs

    p = object.__new__(Pipeline)
    p.broadcaster = Broadcaster()
    p.bible_lookup = None  # find_references() on plain sentences below finds nothing, so this is never touched
    p._committed_text = ""
    p._pending_confirmation = None
    p._original_seq = 0
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


def test_a_sentence_stays_held_back_on_its_first_sighting():
    """A sentence that just completed - real text after its period too, so it
    passes the tail-confirmation check - still isn't released on the very
    re-run that first produces it: it needs the *next* re-run to reproduce it
    identically first (see the next test)."""
    p = _stub_pipeline()
    p._commit_new_asr_text("Первое предложение.", final=False)  # held back, no tail yet
    assert p.broadcaster.original == []

    p._commit_new_asr_text("Первое предложение. Второе еще не", final=False)

    assert p.broadcaster.original == []  # first sighting - held pending confirmation
    assert p._committed_text == ""
    assert p._pending_confirmation == "Первое предложение."


def test_a_sentence_is_released_once_confirmed_by_a_second_re_run():
    p = _stub_pipeline()
    p._commit_new_asr_text("Первое предложение. Второе еще не", final=False)  # first sighting - held back
    assert p.broadcaster.original == []

    # more audio, same sentence reproduced identically - that confirms
    # "Первое предложение." wasn't itself still subject to revision.
    p._commit_new_asr_text("Первое предложение. Второе еще не совсем", final=False)

    assert p.broadcaster.original == ["Первое предложение."]
    assert p._committed_text == "Первое предложение."
    assert p._pending_confirmation is None
    assert list(p._transcripts.queue) == [("Первое предложение.", 0)]  # (text, first sentence's seq)


def test_a_revised_first_sighting_replaces_the_pending_candidate_not_release_it():
    """The real regression this guards against, seen in live use: a re-run's
    first pass at a sentence was revised by the very next re-run once a
    little more audio context became available - releasing on first sighting
    would have shown the first version, then shown the revision again on top
    of it once the segment closed. The revised wording simply becomes the new
    pending candidate, awaiting its own confirmation."""
    p = _stub_pipeline()
    p._commit_new_asr_text("Вопрос. Дальше еще не", final=False)
    assert p._pending_confirmation == "Вопрос."

    p._commit_new_asr_text("Возникает вопрос. Дальше еще не совсем", final=False)

    assert p.broadcaster.original == []  # never released the first, unconfirmed version
    assert p._pending_confirmation == "Возникает вопрос."
    assert p._committed_text == ""


def test_second_sentence_released_once_confirmed_by_a_later_re_run():
    p = _stub_pipeline()
    p._commit_new_asr_text("Первое предложение. Второе еще не", final=False)
    p._commit_new_asr_text("Первое предложение. Второе еще не совсем", final=False)  # confirms + releases first
    assert p.broadcaster.original == ["Первое предложение."]

    # "Второе..." now ends in a period for the first time - held back pending
    # its own confirmation, same as sentence one was.
    p._commit_new_asr_text("Первое предложение. Второе теперь готово. Третье начина", final=False)
    assert p.broadcaster.original == ["Первое предложение."]
    assert p._committed_text == "Первое предложение."

    # confirmed identically by the next re-run - now released.
    p._commit_new_asr_text("Первое предложение. Второе теперь готово. Третье начинает", final=False)
    assert p.broadcaster.original == ["Первое предложение.", "Второе теперь готово."]
    assert p._committed_text == "Первое предложение. Второе теперь готово."


def test_final_releases_everything_left_over_even_without_a_period():
    """Unlike a partial, the final pass needs no tail-confirmation *or*
    second-re-run confirmation - the segment is genuinely done, so whatever's
    left over (finished or not, confirmed or not) is released; there's no
    "next re-run" left to catch it otherwise."""
    p = _stub_pipeline()
    p._commit_new_asr_text("Первое предложение. Второе тоже", final=False)
    p._commit_new_asr_text("Первое предложение. Второе тоже сейчас", final=False)  # confirms + releases sentence one
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
    p._commit_new_asr_text("Всё уже сказано. И тут", final=False)  # first sighting - held back
    p._commit_new_asr_text("Всё уже сказано. И тут же", final=False)  # confirms + releases "Всё уже сказано."
    before = list(p.broadcaster.original)
    assert before == ["Всё уже сказано."]

    p._commit_new_asr_text("Всё уже сказано. И тут же", final=True)

    assert p.broadcaster.original == before + ["И тут же"]
    assert p._committed_text == ""


def test_empty_asr_output_is_a_safe_no_op():
    p = _stub_pipeline()
    p._commit_new_asr_text("", final=False)
    assert p.broadcaster.original == [] and p._committed_text == ""
    p._commit_new_asr_text("", final=True)
    assert p.broadcaster.original == [] and p._committed_text == ""

    p._commit_new_asr_text("Кое-что уже есть. И еще", final=False)  # first sighting - held back
    p._commit_new_asr_text("Кое-что уже есть. И еще раз", final=False)  # confirms + releases
    p._commit_new_asr_text("", final=True)  # e.g. a transient hiccup on the closing pass
    assert p.broadcaster.original == ["Кое-что уже есть."]
    assert p._committed_text == ""  # still reset, ready for the next segment


def test_diverged_partial_is_ignored_until_it_resolves():
    """A later re-run's full self-attention occasionally hears an earlier part
    differently once more context arrives. Mid-segment this is just skipped
    (wait for the next re-run or the final pass) rather than guessed at."""
    p = _stub_pipeline()
    p._commit_new_asr_text("Он сказал одно. И еще", final=False)  # first sighting - held back
    p._commit_new_asr_text("Он сказал одно. И еще раз", final=False)  # confirms + releases "Он сказал одно."
    assert p.broadcaster.original == ["Он сказал одно."]

    p._commit_new_asr_text("Он сказал другое. И еще", final=False)  # diverges from what's already committed

    assert p.broadcaster.original == ["Он сказал одно."]  # unchanged - nothing new pushed
    assert p._committed_text == "Он сказал одно."  # unchanged, not overwritten either


def test_final_commit_broadcasts_one_message_per_sentence():
    """A "final" commit can hold several sentences at once (see
    test_final_releases_everything_left_over_even_without_a_period). Each one
    is broadcast as its own "original" message - not the whole commit as one
    message - so the caption page's portrait layout can pair each with the
    matching "translated" message (also one per sentence) by arrival order."""
    p = _stub_pipeline()

    p._commit_new_asr_text("Первое предложение. Второе предложение. Третье предложение.", final=True)

    assert p.broadcaster.original == ["Первое предложение.", "Второе предложение.", "Третье предложение."]


def test_diverged_final_sends_the_full_text_rather_than_risk_dropping_content():
    """On the final pass, a divergence from committed text means whatever was
    already shown/spoken for this segment can't be trusted to still be a
    clean prefix of the truth. Measured against real audio, requiring an
    exact prefix match here caused the final pass to silently drop every
    sentence after the divergence point - so instead everything not already
    covered is always sent (see new_text_since()), even though that means a
    couple of already-shown words may occasionally appear a second time.
    Duplication is visibly wrong; silently missing content is worse and easy
    to miss. Here nothing in the final pass matches the committed text at
    all, so the whole thing is resent."""
    p = _stub_pipeline()
    p._commit_new_asr_text("Он сказал одно. И еще", final=False)  # first sighting - held back
    p._commit_new_asr_text("Он сказал одно. И еще раз", final=False)  # confirms + releases
    assert p.broadcaster.original == ["Он сказал одно."]

    p._commit_new_asr_text("Что-то совсем другое было сказано.", final=True)

    assert p.broadcaster.original == ["Он сказал одно.", "Что-то совсем другое было сказано."]
    assert p._committed_text == ""  # ready for the next segment


def test_diverged_final_only_resends_the_part_not_already_covered():
    """A final pass that only reworded a couple of already-shown words used
    to resend the whole segment (see the test above for when that's still
    correct). Comparing word by word instead of the whole string finds the
    actual divergence point and only resends from there, so "Он сказал" -
    heard identically both times - isn't duplicated just because "одно"
    became something else."""
    p = _stub_pipeline()
    p._commit_new_asr_text("Он сказал одно. И еще", final=False)  # first sighting - held back
    p._commit_new_asr_text("Он сказал одно. И еще раз", final=False)  # confirms + releases
    assert p.broadcaster.original == ["Он сказал одно."]

    p._commit_new_asr_text("Он сказал на самом деле другое, и продолжил дальше.", final=True)

    assert p.broadcaster.original == ["Он сказал одно.", "на самом деле другое, и продолжил дальше."]
    assert p._committed_text == ""


def test_diverged_final_ignores_a_repunctuated_boundary():
    """The real regression this guards against, seen in live use: a final
    pass that hears the same words at a sentence boundary but with different
    punctuation (a period becoming a comma, continuing the sentence instead
    of ending it) used to look like a total divergence - `full_text` no
    longer *starts with* `committed_text` character for character - and
    resent the entire first sentence as a duplicate, verbatim, ahead of the
    genuinely new content. Word-by-word comparison (ignoring punctuation)
    sees through the re-punctuation and only resends what's actually new."""
    p = _stub_pipeline()
    p._commit_new_asr_text("Первое предложение полностью произнесено. И еще", final=False)  # first sighting
    p._commit_new_asr_text("Первое предложение полностью произнесено. И еще раз", final=False)  # confirms
    assert p.broadcaster.original == ["Первое предложение полностью произнесено."]

    # Same words up through "произнесено", but a comma instead of a period -
    # the sentence continues instead of ending, then adds genuinely new content.
    p._commit_new_asr_text(
        "Первое предложение полностью произнесено, а затем добавлено что-то новое.", final=True
    )

    assert p.broadcaster.original == [
        "Первое предложение полностью произнесено.",
        "а затем добавлено что-то новое.",
    ]
    assert p._committed_text == ""


def test_each_broadcast_sentence_gets_the_next_sequence_number():
    """The caption page pairs an original with its translation by this
    number, so it must be unique and keep counting across commits - and the
    queued chunk records its *first* sentence's number, so translation can
    number its own sentences to match."""
    p = _stub_pipeline()

    p._commit_new_asr_text("Первое предложение. Второе предложение.", final=True)
    p._commit_new_asr_text("Третье предложение.", final=True)

    assert p.broadcaster.seqs == [0, 1, 2]
    assert list(p._transcripts.queue) == [("Первое предложение. Второе предложение.", 0), ("Третье предложение.", 2)]


def test_sequence_numbers_advance_even_with_no_broadcaster():
    p = _stub_pipeline()
    p.broadcaster = None

    p._commit_new_asr_text("Первое предложение. Второе предложение.", final=True)
    p._commit_new_asr_text("Третье предложение.", final=True)

    assert list(p._transcripts.queue)[1] == ("Третье предложение.", 2)


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


def test_a_commits_sentences_share_one_batch_id():
    """The page keeps every sentence of one commit highlighted together and
    dims the previous commit only when a new one starts - so each commit's
    sentences must carry the same batch id (its first sequence number), and
    the next commit a different one."""
    p = _stub_pipeline()

    p._commit_new_asr_text("Первое предложение. Второе предложение.", final=True)
    p._commit_new_asr_text("Третье предложение.", final=True)

    assert p.broadcaster.batches == [0, 0, 2]
