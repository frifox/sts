"""Verifies the live pipeline pushes each translated sentence through as soon
as it's ready, instead of waiting for a whole (possibly multi-sentence) ASR
segment to finish translating before any of it reaches captions/TTS."""

import queue

from sts.pipeline import Pipeline


def _stub_pipeline(translate_sentences):
    class FakeTranslator:
        def translate_sentences(self, text, context=None):
            return translate_sentences(text)

    class Broadcaster:
        def __init__(self):
            self.translated = []
            self.seqs = []
            self.batches = []

        def broadcast_translated(self, text, seq=None, batch=None):
            self.translated.append(text)
            self.seqs.append(seq)
            self.batches.append(batch)

    p = object.__new__(Pipeline)
    p.translator = FakeTranslator()
    p.broadcaster = Broadcaster()
    p._translations = queue.Queue()
    p._last_asr_audio_s = 0.0  # no segment transcribed yet - _broadcast_rtf("translate", ...) is a no-op
    p._last_sentence_translated = None
    return p


def test_each_sentence_is_pushed_and_broadcast_as_it_is_translated():
    """The real regression this guards against: translating "A. B. C." used to
    block on B and C before A reached the queue/broadcast at all."""
    order: list[str] = []

    def translate_sentences(text):
        for sentence in ("A translated.", "B translated.", "C translated."):
            order.append(f"about to yield {sentence}")
            # The consumer must have already drained *this* sentence from the
            # queue and broadcast it before the generator is asked for the next
            # one - proving sentence 1 doesn't wait on sentence 2/3.
            yield sentence
            order.append(f"queue after yielding {sentence}: {list(p._translations.queue)}")

    p = _stub_pipeline(translate_sentences)

    Pipeline._translate_segment(p, "A. B. C.")

    assert p.broadcaster.translated == ["A translated.", "B translated.", "C translated."]
    assert list(p._translations.queue) == ["A translated.", "B translated.", "C translated."]
    assert order == [
        "about to yield A translated.",
        "queue after yielding A translated.: ['A translated.']",
        "about to yield B translated.",
        "queue after yielding B translated.: ['A translated.', 'B translated.']",
        "about to yield C translated.",
        "queue after yielding C translated.: ['A translated.', 'B translated.', 'C translated.']",
    ]


def test_empty_translated_sentences_are_not_queued_or_broadcast():
    def translate_sentences(text):
        yield "Kept."
        yield ""  # e.g. a fragment that translated to nothing
        yield "Also kept."

    p = _stub_pipeline(translate_sentences)

    Pipeline._translate_segment(p, "text")

    assert p.broadcaster.translated == ["Kept.", "Also kept."]
    assert list(p._translations.queue) == ["Kept.", "Also kept."]


def test_each_translation_is_broadcast_with_its_originals_sequence_number():
    """The caption page matches a translation to its original by this number
    (see Pipeline._commit_new_asr_text), so sentence i of a chunk whose first
    sentence was numbered `first_seq` must go out as first_seq + i."""
    p = _stub_pipeline(lambda text: iter(["A.", "B.", "C."]))

    Pipeline._translate_segment(p, "x. y. z.", first_seq=7)

    assert p.broadcaster.seqs == [7, 8, 9]


def test_an_empty_translation_leaves_its_number_unused_instead_of_shifting_later_ones():
    """A sentence that translates to nothing is skipped, but the sentences
    after it must keep their own numbers - not slide down into its place."""
    p = _stub_pipeline(lambda text: iter(["A.", "", "C."]))

    Pipeline._translate_segment(p, "x. y. z.", first_seq=0)

    assert p.broadcaster.translated == ["A.", "C."]
    assert p.broadcaster.seqs == [0, 2]


def test_captions_first_then_translation_pairs_by_number_not_by_arrival_order():
    """The reported bug: Caption runs alone for a few sentences (translation
    off, so those originals are never translated), then Translate is turned
    on. Arrival order would pair the first translation with the first, long
    since untranslated, original - shifting every later pair. Each translation
    instead carries the number of the original it actually belongs to."""
    import threading

    from sts.control import RecognitionControl

    p = _stub_pipeline(lambda text: iter([f"EN({text})"]))
    p.control = RecognitionControl(16000)
    p.control.disable_translation()
    p._transcripts = queue.Queue()
    p._stop = threading.Event()
    p._committed_text = ""
    p._pending_confirmation = None
    p._original_seq = 0
    p._last_sentence_original = None
    p.bible_lookup = None
    p.broadcaster.broadcast_original = lambda text, seq=None, batch=None: originals.append((seq, text))
    originals: list = []

    p._commit_new_asr_text("Первое предложение.", final=True)   # translation off
    p._commit_new_asr_text("Второе предложение.", final=True)   # translation off
    p.control.enable_translation()
    p._commit_new_asr_text("Третье предложение.", final=True)   # translation on

    # Drain the way _translate_loop does: the two skipped ones are dropped,
    # the third is translated.
    while not p._transcripts.empty():
        text, first_seq, context = p._transcripts.get()
        if p.control.translation_enabled.is_set() and text.startswith("Третье"):
            Pipeline._translate_segment(p, text, first_seq, context)

    assert [seq for seq, _ in originals] == [0, 1, 2]
    assert p.broadcaster.seqs == [2]  # the translation belongs to original #2, not #0


def test_translations_carry_their_commits_batch_id_however_slowly_they_arrive():
    """A slow translation (GPU contention with a slow TTS, say) must still
    group with its own commit on the page, so every sentence of the chunk
    goes out with the chunk's first sequence number as its batch id."""
    p = _stub_pipeline(lambda text: iter(["A.", "B."]))

    Pipeline._translate_segment(p, "x. y.", first_seq=4)

    assert p.broadcaster.batches == [4, 4]


# -- _strip_repeated_prefix ---------------------------------------------------
# The real bug this guards: translating a continuation with the previous
# sentence as context (see _commit_new_asr_text) sometimes has NLLB restate
# that whole sentence verbatim before continuing, rather than picking up
# mid-thought - reproduced against real transcripts (and the real model): a
# caption showing "These are the words of our Lord Jesus Christ." followed
# immediately by "These are the words of our Lord Jesus Christ during his
# last long talk..." reads as a broken, duplicated repeat.


def test_strip_repeated_prefix_removes_an_exact_leading_duplicate():
    from sts.pipeline import _strip_repeated_prefix

    previous = "These are the words of our Lord Jesus Christ."
    combined = "These are the words of our Lord Jesus Christ during his last long talk with his disciples."

    assert _strip_repeated_prefix(combined, previous) == "during his last long talk with his disciples."


def test_strip_repeated_prefix_is_case_insensitive_and_ignores_previous_punctuation():
    from sts.pipeline import _strip_repeated_prefix

    assert _strip_repeated_prefix("SOMETHING extra here.", "something") == "extra here."


def test_strip_repeated_prefix_leaves_a_non_matching_translation_unchanged():
    """The ordinary, more common case: NLLB paraphrases rather than repeats -
    nothing to safely strip, so the (mildly redundant but coherent) combined
    translation is shown as-is."""
    from sts.pipeline import _strip_repeated_prefix

    previous = "Each one will contain several items, different numbers of items."
    combined = "Each one will have several items, different numbers of items in each of these sections."

    assert _strip_repeated_prefix(combined, previous) == combined


def test_strip_repeated_prefix_handles_no_previous_sentence():
    from sts.pipeline import _strip_repeated_prefix

    assert _strip_repeated_prefix("Anything.", None) == "Anything."


def test_strip_repeated_prefix_falls_back_when_nothing_would_remain():
    """A pathological case (NLLB adds no new content at all) - better to show
    the duplicate than nothing."""
    from sts.pipeline import _strip_repeated_prefix

    assert _strip_repeated_prefix("Exactly the same.", "Exactly the same.") == "Exactly the same."


def test_translate_segment_strips_a_repeated_prefix_from_a_continuations_first_sentence():
    """Integration: _translate_segment applies the strip only to the first
    sentence of a commit that actually used context, and only ever tracks
    _last_sentence_translated from what was actually broadcast."""
    p = _stub_pipeline(lambda text: iter([
        "These are the words of our Lord Jesus Christ during his last long talk.",
        "New unrelated sentence.",
    ]))
    p._last_sentence_translated = "These are the words of our Lord Jesus Christ."

    Pipeline._translate_segment(p, "text", context="Это слова нашего Господа Иисуса Христа.")

    assert p.broadcaster.translated == ["during his last long talk.", "New unrelated sentence."]
    assert p._last_sentence_translated == "New unrelated sentence."  # advances to the newest one shown


def test_translate_segment_does_not_strip_when_there_is_no_context():
    """A sentence translated without context (the ordinary case) must never
    have its start clipped just because it happens to share wording with
    whatever was last shown."""
    p = _stub_pipeline(lambda text: iter(["These are the words of our Lord Jesus Christ, spoken plainly."]))
    p._last_sentence_translated = "These are the words of our Lord Jesus Christ."

    Pipeline._translate_segment(p, "text")

    assert p.broadcaster.translated == ["These are the words of our Lord Jesus Christ, spoken plainly."]
