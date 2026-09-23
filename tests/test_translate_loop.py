"""Verifies the live pipeline pushes each translated sentence through as soon
as it's ready, instead of waiting for a whole (possibly multi-sentence) ASR
segment to finish translating before any of it reaches captions/TTS."""

import queue

from sts.pipeline import Pipeline


def _stub_pipeline(translate_sentences):
    class FakeTranslator:
        def translate_sentences(self, text):
            return translate_sentences(text)

    class Broadcaster:
        def __init__(self):
            self.translated = []
            self.seqs = []

        def broadcast_translated(self, text, seq=None):
            self.translated.append(text)
            self.seqs.append(seq)

    p = object.__new__(Pipeline)
    p.translator = FakeTranslator()
    p.broadcaster = Broadcaster()
    p._translations = queue.Queue()
    p._last_asr_audio_s = 0.0  # no segment transcribed yet - _broadcast_rtf("translate", ...) is a no-op
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
    p.bible_lookup = None
    p.broadcaster.broadcast_original = lambda text, seq=None: originals.append((seq, text))
    originals: list = []

    p._commit_new_asr_text("Первое предложение.", final=True)   # translation off
    p._commit_new_asr_text("Второе предложение.", final=True)   # translation off
    p.control.enable_translation()
    p._commit_new_asr_text("Третье предложение.", final=True)   # translation on

    # Drain the way _translate_loop does: the two skipped ones are dropped,
    # the third is translated.
    while not p._transcripts.empty():
        text, first_seq = p._transcripts.get()
        if p.control.translation_enabled.is_set() and text.startswith("Третье"):
            Pipeline._translate_segment(p, text, first_seq)

    assert [seq for seq, _ in originals] == [0, 1, 2]
    assert p.broadcaster.seqs == [2]  # the translation belongs to original #2, not #0
