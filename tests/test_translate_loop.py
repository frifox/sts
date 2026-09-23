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

        def broadcast_translated(self, text):
            self.translated.append(text)

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
