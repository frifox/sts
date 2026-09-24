"""Captions -> translation: locked sentences are numbered, shown, checked for
Bible references, and translated in order; the still-open text goes to the
page as one live line (see Pipeline._publish_captions / _translate_sentence)."""

import queue

from sts.pipeline import Pipeline
from sts.transcript import Sentence


class _Broadcaster:
    def __init__(self):
        self.original = []
        self.recent = []
        self.translated = []

    def broadcast_original(self, text, seq=None, batch=None):
        self.original.append((text, seq))

    def broadcast_original_recent(self, text):
        self.recent.append(text)

    def broadcast_translated(self, text, seq=None, batch=None):
        self.translated.append((text, seq))

    def broadcast_verse(self, *args):
        pass

    def broadcast_rtf(self, stage, value):
        pass


def _pipeline():
    p = object.__new__(Pipeline)
    p.broadcaster = _Broadcaster()
    p._transcripts = queue.Queue()
    p._translations = queue.Queue()
    p._original_seq = 0
    p._last_final_end = None
    p._recent_text = ""
    p._rtf_window = {}
    p.bible_lookup = None  # none of the texts below cite a verse
    return p


def test_locked_sentences_are_numbered_shown_and_queued_for_translation():
    p = _pipeline()
    p._publish_captions([Sentence("Первое.", 0.0, 1.0, True), Sentence("Второе.", 1.3, 2.0, True)], ["Трет"])
    p._publish_captions([Sentence("Третье.", 5.0, 6.0, True)], [])

    assert p.broadcaster.original == [("Первое.", 0), ("Второе.", 1), ("Третье.", 2)]
    queued = [p._transcripts.get_nowait() for _ in range(3)]
    assert [(text, seq) for text, seq, _, _ in queued] == [("Первое.", 0), ("Второе.", 1), ("Третье.", 2)]
    assert [spoken for _, _, spoken, _ in queued] == [1.0, 0.7, 1.0]
    # Straight on from the one before -> spoken as the same thought; after a
    # real pause (3 s here) -> the longer between-thoughts gap.
    assert [continues for _, _, _, continues in queued] == [False, True, False]


def test_the_live_line_is_the_open_sentences_joined_and_only_resent_when_it_changes():
    p = _pipeline()
    p._publish_captions([], ["Братья и сёстры.", "Когда мы"])
    p._publish_captions([], ["Братья и сёстры.", "Когда мы"])  # unchanged
    p._publish_captions([Sentence("Братья и сёстры.", 0, 1, True)], [])
    assert p.broadcaster.recent == ["Братья и сёстры. Когда мы", ""]


def _translating_pipeline(translate):
    p = _pipeline()

    class Translator:
        def translate_sentences(self, text):
            yield from translate(text)

    p.translator = Translator()
    return p


def test_a_locked_sentence_is_translated_whole_and_paired_by_its_number():
    p = _translating_pipeline(lambda text: iter(["He said so.", "Then left."]))  # split by NLLB's sentence splitter
    p._translate_sentence("Он так сказал. И ушёл.", seq=7, spoken_s=2.0, continues=True)
    assert p.broadcaster.translated == [("He said so. Then left.", 7)]
    assert p._translations.get_nowait() == ("He said so. Then left.", True)


def test_an_empty_translation_is_neither_shown_nor_spoken():
    p = _translating_pipeline(lambda text: iter([""]))
    p._translate_sentence("…", seq=3)
    assert p.broadcaster.translated == []
    assert p._translations.empty()


def test_translations_pair_with_their_originals_whatever_order_they_finish_in():
    """Captions keep flowing while translation lags or is off; the number,
    not arrival order, says which original a translation belongs to."""
    p = _translating_pipeline(lambda text: iter([text.upper()]))
    p._publish_captions([Sentence(f"s{i}.", i, i + 0.5, True) for i in range(3)], [])
    items = [p._transcripts.get_nowait() for _ in range(3)]
    for text, seq, spoken, continues in reversed(items):
        p._translate_sentence(text, seq, spoken, continues)
    assert p.broadcaster.translated == [("S2.", 2), ("S1.", 1), ("S0.", 0)]
