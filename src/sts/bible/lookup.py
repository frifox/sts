"""Looks up exact verse text (Russian Synodal + English KJV, Pure Cambridge
Edition) for a parsed `BibleReference`.

Both texts are public domain and bundled as a single gzip'd JSON file
(`sts/bible/data/bible.json.gz`, built from
github.com/scrollmapper/bible_databases) so lookups work fully offline.

Psalms needs special handling: the Synodal Psalter follows Septuagint
numbering while KJV follows the Masoretic numbering it's offset from by
(usually) exactly one for chapters 10-146. Getting this wrong doesn't just
misalign a verse, it shows the text of a *different psalm* entirely, so it's
corrected for; the handful of psalms that split or merge across the two
traditions (9/10, 114-116, 147) have no clean 1:1 mapping and are left
without an English match rather than showing the wrong content. No other
book has a full chapter-level shift like this, only the occasional
off-by-one verse near a section boundary, which is left uncorrected.
"""

import gzip
import json
from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class VerseQuote:
    book: str
    ru_label: str
    chapter: int
    verse: int
    verse_end: int | None
    ru_text: str | None
    en_text: str | None
    # The same reference in English (KJV) naming and numbering, e.g. "Psalm
    # 23:1" for the Synodal "Псалтирь 22:1" - the numbers en_text was taken
    # from.
    en_reference: str | None = None


# Dataset book name -> how it's written in an English reference, where different.
_EN_NAMES = {"Psalms": "Psalm", "Revelation of John": "Revelation"}
_ROMAN = {"I": "1", "II": "2", "III": "3"}


def english_book_name(book: str) -> str:
    if book in _EN_NAMES:
        return _EN_NAMES[book]
    first, _, rest = book.partition(" ")
    return f"{_ROMAN[first]} {rest}" if first in _ROMAN and rest else book


def english_reference(book: str, chapter: int, verse: int, verse_end: int | None) -> str:
    """The reference as written in English, with a Psalm's chapter and verses
    in KJV numbering (see _synodal_psalm_to_kjv) - left as they are for the
    few psalms without a 1:1 mapping."""
    if book == "Psalms":
        start = _synodal_psalm_to_kjv(chapter, verse)
        end = _synodal_psalm_to_kjv(chapter, verse_end) if verse_end else None
        if start and (end or not verse_end):
            chapter, verse = start
            verse_end = end[1] if end else None
    return f"{english_book_name(book)} {chapter}:{verse}" + (f"-{verse_end}" if verse_end else "")


def _synodal_psalm_to_kjv(chapter: int, verse: int) -> tuple[int, int] | None:
    if chapter <= 8 or 148 <= chapter <= 150:
        return chapter, verse
    if 10 <= chapter <= 112:
        return chapter + 1, verse
    if 116 <= chapter <= 145:
        return chapter + 1, verse
    return None  # 9, 113-115, 146-147: split/merged, no 1:1 verse mapping


class BibleLookup:
    def __init__(self) -> None:
        raw = resources.files("sts.bible.data").joinpath("bible.json.gz").read_bytes()
        self._data: dict = json.loads(gzip.decompress(raw))

    def _verse_text(self, book: str, chapter: int | None, verse: int | None, lang: int) -> str | None:
        if chapter is None or verse is None:
            return None
        entry = self._data.get(book, {}).get(str(chapter), {}).get(str(verse))
        text = entry[lang] if entry else None
        # KJVPCE marks paragraph starts with a leading pilcrow; not part of the verse text.
        return text.lstrip("¶ ") if text else text

    def lookup(self, book: str, ru_label: str, chapter: int, verse: int, verse_end: int | None = None) -> VerseQuote:
        verses = range(verse, (verse_end or verse) + 1)
        # Only worth numbering when there's more than one verse to tell apart -
        # a single verse's own reference (e.g. "Иоанна 3:16") already says
        # which one it is. Both languages are labeled with the same (Synodal)
        # verse number, even though a Psalm's English text is pulled from a
        # different underlying KJV verse (see _synodal_psalm_to_kjv) - that
        # KJV numbering is never spoken or shown elsewhere, so labeling by it
        # instead would just be confusing next to the Russian side.
        numbered = len(verses) > 1

        if book == "Psalms":
            en_targets = [_synodal_psalm_to_kjv(chapter, v) for v in verses]
        else:
            en_targets = [(chapter, v) for v in verses]

        ru_parts = [self._verse_text(book, chapter, v, lang=0) for v in verses]
        en_parts = [
            self._verse_text(book, target[0], target[1], lang=1) if target else None
            for target in en_targets
        ]

        def join(parts: list[str | None]) -> str | None:
            pieces = [f"({v}) {p}" if numbered else p for v, p in zip(verses, parts) if p]
            return " ".join(pieces) or None

        ru_text = join(ru_parts)
        en_text = join(en_parts)

        return VerseQuote(
            book=book,
            ru_label=ru_label,
            chapter=chapter,
            verse=verse,
            verse_end=verse_end,
            ru_text=ru_text,
            en_text=en_text,
            en_reference=english_reference(book, chapter, verse, verse_end),
        )
