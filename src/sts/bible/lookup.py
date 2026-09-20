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

        if book == "Psalms":
            en_targets = [_synodal_psalm_to_kjv(chapter, v) for v in verses]
        else:
            en_targets = [(chapter, v) for v in verses]

        ru_parts = [self._verse_text(book, chapter, v, lang=0) for v in verses]
        en_parts = [
            self._verse_text(book, target[0], target[1], lang=1) if target else None
            for target in en_targets
        ]

        ru_text = " ".join(p for p in ru_parts if p) or None
        en_text = " ".join(p for p in en_parts if p) or None

        return VerseQuote(
            book=book,
            ru_label=ru_label,
            chapter=chapter,
            verse=verse,
            verse_end=verse_end,
            ru_text=ru_text,
            en_text=en_text,
        )
