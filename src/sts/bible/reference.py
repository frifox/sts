"""Spots spoken Bible references ("Матфея 1:15", "от Иоанна, глава 3, стих
16", "Псалом 22") in a Russian ASR transcript.

Works on word/digit tokens rather than raw text or a single do-everything
regex: at each token, try every book's stem (longest/most specific first);
on a match, require chapter and verse numbers (digits or Russian number
words) immediately after, then advance the scan past what was consumed so
the same words can't be claimed by a second, looser match.
"""

import re
from dataclasses import dataclass

from sts.bible.books import BOOKS_BY_SPECIFICITY, BookInfo
from sts.bible.numbers import parse_number_words

_TOKEN_RE = re.compile(r"[а-яё]+|\d+")

_CHAPTER_KEYWORDS = {"глава", "гл"}
_VERSE_KEYWORDS = {"стих", "стиха", "стихи", "ст"}
_RANGE_KEYWORDS = {"по", "до"}
_ORDINAL_STEMS = {"перв": 1, "втор": 2, "трет": 3, "четверт": 4}
_ORDINAL_SUFFIX_TOLERANCE = 3
# Words that can sit between an ordinal and the book name it modifies, e.g.
# "Первое послание к Коринфянам" - skipped over while looking back for the
# ordinal instead of stopping the search.
_CONNECTOR_WORDS = {
    "послание", "послания", "книга", "книги", "письмо",
    "к", "от", "апостола", "апостолу", "святого", "святому",
}


@dataclass(frozen=True)
class BibleReference:
    book: str  # dataset key, e.g. "Matthew"
    ru_label: str  # short Russian display form, e.g. "Матфея"
    chapter: int
    verse: int
    verse_end: int | None = None


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _read_number(tokens: list[str], idx: int) -> tuple[int, int] | None:
    if idx >= len(tokens):
        return None
    if tokens[idx].isdigit():
        return int(tokens[idx]), 1
    return parse_number_words(tokens, idx)


def _ordinal_at(tokens: list[str], idx: int) -> int | None:
    if idx < 0 or idx >= len(tokens):
        return None
    token = tokens[idx]
    if token.isdigit():
        value = int(token)
        return value if 1 <= value <= 4 else None
    for stem, value in _ORDINAL_STEMS.items():
        if token.startswith(stem) and len(token) - len(stem) <= _ORDINAL_SUFFIX_TOLERANCE:
            return value
    return None


def _preceding_ordinal(tokens: list[str], stem_start: int, max_back: int = 5) -> int | None:
    # Handles a single word ("первое"), a split digit + declension remnant
    # ("1", "е"), and connector words in between ("Первое послание к
    # Коринфянам" has two words between the ordinal and the book name).
    idx = stem_start - 1
    steps = 0
    while idx >= 0 and steps < max_back:
        ordinal = _ordinal_at(tokens, idx)
        if ordinal is not None:
            return ordinal
        token = tokens[idx]
        if token in _CONNECTOR_WORDS or len(token) <= 3:
            idx -= 1
            steps += 1
            continue
        break  # a real, unrelated word - stop looking
    return None


def _match_book_at(tokens: list[str], idx: int) -> BookInfo | None:
    preceding_ordinal = _preceding_ordinal(tokens, idx)
    for book in BOOKS_BY_SPECIFICITY:
        if book.number is not None and book.number != preceding_ordinal:
            continue
        if book.number is None and preceding_ordinal is not None:
            continue

        end = idx + len(book.stems)
        if end > len(tokens):
            continue
        if all(
            tokens[idx + offset].startswith(stem) and len(tokens[idx + offset]) - len(stem) <= 5
            for offset, stem in enumerate(book.stems)
        ):
            return book
    return None


def find_references(text: str) -> list[BibleReference]:
    tokens = _tokenize(text)
    references: list[BibleReference] = []

    i = 0
    while i < len(tokens):
        book = _match_book_at(tokens, i)
        if book is None:
            i += 1
            continue

        j = i + len(book.stems)
        if j < len(tokens) and tokens[j] in _CHAPTER_KEYWORDS:
            j += 1

        chapter_match = _read_number(tokens, j)
        if chapter_match is None:
            i += 1
            continue
        chapter, consumed = chapter_match
        j += consumed

        if j < len(tokens) and tokens[j] in _VERSE_KEYWORDS:
            j += 1

        verse_match = _read_number(tokens, j)
        if verse_match is None:
            i += 1
            continue
        verse, consumed = verse_match
        j += consumed

        verse_end = None
        if j < len(tokens) and tokens[j] in _RANGE_KEYWORDS:
            end_match = _read_number(tokens, j + 1)
            if end_match is not None:
                verse_end, consumed = end_match
                j += 1 + consumed
        elif j < len(tokens) and tokens[j].isdigit() and int(tokens[j]) > verse:
            # Written shorthand "22-23" tokenizes as adjacent bare digits
            # (the hyphen is dropped) with no "по"/"до" between them.
            verse_end = int(tokens[j])
            j += 1

        references.append(
            BibleReference(
                book=book.key, ru_label=book.ru_label, chapter=chapter, verse=verse, verse_end=verse_end
            )
        )
        i = j

    return references
