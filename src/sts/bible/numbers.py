"""Parses Russian number words (cardinal and ordinal, e.g. "пятнадцать" /
"пятнадцатый") into integers, for chapter/verse numbers a speaker says as
words instead of reading as digits.

Whisper normalizes most spoken numbers to digits in its transcript already,
so digit parsing (in `sts.bible.reference`) is the primary path and this is
a fallback. It covers 0-176 (enough for every chapter and verse number in
the Bible, e.g. Psalm 119's 176 verses) but doesn't chase every irregular
ordinal ending for the larger tens (50-90) - those are rarely spoken as
ordinals for a chapter/verse number anyway.

Matching is stem-based (see `sts.translate.glossary` for the same
technique): a word matches if it starts with a known stem and has no more
than 3 trailing characters left over, which lets one stem cover a word's
cardinal, ordinal, and cased/declined forms.
"""

_UNITS: dict[str, int] = {
    "ноль": 0,
    "один": 1,
    "одна": 1,
    "одно": 1,
    "перв": 1,
    "два": 2,
    "две": 2,
    "втор": 2,
    "три": 3,
    "трет": 3,
    "четыр": 4,
    "четверт": 4,
    "пят": 5,
    "шест": 6,
    "седьм": 7,  # седьмой, седьмое, седьмых - "сед" alone left too much of the
                 # word ("ьмой") beyond the 3-char declension tolerance below
    "сем": 7,  # семь
    "восем": 8,
    "восьм": 8,
    "девят": 9,
}

_TEENS: dict[str, int] = {
    "десят": 10,
    "одиннадцат": 11,
    "двенадцат": 12,
    "тринадцат": 13,
    "четырнадцат": 14,
    "пятнадцат": 15,
    "шестнадцат": 16,
    "семнадцат": 17,
    "восемнадцат": 18,
    "девятнадцат": 19,
}

_TENS: dict[str, int] = {
    "двадцат": 20,
    "тридцат": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяност": 90,
}

_HUNDREDS: dict[str, int] = {
    "сто": 100,
    "сот": 100,
}

_ALL_STEMS: list[tuple[str, int, str]] = sorted(
    (
        [(stem, value, "hundred") for stem, value in _HUNDREDS.items()]
        + [(stem, value, "ten") for stem, value in _TENS.items()]
        + [(stem, value, "teen") for stem, value in _TEENS.items()]
        + [(stem, value, "unit") for stem, value in _UNITS.items()]
    ),
    key=lambda entry: len(entry[0]),
    reverse=True,
)

_MAX_SUFFIX = 3


def _match_word(token: str) -> tuple[int, str] | None:
    for stem, value, category in _ALL_STEMS:
        if token.startswith(stem) and len(token) - len(stem) <= _MAX_SUFFIX:
            return value, category
    return None


def parse_number_words(tokens: list[str], start: int) -> tuple[int, int] | None:
    """Reads a compositional number (hundred? + ten? + unit-or-teen?) starting
    at `tokens[start]`. Returns (value, tokens_consumed), or None if
    `tokens[start]` isn't a number word at all.
    """
    idx = start
    total = 0
    consumed = 0

    match = _match_word(tokens[idx]) if idx < len(tokens) else None
    if match and match[1] == "hundred":
        total += match[0]
        idx += 1
        consumed += 1
        match = _match_word(tokens[idx]) if idx < len(tokens) else None

    if match and match[1] == "ten":
        total += match[0]
        idx += 1
        consumed += 1
        match = _match_word(tokens[idx]) if idx < len(tokens) else None

    if match and match[1] in ("unit", "teen"):
        total += match[0]
        consumed += 1

    if consumed == 0:
        return None
    return total, consumed
