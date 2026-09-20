"""Russian -> English terms for Baptist/Christian vocabulary that generic
translation models tend to render inconsistently or too literally.

Terms are substituted with placeholder tokens before translation and swapped
back to their canonical English form afterward, so the translation model
never has a chance to mistranslate them. Placeholders use a form (ASCII,
digits, no spaces) that NLLB-style models are unlikely to alter in transit.

Russian is heavily inflected (grammatical case, number) so speech rarely
contains a glossary term in its dictionary form. Matching uses a crude stem
(each word minus its last 1-2 characters, plus up to 3 trailing word
characters) rather than exact strings, which is unreliable for short words
that share a stem with unrelated ones but works well for the multi-syllable
theological vocabulary this glossary targets.
"""

import re

# Russian term (lowercase) -> canonical English translation.
GLOSSARY: dict[str, str] = {
    "благодать": "grace",
    "покаяние": "repentance",
    "оправдание": "justification",
    "освящение": "sanctification",
    "искупление": "redemption",
    "крещение": "baptism",
    "вечеря господня": "the Lord's Supper",
    "причастие": "communion",
    "проповедь": "sermon",
    "пастор": "pastor",
    "прихожане": "congregation",
    "евангелие": "the Gospel",
    "спасение": "salvation",
    "воскресение": "resurrection",
    "святой дух": "the Holy Spirit",
    "господь": "the Lord",
    "иисус христос": "Jesus Christ",
    "церковь": "church",
    "молитва": "prayer",
    "грех": "sin",
    "вера": "faith",
    "библия": "the Bible",
    "священное писание": "Scripture",
}

_PLACEHOLDER = "ZZGLOSSTERMZZ{i}ZZ"
_PLACEHOLDER_RE = re.compile(r"ZZGLOSSTERMZZ(\d+)ZZ")


def _stem(word: str) -> str:
    if len(word) <= 2:
        return word
    strip = 2 if len(word) > 4 else 1
    return word[:-strip]


def _term_pattern(term: str) -> re.Pattern:
    parts = [re.escape(_stem(w)) + r"\w{0,3}" for w in term.split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.IGNORECASE)


# Longest terms first so multi-word phrases match before their substrings do.
_TERMS_BY_LENGTH = sorted(GLOSSARY.keys(), key=len, reverse=True)
_PATTERNS = [(term, _term_pattern(term)) for term in _TERMS_BY_LENGTH]


def mask(text: str) -> tuple[str, list[str]]:
    """Replaces known Russian terms (in any inflected form) with placeholders.

    Returns (masked_text, translations).
    """
    translations: list[str] = []
    for term, pattern in _PATTERNS:

        def _replace(m: re.Match, term=term) -> str:
            translations.append(GLOSSARY[term])
            return _PLACEHOLDER.format(i=len(translations) - 1)

        text = pattern.sub(_replace, text)
    return text, translations


def unmask(text: str, translations: list[str]) -> str:
    """Swaps placeholders back to their canonical English translation."""

    def _replace(m: re.Match) -> str:
        return translations[int(m.group(1))]

    return _PLACEHOLDER_RE.sub(_replace, text)
