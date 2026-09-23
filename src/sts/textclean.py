"""Cleans up disfluencies in transcribed speech before it is translated:
stutters ("я я думаю"), abandoned word fragments ("гово- говорит",
"по-по-покаяние"), filler sounds ("э-э", "ммм"), and runaway repetition.

A speaker who restarts a word or swaps it mid-way leaves both versions in the
transcript, and the translator then faithfully translates the mess. These
rules only remove what is safely recognizable as a disfluency; they can't
know that a speaker *meant* a different word than the one they abandoned.
"""

import re

# A truncated word followed by the word it was heading for: "гово- говорит",
# "по-по-покаяние", "пре... преступление". The fragment must be a prefix of
# what follows, which keeps real hyphenated words ("по-русски", "кто-то").
_FRAGMENT = re.compile(r"\b(\w{1,8})(?:-|…|\.\.\.)\s*(?=\1)", re.IGNORECASE)

# Filler sounds only ("э", "э-э", "эм", "мм", "гм", "хм"), never real words.
_FILLER = re.compile(r"\b(?:э(?:-э)*|э+м*|м{2,}|гм+|хм+)\b[,.]?\s*", re.IGNORECASE)

_WORD_RE = re.compile(r"\S+")
_STRIP_RE = re.compile(r"^\W+|\W+$")

# Single words this short are stutter ("я я думаю"); longer ones repeated are
# usually deliberate emphasis ("истинно, истинно говорю", "свят, свят, свят").
_STUTTER_MAX_LETTERS = 3
# Anything repeated more often than this is a stuck loop, not emphasis.
_MAX_EMPHASIS_REPEATS = 3
_LOOP_CAP = 2


def _norm(token: str) -> str:
    return _STRIP_RE.sub("", token).lower()


def collapse_repeats(text: str) -> str:
    """Collapses stutters and loops: runs of a repeated word/phrase."""
    tokens = _WORD_RE.findall(text)
    for n in (4, 3, 2, 1):
        i = 0
        out: list[str] = []
        while i < len(tokens):
            gram = [_norm(t) for t in tokens[i : i + n]]
            # A phrase made of one word repeated is the n=1 case's job.
            if len(gram) < n or not all(gram) or (n > 1 and len(set(gram)) == 1):
                out.append(tokens[i])
                i += 1
                continue
            run = 1
            while [_norm(t) for t in tokens[i + run * n : i + (run + 1) * n]] == gram:
                run += 1
            if run == 1:
                out.append(tokens[i])
                i += 1
                continue
            stutter = n > 1 or len(gram[0]) <= _STUTTER_MAX_LETTERS
            keep = 1 if stutter else (run if run <= _MAX_EMPHASIS_REPEATS else _LOOP_CAP)
            # Keep the *last* copy's tokens so trailing punctuation survives.
            last = tokens[i + (run - 1) * n : i + run * n]
            for k in range(keep - 1):
                out.extend(tokens[i + k * n : i + (k + 1) * n])
            out.extend(last)
            i += run * n
        tokens = out
    return " ".join(tokens)


def clean_transcript(text: str) -> str:
    text = _FILLER.sub("", text)
    previous = None
    while previous != text:  # "по-по-покаяние" needs more than one pass
        previous = text
        text = _FRAGMENT.sub("", text)
    text = collapse_repeats(text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.;:])\1+", r"\1", text)
    return text.strip()


def split_for_tts(text: str, limit: int) -> list[str]:
    """One piece per sentence, so each gets its own natural intonation and can
    be checked and retried on its own. Only a sentence longer than `limit`
    characters is cut further: at commas/semicolons first, then at spaces.
    Nothing is dropped."""
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sentence:
            continue
        current = ""
        for clause in re.split(r"(?<=[,;:])\s+", sentence):
            for word in clause.split():
                if current and len(current) + 1 + len(word) > limit:
                    pieces.append(current)
                    current = word
                else:
                    current = f"{current} {word}".strip()
            if current and len(current) > limit * 0.6 and clause.endswith((",", ";", ":")):
                pieces.append(current)
                current = ""
        if current:
            pieces.append(current)
    return pieces


# Matches the *last* sentence-ending punctuation in a string, however much
# text follows it (used to split "confirmed" text from a still-growing tail).
_LAST_SENTENCE_END_RE = re.compile(r"[.!?…](?!.*[.!?…])")


def split_complete(text: str) -> tuple[str, str]:
    """(complete, tail): `text` through its last sentence-ending punctuation,
    and whatever trails after it. Used when re-transcribing a still-growing
    utterance periodically (rather than waiting for it to finish): the
    `complete` part won't grow any more sentences before it, so it's safe to
    commit; `tail` is the not-yet-finished remainder, left to grow on the next
    pass. `complete` is "" if there's no sentence-ending punctuation yet."""
    match = _LAST_SENTENCE_END_RE.search(text)
    if not match:
        return "", text.strip()
    return text[: match.end()].strip(), text[match.end() :].strip()
