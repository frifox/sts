"""Spells out numbers in English text before it's voiced ("verse 6" ->
"verse six"). A voice cloned from a Russian speaker reads bare digits the
way its reference would: in Russian. Words leave the model nothing to guess.
"""

import re

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"))
_ORDINAL_ENDINGS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                    "nine": "ninth", "twelve": "twelfth"}


def number_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + number_words(n % 100) if n % 100 else "")
    for size, name in _SCALES:
        if n >= size:
            return number_words(n // size) + " " + name + (" " + number_words(n % size) if n % size else "")
    return str(n)


def ordinal_words(n: int) -> str:
    words = number_words(n)
    cut = max(words.rfind(" "), words.rfind("-")) + 1
    head, last = words[:cut], words[cut:]
    if last in _ORDINAL_ENDINGS:
        last = _ORDINAL_ENDINGS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + last


def _year_words(n: int) -> str:
    """1990 -> "nineteen ninety", 1905 -> "nineteen oh five", 2000 -> "two thousand", 2024 -> "twenty twenty-four"."""
    high, low = divmod(n, 100)
    if low == 0:
        return number_words(n) if n % 1000 == 0 else number_words(high) + " hundred"
    if n < 2010 and n >= 2000:
        return number_words(n)
    return number_words(high) + " " + (("oh " + _ONES[low]) if low < 10 else number_words(low))


_NUMBER = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)"          # 1,000 or 1000
    r"(?:(\.\d+)|(st|nd|rd|th)\b|(:)(\d+)|(%))?"   # .5 | 1st | 3:16 | 50%
)


_RANGE = re.compile(r"(?<![\w.,])(\d+)\s*[-–]\s*(\d+)(?![\w.,:]\d)")  # verses 1-8


def spell_numbers(text: str) -> str:
    text = _RANGE.sub(lambda m: f"{m.group(1)} to {m.group(2)}", text)

    def spell(match: re.Match) -> str:
        whole, decimals, ordinal, colon, after_colon, percent = match.groups()
        n = int(whole.replace(",", ""))
        if ordinal:
            return ordinal_words(n)
        if colon:  # chapter:verse (John 3:16) or a time (6:30): read as two numbers
            return f"{number_words(n)} {number_words(int(after_colon))}"
        if decimals:
            return number_words(n) + " point " + " ".join(_ONES[int(d)] for d in decimals[1:])
        words = _year_words(n) if len(whole) == 4 and 1100 <= n <= 2099 else number_words(n)
        return words + (" percent" if percent else "")

    return _NUMBER.sub(spell, text)
