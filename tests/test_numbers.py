"""Numbers are spelled out in English before voicing (a voice cloned from a
Russian speaker otherwise reads digits in Russian)."""

from sts.tts.numbers import ordinal_words, spell_numbers


def test_numbers_in_sermon_text():
    assert spell_numbers("And again in verse 6, the Lord says.") == "And again in verse six, the Lord says."
    assert spell_numbers("John 3:16-18.") == "John three sixteen to eighteen."
    assert spell_numbers("Chapter 15, verses 1-8.") == "Chapter fifteen, verses one to eight."
    assert spell_numbers("Psalm 119 has 176 verses.") == "Psalm one hundred nineteen has one hundred seventy-six verses."


def test_years_amounts_and_ordinals():
    assert spell_numbers("In 1990 and 2024, 2005.") == "In nineteen ninety and twenty twenty-four, two thousand five."
    assert spell_numbers("1,500 people, 100%, 2.5 hours") == "one thousand five hundred people, one hundred percent, two point five hours"
    assert spell_numbers("the 1st, 3rd and 21st") == "the first, third and twenty-first"
    assert [ordinal_words(n) for n in (12, 40, 100)] == ["twelfth", "fortieth", "one hundredth"]


def test_text_without_numbers_is_unchanged():
    assert spell_numbers("I am the vine, you are the branches.") == "I am the vine, you are the branches."
