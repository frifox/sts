from sts.bible.lookup import BibleLookup

_lookup = BibleLookup()


def test_matthew_1_15():
    quote = _lookup.lookup("Matthew", "Матфея", 1, 15)
    assert "Иаков" in quote.ru_text
    assert "Jacob" in quote.en_text


def test_verse_range_joins_text():
    quote = _lookup.lookup("Lamentations", "Плач Иеремии", 3, 22, 23)
    assert "милость" in quote.ru_text or "милости" in quote.ru_text
    assert "morning" in quote.en_text


def test_psalm_numbering_is_converted_between_synodal_and_kjv():
    # Synodal Psalm 22 (LXX numbering) == KJV Psalm 23 ("The Lord is my shepherd").
    quote = _lookup.lookup("Psalms", "Псалтирь", 22, 1)
    assert "Пастырь" in quote.ru_text
    assert "shepherd" in quote.en_text.lower()


def test_psalm_without_a_clean_mapping_has_no_english_text():
    # Synodal Psalm 114 has no 1:1 KJV counterpart (splits/merges across
    # Psalms 114-116 in the Masoretic numbering).
    quote = _lookup.lookup("Psalms", "Псалтирь", 114, 1)
    assert quote.ru_text is not None
    assert quote.en_text is None


def test_pilcrow_paragraph_markers_are_stripped():
    quote = _lookup.lookup("John", "Иоанна", 3, 16)
    assert not quote.en_text.startswith("¶")
