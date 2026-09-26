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


def test_verse_range_is_numbered():
    quote = _lookup.lookup("John", "Иоанна", 15, 1, 3)
    assert quote.ru_text.startswith("(1) Я есмь истинная виноградная лоза")
    assert " (2) " in quote.ru_text and " (3) " in quote.ru_text
    assert quote.en_text.startswith("(1) I AM the true vine")
    assert " (2) " in quote.en_text and " (3) " in quote.en_text


def test_single_verse_is_not_numbered():
    quote = _lookup.lookup("John", "Иоанна", 3, 16)
    assert not quote.ru_text.startswith("(16)")
    assert not quote.en_text.startswith("(16)")


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


def test_english_reference_uses_english_names_and_kjv_psalm_numbering():
    from sts.bible.lookup import english_reference

    assert english_reference("John", 3, 14, 15) == "John 3:14-15"
    assert english_reference("I Corinthians", 13, 4, None) == "1 Corinthians 13:4"
    assert english_reference("III John", 1, 2, None) == "3 John 1:2"
    assert english_reference("Revelation of John", 21, 4, None) == "Revelation 21:4"
    # Synodal Psalm 22 is KJV Psalm 23.
    assert english_reference("Psalms", 22, 1, None) == "Psalm 23:1"
    assert english_reference("Psalms", 22, 1, 3) == "Psalm 23:1-3"
    assert english_reference("Psalms", 1, 1, None) == "Psalm 1:1"
    # No 1:1 mapping (Synodal 9 = KJV 9 + 10): left as it is.
    assert english_reference("Psalms", 9, 5, None) == "Psalm 9:5"


def test_a_lookup_carries_its_english_reference():
    from sts.bible.lookup import BibleLookup

    quote = BibleLookup().lookup("Psalms", "Псалтирь", 22, 1)
    assert quote.en_reference == "Psalm 23:1"
    assert "Lord is my shepherd" in quote.en_text
