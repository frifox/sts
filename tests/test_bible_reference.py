from sts.bible.reference import find_references


def _single(text):
    refs = find_references(text)
    assert len(refs) == 1, f"expected exactly one reference in {text!r}, got {refs}"
    return refs[0]


def test_book_colon_form():
    ref = _single("Откроем от Матфея 1:15.")
    assert (ref.book, ref.chapter, ref.verse) == ("Matthew", 1, 15)


def test_book_chapter_verse_keywords():
    ref = _single("Сегодня мы откроем Евангелие от Матфея, глава 1, стих 15.")
    assert (ref.book, ref.chapter, ref.verse) == ("Matthew", 1, 15)


def test_verse_range_with_hyphen():
    ref = _single("Также вспомним Плач Иеремии 3:22-23.")
    assert (ref.book, ref.chapter, ref.verse, ref.verse_end) == ("Lamentations", 3, 22, 23)


def test_numbered_book_with_digit_prefix():
    ref = _single("1-е Иоанна 4:8.")
    assert (ref.book, ref.chapter, ref.verse) == ("I John", 4, 8)


def test_numbered_book_with_full_phrase():
    ref = _single("Первое послание к Коринфянам, глава тринадцать, стих четыре.")
    assert (ref.book, ref.chapter, ref.verse) == ("I Corinthians", 13, 4)


def test_bare_gospel_of_john_not_confused_with_epistle():
    ref = _single("Читаем Иоанна 3:16.")
    assert ref.book == "John"


def test_spelled_out_numbers():
    ref = _single("Иакова глава один стих двадцать два.")
    assert (ref.book, ref.chapter, ref.verse) == ("James", 1, 22)


def test_lamentations_preferred_over_bare_jeremiah():
    ref = _single("Плач Иеремии 3:22.")
    assert ref.book == "Lamentations"


def test_bare_jeremiah_without_lamentations_prefix():
    ref = _single("Иеремии 3:22.")
    assert ref.book == "Jeremiah"


def test_no_reference_in_plain_speech():
    assert find_references("Просто разговор без ссылок на Библию.") == []


def test_book_name_alone_without_numbers_is_not_a_reference():
    assert find_references("Сегодня мы говорим об Иакове и его вере.") == []


def test_verse_range_with_spoken_from_keyword():
    """"стихи с 5 по 7" - the "с" ("from") before the range's starting verse
    must be skipped, not treated as breaking the number that follows it."""
    ref = _single("Евангелие от Иоанна, глава 15, стихи с 5 по 7.")
    assert (ref.book, ref.chapter, ref.verse, ref.verse_end) == ("John", 15, 5, 7)


def test_verse_range_with_spoken_from_keyword_and_word_numbers():
    ref = _single("От Иоанна глава пятнадцать стихи с пятого по седьмое.")
    assert (ref.book, ref.chapter, ref.verse, ref.verse_end) == ("John", 15, 5, 7)


def test_ordinal_seven_is_parsed():
    """Regression: the "сед" stem alone left "ьмой"/"ьмое" (4 chars) beyond
    the declension tolerance, so no spoken form of "seventh" ever matched."""
    ref = _single("От Иоанна глава пятнадцать стих седьмой.")
    assert (ref.book, ref.chapter, ref.verse) == ("John", 15, 7)


def test_first_n_verses_stated_before_the_book():
    """Regression: "N первых стихов в M главе [Книга]" puts the book name
    *last*, after the verse count and chapter - the reverse of every other
    supported phrasing. Means verses 1 through N, not a single verse N."""
    ref = _single(
        "В начале своей проповеди я прочитаю 8 первых стихов в 15 главе Евангелия от Иоанна."
    )
    assert (ref.book, ref.chapter, ref.verse, ref.verse_end) == ("John", 15, 1, 8)


def test_first_n_verses_with_first_word_before_the_count():
    ref = _single("Я прочитаю первых 8 стихов в 15 главе Иоанна.")
    assert (ref.book, ref.chapter, ref.verse, ref.verse_end) == ("John", 15, 1, 8)


def test_first_n_verses_with_word_numbers():
    ref = _single("Прочитаю восемь первых стихов в пятнадцатой главе Иоанна.")
    assert (ref.book, ref.chapter, ref.verse, ref.verse_end) == ("John", 15, 1, 8)


def test_first_one_verse_has_no_verse_end():
    """"1 первый стих" (count of exactly 1) shouldn't render as a "1-1" range."""
    ref = _single("Прочитаю 1 первый стих 15 главы Иоанна.")
    assert (ref.book, ref.chapter, ref.verse, ref.verse_end) == ("John", 15, 1, None)


def test_bare_verse_count_without_first_is_not_a_reference():
    """"8 стихов" alone doesn't say which 8 verses are meant (could start
    anywhere) - showing a guess would risk displaying the wrong passage, so
    this is deliberately left undetected rather than assumed to mean 1-8."""
    assert find_references("Прочитаю 8 стихов в 15 главе Иоанна.") == []
