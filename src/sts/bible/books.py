"""The 66 Bible books, indexed for spotting spoken Russian references.

Russian bible citations use one fixed short form per book (e.g. always
"Матфея 1:15", never a further-declined variant), so matching is done
against a stem of that canonical form rather than full declension. A `\\w*`
suffix is appended when matching to tolerate the odd surrounding suffix, and
the numbers gate (a reference only counts if chapter:verse numbers are found
right after the book name) keeps short/common stems from false-triggering on
unrelated words.

`number` disambiguates a series that shares one Russian stem across several
books (all four Samuel/Kings books cite as "N-я Царств"; John's three
epistles all cite as "N-е Иоанна"). `key` matches the book name used in
`sts.bible.data.bible.json.gz`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BookInfo:
    key: str  # matches the bundled dataset's book name
    ru_label: str  # short display form, e.g. "Матфея"
    stems: tuple[str, ...]  # one stem, or several for a multi-word title
    number: int | None = None  # required leading ordinal (1-4) for a shared stem


BOOKS: tuple[BookInfo, ...] = (
    BookInfo("Genesis", "Бытие", ("быти",)),
    BookInfo("Exodus", "Исход", ("исход",)),
    BookInfo("Leviticus", "Левит", ("левит",)),
    BookInfo("Numbers", "Числа", ("числ",)),
    BookInfo("Deuteronomy", "Второзаконие", ("второзакони",)),
    BookInfo("Joshua", "Иисуса Навина", ("навин",)),
    BookInfo("Judges", "Судей", ("суде",)),
    BookInfo("Ruth", "Руфь", ("руф",)),
    BookInfo("I Samuel", "Первая Царств", ("царств",), number=1),
    BookInfo("II Samuel", "Вторая Царств", ("царств",), number=2),
    BookInfo("I Kings", "Третья Царств", ("царств",), number=3),
    BookInfo("II Kings", "Четвёртая Царств", ("царств",), number=4),
    BookInfo("I Chronicles", "Первая Паралипоменон", ("паралипоменон",), number=1),
    BookInfo("II Chronicles", "Вторая Паралипоменон", ("паралипоменон",), number=2),
    BookInfo("Ezra", "Ездры", ("ездр",)),
    BookInfo("Nehemiah", "Неемии", ("неем",)),
    BookInfo("Esther", "Есфирь", ("есфир",)),
    BookInfo("Job", "Иова", ("иов",)),
    BookInfo("Psalms", "Псалтирь", ("псал",)),
    BookInfo("Proverbs", "Притчи", ("притч",)),
    BookInfo("Ecclesiastes", "Екклесиаст", ("екклесиаст",)),
    BookInfo("Song of Solomon", "Песнь Песней", ("песн", "песн")),
    BookInfo("Isaiah", "Исаии", ("исаи",)),
    BookInfo("Jeremiah", "Иеремии", ("иерем",)),
    BookInfo("Lamentations", "Плач Иеремии", ("плач", "иерем")),
    BookInfo("Ezekiel", "Иезекииля", ("иезекии",)),
    BookInfo("Daniel", "Даниила", ("даниил",)),
    BookInfo("Hosea", "Осии", ("оси",)),
    BookInfo("Joel", "Иоиля", ("иоил",)),
    BookInfo("Amos", "Амоса", ("амос",)),
    BookInfo("Obadiah", "Авдия", ("авди",)),
    BookInfo("Jonah", "Ионы", ("ион",)),
    BookInfo("Micah", "Михея", ("мих",)),
    BookInfo("Nahum", "Наума", ("наум",)),
    BookInfo("Habakkuk", "Аввакума", ("аввакум",)),
    BookInfo("Zephaniah", "Софонии", ("софони",)),
    BookInfo("Haggai", "Аггея", ("агге",)),
    BookInfo("Zechariah", "Захарии", ("захари",)),
    BookInfo("Malachi", "Малахии", ("малахи",)),
    BookInfo("Matthew", "Матфея", ("матфе",)),
    BookInfo("Mark", "Марка", ("марк",)),
    BookInfo("Luke", "Луки", ("лук",)),
    BookInfo("John", "Иоанна", ("иоанн",)),
    BookInfo("Acts", "Деяния", ("деяни",)),
    BookInfo("Romans", "Римлянам", ("римлян",)),
    BookInfo("I Corinthians", "1-е Коринфянам", ("коринф",), number=1),
    BookInfo("II Corinthians", "2-е Коринфянам", ("коринф",), number=2),
    BookInfo("Galatians", "Галатам", ("галат",)),
    BookInfo("Ephesians", "Ефесянам", ("ефесян",)),
    BookInfo("Philippians", "Филиппийцам", ("филиппи",)),
    BookInfo("Colossians", "Колоссянам", ("колосс",)),
    BookInfo("I Thessalonians", "1-е Фессалоникийцам", ("фессалоник",), number=1),
    BookInfo("II Thessalonians", "2-е Фессалоникийцам", ("фессалоник",), number=2),
    BookInfo("I Timothy", "1-е Тимофею", ("тимофе",), number=1),
    BookInfo("II Timothy", "2-е Тимофею", ("тимофе",), number=2),
    BookInfo("Titus", "Титу", ("тит",)),
    BookInfo("Philemon", "Филимону", ("филимон",)),
    BookInfo("Hebrews", "Евреям", ("евре",)),
    BookInfo("James", "Иакова", ("иаков",)),
    BookInfo("I Peter", "1-е Петра", ("петр",), number=1),
    BookInfo("II Peter", "2-е Петра", ("петр",), number=2),
    BookInfo("I John", "1-е Иоанна", ("иоанн",), number=1),
    BookInfo("II John", "2-е Иоанна", ("иоанн",), number=2),
    BookInfo("III John", "3-е Иоанна", ("иоанн",), number=3),
    BookInfo("Jude", "Иуды", ("иуд",)),
    BookInfo("Revelation of John", "Откровение", ("открове",)),
)

BOOKS_BY_KEY: dict[str, BookInfo] = {book.key: book for book in BOOKS}

# Longest (most specific) stem sequences first, so e.g. "Плач Иеремии"
# (Lamentations) is tried before the bare "Иеремии" (Jeremiah) stem.
BOOKS_BY_SPECIFICITY: tuple[BookInfo, ...] = tuple(
    sorted(BOOKS, key=lambda b: len(b.stems), reverse=True)
)
