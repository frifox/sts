from sts.textclean import clean_transcript, collapse_repeats


def test_short_word_stutter_collapses():
    assert clean_transcript("Я я я думаю, что это так.") == "я думаю, что это так."
    assert clean_transcript("Мы в в церкви.") == "Мы в церкви."


def test_phrase_restart_collapses():
    assert clean_transcript("Я думаю я думаю что надо идти.") == "я думаю что надо идти."


def test_deliberate_emphasis_kept():
    assert clean_transcript("Истинно, истинно говорю вам.") == "Истинно, истинно говорю вам."
    assert clean_transcript("Свят, свят, свят Господь.") == "Свят, свят, свят Господь."


def test_runaway_loop_capped():
    assert collapse_repeats(" ".join(["once"] * 100)) == "once once"
    assert collapse_repeats("Тогда " + " ".join(["снова,"] * 50) + " всё.") == "Тогда снова, снова, всё."


def test_fragments_removed():
    assert clean_transcript("Христос гово- говорит нам.") == "Христос говорит нам."
    assert clean_transcript("Нужно по-по-покаяние.") == "Нужно покаяние."
    assert clean_transcript("Он пре... преступил закон.") == "Он преступил закон."


def test_real_hyphenated_words_kept():
    assert clean_transcript("Он сказал по-русски, кто-то слышал.") == "Он сказал по-русски, кто-то слышал."


def test_fillers_removed():
    assert clean_transcript("Ну, э-э, вот мы э пришли, ммм сюда.") == "Ну, вот мы пришли, сюда."


def test_normal_text_untouched():
    text = "Иисус серьезно предупреждает. Нет, здесь Христос говорит: всякую ветвь."
    assert clean_transcript(text) == text


def test_asr_drops_prompt_echo():
    from sts.asr.whisper_asr import is_hallucination
    from sts.config import ASRConfig

    cfg = ASRConfig()
    assert is_hallucination("Библия, церковь, молитва, благодать, покаяние, грех.", 0.1, -0.2, cfg)
    assert not is_hallucination("Молитва есть разговор с Богом.", 0.1, -0.2, cfg)
    assert not is_hallucination("Грех.", 0.1, -0.2, cfg)  # single real word is fine


def test_xtts_chunking():
    import pytest

    pytest.importorskip("TTS")  # this venv may only have the qwen3 backend installed
    from sts.tts.xtts import split_for_xtts

    assert split_for_xtts("Short one.") == ["Short one."]
    three = "Christians live fruitful lives. You need to repent. Jesus speaks, and we should listen."
    assert split_for_xtts(three) == [
        "Christians live fruitful lives.",
        "You need to repent.",
        "Jesus speaks, and we should listen.",
    ]  # one generation per sentence, not packed together
    long = "of course we live in a time when they say " + ", ".join(["once"] * 200)
    chunks = split_for_xtts(long)
    assert all(len(c) <= 220 for c in chunks) and len(chunks) > 1
    assert split_for_xtts("  ") == []


def test_split_sentences():
    from sts.translate.translator import split_sentences

    assert split_sentences("Иисус говорит. Так и есть. Да.") == ["Иисус говорит.", "Так и есть. Да."]
    assert split_sentences("Нужно покаяние. Аминь.") == ["Нужно покаяние. Аминь."]
    assert split_sentences("Одно. Нужно покаяние. Так.") == ["Одно.", "Нужно покаяние. Так."]
    assert split_sentences("Иисус предупреждает нас. Здесь речь не идет о ветвях.") == [
        "Иисус предупреждает нас.",
        "Здесь речь не идет о ветвях.",
    ]
    assert split_sentences("  ") == []


def test_known_one_word_replies_are_translated_exactly():
    from sts.translate.translator import short_phrase

    assert short_phrase("Аминь.") == "Amen."
    assert short_phrase("  аминь!  ") == "Amen."
    assert short_phrase("Слава Богу...") == "Praise God."
    assert short_phrase("Аминь, аминь.") is None          # not in the table: left to the model
    assert short_phrase("Аминь всем нам.") is None
    assert short_phrase("") is None


def test_translate_sentences_yields_incrementally_and_translate_still_joins():
    """translate_sentences() must be a real generator (not build a list first) so a
    caller can act on sentence 1 before sentence 2 is even translated; translate()
    is unchanged for its existing callers - still one joined string."""
    from sts.translate.translator import Translator

    translator = Translator.__new__(Translator)  # skip __init__: no real NLLB model needed for this
    calls = []

    def fake_translate_sentence(text):
        calls.append(text)
        return text.upper()

    translator._translate_sentence = fake_translate_sentence

    text = "One thing happened. Two more things happened. Three final things happened."
    generator = translator.translate_sentences(text)
    assert calls == []  # nothing translated yet - it's lazy until iterated

    first = next(generator)
    assert first == "ONE THING HAPPENED." and calls == ["One thing happened."]  # only the first sentence ran

    remaining = list(generator)
    assert remaining == ["TWO MORE THINGS HAPPENED.", "THREE FINAL THINGS HAPPENED."]
    assert calls == ["One thing happened.", "Two more things happened.", "Three final things happened."]

    assert translator.translate(text) == first + " " + " ".join(remaining)
    assert translator.translate("  ") == ""
    assert list(translator.translate_sentences("  ")) == []


def test_split_complete():
    from sts.textclean import split_complete

    assert split_complete("Привет. Как дела") == ("Привет.", "Как дела")
    assert split_complete("Привет. Как дела сегодня") == ("Привет.", "Как дела сегодня")
    assert split_complete("Одно предложение без точки в конце") == ("", "Одно предложение без точки в конце")
    assert split_complete("") == ("", "")
    assert split_complete("   ") == ("", "")
    assert split_complete("Только одно.") == ("Только одно.", "")
    assert split_complete("А? Б! В...") == ("А? Б! В...", "")
    # only the LAST terminator matters - everything up to it is "complete"
    assert split_complete("Первое. Второе. третье без точки") == ("Первое. Второе.", "третье без точки")


def test_restore_unk_quotes():
    from sts.textclean import restore_unk_quotes

    assert restore_unk_quotes(
        "И вот в этом разделе первым пунктом я называю ⁇ Пренесение плода "
        "свидетельствует о нашей связи со Христом ⁇."
    ) == (
        "И вот в этом разделе первым пунктом я называю "
        '"Пренесение плода свидетельствует о нашей связи со Христом".'
    )
    assert restore_unk_quotes("нет кавычек тут") == "нет кавычек тут"
    assert restore_unk_quotes("") == ""
    assert restore_unk_quotes("А ⁇ первая ⁇, потом ⁇ вторая ⁇ фраза.") == 'А "первая", потом "вторая" фраза.'
    # an unpaired marker (e.g. a segment boundary cut mid-quote) still becomes
    # a quote character rather than staying as the raw placeholder
    assert restore_unk_quotes("⁇ только открывающая") == '"только открывающая'
