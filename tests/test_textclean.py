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
    from sts.tts.xtts import split_for_xtts

    assert split_for_xtts("Short one.") == ["Short one."]
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
