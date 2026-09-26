"""sts-caption's "x to clear" key handling."""

import io
import os
import sys
import threading

import pytest

from sts.caption import watch_keys

termios = pytest.importorskip("termios")  # the pty below is Unix-only


def test_keys_arrive_without_enter_and_the_terminal_is_restored(monkeypatch):
    master, slave = os.openpty()
    stdin = io.TextIOWrapper(io.FileIO(slave, "r", closefd=False), encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin)
    before = termios.tcgetattr(slave)
    keys, got_x = [], threading.Event()

    def on_key(key):
        keys.append(key)
        if key == "x":
            got_x.set()

    restore = watch_keys(on_key)
    assert termios.tcgetattr(slave) != before  # cbreak: no line buffering
    os.write(master, b"ax")  # no newline
    assert got_x.wait(2)
    assert keys == ["a", "x"]

    restore()
    assert termios.tcgetattr(slave) == before
    os.close(master)
    os.close(slave)


def test_does_nothing_without_a_terminal(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("x"))
    calls = []
    watch_keys(calls.append)()
    assert calls == []


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def _size(monkeypatch, columns, lines):
    monkeypatch.setattr("sts.caption.shutil.get_terminal_size", lambda: os.terminal_size((columns, lines)))


def _screen(raw: str, columns: int, lines: int) -> list[str]:
    """Plays `raw` terminal output (\\r, \\n, cursor-up, clear-to-end) onto a
    `lines`-row screen with scrolling, and returns what's visible."""
    import re

    screen, row, col = [""], 0, 0
    for token in re.findall(r"\x1b\[(\d*)A|\x1b\[J|(\r)|(\n)|([^\x1b\r\n])", raw):
        up, cr, nl, ch = token
        if up:
            row = max(0, row - int(up))
        elif cr:
            col = 0
        elif nl:
            row += 1
            if row == len(screen):
                screen.append("")
            if len(screen) > lines:
                screen.pop(0)
                row -= 1
            col = 0
        elif ch:
            if col == columns:  # deferred wrap
                row, col = row + 1, 0
                if row == len(screen):
                    screen.append("")
                if len(screen) > lines:
                    screen.pop(0)
                    row -= 1
            screen[row] = screen[row][:col] + ch
            col += 1
        else:  # clear to end of screen
            screen[row] = screen[row][:col]
            del screen[row + 1 :]
    return screen


def test_one_sentence_per_line():
    from sts.caption import one_sentence_per_line

    assert one_sentence_per_line("Братья и сестры. Вот так! Почему? И") == "Братья и сестры.\nВот так!\nПочему?\nИ"
    assert one_sentence_per_line("Кто пребывает во мне.") == "Кто пребывает во мне."  # nothing after it yet
    assert one_sentence_per_line("стих 3.5 и т.д") == "стих 3.5 и т.д"  # no space after the dot: not an end
    assert one_sentence_per_line("это «необходимо». Всё") == "это «необходимо».\nВсё"


def test_sentence_breaks_that_appear_and_disappear_are_redrawn(monkeypatch):
    from sts.caption import LiveLine

    _size(monkeypatch, 40, 24)
    out = _Terminal()
    line = LiveLine(out)
    line.update("Вопрос. Как нам быть")
    assert _screen(out.getvalue(), 40, 24) == ["Вопрос.", "Как нам быть"]
    line.update("Возникает вопрос, как нам быть?")  # the break went away on revision
    assert _screen(out.getvalue(), 40, 24) == ["Возникает вопрос, как нам быть?"]
    line.update("Возникает вопрос, как нам быть? Ответ")
    assert _screen(out.getvalue(), 40, 24) == ["Возникает вопрос, как нам быть?", "Ответ"]
    line.update("Возникает вопрос, как нам быть? Ответ")  # unchanged: nothing redrawn
    before = out.getvalue()
    line.update("Возникает вопрос, как нам быть? Ответ")
    assert out.getvalue() == before


def test_long_sentences_wrap_and_are_redrawn_cleanly(monkeypatch):
    from sts.caption import LiveLine

    _size(monkeypatch, 20, 24)
    out = _Terminal()
    line = LiveLine(out)
    line.update("Братья и сестры, вот так говорил. Наш")
    line.update("Братья и сестры, вот так говорил наш Господь.")
    assert _screen(out.getvalue(), 20, 24) == ["Братья и сестры, вот", " так говорил наш Гос", "подь."]


def test_a_block_taller_than_the_screen_redraws_what_is_still_visible(monkeypatch):
    from sts.caption import LiveLine

    _size(monkeypatch, 40, 4)
    out = _Terminal()
    line = LiveLine(out)
    sentences = [f"Предложение {i}." for i in range(1, 7)]
    line.update(" ".join(sentences))
    line.update(" ".join(sentences[:-1] + ["Предложение шесть!", "Семь"]))
    assert _screen(out.getvalue(), 40, 4) == ["Предложение 4.", "Предложение 5.", "Предложение шесть!", "Семь"]


def test_clearing_leaves_the_old_block_and_starts_below(monkeypatch):
    from sts.caption import LiveLine

    _size(monkeypatch, 40, 24)
    out = _Terminal()
    line = LiveLine(out)
    line.update("Первый. Второй")
    line.finish()
    out.write("[caption] cleared\n")
    line.update("Новое")
    assert _screen(out.getvalue(), 40, 24) == ["Первый.", "Второй", "[caption] cleared", "Новое"]


def test_piped_output_is_only_final_text_plus_what_is_open_at_the_end():
    from sts.caption import LiveLine

    out = io.StringIO()  # not a terminal
    line = LiveLine(out)
    line.update("\nА. Б")
    line.update("\nА. Б, В")
    line.update("\nБ, В. Г", above=["А."])  # "А." became final
    line.finish()
    assert out.getvalue() == "А.\nБ, В.\nГ\n"


def test_other_output_goes_above_the_block_and_the_block_is_redrawn_below(monkeypatch):
    from sts.caption import LiveLine, _AboveLiveLine

    _size(monkeypatch, 40, 24)
    out = _Terminal()
    line = LiveLine(out)
    line.update("Первый. Второй")
    other = _AboveLiveLine(line)
    other.write("[audio] input status: ")  # arrives in pieces, like print() writes it
    other.write("input overflow\n")
    line.update("Первый. Второй, третий")
    assert _screen(out.getvalue(), 40, 24) == ["[audio] input status: input overflow", "Первый.", "Второй, третий"]


def test_other_output_while_the_block_is_taller_than_the_screen(monkeypatch):
    from sts.caption import LiveLine

    _size(monkeypatch, 40, 4)
    out = _Terminal()
    line = LiveLine(out)
    sentences = [f"Предложение {i}." for i in range(1, 7)]
    line.update(" ".join(sentences))
    line.print_above("[audio] warning")
    line.update(" ".join(sentences + ["Семь"]))
    assert _screen(out.getvalue(), 40, 4) == ["Предложение 4.", "Предложение 5.", "Предложение 6.", "Семь"]
    line.update(" ".join(sentences[:-1] + ["Шесть!", "Семь"]))
    assert _screen(out.getvalue(), 40, 4) == ["Предложение 4.", "Предложение 5.", "Шесть!", "Семь"]


# -- sentences and what's final ------------------------------------------------


def _words(*items):
    from sts.asr import TimedWord

    return [TimedWord(text, start, end) for text, start, end in items]


def test_sentences_are_cut_at_end_marks_and_the_open_tail_is_incomplete():
    from sts.transcript import sentences_from_words

    sentences = sentences_from_words(_words(
        ("Я", 0.0, 0.2), ("есмь", 0.3, 0.6), ("лоза.", 0.7, 1.0),
        ("Кто", 1.5, 1.7), ("пребывает", 1.8, 2.3), ("во", 2.4, 2.5), ("мне?", 2.6, 2.9),
        ("И", 3.5, 3.6), ("я", 3.7, 3.8),
    ))
    assert [(x.text, x.start, x.end, x.complete) for x in sentences] == [
        ("Я есмь лоза.", 0.0, 1.0, True),
        ("Кто пребывает во мне?", 1.5, 2.9, True),
        ("И я", 3.5, 3.8, False),
    ]


def test_a_sentence_never_starts_with_punctuation():
    from sts.transcript import sentences_from_words

    sentences = sentences_from_words(_words((".способны", 0.0, 0.5), ("приносить.", 0.6, 1.0), (".", 1.2, 1.3), ("Ну", 1.5, 1.7)))
    assert [x.text for x in sentences] == ["способны приносить.", "Ну"]


def test_locked_by_text_after_it_by_being_unchanged_or_by_force():
    from sts.transcript import Sentence, count_final

    s = [Sentence("A.", 0, 2, True), Sentence("B.", 3, 5, True), Sentence("C", 6, 7, False)]
    changing = [9.0, 9.0, 9.0]  # every text just changed at now_s=9
    # Text after it is enough, complete or not: A and B; the last one waits.
    assert count_final(s, now_s=9, unchanged_since=changing, after_unchanged=5, force_after=30) == 2
    # The last one, unchanged for 5 s.
    assert count_final(s, now_s=14, unchanged_since=[9.0, 9.0, 9.0], after_unchanged=5, force_after=30) == 3
    assert count_final(s, now_s=13.9, unchanged_since=[9.0, 9.0, 9.0], after_unchanged=5, force_after=30) == 2
    # Still changing, but 30 s since its first word (at 6).
    assert count_final(s, now_s=36, unchanged_since=[36.0] * 3, after_unchanged=5, force_after=30) == 3
    # 0 turns the time rules off; the text-after rule always applies.
    assert count_final(s, now_s=100, unchanged_since=[0.0] * 3, after_unchanged=0, force_after=0) == 2
    # Only a leading run: a sentence that isn't final keeps everything after it open.
    lone = [Sentence("A", 0, 20, False)]
    assert count_final(lone, now_s=21, unchanged_since=[21.0], after_unchanged=5, force_after=30) == 0


class _ScriptedAsr:
    """Recognizes a fixed script of timed words. The fake audio's samples
    hold their own absolute sample index, so this knows which stretch of
    the recording it was handed - the way a real model would hear it.
    `revise(text, heard_from_s)` can change a word depending on where the
    audio it was given starts (context changes what a model hears)."""

    def __init__(self, script, revise=None):
        self.script = script
        self.revise = revise or (lambda text, heard_from_s: text)
        self.heard = []

    def transcribe_words(self, audio):
        from sts.asr import TimedWord

        begin, end = audio[0] / 16_000, (audio[-1] + 1) / 16_000
        self.heard.append((begin, end))
        return [TimedWord(self.revise(w.text, begin), w.start - begin, w.end - begin)
                for w in self.script if w.start >= begin - 0.01 and w.end <= end]


def _texts(step_result):
    final, still_open = step_result
    return [sentence.text for sentence in final], still_open


def _feed(transcript, until_s, fed):
    import numpy as np

    samples = np.arange(fed[0], int(until_s * 16_000), dtype=np.float32)
    fed[0] = int(until_s * 16_000)
    transcript.add(samples)


def test_a_sentence_is_final_once_text_follows_it():
    from sts.transcript import Transcript

    script = _words(
        ("Первое.", 0.5, 1.5), ("Второе", 2.0, 2.5), ("предложение.", 2.6, 3.5),
        ("Третье", 4.0, 4.5),
    )
    asr = _ScriptedAsr(script)
    t = Transcript(asr, after_unchanged=5, force_after=30)
    fed = [0]

    _feed(t, 1.6, fed)
    assert _texts(t.step()) == ([], ["Первое."])
    _feed(t, 2.6, fed)
    # "Первое." has text after it now, even an unfinished sentence.
    assert _texts(t.step()) == (["Первое."], ["Второе"])
    _feed(t, 4.6, fed)
    assert _texts(t.step()) == (["Второе предложение."], ["Третье"])
    # The later passes heard audio from shortly before the first open word, not the whole recording.
    assert asr.heard[-1][0] == pytest.approx(2.0 - Transcript.CONTEXT_S)


def test_the_last_sentence_is_final_once_unchanged_for_5_seconds():
    from sts.transcript import Transcript

    script = _words(("Первое.", 0.5, 1.5), ("Второе", 2.0, 2.5))
    t = Transcript(_ScriptedAsr(script), after_unchanged=5, force_after=30)
    fed = [0]
    _feed(t, 3.0, fed)
    assert _texts(t.step()) == (["Первое."], ["Второе"])  # "Второе" first seen at 3.0
    _feed(t, 7.9, fed)
    assert _texts(t.step()) == ([], ["Второе"])
    assert t.finalize_in_s == pytest.approx(0.1)
    _feed(t, 8.0, fed)
    assert _texts(t.step()) == (["Второе"], [])
    assert t.finalize_in_s is None


def test_a_changing_sentence_is_forced_final_30_seconds_after_its_first_word():
    from sts.transcript import Transcript

    # One run-on sentence, a new word every second, never ending.
    script = _words(*[(f"слово{i}", 1.0 + i, 1.5 + i) for i in range(40)])
    t = Transcript(_ScriptedAsr(script), after_unchanged=5, force_after=30)
    fed = [0]
    for second in range(2, 31):
        _feed(t, second, fed)
        final, _ = _texts(t.step())
        assert final == []
    _feed(t, 31.0, fed)
    final, still_open = _texts(t.step())
    assert final and final[0].startswith("слово0 слово1") and still_open == []
