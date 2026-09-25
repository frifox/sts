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


def test_a_lowercase_sentence_continues_the_previous_one():
    """The reported case: "...он начал." + "с ними беседовать." is one sentence."""
    from sts.transcript import Sentence, join_continuations

    joined = join_continuations([
        Sentence("Я думаю, все читающие знают это, он начал.", 0.0, 9.0, True),
        Sentence("с ними беседовать.", 9.4, 10.5, True),
        Sentence("Это слова Господа.", 11.0, 12.0, True),
        Sentence("«и сказал", 12.5, 13.0, False),  # first *letter* decides, not a leading quote
    ])
    assert [(x.text, x.start, x.end, x.complete) for x in joined] == [
        ("Я думаю, все читающие знают это, он начал с ними беседовать.", 0.0, 10.5, True),
        ("Это слова Господа «и сказал", 11.0, 13.0, False),
    ]


def test_final_by_position_and_by_age():
    from sts.transcript import Sentence, count_final

    s = [Sentence(f"S{i}.", i * 10.0, i * 10.0 + 5, True) for i in range(4)] + [Sentence("open", 40, 41, False)]
    # 5 sentences: the first three have 2+ after them.
    assert count_final(s, now_s=41, after_sentences=2, after_seconds=0) == 3
    # By age alone: S0 ended at 5, S1 at 15 - 30 s before 45 means ended by 15.
    assert count_final(s, now_s=45, after_sentences=0, after_seconds=30) == 2
    # Either rule is enough; both off means nothing is final.
    assert count_final(s, now_s=100, after_sentences=2, after_seconds=30) == 4  # the open tail never ages out
    assert count_final(s, now_s=100, after_sentences=0, after_seconds=0) == 0
    # A sentence that isn't final keeps everything after it open, too.
    lone = [Sentence("A.", 0, 1, True), Sentence("B.", 50, 51, True), Sentence("C", 52, 53, False)]
    assert count_final(lone, now_s=60, after_sentences=5, after_seconds=30) == 1


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


def test_transcript_locks_sentences_once_and_stops_rehearing_their_audio():
    from sts.transcript import Transcript

    script = _words(
        ("Первое.", 0.5, 1.5), ("Второе", 2.0, 2.5), ("предложение.", 2.6, 3.5),
        ("Третье.", 4.0, 5.0), ("Четвёртое.", 6.0, 7.0), ("Пятое", 8.0, 8.5),
    )
    asr = _ScriptedAsr(script)
    t = Transcript(asr, after_sentences=2, after_seconds=30, after_silence=0)
    fed = [0]

    _feed(t, 5.5, fed)
    # "Первое." is locked, but shown as final only once the next one is
    # locked too (it could still turn out to continue).
    assert _texts(t.step()) == ([], ["Первое.", "Второе предложение.", "Третье."])
    _feed(t, 9.0, fed)
    assert _texts(t.step()) == (["Первое.", "Второе предложение."], ["Третье.", "Четвёртое.", "Пятое"])
    # The second pass heard audio from shortly before the first open word, not the whole recording.
    assert asr.heard[1][0] == pytest.approx(2.0 - Transcript.CONTEXT_S)
    _feed(t, 45.0, fed)
    assert _texts(t.step()) == (["Третье."], ["Четвёртое.", "Пятое"])  # 30+ s after it ended; the open tail stays open


def test_a_continuation_heard_only_after_the_lock_still_joins_the_locked_sentence():
    """The reported case: locked while the model heard "Для чего..." as a new
    sentence; with only the end of the locked one left as context, it came
    back as "для чего..." - which belongs on the end of the locked one."""
    from sts.transcript import Transcript

    script = _words(
        ("Мы", 0.5, 0.7), ("об", 0.8, 0.9), ("этом", 1.0, 1.3), ("рассуждаем.", 1.4, 2.4),
        ("Для", 2.8, 3.0), ("чего", 3.1, 3.4), ("Христос", 3.5, 4.0), ("приходил.", 4.1, 4.8),
        ("Мы", 5.5, 5.7), ("обычно", 5.8, 6.3), ("такие.", 6.4, 7.0),
        ("Тогда.", 8.0, 8.6), ("Потом", 9.5, 10.0),
    )
    # Once the audio it's given starts after the beginning, the model hears
    # "для" as a continuation.
    asr = _ScriptedAsr(script, revise=lambda text, begin: "для" if text == "Для" and begin > 0 else text)
    t = Transcript(asr, after_sentences=2, after_seconds=30)
    fed = [0]

    _feed(t, 6.5, fed)
    assert _texts(t.step()) == ([], ["Мы об этом рассуждаем.", "Для чего Христос приходил.", "Мы обычно"])
    _feed(t, 10.5, fed)
    final, still_open = _texts(t.step())
    assert final == ["Мы об этом рассуждаем для чего Христос приходил."]
    assert still_open == ["Мы обычно такие.", "Тогда.", "Потом"]


def test_transcript_by_age_only():
    from sts.transcript import Transcript

    script = _words(("Одно.", 0.5, 1.5), ("Другое.", 2.0, 3.0))
    t = Transcript(_ScriptedAsr(script), after_sentences=0, after_seconds=5)
    fed = [0]
    _feed(t, 4.0, fed)
    assert _texts(t.step()) == ([], ["Одно.", "Другое."])
    _feed(t, 7.0, fed)
    assert _texts(t.step()) == ([], ["Одно.", "Другое."])  # "Одно." is locked, held until the next one is
    _feed(t, 9.0, fed)
    assert _texts(t.step()) == (["Одно."], ["Другое."])
