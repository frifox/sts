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


def test_live_line_prints_each_change_in_full_when_piped():
    from sts.caption import LiveLine

    out = io.StringIO()  # not a terminal
    line = LiveLine(out)
    for text in ("А.", "А. Б", "А. Б"):
        line.update(text)
    line.finish()
    assert out.getvalue() == "А.\nА.\nБ\n"


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
