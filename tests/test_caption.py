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


def test_live_line_rewrites_the_transcript_in_place(monkeypatch):
    from sts.caption import LiveLine

    monkeypatch.setattr("sts.caption.shutil.get_terminal_size", lambda: os.terminal_size((20, 24)))
    out = _Terminal()
    line = LiveLine(out)

    line.update("Братья")
    line.update("Братья и сестры")
    assert out.getvalue() == "\r\x1b[JБратья\r\x1b[JБратья и сестры"  # same row, no newline

    long = "Братья и сестры, вот так говорил"  # 32 chars: two rows at width 20
    line.update(long)
    line.update(long + " наш Господь")
    assert out.getvalue().endswith(f"\r\x1b[J{long}\r\x1b[1A\x1b[J{long} наш Господь")  # back up over both rows

    line.update(long + " наш Господь")  # unchanged: nothing redrawn
    assert out.getvalue().endswith("наш Господь")
    line.finish()
    assert out.getvalue().endswith("\n")


def test_live_line_prints_each_change_on_its_own_line_when_piped():
    from sts.caption import LiveLine

    out = io.StringIO()  # not a terminal
    line = LiveLine(out)
    for text in ("a", "a b", "a b", "a b c"):
        line.update(text)
    line.finish()
    assert out.getvalue() == "a\na b\na b c\n"
