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
