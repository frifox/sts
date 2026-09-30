"""Cloning a voice that first needs its English reference made happens
alongside the current voice, not in its way."""

import threading

from sts.config import PipelineConfig
from sts.pipeline import Pipeline


class _TTS:
    def __init__(self):
        self.cloned = []
        self.made = threading.Event()
        self.release = threading.Event()

    def make_english_reference(self, wav_path):
        self.release.wait(5)  # the reference takes a while
        self.made.set()

    def clone_voice(self, path, ref_text):
        self.cloned.append(path)


def _pipeline(monkeypatch):
    p = Pipeline.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.cfg.tts.backend = "qwen3"
    p.tts = _TTS()
    p.broadcaster = None
    p._commands = None  # control-thread work runs inline
    p._started = True
    monkeypatch.setattr("sts.tts.english_reference.needs_english_reference", lambda cfg, wav, text: True)
    return p


def test_the_current_voice_keeps_going_until_the_new_one_is_ready(monkeypatch):
    p = _pipeline(monkeypatch)
    p._clone_when_ready("new.wav", "Текст.", lambda: p.tts.clone_voice("new.wav", "Текст."), "New", None)
    assert p.tts.cloned == [] and p._preparing_voice == ("New", None)  # not switched yet
    p.tts.release.set()
    p.tts.made.wait(5)
    for _ in range(100):
        if p.tts.cloned:
            break
        threading.Event().wait(0.02)
    assert p.tts.cloned == ["new.wav"] and p._preparing_voice is None


def test_a_voice_chosen_meanwhile_wins(monkeypatch):
    p = _pipeline(monkeypatch)
    p._clone_when_ready("slow.wav", "Текст.", lambda: p.tts.clone_voice("slow.wav", None), "Slow", None)
    p._next_voice_request()  # e.g. a built-in voice picked while it prepares
    p.tts.release.set()
    p.tts.made.wait(5)
    threading.Event().wait(0.2)
    assert p.tts.cloned == []
