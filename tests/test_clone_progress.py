"""Clone's step-by-step progress, as the Voice dropdown shows it."""

import numpy as np

from sts.config import PipelineConfig
from sts.pipeline import Pipeline


class _TTS:
    def clone_voice(self, path, ref_text):
        self.cloned = (path, ref_text)


def _pipeline(monkeypatch, transcript):
    p = Pipeline.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.cfg.tts.backend = "qwen3"
    p.tts = _TTS()
    p.broadcaster = None
    p._commands = None
    p._started = True
    p._write_training_wav = lambda audio: "clip.wav"
    p._transcribe_or_none = lambda audio: transcript
    monkeypatch.setattr("sts.pipeline.CLONE_PROGRESS_LINGER_S", 60)
    return p


def _states(p):
    return {step["key"]: step["state"] for step in p._clone_progress}


def test_the_steps_move_along_as_the_clone_goes(monkeypatch):
    p = _pipeline(monkeypatch, "Братья и сёстры.")
    monkeypatch.setattr("sts.tts.english_reference.needs_english_reference", lambda cfg, wav, text: False)
    p._begin_clone_progress()
    assert _states(p) == {"record": "running", "transcribe": "pending", "english": "pending", "activate": "pending"}
    p._clone_step("record", "done")
    p._activate_trained_voice(np.zeros(16000, dtype=np.float32))
    # its English reference was already made: nothing to do there
    assert _states(p) == {"record": "done", "transcribe": "done", "english": "skipped", "activate": "done"}


def test_a_failed_transcript_shows_but_the_clone_carries_on(monkeypatch):
    p = _pipeline(monkeypatch, None)
    p._begin_clone_progress()
    p._clone_step("record", "done")
    p._activate_trained_voice(np.zeros(16000, dtype=np.float32))
    assert _states(p)["transcribe"] == "failed" and _states(p)["activate"] == "done"
    assert p.tts.cloned == ("clip.wav", None)
