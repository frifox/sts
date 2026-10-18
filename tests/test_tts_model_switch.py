"""Switching the voice model (Qwen3 / Drama 3) from the page."""

import threading
import time

from sts.config import PipelineConfig
from sts.persistent_config import PersistentConfig
from sts.pipeline import Pipeline
from sts.tts.english_reference import uses_english_reference


class _TTS:
    def __init__(self, name):
        self.name, self.cloned, self.reset = name, [], 0

    def clone_from_entry(self, entry):
        self.cloned.append(entry.speaker_id)

    def reset_voice(self):
        self.reset += 1


class _Library:
    def get(self, speaker_id):
        return type("E", (), {"speaker_id": speaker_id, "kind": "zeroshot", "name": speaker_id,
                              "wav_path": "r.wav", "ref_text": "Brothers and sisters."})()


def _pipeline(tmp_path, monkeypatch):
    p = Pipeline.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.cfg.tts.backend, p.cfg.tts.device = "qwen3", "mps"
    p.tts = _TTS("qwen3")
    p.voice_library = _Library()
    p._active_speaker_id, p._pending_voice, p._voice_cloned = "default-1-25", None, True
    p._persistent_config = PersistentConfig(str(tmp_path / "sts.db"))
    p._commands = None
    p.broadcaster = None
    monkeypatch.setattr(p, "_tts_models", lambda: [{"id": "qwen3"}, {"id": "drama3"}])
    monkeypatch.setattr("sts.pipeline._stage_model", lambda factory, cfg, name: _TTS(cfg.backend))
    monkeypatch.setattr("sts.pipeline._wait_ready", lambda model: None)
    monkeypatch.setattr("sts.pipeline._close_stage", lambda model: None)
    return p


def test_the_new_model_takes_the_current_voice_and_is_remembered(tmp_path, monkeypatch):
    p = _pipeline(tmp_path, monkeypatch)
    p._switch_tts_model("drama3")
    for _ in range(100):
        if p.cfg.tts.backend == "drama3":
            break
        time.sleep(0.02)
    assert p.cfg.tts.backend == "drama3" and p.tts.name == "drama3"
    assert p.tts.cloned == ["default-1-25"]  # the voice in use carried over
    assert p._tts_switching is None and p._persistent_config.get("tts_backend") == "drama3"
    assert p.cfg.tts.voice_groups == {}  # Drama 3 has no built-in voices


def test_unknown_or_current_models_are_ignored(tmp_path, monkeypatch):
    p = _pipeline(tmp_path, monkeypatch)
    p._switch_tts_model("qwen3")
    p._switch_tts_model("xtts")
    assert p.cfg.tts.backend == "qwen3" and not getattr(p, "_tts_switching", None)


def test_drama3_clones_non_english_voices_from_their_english_reference():
    cfg = PipelineConfig().tts
    cfg.backend = "drama3"
    assert uses_english_reference(cfg, "Братья и сёстры.") and uses_english_reference(cfg, None)
    assert not uses_english_reference(cfg, "Brothers and sisters.")
