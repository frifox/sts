"""A voice can bring its own pause between sentences (meta.json
"sentence_pause_s"), set when it's switched to."""

import json

from sts.config import PipelineConfig
from sts.persistent_config import PersistentConfig
from sts.pipeline import Pipeline
from sts.tts.voice_library import VoiceLibrary


def _pipeline(tmp_path):
    root = tmp_path / "voices"
    library = VoiceLibrary(root)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    for name, pause in (("Default", 1.0), ("Default 1.25", 0.8), ("Kamke Fedor", None)):
        entry = library.save(name, wav, "Text.")
        if pause is not None:
            meta = root / entry.speaker_id / "meta.json"
            meta.write_text(json.dumps({**json.loads(meta.read_text()), "sentence_pause_s": pause}))
    p = Pipeline.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.cfg.tts.backend = "qwen3"
    p.voice_library = library
    p._persistent_config = PersistentConfig(str(tmp_path / "sts.db"))
    p._select_speaker = lambda speaker_id: None  # the switch itself isn't what's tested
    p._broadcast_status = lambda: None
    return p


def test_switching_voices_brings_their_pause(tmp_path):
    p = _pipeline(tmp_path)
    p._execute_control({"action": "select_speaker", "speaker_id": "default-1-25"})
    assert p.cfg.audio.pause_between_s == 0.8 and p._persistent_config.get("pause_between_s") == "0.8"
    p._execute_control({"action": "select_speaker", "speaker_id": "default"})
    assert p.cfg.audio.pause_between_s == 1.0


def test_a_voice_without_a_pause_of_its_own_leaves_it_alone(tmp_path):
    p = _pipeline(tmp_path)
    p.cfg.audio.pause_between_s = 0.6
    p._execute_control({"action": "select_speaker", "speaker_id": "kamke-fedor"})
    assert p.cfg.audio.pause_between_s == 0.6


def test_the_shipped_default_voice_has_its_pause():
    from sts.tts.voice_library import BUNDLED_VOICES_DIR

    entry = VoiceLibrary(BUNDLED_VOICES_DIR / "_none_", bundled_dir=BUNDLED_VOICES_DIR).get("default")
    assert entry.sentence_pause_s == 1.0
