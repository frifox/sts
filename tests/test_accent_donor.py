"""The Voice dropdown's "English accent" setting: which American preset voice
English references for non-English cloned voices imitate, or none."""

from sts.config import PipelineConfig
from sts.persistent_config import PersistentConfig
from sts.pipeline import Pipeline


class _TTS:
    def __init__(self):
        self.donor = None
        self.cloned = []

    def set_accent_donor(self, name):
        self.donor = name

    def clone_from_entry(self, entry):
        self.cloned.append(entry)


class _Library:
    def get(self, speaker_id):
        return type("Entry", (), {"speaker_id": speaker_id, "kind": "zeroshot"})()


def _pipeline(tmp_path):
    p = Pipeline.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.cfg.tts.backend = "qwen3"
    p.tts = _TTS()
    p.voice_library = _Library()
    p._persistent_config = PersistentConfig(str(tmp_path / "sts.db"))
    p._active_speaker_id = "neyman"
    p._pending_voice = None
    return p


def test_choosing_a_donor_re_clones_the_active_voice_and_is_remembered(tmp_path):
    p = _pipeline(tmp_path)
    p._set_accent_donor("Ryan")
    assert p.cfg.tts.qwen3_accent_donor == "Ryan" and p.tts.donor == "Ryan"
    assert [entry.speaker_id for entry in p.tts.cloned] == ["neyman"]
    assert p._persistent_config.get("qwen3_accent_donor") == "Ryan"
    p._set_accent_donor("")  # the speaker alone
    assert p.cfg.tts.qwen3_accent_donor == "" and p.tts.donor == ""


def test_an_unknown_donor_is_ignored(tmp_path):
    p = _pipeline(tmp_path)
    p._set_accent_donor("Vivian")
    assert p.cfg.tts.qwen3_accent_donor == "" and p.tts.cloned == []  # None, the default
