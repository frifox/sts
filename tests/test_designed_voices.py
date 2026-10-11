"""Voices made with the Design dialog are marked as such (the page lists them
under Default, not Custom)."""

from sts.pipeline import DESIGN_REFERENCE_TEXT, Pipeline
from sts.tts.voice_library import VoiceLibrary


def test_a_designed_voice_is_marked_and_a_cloned_one_is_not(tmp_path):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF")
    library = VoiceLibrary(tmp_path / "voices")
    library.save("Narrator", wav, "Some English text.", designed=True)
    library.save("Pastor", wav, "Братья и сёстры.")
    library.save("Old Design", wav, DESIGN_REFERENCE_TEXT)  # designed before the mark existed
    p = Pipeline.__new__(Pipeline)
    p.voice_library = VoiceLibrary(tmp_path / "voices")
    designed = {s["id"]: s["designed"] for s in p._library_speakers() if not s["builtin"]}
    assert designed == {"narrator": True, "pastor": False, "old-design": True}
