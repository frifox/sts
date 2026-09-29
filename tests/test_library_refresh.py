"""Voices added to the library while the app runs (sts-add-voices) reach the
caption page's Voice list without a restart."""

from sts.pipeline import Pipeline
from sts.tts.voice_library import VoiceLibrary


class _Broadcaster:
    def __init__(self):
        self.sent = []

    def broadcast_library(self, speakers):
        self.sent.append([s["id"] for s in speakers])


def test_a_voice_added_behind_the_app_s_back_is_sent_to_the_page(tmp_path, monkeypatch):
    p = Pipeline.__new__(Pipeline)
    p.voice_library = VoiceLibrary(tmp_path / "voices")
    p.broadcaster = _Broadcaster()
    p._broadcast_library()
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF")
    VoiceLibrary(tmp_path / "voices").save("Goldinov Venya", wav, "Текст.")  # another process
    clock = [1000.0]
    monkeypatch.setattr("sts.pipeline.time.monotonic", lambda: clock[0])
    p._check_library()
    assert "goldinov-venya" in p.broadcaster.sent[-1]
    count = len(p.broadcaster.sent)
    clock[0] += 10
    p._check_library()  # nothing changed: nothing sent
    assert len(p.broadcaster.sent) == count
