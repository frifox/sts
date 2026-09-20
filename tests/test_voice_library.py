from sts.tts.voice_library import VoiceLibrary


def test_save_and_get_roundtrip(tmp_path):
    library = VoiceLibrary(tmp_path)
    entry = library.save("Pastor Ivan", _make_wav(tmp_path, "src.wav"), ref_text="Здравствуйте")

    fetched = library.get(entry.speaker_id)
    assert fetched is not None
    assert fetched.name == "Pastor Ivan"
    assert fetched.ref_text == "Здравствуйте"
    assert fetched.wav_path.exists()


def test_name_is_slugified_into_a_stable_id(tmp_path):
    library = VoiceLibrary(tmp_path)
    entry = library.save("Pastor Ivan Petrov!", _make_wav(tmp_path, "a.wav"))
    assert entry.speaker_id == "pastor-ivan-petrov"


def test_list_returns_all_saved_entries_oldest_first(tmp_path):
    library = VoiceLibrary(tmp_path)
    library.save("Alice", _make_wav(tmp_path, "a.wav"))
    library.save("Bob", _make_wav(tmp_path, "b.wav"))

    names = [e.name for e in library.list()]
    assert names == ["Alice", "Bob"]


def test_get_missing_speaker_returns_none(tmp_path):
    library = VoiceLibrary(tmp_path)
    assert library.get("nobody") is None


def test_delete_removes_entry(tmp_path):
    library = VoiceLibrary(tmp_path)
    entry = library.save("Alice", _make_wav(tmp_path, "a.wav"))
    library.delete(entry.speaker_id)
    assert library.get(entry.speaker_id) is None


def test_ref_text_is_optional(tmp_path):
    library = VoiceLibrary(tmp_path)
    entry = library.save("Alice", _make_wav(tmp_path, "a.wav"))
    assert entry.ref_text is None


def test_saving_same_name_twice_overwrites_the_same_entry(tmp_path):
    library = VoiceLibrary(tmp_path)
    first = library.save("Alice", _make_wav(tmp_path, "a.wav"), ref_text="first")
    second = library.save("Alice", _make_wav(tmp_path, "a.wav"), ref_text="second")

    assert first.speaker_id == second.speaker_id
    assert len(library.list()) == 1
    assert library.get(first.speaker_id).ref_text == "second"


def _make_wav(tmp_path, name: str):
    path = tmp_path / name
    path.write_bytes(b"RIFF....WAVEfmt ")
    return path
