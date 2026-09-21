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


def test_save_lora_adapter_roundtrip(tmp_path):
    library = VoiceLibrary(tmp_path)
    adapter_src = _make_adapter_dir(tmp_path)

    entry = library.save_lora_adapter("Pastor Ivan", adapter_src, speaker_name="pastor_ivan", lora_scale=0.35)

    assert entry.kind == "lora"
    assert entry.speaker_name == "pastor_ivan"
    assert entry.lora_scale == 0.35
    assert (entry.adapter_dir / "adapter_config.json").exists()

    fetched = library.get(entry.speaker_id)
    assert fetched.kind == "lora"
    assert fetched.speaker_name == "pastor_ivan"
    assert fetched.lora_scale == 0.35
    assert (fetched.adapter_dir / "adapter_config.json").read_text() == "fake adapter config"


def test_lora_entry_has_no_zeroshot_fields(tmp_path):
    library = VoiceLibrary(tmp_path)
    entry = library.save_lora_adapter("Alice", _make_adapter_dir(tmp_path), speaker_name="alice")
    assert entry.wav_path is None
    assert entry.ref_text is None


def test_zeroshot_and_lora_entries_coexist_in_list(tmp_path):
    library = VoiceLibrary(tmp_path)
    library.save("Alice", _make_wav(tmp_path, "a.wav"))
    library.save_lora_adapter("Bob", _make_adapter_dir(tmp_path), speaker_name="bob")

    kinds = {e.name: e.kind for e in library.list()}
    assert kinds == {"Alice": "zeroshot", "Bob": "lora"}


def test_lora_default_scale(tmp_path):
    library = VoiceLibrary(tmp_path)
    entry = library.save_lora_adapter("Alice", _make_adapter_dir(tmp_path), speaker_name="alice")
    assert entry.lora_scale == 0.3


def _make_wav(tmp_path, name: str):
    path = tmp_path / name
    path.write_bytes(b"RIFF....WAVEfmt ")
    return path


def _make_adapter_dir(tmp_path):
    import uuid

    src = tmp_path / f"adapter-src-{uuid.uuid4().hex}"
    src.mkdir()
    (src / "adapter_config.json").write_text("fake adapter config")
    (src / "adapter_model.safetensors").write_bytes(b"fake-weights")
    return src


def test_delete_removes_only_that_voice(tmp_path):
    from sts.tts.voice_library import VoiceLibrary

    library = VoiceLibrary(tmp_path / "voices")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    keep = library.save("Keep me", wav)
    gone = library.save("Delete me", wav)

    assert library.delete(gone.speaker_id) is True
    assert [e.name for e in library.list()] == ["Keep me"]
    assert library.get(gone.speaker_id) is None
    assert library.delete(gone.speaker_id) is False  # already gone
    assert library.get(keep.speaker_id) is not None


def test_delete_and_get_reject_path_traversal(tmp_path):
    from sts.tts.voice_library import VoiceLibrary

    library = VoiceLibrary(tmp_path / "voices")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    library.save("Voice", wav)
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "meta.json").write_text("{}")

    for bad in ("../precious", "..", ".", "", "/tmp", str(outside), "voice/../../precious"):
        assert library.delete(bad) is False
        assert library.get(bad) is None
    assert outside.exists()
    assert [e.name for e in library.list()] == ["Voice"]


def make_bundle(tmp_path):
    bundle = tmp_path / "bundled"
    (bundle / "default").mkdir(parents=True)
    (bundle / "default" / "reference.wav").write_bytes(b"RIFF")
    (bundle / "default" / "meta.json").write_text(
        '{"kind": "zeroshot", "name": "Default", "ref_text": "Hello there.", "created_at": 1.0}'
    )
    return bundle


def test_bundled_voices_are_listed_first_and_flagged(tmp_path):
    from sts.tts.voice_library import VoiceLibrary

    library = VoiceLibrary(tmp_path / "voices", bundled_dir=make_bundle(tmp_path))
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    library.save("Mine", wav)

    entries = library.list()
    assert [(e.name, e.builtin) for e in entries] == [("Default", True), ("Mine", False)]
    assert library.get("default").ref_text == "Hello there."
    assert library.get("default").wav_path.name == "reference.wav"


def test_bundled_voices_cannot_be_deleted_but_can_be_overridden(tmp_path):
    from sts.tts.voice_library import VoiceLibrary

    library = VoiceLibrary(tmp_path / "voices", bundled_dir=make_bundle(tmp_path))
    assert library.delete("default") is False                      # shipped: nothing of the user's to delete
    assert library.get("default") is not None

    wav = tmp_path / "mine.wav"
    wav.write_bytes(b"RIFF")
    library.save("Default", wav, "My own words.")                   # same id: the user's copy wins
    assert [e.name for e in library.list()] == ["Default"]
    assert library.get("default").ref_text == "My own words." and not library.get("default").builtin

    assert library.delete("default") is True                        # deleting the override...
    assert library.get("default").ref_text == "Hello there."       # ...brings the shipped voice back


def test_bundled_lookup_ignores_path_tricks(tmp_path):
    from sts.tts.voice_library import VoiceLibrary

    library = VoiceLibrary(tmp_path / "voices", bundled_dir=make_bundle(tmp_path))
    for bad in ("../bundled/default", "default/..", ".."):
        assert library.get(bad) is None


def test_the_real_bundle_ships_the_default_voice():
    from sts.tts.voice_library import BUNDLED_VOICES_DIR, VoiceLibrary

    entry = VoiceLibrary(BUNDLED_VOICES_DIR / "_nothing_", bundled_dir=BUNDLED_VOICES_DIR).get("default")
    assert entry is not None and entry.name == "Default" and entry.builtin
    assert entry.wav_path.stat().st_size > 10_000 and entry.ref_text
