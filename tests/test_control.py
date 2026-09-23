import numpy as np

from sts.control import RecognitionControl


def test_recognition_starts_on():
    control = RecognitionControl(sample_rate=16_000)
    assert control.active.is_set()


def test_voice_starts_off():
    control = RecognitionControl(sample_rate=16_000)
    assert not control.voice_enabled.is_set()


def test_start_stop_toggle_active():
    control = RecognitionControl(sample_rate=16_000)
    control.stop()
    assert not control.active.is_set()
    control.start()
    assert control.active.is_set()


def test_enable_disable_toggle_voice():
    control = RecognitionControl(sample_rate=16_000)
    control.enable_voice()
    assert control.voice_enabled.is_set()
    control.disable_voice()
    assert not control.voice_enabled.is_set()


def test_training_accumulates_and_reports_duration():
    control = RecognitionControl(sample_rate=16_000, target_seconds=1.0)
    control.start_training()
    assert control.training.is_set()

    reached = control.accumulate_training_segment(np.zeros(8_000, dtype=np.float32))  # 0.5s
    assert reached is False
    assert control.training_seconds == 0.5


def test_training_auto_stops_at_target_duration():
    control = RecognitionControl(sample_rate=16_000, target_seconds=1.0)
    control.start_training()
    control.accumulate_training_segment(np.zeros(8_000, dtype=np.float32))  # 0.5s
    reached = control.accumulate_training_segment(np.zeros(8_000, dtype=np.float32))  # 1.0s total
    assert reached is True


def test_stop_training_returns_concatenated_audio():
    control = RecognitionControl(sample_rate=16_000)
    control.start_training()
    control.accumulate_training_segment(np.ones(100, dtype=np.float32))
    control.accumulate_training_segment(np.ones(50, dtype=np.float32) * 2)

    audio = control.stop_training()
    assert not control.training.is_set()
    assert audio is not None
    assert len(audio) == 150
    assert control.training_seconds == 0.0


def test_stop_training_with_discard_returns_none_and_drops_buffer():
    control = RecognitionControl(sample_rate=16_000)
    control.start_training()
    control.accumulate_training_segment(np.ones(100, dtype=np.float32))

    audio = control.stop_training(discard=True)
    assert audio is None


def test_stop_training_without_any_samples_returns_none():
    control = RecognitionControl(sample_rate=16_000)
    control.start_training()
    assert control.stop_training() is None


def test_manual_stop_before_target_uses_whatever_was_captured():
    control = RecognitionControl(sample_rate=16_000, target_seconds=100.0)
    control.start_training()
    control.accumulate_training_segment(np.ones(16_000, dtype=np.float32))  # 1s, far short of target

    audio = control.stop_training()
    assert audio is not None
    assert len(audio) == 16_000


def test_pause_threshold_changes_live_and_is_clamped():
    import queue

    from sts.config import PipelineConfig
    from sts.pipeline import Pipeline

    class Broadcaster:
        status = None

        def broadcast_status(self, **kwargs):
            self.status = kwargs

    class Tts:
        cloned_speaker_wav = None

    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.broadcaster = Broadcaster()
    p.tts = Tts()
    p.control = RecognitionControl(16000)
    p._pending_voice = None
    p._active_speaker_id = None

    p.handle_control({"action": "set_min_silence", "ms": 400})
    assert p.cfg.vad.min_silence_ms == 400
    assert p.broadcaster.status["min_silence_ms"] == 400          # other open pages follow along
    p.handle_control({"action": "set_min_silence", "ms": 5})
    assert p.cfg.vad.min_silence_ms == 200                        # clamped to the slider's range
    p.handle_control({"action": "set_min_silence", "ms": 999999})
    assert p.cfg.vad.min_silence_ms == 1500
    p.handle_control({"action": "set_min_silence", "ms": "abc"})  # junk is ignored
    assert p.cfg.vad.min_silence_ms == 1500


def test_voice_off_drops_already_synthesized_but_unplayed_sentences():
    """voice_off must drop anything already synthesized and queued but not
    yet handed to the sound card - not just stop *new* audio from being
    queued - so a sentence or two that raced ahead of playback doesn't still
    play out after the operator unchecks Voice. (The clip playing *right
    now*, if any, is cut short by _playback_loop's should_continue polling
    - see test_playback.py - rather than by anything handle_control does
    directly: aborting/closing the output stream from this thread while
    _playback_loop's thread is mid-write() has been observed to segfault
    PortAudio's ALSA backend.)"""
    import queue

    from sts.config import PipelineConfig
    from sts.pipeline import Pipeline

    class Broadcaster:
        def broadcast_status(self, **kwargs):
            pass

    class Tts:
        cloned_speaker_wav = None

    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.broadcaster = Broadcaster()
    p.tts = Tts()
    p._pending_voice = None
    p._active_speaker_id = None
    p.control = RecognitionControl(16000)
    p.control.enable_voice()
    for name in ("_segments", "_transcripts", "_translations", "_audio"):
        setattr(p, name, queue.Queue())
    p._audio.put((None, 24000, 0.0))
    p._audio.put((None, 24000, 0.0))  # a couple of sentences already synthesized ahead of playback

    p.handle_control({"action": "voice_off"})

    assert not p.control.voice_enabled.is_set()
    assert p._audio.empty()  # nothing left for _playback_loop to still play


def _startup_voice_pipeline(tmp_path):
    """A Pipeline stub with a real library (one shipped voice + one of the user's) and a recording TTS."""
    import queue

    from sts.config import PipelineConfig
    from sts.pipeline import Pipeline
    from sts.tts.voice_library import VoiceLibrary

    bundle = tmp_path / "bundled" / "default"
    bundle.mkdir(parents=True)
    (bundle / "reference.wav").write_bytes(b"RIFF")
    (bundle / "meta.json").write_text('{"kind": "zeroshot", "name": "Default", "ref_text": "Hi.", "created_at": 1.0}')
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    class Tts:
        cloned_speaker_wav = None
        def clone_from_entry(self, entry): self.cloned_speaker_wav = str(entry.wav_path)
        def reset_voice(self): self.cloned_speaker_wav = None
        def set_default_voice(self, voice): pass

    class Broadcaster:
        status = None
        def broadcast_status(self, **kwargs): Broadcaster.status = kwargs
        def broadcast_library(self, speakers): Broadcaster.library = speakers
        def broadcast_reset(self): pass

    class Playback:
        def stop(self): pass

    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.broadcaster, p.tts, p.playback = Broadcaster(), Tts(), Playback()
    p.voice_library = VoiceLibrary(tmp_path / "voices", bundled_dir=tmp_path / "bundled")
    p.control = RecognitionControl(16000)
    p._pending_voice = None
    p._active_speaker_id = None
    for name in ("_segments", "_transcripts", "_translations", "_audio"):
        setattr(p, name, queue.Queue())
    mine = p.voice_library.save("Mine", wav)
    return p, mine


def test_session_starts_on_the_bundled_default_voice_and_reset_returns_to_it(tmp_path):
    p, mine = _startup_voice_pipeline(tmp_path)
    p._use_startup_voice()
    assert p._active_speaker_id == "default" and p.tts.cloned_speaker_wav is not None

    p.handle_control({"action": "select_speaker", "speaker_id": mine.speaker_id})
    assert p._active_speaker_id == mine.speaker_id
    p.handle_control({"action": "reset"})
    assert p._active_speaker_id == "default"                       # Reset goes back to the Default voice


def test_startup_voice_can_be_disabled(tmp_path):
    p, _ = _startup_voice_pipeline(tmp_path)
    p.cfg.tts.startup_voice = None
    p._use_startup_voice()
    assert p._active_speaker_id is None and p.tts.cloned_speaker_wav is None   # plain built-in voice


def test_shipped_voice_is_undeletable_and_deleting_the_active_custom_voice_returns_to_default(tmp_path):
    p, mine = _startup_voice_pipeline(tmp_path)
    p._use_startup_voice()
    p.handle_control({"action": "delete_voice", "speaker_id": "default"})
    assert p.voice_library.get("default") is not None and p._active_speaker_id == "default"

    p.handle_control({"action": "select_speaker", "speaker_id": mine.speaker_id})
    p.handle_control({"action": "delete_voice", "speaker_id": mine.speaker_id})
    assert p.voice_library.get(mine.speaker_id) is None
    assert p._active_speaker_id == "default"

    # deleting an inactive voice while on a built-in voice must not switch voices
    other = p.voice_library.save("Other", tmp_path / "a.wav")
    p.handle_control({"action": "default_voice", "voice": "Adde Michal"})
    assert p._active_speaker_id is None
    p.handle_control({"action": "delete_voice", "speaker_id": other.speaker_id})
    assert p._active_speaker_id is None
