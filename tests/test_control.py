import numpy as np

from sts.control import RecognitionControl
from sts.transcript import Transcript


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


def test_output_volume_changes_live_is_clamped_and_persisted():
    from sts.config import PipelineConfig
    from sts.pipeline import Pipeline, _Clip

    class Broadcaster:
        status = None

        def broadcast_status(self, **kwargs):
            self.status = kwargs

    class Tts:
        cloned_speaker_wav = None

    class PersistentConfig:
        saved = None

        def set(self, key, value):
            self.saved = (key, value)

    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.broadcaster = Broadcaster()
    p.tts = Tts()
    p.control = RecognitionControl(16000)
    p._pending_voice = None
    p._active_speaker_id = None
    p._persistent_config = PersistentConfig()

    p.handle_control({"action": "set_output_volume", "percent": 40})
    assert p.cfg.audio.volume == 0.4
    assert p.broadcaster.status["output_volume"] == 40  # other open pages follow along
    assert p._persistent_config.saved == ("output_volume", "40")  # remembered across restarts

    p.handle_control({"action": "set_output_volume", "percent": -5})
    assert p.cfg.audio.volume == 0.0  # clamped to the slider's range
    p.handle_control({"action": "set_output_volume", "percent": 150})
    assert p.cfg.audio.volume == 1.0
    p.handle_control({"action": "set_output_volume", "percent": "abc"})  # junk is ignored
    assert p.cfg.audio.volume == 1.0


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
    from sts.pipeline import Pipeline, _Clip

    class Broadcaster:
        output_levels = []

        def broadcast_status(self, **kwargs):
            pass

        def broadcast_output_level(self, level):
            self.output_levels.append(level)

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
    for name in ("_transcripts", "_translations", "_audio"):
        setattr(p, name, queue.Queue())
    p.transcript = Transcript(None, 2, 30)
    p._finalize_timer = None
    p._audio.put((None, 24000, 0.0))
    p._audio.put((None, 24000, 0.0))  # a couple of sentences already synthesized ahead of playback

    p.handle_control({"action": "voice_off"})

    assert not p.control.voice_enabled.is_set()
    assert p._audio.empty()  # nothing left for _playback_loop to still play
    # No more on_chunk calls will happen until something plays again, so the
    # output meter must be told explicitly, or it'd show a stale level forever.
    assert p.broadcaster.output_levels == [0.0]


def test_reset_stops_playback_without_touching_the_stream_directly():
    """Reset must cut whatever's currently playing (the reported bug: it
    used to leave it playing to the end), but the same way voice_off does -
    bumping _playback_generation so _play()'s should_continue notices on its
    next ~50ms poll and _playback_loop's own thread stops writing - not by
    calling playback.stop() (abort()/close() on the stream) from this
    thread, which has been observed to segfault PortAudio's ALSA backend
    when _playback_loop's thread is mid-write(). The stub Playback below has
    no stop() method at all, so calling it would fail this test outright."""
    import queue

    from sts.config import PipelineConfig
    from sts.pipeline import Pipeline, _Clip

    class Broadcaster:
        output_levels = []

        def broadcast_status(self, **kwargs):
            pass

        def broadcast_reset(self):
            pass

        def broadcast_output_level(self, level):
            self.output_levels.append(level)

    class Tts:
        cloned_speaker_wav = None
        def reset_voice(self): pass

    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.cfg.tts.startup_voice = None  # skip voice_library.get() - irrelevant to what this test covers
    p.broadcaster = Broadcaster()
    p.tts = Tts()
    p._pending_voice = None
    p._active_speaker_id = None
    p._playback_generation = 0
    p.control = RecognitionControl(16000)
    p.control.enable_voice()
    for name in ("_transcripts", "_translations", "_audio"):
        setattr(p, name, queue.Queue())
    p.transcript = Transcript(None, 2, 30)
    p._finalize_timer = None
    p._audio.put((None, 24000, 0.0))  # a sentence already synthesized ahead of playback

    p.handle_control({"action": "reset"})

    assert p._audio.empty()  # nothing left for _playback_loop to still play
    assert p._playback_generation == 1  # _play()'s should_continue will now see a mismatch and stop
    assert p.broadcaster.output_levels == [0.0]


def test_playback_loop_broadcasts_before_playing_locally_not_after():
    """playback.play() blocks for close to a clip's real duration (it paces
    with the sound card draining its buffer), so broadcasting to connected
    browsers only after it returns meant a browser wouldn't get the clip
    until local playback had *already finished* - a multi-second "streaming
    delay" that scaled with clip length. The broadcast must happen before
    the blocking local play() call, not after."""
    import queue
    import threading
    import time

    from sts.config import PipelineConfig
    from sts.pipeline import Pipeline, _Clip

    calls = []

    class Playback:
        holding_tail = False

        def pending_s(self):
            return 0.0

        def play(self, *args, **kwargs):
            calls.append("play")
            time.sleep(0.05)  # stands in for play() blocking near-realtime

    class Broadcaster:
        def broadcast_audio(self, audio, sample_rate):
            calls.append("broadcast")

    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.playback = Playback()
    p.broadcaster = Broadcaster()
    p.control = RecognitionControl(16000)
    p.control.enable_voice()
    p._audio = queue.Queue()
    p._stop = threading.Event()
    p._playback_generation = 0
    p._audio.put(_Clip(np.zeros(10, dtype=np.float32), 24000))

    thread = threading.Thread(target=p._playback_loop, daemon=True)
    thread.start()
    thread.join(timeout=2)

    assert calls == ["broadcast", "play"]


def test_queue_test_output_loads_the_bundled_clip_and_marks_it_forced():
    """The Test Output button queues the pre-rendered "Testing 1 2 3" clip
    on the normal playback path, tagged so _playback_loop plays it even with
    Voice off - this is meant to test the hardware output, not the pipeline."""
    import queue

    from sts.pipeline import Pipeline, _Clip

    p = object.__new__(Pipeline)
    p._audio = queue.Queue()

    p._queue_test_output()

    clip = p._audio.get_nowait()
    audio, sample_rate, pause_before, force = clip.audio, clip.sample_rate, clip.pause_before, clip.force
    assert force is True
    assert pause_before == 0.0
    assert sample_rate > 0
    assert audio.dtype == np.float32
    assert len(audio) > 0
    assert audio.min() >= -1.0 and audio.max() <= 1.0


def test_playback_loop_plays_a_forced_clip_even_with_voice_off():
    import queue
    import threading

    from sts.pipeline import Pipeline, _Clip

    should_continues = []

    class Playback:
        holding_tail = False

        def pending_s(self):
            return 0.0

        def play(self, audio, sample_rate, pause_before=0.0, should_continue=None, on_chunk=None, seamless=False):
            should_continues.append(should_continue() if should_continue else None)

    p = object.__new__(Pipeline)
    p.playback = Playback()
    p.broadcaster = None
    p.control = RecognitionControl(16000)  # voice_enabled left off
    p._audio = queue.Queue()
    p._stop = threading.Event()
    p._playback_generation = 0
    p._audio.put(_Clip(np.zeros(10, dtype=np.float32), 24000, force=True))

    thread = threading.Thread(target=p._playback_loop, daemon=True)
    thread.start()
    thread.join(timeout=2)

    # A forced clip both gets played (not skipped by the voice_enabled gate)
    # and isn't cut short mid-playback by the should_continue polling either.
    assert should_continues == [True]


def test_broadcast_input_level_sends_the_frames_rms():
    from sts.pipeline import Pipeline, _Clip

    levels = []

    class Broadcaster:
        def broadcast_input_level(self, level):
            levels.append(level)

    p = object.__new__(Pipeline)
    p.broadcaster = Broadcaster()

    # A full-scale square wave has RMS 1.0 - a clean, exact value to assert on.
    p._broadcast_input_level(np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32))
    assert levels == [1.0]

    p._broadcast_input_level(np.zeros(10, dtype=np.float32))
    assert levels == [1.0, 0.0]


def test_broadcast_input_level_skips_an_empty_frame():
    from sts.pipeline import Pipeline, _Clip

    class Broadcaster:
        def broadcast_input_level(self, level):
            raise AssertionError("should never be called for an empty frame")

    p = object.__new__(Pipeline)
    p.broadcaster = Broadcaster()

    p._broadcast_input_level(np.array([], dtype=np.float32))  # no crash, no broadcast


def test_broadcast_output_level_sends_the_chunks_rms_throttled():
    from sts.pipeline import Pipeline, _Clip

    levels = []

    class Broadcaster:
        def broadcast_output_level(self, level):
            levels.append(level)

    p = object.__new__(Pipeline)
    p.broadcaster = Broadcaster()
    p._last_output_level_broadcast = 0.0

    # A full-scale square wave has RMS 1.0 - a clean, exact value to assert on.
    p._broadcast_output_level(np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32))
    assert levels == [1.0]

    # Immediately after: throttled, so a second chunk within the same
    # window (unlike _broadcast_input_level, this method owns its own
    # timestamp since it's called from inside AudioPlayback.play()'s chunk
    # loop, which has no loop-local state of its own to throttle with).
    p._broadcast_output_level(np.zeros(10, dtype=np.float32))
    assert levels == [1.0]

    # Once the interval has passed, the next chunk gets through.
    p._last_output_level_broadcast = 0.0
    p._broadcast_output_level(np.zeros(10, dtype=np.float32))
    assert levels == [1.0, 0.0]


def test_broadcast_output_level_skips_an_empty_chunk():
    from sts.pipeline import Pipeline, _Clip

    class Broadcaster:
        def broadcast_output_level(self, level):
            raise AssertionError("should never be called for an empty chunk")

    p = object.__new__(Pipeline)
    p.broadcaster = Broadcaster()
    p._last_output_level_broadcast = 0.0

    p._broadcast_output_level(np.array([], dtype=np.float32))  # no crash, no broadcast


def _startup_voice_pipeline(tmp_path):
    """A Pipeline stub with a real library (one shipped voice + one of the user's) and a recording TTS."""
    import queue

    from sts.config import PipelineConfig
    from sts.pipeline import Pipeline, _Clip
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
        def broadcast_output_level(self, level): pass

    class Playback:
        holding_tail = False

        def pending_s(self):
            return 0.0

        def stop(self): pass

    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.broadcaster, p.tts, p.playback = Broadcaster(), Tts(), Playback()
    p.voice_library = VoiceLibrary(tmp_path / "voices", bundled_dir=tmp_path / "bundled")
    p.control = RecognitionControl(16000)
    p._pending_voice = None
    p._active_speaker_id = None
    p._persistent_config = None
    p._playback_generation = 0
    for name in ("_transcripts", "_translations", "_audio"):
        setattr(p, name, queue.Queue())
    p.transcript = Transcript(None, 2, 30)
    p._finalize_timer = None
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
