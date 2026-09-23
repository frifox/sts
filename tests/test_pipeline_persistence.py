import threading

import numpy as np

from sts.audio.capture import AudioCapture
from sts.config import PipelineConfig
from sts.control import RecognitionControl
from sts.persistent_config import PersistentConfig
from sts.pipeline import Pipeline
from sts.tts.voice_library import VoiceLibrary


class _Broadcaster:
    engines = None
    status = None

    def broadcast_engines(self, info):
        self.engines = info

    def broadcast_status(self, **kwargs):
        self.status = kwargs

    def broadcast_stage(self, text, ready=False):
        pass

    def broadcast_devices(self, message):
        pass

    def broadcast_library(self, speakers):
        pass

    def broadcast_reset(self):
        pass


class _Tts:
    cloned_speaker_wav = None
    default_voice = None

    def clone_from_entry(self, entry):
        self.cloned_speaker_wav = str(entry.wav_path)

    def reset_voice(self):
        self.cloned_speaker_wav = None

    def set_default_voice(self, voice):
        self.default_voice = voice


def _voice_pipeline(tmp_path, persistent_config=None):
    """A Pipeline stub with a real voice library (one shipped voice + one
    user voice) and a recording TTS, for exercising voice restore/persist."""
    bundle = tmp_path / "bundled" / "default"
    bundle.mkdir(parents=True)
    (bundle / "reference.wav").write_bytes(b"RIFF")
    (bundle / "meta.json").write_text('{"kind": "zeroshot", "name": "Default", "ref_text": "Hi.", "created_at": 1.0}')
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.broadcaster, p.tts = _Broadcaster(), _Tts()
    p.voice_library = VoiceLibrary(tmp_path / "voices", bundled_dir=tmp_path / "bundled")
    p.control = RecognitionControl(16000)
    p._pending_voice = None
    p._active_speaker_id = None
    p._persistent_config = persistent_config
    mine = p.voice_library.save("Mine", wav)
    return p, mine


# -- restoring a saved voice at startup --------------------------------------


def test_restore_persisted_voice_applies_a_saved_library_voice(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p, mine = _voice_pipeline(tmp_path, pc)
    pc.set(f"voice:{p.cfg.tts.backend}", f"saved:{mine.speaker_id}")

    p._restore_persisted_voice()

    assert p._active_speaker_id == mine.speaker_id
    assert p.tts.cloned_speaker_wav is not None


def test_restore_persisted_voice_applies_a_saved_default_voice(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p, _ = _voice_pipeline(tmp_path, pc)
    pc.set(f"voice:{p.cfg.tts.backend}", "default:Ana Florence")

    p._restore_persisted_voice()

    assert p.cfg.tts.default_voice == "Ana Florence"
    assert p.tts.default_voice == "Ana Florence"
    assert p._active_speaker_id is None  # a built-in voice drops any cloned one


def test_restore_persisted_voice_is_a_noop_when_nothing_saved(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p, _ = _voice_pipeline(tmp_path, pc)

    p._restore_persisted_voice()

    assert p._active_speaker_id is None
    assert p.tts.cloned_speaker_wav is None


def test_restore_persisted_voice_ignores_an_unknown_default_voice_name(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p, _ = _voice_pipeline(tmp_path, pc)
    pc.set(f"voice:{p.cfg.tts.backend}", "default:Not A Real Voice")

    p._restore_persisted_voice()

    assert p.tts.default_voice is None  # ignored, not applied


def test_restore_persisted_voice_ignores_a_missing_saved_speaker(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p, _ = _voice_pipeline(tmp_path, pc)
    pc.set(f"voice:{p.cfg.tts.backend}", "saved:does-not-exist")

    p._restore_persisted_voice()

    assert p._active_speaker_id is None
    assert p.tts.cloned_speaker_wav is None


def test_restore_persisted_voice_is_scoped_per_tts_backend(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p, mine = _voice_pipeline(tmp_path, pc)
    pc.set("voice:qwen3", f"saved:{mine.speaker_id}")  # a different backend's saved voice

    p._restore_persisted_voice()

    assert p._active_speaker_id is None  # not applied - p.cfg.tts.backend is "xtts"


# -- persisting a voice choice made live -------------------------------------


def test_default_voice_action_persists_the_choice(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p, _ = _voice_pipeline(tmp_path, pc)

    p.handle_control({"action": "default_voice", "voice": "Ana Florence"})

    assert pc.get(f"voice:{p.cfg.tts.backend}") == "default:Ana Florence"


def test_select_speaker_action_persists_the_choice(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p, mine = _voice_pipeline(tmp_path, pc)

    p.handle_control({"action": "select_speaker", "speaker_id": mine.speaker_id})

    assert pc.get(f"voice:{p.cfg.tts.backend}") == f"saved:{mine.speaker_id}"


def test_select_speaker_with_no_id_does_not_persist(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p, mine = _voice_pipeline(tmp_path, pc)
    pc.set(f"voice:{p.cfg.tts.backend}", f"saved:{mine.speaker_id}")

    p.handle_control({"action": "select_speaker", "speaker_id": None})

    assert pc.get(f"voice:{p.cfg.tts.backend}") == f"saved:{mine.speaker_id}"  # untouched


def test_voice_actions_are_fine_with_no_persistent_config(tmp_path):
    p, mine = _voice_pipeline(tmp_path, persistent_config=None)
    p.handle_control({"action": "default_voice", "voice": "Ana Florence"})
    p.handle_control({"action": "select_speaker", "speaker_id": mine.speaker_id})  # must not raise


# -- persisting the ASR backend on a live switch -----------------------------


def _asr_pipeline(monkeypatch, asr_factory, persistent_config=None):
    monkeypatch.setattr("sts.pipeline.create_asr", asr_factory)
    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.broadcaster = _Broadcaster()
    p.tts = _Tts()
    p._pending_voice = None
    p._active_speaker_id = None
    p._asr_lock = threading.Lock()
    p._asr_busy = False
    p._persistent_config = persistent_config
    p.control = RecognitionControl(16000)
    return p


def test_switch_asr_backend_persists_on_success(tmp_path, monkeypatch):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p = _asr_pipeline(monkeypatch, lambda cfg: f"{cfg.backend}-engine", pc)
    p.asr = "whisper-engine"
    p.cfg.asr.backend = "whisper"

    p._switch_asr_backend("parakeet")

    assert pc.get("asr_backend") == "parakeet"


def test_switch_asr_backend_does_not_persist_a_same_backend_noop(tmp_path, monkeypatch):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p = _asr_pipeline(monkeypatch, lambda cfg: f"{cfg.backend}-engine", pc)
    p.asr = "whisper-engine"
    p.cfg.asr.backend = "whisper"
    p._asr_busy = True

    p._switch_asr_backend("whisper")

    assert pc.get("asr_backend") is None


def test_switch_asr_backend_does_not_persist_on_failure(tmp_path, monkeypatch):
    def flaky(cfg):
        if cfg.backend == "parakeet":
            raise RuntimeError("out of memory")
        return "whisper-engine"

    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p = _asr_pipeline(monkeypatch, flaky, pc)
    p.asr = "whisper-engine"
    p.cfg.asr.backend = "whisper"

    p._switch_asr_backend("parakeet")

    assert pc.get("asr_backend") is None


# -- persisting audio devices on a live change -------------------------------


def _device_pipeline(persistent_config=None):
    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.capture = AudioCapture(p.cfg.audio)
    p.playback = _StubPlayback()
    p.tts = _Tts()
    p.control = RecognitionControl(16000)
    p._pending_voice = None
    p._active_speaker_id = None
    p._device_lock = threading.Lock()
    p.broadcaster = _Broadcaster()
    p._persistent_config = persistent_config
    return p


class _StubStream:
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class _StubPlayback:
    def stop(self):
        pass


def test_handle_device_action_persists_input_and_output_when_both_change(tmp_path, monkeypatch):
    monkeypatch.setattr("sts.audio.devices.check_input", lambda *a, **k: None)
    monkeypatch.setattr("sts.audio.devices.check_output", lambda *a, **k: None)
    monkeypatch.setattr("sts.audio.capture.sd.InputStream", lambda **kw: _StubStream(**kw))
    monkeypatch.setattr(
        "sts.audio.capture.sd.query_devices", lambda device, kind: {"index": device or 0, "name": "Stub"}
    )
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p = _device_pipeline(pc)

    p._handle_device_action("set_audio_devices", {"input": "3", "output": "5"})

    assert pc.get("input_device") == "3"
    assert pc.get("output_device") == "5"


def test_handle_device_action_only_persists_what_changed(tmp_path, monkeypatch):
    monkeypatch.setattr("sts.audio.devices.check_input", lambda *a, **k: None)
    monkeypatch.setattr("sts.audio.devices.check_output", lambda *a, **k: None)
    monkeypatch.setattr("sts.audio.capture.sd.InputStream", lambda **kw: _StubStream(**kw))
    monkeypatch.setattr(
        "sts.audio.capture.sd.query_devices", lambda device, kind: {"index": device or 0, "name": "Stub"}
    )
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p = _device_pipeline(pc)

    p._handle_device_action("set_audio_devices", {"input": None, "output": "5"})

    assert pc.get("output_device") == "5"
    assert pc.get("input_device") is None  # the input was already "None"/system default - nothing changed


def test_handle_device_action_persists_system_default_as_empty_string(tmp_path, monkeypatch):
    monkeypatch.setattr("sts.audio.devices.check_input", lambda *a, **k: None)
    monkeypatch.setattr("sts.audio.devices.check_output", lambda *a, **k: None)
    monkeypatch.setattr("sts.audio.capture.sd.InputStream", lambda **kw: _StubStream(**kw))
    monkeypatch.setattr(
        "sts.audio.capture.sd.query_devices", lambda device, kind: {"index": device or 0, "name": "Stub"}
    )
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p = _device_pipeline(pc)
    p.cfg.audio.output_device = 5  # something previously set, now cleared back to system default

    p._handle_device_action("set_audio_devices", {"input": "3", "output": None})

    assert pc.get("output_device") == ""


def test_handle_device_action_does_not_persist_on_a_failed_apply(tmp_path, monkeypatch):
    def raising_check_input(*a, **k):
        raise RuntimeError("device busy")

    monkeypatch.setattr("sts.audio.devices.check_input", raising_check_input)
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p = _device_pipeline(pc)

    p._handle_device_action("set_audio_devices", {"input": "3", "output": None})

    assert pc.get("input_device") is None


def test_mic_start_stop_do_not_touch_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr("sts.audio.capture.sd.InputStream", lambda **kw: _StubStream(**kw))
    monkeypatch.setattr(
        "sts.audio.capture.sd.query_devices", lambda device, kind: {"index": device or 0, "name": "Stub"}
    )
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    p = _device_pipeline(pc)

    p._handle_device_action("mic_start", {})
    p._handle_device_action("mic_stop", {})

    assert pc.get("input_device") is None
    assert pc.get("output_device") is None
