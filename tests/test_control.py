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
