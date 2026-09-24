"""After the laptop sleeps and wakes (or a device is plugged in/removed, or the
default output changes), PortAudio's device snapshot is stale: output failed
with "Internal PortAudio error -9986" and the input could stop delivering,
until `sts` was restarted. These cover the recovery that replaces the restart."""

import queue
import threading
import time

import pytest

from sts.audio.capture import AudioCapture
from sts.config import AudioConfig, PipelineConfig
from sts.control import RecognitionControl
from sts.pipeline import Pipeline, _Clip


# -- AudioCapture.is_stalled ------------------------------------------------


class _Stream:
    def __init__(self, active=True):
        self._active = active

    @property
    def active(self):
        if isinstance(self._active, Exception):
            raise self._active
        return self._active


def _capture(stream, source="alsa"):
    capture = AudioCapture(AudioConfig(source=source))
    capture._stream = stream
    return capture


def test_no_open_stream_is_not_a_stall():
    assert not _capture(None).is_stalled(3.0)


def test_the_browser_microphone_is_never_a_stall():
    assert not _capture(_Stream(active=False), source="browser").is_stalled(3.0)


def test_frames_arriving_normally_is_not_a_stall():
    capture = _capture(_Stream())
    capture._last_frame_at = time.monotonic()
    assert not capture.is_stalled(3.0)


def test_silence_in_the_room_is_not_a_stall():
    """A live stream delivers a frame every block_ms whether or not anyone is
    speaking - what's measured is frames arriving, never what's in them."""
    capture = _capture(_Stream())
    capture._callback(__import__("numpy").zeros((512, 1), dtype="float32"), 512, None, None)
    assert not capture.is_stalled(3.0)


def test_no_frames_for_longer_than_the_timeout_is_a_stall():
    capture = _capture(_Stream())
    capture._last_frame_at = time.monotonic() - 10
    assert capture.is_stalled(3.0)


def test_a_stream_that_stopped_or_cant_be_asked_is_a_stall():
    assert _capture(_Stream(active=False)).is_stalled(3.0)
    assert _capture(_Stream(active=RuntimeError("stream is broken"))).is_stalled(3.0)


# -- Pipeline._recover_audio --------------------------------------------------


class _Log:
    def __init__(self):
        self.calls = []


def _pipeline(monkeypatch, source="alsa", input_device=None, output_device=None, start_fails=False):
    log = _Log()

    class Capture:
        def stop(self):
            log.calls.append("capture.stop")

        def start(self):
            log.calls.append("capture.start")
            if start_fails:
                raise RuntimeError("no input device")

        def is_stalled(self, timeout_s):
            return log.stalled

    class Playback:
        def forget_devices(self):
            log.calls.append("playback.forget")

    log.stalled = False
    monkeypatch.setattr("sts.pipeline.reinitialize_audio_system", lambda: log.calls.append("reinit"))
    monkeypatch.setattr("sts.pipeline.check_input", lambda *a, **k: log.calls.append("check_input"))
    monkeypatch.setattr("sts.pipeline.check_output", lambda *a, **k: log.calls.append("check_output"))

    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.cfg.audio.source = source
    p.cfg.audio.input_device = input_device
    p.cfg.audio.output_device = output_device
    p.capture, p.playback = Capture(), Playback()
    p.broadcaster = None
    p._device_lock = threading.Lock()
    p._audio_recovery_lock = threading.Lock()
    p._last_audio_recovery = float("-inf")
    p._last_audio_recovery_ok = True
    return p, log


def test_recovery_closes_everything_before_reinitializing_then_reopens_capture(monkeypatch):
    p, log = _pipeline(monkeypatch)

    assert p._recover_audio("test") is True

    # Terminating PortAudio invalidates open streams, so both must already be
    # closed when it happens; capture comes back afterwards.
    assert log.calls == ["capture.stop", "playback.forget", "reinit", "capture.start"]


def test_recovery_leaves_the_browser_mic_alone(monkeypatch):
    p, log = _pipeline(monkeypatch, source="browser")

    p._recover_audio("test")

    assert "capture.start" not in log.calls  # frames come over the websocket, not a local stream


def test_a_selected_device_that_no_longer_exists_falls_back_to_the_default(monkeypatch):
    p, log = _pipeline(monkeypatch, input_device=13, output_device=9)

    def gone(*a, **k):
        raise RuntimeError("Error querying device")

    monkeypatch.setattr("sts.pipeline.check_input", gone)
    monkeypatch.setattr("sts.pipeline.check_output", gone)

    assert p._recover_audio("test") is True

    assert p.cfg.audio.input_device is None and p.cfg.audio.output_device is None
    assert "capture.start" in log.calls  # and capture opened on the default


def test_a_still_usable_selected_device_is_kept(monkeypatch):
    p, _ = _pipeline(monkeypatch, input_device=3, output_device=4)

    p._recover_audio("test")

    assert (p.cfg.audio.input_device, p.cfg.audio.output_device) == (3, 4)


def test_recovery_reports_failure_when_capture_cant_reopen(monkeypatch):
    p, _ = _pipeline(monkeypatch, start_fails=True)

    assert p._recover_audio("test") is False


def test_a_recovery_right_after_another_reuses_its_result_instead_of_repeating(monkeypatch):
    """The watchdog and a failing clip can both notice the same wake-up; a
    device that's genuinely gone must not be re-initialized in a loop."""
    p, log = _pipeline(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr("sts.pipeline.time.monotonic", lambda: clock[0])

    p._recover_audio("first")
    calls_after_first = list(log.calls)
    clock[0] += 1.0  # well inside AUDIO_RECOVERY_MIN_INTERVAL_S
    assert p._recover_audio("second") is True

    assert log.calls == calls_after_first  # nothing ran twice

    clock[0] += 10.0  # long enough ago - genuinely recovers again
    p._recover_audio("third")
    assert log.calls.count("reinit") == 2


def test_a_failed_recovery_is_also_not_retried_immediately(monkeypatch):
    p, log = _pipeline(monkeypatch, start_fails=True)
    clock = [100.0]
    monkeypatch.setattr("sts.pipeline.time.monotonic", lambda: clock[0])

    assert p._recover_audio("first") is False
    clock[0] += 1.0

    assert p._recover_audio("second") is False
    assert log.calls.count("reinit") == 1


# -- the watchdog and the playback retry -------------------------------------


def test_the_watchdog_recovers_a_stalled_input_and_ignores_a_healthy_one(monkeypatch):
    p, log = _pipeline(monkeypatch)

    log.stalled = False
    p._check_audio_health()
    assert "reinit" not in log.calls

    log.stalled = True
    p._check_audio_health()
    assert "reinit" in log.calls


class _FlakyPlayback:
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.played = []
        self.forgot = 0

    def play(self, audio, sample_rate, pause_before, should_continue, on_chunk=None, seamless=False):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("Error opening OutputStream: Internal PortAudio error -9986")
        self.played.append(sample_rate)

    def forget_devices(self):
        self.forgot += 1

    holding_tail = False

    def pending_s(self):
        return 0.0


def _run_playback_loop(p):
    p._stop = threading.Event()
    p._audio = queue.Queue()
    p._audio.put(_Clip(__import__("numpy").zeros(10, dtype="float32"), 24_000))
    p.control = RecognitionControl(16000)
    p.control.enable_voice()
    p._playback_generation = 0
    thread = threading.Thread(target=Pipeline._playback_loop, args=(p,), daemon=True)
    thread.start()
    thread.join(timeout=0.5)  # loops forever on the next get(); a daemon thread is fine to leave


def test_a_clip_whose_playback_fails_is_retried_once_after_recovery(monkeypatch):
    """The real failure: after wake every clip failed with -9986 until
    restart. Now the first failure re-initializes audio and the same clip
    plays - it isn't lost."""
    p, log = _pipeline(monkeypatch)
    p.playback = _FlakyPlayback(fail_times=1)

    _run_playback_loop(p)

    assert "reinit" in log.calls
    assert p.playback.played == [24_000]


def test_playback_is_not_retried_when_recovery_itself_failed(monkeypatch):
    p, log = _pipeline(monkeypatch, start_fails=True)
    p.playback = _FlakyPlayback(fail_times=1)

    _run_playback_loop(p)

    assert p.playback.played == []  # gave up on this clip; didn't loop


def test_a_clip_still_failing_after_recovery_is_dropped_not_retried_forever(monkeypatch):
    p, log = _pipeline(monkeypatch)
    p.playback = _FlakyPlayback(fail_times=5)

    _run_playback_loop(p)

    assert p.playback.played == []
    assert log.calls.count("reinit") == 1
