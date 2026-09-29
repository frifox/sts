"""A Voice off click's status goes out at once, even while the voice stage is
busy synthesizing a sentence (on mps it runs in a process of its own, whose
proxy handles one request at a time)."""

import threading
import time

from sts.worker import StageProcess


def test_looking_up_a_method_does_not_wait_for_a_request_in_progress():
    proxy = object.__new__(StageProcess)
    proxy._lock = threading.Lock()
    proxy._methods = {"synthesize", "design_reference"}
    proxy._lock.acquire()  # a sentence being synthesized
    try:
        t0 = time.monotonic()
        assert hasattr(proxy, "design_reference")
        assert callable(proxy.synthesize)
        assert time.monotonic() - t0 < 0.5
    finally:
        proxy._lock.release()


def test_the_status_does_not_ask_the_voice_stage_anything():
    from sts.pipeline import Pipeline

    class _Busy:
        def __getattr__(self, name):
            if name == "design_reference":
                return lambda *a: None
            raise AssertionError(f"status asked the voice stage for {name!r}")

    sent = {}

    class _Broadcaster:
        def broadcast_status(self, **status):
            sent.update(status)

    p = Pipeline.__new__(Pipeline)
    from sts.config import PipelineConfig
    from sts.control import RecognitionControl

    p.cfg = PipelineConfig()
    p.control = RecognitionControl(sample_rate=16_000)
    p.tts = _Busy()
    p.broadcaster = _Broadcaster()
    p._pending_voice = None
    p._active_speaker_id = None
    p._voice_cloned = True
    p._broadcast_status()
    assert sent["cloned"] is True
