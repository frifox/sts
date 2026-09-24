"""Stage threads survive a failing item, the watchdog restarts a dead one,
and Reset drops work that was already in flight."""

import queue
import threading
import time

import numpy as np
import pytest

from sts.config import PipelineConfig
from sts.control import RecognitionControl
from sts.pipeline import Pipeline


def _stub():
    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.control = RecognitionControl(16000)
    p.broadcaster = None
    p._stop = threading.Event()
    p._threads = {}
    p._rtf_window = {}
    p._last_asr_audio_s = 0.0
    p._last_sentence_translated = None
    return p


def test_a_failing_item_is_skipped_and_the_stage_keeps_going(monkeypatch):
    """The real failure: one exception (e.g. a CUDA out-of-memory error in
    translation) ended the stage's thread, so nothing after it was ever
    translated until a restart."""
    monkeypatch.setattr("sts.pipeline.STAGE_ERROR_BACKOFF_S", 0.0)
    p = _stub()
    p._transcripts = queue.Queue()
    p._translations = queue.Queue()

    class Translator:
        def translate_sentences(self, text, context=None):
            if text == "boom":
                raise RuntimeError("CUDA out of memory")
            yield text.upper()

    p.translator = Translator()
    for text in ("boom", "fine"):
        p._transcripts.put((text, 0, None))

    thread = threading.Thread(target=p._translate_loop, daemon=True)
    thread.start()
    translated = p._translations.get(timeout=2)
    p._stop.set()
    thread.join(timeout=2)

    assert translated == ("FINE", False)
    assert not thread.is_alive()  # and stop() can actually end it now


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # the thread dies on purpose
def test_the_watchdog_restarts_a_stage_thread_that_died():
    p = _stub()
    runs = []

    def stage():
        runs.append(1)
        if len(runs) == 1:
            raise SystemExit  # gets past _run_stage's `except Exception`
        p._stop.wait()

    p._start_thread("test", stage)
    p._threads["test"][1].join(timeout=2)

    p._check_stage_threads()
    time.sleep(0.1)

    assert len(runs) == 2
    assert p._threads["test"][1].is_alive()
    p._stop.set()


def test_translation_in_flight_during_reset_is_dropped():
    p = _stub()
    p._translations = queue.Queue()

    class Translator:
        def translate_sentences(self, text, context=None):
            yield "first"
            p._reset_generation += 1  # the operator hits Reset mid-segment
            yield "second"

    p.translator = Translator()
    p._translate_segment("x. y.")

    assert p._translations.get_nowait() == ("first", False)
    assert p._translations.empty()


def test_a_transcript_finished_after_reset_is_not_committed():
    p = _stub()
    p._segments = queue.Queue()
    p._asr_lock = threading.Lock()
    committed = []
    p._commit_new_asr_text = lambda text, final: committed.append(text)

    class Asr:
        def transcribe(self, audio):
            p._reset_generation += 1  # Reset lands while this is transcribing
            return "old speaker"

    p.asr = Asr()
    p._segments.put(("final", np.zeros(1600, dtype=np.float32)))
    p._asr_step()

    assert committed == []


def test_tts_stops_synthesizing_a_sentence_once_voice_is_turned_off():
    p = _stub()
    p.control.enable_voice()
    p._translations = queue.Queue()
    p._audio = queue.Queue()
    pieces = []

    class Tts:
        def synthesize_stream(self, text):
            for i in range(3):
                pieces.append(i)
                if i == 1:
                    p.control.disable_voice()
                yield np.zeros(100, dtype=np.float32), 24_000

    p.tts = Tts()
    p._translations.put(("One. Two. Three.", False))
    p._tts_step()

    assert p._audio.qsize() == 1  # only the piece made before Voice went off
    assert pieces == [0, 1]  # and the third was never started


# -- operator actions off the web server's event loop --------------------------


def test_slow_operator_actions_queue_for_the_control_thread_and_quick_ones_run_now():
    p = _stub()
    p._commands = queue.Queue()
    p._pending_voice = None
    p._audio = queue.Queue()
    ran = []
    p._select_speaker = lambda speaker_id: ran.append(("select", speaker_id))
    p._persistent_config = None

    p.handle_control({"action": "select_speaker", "speaker_id": "abc"})
    p.handle_control({"action": "voice_on"})

    assert ran == []  # not run on the calling (event loop) thread
    assert p.control.voice_enabled.is_set()  # but the switch flipped right away
    fn, args = p._commands.get_nowait()
    fn(*args)
    assert ran == [("select", "abc")]


def test_finishing_training_stops_capture_now_and_clones_on_the_control_thread():
    p = _stub()
    p._commands = queue.Queue()
    p.control.start_training()
    p.control.accumulate_training_segment(np.zeros(16000, dtype=np.float32))

    p.handle_control({"action": "train_stop"})

    assert not p.control.training.is_set()  # no more audio goes into it
    fn, args = p._commands.get_nowait()
    assert fn.__name__ == "_activate_trained_voice"
    assert len(args[0]) == 16000


def test_reset_restores_the_voice_after_anything_already_queued():
    """A voice picked just before Reset must not come back after it."""
    p = _stub()
    p._commands = queue.Queue()
    for name in ("_segments", "_transcripts", "_translations", "_audio"):
        setattr(p, name, queue.Queue())
    p._playback_generation = 0
    order = []
    p._select_speaker = lambda speaker_id: order.append(("select", speaker_id))
    p._use_startup_voice = lambda: order.append("startup voice")
    p._persistent_config = None

    p.handle_control({"action": "select_speaker", "speaker_id": "abc"})
    p.handle_control({"action": "reset"})
    while not p._commands.empty():
        fn, args = p._commands.get_nowait()
        fn(*args)

    assert order == [("select", "abc"), "startup voice"]


def test_a_broadcast_survives_a_page_disconnecting_mid_send():
    import asyncio

    from sts.web.server import CaptionBroadcaster

    b = CaptionBroadcaster()
    received = []

    class Client:
        def __init__(self, name, leaves=False):
            self.name, self.leaves = name, leaves

        async def send_text(self, payload):
            received.append(self.name)
            if self.leaves:
                b._clients.discard(self)  # what unregister() does, during the await
            await asyncio.sleep(0)

    b._clients = {Client("a", leaves=True), Client("b"), Client("c")}
    asyncio.run(b._send_to_all({"type": "x"}))
    assert sorted(received) == ["a", "b", "c"]
